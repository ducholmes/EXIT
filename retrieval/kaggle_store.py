"""Public Dataset snapshots. Kaggle imports and authentication are lazy."""
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
import zipfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .storage import digest, write_json


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def http_status(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        status = getattr(getattr(exc, 'response', None), 'status_code', None)
        if status is not None:
            return status
        exc = exc.__cause__
    return None


def transient(exc):
    status = http_status(exc)
    if status is not None:
        return status == 429 or 500 <= status < 600
    return isinstance(exc, (ConnectionError, TimeoutError, OSError))


class KaggleBackend:
    """Small official-client adapter, replaceable by a fake in offline tests."""
    def __init__(self):
        # The SDK otherwise prefers Kaggle's notebook token file over an explicit
        # personal token. Scope this change to the uploader process, not the notebook.
        if os.environ.get('KAGGLE_API_TOKEN'):
            os.environ.pop('KAGGLE_API_V1_TOKEN', None)
        from kaggle.api.kaggle_api_extended import KaggleApi
        self.api = KaggleApi()
        self.api.authenticate()

    def check_access(self, owner):
        known_owner = getattr(self.api, 'config_values', {}).get('username')
        if known_owner and known_owner.lower() != owner.lower():
            raise PermissionError(f'Configured owner {owner} differs from authenticated account {known_owner}')
        if self.api.dataset_list(mine=True, page=1) is None:
            raise PermissionError('Kaggle did not return an authenticated Dataset listing')

    def _get_dataset(self, ref):
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
        owner, slug = ref.split('/')
        request = ApiGetDatasetRequest()
        request.owner_slug, request.dataset_slug = owner, slug
        with self.api.build_kaggle_client() as client:
            return client.datasets.dataset_api_client.get_dataset(request)

    def _listed_info(self, ref, row):
        """GetDataset may lag behind ListDatasets/CreateDataset; use status alone.

        A listed Dataset is not proof of a usable upload: publish still requires
        ready status, a version number, and exact-version manifest read-back.
        """
        private = getattr(row, 'is_private', None)
        if not isinstance(private, bool):
            raise PermissionError(f'Kaggle lists {ref} but denies reading its visibility metadata')
        version = int(getattr(row, 'current_version_number', 0) or 0)
        try:
            status = self.api.dataset_status(ref)
        except Exception as exc:
            if http_status(exc) not in (403, 404):
                raise
            return {'private': private, 'version': version, 'versions': [], 'status': 'metadata_pending',
                    'metadata_error': f'Dataset status temporarily unavailable (HTTP {http_status(exc)})'}
        if str(status).lower() in ('ready', 'complete', 'completed') and version <= 0:
            status = 'metadata_pending'
        return {'private': private, 'version': version, 'versions': [], 'status': status}

    def info(self, ref):
        try:
            result = self._get_dataset(ref)
        except Exception as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            if status == 404:
                return None
            if status == 403:
                # Some Kaggle endpoints hide a missing Dataset behind Forbidden.
                # Confirm absence using the authenticated owner's listing; never
                # blindly treat every permission error as "does not exist".
                try:
                    page = 1
                    while True:
                        rows = self.api.dataset_list(mine=True, page=page)
                        if rows is None:
                            raise PermissionError('Kaggle did not return an authenticated Dataset listing')
                        match = next((row for row in rows if row and row.ref == ref), None)
                        if match is not None:
                            return self._listed_info(ref, match)
                        if not rows:
                            print(f'{ref} is absent from your Dataset listing; will create it.', flush=True)
                            return None
                        page += 1
                except PermissionError:
                    raise
                except Exception as listing_error:
                    if transient(listing_error):
                        raise
                    raise PermissionError('Kaggle denied Dataset access. Reload a personal KAGGLE_API_TOKEN from Secrets '
                                          'and confirm config owner matches that account.') from listing_error
            if status == 401:
                raise PermissionError('Kaggle authentication failed; reload KAGGLE_API_TOKEN from Secrets') from exc
            raise
        versions = [{'number': v.version_number, 'notes': v.version_notes, 'status': v.status}
                    for v in result.versions]
        return {'private': result.is_private, 'version': result.current_version_number,
                'versions': versions, 'status': self.api.dataset_status(ref)}

    def upload(self, ref, folder, exists, snapshot_id):
        if exists:
            result = self.api.dataset_create_version(str(folder), f'EXIT snapshot {snapshot_id}',
                                                     convert_to_csv=False, delete_old_versions=False,
                                                     dir_mode='skip', quiet=False)
        else:
            result = self.api.dataset_create_new(str(folder), public=True, convert_to_csv=False,
                                                 dir_mode='skip', quiet=False)
        if getattr(result, 'error', None):
            raise RuntimeError('Kaggle rejected Dataset upload: ' + str(result.error))

    def fetch(self, ref, version, name, folder):
        self.api.dataset_download_file(f'{ref}/{version}', name, path=str(folder), force=True,
                                       quiet=False)
        target = Path(folder) / name
        # Kaggle wraps some large files in a ZIP; avoid relying on a newer unzip kwarg.
        archive = target.with_name(target.name + '.zip')
        if not target.exists() and archive.exists():
            with zipfile.ZipFile(archive) as zipped:
                matches = [entry for entry in zipped.infolist() if not entry.is_dir() and Path(entry.filename).name == name]
                if len(matches) != 1:
                    raise ValueError('Unexpected Kaggle download archive')
                with zipped.open(matches[0]) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
            archive.unlink()
        if not target.exists():
            raise FileNotFoundError(f'Kaggle did not return requested file: {name}')
        return target


class DatasetStore:
    def __init__(self, config, backend=None, clock=time.monotonic, sleep=time.sleep):
        self.config = config
        self._backend = backend
        self.clock, self.sleep = clock, sleep

    @property
    def backend(self):
        if self._backend is None:
            self._backend = KaggleBackend()
        return self._backend

    def ref(self, kind):
        return f"{self.config['owner']}/{self.config['dataset_prefix']}-{kind}"

    def check_api(self):
        """Prove authenticated listing, writing and exact-version reading work."""
        print('API CHECK: checking authenticated Dataset listing', flush=True)
        for attempt in range(3):
            try:
                self.backend.check_access(self.config['owner'])
                break
            except Exception as exc:
                if http_status(exc) in (401, 403):
                    raise PermissionError(f'API CHECK: authenticated listing denied (HTTP {http_status(exc)}); '
                                          'check Secrets token and config owner') from exc
                if not transient(exc) or attempt == 2:
                    raise
                self.sleep(5 * 2 ** attempt)
        root = Path(self.config['root'])
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='api-probe-', dir=root) as temporary:
            temporary = Path(temporary)
            probe = {'probe_id': uuid.uuid4().hex, 'purpose': 'EXIT Kaggle API read/write check'}
            write_json(temporary / 'probe.json', probe)
            folder = make_snapshot(self.config, 'api-check', {'probe.json': (temporary / 'probe.json', 'probe.json')},
                                   {'purpose': 'api-check'})
            receipt = self.publish(folder, timeout_minutes=self.config.get('api_check_timeout_minutes', 10))
            print('API CHECK: reading probe from confirmed Dataset version', flush=True)
            for attempt in range(3):
                try:
                    returned = self.backend.fetch(receipt['dataset'], receipt['version'], 'probe.json', temporary)
                    if json.loads(returned.read_text(encoding='utf-8')) != probe:
                        raise ValueError('API CHECK failed: downloaded probe does not match uploaded content')
                    break
                except Exception as exc:
                    if http_status(exc) in (401, 403):
                        raise PermissionError(f'API CHECK: read-back denied for {receipt["dataset"]} '
                                              f'(HTTP {http_status(exc)})') from exc
                    if not transient(exc) or attempt == 2:
                        raise
                    self.sleep(5 * 2 ** attempt)
            shutil.rmtree(folder)
        print('API CHECK: API OK (authenticated listing + upload + verified read-back)', flush=True)
        return {**receipt, 'status': 'ok', 'checks': ['authenticated-listing', 'upload', 'read-back']}

    def _remote_snapshot(self, ref, version):
        with tempfile.TemporaryDirectory(prefix='inspect-', dir=self.config['root']) as folder:
            path = self.backend.fetch(ref, version, 'snapshot.json', Path(folder))
            return json.loads(path.read_text(encoding='utf-8'))

    def _matching_version(self, ref, info, snapshot_id):
        if not info:
            return None
        # Version notes let us find an upload accepted before a connection failure.
        versions = [v['number'] for v in info.get('versions', []) if snapshot_id in v.get('notes', '')]
        versions.append(info['version'])
        for version in dict.fromkeys(versions):
            try:
                remote = self._remote_snapshot(ref, version)
            except Exception as exc:
                if getattr(getattr(exc, 'response', None), 'status_code', None) == 404:
                    continue
                raise
            if remote.get('snapshot_id') == snapshot_id:
                return version
            if remote.get('config_id') != self.config['config_id']:
                raise ValueError(f'{ref} belongs to a different retrieval configuration; choose another prefix')
        return None

    def publish(self, folder, timeout_minutes=None):
        """Never mark durable before the exact snapshot is downloadable and ready."""
        manifest = json.loads((folder / 'snapshot.json').read_text(encoding='utf-8'))
        ref = self.ref(manifest['kind'])
        total = sum(x['bytes'] for x in manifest['files'].values())
        if total >= self.config.get('max_dataset_bytes', 200_000_000_000):
            raise ValueError(f'Dataset {ref} exceeds configured size limit ({total} bytes)')
        title = ref.split('/')[1][:50]
        write_json(folder / 'dataset-metadata.json', {
            'id': ref, 'title': title, 'licenses': [{'name': self.config.get('license', 'other')}],
            'description': ('EXIT retrieval snapshot ' + manifest['snapshot_id'] + '\n\n'
                'Sources: https://github.com/facebookresearch/contriever and '
                'https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz . '
                'Wikipedia text retains its original attribution/share-alike terms; '
                'see https://en.wikipedia.org/wiki/Wikipedia:Copyrights . '
                'Embeddings are the official Contriever-MSMARCO artifacts. '
                'This Dataset contains selected retrieval state, not credentials or raw downloads.')})
        started = self.clock()
        timeout = (timeout_minutes if timeout_minutes is not None else self.config.get('upload_timeout_minutes', 90)) * 60
        last_error = None
        metadata_wait_started = None
        last_status_log = None

        def read_info():
            nonlocal metadata_wait_started, last_status_log
            info = self.backend.info(ref)
            if last_status_log is None or self.clock() - last_status_log >= 30:
                print(f'DATASET STATUS: {ref}; state={info["status"] if info else "not visible"}; '
                      f'version={info["version"] if info else "unknown"}; elapsed={self.clock() - started:.0f}s', flush=True)
                last_status_log = self.clock()
            if info and info['status'] == 'metadata_pending':
                if metadata_wait_started is None:
                    metadata_wait_started = self.clock()
                    print(f'Waiting for Dataset metadata to become readable: {ref}', flush=True)
                if self.clock() - metadata_wait_started >= min(timeout, 120):
                    raise PermissionError(f'{ref} is listed but its status/version metadata remains unavailable '
                                          'after the propagation wait. Check Dataset visibility and token read permissions; '
                                          f'local snapshot retained at {folder}')
            else:
                metadata_wait_started = None
            return info

        for attempt in range(3):
            try:
                info = read_info()
                if info and info['private']:
                    raise ValueError(f'{ref} is private; expected public. Choose a new public Dataset.')
                version = self._matching_version(ref, info, manifest['snapshot_id']) if info and self._ready(info['status']) else None
                if version is None:
                    # Do not submit another version while an earlier upload is processing.
                    while info and not self._ready(info['status']):
                        if self._failed(info['status']):
                            raise RuntimeError('Kaggle Dataset processing failed')
                        if self.clock() - started >= timeout:
                            raise TimeoutError('Kaggle Dataset processing timed out')
                        self.sleep(15)
                        info = read_info()
                    version = self._matching_version(ref, info, manifest['snapshot_id']) if info else None
                    if version is None:
                        label = 'API CHECK' if manifest['kind'] == 'api-check' else 'ARTIFACT UPLOAD' if manifest['kind'] == 'artifact' else 'CHECKPOINT UPLOAD'
                        print(f'{label}: {manifest["kind"]}, {total / 1024**3:.2f} GiB to {ref}', flush=True)
                        self.backend.upload(ref, folder, bool(info), manifest['snapshot_id'])
                while True:
                    info = read_info()
                    if info and self._failed(info['status']):
                        raise RuntimeError('Kaggle Dataset processing failed')
                    if info and self._ready(info['status']):
                        version = self._matching_version(ref, info, manifest['snapshot_id'])
                        if version is not None:
                            receipt = {'dataset': ref, 'version': version, 'snapshot_id': manifest['snapshot_id'],
                                       'confirmed_at': utc_now(), 'url': f'https://www.kaggle.com/datasets/{ref}/versions/{version}'}
                            print('Saved: ' + receipt['url'], flush=True)
                            return receipt
                    if self.clock() - started >= timeout:
                        raise TimeoutError('Kaggle did not confirm the uploaded snapshot in time')
                    self.sleep(15)
            except (ValueError, PermissionError):
                raise
            except Exception as exc:
                if getattr(getattr(exc, 'response', None), 'status_code', None) in (401, 403):
                    raise PermissionError(f'Kaggle denied access to {ref} (HTTP {http_status(exc)}); '
                                          'check personal token permissions and Dataset owner. '
                                          f'Local snapshot retained at {folder}') from exc
                last_error = exc
                # Persist only exception type in public state; never raw HTTP/credential data.
                print(f'Upload attempt {attempt + 1}/3 failed ({type(exc).__name__}); snapshot retained at {folder}', flush=True)
                if not transient(exc):
                    break
                if self.clock() - started >= timeout:
                    break
                if attempt < 2:
                    self.sleep(5 * 2 ** attempt)
        raise RuntimeError(f'Upload not confirmed; local snapshot retained at {folder}. Last error: {type(last_error).__name__}') from last_error

    @staticmethod
    def _ready(status):
        return str(status).lower() in ('ready', 'complete', 'completed')

    @staticmethod
    def _failed(status):
        return str(status).lower() in ('error', 'failed')


def make_snapshot(config, kind, files, progress, corpus_ref=None):
    """Explicit allowlist, flat filenames and immutable copies on scratch disk."""
    root = Path(config['root'])
    parent = root / 'uploads'
    parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=f'{kind}-', dir=parent))
    entries = {}
    for name, (source, destination) in files.items():
        if Path(name).name != name or Path(destination).is_absolute() or '..' in Path(destination).parts:
            raise ValueError('Unsafe snapshot filename')
        target = folder / name
        if name == 'corpus.sqlite' and kind == 'corpus':
            src = sqlite3.connect(Path(source).resolve().as_uri() + '?mode=ro', uri=True)
            dst = sqlite3.connect(target)
            try:
                src.backup(dst)
            finally:
                dst.close()
                src.close()
        else:
            shutil.copyfile(source, target)
        entries[name] = {'destination': destination, 'bytes': target.stat().st_size, 'sha256': digest(target)}
    identity = {'format': 1, 'kind': kind, 'config_id': config['config_id'], 'files': entries,
                'progress': progress, 'corpus_ref': corpus_ref}
    snapshot_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    write_json(folder / 'snapshot.json', dict(identity, snapshot_id=snapshot_id))
    return folder


def restore_snapshot(config, manifest, fetch):
    """Restore to scratch; reject conflicting existing files, never overwrite them."""
    if manifest.get('format') != 1 or manifest['config_id'] != config['config_id']:
        raise ValueError('Snapshot format/configuration mismatch')
    identity = {k: v for k, v in manifest.items() if k != 'snapshot_id'}
    expected = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if expected != manifest.get('snapshot_id'):
        raise ValueError('Snapshot manifest checksum mismatch')
    root = Path(config['root']).resolve()
    for name, entry in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('Unsafe snapshot filename')
        target = (root / entry['destination']).resolve()
        if not target.is_relative_to(root):
            raise ValueError('Snapshot destination escapes work root')
        if target.exists():
            if digest(target) != entry['sha256']:
                raise ValueError(f'Existing file differs: {target}; restore into a fresh root')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.tmp')
        fetch(name, temporary)
        if temporary.stat().st_size != entry['bytes'] or digest(temporary) != entry['sha256']:
            raise ValueError(f'Corrupt snapshot file: {name}')
        os.replace(temporary, target)
