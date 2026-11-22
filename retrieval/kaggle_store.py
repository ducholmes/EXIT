"""Public Dataset snapshots. Kaggle imports and authentication are lazy."""
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .storage import digest, write_json


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class KaggleBackend:
    """Small official-client adapter, replaceable by a fake in offline tests."""
    def __init__(self):
        from kaggle.api.kaggle_api_extended import KaggleApi
        self.api = KaggleApi()
        self.api.authenticate()

    def info(self, ref):
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
        owner, slug = ref.split('/')
        request = ApiGetDatasetRequest()
        request.owner_slug, request.dataset_slug = owner, slug
        try:
            with self.api.build_kaggle_client() as client:
                result = client.datasets.dataset_api_client.get_dataset(request)
        except Exception as exc:
            if getattr(getattr(exc, 'response', None), 'status_code', None) == 404:
                return None
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

    def publish(self, folder):
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
        timeout = self.config.get('upload_timeout_minutes', 90) * 60
        last_error = None
        for attempt in range(3):
            try:
                info = self.backend.info(ref)
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
                        info = self.backend.info(ref)
                    version = self._matching_version(ref, info, manifest['snapshot_id']) if info else None
                    if version is None:
                        print(f'Uploading {manifest["kind"]}: {total / 1024**3:.2f} GiB to {ref}', flush=True)
                        self.backend.upload(ref, folder, bool(info), manifest['snapshot_id'])
                while True:
                    info = self.backend.info(ref)
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
            except ValueError:
                raise
            except Exception as exc:
                last_error = exc
                # Persist only exception type in public state; never raw HTTP/credential data.
                print(f'Upload attempt {attempt + 1}/3 failed ({type(exc).__name__}); snapshot retained at {folder}', flush=True)
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
