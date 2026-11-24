"""Immutable, PQ-independent publication and validation of embedding sources."""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from .prepare import load_shard, latest_checkpoint
from .storage import digest, write_json
from .kaggle_store import make_snapshot

MANIFEST = 'embeddings-manifest.json'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def family(dimension):
    return {'format': 1, 'model': 'facebook/contriever-msmarco',
            'corpus': 'DPR Wikipedia 2018-12-20', 'dimension': dimension}


def validate_manifest(manifest, dimension):
    expected = family(dimension)
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('Embedding model/corpus/dimension mismatch')
    payload = {k: v for k, v in manifest.items() if k != 'source_id'}
    if fingerprint(payload) != manifest.get('source_id'):
        raise ValueError('Embedding manifest checksum mismatch')
    names = [row['name'] for row in manifest['shards']]
    if not names or len(names) != len(set(names)) or any(Path(n).name != n or n in ('.', '..') for n in names):
        raise ValueError('Unsafe or duplicate embedding shard names')
    if names != sorted(names):
        raise ValueError('Embedding shards must retain deterministic filename order')
    if sum(row['vectors'] for row in manifest['shards']) != manifest['vectors']:
        raise ValueError('Embedding manifest vector total mismatch')


class EmbeddingSource:
    def __init__(self, config, store):
        self.config, self.store = config, store
        self.root = Path(config['root'])
        self.root.mkdir(parents=True, exist_ok=True)
        self.validated = None

    def remote(self, version=None):
        ref = self.store.ref('embeddings')
        if version is None:
            info = self.store.backend.info(ref)
            if info is None:
                return None
            if info['private'] or not self.store._ready(info['status']):
                raise ValueError(f'Embedding Dataset must be public and ready: {ref}')
            version = info['version']
        with tempfile.TemporaryDirectory(prefix='embedding-info-', dir=self.root) as directory:
            path = self.store.backend.fetch(ref, version, MANIFEST, Path(directory))
            manifest = json.loads(path.read_text(encoding='utf-8'))
            manifest_hash = digest(path)
            snapshot = json.loads(self.store.backend.fetch(ref, version, 'snapshot.json', Path(directory)).read_text(encoding='utf-8'))
        validate_manifest(manifest, self.config['dimension'])
        if snapshot.get('kind') != 'embeddings' or snapshot.get('progress', {}).get('source_id') != manifest['source_id']:
            raise ValueError('Embedding Dataset snapshot does not match source manifest')
        entry = snapshot.get('files', {}).get(MANIFEST, {})
        # Source manifest is also authenticated by its hash in snapshot.json.
        if entry.get('sha256') != manifest_hash or fingerprint({k: v for k, v in snapshot.items() if k != 'snapshot_id'}) != snapshot.get('snapshot_id'):
            raise ValueError('Embedding snapshot/manifest hash mismatch')
        return {'dataset': ref, 'version': version, 'snapshot_id': snapshot['snapshot_id'],
                'source_id': manifest['source_id'], 'manifest': manifest}

    def require_input_if_remote(self):
        source = self.remote()
        if source:
            raise ValueError(f'Embedding source already saved: https://www.kaggle.com/datasets/{source["dataset"]}/versions/{source["version"]}. '
                'Use Add Input, set embeddings_input to its mounted directory, then resume. '
                'The Facebook archive will not be downloaded again.')

    def _legacy_expected(self):
        expected = None
        sample = self.root / 'work/sample.json'
        if sample.exists():
            expected = json.loads(sample.read_text(encoding='utf-8')).get('shards')
        # Legacy sample files did not have a sidecar. Only trusted local state.
        path = self.root / 'work/sample.pkl'
        if expected is None and path.exists():
            import pickle
            with path.open('rb') as f:
                expected = pickle.load(f).get('shards')
        generation = latest_checkpoint(self.root / 'work')
        if generation:
            index_expected = json.loads((generation / 'complete.json').read_text(encoding='utf-8')).get('identity', {}).get('shards')
            if expected is not None and index_expected is not None and expected != index_expected:
                raise ValueError('Existing sample and index checkpoint refer to different embedding sources')
            expected = expected if expected is not None else index_expected
        return expected

    def check_compatibility(self, manifest, expected_ref=None):
        if expected_ref and expected_ref.get('source_id') != manifest['source_id']:
            raise ValueError('Embedding source_id differs from checkpoint reference')
        expected = self._legacy_expected()
        # Existing index generations independently validate this list in build().
        if expected is not None:
            actual = [{k: row[k] for k in ('name', 'size', 'sha256')} for row in manifest['shards']]
            if actual != expected:
                raise ValueError('Embedding shard names/hashes differ from existing sample/checkpoint')

    def local(self, paths, expected_ref=None):
        paths = sorted(Path(p) for p in paths)
        if not paths or len({p.name for p in paths}) != len(paths):
            raise ValueError('Embedding source has no shards or duplicate filenames')
        stats = [{'path': str(p.resolve()), 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in paths]
        cache_path = self.root / 'embedding-source-cache.json'
        cache = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else {}
        if cache.get('stats') == stats:
            manifest = cache['manifest']
            validate_manifest(manifest, self.config['dimension'])
        else:
            rows = []
            for p in paths:
                print(f'EMBEDDING SOURCE: validating {p.name}', flush=True)
                ids, vectors = load_shard(p, self.config['dimension'])
                if not ids:
                    raise ValueError(f'Empty embedding shard: {p.name}')
                rows.append({'name': p.name, 'size': p.stat().st_size, 'sha256': digest(p),
                             'vectors': len(ids), 'dtype': str(vectors.dtype)})
                del ids, vectors
            manifest = {**family(self.config['dimension']), 'shards': rows, 'vectors': sum(r['vectors'] for r in rows)}
            manifest['source_id'] = fingerprint(manifest)
            write_json(cache_path, {'stats': stats, 'manifest': manifest})
        self.check_compatibility(manifest, expected_ref)
        self.validated = manifest, paths
        return manifest, paths

    def attached(self, directory, expected_ref=None):
        directory = Path(directory)
        manifest_path = directory / MANIFEST
        if not manifest_path.is_file():
            raise ValueError(f'Attached embedding Input has no {MANIFEST}: {directory}')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        validate_manifest(manifest, self.config['dimension'])
        self.check_compatibility(manifest, expected_ref)
        snapshot = json.loads((directory / 'snapshot.json').read_text(encoding='utf-8'))
        if snapshot.get('kind') != 'embeddings' or snapshot.get('progress', {}).get('source_id') != manifest['source_id']:
            raise ValueError('Attached embedding snapshot differs from source manifest')
        if snapshot.get('files', {}).get(MANIFEST, {}).get('sha256') != digest(manifest_path) or fingerprint({k: v for k, v in snapshot.items() if k != 'snapshot_id'}) != snapshot.get('snapshot_id'):
            raise ValueError('Attached embedding manifest/snapshot hash mismatch')
        if expected_ref and snapshot.get('snapshot_id') != expected_ref['snapshot_id']:
            raise ValueError('Attached Input differs from pinned embedding Dataset version')
        paths = []
        for row in manifest['shards']:
            path = directory / row['name']
            if not path.is_file() or path.stat().st_size != row['size'] or digest(path) != row['sha256']:
                raise ValueError(f'Missing/corrupt embedding Input shard: {row["name"]}')
            paths.append(path)
        self.validated = manifest, paths
        return manifest, paths

    def publish(self, manifest, paths):
        existing = self.remote()
        if existing:
            if existing['source_id'] != manifest['source_id']:
                raise ValueError('Embedding Dataset contains a different source; choose another embeddings_dataset')
            print('EMBEDDING SOURCE: already published; skipping upload', flush=True)
            return {k: v for k, v in existing.items() if k != 'manifest'}
        total = sum(p.stat().st_size for p in paths)
        free = shutil.disk_usage(self.root).free
        if free < total + 1024 ** 3:
            raise OSError(f'Not enough scratch for immutable embedding staging: need about {total / 1024**3:.1f} GiB plus margin')
        family_config = {**self.config, 'config_id': fingerprint(family(self.config['dimension']))}
        with tempfile.TemporaryDirectory(prefix='embedding-manifest-', dir=self.root) as temporary:
            path = Path(temporary) / MANIFEST
            write_json(path, manifest)
            files = {p.name: (p, 'embedding-source/' + p.name) for p in paths}
            files[MANIFEST] = (path, 'embedding-source/' + MANIFEST)
            folder = make_snapshot(family_config, 'embeddings', files, {'source_id': manifest['source_id']})
        receipt = self.store.publish(folder)
        confirmed = self.remote(version=receipt['version'])
        if not confirmed or confirmed['source_id'] != manifest['source_id'] or confirmed['snapshot_id'] != receipt['snapshot_id']:
            raise ValueError('Published embedding source did not match confirmed Dataset manifest')
        shutil.rmtree(folder)
        return {**receipt, 'source_id': manifest['source_id']}
