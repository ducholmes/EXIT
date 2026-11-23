"""Preparation orchestration, status and public Kaggle Dataset persistence."""
import argparse
import glob
import hashlib
import json
import pickle
import re
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from . import prepare
from .kaggle_store import DatasetStore, http_status, make_snapshot, restore_snapshot, utc_now
from .storage import digest, prepare_corpus, write_json
from .progress import activity

DEFAULTS = {'root': '/tmp/retrieval-state', 'downloads': '/tmp/retrieval-downloads',
            'sources': '/tmp/retrieval-sources', 'dimension': 768, 'pq_m': 96,
            'bits': 8, 'sample_size': 100000, 'seed': 42, 'batch_size': 50000,
            'threads': 4, 'checkpoint_minutes': 120, 'time_budget_minutes': 600,
            'upload_timeout_minutes': 90, 'api_check_timeout_minutes': 10, 'license': 'other'}
SETTINGS = ('dimension', 'pq_m', 'bits', 'sample_size', 'seed')
STEPS = ('download-corpus', 'extract-corpus', 'download-embeddings', 'extract-embeddings',
         'prepare-corpus', 'sample-pq', 'train-pq', 'build-index', 'verify', 'publish-artifact')
SAVE_AFTER = {'prepare-corpus', 'sample-pq', 'train-pq', 'build-index', 'publish-artifact'}


def load_config(path):
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    allowed = set(DEFAULTS) | {'owner', 'dataset_prefix', 'tsv', 'shards', 'checkpoint_input', 'corpus_input',
                               'max_dataset_bytes', 'public'}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError('Unknown config keys (credentials belong in environment/Secrets): ' + ', '.join(sorted(unknown)))
    config = {**DEFAULTS, **raw}
    if not re.fullmatch(r'[A-Za-z0-9_-]+', config.get('owner', '')) or config['owner'] == 'YOUR_USERNAME':
        raise ValueError('Set owner to your actual Kaggle username')
    prefix = config.get('dataset_prefix', '')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{4,38}', prefix):
        raise ValueError('dataset_prefix must contain 5-39 lowercase letters, numbers or hyphens')
    if config.get('public', True) is not True:
        raise ValueError('This workflow is configured for public Datasets only')
    if any(config[k] <= 0 for k in ('dimension', 'pq_m', 'sample_size', 'batch_size', 'threads',
                                   'checkpoint_minutes', 'time_budget_minutes', 'upload_timeout_minutes', 'api_check_timeout_minutes')):
        raise ValueError('Sizes, intervals and budgets must be positive')
    if config['dimension'] % config['pq_m'] or not 1 <= config['bits'] <= 16:
        raise ValueError('Invalid PQ configuration')
    config['root'] = str(Path(config['root']).resolve())
    for name in ('downloads', 'sources'):
        config[name] = str(Path(config[name]).resolve())
    config.setdefault('tsv', str(Path(config['sources']) / 'psgs_w100.tsv'))
    config.setdefault('shards', str(Path(config['sources']) / 'wikipedia_embeddings' / 'passages_*'))
    config['auto_shards'] = 'shards' not in raw
    config['config_id'] = hashlib.sha256(json.dumps({k: config[k] for k in SETTINGS}, sort_keys=True).encode()).hexdigest()
    return config


def read_json(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else ({} if default is None else default)


def shard_pattern(config):
    if glob.glob(config['shards']) or not config.get('auto_shards'):
        return config['shards']
    candidates = []
    for prefix in ('passages_', 'wiki_'):
        files = [p for p in Path(config['sources']).rglob(prefix + '*')
                 if p.is_file() and not p.name.endswith(('.tmp', '.json', '.faiss'))]
        if files:
            parents = {p.parent for p in files}
            if len(parents) != 1:
                raise ValueError('Multiple embedding folders found; set shards explicitly in config')
            candidates.append(str(next(iter(parents)) / (prefix + '*')))
    if len(candidates) > 1:
        raise ValueError('Multiple shard prefixes found; set shards explicitly in config')
    return candidates[0] if candidates else config['shards']


def corpus_progress(root):
    path = root / 'corpus.sqlite'
    if not path.exists():
        return {}
    db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'progress' not in tables:
            raise ValueError('Corpus has no progress table')
        row = db.execute("SELECT value FROM progress WHERE key='state'").fetchone()
        return json.loads(row[0]) if row else {}
    finally:
        db.close()


def sample_progress(root):
    path = root / 'work' / 'sample.pkl'
    if not path.exists():
        return {}
    sidecar = read_json(path.with_suffix('.json'))
    if sidecar:
        return sidecar
    # Compatibility with trusted legacy local checkpoints, before sidecars existed.
    with path.open('rb') as source:
        state = pickle.load(source)
    return {key: value for key, value in state.items() if key not in ('sample', 'rng')}


def inspect(config):
    """Read-only, local-only status: never authenticate, download or create dirs."""
    root = Path(config['root'])
    registry = read_json(root / 'pipeline.json')
    if registry.get('config_id', config['config_id']) != config['config_id']:
        raise ValueError('Configuration differs from existing pipeline state')
    corpus = corpus_progress(root)
    sampled = sample_progress(root)
    training = read_json(root / 'work' / 'trained' / 'complete.json')
    if sampled and any(sampled.get(k) != config[v] for k, v in (
            ('dimension', 'dimension'), ('size', 'sample_size'), ('seed', 'seed'))):
        raise ValueError('Sampling checkpoint configuration differs')
    if training and any(training.get(k) != config[k] for k in ('dimension', 'pq_m', 'bits', 'seed')):
        raise ValueError('Training checkpoint configuration differs')
    generation = prepare.latest_checkpoint(root / 'work')
    indexed = read_json(generation / 'complete.json') if generation else {}
    if generation and any(not (generation / f).exists() for f in ('index.faiss', 'index_meta.faiss')):
        raise ValueError('Checkpoint marker exists but index files are missing')
    artifact = read_json(root / 'artifact' / 'manifest.json')
    if artifact and any(not (root / 'artifact' / f).exists() for f in artifact.get('files', {})):
        raise ValueError('Artifact manifest exists but files are missing')
    completed = set(registry.get('completed', []))
    # On-disk evidence wins over registry flags after an interrupted restore.
    completed.difference_update(('prepare-corpus', 'sample-pq', 'train-pq'))
    if corpus.get('complete'):
        completed.add('prepare-corpus')
    if sampled and sampled['next_shard'] == len(sampled['shards']):
        completed.add('sample-pq')
    if training:
        if not all((root / 'work' / 'trained' / f).exists() for f in ('index.faiss', 'index_meta.faiss')):
            raise ValueError('Trained index files missing')
        completed.add('train-pq')
    if artifact.get('complete'):
        completed.add('build-index')
    else:
        completed.difference_update(('build-index', 'verify', 'publish-artifact'))
    downloads = Path(config['downloads'])
    shards = shard_pattern(config)
    for kind in ('corpus', 'embeddings'):
        name = prepare.SOURCES[kind].rsplit('/', 1)[1]
        if (downloads / name).exists():
            completed.add('download-' + kind)
    if Path(config['tsv']).exists():
        completed.add('extract-corpus')
    else:
        completed.discard('extract-corpus')
    completed.discard('extract-embeddings')
    if (glob.glob(shards) and read_json(root / 'extraction.json').get('embeddings')) or (
            '/kaggle/input/' in shards.replace('\\', '/') and glob.glob(shards)):
        completed.add('extract-embeddings')
    # A restored corpus/sample eliminates unnecessary source work for those stages.
    required = []
    need_shards = not artifact.get('complete')
    if not corpus.get('complete'):
        if not Path(config['tsv']).exists():
            required.extend(['download-corpus', 'extract-corpus'])
        required.append('prepare-corpus')
    if need_shards:
        if not glob.glob(shards) or 'extract-embeddings' not in completed:
            required.extend(['download-embeddings', 'extract-embeddings'])
        required.extend(['sample-pq', 'train-pq', 'build-index'])
    required.extend(['verify', 'publish-artifact'])
    # Completed source markers cannot override a missing source in a fresh session.
    for kind in ('corpus', 'embeddings'):
        if 'extract-' + kind in required and 'extract-' + kind not in completed:
            if not (downloads / prepare.SOURCES[kind].rsplit('/', 1)[1]).exists():
                completed.discard('download-' + kind)
    pending = [step for step in required if step not in completed]
    remote = read_json(root / 'remote.json')
    statuses = {step: ('complete' if step in completed else 'not_needed' if step not in required else 'pending') for step in STEPS}
    current = registry.get('current')
    if current in pending:
        statuses[current] = 'error' if registry.get('error') else 'in_progress'
    if corpus and not corpus.get('complete'):
        statuses['prepare-corpus'] = 'in_progress'
    if sampled and 'sample-pq' not in completed:
        statuses['sample-pq'] = 'in_progress'
    if indexed and 'build-index' not in completed:
        statuses['build-index'] = 'in_progress'
    return {'steps': statuses, 'next_step': pending[0] if pending else None,
            'corpus': corpus, 'sampling': sampled, 'index': {k: v for k, v in indexed.items() if k != 'identity'},
            'local_checkpoint': str(generation) if generation else None, 'remote': remote,
            'last_progress': registry.get('progress'), 'last_error': registry.get('error'),
            'api_check': read_json(root / 'api-check.json'),
            'next_checkpoint_at': registry.get('next_checkpoint_at'),
            'checkpoint_interval_minutes': config['checkpoint_minutes'],
            'limitations': 'Training, pickle decoding, hashing and individual file extraction cannot checkpoint mid-operation.'}


class Workflow:
    def __init__(self, config, store=None, clock=time.monotonic):
        self.config, self.clock = config, clock
        self.root = Path(config['root'])
        self.root.mkdir(parents=True, exist_ok=True)
        self.work = self.root / 'work'
        self.work.mkdir(exist_ok=True)
        self.store = store or DatasetStore(config)
        self._api_checked = False
        self._digest_cache = {}
        self._last_progress_log = None
        self.started = self.last_checkpoint = clock()
        self.registry = read_json(self.root / 'pipeline.json', {'config_id': config['config_id'], 'completed': []})
        if self.registry['config_id'] != config['config_id']:
            raise ValueError('Pipeline configuration mismatch')
        self._set_next_checkpoint()

    def _set_next_checkpoint(self):
        self.registry['next_checkpoint_at'] = (datetime.now(timezone.utc) +
                                               timedelta(minutes=self.config['checkpoint_minutes'])).isoformat()

    def save_registry(self):
        write_json(self.root / 'pipeline.json', self.registry)

    def deadline(self):
        return self.clock() - self.started >= self.config['time_budget_minutes'] * 60

    def tick(self, save=lambda: None, progress=None):
        self.registry['progress'] = progress
        self.save_registry()
        now = self.clock()
        if self._last_progress_log is None or now - self._last_progress_log >= 30:
            print(f'PROGRESS [{self.registry.get("current") or "processing"}]: '
                  + json.dumps(progress or {}, ensure_ascii=False), flush=True)
            self._last_progress_log = now
        if self.clock() - self.last_checkpoint >= self.config['checkpoint_minutes'] * 60:
            current = self.registry.get('current') or ''
            if current.startswith(('download-', 'extract-')):
                self._skip_checkpoint('only download/extraction progress; raw sources are not uploaded')
            else:
                save()
                self.publish_checkpoint()
        if self.deadline():
            save()
            raise BudgetReached()

    def args(self):
        return SimpleNamespace(**{**self.config, 'work': self.work, 'corpus': self.root / 'corpus.sqlite',
                                  'output': self.root / 'artifact', 'shards': shard_pattern(self.config), 'control': self})

    def _progress(self, payload_id=None):
        return {'config_id': self.config['config_id'], 'completed': self.registry['completed'],
                'current': self.registry.get('current'), 'progress': self.registry.get('progress'),
                'artifact_ref': read_json(self.root / 'remote.json').get('artifact'), 'payload_id': payload_id}

    def _skip_checkpoint(self, reason):
        print('CHECKPOINT SKIP: ' + reason, flush=True)
        self.last_checkpoint = self.clock()
        self._set_next_checkpoint()
        self.save_registry()
        return False

    def check_api(self):
        if self._api_checked:
            return read_json(self.root / 'api-check.json')
        try:
            with activity('Kaggle API upload/read-back check'):
                result = self.store.check_api()
        except Exception as exc:
            write_json(self.root / 'api-check.json', {'status': 'failed', 'checked_at': utc_now(),
                       'error_type': type(exc).__name__, 'http_status': http_status(exc),
                       'dataset': self.store.ref('api-check')})
            print('API CHECK FAILED: no download or processing was started by resume.', flush=True)
            raise
        write_json(self.root / 'api-check.json', result)
        self._api_checked = True
        self.last_checkpoint = self.clock()
        self._set_next_checkpoint()
        return result

    def _checkpoint_files(self):
        files = {}
        sampled = sample_progress(self.root)
        if sampled.get('seen', 0) > 0:
            for name, path in (('sample.pkl', 'sample.pkl'), ('sample-progress.json', 'sample.json')):
                if (self.work / path).exists():
                    files[name] = (self.work / path, 'work/' + path)
        trained = self.work / 'trained'
        if all((trained / name).exists() for name in ('index.faiss', 'index_meta.faiss', 'complete.json')):
            for name, path in (('trained-index.faiss', 'index.faiss'), ('trained-meta.faiss', 'index_meta.faiss'),
                               ('trained-complete.json', 'complete.json')):
                files[name] = (trained / path, 'work/trained/' + path)
        generation = prepare.latest_checkpoint(self.work)
        if generation:
            state = read_json(generation / 'complete.json')
            if state.get('vectors', 0) > 0 or state.get('shard', 0) > 0 or state.get('offset', 0) > 0:
                for name in ('index.faiss', 'index_meta.faiss', 'complete.json'):
                    files['checkpoint-' + name] = (generation / name, f'work/checkpoints/{generation.name}/{name}')
        return files

    def _file_digest(self, path):
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        if key not in self._digest_cache:
            self._digest_cache[key] = digest(path)
        return self._digest_cache[key]

    def publish_checkpoint(self):
        """Upload only durable corpus/work data, with exact cross-Dataset references."""
        remote = read_json(self.root / 'remote.json')
        corpus = corpus_progress(self.root)
        files = self._checkpoint_files()
        has_corpus = corpus.get('count', 0) > 0
        if not has_corpus and not files:
            return self._skip_checkpoint('no processed data; only download/extraction or empty pipeline state')
        corpus_identity = hashlib.sha256(json.dumps(corpus, sort_keys=True).encode()).hexdigest() if has_corpus else None
        artifact_ref = remote.get('artifact', {})
        payload = {'corpus_identity': corpus_identity,
                   'files': {name: self._file_digest(value[0]) for name, value in files.items()},
                   'artifact': {k: artifact_ref.get(k) for k in ('dataset', 'version', 'snapshot_id')}}
        payload_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if remote.get('checkpoint_payload_id') == payload_id:
            return self._skip_checkpoint('processed data and resume position have not changed')
        if has_corpus:
            # Row count + source + offset identify a committed import revision.
            if remote.get('corpus_identity') != corpus_identity:
                folder = make_snapshot(self.config, 'corpus', {'corpus.sqlite': (self.root / 'corpus.sqlite', 'corpus.sqlite')}, corpus)
                remote['corpus'] = self.store.publish(folder)
                remote['corpus_identity'] = corpus_identity
                write_json(self.root / 'remote.json', remote)
                shutil.rmtree(folder)
        folder = make_snapshot(self.config, 'checkpoint', files, self._progress(payload_id), remote.get('corpus'))
        if remote.get('checkpoint', {}).get('snapshot_id') != read_json(folder / 'snapshot.json')['snapshot_id']:
            remote['checkpoint'] = self.store.publish(folder)
        remote['checkpoint_payload_id'] = payload_id
        write_json(self.root / 'remote.json', remote)
        shutil.rmtree(folder)
        self.last_checkpoint = self.clock()
        self._set_next_checkpoint()
        self.save_registry()
        print(f'Next periodic checkpoint in {self.config["checkpoint_minutes"]} minutes', flush=True)
        return True

    def publish_artifact(self):
        directory = self.root / 'artifact'
        files = {name: (directory / name, 'artifact/' + name) for name in
                 ('corpus.sqlite', 'index.faiss', 'index_meta.faiss', 'manifest.json')}
        folder = make_snapshot(self.config, 'artifact', files, {'complete': True})
        receipt = self.store.publish(folder)
        remote = read_json(self.root / 'remote.json')
        remote['artifact'] = receipt
        write_json(self.root / 'remote.json', remote)
        shutil.rmtree(folder)

    def run(self):
        print(f'RESUME: work root={self.root}; checking Kaggle API before processing', flush=True)
        self.check_api()
        import faiss
        faiss.omp_set_num_threads(self.config['threads'])
        # Only meaningful unsaved payload is retried, never download-only state.
        if self.registry.get('error'):
            self.publish_checkpoint()
            self.registry.pop('error', None)
            self.save_registry()
        while True:
            report = inspect(self.config)
            step = report['next_step']
            if step is None:
                print('All preparation steps complete; artifact saved on Kaggle.', flush=True)
                return
            if self.deadline():
                self.publish_checkpoint()
                return
            self.registry.update(current=step, progress=None)
            self.registry.pop('error', None)
            self.save_registry()
            print(f'Running {step}; checkpoint every {self.config["checkpoint_minutes"]} minutes', flush=True)
            try:
                if step.startswith('download-'):
                    kind = step.removeprefix('download-')
                    url = prepare.SOURCES[kind]
                    prepare.download(url, Path(self.config['downloads']) / url.rsplit('/', 1)[1], self.deadline,
                                     lambda p: self.tick(progress=p))
                elif step.startswith('extract-'):
                    kind = step.removeprefix('extract-')
                    archive = Path(self.config['downloads']) / prepare.SOURCES[kind].rsplit('/', 1)[1]
                    prepare.extract(archive, Path(self.config['sources']), lambda p: self.tick(progress=p))
                    if kind == 'embeddings' and not glob.glob(shard_pattern(self.config)):
                        raise ValueError('No embedding shards found after extraction; set shards in config to the actual files')
                    extracted = read_json(self.root / 'extraction.json')
                    extracted[kind] = True
                    write_json(self.root / 'extraction.json', extracted)
                elif step == 'prepare-corpus':
                    result = prepare_corpus(self.config['tsv'], self.root / 'corpus.sqlite', self.deadline,
                                            on_progress=lambda p: self.tick(progress=p))
                    if not result['complete']:
                        raise BudgetReached()
                elif step == 'sample-pq':
                    prepare.sample(self.args(), self.deadline)
                    p = sample_progress(self.root)
                    if p['next_shard'] != len(p['shards']):
                        raise BudgetReached()
                elif step == 'train-pq':
                    # No callback inside FAISS training; save the reusable sample first.
                    prepare.train(self.args())
                elif step == 'build-index':
                    prepare.build(self.args(), self.deadline)
                    if not (self.root / 'artifact/manifest.json').exists():
                        raise BudgetReached()
                elif step == 'verify':
                    prepare.verify(self.root / 'artifact')
                elif step == 'publish-artifact':
                    self.publish_artifact()
                if step not in self.registry['completed']:
                    self.registry['completed'].append(step)
                self.registry.update(current=None, progress=None)
                self.save_registry()
                if step in SAVE_AFTER:
                    self.publish_checkpoint()
                elif step.startswith(('download-', 'extract-')):
                    print('CHECKPOINT SKIP: source step completed; progress is local-only', flush=True)
            except BudgetReached:
                self.publish_checkpoint()
                print('Time budget reached. Check status for confirmed remote saves; raw sources remain local.', flush=True)
                return
            except Exception as exc:
                self.registry['error'] = type(exc).__name__
                self.save_registry()
                print('Stopped; last confirmed Dataset references are in remote.json. Local snapshot retained.', flush=True)
                raise


class BudgetReached(Exception):
    pass


def restore(config, store=None):
    store = store or DatasetStore(config)
    root = Path(config['root'])
    root.mkdir(parents=True, exist_ok=True)
    local = config.get('checkpoint_input')
    remote = {}

    def restore_one(kind, version=None, receipt=None, local_folder=None):
        ref = store.ref(kind)
        with tempfile.TemporaryDirectory(prefix='restore-', dir=root) as temporary:
            temporary = Path(temporary)
            if local_folder:
                folder = Path(local_folder)
                manifest = read_json(folder / 'snapshot.json')
                if not manifest:
                    raise ValueError('Attached Input has no snapshot.json')
                def fetch(name, target):
                    shutil.copyfile(folder / name, target)
            else:
                info = store.backend.info(ref)
                if not info or not store._ready(info['status']):
                    raise ValueError(f'Dataset missing or not ready: {ref}')
                version = version or info['version']
                manifest = read_json(store.backend.fetch(ref, version, 'snapshot.json', temporary))
                def fetch(name, target):
                    source = store.backend.fetch(ref, version, name, temporary)
                    shutil.copyfile(source, target)
            if manifest.get('kind') != kind:
                raise ValueError('Wrong snapshot kind')
            if receipt and receipt['snapshot_id'] != manifest.get('snapshot_id'):
                raise ValueError('Dataset version does not match checkpoint reference')
            restore_snapshot(config, manifest, fetch)
            return manifest, {'dataset': ref, 'version': version, 'snapshot_id': manifest['snapshot_id'],
                              'restored_at': utc_now()}

    manifest, remote['checkpoint'] = restore_one('checkpoint', local_folder=local)
    corpus_ref = manifest.get('corpus_ref')
    if corpus_ref:
        if corpus_ref['dataset'] != store.ref('corpus'):
            raise ValueError('Corpus Dataset owner/prefix differs from configured target')
        corpus_manifest, remote['corpus'] = restore_one('corpus', version=corpus_ref['version'], receipt=corpus_ref,
                                                       local_folder=config.get('corpus_input'))
        remote['corpus_identity'] = hashlib.sha256(json.dumps(corpus_manifest['progress'], sort_keys=True).encode()).hexdigest()
    progress = manifest['progress']
    if progress.get('payload_id'):
        remote['checkpoint_payload_id'] = progress['payload_id']
    if progress.get('artifact_ref'):
        ref = progress['artifact_ref']
        if ref['dataset'] != store.ref('artifact'):
            raise ValueError('Artifact Dataset owner/prefix mismatch')
        _, remote['artifact'] = restore_one('artifact', version=ref['version'], receipt=ref)
    write_json(root / 'pipeline.json', progress)
    write_json(root / 'remote.json', remote)
    print(json.dumps(inspect(config), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['check-api', 'status', 'resume', 'checkpoint', 'restore'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--json', action='store_true', help='Machine-readable status')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.step == 'status':
        result = inspect(config)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            for step, state in result['steps'].items():
                print(f'{step:22} {state}')
            print('Next step:', result['next_step'] or 'none')
            print('Local checkpoint:', result['local_checkpoint'])
            print('Index progress:', result['index'])
            print('Corpus passages:', result['corpus'].get('count', 0))
            print('Sampling progress:', {k: result['sampling'].get(k) for k in ('next_shard', 'offset', 'seen')})
            print('Last confirmed remote saves:', json.dumps(result['remote'], indent=2))
            print('Last API check:', json.dumps(result['api_check'], indent=2))
            print('Next checkpoint due (active run):', result['next_checkpoint_at'])
            print(result['limitations'])
    elif args.step == 'restore':
        restore(config)
    elif args.step == 'checkpoint':
        Workflow(config).publish_checkpoint()
    elif args.step == 'check-api':
        Workflow(config).check_api()
    else:
        Workflow(config).run()


if __name__ == '__main__':
    main()
