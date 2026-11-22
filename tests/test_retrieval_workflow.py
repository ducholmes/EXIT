"""Offline tests for scheduling, public Dataset uploads and recovery."""
import csv
import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval import prepare
from retrieval.kaggle_store import DatasetStore, KaggleBackend, make_snapshot, restore_snapshot
from retrieval.storage import digest, prepare_corpus, write_json
from retrieval.workflow import BudgetReached, Workflow, inspect, load_config, restore


class Clock:
    def __init__(self):
        self.value = 0
    def __call__(self):
        return self.value
    def sleep(self, seconds):
        self.value += seconds


class FakeBackend:
    def __init__(self, root):
        self.root = root
        self.datasets = {}
        self.upload_calls = 0
        self.fail_before = 0
        self.fail_after = False
        self.processing_polls = 0
        self.failed_processing = False
        self.private = False

    def info(self, ref):
        versions = self.datasets.get(ref)
        if not versions:
            return None
        if self.failed_processing:
            status = 'failed'
        elif self.processing_polls:
            status = 'processing'
            self.processing_polls -= 1
        else:
            status = 'ready'
        return {'private': self.private, 'version': len(versions), 'status': status,
                'versions': [{'number': i + 1, 'notes': 'EXIT snapshot ' + row['id']} for i, row in enumerate(versions)]}

    def upload(self, ref, folder, exists, snapshot_id):
        self.upload_calls += 1
        if self.fail_before:
            self.fail_before -= 1
            raise ConnectionError('simulated')
        assert exists == bool(self.datasets.get(ref))
        versions = self.datasets.setdefault(ref, [])
        target = self.root / ref.replace('/', '-') / str(len(versions) + 1)
        shutil.copytree(folder, target)
        versions.append({'id': snapshot_id, 'folder': target})
        if self.fail_after:
            self.fail_after = False
            raise ConnectionError('connection lost after successful submission')

    def fetch(self, ref, version, name, folder):
        source = self.datasets[ref][version - 1]['folder'] / name
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, folder / name)
        return folder / name


def configuration(tmp_path, name='local'):
    path = tmp_path / (name + '.json')
    write_json(path, {'owner': 'testowner', 'dataset_prefix': 'exit-test',
                      'root': str(tmp_path / name), 'downloads': str(tmp_path / 'downloads'),
                      'sources': str(tmp_path / 'sources'), 'dimension': 8, 'pq_m': 2,
                      'bits': 2, 'sample_size': 300, 'batch_size': 100, 'threads': 1})
    return load_config(path)


def storage(config, tmp_path, **kwargs):
    backend = FakeBackend(tmp_path / 'remote')
    clock = Clock()
    return DatasetStore(config, backend, clock, clock.sleep), backend, clock


def test_status_is_readonly_and_config_rejects_credentials(tmp_path):
    config = configuration(tmp_path)
    result = inspect(config)
    assert result['next_step'] == 'download-corpus'
    assert not Path(config['root']).exists()
    raw = tmp_path / 'bad.json'
    write_json(raw, {'owner': 'testowner', 'dataset_prefix': 'exit-test', 'api_key': 'do-not-save'})
    with pytest.raises(ValueError, match='credentials'):
        load_config(raw)


def test_two_hour_schedule_and_budget(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    clock = Clock()
    flow = Workflow(config, store=object(), clock=clock)
    calls = []
    def publish():
        calls.append('upload')
        flow.last_checkpoint = clock()
    monkeypatch.setattr(flow, 'publish_checkpoint', publish)
    save = lambda: calls.append('save')
    clock.value = 7199
    flow.tick(save)
    assert calls == []
    clock.value = 7200
    flow.tick(save)
    assert calls == ['save', 'upload']
    flow.tick(save)
    assert calls == ['save', 'upload']
    clock.value = 36000
    with pytest.raises(BudgetReached):
        flow.tick(save)


def snapshot_fixture(config, tmp_path):
    Path(config['root']).mkdir(parents=True, exist_ok=True)
    source = tmp_path / 'safe.json'
    source.write_text('{"safe": true}')
    return make_snapshot(config, 'checkpoint', {'trained-complete.json': (source, 'work/trained/complete.json')}, {'completed': []})


def test_publish_idempotent_and_ambiguous_response(tmp_path):
    config = configuration(tmp_path)
    store, backend, _ = storage(config, tmp_path)
    folder = snapshot_fixture(config, tmp_path)
    backend.fail_after = True
    receipt = store.publish(folder)
    assert receipt['version'] == 1 and backend.upload_calls == 1
    assert store.publish(folder)['version'] == 1
    assert backend.upload_calls == 1
    assert json.loads((folder / 'dataset-metadata.json').read_text())['id'] == 'testowner/exit-test-checkpoint'


def test_official_adapter_preserves_binary_and_public_flags(tmp_path):
    from types import SimpleNamespace
    calls = []
    backend = KaggleBackend.__new__(KaggleBackend)
    backend.api = SimpleNamespace(
        dataset_create_new=lambda *a, **kw: calls.append(('create', kw)) or SimpleNamespace(error=None),
        dataset_create_version=lambda *a, **kw: calls.append(('version', kw)) or SimpleNamespace(error=None))
    backend.upload('owner/name', tmp_path, False, 'snapshot-id')
    backend.upload('owner/name', tmp_path, True, 'snapshot-id')
    assert calls[0][1]['public'] is True
    assert calls[0][1]['convert_to_csv'] is False
    assert calls[1][1]['convert_to_csv'] is False
    assert calls[1][1]['delete_old_versions'] is False


def test_forbidden_missing_dataset_requires_authenticated_listing():
    from types import SimpleNamespace
    class HTTPError(Exception):
        response = SimpleNamespace(status_code=403)
    backend = KaggleBackend.__new__(KaggleBackend)
    def forbidden(_):
        raise HTTPError()
    backend._get_dataset = forbidden
    backend.api = SimpleNamespace(dataset_list=lambda **kw: [])
    assert backend.info('owner/new-dataset') is None
    backend.api = SimpleNamespace(dataset_list=lambda **kw: [SimpleNamespace(ref='owner/existing')])
    with pytest.raises(PermissionError, match='denies reading'):
        backend.info('owner/existing')
    backend.api = SimpleNamespace(dataset_list=lambda **kw: forbidden(None))
    with pytest.raises(PermissionError, match='personal KAGGLE_API_TOKEN'):
        backend.info('owner/new-dataset')


def test_retry_processing_visibility_and_size_limit(tmp_path):
    config = configuration(tmp_path)
    store, backend, clock = storage(config, tmp_path)
    folder = snapshot_fixture(config, tmp_path)
    backend.fail_before = 2
    backend.processing_polls = 2
    assert store.publish(folder)['version'] == 1
    assert backend.upload_calls == 3 and clock.value >= 15
    backend.private = True
    with pytest.raises(ValueError, match='private'):
        store.publish(folder)
    config['max_dataset_bytes'] = 1
    with pytest.raises(ValueError, match='size limit'):
        store.publish(folder)


def test_failed_upload_keeps_snapshot_and_no_success_receipt(tmp_path):
    config = configuration(tmp_path)
    store, backend, _ = storage(config, tmp_path)
    folder = snapshot_fixture(config, tmp_path)
    backend.fail_before = 10
    with pytest.raises(RuntimeError, match='not confirmed'):
        store.publish(folder)
    assert backend.upload_calls == 3
    assert (folder / 'snapshot.json').exists()


def test_restore_checks_integrity_and_existing_files(tmp_path):
    config = configuration(tmp_path)
    folder = snapshot_fixture(config, tmp_path)
    manifest = json.loads((folder / 'snapshot.json').read_text())
    def fetch(name, target):
        shutil.copyfile(folder / name, target)
    restore_snapshot(config, manifest, fetch)
    restored = Path(config['root']) / 'work/trained/complete.json'
    assert restored.exists()
    restore_snapshot(config, manifest, fetch)
    restored.write_text('conflict')
    with pytest.raises(ValueError, match='differs'):
        restore_snapshot(config, manifest, fetch)
    manifest['progress']['changed'] = True
    with pytest.raises(ValueError, match='checksum'):
        restore_snapshot(config, manifest, fetch)


def test_corpus_separate_checkpoint_and_restore(tmp_path):
    config = configuration(tmp_path)
    store, backend, _ = storage(config, tmp_path)
    flow = Workflow(config, store)
    tsv = tmp_path / 'corpus.tsv'
    with tsv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['id', 'text', 'title'])
        for i in range(3):
            writer.writerow([str(i), f'text{i}', 'title'])
    prepare_corpus(tsv, flow.root / 'corpus.sqlite')
    flow.registry['completed'] = ['prepare-corpus']
    flow.publish_checkpoint()
    assert backend.upload_calls == 2  # corpus + checkpoint
    flow.publish_checkpoint()
    assert backend.upload_calls == 2  # no changed data, no duplicate versions
    other = configuration(tmp_path, 'restored')
    restore(other, DatasetStore(other, backend))
    report = inspect(other)
    assert report['corpus']['count'] == 3
    assert report['steps']['prepare-corpus'] == 'complete'
    assert report['next_step'] == 'download-embeddings'
    assert report['remote']['corpus']['version'] == 1


def test_legacy_state_and_missing_sources(tmp_path):
    config = configuration(tmp_path)
    work = Path(config['root']) / 'work'
    work.mkdir(parents=True)
    state = {'next_shard': 1, 'shards': [{'name': 'shard-0'}], 'seen': 300, 'size': 300,
             'dimension': 8, 'seed': 42, 'sample': np.zeros((300, 8)), 'rng': {}}
    (work / 'sample.pkl').write_bytes(pickle.dumps(state))
    result = inspect(config)
    assert result['steps']['sample-pq'] == 'complete'
    write_json(Path(config['root']) / 'pipeline.json', {'config_id': config['config_id'], 'completed': [
        'download-corpus', 'extract-corpus', 'download-embeddings', 'extract-embeddings']})
    assert inspect(config)['next_step'] == 'download-corpus'


def test_periodic_sample_resume_mid_shard(tmp_path):
    config = configuration(tmp_path)
    args = Workflow(config, store=object()).args()
    args.shards = str(tmp_path / 'shard-*')
    vectors = np.random.default_rng(3).normal(size=(100001, 8)).astype('float32')
    (tmp_path / 'shard-0').write_bytes(pickle.dumps(([str(i) for i in range(len(vectors))], vectors)))
    class StopControl:
        def tick(self, save, progress):
            save()
            raise BudgetReached()
    args.control = StopControl()
    with pytest.raises(BudgetReached):
        prepare.sample(args, lambda: False)
    state = json.loads((args.work / 'sample.json').read_text())
    assert state['offset'] == 50000
    args.control = None
    prepare.sample(args, lambda: False)
    resumed = pickle.loads((args.work / 'sample.pkl').read_bytes())['sample'].copy()
    (args.work / 'sample.pkl').unlink()
    prepare.sample(args, lambda: False)
    np.testing.assert_array_equal(resumed, pickle.loads((args.work / 'sample.pkl').read_bytes())['sample'])


def test_full_workflow_and_completed_artifact_restore(tmp_path, monkeypatch):
    import importlib.metadata
    from types import SimpleNamespace
    config = configuration(tmp_path)
    source = Path(config['sources'])
    source.mkdir()
    tsv = Path(config['tsv'])
    with tsv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['id', 'text', 'title'])
        for i in range(300):
            writer.writerow([str(i), f'passage {i}', 'title'])
    shards = source / 'wikipedia_embeddings'
    shards.mkdir()
    vectors = np.random.default_rng(5).normal(size=(300, 8)).astype('float32')
    # Exercise auto-discovery of the alternative official wiki_ prefix.
    (shards / 'wiki_00').write_bytes(pickle.dumps(([str(i) for i in range(300)], vectors)))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(model_info=lambda _: SimpleNamespace(sha='fixture-revision')))
    monkeypatch.setattr(importlib.metadata, 'version', lambda _: 'fixture-version')
    store, backend, clock = storage(config, tmp_path)
    flow = Workflow(config, store, clock)
    write_json(flow.root / 'extraction.json', {'embeddings': True})
    flow.run()
    assert inspect(config)['next_step'] is None
    assert 'testowner/exit-test-artifact' in backend.datasets
    corpus_versions = backend.datasets['testowner/exit-test-corpus']
    assert len(corpus_versions) == 1
    # The new session has no source files, but a finished artifact needs none.
    shutil.rmtree(source)
    restored_config = configuration(tmp_path, 'new-session')
    restore(restored_config, DatasetStore(restored_config, backend))
    assert inspect(restored_config)['next_step'] is None
    assert (Path(restored_config['root']) / 'artifact/index.faiss').exists()
    prepare.verify(Path(restored_config['root']) / 'artifact')


def test_processing_timeout_does_not_report_success(tmp_path):
    config = configuration(tmp_path)
    config['upload_timeout_minutes'] = 0.1
    store, backend, _ = storage(config, tmp_path)
    folder = snapshot_fixture(config, tmp_path)
    backend.processing_polls = 100
    with pytest.raises(RuntimeError, match='not confirmed'):
        store.publish(folder)
    assert (folder / 'snapshot.json').exists()


def test_checkpoint_allowlist_excludes_credentials(tmp_path):
    config = configuration(tmp_path)
    store, backend, _ = storage(config, tmp_path)
    flow = Workflow(config, store)
    (flow.root / 'kaggle.json').write_text('{"key":"secret"}')
    (flow.work / 'raw-archive.tar').write_bytes(b'raw data')
    (flow.work / 'trained').mkdir()
    (flow.work / 'trained/index.faiss').write_bytes(b'incomplete training')
    flow.publish_checkpoint()
    uploaded = backend.datasets[store.ref('checkpoint')][0]['folder']
    assert {p.name for p in uploaded.iterdir()} == {'snapshot.json', 'dataset-metadata.json'}
    assert 'secret' not in (uploaded / 'snapshot.json').read_text()


def test_faiss_does_not_checkpoint_each_shard(tmp_path, monkeypatch):
    import importlib.metadata
    from types import SimpleNamespace
    config = configuration(tmp_path)
    flow = Workflow(config, store=object())
    tsv = tmp_path / 'corpus.tsv'
    with tsv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['id', 'text', 'title'])
        for i in range(600):
            writer.writerow([str(i), 'text', 'title'])
    prepare_corpus(tsv, flow.root / 'corpus.sqlite')
    for shard in range(2):
        (tmp_path / f'shard-{shard}').write_bytes(pickle.dumps((
            [str(shard * 300 + i) for i in range(300)],
            np.random.default_rng(shard).normal(size=(300, 8)).astype('float32'))))
    args = flow.args()
    args.shards = str(tmp_path / 'shard-*')
    args.control = None
    prepare.sample(args, lambda: False)
    prepare.train(args)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(model_info=lambda _: SimpleNamespace(sha='revision')))
    monkeypatch.setattr(importlib.metadata, 'version', lambda _: 'fixture-version')
    original = prepare.checkpoint
    checkpoints = []
    def count_checkpoint(index, work, state):
        checkpoints.append(dict(state))
        return original(index, work, state)
    monkeypatch.setattr(prepare, 'checkpoint', count_checkpoint)
    prepare.build(args, lambda: False)
    assert len(checkpoints) == 1
    assert checkpoints[0]['shard'] == 2  # completion only, no per-shard serializations
