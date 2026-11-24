"""Immutable embedding sources, Input binding and cross-session reuse."""
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval.embedding_source import EmbeddingSource, MANIFEST
from retrieval.kaggle_store import DatasetStore
from retrieval.storage import write_json
from retrieval.workflow import Workflow, inspect
from test_retrieval_workflow import configuration, storage


def shards(tmp_path):
    folder = tmp_path / 'shards'
    folder.mkdir()
    for i in range(2):
        (folder / f'passages_{i:02d}').write_bytes(pickle.dumps((
            [str(i * 5 + j) for j in range(5)], np.arange(40, dtype='float32').reshape(5, 8) + i)))
    return sorted(folder.glob('passages_*'))


def test_publish_once_independent_of_pq_and_attach_input(tmp_path):
    config = configuration(tmp_path)
    paths = shards(tmp_path)
    store, backend, _ = storage(config, tmp_path)
    manager = EmbeddingSource(config, store)
    manifest, _ = manager.local(paths)
    receipt = manager.publish(manifest, paths)
    assert manifest['vectors'] == 10 and len(manifest['shards']) == 2
    assert backend.upload_calls == 1
    assert manager.publish(manifest, paths)['source_id'] == receipt['source_id']
    changed_config = dict(config, pq_m=4, config_id='a-different-pq-configuration')
    manager2 = EmbeddingSource(changed_config, DatasetStore(changed_config, backend))
    assert manager2.publish(manifest, paths)['source_id'] == receipt['source_id']
    assert backend.upload_calls == 1
    uploaded = backend.datasets[store.ref('embeddings')][0]['folder']
    mounted = tmp_path / 'input'
    shutil.copytree(uploaded, mounted)
    attached, actual_paths = manager.attached(mounted, receipt)
    assert attached == manifest
    assert [p.name for p in actual_paths] == [p.name for p in paths]
    assert not any(p.name.endswith(('.tar', '.part')) for p in uploaded.iterdir())


def test_remote_exists_stops_before_downloading(tmp_path):
    config = configuration(tmp_path)
    paths = shards(tmp_path)
    store, backend, _ = storage(config, tmp_path)
    manager = EmbeddingSource(config, store)
    manifest, _ = manager.local(paths)
    manager.publish(manifest, paths)
    with pytest.raises(ValueError, match='Add Input'):
        manager.require_input_if_remote()
    assert backend.upload_calls == 1


def test_resume_does_not_call_download_when_source_is_saved(tmp_path, monkeypatch):
    from retrieval import workflow, prepare
    config = configuration(tmp_path)
    paths = shards(tmp_path)
    store, _, clock = storage(config, tmp_path)
    manager = EmbeddingSource(config, store)
    manifest, _ = manager.local(paths)
    manager.publish(manifest, paths)
    flow = Workflow(config, store, clock)
    monkeypatch.setattr(flow, 'check_api', lambda: None)
    monkeypatch.setattr(workflow, 'inspect', lambda _: {'next_step': 'download-embeddings'})
    monkeypatch.setattr(prepare, 'download', lambda *a, **kw: pytest.fail('must not download saved source'))
    with pytest.raises(ValueError, match='Add Input'):
        flow.run()


def test_input_corruption_missing_and_different_source_rejected(tmp_path):
    config = configuration(tmp_path)
    paths = shards(tmp_path)
    store, backend, _ = storage(config, tmp_path)
    manager = EmbeddingSource(config, store)
    manifest, _ = manager.local(paths)
    receipt = manager.publish(manifest, paths)
    uploaded = backend.datasets[store.ref('embeddings')][0]['folder']
    mounted = tmp_path / 'input'
    shutil.copytree(uploaded, mounted)
    (mounted / paths[0].name).write_bytes(b'bad')
    with pytest.raises(ValueError, match='Missing/corrupt'):
        manager.attached(mounted, receipt)
    (mounted / paths[0].name).unlink()
    with pytest.raises(ValueError, match='Missing/corrupt'):
        manager.attached(mounted, receipt)
    with pytest.raises(ValueError, match='source_id'):
        manager.attached(uploaded, dict(receipt, source_id='wrong'))
    # Same filenames, changed vector content must never replace an existing Dataset.
    (paths[0]).write_bytes(pickle.dumps(([str(j) for j in range(5)], np.ones((5, 8), dtype='float32'))))
    new_manifest, _ = manager.local(paths)
    with pytest.raises(ValueError, match='different source'):
        manager.publish(new_manifest, paths)
    assert backend.upload_calls == 1


def test_legacy_sample_shard_hashes_are_checked(tmp_path):
    config = configuration(tmp_path)
    paths = shards(tmp_path)
    store, _, _ = storage(config, tmp_path)
    manager = EmbeddingSource(config, store)
    manifest, _ = manager.local(paths)
    expected = [{k: row[k] for k in ('name', 'size', 'sha256')} for row in manifest['shards']]
    write_json(Path(config['root']) / 'work/sample.json', {'shards': expected})
    manager.local(paths)
    expected[0]['sha256'] = 'changed'
    write_json(Path(config['root']) / 'work/sample.json', {'shards': expected})
    with pytest.raises(ValueError, match='existing sample/checkpoint'):
        manager.local(paths)


def test_legacy_index_reference_checked_without_sample(tmp_path):
    config = configuration(tmp_path)
    paths = shards(tmp_path)
    store, _, _ = storage(config, tmp_path)
    manager = EmbeddingSource(config, store)
    manifest, _ = manager.local(paths)
    expected = [{k: row[k] for k in ('name', 'size', 'sha256')} for row in manifest['shards']]
    generation = Path(config['root']) / 'work/checkpoints/generation-1'
    generation.mkdir(parents=True)
    (generation / 'index.faiss').write_bytes(b'index')
    (generation / 'index_meta.faiss').write_bytes(b'mapping')
    expected[0]['sha256'] = 'wrong'
    write_json(generation / 'complete.json', {'identity': {'shards': expected}})
    with pytest.raises(ValueError, match='existing sample/checkpoint'):
        manager.local(paths)


def test_workflow_uses_input_directly_and_keeps_source_out_of_checkpoint(tmp_path):
    config = configuration(tmp_path)
    paths = shards(tmp_path)
    store, backend, clock = storage(config, tmp_path)
    manager = EmbeddingSource(config, store)
    manifest, _ = manager.local(paths)
    receipt = manager.publish(manifest, paths)
    mounted = backend.datasets[store.ref('embeddings')][0]['folder']
    next_config = configuration(tmp_path, 'new-session')
    next_config['embeddings_input'] = str(mounted)
    flow = Workflow(next_config, DatasetStore(next_config, backend), clock)
    write_json(flow.root / 'remote.json', {'embeddings': receipt})
    flow.publish_embeddings()
    assert flow.args().embedding_files == [mounted / p.name for p in paths]
    assert all(str(p).startswith(str(mounted)) for p in flow.args().embedding_files)
    assert inspect(next_config)['steps']['extract-embeddings'] == 'complete'
    assert inspect(next_config)['steps']['publish-embeddings'] == 'complete'
    assert flow._progress()['embeddings_ref']['source_id'] == manifest['source_id']
    assert not flow._checkpoint_files()  # raw vectors are not periodic checkpoint files
    assert backend.upload_calls == 1


def test_manual_source_publish_marks_local_shards_ready(tmp_path):
    config = configuration(tmp_path)
    paths = shards(tmp_path)
    config['shards'] = str(paths[0].parent / 'passages_*')
    store, _, clock = storage(config, tmp_path)
    flow = Workflow(config, store, clock)
    flow.publish_embeddings()
    report = inspect(config)
    assert report['steps']['extract-embeddings'] == 'complete'
    assert report['steps']['publish-embeddings'] == 'complete'
