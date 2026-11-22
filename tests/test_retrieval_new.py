"""Offline retrieval tests; no downloads or model dependencies."""
import csv
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval.storage import Corpus, prepare_corpus, write_json
from retrieval.prepare import sample, train, load_shard, checkpoint, latest_checkpoint
from retrieval.prepare import build, verify
from retrieval.package import pack, restore
from retrieval.run import export_records
from retrieval.runtime import ContrieverRetriever, RetrievedPassage
from utils.src.index import Indexer

faiss.omp_set_num_threads(1)


def corpus_fixture(tmp_path, count=3):
    tsv = tmp_path / 'passages.tsv'
    with tsv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['id', 'text', 'title'])
        for i in range(count):
            writer.writerow([str(i), f'text\t{i}\nnext line', 'Tiêu đề'])
    db = tmp_path / 'corpus.sqlite'
    return tsv, db


def test_corpus_resume(tmp_path):
    tsv, db = corpus_fixture(tmp_path)
    first = prepare_corpus(tsv, db, lambda: True, batch_size=1)
    assert first['count'] == 1 and not first['complete']
    final = prepare_corpus(tsv, db, batch_size=1)
    assert final['count'] == 3 and final['complete']
    store = Corpus(db)
    assert store.get_many(['2', '0'])[0] == ('2', 'Tiêu đề', 'text\t2\nnext line')
    with pytest.raises(ValueError):
        store.get_many(['missing'])
    store.close()


def test_invalid_corpus(tmp_path):
    tsv, db = corpus_fixture(tmp_path)
    with tsv.open('a') as f:
        f.write('0\tduplicate\ttitle\n')
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        prepare_corpus(tsv, db)


def test_index_validation_roundtrip(tmp_path):
    index = Indexer(4)
    queries = np.eye(4, dtype='float32')[:2]
    assert index.search_knn(queries, 20) == [([], []), ([], [])]
    with pytest.raises(ValueError):
        index.index_data(['a'], queries)
    assert index.index_id_to_db_id == []
    index.index_data(['a', 'b'], queries)
    before = index.search_knn(queries, 20)
    assert before[0][0] == ['a', 'b']
    index.serialize(str(tmp_path))
    restored = Indexer(4)
    restored.deserialize_from(str(tmp_path))
    after = restored.search_knn(queries, 20)
    assert after[0][0] == before[0][0]
    np.testing.assert_array_equal(after[0][1], before[0][1])
    with pytest.raises(ValueError):
        index.search_knn(queries, 0)


def test_sample_resume_pq_checkpoint(tmp_path):
    rng = np.random.default_rng(3)
    for i in range(2):
        with (tmp_path / f'shard-{i}').open('wb') as f:
            pickle.dump(([str(i * 300 + j) for j in range(300)], rng.normal(size=(300, 8)).astype('float32')), f)
    args = SimpleNamespace(shards=str(tmp_path / 'shard-*'), work=tmp_path,
                           sample_size=300, dimension=8, seed=42, pq_m=2, bits=2)
    sample(args, lambda: True)
    sample(args, lambda: False)
    resumed = pickle.loads((tmp_path / 'sample.pkl').read_bytes())['sample'].copy()
    (tmp_path / 'sample.pkl').unlink()
    sample(args, lambda: False)
    np.testing.assert_array_equal(resumed, pickle.loads((tmp_path / 'sample.pkl').read_bytes())['sample'])
    train(args)
    index = Indexer(8)
    index.deserialize_from(str(tmp_path / 'trained'))
    ids, vectors = load_shard(tmp_path / 'shard-0', 8)
    index.index_data(ids, vectors)
    generation = checkpoint(index, tmp_path, {'offset': 300})
    (tmp_path / 'checkpoints' / 'generation-999999999999999999999').mkdir()
    assert latest_checkpoint(tmp_path) == generation
    restored = Indexer(8)
    restored.deserialize_from(str(generation))
    np.testing.assert_array_equal(index.search_knn(vectors[:1], 5)[0][1], restored.search_knn(vectors[:1], 5)[0][1])


def test_bad_shard(tmp_path):
    path = tmp_path / 'bad'
    path.write_bytes(pickle.dumps((['a'], np.zeros((2, 8)))))
    with pytest.raises(ValueError):
        load_shard(path, 8)


def test_package_roundtrip_and_corruption(tmp_path):
    source, output, restored = tmp_path / 'source', tmp_path / 'package', tmp_path / 'restored'
    source.mkdir()
    (source / 'test.txt').write_text('hello' * 5000)
    pack(source, output, 4000)
    restore(output / 'package.json', output, restored)
    assert (restored / 'test.txt').read_bytes() == (source / 'test.txt').read_bytes()
    (output / 'part-00000.tar').write_bytes(b'bad')
    with pytest.raises(ValueError):
        restore(output / 'package.json', output, restored)


def test_export_resume(tmp_path):
    class Retriever:
        fingerprint = 'fixture'
        calls = 0
        def retrieve_batch(self, queries, top_k):
            self.calls += 1
            return [[RetrievedPassage('1', 'title', q, 1.0)] for q in queries]
    retriever = Retriever()
    source, output = tmp_path / 'input.jsonl', tmp_path / 'output.jsonl'
    source.write_text('\n'.join(json.dumps({'question': f'q{i}', 'answers': ['a']}) for i in range(3)))
    assert not export_records(retriever, source, output, batch_size=1, deadline=lambda: True)
    assert export_records(retriever, source, output, batch_size=1)
    assert retriever.calls == 3
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 3 and rows[0]['ctxs'][0]['id'] == '1'
    with pytest.raises(ValueError):
        export_records(retriever, source, output, top_k=5, batch_size=1)


def test_retriever_fake_encoder(tmp_path):
    tsv, db = corpus_fixture(tmp_path)
    prepare_corpus(tsv, db)
    index = Indexer(768)
    vectors = np.zeros((3, 768), dtype='float32')
    vectors[:, :3] = np.eye(3)
    index.index_data(['0', '1', '2'], vectors)
    index.serialize(str(tmp_path))
    write_json(tmp_path / 'manifest.json', {'complete': True, 'dimension': 768, 'vectors': 3})
    class Encoder:
        def encode(self, queries):
            return np.repeat(vectors[:1], len(queries), axis=0)
    retriever = ContrieverRetriever(tmp_path, encoder=Encoder())
    assert retriever.retrieve('question', 1)[0].doc_id == '0'
    assert retriever.retrieve_batch([]) == []
    with pytest.raises(ValueError):
        retriever.retrieve(' ')
    retriever.close()


def test_build_resume_complete_artifact(tmp_path, monkeypatch):
    import importlib.metadata
    import types
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(model_info=lambda _: SimpleNamespace(sha='fixture-revision')))
    monkeypatch.setattr(importlib.metadata, 'version', lambda _: 'fixture-version')
    tsv, db = corpus_fixture(tmp_path, 300)
    prepare_corpus(tsv, db)
    vectors = np.random.default_rng(5).normal(size=(300, 8)).astype('float32')
    (tmp_path / 'shard-0').write_bytes(pickle.dumps(([str(i) for i in range(300)], vectors)))
    args = SimpleNamespace(shards=str(tmp_path / 'shard-*'), work=tmp_path / 'work',
                           sample_size=300, dimension=8, seed=42, pq_m=2, bits=2,
                           corpus=db, output=tmp_path / 'artifact', batch_size=100, checkpoint_minutes=30)
    args.work.mkdir()
    sample(args, lambda: False)
    train(args)
    build(args, lambda: True)
    assert not (args.output / 'manifest.json').exists()
    build(args, lambda: False)
    verify(args.output)
    metadata = json.loads((args.output / 'manifest.json').read_text())
    assert metadata['vectors'] == 300
    index = Indexer(8)
    index.deserialize_from(str(args.output))
    assert index.index_id_to_db_id == [str(i) for i in range(300)]


def test_selected_package_matches_full(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'payload').write_bytes(b'x' * 10000)
    full, selected = tmp_path / 'full', tmp_path / 'selected'
    pack(source, full, 4000)
    pack(source, selected, 4000, select=1)
    assert (full / 'package.json').read_bytes() == (selected / 'package.json').read_bytes()
    assert (full / 'part-00001.tar').read_bytes() == (selected / 'part-00001.tar').read_bytes()
    assert not (selected / 'part-00000.tar').exists()
