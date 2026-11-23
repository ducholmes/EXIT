"""The notebook's fixture extraction must preserve passage/vector identity."""
import csv
import gzip
import io
import pickle
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval.smoke import subset, extract_first_shard


def test_small_subset_uses_embedding_order_and_handles_gzip(tmp_path):
    shard = tmp_path / 'wiki_00'
    vectors = np.arange(16, dtype='float32').reshape(4, 4)
    shard.write_bytes(pickle.dumps((['3', '1', '4', '2'], vectors)))
    corpus = tmp_path / 'corpus.tsv.gz'
    with gzip.open(corpus, 'wt', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['id', 'text', 'title'])
        for i in range(1, 5):
            writer.writerow([str(i), f'text\t{i}\nline', f'title {i}'])
    ids, result = subset(shard, corpus, tmp_path / 'small', count=3, dimension=4)
    assert ids == ['3', '1', '4']
    np.testing.assert_array_equal(result, vectors[:3])
    with (tmp_path / 'small/small.tsv').open(encoding='utf-8', newline='') as file:
        rows = list(csv.reader(file, delimiter='\t'))
    assert rows[1] == ['3', 'text\t3\nline', 'title 3']
    with pytest.raises(ValueError, match='fresh'):
        subset(shard, corpus, tmp_path / 'small', 3, 4)


def test_extract_only_one_tar_member(tmp_path):
    archive = tmp_path / 'embeddings.tar'
    with tarfile.open(archive, 'w') as target:
        for name in ('readme.txt', 'folder/wiki_00', 'folder/wiki_01'):
            entry = tarfile.TarInfo(name)
            entry.size = 3
            target.addfile(entry, io.BytesIO(b'abc'))
    result = extract_first_shard(archive, tmp_path / 'extracted')
    assert result.name == 'wiki_00'
    assert [p.name for p in result.parent.iterdir()] == ['wiki_00']


def test_notebook_cells_compile_and_outputs_are_clear():
    import json
    path = Path(__file__).resolve().parents[1] / 'notebooks/kaggle_retrieval_smoke_test.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    assert notebook['nbformat'] == 4
    assert 'Kiểm thử retrieval EXIT' in ''.join(notebook['cells'][0]['source'])
    cell_ids = [cell['id'] for cell in notebook['cells']]
    assert len(set(cell_ids)) == len(cell_ids)
    for i, cell in enumerate(notebook['cells']):
        text = ''.join(cell['source'])
        assert '\ufffd' not in text
        # The only intended question marks are English query punctuation.
        for line in text.splitlines():
            assert '?' not in line or 'What is ' in line
        if cell['cell_type'] == 'code':
            assert cell['outputs'] == [] and cell['execution_count'] is None
            source = ''.join(cell['source'])
            if not source.lstrip().startswith('%'):
                compile(source, f'smoke-cell-{i}', 'exec')
