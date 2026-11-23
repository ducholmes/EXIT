"""Build an ID-matched small fixture from trusted, real DPR/Contriever inputs."""
import csv
import gzip
import pickle
import tarfile
from pathlib import Path

from .prepare import load_shard
from .progress import activity, copy_stream
from .storage import Corpus, write_json


def extract_first_shard(archive_path, output):
    """Extract only one official shard, never the whole embedding archive."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as archive:
        for member in archive:
            name = Path(member.name).name
            if member.isfile() and name.startswith(('passages_', 'wiki_')) and not name.endswith(('.json', '.faiss', '.tmp')):
                target = output / name
                if not target.exists() or target.stat().st_size != member.size:
                    temporary = target.with_name(target.name + '.tmp')
                    with archive.extractfile(member) as src, temporary.open('wb') as dst:
                        copy_stream(src, dst, f'Extracting one test shard: {name}')
                    temporary.replace(target)
                return target
    raise ValueError('Archive contains no recognized embedding shard')


def subset(shard, corpus_source, output, count=10000, dimension=768):
    """Retain first N vectors and exactly their passage IDs, in embedding order."""
    if count <= 0:
        raise ValueError('count must be positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'small.tsv').exists() or (output / 'small.pkl').exists():
        raise ValueError('Choose a fresh small-data directory to avoid mixing runs')
    ids, vectors = load_shard(shard, dimension)
    if len(ids) < count:
        raise ValueError(f'Only {len(ids)} vectors in shard; requested {count}')
    selected = ids[:count]
    if len(set(selected)) != count:
        raise ValueError('Selected embedding IDs contain duplicates')
    # Copy releases the full shard after loading; pickle itself is not streamable.
    small_vectors = vectors[:count].astype('float32', copy=True)
    del ids, vectors
    source = Path(corpus_source)
    found = {}
    if source.suffix in ('.sqlite', '.db'):
        store = Corpus(source)
        try:
            found = {row[0]: row for row in store.get_many(selected)}
        finally:
            store.close()
    else:
        wanted = set(selected)
        opener = gzip.open if source.suffix == '.gz' else open
        with activity('Scanning corpus for selected IDs', lambda: f'{len(found):,}/{count:,} found'), opener(source, 'rt', encoding='utf-8', newline='') as file:
            reader = csv.reader(file, delimiter='\t')
            if next(reader) != ['id', 'text', 'title']:
                raise ValueError('Expected DPR TSV header: id, text, title')
            for row in reader:
                if len(row) != 3:
                    raise ValueError('Invalid corpus record')
                if row[0] in wanted:
                    if row[0] in found:
                        raise ValueError(f'Duplicate corpus ID: {row[0]}')
                    found[row[0]] = (row[0], row[2], row[1])
                    if len(found) == count:
                        break
    missing = [identifier for identifier in selected if identifier not in found]
    if missing:
        raise ValueError(f'Missing {len(missing)} corpus IDs, e.g. {missing[:5]}')
    tsv = output / 'small.tsv'
    with tsv.open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['id', 'text', 'title'])
        for identifier in selected:
            _, title, text = found[identifier]
            writer.writerow([identifier, text, title])
    with (output / 'small.pkl').open('wb') as file:
        pickle.dump((selected, small_vectors), file, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(output / 'subset.json', {'count': count, 'dimension': dimension,
               'shard': str(Path(shard)), 'corpus_source': str(source)})
    print(f'Small fixture ready: {count:,} ID-matched passages and vectors', flush=True)
    return selected, small_vectors
