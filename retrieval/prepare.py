"""Kaggle-friendly artifact preparation: python -m retrieval.prepare --help."""
import argparse
import glob
import gzip
import json
import os
import pickle
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

from .storage import Corpus, digest, prepare_corpus, write_json

SOURCES = {
    'corpus': 'https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz',
    'embeddings': 'https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar',
}


def download(url, target, deadline, on_progress=None):
    target = Path(target)
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + '.part')
    meta = part.with_name(part.name + '.json')
    old = json.loads(meta.read_text()) if meta.exists() else {}
    size = part.stat().st_size if part.exists() else 0
    headers = {}
    if size and old.get('url') == url and old.get('validator'):
        headers = {'Range': f'bytes={size}-', 'If-Range': old['validator']}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
        append = response.status == 206 and response.headers.get('Content-Range', '').startswith(f'bytes {size}-')
        write_json(meta, {'url': url, 'validator': response.headers.get('ETag') or response.headers.get('Last-Modified')})
        expected = response.headers.get('Content-Length')
        received = 0
        with part.open('ab' if append else 'wb') as out:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                out.write(block)
                received += len(block)
                if on_progress:
                    out.flush()
                    on_progress({'file': target.name, 'bytes': out.tell()})
                if deadline():
                    return
        if expected is not None and received != int(expected):
            raise ValueError('Incomplete HTTP response')
    os.replace(part, target)
    write_json(target.with_name(target.name + '.json'), {'url': url, 'sha256': digest(target)})


def extract(source, output, on_progress=None):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if source.suffix == '.gz':
        target = output / source.stem
        if not target.exists():
            temporary = target.with_name(target.name + '.tmp')
            with gzip.open(source, 'rb') as src, temporary.open('wb') as dst:
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            os.replace(temporary, target)
            if on_progress:
                on_progress({'file': target.name})
    else:
        with tarfile.open(source) as archive:
            for member in archive:
                target = (output / member.name).resolve()
                if not target.is_relative_to(output.resolve()) or not (member.isfile() or member.isdir()):
                    raise ValueError(f'Unsafe archive entry: {member.name}')
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if target.exists() and target.stat().st_size == member.size:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + '.tmp')
                with archive.extractfile(member) as src, temporary.open('wb') as dst:
                    shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
                os.replace(temporary, target)
                if on_progress:
                    on_progress({'file': member.name})


def load_shard(path, dimension):
    import numpy as np
    # Only load the trusted official archive: its shards use Python pickle.
    with open(path, 'rb') as source:
        ids, vectors = pickle.load(source)
    vectors = np.asarray(vectors)
    if vectors.ndim != 2 or vectors.shape != (len(ids), dimension) or not np.isfinite(vectors).all():
        raise ValueError(f'Invalid embedding shard: {path}')
    return [str(x) for x in ids], vectors


def shard_manifest(pattern):
    files = sorted(Path(p) for p in glob.glob(pattern))
    if not files:
        raise ValueError('No embedding shards matched')
    return [{'name': p.name, 'size': p.stat().st_size, 'sha256': digest(p)} for p in files], files


def save_sample(path, state):
    temporary = Path(str(path) + '.tmp')
    with temporary.open('wb') as output:
        pickle.dump(state, output, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)
    write_json(Path(path).with_suffix('.json'), {k: v for k, v in state.items() if k not in ('sample', 'rng')})


def sample(args, deadline):
    import numpy as np
    manifest, paths = shard_manifest(args.shards)
    state_path = args.work / 'sample.pkl'
    state = pickle.loads(state_path.read_bytes()) if state_path.exists() else {
        'shards': manifest, 'next_shard': 0, 'seen': 0, 'size': args.sample_size,
        'dimension': args.dimension, 'seed': args.seed,
        'sample': np.empty((args.sample_size, args.dimension), dtype='float32'),
        'rng': np.random.default_rng(args.seed).bit_generator.state,
    }
    if (state['shards'], state['size'], state['dimension'], state['seed']) != (manifest, args.sample_size, args.dimension, args.seed):
        raise ValueError('Sampling configuration/source changed')
    rng = np.random.default_rng()
    rng.bit_generator.state = state['rng']
    control = getattr(args, 'control', None)
    for i in range(state['next_shard'], len(paths)):
        _, vectors = load_shard(paths[i], args.dimension)
        # Vectorized reservoir updates, applying replacements in stream order.
        first = state.get('offset', 0) if i == state['next_shard'] else 0
        for start in range(first, len(vectors), 50000):
            block = vectors[start:start + 50000]
            seen = state['seen']
            fill = min(len(block), max(0, args.sample_size - seen))
            state['sample'][seen:seen + fill] = block[:fill]
            rest = block[fill:]
            if len(rest):
                positions = rng.integers(0, np.arange(seen + fill + 1, seen + len(block) + 1))
                for row in np.flatnonzero(positions < args.sample_size):
                    state['sample'][positions[row]] = rest[row]
            state['seen'] += len(block)
            state.update(next_shard=i, offset=start + len(block), rng=rng.bit_generator.state)
            save = lambda: save_sample(state_path, state)
            if control:
                control.tick(save, {'shard': i, 'offset': state['offset'], 'seen': state['seen']})
            if deadline():
                save()
                return
        state.update(next_shard=i + 1, offset=0, rng=rng.bit_generator.state)
        if not control:
            save_sample(state_path, state)
        del vectors
        if deadline():
            save_sample(state_path, state)
            return
    save_sample(state_path, state)
    print(f"Sample ready: {min(state['seen'], args.sample_size)} vectors")


def train(args):
    from utils.src.index import Indexer
    state = pickle.loads((args.work / 'sample.pkl').read_bytes())
    if state['next_shard'] != len(state['shards']) or state['dimension'] != args.dimension:
        raise ValueError('Sampling is incomplete or dimension differs')
    target = args.work / 'trained'
    config = {'dimension': args.dimension, 'pq_m': args.pq_m, 'bits': args.bits,
              'sample_sha256': digest(args.work / 'sample.pkl'), 'seed': args.seed}
    if (target / 'complete.json').exists():
        if json.loads((target / 'complete.json').read_text()) != config:
            raise ValueError('Training configuration changed; use a new work directory')
        return
    if args.dimension % args.pq_m or min(state['seen'], state['size']) < 2 ** args.bits:
        raise ValueError('Invalid PQ dimensions or too few training vectors')
    index = Indexer(args.dimension, args.pq_m, args.bits)
    index.index.pq.cp.seed = args.seed
    index.index.train(state['sample'][:min(state['seen'], state['size'])])
    index.serialize(str(target))
    write_json(target / 'complete.json', config)


def latest_checkpoint(work):
    candidates = sorted((work / 'checkpoints').glob('generation-*'), reverse=True)
    for candidate in candidates:
        if all((candidate / name).exists() for name in ('complete.json', 'index.faiss', 'index_meta.faiss')):
            return candidate
    return None


def checkpoint(index, work, state):
    parent = work / 'checkpoints'
    parent.mkdir(exist_ok=True)
    target = parent / f"generation-{time.time_ns():020d}"
    target.mkdir()
    index.serialize(str(target))
    write_json(target / 'complete.json', state)
    # Keep the newest two valid generations.
    generations = sorted(p for p in parent.glob('generation-*') if (p / 'complete.json').exists())
    for old in generations[:-2]:
        shutil.rmtree(old)
    return target


def build(args, deadline):
    import importlib.metadata
    from huggingface_hub import model_info
    from utils.src.index import Indexer
    manifest, paths = shard_manifest(args.shards)
    trained = args.work / 'trained'
    training = json.loads((trained / 'complete.json').read_text())
    if training['dimension'] != args.dimension:
        raise ValueError('Build dimension differs from trained index')
    corpus = Corpus(args.corpus)
    row = corpus.connection.execute("SELECT value FROM progress WHERE key='state'").fetchone()
    corpus_state = json.loads(row[0])
    if not corpus_state['complete']:
        raise ValueError('Corpus import is incomplete')
    identity = {'shards': manifest, 'training': training, 'corpus': corpus_state['source']}
    previous = latest_checkpoint(args.work)
    index = Indexer(args.dimension)
    index.deserialize_from(str(previous or trained))
    if previous:
        state = json.loads((previous / 'complete.json').read_text())
        if state['identity'] != identity:
            raise ValueError('Checkpoint inputs changed')
    else:
        state = {'identity': identity, 'shard': 0, 'offset': 0,
                 'model_revision': model_info('facebook/contriever-msmarco').sha}
    saved_at = time.monotonic()
    control = getattr(args, 'control', None)
    try:
        for i in range(state['shard'], len(paths)):
            ids, vectors = load_shard(paths[i], args.dimension)
            start = state['offset'] if i == state['shard'] else 0
            for offset in range(start, len(ids), args.batch_size):
                end = min(offset + args.batch_size, len(ids))
                corpus.get_many(ids[offset:end])
                index.index_data(ids[offset:end], vectors[offset:end])
                state.update(shard=i, offset=end, vectors=index.index.ntotal)
                if control:
                    control.tick(lambda: checkpoint(index, args.work, state), dict(state, identity=None))
                if deadline() or (not control and time.monotonic() - saved_at >= args.checkpoint_minutes * 60):
                    checkpoint(index, args.work, state)
                    saved_at = time.monotonic()
                    if deadline():
                        return
            state.update(shard=i + 1, offset=0)
            del ids, vectors
        checkpoint(index, args.work, state)
        args.output.mkdir(parents=True, exist_ok=True)
        if (args.output / 'manifest.json').exists():
            raise ValueError('Output artifact already exists; select a new output directory')
        index.serialize(str(args.output))
        shutil.copyfile(args.corpus, args.output / 'corpus.sqlite')
        versions = {p: importlib.metadata.version(p) for p in ('numpy', 'faiss-cpu', 'transformers')}
        result = {'complete': True, 'dimension': args.dimension, 'vectors': index.index.ntotal,
                  'passages': corpus_state['count'], 'model': 'facebook/contriever-msmarco',
                  'model_revision': state['model_revision'], 'identity': identity, 'versions': versions,
                  'sources': SOURCES}
        result['files'] = {name: digest(args.output / name) for name in ('corpus.sqlite', 'index.faiss', 'index_meta.faiss')}
        write_json(args.output / 'manifest.json', result)
    finally:
        corpus.close()


def verify(path):
    from utils.src.index import Indexer
    manifest = json.loads((path / 'manifest.json').read_text())
    if not manifest['complete']:
        raise ValueError('Incomplete artifact')
    for name, expected in manifest['files'].items():
        if digest(path / name) != expected:
            raise ValueError(f'Hash mismatch: {name}')
    index = Indexer(manifest['dimension'])
    index.deserialize_from(str(path))
    corpus = Corpus(path / 'corpus.sqlite')
    try:
        if corpus.connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('SQLite integrity check failed')
        if corpus.connection.execute('SELECT COUNT(*) FROM passages').fetchone()[0] != manifest['passages']:
            raise ValueError('Corpus passage count mismatch')
        if index.index.ntotal != manifest['vectors']:
            raise ValueError('Vector count mismatch')
        for start in range(0, index.index.ntotal, 10000):
            corpus.get_many(index.index_id_to_db_id[start:start + 10000])
    finally:
        corpus.close()
    print('Artifact verified')


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ('check-api', 'status', 'resume', 'checkpoint', 'restore'):
        from .workflow import main as workflow_main
        return workflow_main()
    parser = argparse.ArgumentParser(description=__doc__,
        epilog='Automatic workflow: check-api / status / resume / checkpoint / restore --config CONFIG.json')
    parser.add_argument('step', choices=['preflight', 'download', 'extract', 'prepare-corpus', 'sample-pq', 'train-pq', 'build-index', 'verify'])
    parser.add_argument('--work', type=Path, default=Path('data/retrieval/work'))
    parser.add_argument('--output', type=Path, default=Path('data/retrieval/artifact'))
    parser.add_argument('--source', choices=SOURCES, default='corpus')
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--tsv', type=Path)
    parser.add_argument('--corpus', type=Path)
    parser.add_argument('--shards')
    parser.add_argument('--dimension', type=int, default=768)
    parser.add_argument('--sample-size', type=int, default=100000)
    parser.add_argument('--pq-m', type=int, default=96)
    parser.add_argument('--bits', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=50000)
    parser.add_argument('--checkpoint-minutes', type=float, default=120)
    parser.add_argument('--time-budget-minutes', type=float, default=600)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if min(args.batch_size, args.sample_size, args.dimension, args.pq_m, args.checkpoint_minutes, args.time_budget_minutes, args.threads) <= 0:
        parser.error('Sizes, budgets and threads must be positive')
    if not 1 <= args.bits <= 16:
        parser.error('PQ bits must be between 1 and 16')
    args.work.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = lambda: time.monotonic() - started >= args.time_budget_minutes * 60
    if args.step in ('sample-pq', 'train-pq', 'build-index', 'verify'):
        import faiss
        faiss.omp_set_num_threads(args.threads)
    if args.step == 'preflight':
        import platform
        info = {'platform': platform.platform(), 'cpu_count': os.cpu_count(),
                'work_disk_free_gib': shutil.disk_usage(args.work).free / 1024 ** 3,
                'note': 'Hashing, extraction, PQ training and checkpoint writes need time beyond batch boundaries.'}
        try:
            import psutil
            info['available_ram_gib'] = psutil.virtual_memory().available / 1024 ** 3
        except ImportError:
            pass
        if args.shards:
            info['shard_bytes'] = {p: Path(p).stat().st_size for p in sorted(glob.glob(args.shards))}
        print(json.dumps(info, indent=2))
    elif args.step == 'download':
        url = SOURCES[args.source]
        download(url, args.work / url.rsplit('/', 1)[1], deadline)
    elif args.step == 'extract':
        if not args.archive:
            parser.error('--archive required')
        extract(args.archive, args.output)
    elif args.step == 'prepare-corpus':
        if not args.tsv or not args.corpus:
            parser.error('--tsv and --corpus required')
        print(prepare_corpus(args.tsv, args.corpus, deadline))
    elif args.step == 'sample-pq':
        if not args.shards:
            parser.error('--shards required')
        sample(args, deadline)
    elif args.step == 'train-pq':
        train(args)
    elif args.step == 'build-index':
        if not args.shards or not args.corpus:
            parser.error('--shards and --corpus required')
        build(args, deadline)
    else:
        verify(args.output)


if __name__ == '__main__':
    main()
