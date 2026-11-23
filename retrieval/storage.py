"""Disk-backed corpus and atomic progress records (standard library only)."""
import csv
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from .progress import activity


def digest(path):
    h = hashlib.sha256()
    total = Path(path).stat().st_size
    read = 0
    with activity(f'Hashing {Path(path).name}', lambda: f'{read / 1024**3:.2f}/{total / 1024**3:.2f} GiB'), open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
            read += len(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


class Corpus:
    def __init__(self, path):
        self.connection = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)

    def get_many(self, ids):
        found = {}
        for start in range(0, len(ids), 500):
            chunk = [str(x) for x in ids[start:start + 500]]
            if chunk:
                rows = self.connection.execute(
                    'SELECT doc_id,title,text FROM passages WHERE doc_id IN (' + ','.join('?' * len(chunk)) + ')', chunk)
                found.update((row[0], row) for row in rows)
        missing = [str(x) for x in ids if str(x) not in found]
        if missing:
            raise ValueError(f'Corpus missing IDs: {missing[:10]}')
        return [found[str(x)] for x in ids]

    def close(self):
        self.connection.close()


def prepare_corpus(tsv, output, deadline=lambda: False, batch_size=10000, on_progress=None):
    """Resume at CSV record boundaries; progress and rows commit together."""
    tsv, output = Path(tsv), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = {'sha256': digest(tsv), 'bytes': tsv.stat().st_size}
    db = sqlite3.connect(output)
    try:
        db.execute('CREATE TABLE IF NOT EXISTS passages(doc_id TEXT PRIMARY KEY,title TEXT NOT NULL,text TEXT NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS progress(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        state_row = db.execute("SELECT value FROM progress WHERE key='state'").fetchone()
        state = json.loads(state_row[0]) if state_row else {'source': identity, 'offset': 0, 'count': 0, 'complete': False}
        if state['source'] != identity:
            raise ValueError('Corpus source changed; choose a new output path')
        if state['complete']:
            return state
        # Binary lines preserve exact byte offsets, including quoted multiline records.
        with tsv.open('rb') as source:
            source.seek(state['offset'])
            lines = (line.decode('utf-8') for line in iter(source.readline, b''))
            reader = csv.reader(lines, delimiter='\t')
            if state['offset'] == 0:
                header = next(reader)
                if header != ['id', 'text', 'title']:
                    raise ValueError(f'Unexpected TSV header: {header}')
            rows = []
            for record in reader:
                if len(record) != 3 or not record[0]:
                    raise ValueError(f'Invalid passage near byte {source.tell()}')
                rows.append((record[0], record[2], record[1]))
                if len(rows) >= batch_size:
                    state.update(offset=source.tell(), count=state['count'] + len(rows))
                    with db:
                        db.executemany('INSERT INTO passages VALUES(?,?,?)', rows)
                        db.execute('INSERT OR REPLACE INTO progress VALUES(?,?)', ('state', json.dumps(state)))
                    rows = []
                    if on_progress:
                        on_progress(dict(state))
                    if deadline():
                        return state
            state.update(offset=source.tell(), count=state['count'] + len(rows), complete=True)
            with db:
                db.executemany('INSERT INTO passages VALUES(?,?,?)', rows)
                db.execute('INSERT OR REPLACE INTO progress VALUES(?,?)', ('state', json.dumps(state)))
        return state
    finally:
        db.close()
