"""Split/restore artifact or work checkpoints without a second full tar copy."""
import argparse
import io
import hashlib
import json
import tarfile
from pathlib import Path

from .storage import digest, write_json


class SplitWriter:
    def __init__(self, folder, limit, select=None):
        self.folder, self.limit = folder, limit
        self.file = None
        self.size = 0
        self.parts = []
        self.select = select
        self.hashes = []
        self.hasher = None

    def write(self, data):
        total = len(data)
        while data:
            if self.hasher is None or self.size == self.limit:
                if self.file:
                    self.file.close()
                if self.hasher is not None:
                    self.hashes.append({'name': self.parts[-1], 'bytes': self.size, 'sha256': self.hasher.hexdigest()})
                name = f'part-{len(self.parts):05d}.tar'
                self.parts.append(name)
                self.file = (self.folder / name).open('wb') if self.select is None or self.select == len(self.parts) - 1 else None
                self.hasher = hashlib.sha256()
                self.size = 0
            length = min(len(data), self.limit - self.size)
            if self.file:
                self.file.write(data[:length])
            self.hasher.update(data[:length])
            self.size += length
            data = data[length:]
        return total

    def close(self):
        if self.file:
            self.file.close()
        if self.hasher is not None:
            self.hashes.append({'name': self.parts[-1], 'bytes': self.size, 'sha256': self.hasher.hexdigest()})
            self.hasher = None


class JoinedReader(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.file = None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.file is None:
                path = next(self.paths, None)
                if path is None:
                    return 0
                self.file = path.open('rb')
            count = self.file.readinto(buffer)
            if count:
                return count
            self.file.close()
            self.file = None

    def close(self):
        if self.file:
            self.file.close()
        super().close()


def pack(source, output, part_bytes, select=None):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Package output must be empty')
    if select is not None and select < 0:
        raise ValueError('Selected part must be nonnegative')
    writer = SplitWriter(output, part_bytes, select)
    try:
        with tarfile.open(fileobj=writer, mode='w|') as archive:
            for path in sorted(source.rglob('*')):
                if path.is_file() and '.tmp' not in path.name:
                    info = archive.gettarinfo(str(path), arcname=path.relative_to(source).as_posix())
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    info.mtime = 0
                    with path.open('rb') as data:
                        archive.addfile(info, data)
    finally:
        writer.close()
    manifest = {'parts': writer.hashes}
    if select is not None and select >= len(writer.parts):
        raise ValueError('Selected part does not exist')
    write_json(output / 'package.json', manifest)


def restore(manifest_path, parts, output):
    manifest = json.loads(manifest_path.read_text())
    paths = []
    for entry in manifest['parts']:
        matches = list(parts.rglob(entry['name']))
        if len(matches) != 1 or digest(matches[0]) != entry['sha256']:
            raise ValueError(f"Missing, ambiguous or corrupt part: {entry['name']}")
        paths.append(matches[0])
    output.mkdir(parents=True, exist_ok=True)
    with io.BufferedReader(JoinedReader(paths)) as reader, tarfile.open(fileobj=reader, mode='r|') as archive:
        for member in archive:
            target = (output / member.name).resolve()
            if not target.is_relative_to(output.resolve()) or not member.isfile():
                raise ValueError('Unsafe package member')
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.tmp')
            import shutil
            with archive.extractfile(member) as src, temporary.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['pack', 'restore'])
    parser.add_argument('--source', type=Path, required=True, help='Directory to pack, or package.json to restore')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parts', type=Path, help='Search attached inputs recursively for package parts')
    parser.add_argument('--part-gib', type=float, default=8)
    parser.add_argument('--select-part', type=int, help='Keep only this zero-based part for saving one Kaggle session')
    args = parser.parse_args()
    if args.mode == 'pack':
        if args.part_gib <= 0 or args.output.resolve().is_relative_to(args.source.resolve()):
            parser.error('Positive part size and output outside source required')
        pack(args.source, args.output, max(1, int(args.part_gib * 1024 ** 3)), args.select_part)
    else:
        if not args.parts:
            parser.error('--parts required')
        restore(args.source, args.parts, args.output)


if __name__ == '__main__':
    main()
