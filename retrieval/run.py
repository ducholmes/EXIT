"""Query EXIT or export resumable pre-retrieved evaluation records."""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from .runtime import ContrieverRetriever
from .storage import digest, write_json


def export_records(retriever, source, output, top_k=20, batch_size=16, deadline=lambda: False):
    """Batch files are the commit log; a final JSONL is assembled atomically."""
    source, output = Path(source), Path(output)
    folder = output.with_name(output.name + '.batches')
    folder.mkdir(parents=True, exist_ok=True)
    identity = {'input': digest(source), 'top_k': top_k, 'batch_size': batch_size,
                'artifact': retriever.fingerprint}
    config = folder / 'config.json'
    if config.exists() and json.loads(config.read_text()) != identity:
        raise ValueError('Resume input/configuration changed; select a new output')
    write_json(config, identity)
    if source.suffix == '.json':
        records = iter(json.loads(source.read_text(encoding='utf-8')))
        stream = None
    else:
        stream = source.open(encoding='utf-8')
        records = (json.loads(line) for line in stream if line.strip())
    batch, number = [], 0

    def commit(items, number):
        target = folder / f'{number:09d}.json'
        if not target.exists():
            for record in items:
                if not isinstance(record.get('answers'), list):
                    raise ValueError('Each record must contain an answers list')
            hits = retriever.retrieve_batch([item['question'] for item in items], top_k)
            result = []
            for record, passages in zip(items, hits):
                ctxs = [{'id': p.doc_id, 'title': p.title, 'text': p.text, 'score': p.score} for p in passages]
                result.append({**record, 'ctxs': ctxs})
            write_json(target, result)

    try:
        for record in records:
            batch.append(record)
            if len(batch) == batch_size:
                commit(batch, number)
                number += 1
                batch = []
                if deadline():
                    return False
        if batch:
            commit(batch, number)
            number += 1
        temporary = output.with_name(output.name + '.tmp')
        with temporary.open('w', encoding='utf-8') as dst:
            for i in range(number):
                for record in json.loads((folder / f'{i:09d}.json').read_text(encoding='utf-8')):
                    dst.write(json.dumps(record, ensure_ascii=False) + '\n')
        temporary.replace(output)
        return True
    finally:
        if stream:
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['query', 'export'])
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--query')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/retrieved.jsonl'))
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--retriever-device')
    parser.add_argument('--exit-device', default=None)
    parser.add_argument('--reader-backend', choices=['gemini', 'transformers'], default='gemini')
    parser.add_argument('--reader-model', default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--gemini-config', type=Path, default=Path('configs/gemini_api.json'))
    parser.add_argument('--no-compressor', action='store_true')
    parser.add_argument('--retrieval-only', action='store_true')
    parser.add_argument('--time-budget-minutes', type=float, default=600)
    args = parser.parse_args()
    if args.top_k <= 0 or args.batch_size <= 0 or args.time_budget_minutes <= 0:
        parser.error('top-k, batch-size and time budget must be positive')
    if args.mode == 'export' and not args.input:
        parser.error('--input required')
    if args.mode == 'query' and not args.query:
        parser.error('--query required')
    started = time.monotonic()
    retriever = ContrieverRetriever(args.artifact, args.retriever_device)
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.mode == 'export':
            complete = export_records(retriever, args.input, args.output, args.top_k, args.batch_size,
                                      lambda: time.monotonic() - started >= args.time_budget_minutes * 60)
            print('Export complete' if complete else 'Checkpoint saved; resume with the same command')
            return
        retrieval_start = time.monotonic()
        hits = retriever.retrieve(args.query, args.top_k)
        retrieval_time = time.monotonic() - retrieval_start
        result = {'query': args.query, 'passages': [asdict(hit) for hit in hits], 'retrieval_time': retrieval_time}
        if not args.retrieval_only:
            import torch
            from exit_rag import Document, ExitRAG
            kwargs = {}
            if args.reader_backend == 'gemini':
                from gemini_reader import GeminiQuota
                config = json.loads(args.gemini_config.read_text())
                kwargs = {'gemini_model': config['model'], 'gemini_project_id': config.get('project_id'),
                          'gemini_quota': GeminiQuota(**config['quotas']),
                          'gemini_quota_state': Path(config['quota_state'])}
            device = args.exit_device or ('cuda:1' if torch.cuda.device_count() > 1 else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
            rag = ExitRAG(device=device, reader_backend=args.reader_backend, reader_model=args.reader_model,
                          use_compressor=not args.no_compressor, **kwargs)
            inference_start = time.monotonic()
            result.update(rag.run_rag(args.query, [Document(h.title, h.text, h.score) for h in hits]))
            result['pipeline_total_time'] = retrieval_time + time.monotonic() - inference_start
        write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        retriever.close()


if __name__ == '__main__':
    main()
