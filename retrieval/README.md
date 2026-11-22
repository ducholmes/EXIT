# Wikipedia retrieval on Kaggle

Run from the repository root. Install `requirements-retrieval.txt`; Kaggle supplies PyTorch. EXIT additionally needs the main requirements and spaCy setup. No data downloads happen at import time.

Starter notebooks: [preparation](../notebooks/kaggle_retrieval_prepare.ipynb) and [inference](../notebooks/kaggle_retrieval_inference.ipynb). Adjust the checkout and attached Dataset paths before running. Run preparation cells individually rather than blindly running every stage after a time-budget stop.

## Prepare once

```bash
python -m retrieval.prepare preflight --work /tmp/retrieval-state
python -m retrieval.prepare download --source corpus --work /tmp/retrieval-downloads
python -m retrieval.prepare download --source embeddings --work /tmp/retrieval-downloads
python -m retrieval.prepare extract --archive /tmp/retrieval-downloads/psgs_w100.tsv.gz --output /tmp/retrieval-sources
python -m retrieval.prepare extract --archive /tmp/retrieval-downloads/wikipedia_embeddings.tar --output /tmp/retrieval-sources
python -m retrieval.prepare prepare-corpus --tsv /tmp/retrieval-sources/psgs_w100.tsv --corpus /tmp/retrieval-state/corpus.sqlite
```

Locate the actual extracted embedding shard names, typically `wikipedia_embeddings/passages_*`, before running:

```bash
python -m retrieval.prepare sample-pq --shards '/tmp/retrieval-sources/wikipedia_embeddings/passages_*' --work /tmp/retrieval-state/work
python -m retrieval.prepare train-pq --work /tmp/retrieval-state/work
python -m retrieval.prepare build-index --shards '/tmp/retrieval-sources/wikipedia_embeddings/passages_*' --corpus /tmp/retrieval-state/corpus.sqlite --work /tmp/retrieval-state/work --output /tmp/retrieval-artifact
python -m retrieval.prepare verify --output /tmp/retrieval-artifact
```

Use only the [official Contriever archive](https://github.com/facebookresearch/contriever#question-answering-retrieval): shards contain Python pickle. Corpus is DPR Wikipedia December 2018, already split into passages. No re-encoding or rechunking is required.

Defaults: PQ96, 8 bits, inner product, 100,000 training vectors, seed 42, 50,000 vectors/add batch, 4 FAISS CPU threads. GPU is used for query encoding, not this FAISS implementation. Repeat custom dimension/PQ/sample settings across corresponding commands. The initial index build resolves the encoder revision online and records it for inference.

## Resume across sessions

Re-run the same command with the same sources/configuration. Changes are rejected rather than silently mixing artifacts.

| Step | Durable progress |
|---|---|
| Download | `.part` plus HTTP validator; resume only if server supports it |
| Extract | Complete files retained; an interrupted file is extracted again |
| SQLite | Rows and TSV byte position committed together every 10,000 passages |
| Sample PQ | Reservoir, RNG state and next shard; interrupted shard is replayed |
| Train PQ | Completed trained index; interrupted training restarts from saved sample |
| Build index | Index, mapping, next shard/row; last two complete generations retained |
| Export questions | Atomic batch results with input/config/artifact fingerprints |

The default `--time-budget-minutes 600` stops import/build/export at safe boundaries. Build checkpoints occur after each shard or `--checkpoint-minutes 30`, checked between batches. Hashing, extraction, sampling one shard, PQ training and checkpoint writes can exceed these boundaries; reserve time to save outputs. A pickle shard must fit RAM when decoded even though the final PQ index is small. Inspect shard sizes and benchmark first.

Scratch files do not survive automatically. Save the state folder before ending your Kaggle session; do not include raw downloads in it. Source TSV/shards must also be preserved separately or downloaded again when resuming.

```bash
python -m retrieval.package pack --source /tmp/retrieval-state --output /kaggle/working/checkpoint-package
python -m retrieval.package restore --source /kaggle/input/checkpoint-part-0/package.json --parts /kaggle/input --output /tmp/retrieval-state
```

Parts default to 8 GiB. Splitting does not increase Kaggle's total output allowance. If total output is too large, `--select-part 0` writes only that part; repeat for subsequent parts from the **same immutable source snapshot**, saving separate sessions. All source bytes are streamed for each selected part to compute the same manifest. If the source cannot be preserved across those sessions, use persistent external storage or a larger preparation machine. The tools do not automatically save notebook outputs or publish datasets.

Restore verifies hashes and searches parts recursively. Use a dedicated parts directory if multiple packages with the same filenames are attached. Copy restored state to a writable path; `/kaggle/input` is read-only.

The final artifact contains `corpus.sqlite`, `index.faiss`, `index_meta.faiss`, `manifest.json`. Package it similarly, or attach these unpacked files as a Kaggle Dataset. Inference can then read them directly from Input, without keeping raw embeddings or rebuilding PQ. Hashes are local fingerprints, not publisher-provided authenticity checks.

## Query and export

```bash
python -m retrieval.run query --artifact /kaggle/input/exit-retrieval --query 'Who invented the telephone?' --retrieval-only --output /kaggle/working/hits.json
python -m retrieval.run query --artifact /kaggle/input/exit-retrieval --query 'Who invented the telephone?' --retriever-device cuda:0 --exit-device cuda:1 --reader-backend gemini --output /kaggle/working/answer.json
python -m retrieval.run export --artifact /kaggle/input/exit-retrieval --input /kaggle/input/questions/dev.jsonl --output /kaggle/working/retrieved.jsonl --batch-size 16 --top-k 20
```

Gemini uses `configs/gemini_api.json`; supply credentials with Kaggle Secrets as described by the existing reader. Local readers use `--reader-backend transformers`; their VRAM requirements need separate validation. No automatic multi-GPU model distribution is added. Contriever defaults to CUDA 0; EXIT uses CUDA 1 when available. Existing EXIT timing fields are unchanged; `pipeline_total_time` includes retrieval and inference but excludes model initialization.

Export accepts JSON/JSONL with `question` and an `answers` list. It preserves other fields and adds `ctxs` with `id/title/text/score`, compatible with `reproduce.py`. Preserve the output `.batches/` directory and its config across sessions, then restore it beside the output path to resume. Changing input, artifact, top-k or batch size requires a fresh output.

```python
from retrieval.runtime import ContrieverRetriever
retriever = ContrieverRetriever('/kaggle/input/exit-retrieval', device='cuda:0')
hits = retriever.retrieve('Who invented the telephone?', top_k=20)
# Each hit: doc_id, title, text, score
retriever.close()
```

## Validation

Run `python -m pytest tests/test_retrieval_new.py`. Tests use synthetic vectors and a fake encoder without downloading models. Begin with 10,000 matching passages/vectors, then benchmark one official shard for peak RAM, wall time, disk use and query latency. PQ codes alone are around 1.9 GiB for 21M passages; ID mapping, training/sample/shard buffers and SQLite add overhead. IndexPQ scans compressed codes; full-corpus latency and answer-hit@5/@20 must be measured. Synthetic tests do not establish paper-equivalent accuracy.
