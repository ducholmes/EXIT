# Wikipedia retrieval on Kaggle

Run from the repository root. Install `requirements-retrieval.txt`; Kaggle supplies PyTorch. EXIT additionally needs the main requirements and spaCy setup. No data downloads happen at import time.

Starter notebooks: [preparation](../notebooks/kaggle_retrieval_prepare.ipynb) and [inference](../notebooks/kaggle_retrieval_inference.ipynb). Adjust the checkout and attached Dataset paths before running. The preparation notebook now uses a config-driven status/restore/resume workflow with direct public Dataset uploads.

## Automatic Kaggle Dataset workflow (recommended)

Edit `configs/retrieval_kaggle.json`: set `owner` to your Kaggle username and choose `dataset_prefix`. All three Datasets are **public**. The default license metadata is `other`; the published description identifies the DPR/Contriever sources and Wikipedia terms. Set a more specific compatible license if appropriate for your redistribution.

Authenticate in a Kaggle notebook using Secrets (never put credentials in the config):

```python
import os
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
os.environ['KAGGLE_API_TOKEN'] = secrets.get_secret('KAGGLE_API_TOKEN')
# Alternatively, for a legacy key:
# os.environ['KAGGLE_USERNAME'] = secrets.get_secret('KAGGLE_USERNAME')
# os.environ['KAGGLE_KEY'] = secrets.get_secret('KAGGLE_KEY')
```

The workflow defaults to scratch paths `/tmp/retrieval-state`, `/tmp/retrieval-downloads` and `/tmp/retrieval-sources`. Existing legacy corpus/sample/trained/index state under `retrieval-state` is detected. Final artifacts for the automatic workflow are stored in `/tmp/retrieval-state/artifact`. Existing complete artifacts outside that root remain usable by the manual inference commands below.

```bash
python -m retrieval.prepare status --config configs/retrieval_kaggle.json
python -m retrieval.prepare resume --config configs/retrieval_kaggle.json
```

`status` is read-only and does not authenticate or upload. It reports local stage completion, passage/shard/vector progress, the next step, error type, and the last confirmed remote Dataset versions. `--json` gives machine-readable output. A currently running process also writes lightweight `pipeline.json` progress that status can read from another cell/process.

`resume` runs only unfinished steps in dependency order. It downloads missing sources, extracts files, imports SQLite, samples/trains PQ, builds/verifies the index and publishes the artifact. The official `passages_*` and `wiki_*` shard prefixes are auto-detected. Set `tsv`/`shards` explicitly for attached sources or unusual layouts. Raw inputs are excluded from snapshots: if scratch sources disappear across sessions, they are downloaded/extracted again. This means download progress is local-only across sessions unless you separately preserve the `.part` files.

Every **120 minutes** of processing, the workflow pauses at a safe batch boundary, saves local state and uploads it. It also saves at stage completion and on the default 600-minute processing budget. It no longer serializes the full FAISS index after every shard. Progress commits in SQLite remain frequent and inexpensive. Training, pickle loading, hashing and a single file extraction are not interruptible mid-operation, so a checkpoint can be delayed until those operations finish. Interrupted PQ training restarts from the saved sample.

Three public Datasets are used:

| Dataset suffix | Files |
|---|---|
| `-corpus` | Consistent SQLite backup with import progress |
| `-checkpoint` | Pipeline progress, sample/RNG, trained PQ, latest index generation and exact corpus Dataset/version reference |
| `-artifact` | `corpus.sqlite`, `index.faiss`, `index_meta.faiss`, `manifest.json` plus the upload snapshot manifest |

Completed corpus is not uploaded again for each index checkpoint. No credentials, raw archives, raw embedding shards, temporary files, or arbitrary repo files are uploaded. Snapshots are copied to immutable staging directories under the scratch root. Plan enough scratch space for the corpus/artifact plus their snapshot copies.

Upload uses the official Kaggle client, preserves binary/tabular files, creates or versions the configured Datasets, waits for ready status, and verifies the exact snapshot manifest is downloadable. Only then does it record success in `remote.json`. It retries up to 3 times and checks for an accepted snapshot after ambiguous network failures to avoid duplicate versions. Upload timeout defaults to 90 minutes. Failed snapshots are retained locally and the pipeline stops. No Dataset or old version is automatically deleted. The configured size guard defaults to 200 GB per Dataset; server quota/permission errors still apply.

Save immediately outside the periodic schedule:

```bash
python -m retrieval.prepare checkpoint --config configs/retrieval_kaggle.json
```

Run this only after the active preparation command has stopped. It uploads the latest durable files; it cannot serialize another process's live in-memory FAISS index. Use one writer for a given root/Dataset prefix.

### Continue in a new session

Load Secrets again, then:

```bash
python -m retrieval.prepare restore --config configs/retrieval_kaggle.json
python -m retrieval.prepare status --config configs/retrieval_kaggle.json
python -m retrieval.prepare resume --config configs/retrieval_kaggle.json
```

Restore obtains the latest ready checkpoint, downloads its pinned corpus version, checks hashes and restores files into writable scratch. If the checkpoint references a completed artifact, it restores that pinned artifact too. It does not start heavy computation automatically. Conflicting local files are rejected; use a fresh scratch root instead of overwriting data.

You can instead attach Datasets with Add Input and configure `checkpoint_input` and `corpus_input` to their unpacked directories. The corpus snapshot must match the exact reference in the checkpoint; otherwise restore stops. Legacy `retrieval.package` archives still use the manual restore command below.

After completion, attach `<owner>/<prefix>-artifact` as Input and use its four artifact files directly for inference. No pack/download-to-PC step is required. Full-corpus upload time and quota need validation on your account; a local checkpoint is not durable across sessions until its Dataset upload is confirmed.

## Manual preparation commands (still supported)

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

## Legacy manual checkpoint packages

Re-run the same command with the same sources/configuration. Changes are rejected rather than silently mixing artifacts.

| Step | Durable progress |
|---|---|
| Download | `.part` plus HTTP validator; resume only if server supports it |
| Extract | Complete files retained; an interrupted file is extracted again |
| SQLite | Rows and TSV byte position committed together every 10,000 passages |
| Sample PQ | Reservoir, RNG state and shard/row position; legacy shard-boundary checkpoints also supported |
| Train PQ | Completed trained index; interrupted training restarts from saved sample |
| Build index | Index, mapping, next shard/row; last two complete generations retained |
| Export questions | Atomic batch results with input/config/artifact fingerprints |

The default `--time-budget-minutes 600` stops import/build/export at safe boundaries. Standalone build checkpoints occur every `--checkpoint-minutes 120` and on stop/completion, checked between batches. Hashing, extraction, sampling one shard, PQ training and checkpoint writes can exceed these boundaries; reserve time to save outputs. A pickle shard must fit RAM when decoded even though the final PQ index is small. Inspect shard sizes and benchmark first.

Scratch files do not survive automatically. Save the state folder before ending your Kaggle session; do not include raw downloads in it. Source TSV/shards must also be preserved separately or downloaded again when resuming.

```bash
python -m retrieval.package pack --source /tmp/retrieval-state --output /kaggle/working/checkpoint-package
python -m retrieval.package restore --source /kaggle/input/checkpoint-part-0/package.json --parts /kaggle/input --output /tmp/retrieval-state
```

Parts default to 8 GiB. Splitting does not increase Kaggle's total output allowance. If total output is too large, `--select-part 0` writes only that part; repeat for subsequent parts from the **same immutable source snapshot**, saving separate sessions. All source bytes are streamed for each selected part to compute the same manifest. If the source cannot be preserved across those sessions, use persistent external storage or a larger preparation machine. This legacy pack command does not publish Datasets; use the automatic workflow above for direct upload.

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

Run `python -m pytest tests/test_retrieval_new.py tests/test_retrieval_workflow.py`. Tests use synthetic vectors and a fake encoder without downloading models. Begin with 10,000 matching passages/vectors, then benchmark one official shard for peak RAM, wall time, disk use and query latency. PQ codes alone are around 1.9 GiB for 21M passages; ID mapping, training/sample/shard buffers and SQLite add overhead. IndexPQ scans compressed codes; full-corpus latency and answer-hit@5/@20 must be measured. Synthetic tests do not establish paper-equivalent accuracy.
