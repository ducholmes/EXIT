#!/usr/bin/env python3
"""Download HotpotQA from Hugging Face and prepare EXIT train datasets.

The script reuses the query split and 2:1:1 sampling implementation in
``train.datasampling``. It supports the ``distractor`` and ``fullwiki`` configs.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

try:
    from datasets import Dataset, load_dataset
except ImportError as exc:  # pragma: no cover - depends on optional install
    raise SystemExit(
        "This script requires Hugging Face Datasets. Install it with: "
        "pip install 'datasets>=2.14.7'"
    ) from exc

try:
    from train.datasampling import (
        COMPRESSION_PROMPT_TEMPLATE,
        DATASET_SCHEMA_VERSION,
        PAPER_URL,
        create_balanced_dataset,
        process_example,
        split_queries,
        _query_digest,
        _row_digest,
        _write_json,
    )
except ModuleNotFoundError:  # Support ``python train/prepare_hotpot_hf.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train.datasampling import (
        COMPRESSION_PROMPT_TEMPLATE,
        DATASET_SCHEMA_VERSION,
        PAPER_URL,
        create_balanced_dataset,
        process_example,
        split_queries,
        _query_digest,
        _row_digest,
        _write_json,
    )


DATASET_ID = "hotpotqa/hotpot_qa"
CONFIGS = ("distractor", "fullwiki")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download HotpotQA from Hugging Face and build EXIT datasets"
    )
    parser.add_argument(
        "--config", choices=CONFIGS, required=True,
        help="HotpotQA configuration to use (kept as a separate experiment).",
    )
    parser.add_argument(
        "--save_dir", required=True,
        help="Output directory for train_dataset, validation_dataset, and manifest.",
    )
    parser.add_argument("--validation_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", default="./cache/huggingface")
    parser.add_argument(
        "--max_train_queries", type=int,
        help="Optional small subset for a quick data-pipeline run.",
    )
    parser.add_argument("--no_titles", action="store_true")
    return parser.parse_args()


def normalize_hf_example(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert HF's struct-of-lists fields into datasampling.py's list schema."""
    context = row["context"]
    supporting_facts = row["supporting_facts"]
    titles: Sequence[str] = context["title"]
    sentences: Sequence[Sequence[str]] = context["sentences"]
    fact_titles: Sequence[str] = supporting_facts["title"]
    fact_indices: Sequence[int] = supporting_facts["sent_id"]

    if len(titles) != len(sentences):
        raise ValueError(
            f"context title/sentence count mismatch for id={row.get('id')!r}"
        )
    if len(fact_titles) != len(fact_indices):
        raise ValueError(
            f"supporting fact title/index count mismatch for id={row.get('id')!r}"
        )

    return {
        "_id": str(row["id"]),
        "question": str(row["question"]),
        "context": [
            [str(title), [str(sentence) for sentence in doc_sentences]]
            for title, doc_sentences in zip(titles, sentences)
        ],
        "supporting_facts": [
            [str(title), int(index)]
            for title, index in zip(fact_titles, fact_indices)
        ],
    }


def main() -> None:
    args = parse_args()
    if args.validation_size <= 0:
        raise SystemExit("--validation_size must be positive")
    if args.max_train_queries is not None and args.max_train_queries <= args.validation_size:
        raise SystemExit("--max_train_queries must exceed --validation_size")

    print(f"Loading {DATASET_ID} config={args.config!r}, split='train'...")
    raw_dataset = load_dataset(
        DATASET_ID,
        args.config,
        split="train",
        cache_dir=args.cache_dir,
    )
    if args.max_train_queries is not None:
        count = min(args.max_train_queries, len(raw_dataset))
        raw_dataset = raw_dataset.select(range(count))
        print(f"Using first {count:,} rows for this limited run.")

    data = [normalize_hf_example(row) for row in raw_dataset]
    train_raw, validation_raw = split_queries(
        data,
        validation_size=args.validation_size,
        seed=args.seed,
    )
    include_titles = not args.no_titles

    print(f"Building samples from {len(train_raw):,} train queries...")
    train_processed = [process_example(row, include_titles) for row in train_raw]
    print(f"Building samples from {len(validation_raw):,} validation queries...")
    validation_processed = [
        process_example(row, include_titles) for row in validation_raw
    ]

    train_rows = create_balanced_dataset(
        train_processed, seed=args.seed, split_name="train"
    )
    validation_rows = create_balanced_dataset(
        validation_processed, seed=args.seed, split_name="validation"
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(train_rows).save_to_disk(str(save_dir / "train_dataset"))
    Dataset.from_list(validation_rows).save_to_disk(
        str(save_dir / "validation_dataset")
    )

    # The HF dataset does not have a local source JSON to hash. Record source
    # provenance separately and keep the existing manifest format for training.
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "paper": {
            "url": PAPER_URL,
            "sampling_ratio": {
                "positive": 2,
                "hard_negative": 1,
                "random_negative": 1,
            },
        },
        "source": {
            "dataset_id": DATASET_ID,
            "config": args.config,
            "split": "train",
            "query_count": len(data),
        },
        "construction": {
            "seed": args.seed,
            "validation_size_queries": args.validation_size,
            "split_unit": "query",
            "split_algorithm": "ascending sha256(seed + NUL + query_id)",
            "include_titles": include_titles,
            "random_negative_rule": "query_id != source_query_id",
            "sentence_boundaries": (
                "HotpotQA-provided sentence arrays, retained to align with "
                "supporting-fact indices"
            ),
        },
        "prompt": {
            "compression_template": COMPRESSION_PROMPT_TEMPLATE,
            "sha256": hashlib.sha256(
                COMPRESSION_PROMPT_TEMPLATE.encode("utf-8")
            ).hexdigest(),
        },
        "splits": {
            "train": {
                "query_count": len(train_raw),
                "query_ids_sha256": _query_digest(train_raw),
                "row_count": len(train_rows),
                "rows_sha256": _row_digest(train_rows),
                "sample_counts": {
                    kind: sum(row["sample_type"] == kind for row in train_rows)
                    for kind in ("positive", "hard_negative", "random_negative")
                },
                "random_negative_query_mismatch_verified": all(
                    row["query_id"] != row["source_query_id"]
                    for row in train_rows
                    if row["sample_type"] == "random_negative"
                ),
            },
            "validation": {
                "query_count": len(validation_raw),
                "query_ids_sha256": _query_digest(validation_raw),
                "row_count": len(validation_rows),
                "rows_sha256": _row_digest(validation_rows),
                "sample_counts": {
                    kind: sum(row["sample_type"] == kind for row in validation_rows)
                    for kind in ("positive", "hard_negative", "random_negative")
                },
                "random_negative_query_mismatch_verified": all(
                    row["query_id"] != row["source_query_id"]
                    for row in validation_rows
                    if row["sample_type"] == "random_negative"
                ),
            },
        },
        "software": {
            "python": platform.python_version(),
            "datasets": __import__("datasets").__version__,
        },
    }
    _write_json(save_dir / "manifest.json", manifest)

    print(json.dumps(manifest["splits"], indent=2, sort_keys=True))
    print(f"Saved normalized EXIT data to {save_dir}")


if __name__ == "__main__":
    main()
