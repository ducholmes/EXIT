#!/usr/bin/env python3
"""Export HotpotQA distractor validation contexts as reproduce.py JSONL."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - depends on optional install
    raise SystemExit(
        "This script requires Hugging Face Datasets. Install it with: "
        "pip install 'datasets>=2.14.7'"
    ) from exc


DATASET_ID = "hotpotqa/hotpot_qa"
CONFIG = "distractor"
SPLIT = "validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export HotpotQA distractor validation to reproduce.py JSONL"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("./cache/huggingface"))
    parser.add_argument("--limit", type=int, help="Optional leading subset for a pilot run")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def normalize_example(row: Mapping[str, Any]) -> Dict[str, Any]:
    context = row.get("context")
    if not isinstance(context, Mapping):
        raise ValueError(f"HotpotQA row {row.get('id')!r} has no context object")
    titles: Sequence[str] = context.get("title", [])
    sentence_lists: Sequence[Sequence[str]] = context.get("sentences", [])
    if len(titles) != len(sentence_lists):
        raise ValueError(f"context title/sentence count mismatch for id={row.get('id')!r}")

    contexts = []
    for title, sentences in zip(titles, sentence_lists):
        if not isinstance(sentences, (list, tuple)):
            raise ValueError(f"context sentences must be a sequence for id={row.get('id')!r}")
        contexts.append(
            {
                "title": str(title or ""),
                "text": " ".join(str(sentence) for sentence in sentences).strip(),
                "score": 1.0,
            }
        )

    question = row.get("question")
    answer = row.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"HotpotQA row {row.get('id')!r} has no question")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"HotpotQA row {row.get('id')!r} has no answer")
    if len(contexts) != 10:
        raise ValueError(
            f"distractor validation example {row.get('id')!r} has {len(contexts)} "
            "contexts; expected 10"
        )
    if any(not context["text"] for context in contexts):
        raise ValueError(f"HotpotQA row {row.get('id')!r} has an empty context")

    return {
        "id": str(row.get("id", "")),
        "question": question.strip(),
        "answers": [answer.strip()],
        "ctxs": contexts,
        "type": str(row.get("type", "")),
        "level": str(row.get("level", "")),
        "supporting_facts": row.get("supporting_facts"),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    raw_dataset = load_dataset(
        DATASET_ID,
        CONFIG,
        split=SPLIT,
        cache_dir=str(args.cache_dir),
    )
    if args.limit is not None:
        raw_dataset = raw_dataset.select(range(min(args.limit, len(raw_dataset))))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    count = 0
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for row in raw_dataset:
                output_file.write(
                    json.dumps(normalize_example(row), ensure_ascii=False) + "\n"
                )
                count += 1
        temporary_path.replace(args.output)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    manifest = {
        "source_dataset": DATASET_ID,
        "config": CONFIG,
        "split": SPLIT,
        "examples": count,
        "contexts_per_example": 10,
        "input_sha256": sha256_file(args.output),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Uses the HotpotQA distractor-provided contexts, not a separately "
            "retrieved Top-k corpus."
        ),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
