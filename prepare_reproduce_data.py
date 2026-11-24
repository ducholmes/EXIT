#!/usr/bin/env python3
"""Export HotpotQA distractor validation contexts as reproduce.py JSONL."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
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
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export HotpotQA distractor validation to reproduce.py JSONL"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("./cache/huggingface"))
    parser.add_argument("--limit", type=int, help="Optional leading subset for a pilot run")
    parser.add_argument(
        "--questions-only",
        action="store_true",
        help="Export id/question/answers only; retrieval will supply ctxs later",
    )
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


def normalize_question(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Export evaluation inputs without relying on dataset-provided contexts."""
    question = row.get("question")
    answer = row.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"HotpotQA row {row.get('id')!r} has no question")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"HotpotQA row {row.get('id')!r} has no answer")
    return {
        "id": str(row.get("id", "")),
        "question": question.strip(),
        "answers": [answer.strip()],
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
    source_split_examples = len(raw_dataset)
    if args.limit is not None:
        raw_dataset = raw_dataset.select(range(min(args.limit, len(raw_dataset))))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    count = 0
    skipped_examples = []
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for row in raw_dataset:
                if args.questions_only:
                    output_file.write(
                        json.dumps(normalize_question(row), ensure_ascii=False) + "\n"
                    )
                    count += 1
                    continue
                context = row.get("context")
                if not isinstance(context, Mapping):
                    raise ValueError(f"HotpotQA row {row.get('id')!r} has no context object")
                titles = context.get("title", [])
                sentence_lists = context.get("sentences", [])
                if len(titles) != len(sentence_lists):
                    raise ValueError(
                        f"context title/sentence count mismatch for id={row.get('id')!r}"
                    )
                if len(titles) != 10:
                    skipped_examples.append(
                        {
                            "id": str(row.get("id", "")),
                            "context_count": len(titles),
                            "reason": "expected exactly 10 contexts",
                        }
                    )
                    logger.warning(
                        "Skipping HotpotQA validation example %r: has %d contexts; expected 10",
                        row.get("id"),
                        len(titles),
                    )
                    continue
                output_file.write(
                    json.dumps(normalize_example(row), ensure_ascii=False) + "\n"
                )
                count += 1
        if count == 0:
            raise ValueError(
                "No HotpotQA validation examples were exported"
            )
        temporary_path.replace(args.output)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    manifest = {
        "source_dataset": DATASET_ID,
        "config": CONFIG,
        "split": SPLIT,
        "source_split_examples": source_split_examples,
        "input_examples": len(raw_dataset),
        "export_mode": "questions_only" if args.questions_only else "with_contexts",
        "exported_examples": count,
        "skipped_example_count": len(skipped_examples),
        "skipped_examples": skipped_examples,
        "contexts_per_example": None if args.questions_only else 10,
        "input_sha256": sha256_file(args.output),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Contexts are omitted; attach retrieved passages in the retrieval stage."
            if args.questions_only
            else "Uses the HotpotQA distractor-provided contexts, not a separately "
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
