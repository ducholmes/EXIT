#!/usr/bin/env python3
"""Reproduce EXIT QA metrics from pre-retrieved paper-format datasets.

The runner intentionally starts after retrieval.  It expects the CompAct-style
``question``/``answers``/``ctxs`` artifacts described in the README, then
measures reader generation with optional EXIT compression.  Rebuilding the
December 2018 Wikipedia Contriever index remains a separate retrieval step.
"""

import argparse
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import re
import statistics
import string
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from exit_rag import Document, ExitRAG
from gemini_reader import DailyQuotaReached, GeminiAPIError, GeminiQuota


PAPER_SPLITS = {
    "nq": "dev",
    "triviaqa": "test",
    "hotpotqa": "dev",
    "2wiki": "dev",
}


def normalize_answer(text: str) -> str:
    """Use the standard SQuAD-style normalization used for QA EM/F1."""

    text = text.lower()
    text = "".join(
        character for character in text if character not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answer: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(answer))


def token_f1(prediction: str, answer: str) -> float:
    from collections import Counter

    prediction_tokens = normalize_answer(prediction).split()
    answer_tokens = normalize_answer(answer).split()
    if not prediction_tokens or not answer_tokens:
        return float(prediction_tokens == answer_tokens)

    overlap = Counter(prediction_tokens) & Counter(answer_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(
    prediction: str, answers: Sequence[str]
) -> Tuple[Optional[float], Optional[float]]:
    if not answers:
        return None, None
    return (
        max(exact_match(prediction, answer) for answer in answers),
        max(token_f1(prediction, answer) for answer in answers),
    )


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_examples(path: Path) -> List[Dict]:
    """Load JSON, JSONL, or a top-level ``{"data": [...]}`` artifact."""

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as input_file:
            return [json.loads(line) for line in input_file if line.strip()]

    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    raise ValueError("input must be a JSON list, JSONL file, or {'data': [...]} object")


def get_query(example: Dict) -> str:
    query = example.get("question", example.get("query"))
    if not isinstance(query, str) or not query.strip():
        raise ValueError("every example needs a non-empty 'question' or 'query'")
    return query.strip()


def get_answers(example: Dict) -> Tuple[str, ...]:
    answers = example.get("answers", example.get("answer", ()))
    if isinstance(answers, str):
        return (answers,)
    if isinstance(answers, list) and all(isinstance(answer, str) for answer in answers):
        return tuple(answers)
    if answers in (None, ()):
        return ()
    raise ValueError("'answers' must be a string or list of strings")


def get_documents(example: Dict, top_k: int) -> List[Document]:
    contexts = example.get("ctxs")
    if not isinstance(contexts, list):
        raise ValueError("every example needs a 'ctxs' list of retrieved documents")
    if len(contexts) < top_k:
        raise ValueError(
            f"example has {len(contexts)} retrieved documents, but Top-{top_k} "
            "evaluation requires at least that many"
        )

    documents = []
    for context in contexts[:top_k]:
        if not isinstance(context, dict) or not isinstance(context.get("text"), str):
            raise ValueError("each retrieved context needs a string 'text' field")
        documents.append(
            Document(
                title=str(context.get("title") or ""),
                text=context["text"],
                score=float(context.get("score", 1.0)),
            )
        )
    return documents


def git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def resolved_revision(model) -> Optional[str]:
    return getattr(getattr(model, "config", None), "_commit_hash", None)


def resolved_adapter_revision(model) -> Optional[str]:
    """Return an adapter commit only when PEFT exposes one explicitly.

    ``PeftModel.config`` is the base-model config, so reading its commit hash
    would incorrectly report the base revision as the adapter revision.
    """

    peft_configs = getattr(model, "peft_config", {})
    for config in peft_configs.values():
        commit_hash = getattr(config, "_commit_hash", None)
        if commit_hash:
            return str(commit_hash)
    return None


def package_versions(packages: Sequence[str]) -> Dict[str, Optional[str]]:
    resolved = {}
    for package in packages:
        try:
            resolved[package] = version(package)
        except PackageNotFoundError:
            resolved[package] = None
    return resolved


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def latency_summary(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate EXIT on pre-retrieved NQ/TQA/HQA/2Wiki artifacts"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(PAPER_SPLITS), required=True)
    parser.add_argument("--split", help="Defaults to the paper split for the dataset")
    parser.add_argument("--top-k", type=int, choices=(5, 10, 20), default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--latency-repeats", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--compression-base-model", default="google/gemma-2b-it")
    parser.add_argument("--compression-model", default="doubleyyh/exit-gemma-2b")
    parser.add_argument("--reader-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--reader-backend", choices=("transformers", "gemini"), default="transformers"
    )
    parser.add_argument(
        "--gemini-config",
        type=Path,
        help="JSON with model-specific Gemini quotas; required for --reader-backend gemini",
    )
    parser.add_argument(
        "--gemini-quota-state",
        type=Path,
        help="Persistent state file shared by compressor-on/off runs",
    )
    parser.add_argument("--compression-base-revision")
    parser.add_argument("--compression-revision")
    parser.add_argument("--reader-revision")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--device")
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument(
        "--no-compressor",
        action="store_true",
        help="Pass retrieved documents directly to the reader without EXIT compression",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.latency_repeats < 1:
        parser.error("--latency-repeats must be at least 1")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs cannot be negative")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.reader_backend == "gemini" and args.latency_repeats != 1:
        parser.error("Gemini API runs require --latency-repeats 1 to avoid duplicate billable calls")
    if args.reader_backend == "gemini" and args.warmup_runs:
        parser.error("Gemini API runs do not support warm-up calls; they consume API quota")
    return args


def main() -> None:
    args = parse_args()
    examples = load_examples(args.input)
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError("input contains no examples")

    # Validate the full selected slice before spending time loading GPU models.
    for example in examples:
        get_query(example)
        get_answers(example)
        get_documents(example, args.top_k)
    if args.validate_only:
        print(
            f"Validated {len(examples)} {args.dataset}/{args.split or PAPER_SPLITS[args.dataset]} examples"
        )
        return

    gemini_model = None
    gemini_project_id = None
    gemini_quota = None
    quota_state_path = None
    gemini_timeout_seconds = 300
    gemini_max_attempts = 3
    if args.reader_backend == "gemini":
        if args.gemini_config is None:
            raise ValueError("--gemini-config is required with --reader-backend gemini")
        if not os.environ.get("GEMINI_API_KEY"):
            raise ValueError("Set GEMINI_API_KEY in the environment before using Gemini")
        with args.gemini_config.open("r", encoding="utf-8") as config_file:
            gemini_config = json.load(config_file)
        gemini_model = gemini_config.get("model")
        if not isinstance(gemini_model, str) or not gemini_model.strip():
            raise ValueError("Gemini config needs a non-empty 'model' field")
        gemini_project_id = gemini_config.get("project_id")
        if not isinstance(gemini_project_id, str) or not gemini_project_id.strip():
            raise ValueError("Gemini config needs a non-empty 'project_id' field")
        quota_values = gemini_config.get("quotas")
        if not isinstance(quota_values, dict):
            raise ValueError("Gemini config needs a 'quotas' object")
        gemini_quota = GeminiQuota.from_mapping(quota_values)
        gemini_timeout_seconds = gemini_config.get("timeout_seconds", 300)
        gemini_max_attempts = gemini_config.get("max_attempts", 3)
        for key, value in (
            ("timeout_seconds", gemini_timeout_seconds),
            ("max_attempts", gemini_max_attempts),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"Gemini config '{key}' must be a positive integer"
                )
        configured_state = gemini_config.get("quota_state")
        quota_state_path = args.gemini_quota_state or (
            Path(configured_state) if configured_state else None
        )
        if quota_state_path is None:
            safe_model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", gemini_model)
            quota_state_path = Path(args.cache_dir) / f"gemini-quota-{safe_model_name}.json"

    rag = ExitRAG(
        compression_base_model=args.compression_base_model,
        compression_model=args.compression_model,
        reader_model=args.reader_model,
        device=args.device,
        batch_size=args.batch_size,
        threshold=args.threshold,
        cache_dir=args.cache_dir,
        compression_base_revision=args.compression_base_revision,
        compression_revision=args.compression_revision,
        reader_revision=args.reader_revision,
        load_compressor_in_4bit=not args.no_4bit,
        max_new_tokens=args.max_new_tokens,
        use_compressor=not args.no_compressor,
        reader_backend=args.reader_backend,
        gemini_model=gemini_model,
        gemini_project_id=gemini_project_id,
        gemini_quota=gemini_quota,
        gemini_quota_state=quota_state_path,
        gemini_timeout_seconds=gemini_timeout_seconds,
        gemini_max_attempts=gemini_max_attempts,
    )

    if args.warmup_runs:
        warmup_example = examples[0]
        warmup_query = get_query(warmup_example)
        warmup_documents = get_documents(warmup_example, args.top_k)
        for _ in range(args.warmup_runs):
            rag.run_rag(
                warmup_query,
                warmup_documents,
                compression_threshold=args.threshold,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    input_digest = file_sha256(args.input)
    progress_path = args.output.with_suffix(args.output.suffix + ".progress.json")
    run_fingerprint = {
        "input_sha256": input_digest,
        "dataset": args.dataset,
        "split": args.split or PAPER_SPLITS[args.dataset],
        "top_k": args.top_k,
        "compressor_enabled": not args.no_compressor,
        "threshold": args.threshold,
        "latency_repeats": args.latency_repeats,
        "reader_backend": args.reader_backend,
        "reader_model": gemini_model if args.reader_backend == "gemini" else args.reader_model,
        "gemini_project_id": gemini_project_id,
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit,
        "compression_base_model": args.compression_base_model,
        "compression_model": args.compression_model,
        "batch_size": args.batch_size,
    }
    if progress_path.exists():
        saved_fingerprint = json.loads(progress_path.read_text(encoding="utf-8"))
        if saved_fingerprint != run_fingerprint:
            raise ValueError(
                f"Existing progress file {progress_path} belongs to a different run; "
                "choose a new --output path or remove the stale progress/output files."
            )
    elif args.output.exists() and args.output.stat().st_size:
        raise ValueError(
            f"Output {args.output} exists without its progress manifest; choose a new path."
        )
    else:
        progress_path.write_text(
            json.dumps(run_fingerprint, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    records_by_index = {}
    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as previous_output:
            for line_number, line in enumerate(previous_output, start=1):
                if line.strip():
                    record = json.loads(line)
                    records_by_index[int(record["index"])] = record

    for index, example in enumerate(examples):
        if index in records_by_index:
            continue
        query = get_query(example)
        answers = get_answers(example)
        documents = get_documents(example, args.top_k)
        try:
            runs = [
                rag.run_rag(query, documents, compression_threshold=args.threshold)
                for _ in range(args.latency_repeats)
            ]
        except DailyQuotaReached as exc:
            print(f"Gemini run paused safely: {exc}", file=sys.stderr)
            break
        except GeminiAPIError as exc:
            print(f"Gemini run paused safely: {exc}", file=sys.stderr)
            if exc.status == 429:
                print(
                    "The API returned 429 despite local limits. Check AI Studio "
                    "usage and resume only after the affected quota resets.",
                    file=sys.stderr,
                )
            break
        result = runs[0]
        compression_times = [run["compression_time"] for run in runs]
        reading_times = [run["reading_time"] for run in runs]
        total_times = [run["total_time"] for run in runs]
        em, f1 = score_answer(result["answer"], answers)

        record = {
            "index": index,
            "id": example.get("id", example.get("_id", index)),
            "query": query,
            "answers": answers,
            "prediction": result["answer"],
            "exact_match": em,
            "f1": f1,
            "compressed_context": result["compressed_context"],
            "sentence_selections": result["sentence_selections"],
            "relevance_scores": result["relevance_scores"],
            "sentences": result["sentences"],
            "sentence_document_indices": result["sentence_document_indices"],
            "sentence_indices": result["sentence_indices"],
            "original_tokens": result["original_tokens"],
            "compressed_tokens": result["compressed_tokens"],
            "compression_latency": latency_summary(compression_times),
            "reading_latency": latency_summary(reading_times),
            "total_latency": latency_summary(total_times),
            "reader_usage": result.get("reader_usage", {}),
        }
        with args.output.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        records_by_index[index] = record

    if len(records_by_index) != len(examples):
        print(
            f"Run paused after {len(records_by_index)}/{len(examples)} examples. "
            f"Resume with the same arguments and output path: {args.output}",
            file=sys.stderr,
        )
        return

    records = [records_by_index[index] for index in range(len(examples))]
    exact_matches = [record["exact_match"] for record in records if record["exact_match"] is not None]
    f1_scores = [record["f1"] for record in records if record["f1"] is not None]
    reader_usage_totals = {
        "input_tokens": sum(
            int(record.get("reader_usage", {}).get("input_tokens", 0))
            for record in records
        ),
        "output_tokens": sum(
            int(record.get("reader_usage", {}).get("output_tokens", 0))
            for record in records
        ),
        "thought_tokens": sum(
            int(record.get("reader_usage", {}).get("thought_tokens", 0))
            for record in records
        ),
    }

    def pooled_latency(field: str) -> Dict[str, float]:
        per_example = [record[field] for record in records]
        pooled_mean = mean([item["mean"] for item in per_example])
        pooled_variance = mean(
            [
                item["std"] ** 2 + (item["mean"] - pooled_mean) ** 2
                for item in per_example
            ]
        )
        return {"mean": pooled_mean, "std": pooled_variance**0.5}

    original_tokens = [record["original_tokens"] for record in records]
    compressed_tokens = [record["compressed_tokens"] for record in records]
    summary = {
        "paper": "arXiv:2412.12559v3",
        "git_sha": git_sha(),
        "dataset": args.dataset,
        "split": args.split or PAPER_SPLITS[args.dataset],
        "input": str(args.input.resolve()),
        "input_sha256": input_digest,
        "examples": len(records),
        "top_k": args.top_k,
        "metrics_percent": {
            "exact_match": 100.0 * mean(exact_matches),
            "f1": 100.0 * mean(f1_scores),
            "scored_examples": len(exact_matches),
        },
        "tokens": {
            "original_mean": mean(original_tokens),
            "compressed_mean": mean(compressed_tokens),
            "scope": (
                "reader prompt input tokens, including the question and fixed instructions"
                if args.reader_backend == "gemini"
                else "context tokens"
            ),
            "retained_fraction": (
                sum(compressed_tokens) / sum(original_tokens)
                if sum(original_tokens)
                else 0.0
            ),
        },
        "latency_seconds": {
            "compression": pooled_latency("compression_latency"),
            "reading": pooled_latency("reading_latency"),
            "total": pooled_latency("total_latency"),
            "repeats": args.latency_repeats,
            "warmup_runs": args.warmup_runs,
            "backend": args.reader_backend,
            "paper_backend": "vLLM 0.5.5",
            "reader_api_usage": reader_usage_totals,
        },
        "models": {
            "compression_base": (
                args.compression_base_model if rag.use_compressor else None
            ),
            "compression_adapter": args.compression_model if rag.use_compressor else None,
            "reader": rag.reader_model_id,
            "requested_revisions": {
                "compression_base": (
                    args.compression_base_revision if rag.use_compressor else None
                ),
                "compression_adapter": (
                    args.compression_revision if rag.use_compressor else None
                ),
                "reader": args.reader_revision,
            },
            "resolved_revisions": {
                "compression_base": (
                    resolved_revision(rag.compressor.base_model)
                    if rag.compressor is not None
                    else None
                ),
                "compression_adapter": (
                    resolved_adapter_revision(rag.compressor.model)
                    if rag.compressor is not None
                    else None
                ),
                "reader": (
                    resolved_revision(rag.reader)
                    if rag.reader is not None
                    else None
                ),
            },
        },
        "inference": {
            "compressor_enabled": rag.use_compressor,
            "threshold": args.threshold,
            "batch_size": args.batch_size,
            "temperature": 0.0 if args.reader_backend == "transformers" else None,
            "top_p": 1.0 if args.reader_backend == "transformers" else None,
            "thinking_level": "MINIMAL" if args.reader_backend == "gemini" else None,
            "max_new_tokens": args.max_new_tokens,
            "gemini_request": (
                {
                    "timeout_seconds": gemini_timeout_seconds,
                    "max_attempts": gemini_max_attempts,
                    "retry_backoff_initial_seconds": 2,
                    "retry_backoff_multiplier": 2,
                    "retry_jitter_max_seconds": 0.5,
                    "retryable_http_statuses": [408, 504],
                    "retryable_transport_errors": ["timeouts"],
                }
                if args.reader_backend == "gemini"
                else None
            ),
            "compressor_4bit": not args.no_4bit if rag.use_compressor else None,
            "gemini_quota": (
                {
                    "configured": {
                        "rpm": gemini_quota.rpm,
                        "input_tpm": gemini_quota.input_tpm,
                        "rpd": gemini_quota.rpd,
                    },
                    "safety_fraction": gemini_quota.safety_fraction,
                    "safe_limits": {
                        "rpm": gemini_quota.safe_rpm,
                        "input_tpm": gemini_quota.safe_input_tpm,
                        "rpd": gemini_quota.safe_rpd,
                    },
                }
                if gemini_quota is not None
                else None
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpus": [
                torch.cuda.get_device_name(device_index)
                for device_index in range(torch.cuda.device_count())
            ],
            "packages": package_versions(
                ["transformers", "peft", "accelerate", "bitsandbytes", "spacy", "vllm"]
            ),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "notes": [
            (
                "Retrieval latency is excluded; compression is skipped."
                if args.no_compressor
                else "Retrieval latency is excluded, matching the paper's compression+reader tables."
            ),
            (
                "Gemini API latency includes network/service time and is not directly comparable with local or paper vLLM latency."
                if args.reader_backend == "gemini"
                else "Transformers latency must not be compared directly with the paper's vLLM 0.5.5 latency."
            ),
            "The paper does not publish exact dataset/index/model revisions; pin and record them for strict reruns.",
        ],
    }
    summary_path = args.output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
        summary_file.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
