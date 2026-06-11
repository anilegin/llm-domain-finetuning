#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel

from compare_models import (
    METRIC_NAMES,
    VARIANTS,
    build_comparison_rows,
    build_prompt_records,
    choose_device,
    generate_answers,
    judge_variant,
    load_jsonl,
    print_comparison,
    safe_name,
    save_jsonl,
)
from config import DATA_DIR, HF_TOKEN, JUDGE_MODEL_ID, OUTPUTS_DIR
from data_utils import DATASET_NAME
from evaluate import evaluate_responses
from model_utils import load_judge_model
from config import SMOLLM3_MODEL_ID, SMOLLM3_OUTPUT_DIR
from smollm3_utils import load_smollm3_model, load_smollm3_tokenizer
from utils import free_gpu_memory, load_rankings


SMOLLM3_LABEL = "SmolLM3-3B-LoRA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the SmolLM3 RAG LoRA adapter on test data"
    )
    parser.add_argument("--adapter_path", default=str(SMOLLM3_OUTPUT_DIR))
    parser.add_argument("--dataset_name", default=DATASET_NAME)
    parser.add_argument("--rankings_path", default=str(DATA_DIR / "test_rankings.jsonl"))
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--judge_batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--source_run_dir",
        default=None,
        help=(
            "Reuse the exact test sample from a previous run. The script first "
            "looks for comparison.json sample_indices, then falls back to query_ids "
            "from answers/*.jsonl."
        ),
    )
    parser.add_argument(
        "--compare_report",
        default=None,
        help="Optional previous comparison.json to include old model metrics in the printout.",
    )
    parser.add_argument("--judge_model_id", default=JUDGE_MODEL_ID)
    parser.add_argument("--skip_judge", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse the SmolLM3 answer file in --output_dir if present.",
    )
    return parser.parse_args()


def get_rows_from_source_run(dataset, source_run_dir: Path) -> tuple[list, list[int] | None]:
    comparison_path = source_run_dir / "comparison.json"
    if comparison_path.is_file():
        with comparison_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        indices = data.get("config", {}).get("sample_indices") or data.get("sample_indices")
        if indices:
            return [dataset[int(index)] for index in indices], [int(index) for index in indices]

    answer_files = sorted((source_run_dir / "answers").glob("*.jsonl"))
    if not answer_files:
        raise FileNotFoundError(
            f"No comparison.json sample_indices or answers/*.jsonl found in {source_run_dir}"
        )

    source_answers = load_jsonl(answer_files[0])
    query_ids = [str(record["query_id"]) for record in source_answers]
    by_id = {str(row["query_id"]): row for row in dataset}
    missing = [query_id for query_id in query_ids if query_id not in by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} query_ids from {answer_files[0]} were not found in test split"
        )
    return [by_id[query_id] for query_id in query_ids], None


def select_rows(dataset, args: argparse.Namespace) -> tuple[list, list[int] | None]:
    if args.source_run_dir:
        return get_rows_from_source_run(dataset, Path(args.source_run_dir))
    if args.num_samples is None:
        return list(dataset), list(range(len(dataset)))
    if args.num_samples <= 0 or args.num_samples > len(dataset):
        raise ValueError(
            f"--num_samples must be between 1 and {len(dataset)}, got {args.num_samples}"
        )
    indices = random.Random(args.seed).sample(range(len(dataset)), k=args.num_samples)
    return [dataset[index] for index in indices], indices


def load_smollm3_lora(adapter_path: Path, device: str):
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"SmolLM3 adapter directory not found: {adapter_path}")
    tokenizer = load_smollm3_tokenizer()
    model = load_smollm3_model(device=device, use_qlora=False)
    model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=True)
    return model.merge_and_unload(), tokenizer


def save_comparison_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "setting", "metric", "score"],
        )
        writer.writeheader()
        writer.writerows(rows)


def load_previous_metrics(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)

    metrics = {}
    if "metrics" in report and isinstance(report["metrics"], dict):
        if "comparison" in report and isinstance(report["comparison"], list):
            for row in report["comparison"]:
                metrics.setdefault(row["model"], {})[
                    f"{row['metric']}_{row['setting']}"
                ] = row["score"]
        if report.get("fine_tuned"):
            metrics["Qwen2.5-3B-LoRA"] = report["fine_tuned"]
        for model_name, model_metrics in report.get("additional_model_results", {}).items():
            metrics[Path(model_name).name] = model_metrics
    return metrics


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.judge_batch_size <= 0:
        raise ValueError("Batch sizes must be greater than zero")

    device = choose_device()
    run_dir = (
        Path(args.output_dir)
        if args.output_dir
        else OUTPUTS_DIR / f"smollm3_lora_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    if args.resume and args.output_dir is None:
        raise ValueError("--resume requires --output_dir")
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    answers_dir = run_dir / "answers"
    judge_dir = run_dir / "judge"

    print(f"Device: {device}")
    print(f"Model: {SMOLLM3_MODEL_ID}")
    print(f"Adapter: {args.adapter_path}")
    dataset = load_dataset(
        args.dataset_name,
        split="test",
        token=args.hf_token or HF_TOKEN,
    )
    rows, sample_indices = select_rows(dataset, args)
    print(f"Evaluating {len(rows)} test examples")

    rankings = load_rankings(args.rankings_path)
    prompt_records = build_prompt_records(rows, rankings, args.k)

    answer_path = answers_dir / f"{safe_name(SMOLLM3_LABEL)}.jsonl"
    if args.resume and answer_path.is_file():
        print(f"Reusing saved answers: {answer_path}")
        answers = load_jsonl(answer_path)
        expected_ids = [record["query_id"] for record in prompt_records]
        actual_ids = [record.get("query_id") for record in answers]
        if actual_ids != expected_ids:
            raise ValueError(
                f"Saved answers in {answer_path} do not match the selected test sample"
            )
    else:
        model, tokenizer = load_smollm3_lora(Path(args.adapter_path), device)
        answers = generate_answers(
            model=model,
            tokenizer=tokenizer,
            prompt_records=prompt_records,
            device=device,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            disable_thinking=True,
        )
        save_jsonl(answers, answer_path)
        del model
        del tokenizer
        free_gpu_memory()

    print("Scoring SmolLM3 LoRA")
    metrics = evaluate_responses(answers)

    if not args.skip_judge:
        judge_model, judge_tokenizer = load_judge_model(
            device,
            model_name=args.judge_model_id,
        )
        for variant in VARIANTS:
            metrics[f"LLM_Judge_{variant}"] = judge_variant(
                results=answers,
                variant=variant,
                judge_model=judge_model,
                judge_tokenizer=judge_tokenizer,
                batch_size=args.judge_batch_size,
                save_path=judge_dir / f"{safe_name(SMOLLM3_LABEL)}__{variant}.jsonl",
            )
        del judge_model
        del judge_tokenizer
        free_gpu_memory()

    all_metrics = {SMOLLM3_LABEL: metrics}
    compare_report = Path(args.compare_report) if args.compare_report else None
    if compare_report and compare_report.is_file():
        all_metrics = {**load_previous_metrics(compare_report), **all_metrics}

    comparison_rows = build_comparison_rows(all_metrics)
    save_comparison_csv(comparison_rows, run_dir / "comparison.csv")
    with (run_dir / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": {
                    "dataset_name": args.dataset_name,
                    "split": "test",
                    "num_samples": len(rows),
                    "sample_indices": sample_indices,
                    "source_run_dir": args.source_run_dir,
                    "k": args.k,
                    "rankings_path": args.rankings_path,
                    "adapter_path": args.adapter_path,
                    "judge_model_id": None if args.skip_judge else args.judge_model_id,
                },
                "answer_paths": {SMOLLM3_LABEL: str(answer_path)},
                "answers": answers,
                "metrics": all_metrics,
                "comparison": comparison_rows,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print_comparison(comparison_rows, list(all_metrics.keys()))
    print(f"Answers: {answer_path}")
    print(f"Report: {run_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
