#!/usr/bin/env python3
"""Compare two small instruct models and a LoRA-tuned Qwen2.5-3B on test data.

Each model is evaluated in baseline, RAG, and oracle settings. The oracle
prompt always places the gold chunk first and contains exactly k chunks.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm

from config import (
    DATA_DIR,
    HF_TOKEN,
    JUDGE_MODEL_ID,
    LITTLE_MODEL_1_ID,
    LITTLE_MODEL_2_ID,
    MODEL_ID,
    MODELS_DIR,
    OUTPUTS_DIR,
    SYSTEM_PROMPT,
)
from data_utils import DATASET_NAME, build_chat_messages, format_chunks
from evaluate import (
    _build_judge_prompt,
    _generate_from_prompts_batch,
    _parse_judge_response,
    evaluate_responses,
)
from model_utils import load_base_model, load_judge_model, load_tokenizer
from utils import free_gpu_memory, load_rankings


VARIANTS = ("baseline", "RAG", "oracle")
METRIC_NAMES = ("EM", "SubEM", "METEOR", "BERT_P", "BERT_R", "BERT_F1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and score baseline/RAG/oracle answers from Qwen3-1.7B, "
            "SmolLM2-1.7B-Instruct, and a LoRA-tuned Qwen2.5-3B-Instruct."
        )
    )
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--judge_batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--dataset_name", default=DATASET_NAME)
    parser.add_argument("--rankings_path", default=str(DATA_DIR / "test_rankings.jsonl"))
    parser.add_argument(
        "--adapter_path",
        default=str(MODELS_DIR / "qwen-rag-lora-k3-seq4096-lr1e4" / "checkpoint-1500"),
    )
    parser.add_argument("--qwen3_model_id", default=LITTLE_MODEL_1_ID)
    parser.add_argument("--smollm_model_id", default=LITTLE_MODEL_2_ID)
    parser.add_argument("--judge_model_id", default=JUDGE_MODEL_ID)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse valid answer JSONL files already present under --output_dir/answers.",
    )
    parser.add_argument(
        "--skip_judge",
        action="store_true",
        help="Skip Mistral judge scoring; lexical and BERTScore metrics still run.",
    )
    return parser.parse_args()


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", value).strip("_")


def select_test_rows(dataset, num_samples: int, seed: int) -> tuple[list, list[int]]:
    if num_samples <= 0:
        raise ValueError("--num_samples must be greater than zero")
    if num_samples > len(dataset):
        raise ValueError(
            f"Requested {num_samples} samples, but the test split has only {len(dataset)}"
        )

    indices = random.Random(seed).sample(range(len(dataset)), k=num_samples)
    return [dataset[index] for index in indices], indices


def get_ranking(row: dict, rankings: dict) -> list[int]:
    query_id = str(row["query_id"])
    if query_id not in rankings:
        raise KeyError(f"No test ranking found for query_id={query_id}")

    chunks = row["candidate_chunks"]
    ranking = list(rankings[query_id])
    if len(set(ranking)) != len(ranking):
        raise ValueError(f"Ranking contains duplicate chunk indices for query_id={query_id}")
    if any(index < 0 or index >= len(chunks) for index in ranking):
        raise ValueError(f"Ranking contains an invalid chunk index for query_id={query_id}")
    return ranking


def build_oracle_top_k(ranking: list[int], correct_idx: int, k: int) -> list[int]:
    """Put the correct chunk first while preserving retrieved order."""
    retrieved = ranking[:k]
    if correct_idx in retrieved:
        return [correct_idx] + [index for index in retrieved if index != correct_idx]
    return [correct_idx] + retrieved[: max(0, k - 1)]


def build_prompt_records(rows: list, rankings: dict, k: int) -> list[dict]:
    if k <= 0:
        raise ValueError("--k must be greater than zero")

    records = []
    for row in rows:
        query_id = str(row["query_id"])
        query = row["query"]
        chunks = row["candidate_chunks"]
        ground_truth = row["short_answer"][0]
        correct_idx = int(row.get("answer_pos", -1))
        if correct_idx < 0 or correct_idx >= len(chunks):
            raise ValueError(
                f"Invalid answer_pos={correct_idx} for query_id={query_id}; "
                "oracle evaluation requires a valid gold chunk"
            )

        ranking = get_ranking(row, rankings)
        if len(ranking) < k:
            raise ValueError(
                f"Ranking for query_id={query_id} has {len(ranking)} chunks, fewer than k={k}"
            )

        rag_top_k = ranking[:k]
        oracle_top_k = build_oracle_top_k(ranking, correct_idx, k)
        if len(oracle_top_k) != k or oracle_top_k[0] != correct_idx:
            raise AssertionError(f"Oracle construction failed for query_id={query_id}")

        rag_context = format_chunks(chunks, rag_top_k, k=k)
        oracle_context = format_chunks(chunks, oracle_top_k, k=k)
        records.append(
            {
                "query_id": query_id,
                "query": query,
                "ground_truth": ground_truth,
                "ranking": ranking,
                "correct_chunk_index": correct_idx,
                "correct_in_topk": (
                    rag_top_k.index(correct_idx) if correct_idx in rag_top_k else -1
                ),
                "rag_top_k": rag_top_k,
                "oracle_top_k": oracle_top_k,
                "messages": {
                    "baseline": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": query},
                    ],
                    "RAG": build_chat_messages(query, rag_context, answer=None),
                    "oracle": build_chat_messages(query, oracle_context, answer=None),
                },
            }
        )
    return records


def render_prompt(tokenizer, messages: list[dict], disable_thinking: bool) -> str:
    kwargs = {}
    if disable_thinking:
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


@torch.no_grad()
def generate_answers(
    model,
    tokenizer,
    prompt_records: list[dict],
    device: str,
    batch_size: int,
    max_new_tokens: int,
    disable_thinking: bool = False,
) -> list[dict]:
    model.eval()
    answers = [
        {
            key: value
            for key, value in record.items()
            if key != "messages"
        }
        for record in prompt_records
    ]

    for variant in VARIANTS:
        prompts = [
            render_prompt(tokenizer, record["messages"][variant], disable_thinking)
            for record in prompt_records
        ]
        generated = []
        for start in tqdm(
            range(0, len(prompts), batch_size),
            desc=f"Generating {variant}",
        ):
            generated.extend(
                _generate_from_prompts_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts[start : start + batch_size],
                    device=device,
                    max_new_tokens=max_new_tokens,
                )
            )
        for record, prediction in zip(answers, generated):
            record[variant] = prediction.strip()
    return answers


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def validate_saved_answers(
    answers: list[dict],
    prompt_records: list[dict],
    path: Path,
) -> None:
    expected_ids = [record["query_id"] for record in prompt_records]
    actual_ids = [record.get("query_id") for record in answers]
    if actual_ids != expected_ids:
        raise ValueError(
            f"Saved answers do not match this sampled test set: {path}. "
            "Use the same --seed/--num_samples or start a new output directory."
        )
    for variant in VARIANTS:
        if any(variant not in record for record in answers):
            raise ValueError(f"Saved answers are missing {variant!r} responses: {path}")


def load_standard_model(model_id: str, device: str):
    tokenizer = load_tokenizer(model_name=model_id)
    model = load_base_model(device, use_qlora=False, model_name=model_id)
    return model, tokenizer


def load_fine_tuned_model(adapter_path: Path, device: str):
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"LoRA checkpoint directory not found: {adapter_path}")
    tokenizer = load_tokenizer(model_name=MODEL_ID)
    model = load_base_model(device, use_qlora=False, model_name=MODEL_ID)
    model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=True)
    return model.merge_and_unload(), tokenizer


@torch.no_grad()
def judge_variant(
    results: list[dict],
    variant: str,
    judge_model,
    judge_tokenizer,
    batch_size: int,
    save_path: Path,
) -> float:
    prompts = [
        _build_judge_prompt(
            question=item["query"],
            ground_truth=item["ground_truth"],
            prediction=item[variant],
        )
        for item in results
    ]
    device = str(next(judge_model.parameters()).device)
    judged = []

    for start in tqdm(
        range(0, len(prompts), batch_size),
        desc=f"Judging {variant}",
    ):
        raw_responses = _generate_from_prompts_batch(
            model=judge_model,
            tokenizer=judge_tokenizer,
            prompts=prompts[start : start + batch_size],
            device=device,
            max_new_tokens=16,
        )
        for item, prompt, raw_response in zip(
            results[start : start + batch_size],
            prompts[start : start + batch_size],
            raw_responses,
        ):
            judged.append(
                {
                    "query_id": item["query_id"],
                    "query": item["query"],
                    "ground_truth": item["ground_truth"],
                    "variant": variant,
                    "prediction": item[variant],
                    "judge_prompt": prompt,
                    "judge_response": raw_response,
                    "judge_score": _parse_judge_response(raw_response),
                }
            )

    save_jsonl(judged, save_path)
    return float(np.mean([item["judge_score"] for item in judged]))


def build_comparison_rows(all_metrics: dict[str, dict]) -> list[dict]:
    rows = []
    for model_name, metrics in all_metrics.items():
        for variant in VARIANTS:
            for metric_name in METRIC_NAMES:
                rows.append(
                    {
                        "model": model_name,
                        "setting": variant,
                        "metric": metric_name,
                        "score": float(metrics[f"{metric_name}_{variant}"]),
                    }
                )
            judge_key = f"LLM_Judge_{variant}"
            if judge_key in metrics:
                rows.append(
                    {
                        "model": model_name,
                        "setting": variant,
                        "metric": "LLM_Judge",
                        "score": float(metrics[judge_key]),
                    }
                )
    return rows


def save_comparison_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "setting", "metric", "score"],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_comparison(rows: list[dict], model_names: list[str]) -> None:
    lookup = {
        (row["setting"], row["metric"], row["model"]): row["score"]
        for row in rows
    }
    metrics = list(METRIC_NAMES)
    if any(row["metric"] == "LLM_Judge" for row in rows):
        metrics.append("LLM_Judge")

    print("\nFINAL COMPARISON")
    header = f"{'Setting':<10} {'Metric':<12}" + "".join(
        f" {name:>24}" for name in model_names
    )
    print(header)
    print("-" * len(header))
    for variant in VARIANTS:
        for metric_name in metrics:
            values = [
                lookup.get((variant, metric_name, model_name), float("nan"))
                for model_name in model_names
            ]
            print(
                f"{variant:<10} {metric_name:<12}"
                + "".join(f" {value:>24.4f}" for value in values)
            )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be greater than zero")
    if args.judge_batch_size <= 0:
        raise ValueError("--judge_batch_size must be greater than zero")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be greater than zero")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device()
    adapter_path = Path(args.adapter_path)
    rankings_path = Path(args.rankings_path)
    if not rankings_path.is_file():
        raise FileNotFoundError(f"Rankings file not found: {rankings_path}")

    run_dir = (
        Path(args.output_dir)
        if args.output_dir
        else OUTPUTS_DIR / f"small_models_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    if args.resume and args.output_dir is None:
        raise ValueError("--resume requires --output_dir pointing to the previous run")
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    answers_dir = run_dir / "answers"
    judge_dir = run_dir / "judge"

    print(f"Device: {device}")
    print(f"Loading test split from {args.dataset_name}")
    dataset = load_dataset(
        args.dataset_name,
        split="test",
        token=args.hf_token or HF_TOKEN,
    )
    rows, sample_indices = select_test_rows(dataset, args.num_samples, args.seed)
    rankings = load_rankings(str(rankings_path))
    prompt_records = build_prompt_records(rows, rankings, args.k)

    model_specs = [
        ("Qwen3-1.7B", args.qwen3_model_id, "standard", True),
        ("SmolLM2-1.7B-Instruct", args.smollm_model_id, "standard", False),
        ("Qwen2.5-3B-LoRA", str(adapter_path), "lora", False),
    ]
    all_answers = {}
    answer_paths = {}
    all_metrics = {}

    for label, source, model_type, disable_thinking in model_specs:
        print(f"\n{'=' * 80}\n{label}: {source}\n{'=' * 80}")
        path = answers_dir / f"{safe_name(label)}.jsonl"
        if args.resume and path.is_file():
            print(f"Reusing saved answers from {path}")
            answers = load_jsonl(path)
            validate_saved_answers(answers, prompt_records, path)
        else:
            if model_type == "lora":
                model, tokenizer = load_fine_tuned_model(adapter_path, device)
            else:
                model, tokenizer = load_standard_model(source, device)

            answers = generate_answers(
                model=model,
                tokenizer=tokenizer,
                prompt_records=prompt_records,
                device=device,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                disable_thinking=disable_thinking,
            )
            save_jsonl(answers, path)
            print(f"Saved {label} answers to {path}")
            del model
            del tokenizer
            free_gpu_memory()

        all_answers[label] = answers
        answer_paths[label] = str(path)

        print(f"Scoring {label}")
        all_metrics[label] = evaluate_responses(answers)

    if not args.skip_judge:
        print(f"\nLoading judge model: {args.judge_model_id}")
        judge_model, judge_tokenizer = load_judge_model(
            device,
            model_name=args.judge_model_id,
        )
        for label, answers in all_answers.items():
            for variant in VARIANTS:
                judge_path = judge_dir / f"{safe_name(label)}__{variant}.jsonl"
                score = judge_variant(
                    results=answers,
                    variant=variant,
                    judge_model=judge_model,
                    judge_tokenizer=judge_tokenizer,
                    batch_size=args.judge_batch_size,
                    save_path=judge_path,
                )
                all_metrics[label][f"LLM_Judge_{variant}"] = score
        del judge_model
        del judge_tokenizer
        free_gpu_memory()

    comparison_rows = build_comparison_rows(all_metrics)
    comparison_csv = run_dir / "comparison.csv"
    save_comparison_csv(comparison_rows, comparison_csv)

    report_path = run_dir / "comparison.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": {
                    "dataset_name": args.dataset_name,
                    "split": "test",
                    "num_samples": args.num_samples,
                    "sample_indices": sample_indices,
                    "seed": args.seed,
                    "k": args.k,
                    "rankings_path": str(rankings_path),
                    "max_new_tokens": args.max_new_tokens,
                    "judge_model_id": None if args.skip_judge else args.judge_model_id,
                    "models": {
                        label: source for label, source, _, _ in model_specs
                    },
                },
                "answer_paths": answer_paths,
                "metrics": all_metrics,
                "comparison": comparison_rows,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print_comparison(comparison_rows, [spec[0] for spec in model_specs])
    print(f"\nComparison CSV: {comparison_csv}")
    print(f"Full report: {report_path}")


if __name__ == "__main__":
    main()
