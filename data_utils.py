# To check a sample from the training set:
# python data_utils.py --split train --rankings_path training_data/training_rankings.jsonl --sample_idx 0 --k 3


from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

from config import SYSTEM_PROMPT, HF_TOKEN
from model_utils import load_tokenizer


DATASET_NAME = "sapienzanlp-course-materials/hw-mnlp-2026"


def load_rankings(path: str | Path) -> dict:
    """
    Load JSONL rankings.

    Example line:
    {"4HZKjht3X13n": [0, 4, 23, 10, ...]}
    """
    rankings = {}

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            rankings.update(entry)

    return rankings


def format_chunks(candidate_chunks: list, indices: list, k: int = 3) -> str:
    top_indices = indices[:k]

    if any(idx < 0 or idx >= len(candidate_chunks) for idx in top_indices):
        raise ValueError(
            f"indices contains out-of-range values for candidate_chunks. "
            f"Length: {len(candidate_chunks)}\nRanking: {top_indices}"
        )

    return "\n\n".join(
        f"[Passage {i + 1} | chunk_index={idx}]\n{candidate_chunks[idx]}"
        for i, idx in enumerate(top_indices)
    )


def build_rag_user_message(query: str, chunks_text: str) -> str:
    return (
        "Use the following retrieved information to answer the question.\n\n"
        f"Retrieved information:\n{chunks_text}\n\n"
        f"Question:\n{query}\n\n"
        "Answer with the shortest correct answer supported by the retrieved information."
    )


def build_chat_messages(query: str, context: str, answer: str | None = None) -> list:
    """
    Build ChatML-style messages for Qwen2.5-Instruct.

    If answer is provided, the assistant turn is included for training.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_rag_user_message(query, context)},
    ]

    if answer is not None:
        messages.append({"role": "assistant", "content": answer})

    return messages


def build_one_training_example(
    row,
    tokenizer,
    rankings: dict | None = None,
    k: int = 3,
    rng: random.Random | None = None,
) -> dict:
    """
    Build one formatted training example from one dataset row.

    Returns metadata too, useful for debugging.
    """
    if rng is None:
        rng = random.Random(42)

    query = row["query"]
    answer = row["short_answer"][0]
    chunks = row["candidate_chunks"]
    query_id = str(row["query_id"])

    if rankings and query_id in rankings:
        full_ranking = list(rankings[query_id])
        ranking_source = "rankings_file"
    else:
        full_ranking = list(range(len(chunks)))
        rng.shuffle(full_ranking)
        ranking_source = "random_fallback"

    rag_context = format_chunks(chunks, full_ranking, k=k)
    full_messages = build_chat_messages(query, rag_context, answer)

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {
        "text": full_text,
        "query_id": query_id,
        "query": query,
        "answer": answer,
        "ranking_source": ranking_source,
        "top_k_indices": full_ranking[:k],
        "num_candidate_chunks": len(chunks),
    }


def prepare_training_data(
    dataset,
    tokenizer,
    rankings: dict | None = None,
    k: int = 3,
    seed: int = 42,
) -> list:
    """
    Convert a HuggingFace dataset split into formatted ChatML strings for SFTTrainer.

    Each example contains:
        system message: general assistant instruction
        user message: retrieved chunks + question
        assistant message: gold answer

    Returns:
        list[dict]: each item has {"text": full_chatml_training_text}
    """
    rng = random.Random(seed)
    formatted = []

    used_rankings = 0
    missing_rankings = 0

    for row in dataset:
        example = build_one_training_example(
            row=row,
            tokenizer=tokenizer,
            rankings=rankings,
            k=k,
            rng=rng,
        )

        if example["ranking_source"] == "rankings_file":
            used_rankings += 1
        else:
            missing_rankings += 1

        formatted.append({"text": example["text"]})

    print(f"Used rankings: {used_rankings}")
    print(f"Missing rankings / random fallback: {missing_rankings}")

    rng.shuffle(formatted)
    return formatted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect formatted RAG training samples."
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default=DATASET_NAME,
        help="Hugging Face dataset name.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to inspect, e.g. train/test/blind.",
    )

    parser.add_argument(
        "--rankings_path",
        type=str,
        default="training_data/training_rankings.jsonl",
        help="Path to JSONL rankings file.",
    )

    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Which dataset row to inspect.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of top retrieved chunks to include.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for fallback rankings.",
    )

    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset_name}")
    ds = load_dataset(
        args.dataset_name,
        token=HF_TOKEN,
    )

    if args.split not in ds:
        raise ValueError(f"Split '{args.split}' not found. Available splits: {list(ds.keys())}")

    dataset = ds[args.split]

    print(f"Loading tokenizer...")
    tokenizer = load_tokenizer()

    rankings = None
    rankings_path = Path(args.rankings_path)

    if rankings_path.exists():
        print(f"Loading rankings from: {rankings_path}")
        rankings = load_rankings(rankings_path)
        print(f"Loaded rankings for {len(rankings)} query_ids")
    else:
        print(f"Rankings file not found: {rankings_path}")
        print("Will use random fallback ranking.")

    if args.sample_idx < 0 or args.sample_idx >= len(dataset):
        raise ValueError(
            f"sample_idx out of range. Got {args.sample_idx}, "
            f"but dataset has {len(dataset)} rows."
        )

    rng = random.Random(args.seed)
    row = dataset[args.sample_idx]

    example = build_one_training_example(
        row=row,
        tokenizer=tokenizer,
        rankings=rankings,
        k=args.k,
        rng=rng,
    )

    print("\n" + "=" * 100)
    print("SAMPLE METADATA")
    print("=" * 100)
    print(f"split: {args.split}")
    print(f"sample_idx: {args.sample_idx}")
    print(f"query_id: {example['query_id']}")
    print(f"ranking_source: {example['ranking_source']}")
    print(f"top_k_indices: {example['top_k_indices']}")
    print(f"num_candidate_chunks: {example['num_candidate_chunks']}")

    print("\n" + "=" * 100)
    print("QUERY")
    print("=" * 100)
    print(example["query"])

    print("\n" + "=" * 100)
    print("GOLD ANSWER")
    print("=" * 100)
    print(example["answer"])

    print("\n" + "=" * 100)
    print("FINAL TRAINING TEXT GIVEN TO SFTTRAINER")
    print("=" * 100)
    print(example["text"])
    print("=" * 100)


if __name__ == "__main__":
    main()