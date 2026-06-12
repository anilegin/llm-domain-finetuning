from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

from config import SYSTEM_PROMPT, HF_TOKEN
from model_utils import load_tokenizer


DATASET_NAME = "sapienzanlp-course-materials/hw-mnlp-2026"


UNKNOWN_ANSWER = "I don't know based on the retrieved information."


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


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def chunk_contains_answer(chunk: str, answers: list[str]) -> bool:
    """
    Simple exact-string check.

    If the gold short answer appears inside a chunk, we treat that chunk as answer-supporting.
    """
    chunk_norm = normalize_text(chunk)

    for ans in answers:
        ans_norm = normalize_text(ans)
        if ans_norm and ans_norm in chunk_norm:
            return True

    return False


def format_chunks(candidate_chunks: list, indices: list, k: int = 3) -> str:
    top_indices = indices[:k]

    if any(idx < 0 or idx >= len(candidate_chunks) for idx in top_indices):
        raise ValueError(
            f"indices contains out-of-range values for candidate_chunks. "
            f"Length: {len(candidate_chunks)}\nRanking: {top_indices}"
        )

    if len(top_indices) == 0:
        return "No retrieved passage contains enough information to answer the question."

    return "\n\n".join(
        f"[Passage {i + 1} | chunk_index={idx}]\n{candidate_chunks[idx]}"
        for i, idx in enumerate(top_indices)
    )


def build_rag_user_message(query: str, chunks_text: str) -> str:
    return (
        "Use the following retrieved information to answer the question.\n\n"
        f"Retrieved information:\n{chunks_text}\n\n"
        f"Question:\n{query}\n\n"
        "Answer with the shortest correct answer supported by the retrieved information. "
        "If the retrieved information does not contain the answer, say you don't know."
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
    corrupt_answer: bool = False,
    unknown_answer: str = UNKNOWN_ANSWER,
) -> dict:
    """
    Build one formatted training example from one dataset row.

    If corrupt_answer=True:
        1. Remove chunks that contain the gold answer when possible.
        2. Replace the assistant answer with an unknown-answer target.

    Returns metadata too, useful for debugging.
    """
    if rng is None:
        rng = random.Random(42)

    query = row["query"]
    gold_answers = row["short_answer"]
    gold_answer = gold_answers[0]
    chunks = row["candidate_chunks"]
    query_id = str(row["query_id"])

    if rankings and query_id in rankings:
        full_ranking = list(rankings[query_id])
        ranking_source = "rankings_file"
    else:
        full_ranking = list(range(len(chunks)))
        rng.shuffle(full_ranking)
        ranking_source = "random_fallback"

    original_top_k_indices = full_ranking[:k]

    if corrupt_answer:
        context_ranking = full_ranking[-k:]
        answer = unknown_answer
        corruption_status = "corrupted"
        corruption_note = "Used the last k ranked chunks instead of the top k chunks."
    else:
        context_ranking = full_ranking[:k]
        answer = gold_answer
        corruption_status = "clean"
        corruption_note = "Used the top k ranked chunks."

    rag_context = format_chunks(chunks, context_ranking, k=len(context_ranking))
    final_top_k_indices = context_ranking[:k]

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
        "gold_answer": gold_answer,
        "answer": answer,
        "ranking_source": ranking_source,
        "original_top_k_indices": original_top_k_indices,
        "top_k_indices": final_top_k_indices,
        "num_candidate_chunks": len(chunks),
        "corruption_status": corruption_status,
        "corruption_note": corruption_note,
    }


def prepare_training_data(
    dataset,
    tokenizer,
    rankings: dict | None = None,
    k: int = 3,
    seed: int = 42,
    corrupt_answers: bool = False,
    corrupt_ratio: float = 0.15,
    unknown_answer: str = UNKNOWN_ANSWER,
) -> list:
    """
    Convert a HuggingFace dataset split into formatted ChatML strings for SFTTrainer.

    Normal examples:
        system message: general assistant instruction
        user message: top-k retrieved chunks + question
        assistant message: gold answer

    Corrupted examples:
        user message: last-k ranked chunks + question
        assistant message: "I don't know based on the retrieved information."

    Returns:
        list[dict]: each item has {"text": full_chatml_training_text}
    """
    rng = random.Random(seed)
    formatted = []

    used_rankings = 0
    missing_rankings = 0
    corrupted_count = 0

    dataset_len = len(dataset)

    if corrupt_answers:
        if not 0.0 <= corrupt_ratio <= 1.0:
            raise ValueError(
                f"corrupt_ratio must be between 0 and 1. Got {corrupt_ratio}"
            )

        n_corrupt = int(dataset_len * corrupt_ratio)

        # If corruption is enabled and dataset is not empty, corrupt at least one example.
        if dataset_len > 0 and n_corrupt == 0:
            n_corrupt = 1

        n_corrupt = min(n_corrupt, dataset_len)
        corrupt_indices = set(rng.sample(range(dataset_len), n_corrupt))
    else:
        n_corrupt = 0
        corrupt_indices = set()

    for row_idx, row in enumerate(dataset):
        should_corrupt = row_idx in corrupt_indices

        example = build_one_training_example(
            row=row,
            tokenizer=tokenizer,
            rankings=rankings,
            k=k,
            rng=rng,
            corrupt_answer=should_corrupt,
            unknown_answer=unknown_answer,
        )

        if example["ranking_source"] == "rankings_file":
            used_rankings += 1
        else:
            missing_rankings += 1

        if should_corrupt:
            corrupted_count += 1

        formatted.append({"text": example["text"]})

    print(f"Used rankings: {used_rankings}")
    print(f"Missing rankings / random fallback: {missing_rankings}")
    print(f"Corrupt mode: {corrupt_answers}")
    print(f"Corrupt ratio: {corrupt_ratio}")
    print(f"Corrupted answers: {corrupted_count} / {dataset_len}")

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
        help="Random seed for fallback rankings and corruption selection.",
    )

    parser.add_argument(
        "--corrupt_answers",
        action="store_true",
        help="If passed, corrupt selected training examples.",
    )

    parser.add_argument(
        "--num_corrupt_answers",
        type=int,
        default=10,
        help="Number of examples to corrupt when --corrupt_answers is passed.",
    )

    parser.add_argument(
        "--corrupt_this_sample",
        action="store_true",
        help="Force corruption on the inspected sample.",
    )

    parser.add_argument(
        "--unknown_answer",
        type=str,
        default=UNKNOWN_ANSWER,
        help="Replacement answer used for corrupted examples.",
    )

    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset_name}")
    ds = load_dataset(
        args.dataset_name,
        token=HF_TOKEN,
    )

    if args.split not in ds:
        raise ValueError(
            f"Split '{args.split}' not found. Available splits: {list(ds.keys())}"
        )

    dataset = ds[args.split]

    print("Loading tokenizer...")
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

    should_corrupt_sample = args.corrupt_this_sample

    if args.corrupt_answers and not args.corrupt_this_sample:
        n_corrupt = min(args.num_corrupt_answers, len(dataset))
        corrupt_indices = set(rng.sample(range(len(dataset)), n_corrupt))
        should_corrupt_sample = args.sample_idx in corrupt_indices

        print(f"\nCorruption mode enabled.")
        print(f"Number of corrupted examples: {n_corrupt}")
        print(f"Is inspected sample corrupted? {should_corrupt_sample}")

    example = build_one_training_example(
        row=row,
        tokenizer=tokenizer,
        rankings=rankings,
        k=args.k,
        rng=rng,
        corrupt_answer=should_corrupt_sample,
        unknown_answer=args.unknown_answer,
    )

    print("\n" + "=" * 100)
    print("SAMPLE METADATA")
    print("=" * 100)
    print(f"split: {args.split}")
    print(f"sample_idx: {args.sample_idx}")
    print(f"query_id: {example['query_id']}")
    print(f"ranking_source: {example['ranking_source']}")
    print(f"corruption_status: {example['corruption_status']}")
    print(f"corruption_note: {example['corruption_note']}")
    print(f"original_top_k_indices: {example['original_top_k_indices']}")
    print(f"final_top_k_indices: {example['top_k_indices']}")
    print(f"num_candidate_chunks: {example['num_candidate_chunks']}")

    print("\n" + "=" * 100)
    print("QUERY")
    print("=" * 100)
    print(example["query"])

    print("\n" + "=" * 100)
    print("GOLD ANSWER")
    print("=" * 100)
    print(example["gold_answer"])

    print("\n" + "=" * 100)
    print("TRAINING ANSWER")
    print("=" * 100)
    print(example["answer"])

    print("\n" + "=" * 100)
    print("FINAL TRAINING TEXT GIVEN TO SFTTRAINER")
    print("=" * 100)
    print(example["text"])
    print("=" * 100)


if __name__ == "__main__":
    main()