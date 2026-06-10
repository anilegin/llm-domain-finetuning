from __future__ import annotations

import random

from config import SYSTEM_PROMPT


def format_chunks(candidate_chunks: list, indices: list, k: int = 3) -> str:
    top_indices = indices[:k]
    if any(idx < 0 or idx >= len(candidate_chunks) for idx in top_indices):
        raise ValueError(
            f"indices contains out-of-range values for candidate_chunks "
            f"Length:{len(candidate_chunks)}\n Ranking:{top_indices}"
        )
    return "\n".join(
        f"{i+1}. {candidate_chunks[idx]}" for i, idx in enumerate(top_indices)
    )


def build_rag_user_message(query: str, chunks_text: str) -> str:
    return (
        f"Given the following information \n\n{chunks_text}\n\n"
        f"Reply to this question:\n\n{query} \n\n Be concise in your answer"
    )


def build_chat_messages(query: str, context: str, answer: str | None = None) -> list:
    """
    Build ChatML-style messages for Qwen2.5-Instruct.
    If `answer` is provided the assistant turn is included (for training).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_rag_user_message(query, context)},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def prepare_training_data(
    dataset,
    tokenizer,
    rankings: dict | None = None,
    k: int = 3,
    seed: int = 42,
) -> list:
    """
    Convert a HuggingFace dataset split into formatted chat strings for SFTTrainer.

    For each example we create a training sample with top-k retrieved chunks as
    context.  If no pre-computed rankings are available we fall back to a random
    ordering with the correct chunk guaranteed in the top-k.

    Returns a list of dictionaries with two keys:
        "prompt": The RAG prompt a model would get
        "completion": The gold standard answer to the prompt
    """
    rng = random.Random(seed)
    formatted = []

    for row in dataset:
        query = row["query"]
        answer = row["short_answer"][0]
        chunks = row["candidate_chunks"]
        query_id = str(row["query_id"])

        if rankings and query_id in rankings:
            full_ranking = rankings[query_id]
        else:
            full_ranking = list(range(len(chunks)))
            rng.shuffle(full_ranking)

        rag_context = format_chunks(chunks, full_ranking, k=k)
        full_messages = build_chat_messages(query, rag_context, answer)
        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        formatted.append({"text": full_text})

    rng.shuffle(formatted)
    return formatted
