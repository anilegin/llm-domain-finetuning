from __future__ import annotations

import random

import torch
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from data_utils import build_chat_messages, format_chunks
from config import (
    SMOLLM3_LORA_ALPHA,
    SMOLLM3_LORA_BIAS,
    SMOLLM3_LORA_DROPOUT,
    SMOLLM3_LORA_R,
    SMOLLM3_LORA_TARGET_MODULES,
    SMOLLM3_MODEL_ID,
)


def get_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_smollm3_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        SMOLLM3_MODEL_ID,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_smollm3_model(device: str, use_qlora: bool = False):
    kwargs = {
        "torch_dtype": get_compute_dtype(),
        "device_map": device,
        "local_files_only": True,
    }
    if use_qlora:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=get_compute_dtype(),
            bnb_4bit_use_double_quant=True,
        )
    return AutoModelForCausalLM.from_pretrained(SMOLLM3_MODEL_ID, **kwargs)


def apply_smollm3_chat_template(
    tokenizer,
    messages: list[dict],
    *,
    add_generation_prompt: bool,
    tokenize: bool,
):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def prepare_smollm3_training_data(
    dataset,
    tokenizer,
    rankings: dict | None,
    k: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    formatted = []
    used_rankings = 0
    missing_rankings = 0

    for row in dataset:
        query_id = str(row["query_id"])
        chunks = row["candidate_chunks"]
        if rankings and query_id in rankings:
            ranking = list(rankings[query_id])
            used_rankings += 1
        else:
            ranking = list(range(len(chunks)))
            rng.shuffle(ranking)
            missing_rankings += 1

        context = format_chunks(chunks, ranking, k=k)
        messages = build_chat_messages(
            query=row["query"],
            context=context,
            answer=row["short_answer"][0],
        )
        formatted.append(
            {
                "text": apply_smollm3_chat_template(
                    tokenizer,
                    messages,
                    add_generation_prompt=False,
                    tokenize=False,
                )
            }
        )

    print(f"Used rankings: {used_rankings}")
    print(f"Missing rankings / random fallback: {missing_rankings}")
    rng.shuffle(formatted)
    return formatted


def derive_assistant_prefix_ids(tokenizer) -> list[int]:
    """Derive the completion marker from SmolLM3's installed chat template."""
    messages = [
        {"role": "system", "content": "System marker."},
        {"role": "user", "content": "User marker."},
    ]
    without_generation = apply_smollm3_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=False,
        tokenize=True,
    )
    with_generation = apply_smollm3_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=True,
        tokenize=True,
    )

    common_length = 0
    for left, right in zip(without_generation, with_generation):
        if left != right:
            break
        common_length += 1

    assistant_prefix = with_generation[common_length:]
    if not assistant_prefix:
        raise RuntimeError(
            "Could not derive SmolLM3 assistant prefix from its chat template"
        )
    return assistant_prefix


def validate_completion_marker(
    tokenizer,
    response_template_ids: list[int],
    sample_text: str,
) -> None:
    sample_ids = tokenizer.encode(sample_text, add_special_tokens=False)
    marker_length = len(response_template_ids)
    found = any(
        sample_ids[index : index + marker_length] == response_template_ids
        for index in range(len(sample_ids) - marker_length + 1)
    )
    if not found:
        marker_text = tokenizer.decode(response_template_ids)
        raise RuntimeError(
            "The tokenizer-derived assistant marker was not found in a formatted "
            f"training sample. Marker: {marker_text!r}"
        )


def validate_lora_targets(model) -> list[str]:
    module_suffixes = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing = [
        name for name in SMOLLM3_LORA_TARGET_MODULES if name not in module_suffixes
    ]
    if missing:
        available = sorted(
            name
            for name in module_suffixes
            if name.endswith("_proj")
        )
        raise RuntimeError(
            "SmolLM3 LoRA target modules were not found in the installed model "
            f"architecture. Missing: {missing}. Projection modules found: {available}"
        )
    return list(SMOLLM3_LORA_TARGET_MODULES)


def get_smollm3_lora_config(model) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=SMOLLM3_LORA_R,
        lora_alpha=SMOLLM3_LORA_ALPHA,
        lora_dropout=SMOLLM3_LORA_DROPOUT,
        target_modules=validate_lora_targets(model),
        bias=SMOLLM3_LORA_BIAS,
    )
