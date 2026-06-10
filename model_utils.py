from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType

from config import (
    JUDGE_MODEL_ID,
    MODEL_ID,
    LORA_ALPHA,
    LORA_BIAS,
    LORA_CUSTOM_TARGET_MODULES,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULE_PRESETS,
    LORA_TARGET_MODULES_PRESET,
)


# Prefer bf16 on hardware that supports it (e.g. A100); fall back to fp16.
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

float_type = torch.float16
if use_bf16:
    float_type = torch.bfloat16


def load_tokenizer(model_name: str | None = None):
    if model_name is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(device: str, use_qlora: bool = False, model_name: str | None = None):
    """Load Qwen2.5-3B-Instruct, optionally quantised to 4-bit for QLoRA."""
    kwargs: dict = dict(
        torch_dtype=float_type,
        device_map=device,
        local_files_only=True,
    )
    if use_qlora:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=float_type,
            bnb_4bit_use_double_quant=True,
        )
    if model_name is None:
        return AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    else:
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def load_judge_model(device: str):
    """Load Phi-3.5-mini-instruct as a judge; returns (model, tokenizer).

    The caller is responsible for deleting the model and freeing GPU memory
    when done:
        del judge_model
        free_gpu_memory()
    """
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_ID, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_ID,
        torch_dtype=float_type,
        device_map=device,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    return model, tokenizer


def get_lora_target_modules() -> list[str]:
    if LORA_TARGET_MODULES_PRESET == "custom":
        target_modules = LORA_CUSTOM_TARGET_MODULES
    else:
        target_modules = LORA_TARGET_MODULE_PRESETS.get(LORA_TARGET_MODULES_PRESET)

    if not target_modules:
        valid_presets = list(LORA_TARGET_MODULE_PRESETS.keys()) + ["custom"]
        raise ValueError(
            f"Invalid LORA_TARGET_MODULES_PRESET={LORA_TARGET_MODULES_PRESET!r}. "
            f"Choose one of: {valid_presets}"
        )

    return target_modules


def get_lora_config() -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=get_lora_target_modules(),
        bias=LORA_BIAS,
    )