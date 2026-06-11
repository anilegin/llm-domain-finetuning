from __future__ import annotations

import os
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
LITTLE_MODEL_1_ID = "Qwen/Qwen3-1.7B"
LITTLE_MODEL_2_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
#JUDGE_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
JUDGE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

DATA_DIR = Path("training_data")
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")

SMOLLM3_MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
SMOLLM3_OUTPUT_DIR = Path("models/smollm3-rag-lora-k3-seq4096-lr1e4")

SMOLLM3_LORA_R = 16
SMOLLM3_LORA_ALPHA = 32
SMOLLM3_LORA_DROPOUT = 0.05
SMOLLM3_LORA_BIAS = "none"

# These names are validated against model.named_modules() before PEFT is applied.
SMOLLM3_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Course-provided token for the dataset; override via HF_TOKEN env variable.
HF_TOKEN: str = os.environ.get(
    "HF_TOKEN"
)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the "
    "provided context. Give concise, accurate answers."
)

# LoRA configuration

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_BIAS = "none"

# Choose one of:
# "attention" -> q_proj, k_proj, v_proj, o_proj
# "mlp"       -> gate_proj, up_proj, down_proj
# "all"       -> attention + mlp
# "qv"        -> q_proj, v_proj only
# "custom"    -> use LORA_CUSTOM_TARGET_MODULES below
LORA_TARGET_MODULES_PRESET = "all"

LORA_TARGET_MODULE_PRESETS = {
    "attention": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    "mlp": [
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "all": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "qv": [
        "q_proj",
        "v_proj",
    ],
}

# Used only when LORA_TARGET_MODULES_PRESET = "custom"
LORA_CUSTOM_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]
