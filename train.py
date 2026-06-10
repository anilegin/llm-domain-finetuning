from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, Dataset as HFDataset
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

from config import HF_TOKEN, MODELS_DIR
from data_utils import prepare_training_data
from model_utils import get_lora_config, load_base_model, load_tokenizer
from utils import free_gpu_memory, load_rankings


def train(
    use_qlora: bool = False,
    output_dir: str = str(MODELS_DIR / "qwen-rag-lora"),
    epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    max_seq_length: int = 1024,
    rankings_path: str | None = None,
    hf_token: str | None = None,
    k: int = 3,
    seed: int = 42,
) -> str:
    """Run the full LoRA / QLoRA fine-tuning pipeline."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    print(f"Mode: {'QLoRA (4-bit)' if use_qlora else 'LoRA (float16)'}")

    print("Loading dataset …")
    ds = load_dataset(
        "sapienzanlp-course-materials/hw-mnlp-2026",
        token=hf_token or HF_TOKEN,
    )

    rankings = None
    if rankings_path and Path(rankings_path).exists():
        print(f"Loading pre-computed rankings from {rankings_path}")
        rankings = load_rankings(rankings_path)

    tokenizer = load_tokenizer()

    train_split = "train"
    print(f"Preparing training data from '{train_split}' split …")
    train_data = prepare_training_data(
        ds[train_split],
        tokenizer,
        rankings=rankings,
        k=k,
        seed=seed,
    )
    print(f"Created {len(train_data)} training examples")

    train_dataset = HFDataset.from_list(train_data)

    print(f"Loading model: {__import__('config').MODEL_ID}")
    model = load_base_model(device, use_qlora=use_qlora)

    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prefer bf16 on hardware that supports it (e.g. A100); fall back to fp16.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    steps_per_epoch = max(1, len(train_dataset) // (batch_size * gradient_accumulation_steps))
    warmup_steps = max(1, int(steps_per_epoch * epochs * 0.05))

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        seed=seed,
        report_to="none",
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=max_seq_length,
    )

    # Search for the response template in token space (robust to BPE boundary effects).
    # The template marks where assistant completions begin in Qwen2.5's chat format.
    response_template_ids = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training …")
    trainer.train()

    print(f"Saving LoRA adapter to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    free_gpu_memory()
    print("Training complete!\n")
    return output_dir
