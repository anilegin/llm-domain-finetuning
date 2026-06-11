#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import transformers
import accelerate
from datasets import Dataset as HFDataset
from datasets import load_dataset
from packaging.version import Version
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from config import DATA_DIR, HF_TOKEN
from data_utils import DATASET_NAME
from config import SMOLLM3_MODEL_ID, SMOLLM3_OUTPUT_DIR
from smollm3_utils import (
    derive_assistant_prefix_ids,
    get_smollm3_lora_config,
    load_smollm3_model,
    load_smollm3_tokenizer,
    prepare_smollm3_training_data,
    validate_completion_marker,
)
from utils import free_gpu_memory, load_rankings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune SmolLM3-3B for RAG with a separate LoRA adapter"
    )
    parser.add_argument("--output_dir", default=str(SMOLLM3_OUTPUT_DIR))
    parser.add_argument(
        "--rankings_path",
        default=str(DATA_DIR / "training_rankings.jsonl"),
    )
    parser.add_argument("--dataset_name", default=DATASET_NAME)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()
    if Version(transformers.__version__) < Version("4.53.0"):
        raise RuntimeError(
            f"{SMOLLM3_MODEL_ID} requires transformers>=4.53.0; "
            f"found {transformers.__version__}"
        )
    if Version(accelerate.__version__) < Version("1.10.1"):
        raise RuntimeError(
            "transformers 4.57.x requires a newer Accelerate API for Trainer. "
            f"Install accelerate>=1.10.1; found {accelerate.__version__}"
        )
    if args.k <= 0 or args.batch_size <= 0:
        raise ValueError("--k and --batch_size must be greater than zero")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = choose_device()
    print(f"Using device: {device}")
    print(f"Model: {SMOLLM3_MODEL_ID}")
    print(f"Mode: {'QLoRA (4-bit)' if args.use_qlora else 'LoRA'}")
    print("Extended thinking: disabled")

    dataset = load_dataset(
        args.dataset_name,
        token=args.hf_token or HF_TOKEN,
    )
    rankings_path = Path(args.rankings_path)
    rankings = load_rankings(str(rankings_path)) if rankings_path.is_file() else None

    tokenizer = load_smollm3_tokenizer()
    training_rows = prepare_smollm3_training_data(
        dataset=dataset["train"],
        tokenizer=tokenizer,
        rankings=rankings,
        k=args.k,
        seed=args.seed,
    )
    if not training_rows:
        raise RuntimeError("No SmolLM3 training examples were created")

    response_template_ids = derive_assistant_prefix_ids(tokenizer)
    validate_completion_marker(
        tokenizer,
        response_template_ids,
        training_rows[0]["text"],
    )
    print(
        "Derived assistant completion marker: "
        f"{tokenizer.decode(response_template_ids)!r} "
        f"({len(response_template_ids)} tokens)"
    )

    train_dataset = HFDataset.from_list(training_rows)
    model = load_smollm3_model(device=device, use_qlora=args.use_qlora)
    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora_config = get_smollm3_lora_config(model)
    print(f"Validated LoRA targets: {lora_config.target_modules}")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    steps_per_epoch = max(
        1,
        len(train_dataset)
        // (args.batch_size * args.gradient_accumulation_steps),
    )
    warmup_steps = max(1, int(steps_per_epoch * args.epochs * 0.05))

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        seed=args.seed,
        report_to="none",
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=args.max_seq_length,
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

    print("Starting SmolLM3 training")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    free_gpu_memory()
    print(f"Training complete. Adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
