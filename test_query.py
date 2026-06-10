#!/usr/bin/env python3
"""
test_query.py - A simple CLI script to query the fine-tuned or base Qwen model.

Usage:
    python test_query.py --query "What is the capital of France?"
    python test_query.py --query "Who wrote Romeo and Juliet?" --context "Shakespeare wrote Romeo and Juliet in the late 16th century."
    python test_query.py --query "What is the capital of France?" --use_base
"""

import argparse
import sys
import torch
from peft import PeftModel

from config import MODELS_DIR, SYSTEM_PROMPT
from model_utils import load_base_model, load_tokenizer
from data_utils import build_chat_messages


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query the Qwen model (base or fine-tuned) with an optional context."
    )
    parser.add_argument(
        "--query", "-q", type=str, required=True,
        help="The query/question to send to the model."
    )
    parser.add_argument(
        "--context", "-c", type=str, default=None,
        help="Optional context information for RAG. If provided, the prompt is formatted as RAG."
    )
    parser.add_argument(
        "--adapter_path", "-a", type=str, default=str(MODELS_DIR / "qwen-rag-lora"),
        help="Path to the LoRA adapter directory."
    )
    parser.add_argument(
        "--use_base", "-b", action="store_true",
        help="If set, query the base model without loading the fine-tuned adapter."
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="Maximum number of new tokens to generate."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[INFO] Using device: {device}", flush=True)

    # Load tokenizer
    print("[INFO] Loading tokenizer...", flush=True)
    tokenizer = load_tokenizer()

    # Load model
    if args.use_base:
        print("[INFO] Loading base model...", flush=True)
        model = load_base_model(device, use_qlora=False)
    else:
        print(f"[INFO] Loading base model + adapter from {args.adapter_path}...", flush=True)
        model = load_base_model(device, use_qlora=False)
        try:
            model = PeftModel.from_pretrained(model, args.adapter_path)
            model = model.merge_and_unload()
            print("[INFO] LoRA adapter successfully merged.", flush=True)
        except Exception as e:
            print(f"[ERROR] Failed to load adapter from {args.adapter_path}: {e}", file=sys.stderr, flush=True)
            print("[INFO] Falling back to base model only.", flush=True)

    model.eval()

    # Format the prompt
    if args.context:
        print("[INFO] Formatting prompt with provided context (RAG mode)...", flush=True)
        messages = build_chat_messages(args.query, args.context, answer=None)
    else:
        print("[INFO] Formatting prompt without context (Baseline mode)...", flush=True)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": args.query},
        ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Generate response
    print("[INFO] Generating response...", flush=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("\n" + "=" * 40 + " QUERY " + "=" * 40)
    print(args.query)
    if args.context:
        print("\n" + "=" * 40 + " CONTEXT " + "=" * 40)
        print(args.context)
    print("\n" + "=" * 40 + " RESPONSE " + "=" * 40)
    print(response)
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
