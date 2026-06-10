"""
main.py — entry point for LoRA / QLoRA fine-tuning of Qwen2.5-3B-Instruct.

Usage:
    python main.py                      # LoRA  (float16 / bf16)
    python main.py --use_qlora          # QLoRA (4-bit NF4)
    python main.py --full_eval          # skip training, evaluate a saved adapter
    python main.py --light_eval --responses_path outputs/.../comparison.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import DATA_DIR, MODELS_DIR, LITTLE_MODEL_1_ID, LITTLE_MODEL_2_ID
from evaluate import compare_models, evaluate_from_responses
from train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-3B-Instruct with LoRA/QLoRA for RAG"
    )
    # Paths and directories
    parser.add_argument(
        "--output_dir", type=str, default=str(MODELS_DIR / "qwen-rag-lora"),
        help="Directory to save / load the LoRA adapter",
    )
    parser.add_argument(
        "--rankings_path_training", type=str,
        default=str(DATA_DIR / "training_rankings.jsonl"),
        help="Path to pre-computed chunk rankings for training split (JSONL)",
    )
    parser.add_argument(
        "--rankings_path_test", type=str,
        default=str(DATA_DIR / "test_rankings.jsonl"),
        help="Path to pre-computed chunk rankings for test split (JSONL)",
    )
    parser.add_argument(
        "--rankings_path_blind", type=str,
        default=str(DATA_DIR / "blind_rankings.jsonl"),
        help="Path to pre-computed chunk rankings for blind split (JSONL)",
    )
    parser.add_argument(
        "--responses_path", type=str, default=None,
        help="Path to a saved comparison.json to use with --light_eval",
    )
    parser.add_argument(
        "--llm_judge_dir", type=str, default=None,
        help="Path to saved LLM judge outputs to use with --light_eval",
    )
    # Training arguments
    parser.add_argument(
        "--use_qlora", action="store_true",
        help="Use QLoRA (4-bit NF4 quantisation) instead of standard LoRA",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=8,
        help="Gradient accumulation steps (effective batch = batch_size × this)",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=1024,
        help="Maximum sequence length for training",
    )
    # Evaluation arguments
    parser.add_argument(
        "--k", type=int, default=3,
        help="Number of top-k chunks to include in the RAG prompt",
    )
    parser.add_argument(
        "--eval_samples", type=int, default=200,
        help="Number of samples for the comparison evaluation",
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16,
        help="Batch size to use for evaluation / generation",
    )
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="HuggingFace token for dataset access (overrides HF_TOKEN env var)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--run_llm_judge", action="store_true",
        help="Score RAG responses with Phi-3.5-mini-instruct as an LLM judge",
    )
    parser.add_argument(
        "--run_little_models_eval", action="store_true", dest="little_models_eval",
        help="Run evaluation on little models",
    )
    # Type of run
    parser.add_argument(
        "--train_rag", action="store_true",
        help="Train the RAG model",
    )
    parser.add_argument(
        "--full_eval", action="store_true", default=False,
        help="only runs compare_models, this is a full comparison of 2 models. Complete generation and evaluation",
    )
    parser.add_argument(
        "--light_eval", action="store_true",
        help="Skip response generation; recompute metrics from a saved comparison.json",
    )
    parser.add_argument(
        "--test_only", action="store_true",
        help="Only run --full-eval on the test set",
    )
    parser.add_argument(
        "--blind_only", action="store_true",
        help="Only run --full-eval on the blind set",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.train_rag:
        adapter_path = train(
            use_qlora=args.use_qlora,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_seq_length=args.max_seq_length,
            rankings_path=args.rankings_path_training,
            hf_token=args.hf_token,
            k=args.k,
            seed=args.seed,
        )
        compare_models(
            adapter_path=adapter_path,
            rankings_path=args.rankings_path_test,
            llm_judge_outputs_dir=args.llm_judge_dir,
            hf_token=args.hf_token,
            k=args.k,
            num_samples=args.eval_samples,
            seed=args.seed,
            split="test",
            use_llm_judge=args.run_llm_judge,
            batch_size=args.eval_batch_size,
        )
        compare_models(
            adapter_path=adapter_path,
            rankings_path=args.rankings_path_blind,
            llm_judge_outputs_dir=args.llm_judge_dir,
            hf_token=args.hf_token,
            k=args.k,
            num_samples=args.eval_samples,
            seed=args.seed,
            split="blind",
            use_llm_judge=args.run_llm_judge,
            batch_size=args.eval_batch_size,
        )
    
    elif args.full_eval:
        if not args.test_only and not args.blind_only:
            adapter_path = args.output_dir
            if not args.little_models_eval:
                
                compare_models(
                    adapter_path=adapter_path,
                    rankings_path=args.rankings_path_test,
                    llm_judge_outputs_dir=args.llm_judge_dir,
                    hf_token=args.hf_token,
                    k=args.k,
                    num_samples=args.eval_samples,
                    seed=args.seed,
                    split="test",
                    use_llm_judge=args.run_llm_judge,
                    batch_size=args.eval_batch_size,
                )

                compare_models(
                    adapter_path=adapter_path,
                    rankings_path=args.rankings_path_blind,
                    llm_judge_outputs_dir=args.llm_judge_dir,
                    hf_token=args.hf_token,
                    k=args.k,
                    num_samples=args.eval_samples,
                    seed=args.seed,
                    split="blind",
                    use_llm_judge=args.run_llm_judge,
                    batch_size=args.eval_batch_size,
                )
            else:
                compare_models(
                    adapter_path=adapter_path,
                    rankings_path=args.rankings_path_test,
                    llm_judge_outputs_dir=args.llm_judge_dir,
                    hf_token=args.hf_token,
                    k=args.k,
                    num_samples=args.eval_samples,
                    seed=args.seed,
                    split="test",
                    use_llm_judge=args.run_llm_judge,
                    batch_size=args.eval_batch_size,
                    additional_models=[LITTLE_MODEL_1_ID,LITTLE_MODEL_2_ID]
                )

                compare_models(
                    adapter_path=adapter_path,
                    rankings_path=args.rankings_path_blind,
                    llm_judge_outputs_dir=args.llm_judge_dir,
                    hf_token=args.hf_token,
                    k=args.k,
                    num_samples=args.eval_samples,
                    seed=args.seed,
                    split="blind",
                    use_llm_judge=args.run_llm_judge,
                    batch_size=args.eval_batch_size,
                    additional_models=[LITTLE_MODEL_1_ID,LITTLE_MODEL_2_ID]
                )
        elif args.test_only:
            adapter_path = args.output_dir
            compare_models(
                adapter_path=adapter_path,
                rankings_path=args.rankings_path_test,
                llm_judge_outputs_dir=args.llm_judge_dir,
                hf_token=args.hf_token,
                k=args.k,
                num_samples=args.eval_samples,
                seed=args.seed,
                split="test",
                use_llm_judge=args.run_llm_judge,
                batch_size=args.eval_batch_size,
                additional_models=[LITTLE_MODEL_1_ID,LITTLE_MODEL_2_ID]
            )
        elif args.blind_only:
            adapter_path = args.output_dir
            compare_models(
                adapter_path=adapter_path,
                rankings_path=args.rankings_path_blind,
                llm_judge_outputs_dir=args.llm_judge_dir,
                hf_token=args.hf_token,
                k=args.k,
                num_samples=args.eval_samples,
                seed=args.seed,
                split="blind",
                use_llm_judge=args.run_llm_judge,
                batch_size=args.eval_batch_size,
                additional_models=[LITTLE_MODEL_1_ID,LITTLE_MODEL_2_ID]
            )
        else:
            raise ValueError("--test_only and --blind_only cannot be used together")

    elif args.light_eval:
        if not args.responses_path:
            raise ValueError("--light_eval requires --responses_path")
        # this includes both base and ft results
        if args.little_models_eval:
            evaluate_from_responses(
                responses_path=args.responses_path, 
                llm_judge_dir=args.llm_judge_dir,
                num_samples=args.eval_samples,
                seed=args.seed,
                additional_models=[LITTLE_MODEL_1_ID,LITTLE_MODEL_2_ID]
            )
        else:
            evaluate_from_responses(
                responses_path=args.responses_path, 
                llm_judge_dir=args.llm_judge_dir,
                num_samples=args.eval_samples,
                seed=args.seed,
            )

    else:
        print("No mode selected. Please select a mode to run.")
        print("Options: --train_rag, --full_eval, --light_eval")
            