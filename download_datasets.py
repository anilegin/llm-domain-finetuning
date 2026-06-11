"""
prefetch.py — run this on the LOGIN NODE (internet access required) before
submitting the SLURM job.  Downloads all HF assets into the cache so the
compute nodes can run fully offline:
  1. Qwen2.5-3B-Instruct (main model)
  2. Phi-3.5-mini-instruct (judge model, optional)
  3. Course dataset (sapienzanlp-course-materials/hw-mnlp-2026)
  4. microsoft/deberta-xlarge-mnli (BERTScore)

Note: NLTK data is downloaded by scripts/setup_env.sh — no need to handle it here.

Usage:
    python prefetch.py
    python prefetch.py --hf_token hf_...        # override token
    python prefetch.py --use_llm_judge          # also fetch Phi-3.5-mini-instruct
"""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download
from datasets import load_dataset

from config import JUDGE_MODEL_ID, MODEL_ID, HF_TOKEN, LITTLE_MODEL_1_ID, LITTLE_MODEL_2_ID, SMOLLM3_MODEL_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token (falls back to config.HF_TOKEN)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = args.hf_token or HF_TOKEN

    # ───── 1. Main model weights + tokenizer ───────────────────────────────────
    print(f"Downloading model: {MODEL_ID}")
    path = snapshot_download(repo_id=MODEL_ID, token=token)
    print(f"  cached at: {path}\n")

    # ───── 2. Little models ────────────────────────────────────────────────────
    print(f"Downloading model: {LITTLE_MODEL_1_ID}")
    path = snapshot_download(repo_id=LITTLE_MODEL_1_ID, token=token)
    print(f"  cached at: {path}\n")
    print(f"Downloading model: {LITTLE_MODEL_2_ID}")
    path = snapshot_download(repo_id=LITTLE_MODEL_2_ID, token=token)
    print(f"  cached at: {path}\n")

    # ───── 3. Judge model ──────────────────────────────────────────────────
    print(f"Downloading judge model: {JUDGE_MODEL_ID}")
    path = snapshot_download(repo_id=JUDGE_MODEL_ID, token=token)
    print(f"  cached at: {path}\n")

    # ── 4. Course dataset ─────────────────────────────────────────────────────
    dataset_id = "sapienzanlp-course-materials/hw-mnlp-2026"
    print(f"Downloading dataset: {dataset_id}")
    ds = load_dataset(dataset_id, token=token)
    splits = list(ds.keys())
    print(f"  splits: {splits}")
    for split in splits:
        print(f"  {split}: {len(ds[split])} rows")

    # ───── 5. BERTScore model ───────────────────────────────────────────────────
    bert_model_id = "microsoft/deberta-xlarge-mnli"
    print(f"Downloading BERTScore model: {bert_model_id}")
    path = snapshot_download(repo_id=bert_model_id, token=token)
    print(f"  cached at: {path}\n")

    # ───── 6. nltk ───────────────────────────────────────────────────
    import nltk
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    # ───── 7. SMOLLM ───────────────────────────────────────────────────
    print(f"Downloading SmolLM model: {SMOLLM3_MODEL_ID}")
    path = snapshot_download(
        repo_id=SMOLLM3_MODEL_ID,
        token=args.hf_token or HF_TOKEN,
    )
    print(f"Cached {SMOLLM3_MODEL_ID} at: {path}")

    print("All assets cached. You can now submit the SLURM job.")


if __name__ == "__main__":
    main()
