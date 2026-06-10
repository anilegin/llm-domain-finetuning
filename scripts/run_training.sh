#!/bin/bash
#SBATCH --job-name=qwen-rag-lora-train
#SBATCH --account=iscrc_mnlp26
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80GB
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ==================== Directories ====================
PROJECT_DIR="$HOME/projects/llm-domain-finetuning"
cd "$PROJECT_DIR"

mkdir -p logs models outputs

# ==================== Modules ====================
module load python/3.11.7
module load cuda/12.6

# ==================== Environment ====================
# If anilegin is a venv inside the project folder:
VENV_NAME="anilegin"

if [ -f "$PROJECT_DIR/$VENV_NAME/bin/activate" ]; then
    echo "Activating venv: $PROJECT_DIR/$VENV_NAME"
    source "$PROJECT_DIR/$VENV_NAME/bin/activate"
else
    echo "Could not find venv at $PROJECT_DIR/$VENV_NAME/bin/activate"
    echo "Trying conda activate anilegin instead..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate anilegin
fi

echo "Python executable:"
which python
python --version

# ==================== CUDA / NVIDIA Python libs ====================
# Avoid cluster-wide CUDA package mismatch if pip-installed nvidia packages exist.
if [ -n "${VIRTUAL_ENV:-}" ]; then
    for dir in "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/"*/lib; do
        if [ -d "$dir" ]; then
            export LD_LIBRARY_PATH="$dir:$LD_LIBRARY_PATH"
        fi
    done
fi

# ==================== Offline / cache settings ====================
export NLTK_DATA="$PROJECT_DIR/nltk_data"

# Use cache only. Make sure you already ran prefetch.py on login node.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export TOKENIZERS_PARALLELISM=false

# Optional: put HF cache explicitly if needed
# export HF_HOME="$PROJECT_DIR/.cache/huggingface"

# ==================== Debug info ====================
echo "============================================="
echo "Job started at: $(date)"
echo "Running on node: ${SLURM_NODENAME:-not set}"
echo "Working directory: $(pwd)"
echo "GPU info:"
nvidia-smi
echo "============================================="

# ==================== Cache sanity check ====================
python - <<'EOF'
import sys
from huggingface_hub import scan_cache_dir
from config import MODEL_ID

DATASET_ID = "sapienzanlp-course-materials/hw-mnlp-2026"

info = scan_cache_dir()
cached = {r.repo_id for r in info.repos}

missing = []

if MODEL_ID not in cached:
    missing.append(f"Model not in HF cache: {MODEL_ID}")

if DATASET_ID not in cached:
    missing.append(f"Dataset not in HF cache: {DATASET_ID}")

if missing:
    print("ERROR: Required assets are missing from HF cache.")
    print("Run prefetch.py on the login node first.")
    for m in missing:
        print(f"  - {m}")
    sys.exit(1)

print("Cache check passed.")
print(f"Model found: {MODEL_ID}")
print(f"Dataset found: {DATASET_ID}")
EOF

# ==================== Fine-tuning ====================
# A100 + Qwen2.5-3B + LoRA.
# k=3 because your retrieved examples look clean.
# max_seq_length=4096 because A100 can handle longer RAG context.
# batch_size=2 is safer for 4096 context; grad_accum=8 keeps effective batch size 16.

python main.py \
    --train_rag \
    --epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --max_seq_length 4096 \
    --k 3 \
    --output_dir models/qwen-rag-lora-k3-seq4096-lr1e4 \
    --rankings_path_training training_data/training_rankings.jsonl \
    --rankings_path_test training_data/test_rankings.jsonl \
    --rankings_path_blind training_data/blind_rankings.jsonl

echo "============================================="
echo "Training finished at: $(date)"
echo "Saved adapter to: models/qwen-rag-lora-k3-seq4096-lr1e4"
echo "============================================="