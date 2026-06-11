#!/bin/bash
#SBATCH --job-name=smollm3-rag-lora
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

PROJECT_DIR="$HOME/projects/llm-domain-finetuning"
VENV_DIR="$PROJECT_DIR/anilegin"

cd "$PROJECT_DIR"
mkdir -p models outputs

module purge
module load python/3.11.7
module load cuda/12.6

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: Virtual environment not found: $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "============================================="
echo "Job ID: ${SLURM_JOB_ID:-not-set}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Python: $(which python)"
python --version
python -c 'import torch, transformers, trl, peft; print("torch", torch.__version__); print("transformers", transformers.__version__); print("trl", trl.__version__); print("peft", peft.__version__)'
nvidia-smi
echo "============================================="

python - <<'PY'
import sys
from huggingface_hub import scan_cache_dir
from config import SMOLLM3_MODEL_ID

cached = {repo.repo_id for repo in scan_cache_dir().repos}
if SMOLLM3_MODEL_ID not in cached:
    print(f"ERROR: {SMOLLM3_MODEL_ID} is not in the Hugging Face cache.")
    print("Run this on the login node first: python download_smollm3.py")
    sys.exit(1)
print(f"Cache check passed: {SMOLLM3_MODEL_ID}")
PY

python train_smollm3.py \
    --epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --max_seq_length 4096 \
    --k 3 \
    --output_dir models/smollm3-rag-lora-k3-seq4096-lr1e4 \
    --rankings_path training_data/training_rankings.jsonl

echo "============================================="
echo "Training finished: $(date)"
echo "Adapter: models/smollm3-rag-lora-k3-seq4096-lr1e4"
echo "============================================="
