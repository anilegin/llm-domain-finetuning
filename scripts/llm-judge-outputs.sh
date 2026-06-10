#!/bin/bash
#SBATCH --job-name=qwen-llm-judge
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


python main.py \
    --light_eval \
    --responses_path outputs/YOUR_RUN_DIR/comparison.json \
    --run_little_models_eval

echo "============================================="
echo "LLM Scored outputs at: $(date)"
echo "============================================="