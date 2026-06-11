#!/bin/bash
#SBATCH --job-name=small-model-eval
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

# The logs directory must already exist before sbatch starts.
mkdir -p models outputs

module purge
module load python/3.11.7
module load cuda/12.6

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: Virtual environment not found: $VENV_DIR"
    exit 1
fi

source "$VENV_DIR/bin/activate"

# Do not manually add pip-installed NVIDIA libraries to LD_LIBRARY_PATH.
# nvidia-smi must use the compute node's system NVML library.

export NLTK_DATA="$PROJECT_DIR/nltk_data"

export HF_HOME="$SCRATCH/.cache/huggingface"
export TORCH_HOME="$SCRATCH/.cache/torch"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "============================================="
echo "Job ID: ${SLURM_JOB_ID:-not-set}"
echo "Node: $(hostname)"
echo "Allocated nodes: ${SLURM_JOB_NODELIST:-not-set}"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
python --version

echo "Package versions:"
python -c \
'import torch, transformers, trl, peft; print("torch:", torch.__version__); print("transformers:", transformers.__version__); print("trl:", trl.__version__); print("peft:", peft.__version__)'

echo "GPU information:"
nvidia-smi

echo "PyTorch CUDA check:"
python -c \
'import torch; print("CUDA available:", torch.cuda.is_available()); print("CUDA version:", torch.version.cuda); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")'

echo "============================================="

# python compare_models.py \
#     --num_samples 300 \
#     --k 3 \
#     --batch_size 16 \
#     --judge_batch_size 16 \
#     --max_new_tokens 256 \
#     --adapter_path models/qwen-rag-lora-k3-seq4096-lr1e4/checkpoint-1500 \
#     --rankings_path training_data/blind_rankings.jsonl

# python compare_models.py \
#   --output_dir outputs/small_models_test_20260611_095927 \
#   --resume

python compare_models.py \
    --split test \
    --num_samples 300 \
    --k 3 \
    --batch_size 16 \
    --judge_batch_size 16 \
    --max_new_tokens 256 \
    --adapter_path models/qwen-rag-lora-k3-seq4096-lr1e4/checkpoint-1500 \
    --rankings_path training_data/test_rankings.jsonl

  # python compare_models.py \
  #   --split test \
  #   --num_samples 300 \
  #   --k 3 \
  #   --batch_size 16 \
  #   --judge_batch_size 16 \
  #   --max_new_tokens 256 \
  #   --adapter_path models/qwen-rag-lora-k3-seq4096-lr1e4/checkpoint-1500 \
  #   --rankings_path training_data/test_rankings.jsonl

echo "============================================="
echo "Evaluation completed: $(date)"
echo "============================================="