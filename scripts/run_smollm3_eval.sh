#!/bin/bash
#SBATCH --job-name=smollm3-eval
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

export NLTK_DATA="$PROJECT_DIR/nltk_data"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Default: evaluate on the exact 300-query run you named.
# For the full test set, submit with: SOURCE_RUN_DIR= sbatch scripts/run_smollm3_eval.sh
if [ "${SOURCE_RUN_DIR+x}" ]; then
    SOURCE_RUN_DIR="$SOURCE_RUN_DIR"
else
    SOURCE_RUN_DIR="outputs/small_models_test_20260611_095927"
fi
OUTPUT_DIR="${OUTPUT_DIR:-}"
SKIP_JUDGE="${SKIP_JUDGE:-0}"

ARGS=(
    --adapter_path models/smollm3-rag-lora-k3-seq4096-lr1e4
    --rankings_path training_data/test_rankings.jsonl
    --k 3
    --batch_size 16
    --judge_batch_size 16
    --max_new_tokens 64
)

if [ -n "$SOURCE_RUN_DIR" ]; then
    ARGS+=(--source_run_dir "$SOURCE_RUN_DIR")
    if [ -f "$SOURCE_RUN_DIR/comparison.json" ]; then
        ARGS+=(--compare_report "$SOURCE_RUN_DIR/comparison.json")
    fi
fi

if [ -n "$OUTPUT_DIR" ]; then
    ARGS+=(--output_dir "$OUTPUT_DIR")
fi

if [ "$SKIP_JUDGE" = "1" ]; then
    ARGS+=(--skip_judge)
fi

echo "============================================="
echo "Job ID: ${SLURM_JOB_ID:-not-set}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Python: $(which python)"
python --version
python -c 'import torch, transformers, accelerate, trl, peft; print("torch", torch.__version__); print("transformers", transformers.__version__); print("accelerate", accelerate.__version__); print("trl", trl.__version__); print("peft", peft.__version__)'
nvidia-smi
echo "Source run dir: ${SOURCE_RUN_DIR:-FULL TEST SET}"
echo "Output dir: ${OUTPUT_DIR:-auto}"
echo "Skip judge: $SKIP_JUDGE"
echo "============================================="

python evaluate_smollm3_lora.py "${ARGS[@]}"

echo "============================================="
echo "SmolLM3 evaluation finished: $(date)"
echo "============================================="
