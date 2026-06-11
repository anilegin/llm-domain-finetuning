# Domain RAG Fine-Tuning Experiments

This repository is a small, practical playground for asking a simple question:

**Can a compact instruction model become better at short-answer RAG if we fine-tune it on retrieved passages?**

The project uses the `sapienzanlp-course-materials/hw-mnlp-2026` dataset, precomputed chunk rankings, LoRA fine-tuning, and a few evaluation settings to compare base models, small models, and RAG-tuned adapters.

The main target format is deliberately strict: given a question and retrieved passages, the model should return the shortest correct answer, not a paragraph.

The fine-tuned models can be found at: 

anilegin/Qwen2.5-3B-Instruct-FT-RAG-LoRA: https://huggingface.co/anilegin/Qwen2.5-3B-Instruct-FT-RAG-LoRA

anilegin/SmolLM-3B-Instruct-FT-RAG-LoRA: https://huggingface.co/anilegin/SmolLM-3B-Instruct-FT-RAG-LoRA


## Overall 

We started with two small instruction models under 3B parameters:

```text
Qwen/Qwen3-1.7B
HuggingFaceTB/SmolLM2-1.7B-Instruct
```

For each one, we generated answers on the test set in three settings: plain baseline, retrieved-context RAG, and oracle RAG where the gold chunk is forced to the top. Those runs gave us the first comparison scores for EM, SubEM, METEOR, BERTScore, and LLM-judge accuracy.

Then we trained RAG-specific LoRA adapters for stronger small models:

```text
Qwen/Qwen2.5-3B-Instruct
HuggingFaceTB/SmolLM3-3B
```

Both were fine-tuned with the same retrieved-passage prompt and short-answer target from `row["short_answer"][0]`. After generation, we judged the outputs with LLM-as-a-judge models, including Phi-3.5 in earlier runs and Mistral-7B-Instruct-v0.3 in the current evaluation path. The saved `outputs/` folders keep the actual answer files, judge decisions, and comparison tables.

## Project Tree

```text
.
|-- config.py                         # Qwen2.5 config, little-model IDs, LoRA defaults
|-- smollm3_config.py                 # Separate SmolLM3 config so Qwen training stays unchanged
|-- data_utils.py                     # Ranking loading and RAG prompt formatting
|-- model_utils.py                    # Qwen/base model/tokenizer loading helpers
|-- train.py                          # Qwen2.5 LoRA training pipeline
|-- train_smollm3.py                  # SmolLM3 LoRA training pipeline
|-- evaluate.py                       # Main generation, metrics, and LLM judge utilities
|-- compare_small_models.py           # 300-sample comparison for Qwen3, SmolLM2, Qwen2.5-LoRA
|-- evaluate_smollm3_lora.py          # SmolLM3-LoRA evaluation on full test or a saved 300-query set
|-- download_datasets.py              # Cache dataset/models/BERTScore assets
|-- download_smollm3.py               # Cache SmolLM3 separately
|-- training_data/
|   |-- training_rankings.jsonl
|   |-- test_rankings.jsonl
|   `-- blind_rankings.jsonl
|-- scripts/
|   |-- run_training.sh               # Qwen2.5 LoRA Slurm training
|   |-- run_smollm3_training.sh       # SmolLM3 LoRA Slurm training
|   |-- run_smollm3_eval.sh           # SmolLM3 evaluation Slurm job
|   `-- llm-judge-outputs.sh          # Re-score saved outputs with the judge
|-- models/                           # Local adapters/checkpoints, ignored by git
`-- outputs/                          # Saved generations, metrics, judge outputs, ignored by git
```

## Prompt Format

The shared RAG prompt lives in `data_utils.py`:

```text
Use the following retrieved information to answer the question.

Retrieved information:
[Passage 1 | chunk_index=...]
...

Question:
...

Answer with the shortest correct answer supported by the retrieved information.
```

Training examples append the gold assistant answer from:

```python
row["short_answer"][0]
```

So both Qwen2.5 and SmolLM3 LoRA training use short-answer supervision.

For SmolLM3 evaluation, there is an extra inference-only nudge because SmolLM3 tends to answer in full sentences:

```text
Important: output only the shortest answer string.
Do not write a full sentence, explanation, prefix, citation, or reasoning.
```

That extra line is only for SmolLM3 evaluation; the older Qwen path is left as it was.

## Models and Settings

### Fine-Tuned Qwen2.5

Base model:

```text
Qwen/Qwen2.5-3B-Instruct
```

Adapter path used in the main experiments:

```text
models/qwen-rag-lora-k3-seq4096-lr1e4/checkpoint-1500
```

Training settings:

```text
k = 3 retrieved chunks
sequence length = 4096
learning rate = 1e-4
epochs = 3
batch size = 2
gradient accumulation = 8
LoRA r = 16
LoRA alpha = 32
LoRA dropout = 0.05
target modules = q/k/v/o projections + MLP projections
```

### Fine-Tuned SmolLM3

Base model:

```text
HuggingFaceTB/SmolLM3-3B
```

Adapter path:

```text
models/smollm3-rag-lora-k3-seq4096-lr1e4
```

Training settings match Qwen where possible:

```text
k = 3 retrieved chunks
sequence length = 4096
learning rate = 1e-4
epochs = 3
batch size = 2
gradient accumulation = 8
LoRA r = 16
LoRA alpha = 32
LoRA dropout = 0.05
target modules = q/k/v/o projections + MLP projections
```

SmolLM3 has its own training script because its chat template is different. The script disables extended thinking and derives the assistant completion marker from the tokenizer instead of hardcoding Qwen's `<|im_start|>assistant\n`.

### Small-Model Comparison

The 300-query comparison script evaluates:

```text
Qwen/Qwen3-1.7B
HuggingFaceTB/SmolLM2-1.7B-Instruct
Qwen2.5-3B LoRA adapter
```

Each model is tested in three settings:

```text
baseline: question only
RAG: top-k retrieved chunks from test_rankings.jsonl
oracle: gold chunk forced into Passage 1
```

Oracle follows the expected rule: if the gold chunk is already in top-k, move it to the top; otherwise replace the kth chunk with the gold chunk at rank 1.

## Metrics

The project reports:

```text
EM
SubEM
METEOR
BERTScore precision / recall / F1
LLM judge score
```

The judge model is:

```text
mistralai/Mistral-7B-Instruct-v0.3
```

A note on EM: exact match is intentionally harsh. If the gold answer is `Paris` and the model says `The answer is Paris.`, EM is 0 but SubEM is 1. This is useful because it tells us not only whether the model knows the answer, but whether it follows the short-answer format.

## Setup Notes

The cluster jobs are designed for a cached/offline Hugging Face workflow. Cache the assets on a login node first, then submit Slurm jobs.

Install/verify the current environment:

```bash
pip install -r requirements.txt
python -c "import torch, transformers, accelerate, trl, peft; print(torch.__version__, transformers.__version__, accelerate.__version__, trl.__version__, peft.__version__)"
```

The current stack expects:

```text
torch 2.6.0
transformers 4.57.6
accelerate 1.10.1
trl 0.15.2
```

Torch 2.6 is needed because BERTScore's DeBERTa model uses a legacy `.bin` checkpoint, and newer Transformers blocks `torch.load` on older Torch versions because of CVE-2025-32434.

## Common Commands

Cache SmolLM3:

```bash
python download_smollm3.py
```

Train Qwen2.5 LoRA:

```bash
mkdir -p logs
sbatch scripts/run_training.sh
```

Train SmolLM3 LoRA:

```bash
mkdir -p logs
sbatch scripts/run_smollm3_training.sh
```

Run the 300-query comparison:

```bash
python compare_small_models.py
```

Evaluate SmolLM3 on the same saved 300-query set:

```bash
sbatch scripts/run_smollm3_eval.sh
```

Evaluate SmolLM3 on the full test set:

```bash
SOURCE_RUN_DIR= sbatch scripts/run_smollm3_eval.sh
```

## Output Layout

Runs write to `outputs/`, usually with timestamped names. A typical run contains:

```text
answers/*.jsonl       # one file per model
judge/*.jsonl         # raw Mistral judge decisions
comparison.csv        # flat metric table
comparison.json       # full report with config, paths, metrics, and samples
```

The answer JSONL files are useful for debugging behavior directly. For example, high SubEM but low EM usually means the model contains the answer but wraps it in a sentence.

## What We Learned So Far

Qwen2.5 adapts cleanly to the short-answer RAG format.

SmolLM3 can be trained with the same supervision, but its chat template includes a thinking scaffold. Even with short-answer targets, it can still be more verbose at inference time, so its evaluation path uses a stricter short-answer prompt and strips leftover `<think>...</think>` text before scoring.

That distinction is important: we are not only measuring whether a model can find the answer in the retrieved context. We are also measuring whether it can obey the answer format required by short-answer QA.
