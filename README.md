# Domain RAG Fine-Tuning Experiments

This repository contains the experiments for a short-answer RAG setup on the
`sapienzanlp-course-materials/hw-mnlp-2026` dataset.

This repository is a continuation of the retrieval work in [semantic-retrieval-transformers](https://github.com/anilegin/semantic-retrieval-transformers). In that first project, we focused on the retrieval side: fine-tuning BGE and MiniLM-style bi-encoders, mining hard negatives, and testing reranking/chunking strategies. Those retrieved rankings form the foundation for this project. Here, we take the next step: instead of only measuring whether the correct chunk is retrieved, we use those chunks inside LLM prompts and evaluate the generated short answers.

The main question is simple:

**Can a compact instruction model answer better when it is fine-tuned on retrieved passages?**

The task is deliberately strict. Given a question and retrieved passages, the model should return the shortest correct answer, not a full paragraph.

## Fine-Tuned Adapters

The LoRA adapters used in the main experiments are available on Hugging Face:

- Qwen2.5 RAG LoRA:  
  https://huggingface.co/anilegin/Qwen2.5-3B-Instruct-FT-RAG-LoRA

- SmolLM3 RAG LoRA:  
  https://huggingface.co/anilegin/SmolLM-3B-Instruct-FT-RAG-LoRA

- Qwen2.5 corrupted-context RAG LoRA:  
  https://huggingface.co/anilegin/Qwen2.5-3B-Instruct-FT-RAG-LoRA-corrupt

## Overview

We first evaluate two small instruction models under 3B parameters:

```text
Qwen/Qwen3-1.7B
HuggingFaceTB/SmolLM2-1.7B-Instruct
````

Each model is tested in three settings:

```text
baseline: question only
RAG: top-3 retrieved chunks
oracle: gold chunk forced into the first passage position
```

Then we fine-tune stronger 3B models with LoRA:

```text
Qwen/Qwen2.5-3B-Instruct
HuggingFaceTB/SmolLM3-3B
```

The training format follows the same RAG structure used at inference time:
retrieved passages, question, and the gold short answer from `row["short_answer"][0]`.

We also train a corrupted-context Qwen2.5 variant. In this run, 15% of the training examples use unsupported/worst-ranked retrieved passages and the target answer is replaced with:

```text
I don't know based on the retrieved information.
```

This did not improve the main RAG score, but it helped reduce unsupported guessing in cases where the model had weak or missing evidence.

## Prompt Format

The main RAG prompt is defined in `data_utils.py`:

```text
Use the following retrieved information to answer the question.

Retrieved information:
[Passage 1 | chunk_index=...]
...

Question:
...

Answer with the shortest correct answer supported by the retrieved information.
```

For SmolLM3 evaluation, we add a stricter inference-only instruction because it tends to produce longer answers:

```text
Important: output only the shortest answer string.
Do not write a full sentence, explanation, prefix, citation, or reasoning.
```

This extra instruction is only used during SmolLM3 evaluation.

## Repository Layout

```text
.
|-- config.py                       # Qwen2.5 config and LoRA defaults
|-- smollm3_config.py               # SmolLM3-specific config
|-- data_utils.py                   # Ranking loading and RAG prompt formatting
|-- model_utils.py                  # Model/tokenizer loading helpers
|-- train.py                        # Qwen2.5 LoRA training
|-- train_smollm3.py                # SmolLM3 LoRA training
|-- evaluate.py                     # Generation, metrics, and judge utilities
|-- compare_small_models.py         # 300-sample model comparison
|-- evaluate_smollm3_lora.py        # SmolLM3-LoRA evaluation
|-- download_datasets.py            # Cache dataset/models/assets
|-- training_data/
|   |-- training_rankings.jsonl
|   |-- test_rankings.jsonl
|   `-- blind_rankings.jsonl
|-- scripts/
|   |-- run_training.sh
|   |-- run_smollm3_training.sh
|   |-- run_smollm3_eval.sh
|   `-- llm-judge-outputs.sh
|-- models/                         # Local checkpoints/adapters, ignored by git
`-- outputs/                        # Saved generations and metrics, ignored by git
```

## Metrics

The evaluation reports:

```text
EM
SubEM
METEOR
BERTScore precision / recall / F1
LLM judge score
```

Exact Match is intentionally strict. For example, if the gold answer is `Paris` and the model says `The answer is Paris.`, EM is 0 but SubEM is 1. This helps separate knowing the answer from following the short-answer format.

The final LLM judge is:

```text
mistralai/Mistral-7B-Instruct-v0.3
```

Earlier runs also tested Phi-3.5, but Mistral followed the binary judging format more reliably.

## Setup

The code was mainly used on a cluster with cached Hugging Face assets. Cache the dataset and models first, then submit the training/evaluation jobs.

Install dependencies:

```bash
pip install -r requirements.txt
```

Quick environment check:

```bash
python -c "import torch, transformers, accelerate, trl, peft; print(torch.__version__, transformers.__version__, accelerate.__version__, trl.__version__, peft.__version__)"
```

Main stack used:

```text
torch 2.6.0
transformers 4.57.6
accelerate 1.10.1
trl 0.15.2
```

## Common Commands

Cache datasets and assets:

```bash
python download_datasets.py
```

Train Qwen2.5 LoRA:

```bash
sbatch scripts/run_training.sh
```

Train SmolLM3 LoRA:

```bash
sbatch scripts/run_smollm3_training.sh
```

Evaluate SmolLM3 LoRA:

```bash
sbatch scripts/run_smollm3_eval.sh
```

Re-score saved outputs with the LLM judge:

```bash
sbatch scripts/llm-judge-outputs.sh
```

## Outputs

Runs are saved under `outputs/`, usually with timestamped folders. A typical run contains:

```text
answers/*.jsonl       # generated answers
judge/*.jsonl         # raw LLM judge decisions
comparison.csv        # flat metric table
comparison.json       # full metrics and config
```

The answer files are useful for inspecting model behavior directly. In particular, high SubEM with low EM usually means the model included the correct answer but wrapped it in a longer sentence.

## Main Takeaways

Qwen2.5 adapts well to the short-answer RAG format. LoRA fine-tuning improves both answer quality and format following.

SmolLM3 can use the retrieved information, but it is harder to make it consistently produce only the short answer. This often gives high SubEM but weak EM.

The corrupted-context Qwen2.5 run is useful for studying abstention. It does not improve the main RAG score, but it can reduce hallucinated answers when the model has little or no useful evidence.

Overall, the experiments show that short-answer RAG is not only about retrieving the right chunk. The model also needs to learn when to answer, when to abstain, and how to keep the output in the required format.


