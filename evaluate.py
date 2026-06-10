from __future__ import annotations

import json
import re
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm

from config import HF_TOKEN, MODELS_DIR, OUTPUTS_DIR, SYSTEM_PROMPT
from data_utils import build_chat_messages, format_chunks
from metrics import (
    compute_exact_match,
    compute_meteor,
    compute_sub_em,
    get_bert_scorer,
)
from model_utils import load_base_model, load_judge_model, load_tokenizer
from utils import free_gpu_memory, load_rankings


# ─── Timing helpers ────────────────────────────────────────────────────────
def _tprint(msg: str) -> None:
    """Print to terminal with immediate flush so output isn't buffered."""
    print(msg, flush=True)
    sys.stdout.flush()


def _sync(device: str) -> None:
    """Block until the device finishes pending work, so wall-clock timings
    actually reflect compute (not just kernel-launch latency)."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
        except AttributeError:
            pass  # older torch builds don't expose this


@contextmanager
def timer(label: str, device: str = "cpu"):
    """Context manager that prints elapsed wall-clock time for a code block."""
    _sync(device)
    t0 = time.perf_counter()
    _tprint(f"[TIMER] ▶ {label} ...")
    try:
        yield
    finally:
        _sync(device)
        elapsed = time.perf_counter() - t0
        _tprint(f"[TIMER] ✓ {label}: {elapsed:.2f}s ({elapsed/60:.2f} min)")


def _describe_model(model, name: str) -> None:
    """Print device + dtype + parameter count — catches silent CPU/fp32 fallbacks."""
    try:
        p = next(model.parameters())
        n_params = sum(x.numel() for x in model.parameters())
        _tprint(
            f"[INFO]  {name}: device={p.device}  dtype={p.dtype}  "
            f"params={n_params/1e9:.2f}B"
        )
    except Exception as e:
        _tprint(f"[INFO]  {name}: could not introspect ({e})")


def _describe_environment(device: str) -> None:
    """Print high-level environment info — what's available and what we picked."""
    _tprint(f"[INFO]  Selected device: {device}")
    _tprint(f"[INFO]  torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        _tprint(f"[INFO]  CUDA device name: {torch.cuda.get_device_name(0)}")
        free_b, total_b = torch.cuda.mem_get_info()
        _tprint(
            f"[INFO]  CUDA memory: {free_b/1e9:.1f} GB free / "
            f"{total_b/1e9:.1f} GB total"
        )
    _tprint(f"[INFO]  torch.backends.mps.is_available(): "
            f"{torch.backends.mps.is_available()}")


@torch.no_grad()
def _generate_from_messages(
    model,
    tokenizer,
    messages: list,
    device: str,
    max_new_tokens: int = 256,
) -> str:
    """Generate a model response based on a messages list."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


@torch.no_grad()
def _generate_from_prompts_batch(
    model,
    tokenizer,
    prompts: list[str],
    device: str,
    max_new_tokens: int = 256,
) -> list[str]:
    # 1. Temporarily switch to left-padding for generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 2. Tokenize with padding
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    # 3. Generate
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    # 4. Extract only the generated tokens and decode
    responses = []
    input_len = inputs["input_ids"].shape[-1]
    for i in range(len(prompts)):
        gen_ids = output_ids[i][input_len:]
        responses.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
    # 5. Restore the original padding side
    tokenizer.padding_side = original_padding_side
    return responses


@torch.no_grad()
def _generate_from_messages_batch(
    model,
    tokenizer,
    batch_messages: list[list[dict]],
    device: str,
    max_new_tokens: int = 256,
) -> list[str]:
    # Render prompts with the chat template
    prompts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in batch_messages
    ]
    return _generate_from_prompts_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
        max_new_tokens=max_new_tokens,
    )


def compute_responses(
    model,
    tokenizer,
    test_data: list,
    rankings: dict | None,
    k: int,
    device: str,
    split: str = "test",
    save_path: Path | None = None,
    batch_size: int = 16,
) -> list:
    """
    Run the model over dataset and collect baseline, RAG, and oracle
    responses for each prompt in batches.
    """
    model.eval()
    all_results = []

    # Per-phase wall-clock accumulators so we can see where time actually goes.
    phase_time = {"baseline_gen": 0.0, "rag_gen": 0.0, "oracle_gen": 0.0,
                  "setup": 0.0, "format_chunks": 0.0}

    overall_start = time.perf_counter()
    num_batches = (len(test_data) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(0, len(test_data), batch_size), desc="Generating responses", total=num_batches):
        batch_rows = test_data[batch_idx : batch_idx + batch_size]
        
        batch_baselines = []
        batch_rags = []
        batch_oracles = []
        batch_meta = []

        # Compute the prompts for each query and each type of query
        for row in batch_rows:
            t_setup_start = time.perf_counter()
            query_id     = str(row["query_id"])
            query        = row["query"]
            chunks       = row["candidate_chunks"]
            # In blind mode, we don't have the answer position, so we set it to -1
            if split == "blind":
                correct_idx = -1
                ground_truth = None
            else:
                ground_truth = row["short_answer"][0]
                correct_idx  = row.get("answer_pos", -1)

            if rankings and query_id in rankings:
                ranking = list(rankings[query_id])
            else:
                _tprint(f"[WARNING] No ranking found for query_id {query_id}. Using random ranking")
                ranking = list(range(len(chunks)))
                random.shuffle(ranking)

            correct_in_topk = ranking.index(correct_idx) if correct_idx in ranking[:k] else -1

            batch_meta.append({
                "query_id":        query_id,
                "query":           query,
                "ground_truth":    ground_truth,
                "ranking":         ranking,
                "correct_in_topk": correct_in_topk,
            })

            # ---- baseline ----
            baseline_msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": query},
            ]
            batch_baselines.append(baseline_msgs)
            phase_time["setup"] += time.perf_counter() - t_setup_start

            # ---- RAG ----
            t_fmt = time.perf_counter()
            rag_context = format_chunks(chunks, ranking, k=k)
            phase_time["format_chunks"] += time.perf_counter() - t_fmt

            t_setup_start = time.perf_counter()
            rag_msgs = build_chat_messages(query, rag_context, answer=None)
            batch_rags.append(rag_msgs)
            phase_time["setup"] += time.perf_counter() - t_setup_start

            # ---- oracle (force correct chunk to position 0) ----
            if not split == "blind":
                oracle_ranking = list(ranking)
                if correct_idx in oracle_ranking:
                    # remove correct index so it doesn't appear double
                    oracle_ranking.remove(correct_idx)
                # inserting is always okay, since format_chunks handles slicing to top-k
                oracle_ranking.insert(0, correct_idx)

                t_fmt = time.perf_counter()
                oracle_context = format_chunks(chunks, oracle_ranking, k=k)
                phase_time["format_chunks"] += time.perf_counter() - t_fmt

                t_setup_start = time.perf_counter()
                oracle_msgs = build_chat_messages(query, oracle_context, answer=None)
                batch_oracles.append(oracle_msgs)
                phase_time["setup"] += time.perf_counter() - t_setup_start

        # Perform batched generations
        # Baseline
        _sync(device)
        t0 = time.perf_counter()
        baseline_responses = _generate_from_messages_batch(model, tokenizer, batch_baselines, device)
        _sync(device)
        phase_time["baseline_gen"] += time.perf_counter() - t0

        # RAG
        _sync(device)
        t0 = time.perf_counter()
        rag_responses = _generate_from_messages_batch(model, tokenizer, batch_rags, device)
        _sync(device)
        phase_time["rag_gen"] += time.perf_counter() - t0

        # Oracle
        if not split == "blind":
            _sync(device)
            t0 = time.perf_counter()
            oracle_responses = _generate_from_messages_batch(model, tokenizer, batch_oracles, device)
            _sync(device)
            phase_time["oracle_gen"] += time.perf_counter() - t0

        # Assemble batch results
        for i in range(len(batch_rows)):
            meta = batch_meta[i]
            if not split == "blind":
                all_results.append({
                    "query_id":        meta["query_id"],
                    "query":           meta["query"],
                    "ground_truth":    meta["ground_truth"],
                    "ranking":         meta["ranking"],
                    "correct_in_topk": meta["correct_in_topk"],
                    "baseline":        baseline_responses[i],
                    "RAG":             rag_responses[i],
                    "oracle":          oracle_responses[i],
                })
            else:
                all_results.append({
                    "query_id":        meta["query_id"],
                    "query":           meta["query"],
                    "ranking":         meta["ranking"],
                    "baseline":        baseline_responses[i],
                    "RAG":             rag_responses[i],
                })

    total_elapsed = time.perf_counter() - overall_start
    n = len(test_data)

    _tprint("")
    _tprint(f"[TIMER] compute_responses breakdown over {n} samples "
            f"(total {total_elapsed:.1f}s = {total_elapsed/60:.2f} min):")
    for phase, t in phase_time.items():
        per_sample = (t / n) * 1000 if n else 0
        _tprint(f"[TIMER]   {phase:<15}: {t:7.1f}s  ({per_sample:6.1f} ms/sample)")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            for record in all_results:
                f.write(json.dumps(record) + "\n")

    return all_results


def evaluate_responses(results: list) -> dict:
    """
    Score baseline / RAG / oracle responses with EM, SubEM, METEOR, BERTScore.
    Returns a flat dict of mean scores keyed as <metric>_<variant>.
    """
    variants = ["baseline", "RAG", "oracle"]
    accum = {
        v: {"em": [], "sub_em": [], "meteor": [], "bert_p": [], "bert_r": [], "bert_f1": []}
        for v in variants
    }

    preds = {v: [] for v in variants}
    gts   = {v: [] for v in variants}

    t_classic_start = time.perf_counter()
    for row in tqdm(results, desc="Scoring responses"):
        gt = row["ground_truth"]
        for v in variants:
            pred = row[v]
            accum[v]["em"].append(compute_exact_match(pred, gt))
            accum[v]["sub_em"].append(compute_sub_em(pred, gt))
            accum[v]["meteor"].append(compute_meteor(pred, gt))
            preds[v].append(pred)
            gts[v].append(gt)
    t_classic = time.perf_counter() - t_classic_start

    t_scorer_init_start = time.perf_counter()
    scorer = get_bert_scorer()
    t_scorer_init = time.perf_counter() - t_scorer_init_start

    bert_times = {}
    for v in tqdm(variants, desc="BERTScore"):
        t0 = time.perf_counter()
        P, R, F1 = scorer.score(preds[v], gts[v])
        bert_times[v] = time.perf_counter() - t0
        accum[v]["bert_p"]  = P.tolist()
        accum[v]["bert_r"]  = R.tolist()
        accum[v]["bert_f1"] = F1.tolist()

    _tprint("")
    _tprint(f"[TIMER] evaluate_responses breakdown:")
    _tprint(f"[TIMER]   EM/SubEM/METEOR loop  : {t_classic:7.1f}s")
    _tprint(f"[TIMER]   get_bert_scorer() init: {t_scorer_init:7.1f}s")
    for v in variants:
        _tprint(f"[TIMER]   BERTScore ({v:<8}): {bert_times[v]:7.1f}s "
                f"({len(preds[v])} pairs)")

    final = {}
    for v in variants:
        final[f"EM_{v}"]      = np.mean(accum[v]["em"])
        final[f"SubEM_{v}"]   = np.mean(accum[v]["sub_em"])
        final[f"METEOR_{v}"]  = np.mean(accum[v]["meteor"])
        final[f"BERT_P_{v}"]  = np.mean(accum[v]["bert_p"])
        final[f"BERT_R_{v}"]  = np.mean(accum[v]["bert_r"])
        final[f"BERT_F1_{v}"] = np.mean(accum[v]["bert_f1"])

    print(f"\n{'Generation Quality Metrics':^50}")
    print("-" * 50)
    for v in variants:
        n = len(accum[v]["em"])
        print(f"{v.capitalize()}:")
        print(f"  EM          : {final[f'EM_{v}']:.4f}  ({sum(accum[v]['em'])}/{n})")
        print(f"  SubEM       : {final[f'SubEM_{v}']:.4f}  ({sum(accum[v]['sub_em'])}/{n})")
        print(f"  METEOR      : {final[f'METEOR_{v}']:.4f}")
        print(f"  BERTScore F1: {final[f'BERT_F1_{v}']:.4f}")
    print("-" * 50)

    return final


def _build_judge_prompt(question: str, ground_truth: str, prediction: str) -> str:
    return (
        "You are a strict answer-evaluation judge.\n\n"
        "## Task\n"
        "Given a question, the correct short answer, and a model-generated answer, "
        "decide whether the generated answer contains the correct short answer in "
        "any valid form.\n\n"
        "## Scoring criterion\n"
        "Return 1 if the generated answer contains the correct short answer or an equivalent form.\n"
        "Return 0 if the generated answer does not contain the correct answer, contradicts it, "
        "or gives a different answer.\n\n"
        "## Examples\n\n"
        "Question: When was Rome founded?\n"
        "Correct short answer: 753 BCE\n"
        "Generated answer: Legend says Rome was founded in 753 BCE by Romulus.\n"
        "Output: 1\n\n"
        "Question: Who wrote Romeo and Juliet?\n"
        "Correct short answer: William Shakespeare\n"
        "Generated answer: The play was written by Shakespeare in the late 16th century.\n"
        "Output: 1\n\n"
        "Question: What is the capital of France?\n"
        "Correct short answer: Paris\n"
        "Generated answer: The capital city of France is Lyon.\n"
        "Output: 0\n\n"
        "Question: How many planets are in the solar system?\n"
        "Correct short answer: eight\n"
        "Generated answer: There are 8 planets revolving around the Sun.\n"
        "Output: 1\n\n"
        "## Input\n"
        f"Question: {question}\n"
        f"Correct short answer: {ground_truth}\n"
        f"Generated answer: {prediction}\n\n"
        "## Output\n"
        "Reply with a single integer only: 0 or 1.\n"
        "Output:"
    )


def _parse_judge_response(response: str) -> int:
    response = response.strip()
    if response in ("0", "1"):
        return int(response)
    if response and response[0] in ("0", "1"):
        return int(response[0])
    m = re.search(r"(?:score|answer|result|verdict|label)\s*[:\-]?\s*([01])", response, re.IGNORECASE)
    if m:
        return int(m.group(1))
    digits = set(re.findall(r"\b([01])\b", response))
    if len(digits) == 1:
        return int(digits.pop())
    print(f"[Judge Warning] Could not parse response: {repr(response[:80])}")
    return 0


def compute_llm_judge(
    results: list,
    judge_model,
    judge_tokenizer,
    num_samples: int = 200,
    seed: int = 42,
    save_path: Path | None = None,
    batch_size: int = 16,
) -> list:
    """Score RAG responses with the LLM judge on a random subset of results in batches.
    """
    random.seed(seed)
    n = min(num_samples, len(results))
    subset = random.sample(results, n)

    all_generations: list[dict] = []
    judge_times: list[float] = []
    device = next(judge_model.parameters()).device

    t_loop_start = time.perf_counter()
    num_batches = (len(subset) + batch_size - 1) // batch_size
    for idx in tqdm(range(0, len(subset), batch_size), desc="LLM Judge", total=num_batches):
        batch_items = subset[idx : idx + batch_size]
        
        batch_prompts = []
        # Build prompts for each query in the batch
        for item in batch_items:
            prompt = _build_judge_prompt(
                question=item["query"],
                ground_truth=item["ground_truth"],
                prediction=item["RAG"],
            )
            batch_prompts.append(prompt)
            
        t0 = time.perf_counter()
        _sync(device)
        # Generate the responses of the entire batch
        raw_responses = _generate_from_prompts_batch(
            model=judge_model,
            tokenizer=judge_tokenizer,
            prompts=batch_prompts,
            device=device,
            max_new_tokens=16,
        )
        _sync(device)
        batch_elapsed = time.perf_counter() - t0
        # TODO: whats the use of the for loop below?
        for _ in range(len(batch_items)):
            judge_times.append(batch_elapsed / len(batch_items))
            
        for i, item in enumerate(batch_items):
            raw_response = raw_responses[i]
            score = _parse_judge_response(raw_response)
            record = {
                "query_id":       item["query_id"],
                "query":          item["query"],
                "ground_truth":   item["ground_truth"],
                "prediction":     item["RAG"],
                "judge_prompt":   batch_prompts[i],
                "judge_response": raw_response,
                "judge_score":    score,
            }
            all_generations.append(record)
            
    t_loop = time.perf_counter() - t_loop_start

    if judge_times:
        _tprint("")
        _tprint(f"[TIMER] compute_llm_judge over {n} samples (total {t_loop:.1f}s):")
        _tprint(f"[TIMER]   mean / median / max per judge call: "
                f"{np.mean(judge_times)*1000:.0f} / "
                f"{np.median(judge_times)*1000:.0f} / "
                f"{max(judge_times)*1000:.0f} ms")

    if save_path is not None:
        with open(save_path, "w") as f:
            for record in all_generations:
                f.write(json.dumps(record) + "\n")

    return all_generations


def evaluate_llm_judge(path: Path) -> dict:
    """Compute LLM judge metrics from a saved JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    scores = [r["judge_score"] for r in records]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    return {"LLM_Judge_Score": mean_score, "LLM_Judge_N": len(scores)}


def compare_models(
    adapter_path: str = str(MODELS_DIR / "qwen-rag-lora"),
    rankings_path: str | None = None,
    llm_judge_outputs_dir: str | None = None,
    hf_token: str | None = None,
    k: int = 3,
    num_samples: int = 200,
    seed: int = 42,
    split: str = "test",
    use_llm_judge: bool = False,
    batch_size: int = 16,
    additional_models: list[str] | None = None,
):
    """
    Evaluate base and LoRA-adapted Qwen across baseline, RAG, and oracle
    prompt variants on the same test set. Does both the computation and the evaluation.
    """
    random.seed(seed)
    np.random.seed(seed)

    overall_start = time.perf_counter()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    _tprint(f"\n{'='*60}")
    _tprint("ENVIRONMENT")
    _tprint(f"{'='*60}")
    _describe_environment(device)

    with timer("Load dataset", device):
        ds = load_dataset(
            "sapienzanlp-course-materials/hw-mnlp-2026",
            token=hf_token or HF_TOKEN,
        )

        test_data = list(ds[split])
        _tprint(f"[INFO]  Evaluating on {len(test_data)} samples from split={split!r}")

    rankings = None
    if rankings_path and Path(rankings_path).exists():
        rankings = load_rankings(rankings_path)
        _tprint(f"[INFO]  Loaded rankings from {rankings_path}")

    adapter_name = Path(adapter_path).name
    run_name = f"{adapter_name}_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = OUTPUTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if use_llm_judge or llm_judge_outputs_dir is None:
        judge_out_dir = run_dir / "LLM_judge_outputs"
        judge_out_dir.mkdir(parents=True, exist_ok=True)
        base_judge_path = judge_out_dir / "base.jsonl"
        ft_judge_path = judge_out_dir / "fine_tuned.jsonl"
        additional_model_judge_paths = {m: judge_out_dir / f"{Path(m).name}.jsonl" for m in additional_models} if additional_models else {}
    else:
        base_judge_path = Path(llm_judge_outputs_dir) / "base.jsonl"
        ft_judge_path = Path(llm_judge_outputs_dir) / "fine_tuned.jsonl"
        additional_model_judge_paths = {m: Path(llm_judge_outputs_dir) / f"{Path(m).name}.jsonl" for m in additional_models} if additional_models else {}

    judged_respones_precomputed = False
    if llm_judge_outputs_dir:
        judged_respones_precomputed = True

    with timer("Load tokenizer", device):
        tokenizer = load_tokenizer()
        if additional_models:
            additional_tokenizers = {m: load_tokenizer(model_name=m) for m in additional_models}

    _tprint(f"\n{'='*60}")
    _tprint("BASE MODEL (no LoRA)")
    _tprint(f"{'='*60}")
    with timer("Load base model", device):
        base_model = load_base_model(device, use_qlora=False)
    _describe_model(base_model, "base_model")

    with timer("Base: compute_responses (generation)", device):
        base_raw = compute_responses(model=base_model, tokenizer=tokenizer, test_data=test_data, rankings=rankings, k=k, device=device, split=split, batch_size=batch_size)

    if not split == "blind":
        with timer("Base: evaluate_responses (scoring)", device):
            _tprint("\nBase model scores:")
            base_results = evaluate_responses(base_raw)
    else:
        base_results = None

    del base_model
    with timer("Free GPU memory after base model", device):
        free_gpu_memory()

    _tprint(f"\n{'='*60}")
    _tprint(f"FINE-TUNED MODEL (adapter: {adapter_path})")
    _tprint(f"{'='*60}")
    with timer("Load fine-tuned base + adapter + merge", device):
        ft_model = load_base_model(device, use_qlora=False)
        ft_model = PeftModel.from_pretrained(ft_model, adapter_path)
        ft_model = ft_model.merge_and_unload()
    _describe_model(ft_model, "ft_model")

    with timer("FT: compute_responses (generation)", device):
        ft_raw = compute_responses(model=ft_model, tokenizer=tokenizer, test_data=test_data, rankings=rankings, k=k, device=device, split=split, batch_size=batch_size)

    if not split == "blind":
        with timer("FT: evaluate_responses (scoring)", device):
            _tprint("\nFine-tuned model scores:")
            ft_results = evaluate_responses(ft_raw)
    else:
        ft_results = None

    del ft_model
    with timer("Free GPU memory after FT model", device):
        free_gpu_memory()

    additional_model_raw = {}
    additional_model_results = {}
    if additional_models:
        for model in additional_models:
            with timer("Load additional model", device):
                additional_model = load_base_model(device, use_qlora=False, model_name=model)
            with timer("Additional: compute_responses (generation)", device):
                additional_model_raw[model] = compute_responses(model=additional_model, tokenizer=additional_tokenizers[model], test_data=test_data, rankings=rankings, k=k, device=device, split=split, batch_size=batch_size)
            del additional_model
            with timer("Free GPU memory after additional model", device):
                free_gpu_memory()

            if not split == "blind":
                with timer("Additional: evaluate_responses (scoring)", device):
                    _tprint("\nAdditional model scores:")
                    additional_model_results[model] = evaluate_responses(additional_model_raw[model])

    # ── LLM-as-a-judge ────────────────────────────────────────────
    base_judge = {}
    ft_judge = {}
    additional_model_judge = {}
    if use_llm_judge:
        # only compute LLM-judge on a subset
        if num_samples < len(base_raw):
            indices = random.sample(range(len(base_raw)), k=num_samples)
            base_LLM_judge_test_data = [base_raw[i] for i in indices]
            ft_LLM_judge_test_data = [ft_raw[i] for i in indices]
            additional_model_LLM_judge_test_data = {m: [additional_model_raw[m][i] for i in indices] for m in additional_models} if additional_models else {}
        else:
            base_LLM_judge_test_data = base_raw
            ft_LLM_judge_test_data = ft_raw
            additional_model_LLM_judge_test_data = additional_model_raw

        _tprint(f"\n{'='*60}")
        _tprint("LLM JUDGE (Phi-3.5-mini-instruct)")
        _tprint(f"{'='*60}")
        with timer("Load judge model", device):
            judge_model, judge_tokenizer = load_judge_model(device)
        _describe_model(judge_model, "judge_model")

        _tprint("\nBase model — LLM Judge:")
        with timer("Base: compute_llm_judge", device):
            _ = compute_llm_judge(
                results=base_LLM_judge_test_data,
                judge_model=judge_model,
                judge_tokenizer=judge_tokenizer,
                num_samples=num_samples,
                seed=seed,
                save_path=base_judge_path,
                batch_size=batch_size,
            )
        base_judge = evaluate_llm_judge(base_judge_path)
        _tprint(f"  Score: {base_judge['LLM_Judge_Score']:.4f}  (n={base_judge['LLM_Judge_N']})")

        _tprint("\nFine-tuned model — LLM Judge:")
        with timer("FT: compute_llm_judge", device):
            _ = compute_llm_judge(
                results=ft_LLM_judge_test_data,
                judge_model=judge_model,
                judge_tokenizer=judge_tokenizer,
                num_samples=num_samples,
                seed=seed,
                save_path=ft_judge_path,
                batch_size=batch_size,
            )
        ft_judge = evaluate_llm_judge(ft_judge_path)
        _tprint(f"  Score: {ft_judge['LLM_Judge_Score']:.4f}  (n={ft_judge['LLM_Judge_N']})")

        if additional_models:
            for model in additional_models:
                with timer("Additional: compute_llm_judge", device):
                    _ = compute_llm_judge(
                        results=additional_model_LLM_judge_test_data[model],
                        judge_model=judge_model,
                        judge_tokenizer=judge_tokenizer,
                        num_samples=num_samples,
                        seed=seed,
                        save_path=additional_model_judge_paths[model],
                        batch_size=batch_size,
                    )
                additional_model_judge[model] = evaluate_llm_judge(additional_model_judge_paths[model])
                _tprint(f"  Score: {additional_model_judge[model]['LLM_Judge_Score']:.4f}  (n={additional_model_judge[model]['LLM_Judge_N']})")

        del judge_model
        with timer("Free GPU memory after judge", device):
            free_gpu_memory()

        base_results.update({f"judge_{key}": v for key, v in base_judge.items()})
        ft_results.update({f"judge_{key}": v for key, v in ft_judge.items()})

        if additional_models:
            for model in additional_models:
                additional_model_results[model].update({f"judge_{key}": v for key, v in additional_model_judge[model].items()})

    # Reload from saved outputs if you had them precomputed
    if not use_llm_judge and judged_respones_precomputed:
        base_judge = evaluate_llm_judge(base_judge_path)
        ft_judge   = evaluate_llm_judge(ft_judge_path)
        base_results.update({f"judge_{key}": v for key, v in base_judge.items()})
        ft_results.update({f"judge_{key}": v for key, v in ft_judge.items()})

        if additional_models:
            for model in additional_models:
                additional_model_judge[model] = evaluate_llm_judge(additional_model_judge_paths[model])
                additional_model_results[model].update({f"judge_{key}": v for key, v in additional_model_judge[model].items()})

    if not split == "blind":
        METRICS  = ["EM", "SubEM", "METEOR", "BERT_F1"]
        if judged_respones_precomputed or use_llm_judge:
            METRICS.append("LLM_Judge_Score")
        VARIANTS = ["baseline", "RAG", "oracle"]
        col_w = 12

        metrics = {}
        print(f"\n{'='*70}")
        print("  COMPARISON: Base  vs  Fine-tuned (LoRA)")
        print(f"{'='*70}")
        print(f"  {'Variant':<10} {'Metric':<12} {'Base':>{col_w}} {'Fine-tuned':>{col_w}} {'Δ':>{col_w}}")
        print(f"  {'-'*62}")
        for v in VARIANTS:
            first = True
            for m in METRICS:
                # LLM_Judge_Score only applies to the RAG variant
                if m == "LLM_Judge_Score" and v != "RAG":
                    continue
                key = f"judge_LLM_Judge_Score" if m == "LLM_Judge_Score" else f"{m}_{v}"
                base_val = base_results.get(key, float("nan"))
                ft_val   = ft_results.get(key, float("nan"))
                delta    = ft_val - base_val
                sign     = "+" if delta >= 0 else ""
                label    = v.capitalize() if first else ""
                print(f"  {label:<10} {m:<12} {base_val:>{col_w}.4f} {ft_val:>{col_w}.4f} {sign}{delta:>{col_w-1}.4f}")
                metrics[key] = {
                    "base": base_val,
                    "fine_tuned": ft_val,
                    "delta": delta,
                    'sign': sign,
                    "variant": v,
                    "metric": m
                }
                first = False
            print(f"  {'-'*62}")
        print(f"{'='*70}\n")
    else:
        metrics = None

    out_path = run_dir / "comparison.json"
    if not split == "blind":
        with open(out_path, "w") as f:
            json.dump(
                {
                # raw model output
                "base_raw": base_raw,
                "fine_tuned_raw": ft_raw,
                
                # evaluation of model output (+ LLM as a judge)
                "base": base_results,
                "fine_tuned": ft_results,

                # additional models
                "additional_models": additional_models,
                "additional_model_raw": additional_model_raw,
                "additional_model_results": additional_model_results,
                
                # computed metrics
                "metrics": metrics,
            },
            f,
            indent=2,
        )
        print(f"Results saved to {out_path}")

    else:
        with open(out_path, "w") as f:
            json.dump(
                {
                # raw model output
                "base_raw": base_raw,
                "fine_tuned_raw": ft_raw,
            },
            f,
            indent=2,
        )
        print(f"Results saved to {out_path}")

    total_runtime = time.perf_counter() - overall_start
    _tprint("")
    _tprint(f"{'='*60}")
    _tprint(f"TOTAL compare_models runtime: {total_runtime:.1f}s "
            f"({total_runtime/60:.2f} min)")
    _tprint(f"{'='*60}")

    return base_raw, ft_raw, base_results, ft_results


def evaluate_from_responses(
    responses_path: str | Path,
    llm_judge_dir: str | Path | None = None,
    additional_judge_models: list[str] | None = None,
    num_samples: int = 200,
    seed: int = 42,
):
    """
    Recompute metrics from a previously saved comparison.json without
    regenerating any model responses. Optionally run multiple LLMs as judges.
    """
    responses_path = Path(responses_path)
    with open(responses_path) as f:
        data = json.load(f)
    base_raw = data["base_raw"]
    ft_raw   = data["fine_tuned_raw"]

    print(f"\nLoaded {len(base_raw)} base responses and {len(ft_raw)} fine-tuned responses from {responses_path}")

    # --- classic metrics ---
    print("\nBase model scores:")
    base_results = evaluate_responses(base_raw)
    print("\nFine-tuned model scores:")
    ft_results = evaluate_responses(ft_raw)

    # --- additional models metrics (if any) ---
    additional_model_results = {}
    if additional_judge_models:
        additional_model_raw = data.get("additional_model_raw", {})
        for model_name in additional_judge_models:
            if model_name in additional_model_raw:
                print(f"\nAdditional model: {model_name} scores:")
                additional_model_results[model_name] = evaluate_responses(additional_model_raw[model_name])

    # --- LLM-as-a-judge ---
    if llm_judge_dir:
        llm_judge_dir = Path(llm_judge_dir)
        base_judge_path = llm_judge_dir / "base.jsonl"
        ft_judge_path   = llm_judge_dir / "fine_tuned.jsonl"
        additional_judge_paths = {
            m: llm_judge_dir / f"{Path(m).name}.jsonl" for m in additional_judge_models
        } if additional_judge_models else {}

        # Evaluate previously saved judge outputs
        base_judge = evaluate_llm_judge(base_judge_path)
        ft_judge   = evaluate_llm_judge(ft_judge_path)
        base_results.update({f"judge_{k}": v for k, v in base_judge.items()})
        ft_results.update({f"judge_{k}": v for k, v in ft_judge.items()})

        if additional_judge_models:
            for m in additional_judge_models:
                additional_model_results[m].update(
                    {f"judge_{k}": v for k, v in evaluate_llm_judge(additional_judge_paths[m]).items()}
                )
                print(f"Additional judge {m}: {additional_model_results[m]['judge_LLM_Judge_Score']:.4f}")

    # Save re-evaluated comparison
    out_dir = responses_path.parent / f"reeval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comparison.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "base_raw": base_raw,
                "fine_tuned_raw": ft_raw,
                "base": base_results,
                "fine_tuned": ft_results,
                "additional_model_results": additional_model_results,
            },
            f,
            indent=2,
        )
    print(f"Results saved to {out_path}")

    return base_raw, ft_raw, base_results, ft_results, additional_model_results