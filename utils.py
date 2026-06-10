from __future__ import annotations

import gc
import json

import torch


def free_gpu_memory() -> None:
    """Release cached GPU memory."""
    if not torch.cuda.is_available():
        return
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def load_rankings(path: str) -> dict:
    """Load pre-computed chunk rankings from a JSONL file."""
    rankings: dict = {}
    with open(path, "r") as f:
        for line in f:
            entry = json.loads(line)
            rankings.update(entry)
    return rankings
