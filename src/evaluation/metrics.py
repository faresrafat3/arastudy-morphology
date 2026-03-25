"""Evaluation metrics for AraStudy probing and LM quality."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from src.evaluation.probing import extract_embeddings


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def root_clustering_score(model: Any, word_pairs: dict[str, list[dict[str, str]]], tokenizer: Any) -> dict[str, float]:
    """M1: Root Clustering Score = mean(intra-root sim) - mean(inter-root sim)."""
    pairs_same = word_pairs.get("same_root", [])
    pairs_diff = word_pairs.get("diff_root", [])
    unique_words = sorted({p["w1"] for p in pairs_same + pairs_diff} | {p["w2"] for p in pairs_same + pairs_diff})
    emb = extract_embeddings(model, unique_words, tokenizer)

    intra = [_cos_sim(emb[p["w1"]], emb[p["w2"]]) for p in pairs_same if p["w1"] in emb and p["w2"] in emb]
    inter = [_cos_sim(emb[p["w1"]], emb[p["w2"]]) for p in pairs_diff if p["w1"] in emb and p["w2"] in emb]

    intra_mean = float(np.mean(intra)) if intra else 0.0
    inter_mean = float(np.mean(inter)) if inter else 0.0
    return {
        "intra_similarity": intra_mean,
        "inter_similarity": inter_mean,
        "rcs": intra_mean - inter_mean,
    }


@torch.no_grad()
def perplexity(model: Any, valid_bin: str, block_size: int, batch_size: int = 8, eval_batches: int = 50) -> dict[str, float]:
    """M2: Perplexity and BPC on binary tokenized validation set."""
    arr = np.memmap(valid_bin, dtype=np.uint16, mode="r")
    max_start = len(arr) - block_size - 1
    if max_start <= 0:
        raise ValueError("Validation bin too small for given block_size")

    device = next(model.parameters()).device
    model.eval()
    losses: list[float] = []

    for _ in range(eval_batches):
        starts = np.random.randint(0, max_start, size=(batch_size,))
        x = np.stack([arr[s : s + block_size] for s in starts]).astype(np.int64)
        y = np.stack([arr[s + 1 : s + block_size + 1] for s in starts]).astype(np.int64)
        x_t = torch.from_numpy(x).to(device)
        y_t = torch.from_numpy(y).to(device)
        _, loss = model(x_t, targets=y_t)
        if loss is not None:
            losses.append(float(loss.item()))

    avg_loss = float(np.mean(losses)) if losses else float("inf")
    ppl = float(np.exp(avg_loss)) if avg_loss < 20 else float("inf")
    bpc = float(avg_loss / np.log(2))
    return {"loss": avg_loss, "ppl": ppl, "bpc": bpc}


def control_accuracy(model: Any, word_pairs_random: dict[str, list[dict[str, str]]], tokenizer: Any) -> dict[str, float]:
    """M7: Control score with shuffled labels for sanity check."""
    same_pairs = list(word_pairs_random.get("same_root", []))
    diff_pairs = list(word_pairs_random.get("diff_root", []))

    merged = same_pairs + diff_pairs
    random.shuffle(merged)
    half = len(merged) // 2
    shuffled = {"same_root": merged[:half], "diff_root": merged[half:]}

    scores = root_clustering_score(model, shuffled, tokenizer)
    return {
        "control_intra": scores["intra_similarity"],
        "control_inter": scores["inter_similarity"],
        "control_rcs": scores["rcs"],
    }
