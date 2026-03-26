"""Probing utilities for AraStudy models."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


@torch.no_grad()
def extract_embeddings(model: Any, words: list[str], tokenizer: Any) -> dict[str, np.ndarray]:
    """Extract word embeddings (mean-pooled for multi-token words)."""
    device = next(model.parameters()).device
    model.eval()
    out: dict[str, np.ndarray] = {}

    for word in words:
        ids = tokenizer.encode(word, out_type=int)
        if not ids:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        hidden_states, final_hidden = model.forward_hidden(x)
        pooled = final_hidden[0].mean(dim=0)
        out[word] = pooled.detach().cpu().numpy().astype(np.float32)
    return out


@torch.no_grad()
def layer_wise_probing(model: Any, word_pairs: dict[str, list[dict[str, str]]], tokenizer: Any) -> dict[str, Any]:
    """M6: compute RCS per transformer layer."""
    device = next(model.parameters()).device
    model.eval()

    same_pairs = word_pairs.get("same_root", [])
    diff_pairs = word_pairs.get("diff_root", [])
    unique_words = sorted({p["w1"] for p in same_pairs + diff_pairs} | {p["w2"] for p in same_pairs + diff_pairs})

    per_word_layer: dict[str, list[np.ndarray]] = {}
    for word in unique_words:
        ids = tokenizer.encode(word, out_type=int)
        if not ids:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        hidden_states, _ = model.forward_hidden(x)
        per_word_layer[word] = [h[0].mean(dim=0).detach().cpu().numpy().astype(np.float32) for h in hidden_states]

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    n_layers = len(next(iter(per_word_layer.values()))) if per_word_layer else 0
    layer_scores: list[dict[str, float]] = []

    for layer_idx in range(n_layers):
        intra = [
            cos(per_word_layer[p["w1"]][layer_idx], per_word_layer[p["w2"]][layer_idx])
            for p in same_pairs
            if p["w1"] in per_word_layer and p["w2"] in per_word_layer
        ]
        inter = [
            cos(per_word_layer[p["w1"]][layer_idx], per_word_layer[p["w2"]][layer_idx])
            for p in diff_pairs
            if p["w1"] in per_word_layer and p["w2"] in per_word_layer
        ]
        intra_mean = float(np.mean(intra)) if intra else 0.0
        inter_mean = float(np.mean(inter)) if inter else 0.0
        layer_scores.append(
            {
                "layer": float(layer_idx),
                "intra_similarity": intra_mean,
                "inter_similarity": inter_mean,
                "rcs": intra_mean - inter_mean,
            }
        )

    return {
        "layers": layer_scores,
        "best_layer": max(layer_scores, key=lambda x: x["rcs"]) if layer_scores else None,
    }

