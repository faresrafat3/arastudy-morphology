from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tokenizer import load_tokenizer
from src.evaluation.metrics import control_accuracy, root_clustering_score
from src.models.morph_transformer import RootEmbeddingTransformer
from src.models.transformer import ModelArgs, from_config


def build_phase2_model(vocab_size: int = 16000) -> RootEmbeddingTransformer:
    with open("configs/model/base_30m.yaml", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)["model"]
    with open("data/morphology/token_root_map.json", encoding="utf-8") as handle:
        root_map = json.load(handle)
    num_roots = int(root_map.get("stats", {}).get("unique_roots", 0))
    args = ModelArgs(
        dim=int(cfg.get("dim", 512)),
        n_layers=int(cfg.get("n_layers", 6)),
        n_heads=int(cfg.get("n_heads", 8)),
        vocab_size=vocab_size,
        max_seq_len=int(cfg.get("max_seq_len", 512)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    return RootEmbeddingTransformer(args, num_roots=num_roots + 1)


def _load_checkpoint_model(kind: str, ckpt_path: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if kind == "phase2":
        model = build_phase2_model(vocab_size=16000)
    elif kind == "baseline10m":
        model = from_config("configs/model/small_10m.yaml", vocab_size=16000)
    else:
        model = from_config("configs/model/base_30m.yaml", vocab_size=16000)

    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def run_layerwise(tokenizer) -> dict[str, list[dict[str, float | int]]]:
    with open("data/evaluation/word_pairs.json", encoding="utf-8") as handle:
        pairs = json.load(handle)

    test_words = sorted(
        set([p["w1"] for p in pairs["same_root"][:50]] + [p["w2"] for p in pairs["same_root"][:50]])
    )[:30]
    token_ids_list = [tokenizer.encode(word, out_type=int) for word in test_words]
    max_len = max(len(ids) for ids in token_ids_list)
    padded = [ids + [0] * (max_len - len(ids)) for ids in token_ids_list]
    input_tensor = torch.tensor(padded, dtype=torch.long)

    models = [
        ("Baseline 28M", "results/checkpoints/best.pt", "baseline28m"),
        ("Phase 1", "results/checkpoints_phase1/best.pt", "phase1"),
        ("Phase 2", "results/checkpoints_phase2/best.pt", "phase2"),
        ("Baseline 10M", "results/checkpoints_10m/best.pt", "baseline10m"),
    ]

    all_results: dict[str, list[dict[str, float | int]]] = {}
    for name, ckpt_path, kind in models:
        model = _load_checkpoint_model(kind, ckpt_path)
        with torch.no_grad():
            hidden_states, _ = model.forward_hidden(input_tensor)

        layer_results: list[dict[str, float | int]] = []
        for layer_idx, hidden in enumerate(hidden_states):
            word_vecs = {}
            for i, word in enumerate(test_words):
                n_tokens = len(token_ids_list[i])
                word_vecs[word] = hidden[i, :n_tokens, :].mean(dim=0).cpu().numpy()

            intra = [
                _cos_sim(word_vecs[p["w1"]], word_vecs[p["w2"]])
                for p in pairs["same_root"][:50]
                if p["w1"] in word_vecs and p["w2"] in word_vecs
            ]
            inter = [
                _cos_sim(word_vecs[p["w1"]], word_vecs[p["w2"]])
                for p in pairs["diff_root"][:50]
                if p["w1"] in word_vecs and p["w2"] in word_vecs
            ]

            intra_m = float(np.mean(intra)) if intra else 0.0
            inter_m = float(np.mean(inter)) if inter else 0.0
            layer_results.append(
                {
                    "layer": layer_idx + 1,
                    "intra": round(intra_m, 4),
                    "inter": round(inter_m, 4),
                    "rcs": round(intra_m - inter_m, 4),
                }
            )

        all_results[name] = layer_results
    return all_results


def run_10m_probing(tokenizer) -> dict[str, float | str]:
    ckpt_files = glob.glob("results/*10m*/best.pt") + glob.glob("results/*10M*/best.pt")
    if not ckpt_files:
        ckpt_files = ["results/checkpoints_10m/best.pt"]
    ckpt_path = ckpt_files[0]

    model = _load_checkpoint_model("baseline10m", ckpt_path)
    with open("data/evaluation/word_pairs.json", encoding="utf-8") as handle:
        pairs = json.load(handle)
    rcs = root_clustering_score(model, pairs, tokenizer)
    control = control_accuracy(model, pairs, tokenizer)
    return {
        "checkpoint": ckpt_path,
        "rcs": float(rcs["rcs"]),
        "control_rcs": float(control["control_rcs"]),
        "selectivity": float(rcs["rcs"] - control["control_rcs"]),
    }


def run_generation(tokenizer) -> dict[str, list[dict[str, str]]]:
    prompts = ["اللغة العربية", "في يوم من الايام", "العلم نور", "المدرسة هي", "كتب الطالب"]
    models = [
        ("Baseline", "results/checkpoints/best.pt", "baseline"),
        ("Phase 1", "results/checkpoints_phase1/best.pt", "phase1"),
        ("Phase 2", "results/checkpoints_phase2/best.pt", "phase2"),
    ]
    outputs: dict[str, list[dict[str, str]]] = {}

    root_lookup = np.load("data/morphology/root_ids_lookup.npy")
    for name, ckpt_path, kind in models:
        model = _load_checkpoint_model(kind, ckpt_path)
        outputs[name] = []
        for prompt in prompts:
            ids = tokenizer.encode(prompt, out_type=int) or [2]
            x = torch.tensor([ids], dtype=torch.long)

            with torch.no_grad():
                for _ in range(50):
                    x_cond = x[:, -512:]
                    if kind == "phase2":
                        x_np = x_cond.detach().cpu().numpy()
                        root_ids = torch.from_numpy(root_lookup[x_np]).to(x_cond.device)
                        logits, _ = model(x_cond, root_ids=root_ids)
                    else:
                        logits, _ = model(x_cond)
                    next_logits = logits[0, -1, :] / 0.8
                    probs = torch.softmax(next_logits, dim=-1)
                    next_id = torch.multinomial(probs, 1).item()
                    x = torch.cat([x, torch.tensor([[next_id]], dtype=torch.long)], dim=1)

            outputs[name].append({"prompt": prompt, "output": tokenizer.decode(x[0].tolist())})
    return outputs


def _best_metrics(path: str) -> dict[str, float | int]:
    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    best = min(rows, key=lambda row: float(row["val_loss"]))
    return {
        "best_step": int(best["step"]),
        "best_val_loss": float(best["val_loss"]),
        "tail5": rows[-5:],
    }


def main() -> None:
    out_dir = Path("results/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer("results/tokenizers/bpe_16k.model")

    layerwise = run_layerwise(tokenizer)
    (out_dir / "layer_wise_comparison.json").write_text(
        json.dumps(layerwise, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    probing10m = run_10m_probing(tokenizer)
    (out_dir / "task2_10m_probing.json").write_text(
        json.dumps(probing10m, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    generation = run_generation(tokenizer)
    (out_dir / "task3_generation_outputs.json").write_text(
        json.dumps(generation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "models": {
            "phase1": _best_metrics("results/checkpoints_phase1/train_log.csv"),
            "phase2": _best_metrics("results/checkpoints_phase2/train_log.csv"),
            "baseline10m": _best_metrics("results/checkpoints_10m/train_log.csv"),
        },
        "layerwise": layerwise,
        "probing10m": probing10m,
    }
    (out_dir / "final_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[ok] Saved:")
    print(f" - {out_dir / 'layer_wise_comparison.json'}")
    print(f" - {out_dir / 'task2_10m_probing.json'}")
    print(f" - {out_dir / 'task3_generation_outputs.json'}")
    print(f" - {out_dir / 'final_analysis_summary.json'}")


if __name__ == "__main__":
    main()
