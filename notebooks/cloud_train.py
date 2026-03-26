#!/usr/bin/env python3
'''
AraStudy Cloud Training Script.
Works on Kaggle (T4) and Colab (T4).

Usage:
  # Phase 1 (morphological data!):
  python cloud_train.py --experiment phase1

  # Phase 2 (root embedding!):
  python cloud_train.py --experiment phase2

  # Baseline (standard!):
  python cloud_train.py --experiment baseline
'''

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def setup_environment():
    """Install dependencies if not present."""
    try:
        import torch  # noqa: F401
        import sentencepiece  # noqa: F401
        import yaml  # noqa: F401
    except ImportError:
        print("Installing dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q",
            "torch", "sentencepiece", "pyyaml", "numpy",
        ], check=True)


def download_data():
    """Clone repository if code is not present and verify required files."""
    if not Path("src").exists():
        print("Cloning repository...")
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/YOUR_USERNAME/arastudy.git", "."
        ], check=True)

    required = [
        "results/tokenizers/bpe_16k.model",
        "configs/model/base_30m.yaml",
        "configs/training/default.yaml",
    ]
    for file_path in required:
        if not Path(file_path).exists():
            print(f"WARNING: {file_path} not found!")
            print("Make sure to upload data or clone repo!")
            return False
    return True


def pretokenize_if_needed(train_file, valid_file, tokenizer_model):
    """Pre-tokenize corpus if .bin files don't exist."""
    from src.data.tokenizer import pretokenize

    train_bin = "data/tokenized/train.bin"
    valid_bin = "data/tokenized/valid.bin"

    Path("data/tokenized").mkdir(parents=True, exist_ok=True)

    if not Path(train_bin).exists():
        print(f"Pre-tokenizing {train_file}...")
        meta = pretokenize(train_file, tokenizer_model, train_bin)
        print(f"  Tokens: {meta['total_tokens']:,}")

    if not Path(valid_bin).exists():
        print(f"Pre-tokenizing {valid_file}...")
        meta = pretokenize(valid_file, tokenizer_model, valid_bin)
        print(f"  Tokens: {meta['total_tokens']:,}")

    return train_bin, valid_bin


def train_baseline(total_steps=50000, batch_size=64, grad_accum_steps=1):
    """Train standard baseline model."""
    from src.models.transformer import from_config, count_parameters
    from src.training.trainer import TrainConfig, train_loop
    from src.data.tokenizer import load_tokenizer

    tokenizer_model = "results/tokenizers/bpe_16k.model"
    tokenizer = load_tokenizer(tokenizer_model)

    train_bin, valid_bin = pretokenize_if_needed(
        "data/processed/train.txt",
        "data/processed/valid.txt",
        tokenizer_model,
    )

    model = from_config("configs/model/base_30m.yaml", vocab_size=tokenizer.get_piece_size())
    print(f"Model params: {count_parameters(model)['total']:,}")

    cfg = TrainConfig(
        total_steps=total_steps,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        eval_every=2000,
        save_every=10000,
        generation_every=10000,
        early_stopping_patience=5,
    )

    result = train_loop(
        model=model,
        tokenizer=tokenizer,
        train_bin=train_bin,
        valid_bin=valid_bin,
        cfg=cfg,
        out_dir="results/cloud_baseline",
    )
    print(f"Training done! {result}")
    return result


def train_phase1(total_steps=50000, batch_size=64, grad_accum_steps=1):
    """Phase 1: Train with morphological data."""
    from src.models.transformer import from_config, count_parameters
    from src.training.trainer import TrainConfig, train_loop
    from src.data.tokenizer import load_tokenizer

    tokenizer_model = "results/tokenizers/bpe_16k.model"
    tokenizer = load_tokenizer(tokenizer_model)

    train_bin, valid_bin = pretokenize_if_needed(
        "data/processed/train_phase1.txt",
        "data/processed/valid.txt",
        tokenizer_model,
    )

    model = from_config("configs/model/base_30m.yaml", vocab_size=tokenizer.get_piece_size())
    print(f"Model params: {count_parameters(model)['total']:,}")

    cfg = TrainConfig(
        total_steps=total_steps,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        eval_every=2000,
        save_every=10000,
        generation_every=10000,
        early_stopping_patience=5,
    )

    result = train_loop(
        model=model,
        tokenizer=tokenizer,
        train_bin=train_bin,
        valid_bin=valid_bin,
        cfg=cfg,
        out_dir="results/cloud_phase1",
    )
    print(f"Training done! {result}")
    return result


def train_phase2(total_steps=50000, batch_size=64, grad_accum_steps=1):
    """Phase 2: Train with root embeddings."""
    import numpy as np
    import yaml

    from src.models.morph_transformer import RootEmbeddingTransformer
    from src.models.transformer import ModelArgs, count_parameters
    from src.training.trainer import TrainConfig, train_loop
    from src.data.tokenizer import load_tokenizer

    tokenizer_model = "results/tokenizers/bpe_16k.model"
    tokenizer = load_tokenizer(tokenizer_model)

    root_map_file = "data/morphology/token_root_map.json"
    with open(root_map_file, encoding="utf-8") as handle:
        root_map = json.load(handle)

    num_roots = root_map["stats"]["unique_roots"]
    token_to_root = root_map["token_to_root_id"]

    vocab_size = tokenizer.get_piece_size()
    root_ids_array = np.zeros(vocab_size, dtype=np.int64)
    for tok_id_str, root_id in token_to_root.items():
        tok_id = int(tok_id_str)
        if 0 <= tok_id < vocab_size:
            root_ids_array[tok_id] = root_id

    Path("data/morphology").mkdir(parents=True, exist_ok=True)
    np.save("data/morphology/root_ids_lookup.npy", root_ids_array)

    train_bin, valid_bin = pretokenize_if_needed(
        "data/processed/train_phase1.txt",
        "data/processed/valid.txt",
        tokenizer_model,
    )

    with open("configs/model/base_30m.yaml", encoding="utf-8") as handle:
        model_cfg = yaml.safe_load(handle)["model"]

    args = ModelArgs(
        dim=model_cfg.get("dim", 512),
        n_layers=model_cfg.get("n_layers", 6),
        n_heads=model_cfg.get("n_heads", 8),
        vocab_size=vocab_size,
        max_seq_len=model_cfg.get("max_seq_len", 512),
        dropout=model_cfg.get("dropout", 0.1),
    )

    model = RootEmbeddingTransformer(args, num_roots=num_roots + 1)
    print(f"Model params: {count_parameters(model)['total']:,}")
    print(f"Root embeddings: {num_roots + 1} roots!")

    cfg = TrainConfig(
        total_steps=total_steps,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        eval_every=2000,
        save_every=10000,
        generation_every=10000,
        early_stopping_patience=5,
    )

    result = train_loop(
        model=model,
        tokenizer=tokenizer,
        train_bin=train_bin,
        valid_bin=valid_bin,
        cfg=cfg,
        out_dir="results/cloud_phase2",
    )
    print(f"Training done! {result}")
    return result


def run_probing(checkpoint_dir):
    """Run probing on trained model."""
    import torch

    from src.models.transformer import from_config
    from src.evaluation.metrics import root_clustering_score, control_accuracy
    from src.data.tokenizer import load_tokenizer

    ckpt_path = Path(checkpoint_dir) / "best.pt"
    if not ckpt_path.exists():
        print(f"No checkpoint at {ckpt_path}!")
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = from_config("configs/model/base_30m.yaml", vocab_size=16000)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    tokenizer = load_tokenizer("results/tokenizers/bpe_16k.model")

    with open("data/evaluation/word_pairs.json", encoding="utf-8") as handle:
        pairs = json.load(handle)

    rcs = root_clustering_score(model, pairs, tokenizer)
    control = control_accuracy(model, pairs, tokenizer)

    results = {
        "rcs": rcs,
        "control": control,
        "selectivity": rcs["rcs"] - control["control_rcs"],
    }

    print(f"\n{'=' * 50}")
    print(f"Probing Results for {checkpoint_dir}:")
    print(f"  RCS: {rcs['rcs']:.4f}")
    print(f"  Control: {control['control_rcs']:.4f}")
    print(f"  Selectivity: {results['selectivity']:.4f}")
    print(f"{'=' * 50}")

    out = Path(checkpoint_dir) / "probing_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser(description="AraStudy Cloud Training")
    parser.add_argument("--experiment", required=True, choices=["baseline", "phase1", "phase2", "probe"])
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--probe-dir", type=str, default=None)
    args = parser.parse_args()

    setup_environment()

    if not download_data():
        return

    if args.experiment != "probe":
        effective_batch = args.batch_size * args.grad_accum_steps
        print(
            f"Training config: steps={args.steps}, batch_size={args.batch_size}, "
            f"grad_accum_steps={args.grad_accum_steps}, effective_batch={effective_batch}"
        )

    if args.experiment == "baseline":
        train_baseline(args.steps, batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps)
        run_probing("results/cloud_baseline")
    elif args.experiment == "phase1":
        train_phase1(args.steps, batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps)
        run_probing("results/cloud_phase1")
    elif args.experiment == "phase2":
        train_phase2(args.steps, batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps)
        run_probing("results/cloud_phase2")
    elif args.experiment == "probe":
        if args.probe_dir:
            run_probing(args.probe_dir)
        else:
            print("Need --probe-dir!")


if __name__ == "__main__":
    main()
