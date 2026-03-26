"""Training utilities for AraStudy baseline model."""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.data.tokenizer import pretokenize
from src.models.transformer import AraStudyTransformer, count_parameters, from_config


def _uses_root_embeddings(model: torch.nn.Module) -> bool:
    return hasattr(model, "root_embeddings")


def _forward_with_optional_roots(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    root_lookup: np.ndarray | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if _uses_root_embeddings(model):
        if root_lookup is None:
            raise ValueError(
                "Model uses root embeddings but root lookup was not loaded. "
                "Expected data/morphology/root_ids_lookup.npy"
            )
        x_np = x.detach().cpu().numpy()
        root_ids_np = root_lookup[x_np]
        root_ids = torch.from_numpy(root_ids_np).to(device)
        return model(x, targets=y, root_ids=root_ids)
    return model(x, targets=y)


@dataclass
class TrainConfig:
    total_steps: int = 60_000
    batch_size: int = 16
    grad_accum_steps: int = 4
    learning_rate: float = 3e-4
    warmup_steps: int = 1_000
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    precision: str = "fp16"
    eval_every: int = 2_000
    save_every: int = 5_000
    generation_every: int = 5_000
    early_stopping_patience: int = 10
    block_size: int = 512
    eval_batches: int = 50


class MemmapDataLoader:
    def __init__(self, bin_file: str, block_size: int, batch_size: int):
        self.data = np.memmap(bin_file, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.batch_size = batch_size
        self.max_start = len(self.data) - block_size - 1
        if self.max_start <= 0:
            raise ValueError(f"Insufficient tokens in {bin_file} for block_size={block_size}")

    def get_batch(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        starts = np.random.randint(0, self.max_start, size=(self.batch_size,))
        x = np.stack([self.data[s : s + self.block_size] for s in starts]).astype(np.int64)
        y = np.stack([self.data[s + 1 : s + self.block_size + 1] for s in starts]).astype(np.int64)
        x_t = torch.from_numpy(x).to(device)
        y_t = torch.from_numpy(y).to(device)
        return x_t, y_t


def build_lr_scheduler(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    min_lr = base_lr * 0.1
    return min_lr + (base_lr - min_lr) * cosine


@torch.no_grad()
def evaluate(
    model: AraStudyTransformer,
    valid_loader: MemmapDataLoader,
    device: torch.device,
    eval_batches: int,
    root_lookup: np.ndarray | None = None,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    tokens_count = 0
    for _ in range(eval_batches):
        x, y = valid_loader.get_batch(device)
        _, loss = _forward_with_optional_roots(model, x, y, root_lookup, device)
        if loss is None:
            continue
        losses.append(float(loss.item()))
        tokens_count += x.numel()

    avg_loss = float(np.mean(losses)) if losses else float("inf")
    ppl = float(math.exp(avg_loss)) if avg_loss < 20 else float("inf")
    bpc = float(avg_loss / math.log(2))
    model.train()
    return {
        "loss": avg_loss,
        "ppl": ppl,
        "bpc": bpc,
        "tokens": float(tokens_count),
    }


@torch.no_grad()
def generate(
    model: AraStudyTransformer,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    temperature: float = 1.0,
    top_k: int = 40,
    device: torch.device | None = None,
) -> str:
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    ids = tokenizer.encode(prompt, out_type=int)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        x_cond = x[:, -model.max_seq_len :]
        logits, _ = model(x_cond)
        next_logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k > 0:
            top_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < top_vals[:, [-1]]] = -float("inf")
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_id), dim=1)

    model.train()
    out_ids = x[0].tolist()
    return tokenizer.decode(out_ids)


def train_loop(
    model: AraStudyTransformer,
    tokenizer: Any,
    train_bin: str,
    valid_bin: str,
    cfg: TrainConfig,
    out_dir: str,
) -> dict[str, Any]:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = MemmapDataLoader(train_bin, cfg.block_size, cfg.batch_size)
    valid_loader = MemmapDataLoader(valid_bin, cfg.block_size, cfg.batch_size)

    root_lookup: np.ndarray | None = None
    if _uses_root_embeddings(model):
        root_lookup_path = Path("data/morphology/root_ids_lookup.npy")
        if not root_lookup_path.exists():
            raise FileNotFoundError(
                "Root lookup file not found at data/morphology/root_ids_lookup.npy"
            )
        root_lookup = np.load(root_lookup_path)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
    )

    use_amp = cfg.precision.lower() == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    csv_path = output_dir / "train_log.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "train_loss", "val_loss", "ppl", "lr", "tokens_per_sec"])

    best_val = float("inf")
    patience_count = 0
    best_ckpt = output_dir / "best.pt"

    global_step = 0
    tokens_seen = 0
    started = time.time()

    while global_step < cfg.total_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(cfg.grad_accum_steps):
            x, y = train_loader.get_batch(device)
            tokens_seen += x.numel()

            autocast_device = "cuda" if device.type == "cuda" else "cpu"
            with torch.amp.autocast(device_type=autocast_device, enabled=use_amp, dtype=torch.float16):
                _, loss = _forward_with_optional_roots(model, x, y, root_lookup, device)
                if loss is None:
                    continue
                loss = loss / cfg.grad_accum_steps

            accum_loss += float(loss.item())
            scaler.scale(loss).backward()

        lr = build_lr_scheduler(global_step, cfg.total_steps, cfg.warmup_steps, cfg.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        elapsed = max(time.time() - started, 1e-6)
        tokens_per_sec = tokens_seen / elapsed

        global_step += 1

        if global_step % cfg.eval_every == 0:
            eval_stats = evaluate(model, valid_loader, device, cfg.eval_batches, root_lookup=root_lookup)
            row = [
                global_step,
                round(accum_loss, 6),
                round(eval_stats["loss"], 6),
                round(eval_stats["ppl"], 6) if math.isfinite(eval_stats["ppl"]) else "inf",
                lr,
                round(tokens_per_sec, 2),
            ]
            with open(csv_path, "a", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow(row)

            if eval_stats["loss"] < best_val:
                best_val = eval_stats["loss"]
                patience_count = 0
                torch.save({"model": model.state_dict(), "step": global_step}, best_ckpt)
            else:
                patience_count += 1

            if patience_count >= cfg.early_stopping_patience:
                break

        if global_step % cfg.save_every == 0:
            ckpt_path = output_dir / f"checkpoint_step_{global_step}.pt"
            torch.save({"model": model.state_dict(), "step": global_step}, ckpt_path)

        if global_step % cfg.generation_every == 0:
            sample = generate(model, tokenizer, "اللغة العربية", max_tokens=40, temperature=0.8, top_k=40, device=device)
            sample_path = output_dir / "samples.txt"
            with open(sample_path, "a", encoding="utf-8") as handle:
                handle.write(f"\n[step={global_step}]\n{sample}\n")

    return {
        "best_val_loss": best_val,
        "best_checkpoint": str(best_ckpt),
        "log_csv": str(csv_path),
        "steps_completed": global_step,
    }


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)

def train_experiment(config_path: str) -> None:
    from src.data.tokenizer import load_tokenizer

    exp_cfg = _load_yaml(config_path).get("experiment", {})
    model_cfg_path = exp_cfg.get("model", "configs/model/base_30m.yaml")
    train_cfg_path = exp_cfg.get("training", "configs/training/default.yaml")
    data_cfg = exp_cfg.get("data", {})
    out_dir = exp_cfg.get("out_dir", "results/checkpoints")

    train_text = data_cfg.get("train_file", "data/processed/train.txt")
    valid_text = data_cfg.get("valid_file", "data/processed/valid.txt")
    tokenizer_model = data_cfg.get("tokenizer_model", "results/tokenizers/bpe_16k.model")

    tokenized_dir = Path("data/tokenized")
    tokenized_dir.mkdir(parents=True, exist_ok=True)
    train_bin = tokenized_dir / "train.bin"
    valid_bin = tokenized_dir / "valid.bin"

    if not train_bin.exists():
        pretokenize(train_text, tokenizer_model, str(train_bin))
    if not valid_bin.exists():
        pretokenize(valid_text, tokenizer_model, str(valid_bin))

    tokenizer = load_tokenizer(tokenizer_model)
    model = from_config(model_cfg_path, vocab_size=tokenizer.get_piece_size())

    train_cfg_dict = _load_yaml(train_cfg_path).get("training", {})
    train_cfg = TrainConfig(
        total_steps=int(train_cfg_dict.get("total_steps", 60_000)),
        batch_size=int(train_cfg_dict.get("batch_size", 16)),
        grad_accum_steps=int(train_cfg_dict.get("grad_accum_steps", 4)),
        learning_rate=float(train_cfg_dict.get("learning_rate", 3e-4)),
        warmup_steps=int(train_cfg_dict.get("warmup_steps", 1_000)),
        weight_decay=float(train_cfg_dict.get("weight_decay", 0.1)),
        max_grad_norm=float(train_cfg_dict.get("max_grad_norm", 1.0)),
        precision=str(train_cfg_dict.get("precision", "fp16")),
        eval_every=int(train_cfg_dict.get("eval_every", 2_000)),
        save_every=int(train_cfg_dict.get("save_every", 5_000)),
        generation_every=int(train_cfg_dict.get("generation_every", 5_000)),
        early_stopping_patience=int(train_cfg_dict.get("early_stopping_patience", 10)),
        block_size=int(model.max_seq_len),
    )

    run_summary = train_loop(
        model=model,
        tokenizer=tokenizer,
        train_bin=str(train_bin),
        valid_bin=str(valid_bin),
        cfg=train_cfg,
        out_dir=str(out_dir),
    )

    params = count_parameters(model)
    print("Train summary:")
    print(run_summary)
    print("Parameter count:")
    print(params)
