"""Extract baseline token embeddings from a training checkpoint.

Phase 0 utility:
- Loads a PyTorch checkpoint.
- Finds token embedding matrix key heuristically.
- Exports embedding matrix (.npy) and token list (.txt/.json) from SentencePiece model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm  # type: ignore[import-untyped]
import torch

CANDIDATE_KEYS = [
    "tok_embeddings.weight",
    "model.tok_embeddings.weight",
    "transformer.tok_embeddings.weight",
    "embedding.weight",
    "model.embedding.weight",
    "wte.weight",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract token embeddings from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--sp-model", type=str, required=True, help="Path to SentencePiece .model")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to store outputs")
    parser.add_argument("--embedding-key", type=str, default="", help="Optional explicit state-dict key")
    return parser


def _get_state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
        return payload["model_state_dict"]
    tensor_items = {
        key: value
        for key, value in payload.items()
        if isinstance(value, torch.Tensor)
    }
    if tensor_items:
        return tensor_items
    raise ValueError("Could not find model state dict in checkpoint payload")


def _resolve_embedding_key(state_dict: dict[str, torch.Tensor], preferred: str) -> str:
    if preferred:
        if preferred not in state_dict:
            raise KeyError(f"Provided embedding key not found: {preferred}")
        return preferred

    for key in CANDIDATE_KEYS:
        if key in state_dict and state_dict[key].ndim == 2:
            return key

    for key, value in state_dict.items():
        if value.ndim == 2 and "embed" in key.lower():
            return key

    raise KeyError(
        "Unable to find embedding matrix key automatically. "
        "Use --embedding-key explicitly."
    )


def main() -> None:
    args = build_arg_parser().parse_args()

    ckpt_path = Path(args.checkpoint)
    sp_path = Path(args.sp_model)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("Unsupported checkpoint format: expected dict payload")

    state_dict = _get_state_dict(payload)
    embedding_key = _resolve_embedding_key(state_dict, args.embedding_key)
    embedding_matrix = state_dict[embedding_key].detach().cpu().float().numpy()

    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_path))

    vocab_size = sp.get_piece_size()
    if embedding_matrix.shape[0] != vocab_size:
        print(
            "[warn] Embedding rows != tokenizer vocab size: "
            f"{embedding_matrix.shape[0]} vs {vocab_size}"
        )

    tokens = [sp.id_to_piece(i) for i in range(min(vocab_size, embedding_matrix.shape[0]))]

    npy_path = out_dir / "token_embeddings.npy"
    txt_path = out_dir / "tokens.txt"
    meta_path = out_dir / "metadata.json"

    np.save(npy_path, embedding_matrix[: len(tokens)])
    txt_path.write_text("\n".join(tokens), encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "checkpoint": str(ckpt_path),
                "sp_model": str(sp_path),
                "embedding_key": embedding_key,
                "num_tokens": len(tokens),
                "embedding_dim": int(embedding_matrix.shape[1]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[ok] Saved embeddings: {npy_path}")
    print(f"[ok] Saved tokens: {txt_path}")
    print(f"[ok] Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
