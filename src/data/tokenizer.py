"""Tokenizer training/loading placeholders for AraStudy."""

from __future__ import annotations

from pathlib import Path


def train_bpe_tokenizer(corpus_path: str, output_prefix: str, vocab_size: int) -> Path:
    raise NotImplementedError("Implemented in Day 2 pipeline.")


def load_tokenizer(model_path: str):
    raise NotImplementedError("Implemented in Day 2 pipeline.")
