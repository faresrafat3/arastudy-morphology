"""Tokenizer utilities for AraStudy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sentencepiece as spm  # type: ignore[import-untyped]


def train_bpe_tokenizer(corpus_path: str, output_prefix: str, vocab_size: int) -> Path:
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9999,
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
        byte_fallback=True,
        normalization_rule_name="identity",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )
    return Path(f"{output_prefix}.model")


def load_tokenizer(model_path: str):
    return spm.SentencePieceProcessor(model_file=model_path)


def pretokenize(text_file: str, tokenizer_model: str, output_bin: str) -> dict[str, str | int]:
    """Tokenize text file into uint16 binary format."""
    source = Path(text_file)
    output = Path(output_bin)
    output.parent.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor(model_file=tokenizer_model)

    chunk_size = 1_000_000
    buffered: list[int] = []
    total_tokens = 0
    total_lines = 0
    kept_lines = 0

    with open(source, encoding="utf-8") as in_handle, open(output, "wb") as out_handle:
        for line in in_handle:
            total_lines += 1
            text = line.strip()
            if not text:
                continue
            ids = sp.encode(text, out_type=int)
            if not ids:
                continue
            kept_lines += 1
            buffered.extend(ids)
            total_tokens += len(ids)

            if len(buffered) >= chunk_size:
                np.array(buffered, dtype=np.uint16).tofile(out_handle)
                buffered.clear()

        if buffered:
            np.array(buffered, dtype=np.uint16).tofile(out_handle)
            buffered.clear()

    meta: dict[str, str | int] = {
        "total_tokens": int(total_tokens),
        "total_lines": int(total_lines),
        "kept_lines": int(kept_lines),
        "dtype": "uint16",
        "source": str(source),
        "output": str(output),
        "vocab_size": int(sp.get_piece_size()),
    }
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta
