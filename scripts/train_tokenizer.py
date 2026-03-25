#!/usr/bin/env python3
"""Train BPE tokenizers with 8K, 16K, and 32K vocab sizes."""

from __future__ import annotations

from pathlib import Path

import sentencepiece as spm  # type: ignore[import-untyped]


def train_tokenizer(
    input_file: str,
    model_prefix: str,
    vocab_size: int,
    max_sentences: int = 5_000_000,
) -> None:
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9999,
        num_threads=4,
        input_sentence_size=max_sentences,
        shuffle_input_sentence=True,
        byte_fallback=True,
        normalization_rule_name="identity",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )
    print(f"Trained {model_prefix} with vocab_size={vocab_size}")


def main() -> None:
    corpus = "data/processed/train.txt"
    out_dir = Path("results/tokenizers")
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = [8000, 16000, 32000]
    test_words = [
        "كاتب",
        "مكتوب",
        "مكتبة",
        "استخرجناها",
        "المدرسة",
        "الطالب",
        "يتعلمون",
        "كتب",
    ]

    for size in sizes:
        name = f"bpe_{size // 1000}k"
        prefix = str(out_dir / name)
        print(f"\nTraining {name}...")
        train_tokenizer(corpus, prefix, size)

        sp = spm.SentencePieceProcessor(model_file=f"{prefix}.model")
        print(f"  Vocab size: {sp.get_piece_size()}")

        for word in test_words:
            tokens = sp.encode(word, out_type=str)
            print(f"  {word} -> {tokens} ({len(tokens)} tokens)")

    print("\nDone! Tokenizers saved in results/tokenizers/")


if __name__ == "__main__":
    main()
