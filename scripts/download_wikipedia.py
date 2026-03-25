#!/usr/bin/env python3
"""Download + clean Arabic Wikipedia from Hugging Face datasets.

Outputs:
- data/processed/wikipedia_ar.txt
- data/processed/corpus_stats.json
- data/processed/train.txt
- data/processed/valid.txt
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path


def normalize_arabic(text: str) -> str:
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"[\u064B-\u0652]", "", text)
    text = re.sub(r"\u0640", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_valid_line(line: str) -> bool:
    line = line.strip()
    if len(line) < 20:
        return False

    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", line))
    total_chars = len(line.replace(" ", ""))
    if total_chars == 0:
        return False
    if arabic_chars / total_chars < 0.5:
        return False

    if len(line.split()) < 5:
        return False

    if line.startswith(("*", "#", "|", "{", "}")):
        return False
    if line.startswith(("==", "[[", "]]")):
        return False

    return True


def split_corpus(
    input_file: str,
    train_file: str,
    valid_file: str,
    valid_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[int, int]:
    rng = random.Random(seed)

    inp = Path(input_file)
    train_path = Path(train_file)
    valid_path = Path(valid_file)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    valid_path.parent.mkdir(parents=True, exist_ok=True)

    train_count = 0
    valid_count = 0

    with open(inp, encoding="utf-8") as in_handle, open(
        train_path, "w", encoding="utf-8"
    ) as train_handle, open(valid_path, "w", encoding="utf-8") as valid_handle:
        for line in in_handle:
            if not line.strip():
                continue
            if rng.random() < valid_ratio:
                valid_handle.write(line)
                valid_count += 1
            else:
                train_handle.write(line)
                train_count += 1

    print(f"Train: {train_count:,} lines -> {train_file}")
    print(f"Valid: {valid_count:,} lines -> {valid_file}")
    return train_count, valid_count


def main() -> None:
    from datasets import load_dataset

    output = Path("data/processed/wikipedia_ar.txt")
    output.parent.mkdir(parents=True, exist_ok=True)

    print("Loading Arabic Wikipedia from HuggingFace...")
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.ar",
        split="train",
    )

    total_articles = len(dataset)
    print(f"Total articles: {total_articles:,}")

    stats: dict[str, int] = {
        "total_articles": total_articles,
        "total_lines": 0,
        "kept_lines": 0,
        "total_words": 0,
    }

    with open(output, "w", encoding="utf-8") as handle:
        for index, article in enumerate(dataset):
            if index % 50_000 == 0:
                print(f"  Processing article {index:,}/{total_articles:,}...")

            text = str(article.get("text", ""))
            if not text:
                continue

            for line in text.split("\n"):
                stats["total_lines"] += 1
                line = normalize_arabic(line)

                if not is_valid_line(line):
                    continue

                handle.write(line + "\n")
                stats["kept_lines"] += 1
                stats["total_words"] += len(line.split())

    print("\nDone!")
    print(f"  Articles: {stats['total_articles']:,}")
    print(f"  Total lines: {stats['total_lines']:,}")
    print(f"  Kept lines: {stats['kept_lines']:,}")
    print(f"  Total words: {stats['total_words']:,}")
    print(f"  Output: {output}")
    print(f"  Size: {output.stat().st_size / 1024 / 1024:.1f} MB")

    stats_file = Path("data/processed/corpus_stats.json")
    stats_file.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Stats: {stats_file}")

    split_corpus(
        "data/processed/wikipedia_ar.txt",
        "data/processed/train.txt",
        "data/processed/valid.txt",
        valid_ratio=0.05,
        seed=42,
    )


if __name__ == "__main__":
    main()
