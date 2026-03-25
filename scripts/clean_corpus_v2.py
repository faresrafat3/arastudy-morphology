#!/usr/bin/env python3
"""Second-pass cleaning for Arabic Wikipedia corpus."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
DATE_LINE_RE = re.compile(
    r"^\s*\d{1,2}\s+(يناير|فبراير|مارس|ابريل|أبريل|مايو|يونيو|يوليو|اغسطس|أغسطس|سبتمبر|اكتوبر|أكتوبر|نوفمبر|ديسمبر)\b"
)
NUMERIC_START_RE = re.compile(r"^\s*[\d٠-٩]")
LIST_LIKE_RE = re.compile(r"^\s*[-*•]\s+")
SEPARATOR_HEAVY_RE = re.compile(r"[،,:;؛|/\\-]")

COMMON_INDICATORS = {
    "في",
    "من",
    "على",
    "الى",
    "إلى",
    "عن",
    "هو",
    "هي",
    "كان",
    "كانت",
    "يكون",
    "التي",
    "الذي",
    "هذا",
    "هذه",
    "بين",
    "عند",
    "حيث",
    "بعد",
    "قبل",
    "خلال",
    "مع",
    "تم",
}

CATEGORY_PREFIXES = (
    "تصنيف:",
    "بوابة:",
    "ملف:",
    "قالب:",
    "ويكيبيديا:",
)


def _arabic_ratio(line: str) -> float:
    compact = line.replace(" ", "")
    if not compact:
        return 0.0
    return len(ARABIC_CHAR_RE.findall(compact)) / len(compact)


def _is_title_like(words: list[str], line: str) -> bool:
    if len(words) <= 8 and len(SEPARATOR_HEAVY_RE.findall(line)) <= 1:
        if words and words[0].startswith("ال") and all(len(token) <= 12 for token in words):
            return True
    return False


def _has_indicator(words: list[str]) -> bool:
    return any(token in COMMON_INDICATORS for token in words)


def is_quality_line(line: str) -> bool:
    line = line.strip()
    if not line:
        return False

    words = line.split()
    if len(words) < 10:
        return False

    if NUMERIC_START_RE.match(line):
        return False

    if DATE_LINE_RE.match(line):
        return False

    if LIST_LIKE_RE.match(line):
        return False

    lower_line = line.lower()
    if any(prefix in line for prefix in CATEGORY_PREFIXES):
        return False

    if line.startswith("قائمة") or line.startswith("قوائم"):
        return False

    if "تصنيفات" in line or "بوابات" in line:
        return False

    if "ويكيبيديا" in line or "wikipedia" in lower_line:
        return False

    if _arabic_ratio(line) < 0.7:
        return False

    if _is_title_like(words, line):
        return False

    if not _has_indicator(words):
        return False

    return True


def split_corpus(
    input_file: Path,
    train_file: Path,
    valid_file: Path,
    valid_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[int, int]:
    rng = random.Random(seed)
    train_count = 0
    valid_count = 0

    with open(input_file, encoding="utf-8") as in_handle, open(
        train_file, "w", encoding="utf-8"
    ) as train_handle, open(valid_file, "w", encoding="utf-8") as valid_handle:
        for line in in_handle:
            if not line.strip():
                continue
            if rng.random() < valid_ratio:
                valid_handle.write(line)
                valid_count += 1
            else:
                train_handle.write(line)
                train_count += 1

    return train_count, valid_count


def main() -> None:
    inp = Path("data/processed/wikipedia_ar.txt")
    out = Path("data/processed/wikipedia_ar_v2.txt")
    train = Path("data/processed/train.txt")
    valid = Path("data/processed/valid.txt")
    stats_path = Path("data/processed/corpus_stats_v2.json")

    if not inp.exists():
        raise FileNotFoundError(f"Input corpus not found: {inp}")

    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0

    with open(inp, encoding="utf-8") as in_handle, open(out, "w", encoding="utf-8") as out_handle:
        for line in in_handle:
            total += 1
            if is_quality_line(line):
                out_handle.write(line.rstrip("\n") + "\n")
                kept += 1

    removed = total - kept
    keep_ratio = (kept / total) if total else 0.0

    train_count, valid_count = split_corpus(
        input_file=out,
        train_file=train,
        valid_file=valid,
        valid_ratio=0.05,
        seed=42,
    )

    stats = {
        "input_lines": total,
        "kept_lines": kept,
        "removed_lines": removed,
        "kept_ratio": round(keep_ratio, 4),
        "output_file": str(out),
        "output_size_mb": round(out.stat().st_size / 1024 / 1024, 2),
        "train_lines": train_count,
        "valid_lines": valid_count,
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Cleaning v2 complete")
    print(f"  Input lines : {total:,}")
    print(f"  Kept lines  : {kept:,} ({keep_ratio:.2%})")
    print(f"  Removed     : {removed:,} ({(1 - keep_ratio):.2%})")
    print(f"  Output file : {out}")
    print(f"  Output size : {out.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  Train lines : {train_count:,}")
    print(f"  Valid lines : {valid_count:,}")
    print(f"  Stats JSON  : {stats_path}")


if __name__ == "__main__":
    main()
