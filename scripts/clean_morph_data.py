#!/usr/bin/env python3
"""Clean morphological data from CAMeL artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path


ARABIC_RE = re.compile(r"^[\u0600-\u06FF]+$")
ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")


def is_valid_arabic_root(root: str) -> bool:
    if not root or len(root) < 2:
        return False
    if not ARABIC_RE.match(root):
        return False
    if "#" in root or "." in root:
        return False
    if len(root) > 5:
        return False
    return True


def is_valid_arabic_word(word: str) -> bool:
    if not word or len(word) < 2:
        return False
    arabic_chars = len(ARABIC_CHAR_RE.findall(word))
    return (arabic_chars / max(len(word), 1)) >= 0.8


def clean_morph_data(input_file: str, output_file: str) -> dict[str, object]:
    inp = Path(input_file)
    out = Path(output_file)

    total = 0
    kept = 0
    removed_roots: list[str] = []

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as out_handle:
        for line in inp.read_text(encoding="utf-8").splitlines():
            total += 1

            if not line.startswith("جذر "):
                continue

            parts = line.split(": ", 1)
            if len(parts) != 2:
                continue

            root = parts[0].replace("جذر ", "").strip()
            words = parts[1].strip().split()

            if not is_valid_arabic_root(root):
                removed_roots.append(root)
                continue

            clean_words = [word for word in words if is_valid_arabic_word(word)]
            if len(clean_words) < 3:
                continue

            out_handle.write(f"جذر {root}: {' '.join(clean_words[:10])}\n")
            kept += 1

    removed = total - kept
    return {
        "total": total,
        "kept": kept,
        "removed": removed,
        "removed_roots_sample": removed_roots[:20],
        "output": str(out),
    }


def main() -> None:
    stats = clean_morph_data(
        "data/morphology/root_word_lists.txt",
        "data/morphology/root_word_lists_clean.txt",
    )

    print("Cleaning morph data:")
    print(f"  Total roots: {stats['total']}")
    print(f"  Kept: {stats['kept']} ({stats['kept'] / max(stats['total'], 1):.0%})")
    print(f"  Removed: {stats['removed']}")
    print(f"  Sample removed roots: {stats['removed_roots_sample'][:10]}")

    print("\nFirst 5 clean lines:")
    out = Path(str(stats["output"]))
    for line in out.read_text(encoding="utf-8").splitlines()[:5]:
        print(f"  {line[:80]}")

    meta_path = Path("data/morphology/morph_data_clean_stats.json")
    meta_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved stats: {meta_path}")


if __name__ == "__main__":
    main()
