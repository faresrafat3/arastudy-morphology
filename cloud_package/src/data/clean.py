"""Clean Arabic text for language modeling."""

from __future__ import annotations

import re
from pathlib import Path


def normalize_arabic(text: str) -> str:
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[\u064B-\u0652]", "", text)
    text = re.sub(r"\u0640", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_valid_arabic_line(line: str, min_arabic_ratio: float = 0.5) -> bool:
    line = line.strip()
    if len(line) < 10:
        return False

    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", line))
    total_chars = len(line.replace(" ", ""))
    if total_chars == 0:
        return False

    if arabic_chars / total_chars < min_arabic_ratio:
        return False

    if line.startswith("*") or line.startswith("#") or line.startswith("|"):
        return False

    if len(line.split()) < 5:
        return False

    return True


def clean_corpus(input_dir: str, output_file: str) -> dict[str, int]:
    inp = Path(input_dir)
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    stats = {"total_lines": 0, "kept_lines": 0, "total_words": 0}

    with open(out, "w", encoding="utf-8") as output_handle:
        for txt_file in sorted(inp.rglob("*.txt")):
            for line in txt_file.read_text(encoding="utf-8").splitlines():
                stats["total_lines"] += 1

                if line.startswith("<") and line.endswith(">"):
                    continue

                line = normalize_arabic(line)
                if not is_valid_arabic_line(line):
                    continue

                output_handle.write(line + "\n")
                stats["kept_lines"] += 1
                stats["total_words"] += len(line.split())

    return stats
