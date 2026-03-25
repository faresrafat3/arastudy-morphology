#!/usr/bin/env python3
"""Build same-root and different-root word pairs from morphology guide."""

from __future__ import annotations

import itertools
import json
import random
import re
from collections import defaultdict
from pathlib import Path


ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*$")


def parse_morphology_guide(path: Path) -> list[dict[str, str]]:
    words: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            m = ROW_RE.match(line.strip())
            if not m:
                continue
            _, word, root, pattern, category = m.groups()
            words.append(
                {
                    "word": word.strip(),
                    "root": root.strip(),
                    "pattern": pattern.strip(),
                    "category": category.strip(),
                }
            )
    return words


def build_pairs(entries: list[dict[str, str]], seed: int = 42) -> dict[str, list[dict[str, str]]]:
    by_root: dict[str, list[str]] = defaultdict(list)
    for item in entries:
        root = item["root"]
        word = item["word"]
        if root in {"-", ""}:
            continue
        by_root[root].append(word)

    same_root: list[dict[str, str]] = []
    for root, words in by_root.items():
        unique_words = sorted(set(words))
        if len(unique_words) < 2:
            continue
        for w1, w2 in itertools.combinations(unique_words, 2):
            same_root.append({"w1": w1, "w2": w2, "root": root})

    roots = sorted([r for r, ws in by_root.items() if ws])
    all_words = {r: sorted(set(by_root[r])) for r in roots}

    random.seed(seed)
    diff_root: list[dict[str, str]] = []
    target = len(same_root)
    tries = 0
    while len(diff_root) < target and tries < target * 20:
        r1, r2 = random.sample(roots, 2)
        w1 = random.choice(all_words[r1])
        w2 = random.choice(all_words[r2])
        if w1 == w2:
            tries += 1
            continue
        diff_root.append({"w1": w1, "w2": w2, "root1": r1, "root2": r2})
        tries += 1

    return {"same_root": same_root, "diff_root": diff_root}


def main() -> None:
    guide_path = Path("data/morphology/morphology_guide.md")
    output_path = Path("data/evaluation/word_pairs.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries = parse_morphology_guide(guide_path)
    pairs = build_pairs(entries)
    output_path.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Guide entries parsed: {len(entries)}")
    print(f"same_root pairs: {len(pairs['same_root'])}")
    print(f"diff_root pairs: {len(pairs['diff_root'])}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
