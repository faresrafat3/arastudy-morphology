"""Build Arabic morphology root database."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def build_root_database() -> list[dict[str, Any]]:
    """Bootstrap a small manual root database (extendable to 3K+ roots)."""
    return [
        {
            "root": "كتب",
            "meaning": "writing",
            "words": {
                "فعل": "كتب",
                "فاعل": "كاتب",
                "مفعول": "مكتوب",
                "فعال": "كتاب",
                "فعالة": "كتابة",
                "مفعلة": "مكتبة",
                "فعول": "كتب",
                "فعّال_جمع": "كتاب",
            },
            "frequency_rank": 1,
        },
        {
            "root": "علم",
            "meaning": "knowledge",
            "words": {
                "فعل": "علم",
                "فاعل": "عالم",
                "مفعول": "معلوم",
                "فعال": "علام",
                "تفعيل": "تعليم",
                "مفعلة": "معلمة",
                "فعول": "علوم",
            },
            "frequency_rank": 2,
        },
        {
            "root": "درس",
            "meaning": "study",
            "words": {
                "فعل": "درس",
                "فاعل": "دارس",
                "مفعول": "مدروس",
                "فعالة": "دراسة",
                "مفعلة": "مدرسة",
                "فعول": "دروس",
            },
            "frequency_rank": 3,
        },
        {
            "root": "عمل",
            "meaning": "work",
            "words": {
                "فعل": "عمل",
                "فاعل": "عامل",
                "فعيل": "عميل",
                "مفعول": "معمول",
                "أفعال": "أعمال",
                "مفعل": "معمل",
            },
            "frequency_rank": 4,
        },
        {
            "root": "قرأ",
            "meaning": "reading",
            "words": {
                "فعل": "قرأ",
                "فاعل": "قارئ",
                "فعالة": "قراءة",
                "مفعول": "مقروء",
            },
            "frequency_rank": 5,
        },
    ]


def generate_from_wikipedia(wiki_tokens_path: str, analyzer: Any) -> list[dict[str, Any]]:
    """Generate root candidates from a tokenized corpus using a morphology analyzer."""
    root_counter: Counter[str] = Counter()
    root_words: defaultdict[str, set[str]] = defaultdict(set)

    text = Path(wiki_tokens_path).read_text(encoding="utf-8")

    for line in text.splitlines():
        for word in line.split():
            analysis = analyzer.analyze(word)
            if analysis.confidence >= 0.5 and not analysis.is_function:
                root_counter[analysis.root] += 1
                root_words[analysis.root].add(word)

    top_roots: list[dict[str, Any]] = []
    for rank, (root, count) in enumerate(root_counter.most_common(3000), start=1):
        top_roots.append(
            {
                "root": root,
                "words": sorted(root_words[root])[:20],
                "frequency": count,
                "frequency_rank": rank,
            }
        )

    return top_roots


def main() -> None:
    roots = build_root_database()

    output = Path("data/morphology/root_database.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(roots, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved {len(roots)} roots to {output}")

    total_words = sum(
        len(entry.get("words", {})) if isinstance(entry.get("words"), dict) else len(entry.get("words", []))
        for entry in roots
    )
    print(f"Total words covered: {total_words}")
    print(f"Average words per root: {total_words / max(len(roots), 1):.1f}")


if __name__ == "__main__":
    main()
