#!/usr/bin/env python3
"""Test CAMeL Tools morphological analysis pipeline (quality + speed)."""

from __future__ import annotations

import json
import time
from pathlib import Path


def test_camel() -> None:
    from camel_tools.morphology.analyzer import Analyzer  # type: ignore[import-untyped]
    from camel_tools.morphology.database import MorphologyDB  # type: ignore[import-untyped]

    print("Loading CAMeL morphology database...")
    db = MorphologyDB.builtin_db()
    analyzer = Analyzer(db)

    test_words = [
        "كتب",
        "كاتب",
        "مكتوب",
        "كتاب",
        "مكتبة",
        "علم",
        "عالم",
        "تعليم",
        "معلمة",
        "درس",
        "دارس",
        "مدرسة",
        "دراسة",
        "استخرج",
        "يتعلمون",
        "المدرسة",
        "القاهرة",
        "محمد",
        "كمبيوتر",
        "في",
        "من",
        "على",
    ]

    print(f"\nTesting {len(test_words)} words...\n")

    results: list[dict[str, object]] = []
    total_time = 0.0

    for word in test_words:
        start = time.time()
        analyses = analyzer.analyze(word)
        elapsed = time.time() - start
        total_time += elapsed

        if analyses:
            best = analyses[0]
            root = str(best.get("root", "N/A"))
            pattern = str(best.get("pattern", "N/A"))
            pos = str(best.get("pos", "N/A"))

            result = {
                "word": word,
                "root": root,
                "pattern": pattern,
                "pos": pos,
                "num_analyses": len(analyses),
                "time_ms": round(elapsed * 1000, 1),
            }
            print(
                f"  {word:15} -> root={root:8} pattern={pattern:10} "
                f"pos={pos:8} ({len(analyses)} analyses, {elapsed * 1000:.1f}ms)"
            )
        else:
            result = {
                "word": word,
                "root": "NONE",
                "pattern": "NONE",
                "pos": "NONE",
                "num_analyses": 0,
                "time_ms": round(elapsed * 1000, 1),
            }
            print(f"  {word:15} -> NO ANALYSIS! ({elapsed * 1000:.1f}ms)")

        results.append(result)

    analyzed = sum(1 for item in results if item["root"] != "NONE")
    avg_time = (total_time / len(test_words)) * 1000

    print("\nSummary:")
    print(f"  Analyzed: {analyzed}/{len(test_words)} ({analyzed / len(test_words):.0%})")
    print(f"  Avg time: {avg_time:.1f} ms/word")
    print(f"  Total time: {total_time:.2f}s")

    print("\nSpeed estimate for full corpus:")
    print(f"  238M words × {avg_time:.1f}ms = {238_000_000 * avg_time / 1000 / 3600:.1f} hours")
    print("  <- ده لو pre-compute offline!")

    output = Path("results/camel_test_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    test_camel()
