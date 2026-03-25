#!/usr/bin/env python3
"""Analyze BPE morpheme-boundary alignment for 100 Arabic words."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import sentencepiece as spm  # type: ignore[import-untyped]


class WordMorphemes(TypedDict):
    word: str
    morphemes: list[str]
    category: str


class DetailRow(TypedDict):
    word: str
    category: str
    morphemes: list[str]
    bpe_tokens: list[str]
    morpheme_boundaries: list[int]
    bpe_boundaries: list[int]
    respect_ratio: float
    respected: bool


class CategoryStats(TypedDict):
    total: int
    respected: int
    broken: int
    respect_rate: float


class AnalysisResult(TypedDict):
    total_words: int
    boundary_respected: int
    boundary_broken: int
    respect_rate: float
    category_breakdown: dict[str, CategoryStats]
    worst_5_breaks: list[DetailRow]
    best_5_matches: list[DetailRow]
    details: list[DetailRow]


def analyze_morpheme_boundaries(
    sp: spm.SentencePieceProcessor,
    words_with_morphemes: list[WordMorphemes],
) -> AnalysisResult:
    category_names = ["regular", "weak", "broken", "quad", "function"]
    category_breakdown: dict[str, CategoryStats] = {
        name: {"total": 0, "respected": 0, "broken": 0, "respect_rate": 0.0}
        for name in category_names
    }

    details: list[DetailRow] = []
    total_words = 0
    boundary_respected = 0

    for entry in words_with_morphemes:
        word = entry["word"]
        expected = entry["morphemes"]
        category = entry["category"]

        tokens = sp.encode(word, out_type=str)

        morpheme_boundaries: set[int] = set()
        position = 0
        for morpheme in expected[:-1]:
            position += len(str(morpheme))
            morpheme_boundaries.add(position)

        bpe_boundaries: set[int] = set()
        position = 0
        for token in tokens[:-1]:
            clean = token.replace("▁", "")
            position += len(clean)
            bpe_boundaries.add(position)

        if morpheme_boundaries:
            overlap = morpheme_boundaries & bpe_boundaries
            respect_ratio = len(overlap) / len(morpheme_boundaries)
        else:
            respect_ratio = 1.0

        respected = respect_ratio >= 0.5

        total_words += 1
        if respected:
            boundary_respected += 1

        if category not in category_breakdown:
            category_breakdown[category] = {
                "total": 0,
                "respected": 0,
                "broken": 0,
                "respect_rate": 0.0,
            }
        category_breakdown[category]["total"] += 1
        if respected:
            category_breakdown[category]["respected"] += 1
        else:
            category_breakdown[category]["broken"] += 1

        details.append(
            {
                "word": word,
                "category": category,
                "morphemes": expected,
                "bpe_tokens": tokens,
                "morpheme_boundaries": sorted(morpheme_boundaries),
                "bpe_boundaries": sorted(bpe_boundaries),
                "respect_ratio": round(respect_ratio, 3),
                "respected": respected,
            }
        )

    for category, stats in category_breakdown.items():
        if stats["total"] == 0:
            category_breakdown[category]["respect_rate"] = 0.0
        else:
            category_breakdown[category]["respect_rate"] = round(
                stats["respected"] / stats["total"], 3
            )

    sorted_for_worst = sorted(
        details,
        key=lambda row: (row["respect_ratio"], -len(row["bpe_tokens"]), row["word"]),
    )
    sorted_for_best = sorted(
        details,
        key=lambda row: (-row["respect_ratio"], len(row["bpe_tokens"]), row["word"]),
    )

    result: AnalysisResult = {
        "total_words": total_words,
        "boundary_respected": boundary_respected,
        "boundary_broken": total_words - boundary_respected,
        "respect_rate": round(boundary_respected / max(total_words, 1), 3),
        "category_breakdown": category_breakdown,
        "worst_5_breaks": sorted_for_worst[:5],
        "best_5_matches": sorted_for_best[:5],
        "details": details,
    }
    return result


def build_test_words_100() -> list[WordMorphemes]:
    return [
        {"word": "كتب", "morphemes": ["كتب"], "category": "regular"},
        {"word": "كاتب", "morphemes": ["كاتب"], "category": "regular"},
        {"word": "مكتوب", "morphemes": ["م", "كتوب"], "category": "regular"},
        {"word": "كتاب", "morphemes": ["كتاب"], "category": "regular"},
        {"word": "مكتبة", "morphemes": ["م", "كتب", "ة"], "category": "regular"},
        {"word": "علم", "morphemes": ["علم"], "category": "regular"},
        {"word": "عالم", "morphemes": ["عالم"], "category": "regular"},
        {"word": "معلوم", "morphemes": ["م", "علوم"], "category": "regular"},
        {"word": "تعليم", "morphemes": ["ت", "عليم"], "category": "regular"},
        {"word": "معلمة", "morphemes": ["م", "علم", "ة"], "category": "regular"},
        {"word": "درس", "morphemes": ["درس"], "category": "regular"},
        {"word": "دارس", "morphemes": ["دارس"], "category": "regular"},
        {"word": "مدروس", "morphemes": ["م", "دروس"], "category": "regular"},
        {"word": "دراسة", "morphemes": ["دراس", "ة"], "category": "regular"},
        {"word": "مدرسة", "morphemes": ["م", "درس", "ة"], "category": "regular"},
        {"word": "عمل", "morphemes": ["عمل"], "category": "regular"},
        {"word": "عامل", "morphemes": ["عامل"], "category": "regular"},
        {"word": "معمول", "morphemes": ["م", "عمول"], "category": "regular"},
        {"word": "اعمال", "morphemes": ["ا", "عمال"], "category": "regular"},
        {"word": "معمل", "morphemes": ["م", "عمل"], "category": "regular"},
        {"word": "حكم", "morphemes": ["حكم"], "category": "regular"},
        {"word": "حاكم", "morphemes": ["حاكم"], "category": "regular"},
        {"word": "محكوم", "morphemes": ["م", "حكوم"], "category": "regular"},
        {"word": "حكومة", "morphemes": ["حكوم", "ة"], "category": "regular"},
        {"word": "محكمة", "morphemes": ["م", "حكم", "ة"], "category": "regular"},
        {"word": "فتح", "morphemes": ["فتح"], "category": "regular"},
        {"word": "فاتح", "morphemes": ["فاتح"], "category": "regular"},
        {"word": "مفتوح", "morphemes": ["م", "فتوح"], "category": "regular"},
        {"word": "فتاح", "morphemes": ["فتاح"], "category": "regular"},
        {"word": "مفتاح", "morphemes": ["م", "فتاح"], "category": "regular"},
        {"word": "جمع", "morphemes": ["جمع"], "category": "regular"},
        {"word": "جامع", "morphemes": ["جامع"], "category": "regular"},
        {"word": "مجموع", "morphemes": ["م", "جموع"], "category": "regular"},
        {"word": "جمعية", "morphemes": ["جمع", "ية"], "category": "regular"},
        {"word": "اجتماع", "morphemes": ["اجتماع"], "category": "regular"},
        {"word": "نظر", "morphemes": ["نظر"], "category": "regular"},
        {"word": "ناظر", "morphemes": ["ناظر"], "category": "regular"},
        {"word": "منظور", "morphemes": ["م", "نظور"], "category": "regular"},
        {"word": "نظرية", "morphemes": ["نظر", "ية"], "category": "regular"},
        {"word": "منظار", "morphemes": ["م", "نظار"], "category": "regular"},
        {"word": "صنع", "morphemes": ["صنع"], "category": "regular"},
        {"word": "صانع", "morphemes": ["صانع"], "category": "regular"},
        {"word": "خرج", "morphemes": ["خرج"], "category": "regular"},
        {"word": "مخرج", "morphemes": ["م", "خرج"], "category": "regular"},
        {"word": "استخراج", "morphemes": ["است", "خراج"], "category": "regular"},
        {"word": "كبير", "morphemes": ["كبير"], "category": "regular"},
        {"word": "اكبر", "morphemes": ["ا", "كبر"], "category": "regular"},
        {"word": "جميل", "morphemes": ["جميل"], "category": "regular"},
        {"word": "سريع", "morphemes": ["سريع"], "category": "regular"},
        {"word": "قريب", "morphemes": ["قريب"], "category": "regular"},
        {"word": "قال", "morphemes": ["قال"], "category": "weak"},
        {"word": "قول", "morphemes": ["قول"], "category": "weak"},
        {"word": "قائل", "morphemes": ["قائل"], "category": "weak"},
        {"word": "مقال", "morphemes": ["م", "قال"], "category": "weak"},
        {"word": "نام", "morphemes": ["نام"], "category": "weak"},
        {"word": "نوم", "morphemes": ["نوم"], "category": "weak"},
        {"word": "نائم", "morphemes": ["نائم"], "category": "weak"},
        {"word": "وعد", "morphemes": ["وعد"], "category": "weak"},
        {"word": "مشى", "morphemes": ["مشى"], "category": "weak"},
        {"word": "مشي", "morphemes": ["مشي"], "category": "weak"},
        {"word": "ماشي", "morphemes": ["ماشي"], "category": "weak"},
        {"word": "دعا", "morphemes": ["دعا"], "category": "weak"},
        {"word": "دعوة", "morphemes": ["دعو", "ة"], "category": "weak"},
        {"word": "داعي", "morphemes": ["داعي"], "category": "weak"},
        {"word": "يقول", "morphemes": ["ي", "قول"], "category": "weak"},
        {"word": "علماء", "morphemes": ["علماء"], "category": "broken"},
        {"word": "مدارس", "morphemes": ["مدارس"], "category": "broken"},
        {"word": "رجال", "morphemes": ["رجال"], "category": "broken"},
        {"word": "اطفال", "morphemes": ["اطفال"], "category": "broken"},
        {"word": "دروس", "morphemes": ["دروس"], "category": "broken"},
        {"word": "بيوت", "morphemes": ["بيوت"], "category": "broken"},
        {"word": "شعوب", "morphemes": ["شعوب"], "category": "broken"},
        {"word": "ايام", "morphemes": ["ايام"], "category": "broken"},
        {"word": "بلاد", "morphemes": ["بلاد"], "category": "broken"},
        {"word": "امور", "morphemes": ["امور"], "category": "broken"},
        {"word": "نساء", "morphemes": ["نساء"], "category": "broken"},
        {"word": "ابناء", "morphemes": ["ابناء"], "category": "broken"},
        {"word": "عيون", "morphemes": ["عيون"], "category": "broken"},
        {"word": "قلوب", "morphemes": ["قلوب"], "category": "broken"},
        {"word": "كتب", "morphemes": ["كتب"], "category": "broken"},
        {"word": "ترجم", "morphemes": ["ترجم"], "category": "quad"},
        {"word": "ترجمة", "morphemes": ["ترجم", "ة"], "category": "quad"},
        {"word": "مترجم", "morphemes": ["م", "ترجم"], "category": "quad"},
        {"word": "زلزل", "morphemes": ["زلزل"], "category": "quad"},
        {"word": "زلزال", "morphemes": ["زلزال"], "category": "quad"},
        {"word": "دحرج", "morphemes": ["دحرج"], "category": "quad"},
        {"word": "برمج", "morphemes": ["برمج"], "category": "quad"},
        {"word": "برمجة", "morphemes": ["برمج", "ة"], "category": "quad"},
        {"word": "مبرمج", "morphemes": ["م", "برمج"], "category": "quad"},
        {"word": "فلسف", "morphemes": ["فلسف"], "category": "quad"},
        {"word": "في", "morphemes": ["في"], "category": "function"},
        {"word": "من", "morphemes": ["من"], "category": "function"},
        {"word": "على", "morphemes": ["على"], "category": "function"},
        {"word": "الى", "morphemes": ["الى"], "category": "function"},
        {"word": "هو", "morphemes": ["هو"], "category": "function"},
        {"word": "هي", "morphemes": ["هي"], "category": "function"},
        {"word": "الذي", "morphemes": ["الذي"], "category": "function"},
        {"word": "هذا", "morphemes": ["هذا"], "category": "function"},
        {"word": "لكن", "morphemes": ["لكن"], "category": "function"},
        {"word": "لان", "morphemes": ["لان"], "category": "function"},
    ]


def print_examples(label: str, rows: list[DetailRow]) -> None:
    print(f"  {label}:")
    for row in rows:
        print(
            f"    - {row['word']} [{row['category']}] "
            f"morph={row['morphemes']} bpe={row['bpe_tokens']} "
            f"ratio={row['respect_ratio']:.0%}"
        )


def main() -> None:
    test_words = build_test_words_100()

    tokenizer_dir = Path("results/tokenizers")
    all_results: dict[str, AnalysisResult] = {}

    for size in ["8k", "16k", "32k"]:
        model_file = tokenizer_dir / f"bpe_{size}.model"
        if not model_file.exists():
            print(f"Skipping {size} (not found)")
            continue

        sp = spm.SentencePieceProcessor(model_file=str(model_file))
        print(f"\n{'=' * 72}")
        print(f"BPE-{size} (vocab={sp.get_piece_size()})")
        print(f"{'=' * 72}")

        results = analyze_morpheme_boundaries(sp, test_words)
        all_results[size] = results

        print(f"Respect rate: {results['respect_rate']:.1%}")
        print(f"  Respected: {results['boundary_respected']}")
        print(f"  Broken: {results['boundary_broken']}")
        print("  Category breakdown:")
        for category in ["regular", "weak", "broken", "quad", "function"]:
            stats = results["category_breakdown"][category]
            print(
                f"    - {category}: {stats['respect_rate']:.1%} "
                f"({stats['respected']}/{stats['total']})"
            )

        print_examples("Worst 5 breaks", results["worst_5_breaks"])
        print_examples("Best 5 matches", results["best_5_matches"])

    print(f"\n{'=' * 72}")
    print("SUMMARY: Morpheme Boundary Respect Rate")
    print(f"{'=' * 72}")
    for size, result in all_results.items():
        print(f"  BPE-{size}: {result['respect_rate']:.1%}")

    output = Path("results/tokenizer_analysis.json")
    output.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nResults saved to {output}")

    if all_results:
        best = max(all_results.items(), key=lambda item: item[1]["respect_rate"])
        print(
            f"Recommendation: BPE-{best[0]} "
            f"(respect rate: {best[1]['respect_rate']:.1%})"
        )


if __name__ == "__main__":
    main()
