#!/usr/bin/env python3
"""Analyze BPE morpheme-boundary alignment for Arabic words."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import sentencepiece as spm  # type: ignore[import-untyped]


class WordMorphemes(TypedDict):
    word: str
    morphemes: list[str]


class DetailRow(TypedDict):
    word: str
    morphemes: list[str]
    bpe_tokens: list[str]
    morpheme_boundaries: list[int]
    bpe_boundaries: list[int]
    respect_ratio: float
    respected: bool


class AnalysisResult(TypedDict):
    total_words: int
    boundary_respected: int
    boundary_broken: int
    details: list[DetailRow]
    respect_rate: float


def analyze_morpheme_boundaries(
    sp: spm.SentencePieceProcessor,
    words_with_morphemes: list[WordMorphemes],
) -> AnalysisResult:
    results: AnalysisResult = {
        "total_words": 0,
        "boundary_respected": 0,
        "boundary_broken": 0,
        "details": [],
        "respect_rate": 0.0,
    }

    details: list[DetailRow] = []

    for entry in words_with_morphemes:
        word = entry["word"]
        expected = entry["morphemes"]

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

        results["total_words"] += 1
        if respected:
            results["boundary_respected"] += 1
        else:
            results["boundary_broken"] += 1

        details.append(
            {
                "word": word,
                "morphemes": expected,
                "bpe_tokens": tokens,
                "morpheme_boundaries": sorted(morpheme_boundaries),
                "bpe_boundaries": sorted(bpe_boundaries),
                "respect_ratio": round(respect_ratio, 2),
                "respected": respected,
            }
        )

    total_words = max(int(results["total_words"]), 1)
    results["details"] = details
    results["respect_rate"] = round(results["boundary_respected"] / total_words, 3)
    return results


def main() -> None:
    test_words: list[WordMorphemes] = [
        {"word": "كاتب", "morphemes": ["كاتب"]},
        {"word": "مكتوب", "morphemes": ["مكتوب"]},
        {"word": "المكتبة", "morphemes": ["ال", "مكتب", "ة"]},
        {"word": "كتابة", "morphemes": ["كتاب", "ة"]},
        {"word": "والكاتبون", "morphemes": ["و", "ال", "كاتب", "ون"]},
        {"word": "استخرجناها", "morphemes": ["است", "خرج", "نا", "ها"]},
        {"word": "يتعلمون", "morphemes": ["ي", "تعلم", "ون"]},
        {"word": "بالمدرسة", "morphemes": ["ب", "ال", "مدرس", "ة"]},
        {"word": "فكتبوها", "morphemes": ["ف", "كتب", "وا", "ها"]},
        {"word": "للمعلمين", "morphemes": ["ل", "ال", "معلم", "ين"]},
        {"word": "وعلماؤنا", "morphemes": ["و", "علماء", "نا"]},
        {"word": "تدريبهم", "morphemes": ["تدريب", "هم"]},
        {"word": "اجتماعية", "morphemes": ["اجتماع", "ية"]},
        {"word": "المستخرجات", "morphemes": ["ال", "مستخرج", "ات"]},
        {"word": "سيكتبونها", "morphemes": ["س", "ي", "كتب", "ون", "ها"]},
        {"word": "معلوماتهم", "morphemes": ["معلوم", "ات", "هم"]},
        {"word": "الدراسات", "morphemes": ["ال", "دراس", "ات"]},
        {"word": "مدرسون", "morphemes": ["مدرس", "ون"]},
        {"word": "القاهرة", "morphemes": ["ال", "قاهر", "ة"]},
        {"word": "استقلال", "morphemes": ["است", "قلال"]},
    ]

    tokenizer_dir = Path("results/tokenizers")
    all_results: dict[str, AnalysisResult] = {}

    for size in ["8k", "16k", "32k"]:
        model_file = tokenizer_dir / f"bpe_{size}.model"
        if not model_file.exists():
            print(f"Skipping {size} (not found!)")
            continue

        sp = spm.SentencePieceProcessor(model_file=str(model_file))
        print(f"\n{'=' * 60}")
        print(f"BPE-{size} (vocab={sp.get_piece_size()})")
        print(f"{'=' * 60}")

        results = analyze_morpheme_boundaries(sp, test_words)
        all_results[size] = results

        print(
            "Morpheme boundary respect rate: "
            f"{results['respect_rate']:.1%}"
        )
        print(f"  Respected: {results['boundary_respected']}")
        print(f"  Broken: {results['boundary_broken']}")

        for detail in results["details"]:
            status = "✅" if detail["respected"] else "❌"
            print(f"  {status} {detail['word']}")
            print(f"     morphemes: {detail['morphemes']}")
            print(f"     BPE:       {detail['bpe_tokens']}")
            print(f"     respect:   {detail['respect_ratio']:.0%}")

    print(f"\n{'=' * 60}")
    print("SUMMARY: Morpheme Boundary Respect Rate")
    print(f"{'=' * 60}")
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
            f"\nRecommendation: BPE-{best[0]} "
            f"(respect rate: {best[1]['respect_rate']:.1%})"
        )


if __name__ == "__main__":
    main()
