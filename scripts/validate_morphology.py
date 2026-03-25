"""Manual linguistic validation for the rule-based Arabic analyzer."""

from __future__ import annotations

from src.morphology.arabic_analyzer import ArabicAnalyzer


def validate() -> None:
    analyzer = ArabicAnalyzer("data/morphology/root_database.json")

    test_words = [
        "المدرسة",
        "الكاتب",
        "استخرجنا",
        "يتعلمون",
        "مكتبات",
        "العلماء",
        "تدريب",
        "اجتماعي",
        "القاهرة",
        "محمد",
        "تلفزيون",
        "ديمقراطية",
    ]

    results = {
        "correct": 0,
        "partial": 0,
        "wrong": 0,
        "function": 0,
        "foreign": 0,
        "total": len(test_words),
    }

    for word in test_words:
        analysis = analyzer.analyze(word)

        print(f"{word}:")
        print(
            f"  root={analysis.root} pattern={analysis.pattern} confidence={analysis.confidence:.2f}"
        )
        print(f"  prefixes={analysis.prefixes} suffixes={analysis.suffixes}")

        if analysis.is_function:
            results["function"] += 1

    print(f"\nResults: {results}")

    morph_applicable = results["total"] - results["function"] - results["foreign"]
    if morph_applicable > 0:
        accuracy = results["correct"] / morph_applicable
        print(f"Root accuracy: {accuracy:.1%}")
        print(
            f"Morphologically applicable: {morph_applicable}/{results['total']} = {morph_applicable / results['total']:.1%}"
        )


if __name__ == "__main__":
    validate()
