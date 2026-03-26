"""Arabic Morphological Analyzer — Rule-Based (Tier 1)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MorphAnalysis:
    """Result of morphological analysis for a single Arabic word."""

    original: str
    root: str
    root_id: int
    pattern: str
    pattern_id: int
    prefixes: list[str]
    suffixes: list[str]
    confidence: float
    is_function: bool


PREFIXES: list[tuple[str, list[str]]] = [
    ("وبال", ["و", "ب", "ال"]),
    ("وال", ["و", "ال"]),
    ("بال", ["ب", "ال"]),
    ("لل", ["ل", "ال"]),
    ("وك", ["و", "ك"]),
    ("ول", ["و", "ل"]),
    ("وب", ["و", "ب"]),
    ("ال", ["ال"]),
    ("و", ["و"]),
    ("ب", ["ب"]),
    ("ل", ["ل"]),
    ("ف", ["ف"]),
    ("ك", ["ك"]),
    ("س", ["س"]),
]

SUFFIXES: list[tuple[str, list[str]]] = [
    ("ناها", ["نا", "ها"]),
    ("وها", ["و", "ها"]),
    ("تهم", ["ت", "هم"]),
    ("ونه", ["ون", "ه"]),
    ("اتها", ["ات", "ها"]),
    ("ات", ["ات"]),
    ("ون", ["ون"]),
    ("ين", ["ين"]),
    ("ان", ["ان"]),
    ("وا", ["وا"]),
    ("نا", ["نا"]),
    ("هم", ["هم"]),
    ("هن", ["هن"]),
    ("ها", ["ها"]),
    ("ته", ["ت", "ه"]),
    ("ة", ["ة"]),
    ("ه", ["ه"]),
    ("ي", ["ي"]),
    ("ا", ["ا"]),
]

FUNCTION_WORDS = {
    "في",
    "من",
    "على",
    "إلى",
    "الى",
    "عن",
    "مع",
    "بين",
    "هو",
    "هي",
    "هم",
    "هن",
    "نحن",
    "أنا",
    "انا",
    "أنت",
    "انت",
    "هذا",
    "هذه",
    "ذلك",
    "تلك",
    "هؤلاء",
    "الذي",
    "التي",
    "اللذان",
    "الذين",
    "ما",
    "لا",
    "لم",
    "لن",
    "إن",
    "ان",
    "أن",
    "كان",
    "يكون",
    "ليس",
    "كل",
    "بعض",
    "أي",
    "ثم",
    "لكن",
    "بل",
    "حتى",
    "إذا",
    "اذا",
    "لو",
    "قد",
    "سوف",
    "حيث",
    "أيضا",
    "أيضاً",
    "عند",
    "منذ",
    "خلال",
    "حول",
    "دون",
    "بدون",
    "فوق",
    "تحت",
    "أمام",
    "امام",
    "خلف",
    "قبل",
    "بعد",
}

PATTERNS: dict[str, int] = {
    "فعل": 0,
    "فعيل": 1,
    "فاعل": 2,
    "مفعول": 3,
    "فعال": 4,
    "فعالة": 5,
    "مفعل": 6,
    "مفعلة": 7,
    "تفعيل": 8,
    "إفعال": 9,
    "فعول": 10,
    "افتعال": 11,
    "استفعال": 12,
    "فعلان": 13,
    "فعيلة": 14,
    "أفعل": 15,
    "تفاعل": 16,
    "انفعال": 17,
    "مفاعلة": 18,
    "فعّال_جمع": 19,
    "فعلة": 20,
}


class ArabicAnalyzer:
    """Fast Arabic rule-based morphological analyzer."""

    def __init__(self, root_db_path: str | None = None):
        self.roots: dict[str, int] = {}
        self.root_words: dict[str, list[str] | dict[str, str]] = {}
        if root_db_path:
            self._load_root_db(root_db_path)

    def _load_root_db(self, path: str) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for index, entry in enumerate(data):
            root = str(entry["root"])
            self.roots[root] = index
            self.root_words[root] = entry.get("words", [])

    def analyze(self, word: str) -> MorphAnalysis:
        word = word.strip()

        if not word:
            return MorphAnalysis(
                original="",
                root="",
                root_id=-1,
                pattern="",
                pattern_id=-1,
                prefixes=[],
                suffixes=[],
                confidence=0.0,
                is_function=False,
            )

        if word in FUNCTION_WORDS:
            return MorphAnalysis(
                original=word,
                root=word,
                root_id=-1,
                pattern="",
                pattern_id=-1,
                prefixes=[],
                suffixes=[],
                confidence=1.0,
                is_function=True,
            )

        stem = word
        found_prefixes: list[str] = []
        for prefix_str, prefix_parts in PREFIXES:
            if stem.startswith(prefix_str) and len(stem) > len(prefix_str) + 2:
                stem = stem[len(prefix_str) :]
                found_prefixes = prefix_parts
                break

        found_suffixes: list[str] = []
        for suffix_str, suffix_parts in SUFFIXES:
            if stem.endswith(suffix_str) and len(stem) > len(suffix_str) + 2:
                stem = stem[: -len(suffix_str)]
                found_suffixes = suffix_parts
                break

        root, pattern, confidence = self._extract_root(stem)
        root_id = self.roots.get(root, -1)
        pattern_id = PATTERNS.get(pattern, -1)

        if root_id == -1:
            confidence *= 0.5

        return MorphAnalysis(
            original=word,
            root=root,
            root_id=root_id,
            pattern=pattern,
            pattern_id=pattern_id,
            prefixes=found_prefixes,
            suffixes=found_suffixes,
            confidence=confidence,
            is_function=False,
        )

    def _extract_root(self, stem: str) -> tuple[str, str, float]:
        zawaid = set("اويتمنسه")
        non_root_chars = set("ءآأإؤئةى")

        consonants: list[str] = []
        for character in stem:
            if character not in zawaid and character not in non_root_chars:
                consonants.append(character)

        if len(consonants) >= 3:
            root = "".join(consonants[:3])
            pattern = self._detect_pattern(stem, root)
            return root, pattern, 0.7
        if len(consonants) == 2:
            return "".join(consonants) + "?", "", 0.3

        fallback = stem[:3] if len(stem) >= 3 else stem
        return fallback, "", 0.1

    def _detect_pattern(self, stem: str, root: str) -> str:
        if len(root) != 3:
            return ""

        if len(stem) == 3:
            return "فعل"
        if len(stem) == 4:
            if stem.startswith("م"):
                return "مفعل"
            if len(stem) > 1 and stem[1] == "ا":
                return "فاعل"
            if stem.endswith("ة"):
                return "فعلة"
        if len(stem) == 5:
            if stem.startswith("م") and stem.endswith("ة"):
                return "مفعلة"
            if stem.startswith("ت"):
                return "تفعيل"
            if len(stem) > 1 and stem[1] == "ا" and stem.endswith("ة"):
                return "فعالة"
        if len(stem) >= 6:
            if stem.startswith("است"):
                return "استفعال"
            if stem.startswith("انف") or stem.startswith("انق"):
                return "انفعال"
            if stem.startswith("افت"):
                return "افتعال"

        return "فعل"

    def analyze_batch(self, words: list[str]) -> list[MorphAnalysis]:
        return [self.analyze(word) for word in words]

    def analyze_text(self, text: str) -> list[MorphAnalysis]:
        return self.analyze_batch(text.split())
