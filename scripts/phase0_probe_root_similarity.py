"""Phase 0 probe: are same-root Arabic tokens closer in baseline embeddings?

Inputs:
- token embeddings (.npy)
- tokenizer tokens (.txt)

Root extraction backend priority:
1) CAMeL Tools (preferred)
2) Rule-based ArabicAnalyzer fallback
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import cast
from typing import Protocol

import numpy as np

_ARABIC_RE = re.compile(r"^[\u0621-\u063A\u0641-\u064A]+$")


class RootExtractor(Protocol):
    def root_of(self, word: str) -> str | None:
        ...


class CamelRootExtractor:
    def __init__(self) -> None:
        from camel_tools.morphology.database import (  # type: ignore[import-untyped]
            MorphologyDB,
        )
        from camel_tools.morphology.analyzer import Analyzer  # type: ignore[import-untyped]

        self._analyzer = Analyzer(MorphologyDB.builtin_db())

    def root_of(self, word: str) -> str | None:
        analyses = self._analyzer.analyze(word)
        if not analyses:
            return None

        roots = [analysis.get("root", "") for analysis in analyses]
        roots = [root for root in roots if root and root != "NOAN"]
        if not roots:
            return None

        counts: dict[str, int] = {}
        for root in roots:
            counts[root] = counts.get(root, 0) + 1
        return max(counts, key=lambda root: counts[root])


class RuleBasedRootExtractor:
    def __init__(self) -> None:
        from src.morphology.arabic_analyzer import ArabicAnalyzer

        self._analyzer = ArabicAnalyzer("data/morphology/root_database.json")

    def root_of(self, word: str) -> str | None:
        analysis = self._analyzer.analyze(word)
        if analysis.is_function:
            return None
        if not analysis.root or "?" in analysis.root:
            return None
        if analysis.confidence < 0.5:
            return None
        return analysis.root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe same-root embedding similarity")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to token_embeddings.npy")
    parser.add_argument("--tokens", type=str, required=True, help="Path to tokens.txt")
    parser.add_argument("--output", type=str, required=True, help="Output JSON report")
    parser.add_argument("--min-root-size", type=int, default=3, help="Minimum words per root group")
    parser.add_argument("--max-words", type=int, default=8000, help="Max Arabic words to analyze")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _normalize_sp_piece(piece: str) -> str:
    cleaned = piece.replace("▁", "").strip()
    return cleaned


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def _choose_extractor() -> tuple[RootExtractor, str]:
    try:
        extractor = CamelRootExtractor()
        return extractor, "camel_tools"
    except Exception as exc:
        print(f"[warn] CAMeL unavailable ({exc}); using rule-based fallback")
        return RuleBasedRootExtractor(), "rule_based_fallback"


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    embeddings = np.load(args.embeddings)
    tokens = Path(args.tokens).read_text(encoding="utf-8").splitlines()

    if len(tokens) > embeddings.shape[0]:
        tokens = tokens[: embeddings.shape[0]]

    extractor, backend = _choose_extractor()

    root_to_indices: dict[str, list[int]] = defaultdict(list)
    used_words = 0
    skipped_non_arabic = 0

    for index, piece in enumerate(tokens):
        word = _normalize_sp_piece(piece)
        if len(word) < 3 or not _ARABIC_RE.match(word):
            skipped_non_arabic += 1
            continue

        root = extractor.root_of(word)
        if not root:
            continue

        root_to_indices[root].append(index)
        used_words += 1
        if used_words >= args.max_words:
            break

    valid_groups = {
        root: idxs
        for root, idxs in root_to_indices.items()
        if len(idxs) >= args.min_root_size
    }

    same_root_scores: list[float] = []
    random_scores: list[float] = []

    sampled_pairs = 0
    all_indices = [i for idxs in valid_groups.values() for i in idxs]
    if len(all_indices) < 2 or not valid_groups:
        raise RuntimeError("Not enough valid root groups to run probe")

    for idxs in valid_groups.values():
        for _ in range(min(len(idxs), 8)):
            i, j = random.sample(idxs, 2)
            same_root_scores.append(_cosine(embeddings[i], embeddings[j]))

            r1, r2 = random.sample(all_indices, 2)
            random_scores.append(_cosine(embeddings[r1], embeddings[r2]))
            sampled_pairs += 1

    same_mean = float(np.mean(same_root_scores)) if same_root_scores else 0.0
    random_mean = float(np.mean(random_scores)) if random_scores else 0.0
    delta = same_mean - random_mean

    top_roots = [
        {"root": root, "count": len(idxs)} for root, idxs in valid_groups.items()
    ]
    top_roots_sorted = sorted(
        top_roots,
        key=lambda item: cast(int, item["count"]),
        reverse=True,
    )[:20]

    report = {
        "backend": backend,
        "num_tokens": len(tokens),
        "used_words": used_words,
        "skipped_non_arabic": skipped_non_arabic,
        "num_roots_total": len(root_to_indices),
        "num_roots_valid": len(valid_groups),
        "sampled_pairs": sampled_pairs,
        "same_root_cosine_mean": same_mean,
        "random_pair_cosine_mean": random_mean,
        "delta": delta,
        "interpretation": (
            "positive_delta_same_root_closer"
            if delta > 0
            else "non_positive_delta_no_clear_root_clustering"
        ),
        "top_roots_by_count": top_roots_sorted,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] Probe report: {output_path}")
    print(
        "[probe] "
        f"same_root={same_mean:.4f} random={random_mean:.4f} delta={delta:.4f}"
    )


if __name__ == "__main__":
    main()
