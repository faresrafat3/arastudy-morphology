"""Build morphology-augmented text lines (Phase 1: Data-Only).

Generates lines like:
  جذر كتب: كتب كاتب مكتوب كتاب كتابة مكتبة

These lines can be concatenated with normal training corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build morphology data-only corpus")
    parser.add_argument(
        "--root-db",
        type=str,
        default="data/morphology/root_database.json",
        help="Path to root database JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/morphology/morph_root_lines.txt",
        help="Output text file",
    )
    parser.add_argument(
        "--max-words-per-root",
        type=int,
        default=6,
        help="Max number of forms used per root",
    )
    return parser


def _forms_from_entry(entry: dict[str, Any], max_words: int) -> list[str]:
    words = entry.get("words", [])

    if isinstance(words, dict):
        forms = [str(value).strip() for value in words.values() if str(value).strip()]
    elif isinstance(words, list):
        forms = [str(value).strip() for value in words if str(value).strip()]
    else:
        forms = []

    deduped: list[str] = []
    seen: set[str] = set()
    for form in forms:
        if form not in seen:
            seen.add(form)
            deduped.append(form)

    return deduped[:max_words]


def main() -> None:
    args = build_arg_parser().parse_args()

    root_db_path = Path(args.root_db)
    output_path = Path(args.output)

    entries = json.loads(root_db_path.read_text(encoding="utf-8"))

    lines: list[str] = []
    for entry in entries:
        root = str(entry.get("root", "")).strip()
        if not root:
            continue
        forms = _forms_from_entry(entry, args.max_words_per_root)
        if not forms:
            continue
        lines.append(f"جذر {root}: {' '.join(forms)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[ok] Wrote {len(lines)} root lines to {output_path}")


if __name__ == "__main__":
    main()
