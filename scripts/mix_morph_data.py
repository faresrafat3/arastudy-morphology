#!/usr/bin/env python3
'''
Mix morphological data with Wikipedia for Phase 1 training.
CONTROLLED: same total tokens!
Replace some Wikipedia lines with morphological lines.
'''

from __future__ import annotations

import random
from pathlib import Path


def mix_data(
    wiki_file: str,
    morph_file: str,
    output_file: str,
    morph_ratio: float = 0.05,
    seed: int = 42,
) -> None:
    random.seed(seed)

    wiki_lines = Path(wiki_file).read_text(encoding='utf-8').splitlines()
    morph_lines = Path(morph_file).read_text(encoding='utf-8').splitlines()

    n_morph = int(len(wiki_lines) * morph_ratio)
    n_wiki = len(wiki_lines) - n_morph

    random.shuffle(wiki_lines)
    selected_wiki = wiki_lines[:n_wiki]

    selected_morph: list[str] = []
    while len(selected_morph) < n_morph:
        selected_morph.extend(morph_lines)
    selected_morph = selected_morph[:n_morph]

    mixed = selected_wiki + selected_morph
    random.shuffle(mixed)

    Path(output_file).write_text('\n'.join(mixed) + '\n', encoding='utf-8')

    print('Mixed data:')
    print(f'  Wiki lines: {n_wiki}')
    print(f'  Morph lines: {n_morph} ({morph_ratio:.0%})')
    print(f'  Total: {len(mixed)}')
    print(f'  Output: {output_file}')


if __name__ == '__main__':
    mix_data(
        'data/processed/train.txt',
        'data/morphology/root_word_lists.txt',
        'data/processed/train_phase1.txt',
        morph_ratio=0.05,
    )
