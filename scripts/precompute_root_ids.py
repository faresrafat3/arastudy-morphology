#!/usr/bin/env python3
'''
Pre-compute root_id for every token in the vocabulary.
Maps: BPE token_id -> root_id (or 0 for unknown)

Output: data/morphology/token_root_map.json
'''

from __future__ import annotations

import json
import re
from pathlib import Path

import sentencepiece as spm
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB


ARABIC_ROOT_RE = re.compile(r'^[\u0600-\u06FF]{2,5}$')


def is_valid_arabic_root(root: str) -> bool:
    if not root:
        return False
    if '#' in root or '.' in root:
        return False
    return bool(ARABIC_ROOT_RE.match(root))


def build_token_root_map(
    tokenizer_model: str,
    output_file: str,
) -> None:
    sp = spm.SentencePieceProcessor(model_file=tokenizer_model)
    db = MorphologyDB.builtin_db()
    analyzer = Analyzer(db)

    vocab_size = sp.get_piece_size()

    root_to_id = {'<UNK>': 0}
    token_to_root_id: dict[int, int] = {}

    print(f'Analyzing {vocab_size} tokens...')

    for token_id in range(vocab_size):
        piece = sp.id_to_piece(token_id)
        word = piece.replace('▁', '').strip()

        if not word or len(word) < 2:
            token_to_root_id[token_id] = 0
            continue

        analyses = analyzer.analyze(word)
        if analyses:
            root = analyses[0].get('root', '')
            if root and root != 'NOAN':
                clean_root = root.replace('.', '').replace(' ', '')
                if is_valid_arabic_root(clean_root):
                    if clean_root not in root_to_id:
                        root_to_id[clean_root] = len(root_to_id)
                    token_to_root_id[token_id] = root_to_id[clean_root]
                else:
                    token_to_root_id[token_id] = 0
            else:
                token_to_root_id[token_id] = 0
        else:
            token_to_root_id[token_id] = 0

    known = sum(1 for value in token_to_root_id.values() if value > 0)
    print(f'Tokens with known root: {known}/{vocab_size} ({known / vocab_size:.0%})')
    print(f'Unique roots: {len(root_to_id)}')

    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                'token_to_root_id': token_to_root_id,
                'root_to_id': root_to_id,
                'stats': {
                    'vocab_size': vocab_size,
                    'known_roots': known,
                    'unique_roots': len(root_to_id),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )

    print(f'Saved to {output}')


if __name__ == '__main__':
    build_token_root_map(
        'results/tokenizers/bpe_16k.model',
        'data/morphology/token_root_map.json',
    )
