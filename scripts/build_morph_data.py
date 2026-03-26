#!/usr/bin/env python3
'''
Build morphological training data from CAMeL Tools.
Extract root-word families from Wikipedia corpus.

Output: data/morphology/root_word_lists.txt
Format: one line per root family:
  "جذر كتب: كتب كاتب مكتوب كتاب كتابة مكتبة"
  "جذر علم: علم عالم معلوم تعليم علوم معلمة"
'''

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB


def extract_unique_words(corpus_file: str, max_words: int = 100000) -> list[str]:
    '''Extract top N unique words from corpus.'''
    word_freq = Counter()
    with open(corpus_file, encoding='utf-8') as handle:
        for line in handle:
            for word in line.split():
                if re.search(r'[\u0600-\u06FF]', word) and len(word) >= 2:
                    clean = re.sub(r'[^\u0600-\u06FF]', '', word)
                    if clean:
                        word_freq[clean] += 1

    return [w for w, _ in word_freq.most_common(max_words)]


def analyze_words(words: list[str]) -> dict[str, list[str]]:
    '''Group words by their root using CAMeL Tools.'''
    print('Loading CAMeL morphology database...')
    db = MorphologyDB.builtin_db()
    analyzer = Analyzer(db)

    root_words = defaultdict(set)
    analyzed = 0

    for index, word in enumerate(words):
        if index % 10000 == 0:
            print(f'  Analyzing word {index}/{len(words)}...')

        analyses = analyzer.analyze(word)
        if analyses:
            root = analyses[0].get('root', '')
            if root and root != 'NOAN':
                clean_root = root.replace('.', '').replace(' ', '')
                if len(clean_root) >= 2:
                    root_words[clean_root].add(word)
                    analyzed += 1

    print(f'Analyzed: {analyzed}/{len(words)} words!')
    print(f'Unique roots: {len(root_words)}!')
    return {r: sorted(ws) for r, ws in root_words.items()}


def build_morph_data(
    root_words: dict[str, list[str]],
    output_file: str,
    min_family_size: int = 3,
    max_words_per_root: int = 10,
) -> dict[str, float | int]:
    '''Build morphological training data.'''
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    families = 0
    total_words = 0

    with open(out, 'w', encoding='utf-8') as handle:
        for root, words in sorted(root_words.items(), key=lambda item: -len(item[1])):
            if len(words) < min_family_size:
                continue

            selected = words[:max_words_per_root]
            line = f"جذر {root}: {' '.join(selected)}"
            handle.write(line + '\n')
            families += 1
            total_words += len(selected)

    avg_words = total_words / max(families, 1)
    print(f'Root families: {families}')
    print(f'Total words: {total_words}')
    print(f'Avg words/root: {avg_words:.1f}')
    print(f'Output: {out}')

    return {
        'families': families,
        'total_words': total_words,
        'avg_words_per_root': round(avg_words, 2),
    }


def main() -> None:
    corpus = 'data/processed/train.txt'
    output = 'data/morphology/root_word_lists.txt'

    print('Step 1: Extract unique words...')
    words = extract_unique_words(corpus, max_words=50000)
    print(f'  Unique words: {len(words)}')

    print('Step 2: Analyze with CAMeL...')
    root_words = analyze_words(words)

    print('Step 3: Build training data...')
    stats = build_morph_data(root_words, output)

    stats_file = Path('data/morphology/morph_data_stats.json')
    stats_file.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')

    print('\nSamples:')
    shown = 0
    with open(output, encoding='utf-8') as handle:
        for line in handle:
            print(f'  {line.strip()[:80]}')
            shown += 1
            if shown >= 5:
                break


if __name__ == '__main__':
    main()
