#!/usr/bin/env python3
'''
Kaggle/Colab training script.
Upload this + code + data -> train

Usage:
  python kaggle_training.py --experiment exp002_morph_data
'''

from __future__ import annotations

import argparse
import subprocess
import sys


def setup() -> None:
    subprocess.run(
        [
            sys.executable,
            '-m',
            'pip',
            'install',
            'sentencepiece',
            'pyyaml',
            'camel-tools',
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True)
    args = parser.parse_args()

    from scripts.train_model import main as train_main

    sys.argv = [
        'train_model.py',
        '--config',
        f'configs/experiment/{args.experiment}.yaml',
    ]
    train_main()


if __name__ == '__main__':
    setup()
    main()
