"""Training entrypoint for AraStudy baseline model."""

from __future__ import annotations

import argparse

from src.training.trainer import train_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AraStudy baseline model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment/exp001_baseline.yaml",
        help="Path to experiment yaml",
    )
    args = parser.parse_args()
    train_experiment(args.config)


if __name__ == "__main__":
    main()
