#!/usr/bin/env bash
set -euo pipefail

cd /home/fares/bug-hunter

PYTHON_BIN="/home/fares/bug-hunter/.venv/bin/python"
LOG_DIR="results"
mkdir -p "$LOG_DIR"

echo "[runner] Waiting for 10M process to finish..."
while ps -eo cmd | grep -F "scripts/train_model.py --config configs/experiment/exp004_10m.yaml" | grep -v grep >/dev/null; do
  sleep 30
done

echo "[runner] 10M finished. Starting Phase 1..."
nice -n 10 "$PYTHON_BIN" scripts/train_model.py --config configs/experiment/exp002_morph_data.yaml > "$LOG_DIR/train_phase1_local.log" 2>&1

echo "[runner] Phase 1 finished. Starting Phase 2..."
nice -n 10 "$PYTHON_BIN" scripts/train_model.py --config configs/experiment/exp003_root_emb.yaml > "$LOG_DIR/train_phase2_local.log" 2>&1

echo "[runner] Phase 2 finished. Done."
