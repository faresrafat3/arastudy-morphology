#!/usr/bin/env bash
set -euo pipefail

cd /home/fares/bug-hunter

PYTHON_BIN="/home/fares/bug-hunter/.venv/bin/python"
LOG_DIR="results"
mkdir -p "$LOG_DIR"

echo "[runner] Waiting for 10M process to finish..."
while pgrep -f "scripts/train_model.py --config configs/experiment/exp004_10m.yaml" >/dev/null; do
  sleep 30
done

echo "[runner] 10M finished. Starting Phase 2 (local)..."
nice -n 10 "$PYTHON_BIN" scripts/train_model.py --config configs/experiment/exp003_root_emb.yaml > "$LOG_DIR/train_phase2_local.log" 2>&1

echo "[runner] Phase 2 finished. Done."
