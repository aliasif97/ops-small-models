#!/usr/bin/env bash
# Rebuild data, (optionally) train v2, pick the best checkpoint, evaluate base/v1/v2, write eval/RESULTS.md.
# SKIP_TRAIN=1 ./run.sh   reuses the existing adapters-v2/ checkpoints and train_v2.log (no training).
# Read only on the company pricebook folder (PRICEBOOK_ROOT). No git.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv-mlx/bin/python
if [ ! -x "$PY" ]; then
  python3 -m venv --system-site-packages ../.venv-mlx
  SP=$(python3 -c "import site;print(site.getsitepackages()[0])")
  NSP=$(../.venv-mlx/bin/python -c "import site;print(site.getsitepackages()[0])")
  echo "$SP" > "$NSP/video_stack.pth"; ../.venv-mlx/bin/pip install -q openpyxl
fi
mkdir -p eval
echo "[1/5] build data"; $PY scripts/build_data.py
if [ "${SKIP_TRAIN:-0}" = "1" ]; then echo "[2/5] training skipped (SKIP_TRAIN=1)"; else
  echo "[2/5] train v2: 120 iterations, checkpoint every 20 (about 9 min)"; rm -rf adapters-v2
  date +%s > train_v2_start.txt; $PY -m mlx_lm lora -c lora_config_v2.yaml 2>&1 | tee train_v2.log; date +%s > train_v2_end.txt
  # v1 (400 iterations): $PY -m mlx_lm lora -c lora_config.yaml  (about 28 min; adapters/ already holds it)
fi
echo "[3/5] pick best v2 checkpoint"; $PY scripts/select_best.py
echo "[4/5] predict on held-out docs"
$PY scripts/evaluate.py base_norp none 1.0
$PY scripts/evaluate.py v1_norp adapters 1.0
$PY scripts/evaluate.py base none 1.1
$PY scripts/evaluate.py v1 adapters 1.1
$PY scripts/evaluate.py v2 adapters-v2/best 1.1
echo "[5/5] score"; $PY scripts/make_results.py
