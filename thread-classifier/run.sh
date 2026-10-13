#!/bin/bash
# Full pipeline: dataset -> baseline -> fine-tune -> audit -> latency. Needs data/parts/*.txt (exported from Supabase).
set -e
cd "$(dirname "$0")"
PY="$(cd .. && pwd)/.venv/bin/python"
mkdir -p eval models
cd scripts
$PY 01_build_dataset.py && $PY 02_baseline.py && $PY 03_finetune.py && $PY 04_audit.py && $PY 05_latency.py
