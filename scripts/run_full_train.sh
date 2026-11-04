#!/bin/bash
# Training on all BDD100K train frames (wad/full_train.py), after the images are fetched:
#   .venv/Scripts/python.exe -m wad.full_train names
#   .venv/Scripts/python.exe scripts/get_bdd100k.py --no-val --train-list cache/thresh_full/fetch_names.txt
# then this script: scan, GPU passes (detection and per-box outputs for the new frames), fit and grading.
# Logs in logs/full_train_*.log; results in results/ablation/full_train*.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/Scripts/python.exe
$PY -u -m wad.full_train scan > logs/full_train_scan.log 2>&1 || { echo "scan FAILED"; exit 1; }
$PY -u -m wad.full_train gpu > logs/full_train_gpu.log 2>&1 || { echo "gpu FAILED"; exit 1; }
OMP_NUM_THREADS=8 $PY -u -m wad.full_train fit > logs/full_train_fit.log 2>&1 || { echo "fit FAILED"; exit 1; }
echo "full train: done"
