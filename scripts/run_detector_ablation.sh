#!/bin/bash
# Ablation (d): GPU passes per detector, each detector's CPU fit overlapping the next detector's GPU pass.
# Logs in logs/detector_ablation_*.log; results in results/ablation/ (wad/detector_ablation.py).
cd "$(dirname "$0")/.." || exit 1
PY=.venv/Scripts/python.exe
for d in "${@:-v8x_640 v8s_640 v8s_1280}"; do
  for det in $d; do
    $PY -u -m wad.detector_ablation gpu --det "$det" >> logs/detector_ablation_gpu.log 2>&1 \
      || { echo "GPU $det FAILED" >> logs/detector_ablation_gpu.log; wait; exit 1; }
    OMP_NUM_THREADS=8 $PY -u -m wad.detector_ablation fit --det "$det" > "logs/detector_ablation_fit_$det.log" 2>&1 &
  done
done
wait
$PY -u -m wad.detector_ablation report > logs/detector_ablation_report.log 2>&1
echo "EXIT $?" >> logs/detector_ablation_gpu.log
