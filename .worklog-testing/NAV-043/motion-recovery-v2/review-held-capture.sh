#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
cd /home/caohy/repositories/Paper
task_ev=.worklog-testing/NAV-043/motion-recovery-v2
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export PYTHONPATH=/home/caohy/repositories/Paper/CaMRA/navigation/src:/home/caohy/repositories/Paper/CaMRA/calibration/src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if /pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B -u "$task_ev/review-held-capture.py" > "$task_ev/review-held-capture.log" 2>&1; then
  printf '0\n' > "$task_ev/review-held-capture.exit"
else
  task_status=$?
  printf '%s\n' "$task_status" > "$task_ev/review-held-capture.exit"
  exit "$task_status"
fi
