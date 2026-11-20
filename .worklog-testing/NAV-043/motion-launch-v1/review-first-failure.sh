#!/usr/bin/env bash
set -euo pipefail
cd /home/caohy/repositories/Paper
export PYTHONPATH=/home/caohy/repositories/Paper/CaMRA/navigation/src:/home/caohy/repositories/Paper/CaMRA/calibration/src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
task_evidence=.worklog-testing/NAV-043/motion-launch-v1
if /pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B "$task_evidence/review-first-failure.py" > "$task_evidence/first-failure-review.log" 2>&1; then
  printf '0\n' > "$task_evidence/first-failure-review.exit"
else
  task_status=$?
  printf '%s\n' "$task_status" > "$task_evidence/first-failure-review.exit"
  exit "$task_status"
fi
