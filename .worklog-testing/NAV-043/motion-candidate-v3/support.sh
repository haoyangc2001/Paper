#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
cd /home/caohy/repositories/Paper
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_ev=.worklog-testing/NAV-043/motion-candidate-v3
if /pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B -u "$task_ev/support.py" > "$task_ev/support.log" 2>&1; then
  printf '0\n' > "$task_ev/support.exit"
else
  task_status=$?
  printf '%s\n' "$task_status" > "$task_ev/support.exit"
  exit "$task_status"
fi
