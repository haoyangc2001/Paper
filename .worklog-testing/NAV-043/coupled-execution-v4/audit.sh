#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
cd /home/caohy/repositories/Paper
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_ev=.worklog-testing/NAV-043/coupled-execution-v4
test "$(cat "$task_ev/launcher.exit")" = 0
if /pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B -u CaMRA/navigation/scripts/audit_nav043_coupled_response.py > "$task_ev/audit.log" 2>&1; then
  printf '0\n' > "$task_ev/audit.exit"
else
  task_status=$?
  printf '%s\n' "$task_status" > "$task_ev/audit.exit"
  exit "$task_status"
fi
