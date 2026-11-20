#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
cd /home/caohy/repositories/Paper
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/caohy/repositories/Paper/CaMRA/navigation/scripts
task_ev=.worklog-testing/NAV-043/timeout-extension-v1
test "$(cat "$task_ev/preflight.exit")" = 0
test -f "$task_ev/prelaunch-commit.json"
if /pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B -u CaMRA/navigation/scripts/run_nav043_timeout.py > "$task_ev/run.log" 2>&1; then
  printf '0\n' > "$task_ev/run.exit"
else
  task_status=$?
  printf '%s\n' "$task_status" > "$task_ev/run.exit"
fi
export CUDA_VISIBLE_DEVICES=''
if /pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B -u CaMRA/navigation/scripts/audit_nav043_timeout.py > "$task_ev/audit.log" 2>&1; then
  printf '0\n' > "$task_ev/audit.exit"
else
  task_status=$?
  printf '%s\n' "$task_status" > "$task_ev/audit.exit"
  exit "$task_status"
fi
