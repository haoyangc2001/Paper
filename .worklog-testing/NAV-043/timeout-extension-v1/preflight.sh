#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
cd /home/caohy/repositories/Paper
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/caohy/repositories/Paper/CaMRA/navigation/scripts
task_ev=.worklog-testing/NAV-043/timeout-extension-v1
test "$(cat "$task_ev/basic_v2.exit")" = 0
if /pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B -u CaMRA/navigation/scripts/run_nav043_timeout.py --check > "$task_ev/preflight.log" 2>&1; then
  printf '0\n' > "$task_ev/preflight.exit"
else
  task_status=$?
  printf '%s\n' "$task_status" > "$task_ev/preflight.exit"
  exit "$task_status"
fi
/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python /home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py doc/plan/20261109_camra_navigation_implementation.json --mode update > "$task_ev/plan-validation.log" 2>&1
