#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
cd /home/caohy/repositories/Paper
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_ev=.worklog-testing/NAV-043/motion-candidate-v3
if /pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B -u "$task_ev/close.py" > "$task_ev/close.log" 2>&1; then
  printf '0\n' > "$task_ev/close.exit"
else
  task_status=$?
  printf '%s\n' "$task_status" > "$task_ev/close.exit"
  exit "$task_status"
fi
/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B /home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py doc/plan/20261109_camra_navigation_implementation.json --mode update > "$task_ev/close-plan.log" 2>&1
git diff --check > "$task_ev/close-diff-check.log" 2>&1
