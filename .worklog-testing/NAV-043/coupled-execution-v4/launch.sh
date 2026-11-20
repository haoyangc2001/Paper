#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
cd /home/caohy/repositories/Paper
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_ev=.worklog-testing/NAV-043/coupled-execution-v4
task_commit=$(cat "$task_ev/approval-commit.txt")
test "$(git show "$task_commit:CaMRA/navigation/config/nav043_coupled_response_approval_v4.json.sha256")" = "$(cat CaMRA/navigation/config/nav043_coupled_response_approval_v4.json.sha256)"
if /pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B -u CaMRA/navigation/scripts/launch_nav043_coupled_response_once.py > "$task_ev/launcher.log" 2>&1; then
  printf '0\n' > "$task_ev/launcher.exit"
else
  task_status=$?
  printf '%s\n' "$task_status" > "$task_ev/launcher.exit"
  exit "$task_status"
fi
