#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
cd /home/caohy/repositories/Paper
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_ev=.worklog-testing/NAV-043/supported-execution-v3
test "$(cat "$task_ev/preflight.exit")" = 0
git -c core.preloadIndex=false -c core.fsmonitor=false rev-parse HEAD > "$task_ev/launch-source-commit.txt"
if /pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B -u CaMRA/navigation/scripts/launch_nav043_supported_response_once.py > "$task_ev/launcher.log" 2>&1; then
  printf '0\n' > "$task_ev/launcher.exit"
else
  task_status=$?
  printf '%s\n' "$task_status" > "$task_ev/launcher.exit"
  exit "$task_status"
fi
