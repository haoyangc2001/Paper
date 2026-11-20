#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
cd /home/caohy/repositories/Paper
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
task_ev=.worklog-testing/NAV-043/motion-recovery-v2
if /pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B "$task_ev/inspect-held.py" > "$task_ev/inspect-held.log" 2>&1; then
  printf '0\n' > "$task_ev/inspect-held.exit"
else
  task_status=$?
  printf '%s\n' "$task_status" > "$task_ev/inspect-held.exit"
  exit "$task_status"
fi
