#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
cd /home/caohy/repositories/Paper
task_python=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
task_evidence=.worklog-testing/NAV-043/motion-launch-v1
export PYTHONPATH=/home/caohy/repositories/Paper/CaMRA/navigation/src:/home/caohy/repositories/Paper/CaMRA/calibration/src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=6
test -s "$task_evidence/approval-preflight.json"
test ! -e "$task_evidence/launch.log"
test ! -e "$task_evidence/launch.exit"
git cat-file -e HEAD:CaMRA/navigation/config/nav043_motion_probe_approval_v1.json
if "$task_python" -B CaMRA/navigation/scripts/run_nav043_motion_probe.py > "$task_evidence/launch.log" 2>&1; then
  printf '0\n' > "$task_evidence/launch.exit"
else
  task_status=$?
  printf '%s\n' "$task_status" > "$task_evidence/launch.exit"
  exit "$task_status"
fi
export CUDA_VISIBLE_DEVICES=''
if "$task_python" -B CaMRA/navigation/scripts/audit_nav043_motion_probe.py > "$task_evidence/saved-review.log" 2>&1; then
  printf '0\n' > "$task_evidence/saved-review.exit"
else
  task_status=$?
  printf '%s\n' "$task_status" > "$task_evidence/saved-review.exit"
  exit "$task_status"
fi
