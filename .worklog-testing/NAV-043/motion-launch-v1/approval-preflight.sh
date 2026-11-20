#!/usr/bin/env bash
set -euo pipefail
cd /home/caohy/repositories/Paper
task_python=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
task_evidence=.worklog-testing/NAV-043/motion-launch-v1
export PYTHONPATH=/home/caohy/repositories/Paper/CaMRA/navigation/src:/home/caohy/repositories/Paper/CaMRA/calibration/src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=6
test ! -e "$task_evidence/approval-preflight.exit"
if "$task_python" -B "$task_evidence/approval-preflight.py" > "$task_evidence/approval-preflight.log" 2>&1; then
  printf '0\n' > "$task_evidence/approval-preflight.exit"
else
  task_status=$?
  printf '%s\n' "$task_status" > "$task_evidence/approval-preflight.exit"
  exit "$task_status"
fi
"$task_python" -B /home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py doc/plan/20261109_camra_navigation_implementation.json --mode update > "$task_evidence/approval-plan.log" 2>&1
git diff --check > "$task_evidence/approval-diff-check.log" 2>&1
