#!/usr/bin/env bash
set -euo pipefail
cd /home/caohy/repositories/Paper
task_python=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
task_evidence=.worklog-testing/NAV-043/motion-state-v1
export PYTHONPATH=/home/caohy/repositories/Paper/CaMRA/navigation/src:/home/caohy/repositories/Paper/CaMRA/calibration/src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
test ! -e "$task_evidence/basic-v3.exit"
test ! -e CaMRA/navigation/reports/nav043_affine_support_review_v1.json
if "$task_python" -B -m pytest -q -p no:cacheprovider CaMRA/navigation/tests/test_motion_state.py --junitxml="$task_evidence/basic-v3.xml" > "$task_evidence/basic-v3.log" 2>&1; then
  printf '0\n' > "$task_evidence/basic-v3.exit"
else
  task_status=$?
  printf '%s\n' "$task_status" > "$task_evidence/basic-v3.exit"
  exit "$task_status"
fi
if "$task_python" -B "$task_evidence/review-affine-support.py" > "$task_evidence/affine-support.log" 2>&1; then
  printf '0\n' > "$task_evidence/support.exit"
else
  task_status=$?
  printf '%s\n' "$task_status" > "$task_evidence/support.exit"
  exit "$task_status"
fi
