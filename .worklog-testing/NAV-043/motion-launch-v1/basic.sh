#!/usr/bin/env bash
set -euo pipefail
cd /home/caohy/repositories/Paper
task_python=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
task_evidence=.worklog-testing/NAV-043/motion-launch-v1
export PYTHONPATH=/home/caohy/repositories/Paper/CaMRA/navigation/src:/home/caohy/repositories/Paper/CaMRA/calibration/src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
test ! -e "$task_evidence/basic.exit"
if "$task_python" -B -m pytest -q -p no:cacheprovider CaMRA/navigation/tests/test_nav043_motion_launch.py CaMRA/navigation/tests/test_motion_capture.py --junitxml="$task_evidence/basic.xml" > "$task_evidence/basic.log" 2>&1; then
  printf '0\n' > "$task_evidence/basic.exit"
else
  task_status=$?
  printf '%s\n' "$task_status" > "$task_evidence/basic.exit"
  exit "$task_status"
fi
