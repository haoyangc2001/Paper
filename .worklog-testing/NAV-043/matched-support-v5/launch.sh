#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
cd /home/caohy/repositories/Paper
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/caohy/repositories/Paper/CaMRA/navigation/scripts
task_ev=.worklog-testing/NAV-043/matched-support-v5
/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B -c 'import nav043_matched_support_v5 as m; m.context(); m.approved()'
if /pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B -u CaMRA/navigation/scripts/nav043_matched_support_v5.py > "$task_ev/wrapper.log" 2>&1; then
  printf '0\n' > "$task_ev/wrapper.exit"
else
  task_status=$?
  printf '%s\n' "$task_status" > "$task_ev/wrapper.exit"
  exit "$task_status"
fi
