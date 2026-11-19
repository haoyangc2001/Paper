#!/usr/bin/env bash
set -euo pipefail
cd /home/caohy/repositories/Paper
task_python=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
task_evidence=.worklog-testing/NAV-043/affine-response-v2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
test -e /pub/data/caohy/work/camra_navigation/nav043_affine_response_probe_v2/FINISHED.json
test ! -e /pub/data/caohy/work/camra_navigation/nav043_affine_response_probe_v2/HELD.json
test ! -e CaMRA/navigation/reports/nav043_affine_response_review_v2.json
test ! -e "$task_evidence/audit-saved.log"
if "$task_python" -B CaMRA/navigation/scripts/audit_nav043_affine_response.py > "$task_evidence/audit-saved.log" 2>&1; then
  printf '0\n' > "$task_evidence/audit-saved.exit"
else
  task_status=$?
  printf '%s\n' "$task_status" > "$task_evidence/audit-saved.exit"
  exit "$task_status"
fi
