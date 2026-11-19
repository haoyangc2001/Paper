#!/usr/bin/env bash
set -euo pipefail
cd /home/caohy/repositories/Paper
task_python=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
task_evidence=.worklog-testing/NAV-043/affine-response-v2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
"$task_python" -B "$task_evidence/seal_support.py" > "$task_evidence/seal.log" 2>&1
set +e
"$task_python" -B CaMRA/navigation/scripts/launch_nav043_affine_response_once.py > "$task_evidence/launcher-refusal.log" 2>&1
task_launcher_code=$?
"$task_python" -B CaMRA/navigation/scripts/audit_nav043_affine_response.py > "$task_evidence/auditor-refusal.log" 2>&1
task_auditor_code=$?
set -e
test "$task_launcher_code" -eq 1
test "$task_auditor_code" -eq 1
test ! -e "$task_evidence/launch.lock"
test ! -e /pub/data/caohy/work/camra_navigation/nav043_affine_response_probe_v2
test ! -e CaMRA/navigation/reports/nav043_affine_response_review_v2.json
printf 'launcher=1 auditor=1 no marker/output/report\n' > "$task_evidence/refusal.exit"
