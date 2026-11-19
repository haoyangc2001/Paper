#!/usr/bin/env bash
set -euo pipefail
cd /home/caohy/repositories/Paper
task_python=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
task_evidence=.worklog-testing/NAV-043/affine-response-v2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
"$task_python" -B -m pytest -q -p no:cacheprovider CaMRA/navigation/tests/test_nav043_affine_audit.py --junitxml="$task_evidence/final-basic.xml" > "$task_evidence/final-basic.log" 2>&1
"$task_python" -B "$task_evidence/check_identities.py" > "$task_evidence/identities.log" 2>&1
printf '0\n' > "$task_evidence/final-check.exit"
