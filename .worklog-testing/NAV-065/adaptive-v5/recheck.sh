#!/usr/bin/env bash
set -eu
set -o noclobber
cd /home/caohy/repositories/Paper/CaMRA/navigation
export PYTHONPATH=src:scripts
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
TASK_LOG=/home/caohy/repositories/Paper/.worklog-testing/NAV-065/adaptive-v5
TASK_ATTEMPT=${1:?attempt number required}
/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -m pytest -q -p no:cacheprovider tests/test_nav065_adaptive_refit.py tests/test_nav065_guarded_refit.py tests/test_nav065_boundary_refit.py tests/test_nav065_saved_refit.py > "$TASK_LOG/tests-attempt$TASK_ATTEMPT.log" 2>&1
/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -u scripts/refit_nav065_adaptive.py --diagnose "reports/nav065_train_coverage_v5_attempt$TASK_ATTEMPT.json" > "$TASK_LOG/diagnosis-attempt$TASK_ATTEMPT.log" 2>&1
date -Iseconds > "$TASK_LOG/recheck-finished-attempt$TASK_ATTEMPT.txt"
