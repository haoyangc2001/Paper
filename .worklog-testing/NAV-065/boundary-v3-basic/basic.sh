#!/bin/bash
set -eu
cd /home/caohy/repositories/Paper
task_evidence=/home/caohy/repositories/Paper/.worklog-testing/NAV-065/boundary-v3-basic
task_python=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
hostname > "$task_evidence/host.log"
pwd >> "$task_evidence/host.log"
tmux display-message -p -t "$TMUX_PANE" '#{session_name}:#{window_name}' >> "$task_evidence/host.log"
"$task_python" --version >> "$task_evidence/host.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv >> "$task_evidence/host.log"
"$task_python" /home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py doc/plan/20261109_camra_navigation_implementation.json --mode update > "$task_evidence/validation-opening-attempt2.log" 2>&1
cd CaMRA/navigation
export PYTHONPATH=src:scripts
"$task_python" -m pytest -q -p no:cacheprovider tests/test_nav065_boundary_refit.py tests/test_nav065_saved_refit.py > "$task_evidence/tests.log" 2>&1
"$task_python" -u scripts/refit_nav065_boundary.py --preflight --report reports/nav065_boundary_refit_preflight_v3.json > "$task_evidence/preflight.log" 2>&1
date --iso-8601=seconds > "$task_evidence/BASIC_FINISHED"
