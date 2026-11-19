#!/usr/bin/env bash
set -eu
cd /home/caohy/repositories/Paper
nav_python=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
nav_evidence=.worklog-testing/NAV-043/response-residual-v1
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
hostname > "$nav_evidence/host.log"
"$nav_python" -B /home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py doc/plan/20261109_camra_navigation_implementation.json --mode update > "$nav_evidence/preflight-plan.log" 2>&1
"$nav_python" -B -m pytest -q -p no:cacheprovider CaMRA/navigation/tests/test_nav043_response_residuals.py --junitxml="$nav_evidence/basic.xml" > "$nav_evidence/basic.log" 2>&1
"$nav_python" -B CaMRA/navigation/scripts/analyze_nav043_response_residuals.py > "$nav_evidence/analysis.log" 2>&1
echo 0 > "$nav_evidence/check.exit"
