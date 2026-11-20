#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
cd /home/caohy/repositories/Paper
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
task_python=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
"$task_python" -B /home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py doc/plan/20261109_camra_navigation_implementation.json --mode update
"$task_python" -B -u .worklog-testing/NAV-043/motion-recovery-v2/commit-close.py
git commit -m 'NAV-043: preserve 12 verified motion cases and retained closure failure'
git log -1 --format='%H %s'
git status --short
