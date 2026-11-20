#!/usr/bin/env bash
set -euo pipefail
cd /home/caohy/repositories/Paper
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_ev=.worklog-testing/NAV-043/coupled-execution-v4
test "$(cat "$task_ev/preflight.exit")" = 0
/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B -c 'import json; from pathlib import Path; p=json.loads(Path(".worklog-testing/NAV-043/coupled-response-v4/preflight.json").read_text()); assert p["status"]=="PASS_BEFORE_ONE_APPROVED_V4_ATTEMPT"; assert not Path(".worklog-testing/NAV-043/coupled-response-v4/launch.lock").exists()'
/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B /home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py doc/plan/20261109_camra_navigation_implementation.json --mode update
git diff --check
git add -- AGENTS.md README.md CaMRA/navigation/README.md CaMRA/navigation/docs/implementation_handoff.md CaMRA/navigation/config/nav043_coupled_response_approval_v4.json CaMRA/navigation/config/nav043_coupled_response_approval_v4.json.sha256 doc/plan/20261109_camra_navigation_implementation.json .worklog-testing/NAV-043/coupled-response-v4/approval-resume.log .worklog-testing/NAV-043/coupled-response-v4/preflight.json .worklog-testing/NAV-043/coupled-response-v4/preflight.log .worklog-testing/NAV-043/coupled-response-v4/preflight.exit .worklog-testing/NAV-043/coupled-execution-v4
git diff --cached --stat
git commit -m 'NAV-043: approve and preflight one finite coupled response v4 diagnostic'
git rev-parse HEAD > "$task_ev/approval-commit.txt"
