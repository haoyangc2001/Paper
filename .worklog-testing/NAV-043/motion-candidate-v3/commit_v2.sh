#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
cd /home/caohy/repositories/Paper
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
task_ev=.worklog-testing/NAV-043/motion-candidate-v3
test "$(cat "$task_ev/close.exit")" = 0
test "$(cat "$task_ev/basic.exit")" = 0
git diff --check
git status --short > "$task_ev/precommit-status-v2.log"
git add -- AGENTS.md README.md CaMRA/navigation/README.md CaMRA/navigation/docs/implementation_handoff.md doc/plan/20261109_camra_navigation_implementation.json CaMRA/navigation/scripts/nav043_supported_response.py CaMRA/navigation/scripts/launch_nav043_supported_response_once.py CaMRA/navigation/tests/test_supported_response_v3.py CaMRA/navigation/config/nav043_supported_response_proposal_v3.json CaMRA/navigation/config/nav043_supported_response_proposal_v3.json.sha256 CaMRA/navigation/config/nav043_supported_response_readiness_v3.json CaMRA/navigation/config/nav043_supported_response_readiness_v3.json.sha256 CaMRA/navigation/docs/nav043_supported_response_preparation_20261120.md .worklog-testing/NAV-043/motion-candidate-v3
git diff --cached --check
git diff --cached --name-only
if git diff --cached --name-only | rg '(^manuscripts/|^RC-PPO/|^HiRA/|\.(tex|bib)$)'; then
  exit 3
fi
git commit -m "NAV-043: prepare origin-preserving response candidate and finite CPU diagnostic"
git rev-parse HEAD
git status --short
