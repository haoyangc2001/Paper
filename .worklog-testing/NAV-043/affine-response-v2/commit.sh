#!/usr/bin/env bash
set -euo pipefail
cd /home/caohy/repositories/Paper
test "$(cat .worklog-testing/NAV-043/affine-response-v2/close.exit)" = 0
git diff --cached --quiet
git add -- AGENTS.md CaMRA/navigation/README.md CaMRA/navigation/docs/implementation_handoff.md CaMRA/navigation/docs/nav043_response_model_v2_plan_20261120.md doc/plan/20261109_camra_navigation_implementation.json CaMRA/navigation/scripts/audit_nav043_affine_response.py CaMRA/navigation/scripts/launch_nav043_affine_response_once.py CaMRA/navigation/tests/test_nav043_affine_audit.py CaMRA/navigation/config/nav043_affine_response_readiness_v2.json CaMRA/navigation/config/nav043_affine_response_readiness_v2.json.sha256
git add -f -- .worklog-testing/NAV-043/affine-response-v2
git diff --cached --check
git diff --cached --stat
git commit -m "chore(nav): prepare one-shot affine diagnostic and saved-output audit"
git log -1 --oneline
