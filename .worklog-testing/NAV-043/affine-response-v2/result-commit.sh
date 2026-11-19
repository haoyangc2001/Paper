#!/usr/bin/env bash
set -euo pipefail
cd /home/caohy/repositories/Paper
task_evidence=.worklog-testing/NAV-043/affine-response-v2
test "$(cat "$task_evidence/close-result.exit")" = 0
git diff --cached --quiet
git add -- AGENTS.md CaMRA/navigation/README.md CaMRA/navigation/docs/implementation_handoff.md CaMRA/navigation/docs/nav043_response_model_v2_plan_20261120.md CaMRA/navigation/docs/nav043_affine_response_result_20261120.md CaMRA/navigation/reports/nav043_affine_response_review_v2.json doc/plan/20261109_camra_navigation_implementation.json
git add -f -- "$task_evidence"
git diff --cached --check
git diff --cached --stat
git commit -m "docs(nav043): preserve passing affine diagnostic and saved-output audit"
git rev-parse HEAD
git status --short --untracked-files=normal
printf 'RESULT_COMMIT_COMPLETE\n'
