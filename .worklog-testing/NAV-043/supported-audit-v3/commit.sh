#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
cd /home/caohy/repositories/Paper
task_ev=.worklog-testing/NAV-043/supported-audit-v3
test "$(cat "$task_ev/close.exit")" = 0
test "$(cat "$task_ev/verify.exit")" = 0
git diff --check
git add -- AGENTS.md README.md CaMRA/navigation/README.md CaMRA/navigation/docs/implementation_handoff.md CaMRA/navigation/docs/nav043_supported_response_preparation_20261120.md CaMRA/navigation/scripts/audit_nav043_supported_response.py CaMRA/navigation/tests/test_supported_response_audit_v3.py doc/plan/20261109_camra_navigation_implementation.json .worklog-testing/NAV-043/supported-audit-v3
if git commit -m 'NAV-043: prepare read-only response-v3 result verification' > "$task_ev/commit.log" 2>&1; then
  printf '0\n' > "$task_ev/commit.exit"
else
  task_status=$?
  printf '%s\n' "$task_status" > "$task_ev/commit.exit"
  exit "$task_status"
fi
git rev-parse HEAD > "$task_ev/committed-head.log"
git status --short > "$task_ev/post-commit-status.log"
