#!/usr/bin/env bash
set -euo pipefail
cd /home/caohy/repositories/Paper
task_ev=.worklog-testing/NAV-043/coupled-execution-v4
test "$(cat "$task_ev/preflight.exit")" = 0
git diff --check
git add -- AGENTS.md README.md CaMRA/navigation/README.md CaMRA/navigation/docs/implementation_handoff.md CaMRA/navigation/config/nav043_coupled_response_approval_v4.json CaMRA/navigation/config/nav043_coupled_response_approval_v4.json.sha256 doc/plan/20261109_camra_navigation_implementation.json .worklog-testing/NAV-043/coupled-response-v4/preflight.json .worklog-testing/NAV-043/coupled-response-v4/preflight.exit .worklog-testing/NAV-043/coupled-execution-v4
git add -f -- .worklog-testing/NAV-043/coupled-response-v4/approval-resume.log .worklog-testing/NAV-043/coupled-response-v4/preflight.log .worklog-testing/NAV-043/coupled-execution-v4/preflight.log
git diff --cached --stat
git commit -m 'NAV-043: approve and preflight one finite coupled response v4 diagnostic'
git rev-parse HEAD > "$task_ev/approval-commit.txt"
