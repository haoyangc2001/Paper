#!/usr/bin/env bash
set -euo pipefail
cd /home/caohy/repositories/Paper
task_ev=.worklog-testing/NAV-043/coupled-execution-v4
test "$(cat "$task_ev/final-verify.exit")" = 0
git diff --check
git add -- AGENTS.md README.md CaMRA/navigation/README.md CaMRA/navigation/docs/implementation_handoff.md CaMRA/navigation/docs/nav043_coupled_response_result_20261120.md CaMRA/navigation/reports/nav043_coupled_response_review_v4.json doc/plan/20261109_camra_navigation_implementation.json
git add -f -- .worklog-testing/NAV-043/coupled-execution-v4 .worklog-testing/NAV-043/coupled-response-v4/launch.lock .worklog-testing/NAV-043/coupled-response-v4/launch.json .worklog-testing/NAV-043/coupled-response-v4/exit.json .worklog-testing/NAV-043/coupled-response-v4/run.log
git diff --cached --check
git diff --cached --stat
git commit -m 'NAV-043: retain audited response v4 results and failed diagnostic gates'
git status --short --ignore-submodules=all
git log -2 --oneline
