#!/usr/bin/env bash
set -euo pipefail
cd /home/caohy/repositories/Paper
test "$(cat .worklog-testing/NAV-043/supported-execution-v3/final_verify.exit)" = 0
git -c core.preloadIndex=false -c core.fsmonitor=false add -- AGENTS.md README.md CaMRA/navigation/README.md CaMRA/navigation/docs/implementation_handoff.md CaMRA/navigation/docs/nav043_supported_response_result_20261120.md CaMRA/navigation/reports/nav043_supported_response_review_v3.json doc/plan/20261109_camra_navigation_implementation.json
git -c core.preloadIndex=false -c core.fsmonitor=false add -f -- .worklog-testing/NAV-043/supported-execution-v3 .worklog-testing/NAV-043/motion-candidate-v3/launch.lock .worklog-testing/NAV-043/motion-candidate-v3/launch.json .worklog-testing/NAV-043/motion-candidate-v3/exit.json .worklog-testing/NAV-043/motion-candidate-v3/run.log
while IFS= read -r task_path; do
  case "$task_path" in
    AGENTS.md|README.md|CaMRA/navigation/README.md|CaMRA/navigation/docs/implementation_handoff.md|CaMRA/navigation/docs/nav043_supported_response_result_20261120.md|CaMRA/navigation/reports/nav043_supported_response_review_v3.json|doc/plan/20261109_camra_navigation_implementation.json|.worklog-testing/NAV-043/supported-execution-v3/*|.worklog-testing/NAV-043/motion-candidate-v3/launch.lock|.worklog-testing/NAV-043/motion-candidate-v3/launch.json|.worklog-testing/NAV-043/motion-candidate-v3/exit.json|.worklog-testing/NAV-043/motion-candidate-v3/run.log) ;;
    *) printf 'Unexpected staged file: %s\n' "$task_path"; exit 1 ;;
  esac
done < <(git -c core.preloadIndex=false -c core.fsmonitor=false diff --cached --name-only)
git -c core.preloadIndex=false -c core.fsmonitor=false diff --cached --check
git -c core.preloadIndex=false -c core.fsmonitor=false diff --cached --stat
git -c core.preloadIndex=false -c core.fsmonitor=false commit -m 'NAV-043: retain response-v3 diagnostic failure and verified saved outputs'
git -c core.preloadIndex=false -c core.fsmonitor=false rev-parse HEAD
git -c core.preloadIndex=false -c core.fsmonitor=false status --short --ignore-submodules=all
printf 'COMMIT_AND_STATUS_FINISHED\n'
