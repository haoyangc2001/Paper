#!/usr/bin/env bash
set -euo pipefail
cd /home/caohy/repositories/Paper
test "$(cat .worklog-testing/NAV-043/supported-execution-v3/preflight.exit)" = 0
git -c core.preloadIndex=false -c core.fsmonitor=false diff --check -- AGENTS.md README.md CaMRA/navigation/README.md CaMRA/navigation/docs/implementation_handoff.md doc/plan/20261109_camra_navigation_implementation.json
git -c core.preloadIndex=false -c core.fsmonitor=false add -- AGENTS.md README.md CaMRA/navigation/README.md CaMRA/navigation/docs/implementation_handoff.md doc/plan/20261109_camra_navigation_implementation.json CaMRA/navigation/config/nav043_supported_response_approval_v3.json CaMRA/navigation/config/nav043_supported_response_approval_v3.json.sha256
git -c core.preloadIndex=false -c core.fsmonitor=false add -f -- .worklog-testing/NAV-043/supported-execution-v3
git -c core.preloadIndex=false -c core.fsmonitor=false diff --cached --check
git -c core.preloadIndex=false -c core.fsmonitor=false diff --cached --stat
git -c core.preloadIndex=false -c core.fsmonitor=false commit -m 'NAV-043: register approved response-v3 scope and preflight before one CPU attempt'
git -c core.preloadIndex=false -c core.fsmonitor=false rev-parse HEAD
git -c core.preloadIndex=false -c core.fsmonitor=false status --short --ignore-submodules=all
