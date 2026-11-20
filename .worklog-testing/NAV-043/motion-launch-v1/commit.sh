#!/usr/bin/env bash
set -euo pipefail
cd /home/caohy/repositories/Paper
task_evidence=.worklog-testing/NAV-043/motion-launch-v1
test "$(cat "$task_evidence/basic-v3.exit")" = 0
test "$(cat "$task_evidence/verify.exit")" = 0
git diff --quiet --cached
git add -- AGENTS.md CaMRA/navigation/README.md CaMRA/navigation/docs/implementation_handoff.md CaMRA/navigation/docs/nav043_motion_probe_proposal_20261120.md doc/plan/20261109_camra_navigation_implementation.json CaMRA/navigation/config/nav043_motion_probe_readiness_v1.json CaMRA/navigation/config/nav043_motion_probe_readiness_v1.json.sha256 CaMRA/navigation/scripts/nav043_motion_probe_guard.py CaMRA/navigation/scripts/nav043_motion_probe_simulation.py CaMRA/navigation/scripts/run_nav043_motion_probe.py CaMRA/navigation/scripts/audit_nav043_motion_probe.py CaMRA/navigation/tests/test_nav043_motion_launch.py .worklog-testing/NAV-043/motion-launch-v1
git add -f -- .worklog-testing/NAV-043/motion-launch-v1/basic.log .worklog-testing/NAV-043/motion-launch-v1/basic-final.log .worklog-testing/NAV-043/motion-launch-v1/basic-v3.log .worklog-testing/NAV-043/motion-launch-v1/verify.log .worklog-testing/NAV-043/motion-launch-v1/plan-update.log .worklog-testing/NAV-043/motion-launch-v1/diff-check.log
# Preserve raw diagnostics byte for byte, check all implementation/docs/snapshots.
git diff --cached --check -- . ':(exclude).worklog-testing/NAV-043/motion-launch-v1/*.log' ':(exclude).worklog-testing/NAV-043/motion-launch-v1/*.xml'
git diff --cached --stat
git commit -m "NAV-043: prepare bounded motion probe launcher and saved-data review"
git status --short
git log -1 --format='%h %s'
