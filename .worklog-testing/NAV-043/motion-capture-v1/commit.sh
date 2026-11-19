#!/usr/bin/env bash
set -euo pipefail
cd /home/caohy/repositories/Paper
task_evidence=.worklog-testing/NAV-043/motion-capture-v1
test "$(cat "$task_evidence/basic-v3.exit")" = 0
test "$(cat "$task_evidence/verify.exit")" = 0
git diff --quiet --cached
git add -- AGENTS.md CaMRA/navigation/README.md CaMRA/navigation/docs/implementation_handoff.md CaMRA/navigation/docs/nav043_motion_probe_proposal_20261120.md doc/plan/20261109_camra_navigation_implementation.json CaMRA/navigation/src/camra_navigation/envs/motion_capture_v1.py CaMRA/navigation/tests/test_motion_capture.py .worklog-testing/NAV-043/motion-capture-v1
git add -f -- .worklog-testing/NAV-043/motion-capture-v1/basic.log .worklog-testing/NAV-043/motion-capture-v1/basic-final.log .worklog-testing/NAV-043/motion-capture-v1/basic-v3.log .worklog-testing/NAV-043/motion-capture-v1/verify.log .worklog-testing/NAV-043/motion-capture-v1/plan-update.log .worklog-testing/NAV-043/motion-capture-v1/diff-check.log
# Keep original raw diagnostics unchanged; check all source/docs and snapshots.
git diff --cached --check -- . ':(exclude).worklog-testing/NAV-043/motion-capture-v1/*.log' ':(exclude).worklog-testing/NAV-043/motion-capture-v1/*.xml'
git diff --cached --stat
git commit -m "NAV-043: prepare isolated motion capture core and verify CPU interfaces"
git status --short
git log -1 --format='%h %s'
