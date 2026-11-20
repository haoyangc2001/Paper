#!/usr/bin/env bash
set -euo pipefail
cd /home/caohy/repositories/Paper
task_evidence=.worklog-testing/NAV-043/motion-launch-v1
test -s "$task_evidence/approval-preflight.json"
test -s "$task_evidence/approval-preflight.exit"
git diff --cached --name-only > /tmp/nav043-motion-probe-staged-before.txt
test ! -s /tmp/nav043-motion-probe-staged-before.txt
git add AGENTS.md CaMRA/navigation/docs/implementation_handoff.md doc/plan/20261109_camra_navigation_implementation.json CaMRA/navigation/config/nav043_motion_probe_approval_v1.json CaMRA/navigation/config/nav043_motion_probe_approval_v1.json.sha256
git add -f "$task_evidence/approval-preflight.py" "$task_evidence/approval-preflight.sh" "$task_evidence/approval-preflight.json" "$task_evidence/approval-preflight.log" "$task_evidence/approval-preflight.exit" "$task_evidence/approval-plan.log" "$task_evidence/approval-diff-check.log" "$task_evidence/launch-approved.sh" "$task_evidence/commit-approved.sh"
git diff --cached --check
git diff --cached --stat
git commit -m "NAV-043: register approved single motion capture allocation"
git rev-parse HEAD
