#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
cd /home/caohy/repositories/Paper
task_evidence=.worklog-testing/NAV-043/motion-recovery-v2
test "$(cat "$task_evidence/approval-preflight.exit")" = 0
test "$(cat "$task_evidence/approval-commit.exit")" = 1
git add -f -- "$task_evidence/approval-preflight.log" "$task_evidence/approval-plan.log" "$task_evidence/approval-diff-check.log" "$task_evidence/approval-commit.log"
git add -- AGENTS.md CaMRA/navigation/README.md CaMRA/navigation/docs/implementation_handoff.md CaMRA/navigation/docs/nav043_motion_probe_recovery_v2_20261120.md doc/plan/20261109_camra_navigation_implementation.json CaMRA/navigation/config/nav043_motion_probe_recovery_approval_v2.json CaMRA/navigation/config/nav043_motion_probe_recovery_approval_v2.json.sha256 "$task_evidence/approval-preflight.py" "$task_evidence/approval-preflight.sh" "$task_evidence/approval-preflight.json" "$task_evidence/approval-preflight.exit" "$task_evidence/approval-commit.sh" "$task_evidence/approval-commit.exit" "$task_evidence/approval-commit-v2.sh"
git diff --cached --check
git diff --cached --stat
git commit -m "NAV-043: approve isolated motion capture recovery after preflight"
git log -1 --format='%H %s'
git status --porcelain=v1
