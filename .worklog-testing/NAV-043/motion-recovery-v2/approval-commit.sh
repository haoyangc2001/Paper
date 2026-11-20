#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
cd /home/caohy/repositories/Paper
task_python=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
task_evidence=.worklog-testing/NAV-043/motion-recovery-v2
test "$(cat "$task_evidence/approval-preflight.exit")" = 0
"$task_python" -B /home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py doc/plan/20261109_camra_navigation_implementation.json --mode update
git diff --check
git add -- AGENTS.md CaMRA/navigation/README.md CaMRA/navigation/docs/implementation_handoff.md CaMRA/navigation/docs/nav043_motion_probe_recovery_v2_20261120.md doc/plan/20261109_camra_navigation_implementation.json CaMRA/navigation/config/nav043_motion_probe_recovery_approval_v2.json CaMRA/navigation/config/nav043_motion_probe_recovery_approval_v2.json.sha256 .worklog-testing/NAV-043/motion-recovery-v2/approval-preflight.py .worklog-testing/NAV-043/motion-recovery-v2/approval-preflight.sh .worklog-testing/NAV-043/motion-recovery-v2/approval-preflight.json .worklog-testing/NAV-043/motion-recovery-v2/approval-preflight.log .worklog-testing/NAV-043/motion-recovery-v2/approval-preflight.exit .worklog-testing/NAV-043/motion-recovery-v2/approval-plan.log .worklog-testing/NAV-043/motion-recovery-v2/approval-diff-check.log .worklog-testing/NAV-043/motion-recovery-v2/approval-commit.sh
git diff --cached --check
git diff --cached --stat
git commit -m "NAV-043: approve isolated motion capture recovery after preflight"
git log -1 --format='%H %s'
git status --porcelain=v1
