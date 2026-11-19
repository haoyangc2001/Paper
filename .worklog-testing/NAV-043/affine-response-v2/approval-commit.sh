#!/usr/bin/env bash
set -euo pipefail
cd /home/caohy/repositories/Paper
task_evidence=.worklog-testing/NAV-043/affine-response-v2
test "$(cat "$task_evidence/approval-preflight.exit")" = 0
test ! -e "$task_evidence/launch.lock"
test ! -e /pub/data/caohy/work/camra_navigation/nav043_affine_response_probe_v2
git diff --cached --quiet
git add -- AGENTS.md CaMRA/navigation/docs/implementation_handoff.md doc/plan/20261109_camra_navigation_implementation.json CaMRA/navigation/config/nav043_affine_response_approval_v2.json CaMRA/navigation/config/nav043_affine_response_approval_v2.json.sha256 "$task_evidence/approval-preflight.py" "$task_evidence/approval-preflight.sh" "$task_evidence/approval-preflight.json" "$task_evidence/approval-preflight.exit" "$task_evidence/approval-commit.sh"
git add -f -- "$task_evidence/approval-preflight.log" "$task_evidence/approval-plan.log" "$task_evidence/approval-diff-check.log"
git diff --cached --check
git commit -m "docs(nav043): register approved one-shot affine diagnostic"
git rev-parse HEAD
git status --short --untracked-files=normal
printf '0\n' > "$task_evidence/approval-commit.exit"
