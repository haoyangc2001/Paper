#!/usr/bin/env bash
set -euo pipefail
cd /home/caohy/repositories/Paper
task_evidence=.worklog-testing/NAV-043/motion-coverage-v1
test "$(cat "$task_evidence/final-review.exit")" = 0
test "$(cat "$task_evidence/data-review.exit")" = 0
git diff --quiet --cached
git add -- AGENTS.md CaMRA/navigation/README.md CaMRA/navigation/docs/implementation_handoff.md doc/plan/20261109_camra_navigation_implementation.json CaMRA/navigation/reports/nav043_motion_data_availability_v1.json CaMRA/navigation/config/nav043_motion_probe_proposal_v1.json CaMRA/navigation/config/nav043_motion_probe_proposal_v1.json.sha256 CaMRA/navigation/scripts/check_nav043_motion_probe_proposal.py CaMRA/navigation/docs/nav043_motion_probe_proposal_20261120.md .worklog-testing/NAV-043/motion-coverage-v1
git diff --cached --check
git diff --cached --stat
git commit -m "NAV-043: review saved low-speed data and propose finite response collection"
git status --short
git log -1 --format='%h %s'
