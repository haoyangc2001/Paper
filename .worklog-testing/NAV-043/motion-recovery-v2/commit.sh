#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
cd /home/caohy/repositories/Paper
task_python=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
task_evidence=.worklog-testing/NAV-043/motion-recovery-v2
export PYTHONPATH=/home/caohy/repositories/Paper/CaMRA/navigation/src:/home/caohy/repositories/Paper/CaMRA/calibration/src
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
test -z "$(git diff --cached --name-only)"
"$task_python" -B "$task_evidence/final-check.py" > "$task_evidence/final-check.log" 2>&1
"$task_python" -B /home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py doc/plan/20261109_camra_navigation_implementation.json --mode update > "$task_evidence/final-plan.log" 2>&1
git diff --check
git add -- AGENTS.md CaMRA/navigation/README.md CaMRA/navigation/docs/implementation_handoff.md doc/plan/20261109_camra_navigation_implementation.json
git add -- CaMRA/navigation/docs/nav043_motion_probe_first_failure_20261120.md CaMRA/navigation/docs/nav043_motion_probe_recovery_v2_20261120.md CaMRA/navigation/reports/nav043_motion_probe_first_failure_v1.json CaMRA/navigation/reports/nav043_motion_probe_recovery_preparation_v2.json
git add -- CaMRA/navigation/config/nav043_motion_probe_recovery_proposal_v2.json CaMRA/navigation/config/nav043_motion_probe_recovery_proposal_v2.json.sha256 CaMRA/navigation/config/nav043_motion_probe_recovery_readiness_v2.json CaMRA/navigation/config/nav043_motion_probe_recovery_readiness_v2.json.sha256
git add -- CaMRA/navigation/src/camra_navigation/envs/simulation_clock_v2.py CaMRA/navigation/src/camra_navigation/envs/motion_capture_v2.py CaMRA/navigation/scripts/nav043_process_cleanup_v2.py CaMRA/navigation/scripts/nav043_motion_probe_recovery_guard.py CaMRA/navigation/scripts/nav043_motion_probe_recovery_simulation.py CaMRA/navigation/scripts/run_nav043_motion_probe_recovery_v2.py CaMRA/navigation/scripts/audit_nav043_motion_probe_recovery_v2.py CaMRA/navigation/tests/test_motion_recovery_v2.py
git add -f -- .worklog-testing/NAV-043/motion-launch-v1/exit.json .worklog-testing/NAV-043/motion-launch-v1/first-failure-review.exit .worklog-testing/NAV-043/motion-launch-v1/first-failure-review.log .worklog-testing/NAV-043/motion-launch-v1/launch.exit .worklog-testing/NAV-043/motion-launch-v1/launch.lock .worklog-testing/NAV-043/motion-launch-v1/launch.log .worklog-testing/NAV-043/motion-launch-v1/review-first-failure.py .worklog-testing/NAV-043/motion-launch-v1/review-first-failure.sh .worklog-testing/NAV-043/motion-launch-v1/stop-startup.py .worklog-testing/NAV-043/motion-launch-v1/startup-stop.log
git add -f -- "$task_evidence/basic.sh" "$task_evidence/basic.log" "$task_evidence/basic.xml" "$task_evidence/basic.exit" "$task_evidence/prepare-documents.py" "$task_evidence/launch-once.sh" "$task_evidence/verify.py" "$task_evidence/verify.sh" "$task_evidence/verify.log" "$task_evidence/verify.exit" "$task_evidence/plan-update.log" "$task_evidence/diff-check.log" "$task_evidence/final-check.py" "$task_evidence/final-check.log" "$task_evidence/final-plan.log" "$task_evidence/commit.sh"
git diff --cached --check
git diff --cached --stat
git commit -m 'NAV-043: retain failed capture and prepare isolated clock recovery'
git rev-parse HEAD
git status --short --untracked-files=all
