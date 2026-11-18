#!/bin/bash
set -eu
cd /home/caohy/repositories/Paper
/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B /home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py doc/plan/20261109_camra_navigation_implementation.json --mode update
git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --check
git diff --name-only -- 'manuscripts/**' '*.tex' '*.bib' 'docs/paper*' 'submission*'
git add -- AGENTS.md CaMRA/navigation/README.md CaMRA/navigation/docs/implementation_handoff.md doc/plan/20261109_camra_navigation_implementation.json CaMRA/navigation/reports/nav066_budget_decision_validation_v2.json CaMRA/navigation/docs/nav066_budget_decision_review_v2_20261118.md .worklog-testing/NAV-065/best-decision-v2/audit.py .worklog-testing/NAV-065/best-decision-v2/final-audit.json .worklog-testing/NAV-065/best-decision-v2/activate.py .worklog-testing/NAV-065/best-decision-v2/transition.py .worklog-testing/NAV-065/best-decision-v2/preflight.py .worklog-testing/NAV-065/best-decision-v2/write_review.py .worklog-testing/NAV-065/best-decision-v2/checkpoint.sh .worklog-testing/NAV-065/best-decision-v2/finalize.sh
git add -f -- .worklog-testing/NAV-065/best-decision-v2/audit.log .worklog-testing/NAV-065/best-decision-v2/audit-finished.log .worklog-testing/NAV-065/best-decision-v2/basic-tests.log .worklog-testing/NAV-065/best-decision-v2/evaluate.log .worklog-testing/NAV-065/best-decision-v2/decisions.log .worklog-testing/NAV-065/best-decision-v2/readiness.log .worklog-testing/NAV-065/best-decision-v2/checkpoint.log .worklog-testing/NAV-065/best-decision-v2/checkpoint-fixed.log
git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --cached --check
git diff --cached --stat
git commit -m "nav: record independent best-model budget decisions and efficiency failure"
git rev-parse HEAD
