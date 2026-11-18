#!/bin/bash
set -eu
cd /home/caohy/repositories/Paper
/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B /home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py doc/plan/20261109_camra_navigation_implementation.json --mode update
git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --check
git add -- AGENTS.md CaMRA/navigation/README.md CaMRA/navigation/docs/implementation_handoff.md doc/plan/20261109_camra_navigation_implementation.json CaMRA/navigation/config/nav066_decision_prereg_v2.json CaMRA/navigation/config/nav066_decision_prereg_v2.json.sha256 CaMRA/navigation/reports/nav065_best_existing_model_binding_v2.json CaMRA/navigation/reports/nav065_budget_decision_readiness_v2.json CaMRA/navigation/scripts/run_nav066_decision_validation.py CaMRA/navigation/src/camra_navigation/budget/guarded.py CaMRA/navigation/tests/test_guarded_budget.py
git diff --cached --stat
git commit -m "nav: freeze best existing models and independent budget decision protocol"
git rev-parse HEAD
