#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
cd /home/caohy/repositories/Paper
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_ev=.worklog-testing/NAV-043/supported-audit-v3
test "$(cat "$task_ev/blocked-audit.exit")" = 0
/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B - <<'PY' > "$task_ev/blocked-history.log" 2>&1
import json
from pathlib import Path
import subprocess
path='doc/plan/20261109_camra_navigation_implementation.json'
old=json.loads(subprocess.check_output(['git','show','279a785:'+path],text=True))
now=json.loads(Path(path).read_text())
for a,b in zip(old['tasks'],now['tasks'],strict=True):
    assert {k:v for k,v in a.items() if k!='work_records'}=={k:v for k,v in b.items() if k!='work_records'}
    assert b['work_records'][:len(a['work_records'])]==a['work_records']
    if a['id']!='NAV-043':assert a==b
assert {k:v for k,v in old.items() if k!='tasks'}=={k:v for k,v in now.items() if k!='tasks'}
assert sum(t['status'] for t in now['tasks'])==55
assert len(next(t for t in now['tasks'] if t['id']=='NAV-043')['work_records'])==58
print('57 original NAV-043 records and all 67 definitions/statuses unchanged; one blocking record appended')
PY
/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B /home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py doc/plan/20261109_camra_navigation_implementation.json --mode update > "$task_ev/blocked-plan-validation.log" 2>&1
git -c core.preloadIndex=false -c diff.ignoreSubmodules=all diff --check -- doc/plan/20261109_camra_navigation_implementation.json
git -c core.preloadIndex=false add -- doc/plan/20261109_camra_navigation_implementation.json
git -c core.preloadIndex=false add -f -- "$task_ev"/blocked*
git -c core.preloadIndex=false -c diff.ignoreSubmodules=all commit --no-status -m 'NAV-043: record repeated diagnostic scope approval blocker' > /tmp/nav043_supported_fit_blocked_commit.log 2>&1
git -c core.preloadIndex=false status --short --ignore-submodules=all > /tmp/nav043_supported_fit_blocked_status.log
git rev-parse HEAD > /tmp/nav043_supported_fit_blocked_head.log
