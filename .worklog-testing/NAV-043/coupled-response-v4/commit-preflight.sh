#!/usr/bin/env bash
set -euo pipefail
cd /home/caohy/repositories/Paper
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B - <<'PY'
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, 'CaMRA/navigation/scripts')
import run_nav043_coupled_response as c
c.context()
ev = Path('.worklog-testing/NAV-043/coupled-response-v4')
proof = json.loads((ev/'preflight-preparation.json').read_text())
assert proof['status'] == 'PREPARED_ONLY_PREFLIGHT_REFUSAL_VERIFIED'
assert proof['actual_approved_branch_not_verified'] is True
for path, digest in proof['source_hashes'].items():
    assert c.base.sha(path) == digest, path
assert not c.APPROVAL.exists() and not (ev/'launch.lock').exists()
assert not Path(c.core.old.pin(c.PROPOSAL)['output_root']).exists()
plan_path = 'doc/plan/20261109_camra_navigation_implementation.json'
old = json.loads(subprocess.check_output(['git','show','79bd77a:'+plan_path], text=True))
now = c.base.load(plan_path)
assert {k:v for k,v in old.items() if k!='tasks'} == {k:v for k,v in now.items() if k!='tasks'}
for a,b in zip(old['tasks'],now['tasks'],strict=True):
    assert {k:v for k,v in a.items() if k!='work_records'} == {k:v for k,v in b.items() if k!='work_records'}
    assert b['work_records'][:len(a['work_records'])] == a['work_records']
    if a['id'] != 'NAV-043':
        assert a == b
assert len(next(t for t in now['tasks'] if t['id']=='NAV-043')['work_records']) == 64
assert sum(t['status'] for t in now['tasks']) == 55
print('Original62records/67definitions/source pins preserved; zero real or synthetic solves', flush=True)
PY
/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B /home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py doc/plan/20261109_camra_navigation_implementation.json --mode update
git diff --check
git add -- AGENTS.md CaMRA/navigation/docs/implementation_handoff.md CaMRA/navigation/docs/nav043_coupled_response_preparation_20261120.md doc/plan/20261109_camra_navigation_implementation.json .worklog-testing/NAV-043/coupled-response-v4/preflight.py .worklog-testing/NAV-043/coupled-response-v4/preflight.sh .worklog-testing/NAV-043/coupled-response-v4/prepare-preflight.py .worklog-testing/NAV-043/coupled-response-v4/preflight-preparation.json .worklog-testing/NAV-043/coupled-response-v4/commit-preflight.sh
git diff --cached --check
git diff --cached --stat
git commit -m 'NAV-043: prepare approval-gated read-only preflight without new fits'
git rev-parse HEAD
git status --short
