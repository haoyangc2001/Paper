"""Post-approval preflight only; no fitting or saved-command prediction."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
sys.path.insert(0, str(ROOT/'CaMRA/navigation/scripts'))
import nav043_matched_support_v5 as m
import nav043_motion_probe_recovery_guard as guard

m.context()
p = m.approved()
if Path(p['output_root']).exists() or (m.HERE/'launch.lock').exists():
    raise ValueError('already attempted; do not start another allocation')
identities = guard.protected_sources()
assert identities == dict(protected_identities=295, existing_raw_input_hashes=90, bundle_files=29)
source = m.base.load(m.base.NAV/'reports/nav043_coupled_response_review_v4.json')
for pid in (source['launch']['wrapper_pid'], source['launch']['child_pid']):
    if Path('/proc', str(pid)).exists():
        raise ValueError('prior diagnostic PID present; inspect identity before proceeding')
for item in source['output_index']:
    f = Path(item['path'])
    assert f.stat().st_size == item['bytes'] and m.base.sha(f) == item['sha256']
cases = m.diagonal.load_cases()
parts = m.partitions(cases)
assert len(parts) == 3
plan_path = ROOT/'doc/plan/20261109_camra_navigation_implementation.json'
plan = m.base.load(plan_path)
prior = json.loads(subprocess.check_output(['git', 'show', 'f187b6f:'+str(plan_path.relative_to(ROOT))],
    cwd=ROOT, text=True))
assert {k:v for k,v in prior.items() if k != 'tasks'} == {k:v for k,v in plan.items() if k != 'tasks'}
for a,b in zip(prior['tasks'], plan['tasks'], strict=True):
    assert {k:v for k,v in a.items() if k != 'work_records'} == {k:v for k,v in b.items() if k != 'work_records'}
    assert b['work_records'][:len(a['work_records'])] == a['work_records']
    if a['id'] != 'NAV-043':
        assert a == b
states = {t['id']:t['status'] for t in plan['tasks']}
assert sum(states.values()) == 55 and not states['NAV-043']
validation = subprocess.run([sys.executable,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',
    str(plan_path),'--mode','update'], cwd=ROOT, text=True, capture_output=True)
assert validation.returncode == 0, validation.stdout+validation.stderr
m.base.write_new(m.HERE/'preflight.json', dict(status='APPROVED_MATCHED_SUPPORT_PREFLIGHT_PASS',
    pane=os.environ['TMUX_PANE'], identities=identities, source_cases=len(cases),
    folds=3, structures=2, source_outputs_verified=83,
    proposal_sha256=m.base.sha(m.PROPOSAL), approval_sha256=m.base.sha(m.APPROVAL),
    readiness_sha256=m.base.sha(m.READINESS), extra_parameter_solves=0,
    plan_true=55, NAV043=False, validation=validation.stdout))
print('APPROVED_MATCHED_SUPPORT_PREFLIGHT_PASS', flush=True)
