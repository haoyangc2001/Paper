"""Read current plan and approval scope before this explicitly authorized attempt."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
sys.path.insert(0, str(ROOT/'CaMRA/navigation/scripts'))
import nav043_matched_support_v5 as m

m.context()
plan_path = ROOT/'doc/plan/20261109_camra_navigation_implementation.json'
plan = m.base.load(plan_path)
states = {t['id']:t['status'] for t in plan['tasks']}
eligible = [t for t in plan['tasks'] if not t['status'] and all(states[x] for x in t['depends_on'])]
assert eligible[0]['id'] == 'NAV-043' and sum(states.values()) == 55
task = eligible[0]
print('PLAN_POLICIES', json.dumps({k:plan[k] for k in plan if k != 'tasks' and
    (k in ('version','plan_version','goal','objective','global_constraints','uncertainty_policy') or k.endswith('_policy'))}, ensure_ascii=False))
print('CURRENT_TASK', json.dumps({k:v for k,v in task.items() if k != 'work_records'}, ensure_ascii=False))
for i, record in enumerate(task['work_records'], 1):
    print('PRIOR_RECORD', i, json.dumps(record, ensure_ascii=False))
assert len(task['work_records']) == 69
validation = subprocess.run([sys.executable, '-B', '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',
    str(plan_path), '--mode', 'update'], cwd=ROOT, text=True, capture_output=True)
assert validation.returncode == 0, validation.stdout+validation.stderr
p, ready = [m.diagonal.pin(x) for x in (m.PROPOSAL, m.READINESS)]
assert m.base.sha(m.PROPOSAL) == '71d5a31efe3890a713681d3e95600e87541563197e698c3a32446331a0e178d6'
assert m.base.sha(m.READINESS) == '364fe158b8bacedde45681137f8b4bb9e050560ead4ff5983a9962f8f1c0e982'
assert p['limits'] == m.LIMITS
assert not m.APPROVAL.exists() and not Path(p['output_root']).exists()
assert not (m.HERE/'launch.lock').exists() and not m.REVIEW.exists()
for mapping in (p['source_hashes'], ready['source_hashes']):
    for path, digest in mapping.items():
        assert m.base.sha(path) == digest, path
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() == 'e7766754c1d895ab236c99b1913f273d68def9c5'
dirty = subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
assert not dirty, dirty
m.base.write_new(m.HERE/'opening.json', dict(status='CURRENT_PLAN_AND_EXPLICIT_SCOPE_READ_PASS',
    pane=os.environ['TMUX_PANE'], plan_sha256=m.base.sha(plan_path), plan_version='1.5.0',
    task_id='NAV-043', prior_records=69, plan_true=55, NAV043=False,
    user_response='可以  继续推进', proposal_sha256=m.base.sha(m.PROPOSAL),
    readiness_sha256=m.base.sha(m.READINESS), source_hashes_rechecked=True,
    validation=validation.stdout, real_fits=0))
print('CURRENT_PLAN_AND_EXPLICIT_SCOPE_READ_PASS', flush=True)
