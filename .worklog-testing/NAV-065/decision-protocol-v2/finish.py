"""Validate final append-only documents and create the authorized local commit."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
HERE = Path(__file__).resolve().parent
PYTHON = '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == PYTHON
PLAN = 'doc/plan/20261109_camra_navigation_implementation.json'
before = json.loads((HERE / 'baseline.json').read_bytes())['plan']
after = json.loads((ROOT / PLAN).read_bytes())
assert {k:v for k,v in before.items() if k != 'tasks'} == {k:v for k,v in after.items() if k != 'tasks'}
for old, new in zip(before['tasks'], after['tasks'], strict=True):
    if old['id'] == 'NAV-065':
        assert {k:v for k,v in old.items() if k != 'work_records'} == {k:v for k,v in new.items() if k != 'work_records'}
        n = len(old['work_records'])
        assert new['work_records'][:n] == old['work_records']
        assert len(new['work_records']) == n+2
        required = set(before['document']['work_record_fields'])
        assert all(set(r) == required for r in new['work_records'][n:])
    else:
        assert old == new
protocol = ROOT / 'CaMRA/navigation/config/nav065_budget_decision_protocol_v2.json'
report = json.loads((ROOT / 'CaMRA/navigation/reports/nav065_budget_decision_protocol_v2_preparation_review.json').read_bytes())
assert report['preparation_pass'] and report['check_count'] == 272
assert report['protocol_sha256'] == hashlib.sha256(protocol.read_bytes()).hexdigest()
files = ['AGENTS.md', 'CaMRA/navigation/README.md', 'CaMRA/navigation/config/nav065_budget_decision_protocol_v2.json', 'CaMRA/navigation/docs/nav065_budget_decision_protocol_v2_20261118.md', 'CaMRA/navigation/docs/implementation_handoff.md', 'CaMRA/navigation/reports/nav065_budget_decision_protocol_v2_preparation_review.json', PLAN]
with (HERE / 'validation-final.log').open('x') as log:
    subprocess.run([PYTHON, '-B', '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', str(ROOT / PLAN), '--mode', 'update'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
assert not git('diff', '--cached', '--name-only'), 'Unrelated staged work must be preserved'
assert set(git('diff', '--name-only').splitlines()) <= set(files)
subprocess.run(['git', '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--check'], cwd=ROOT, check=True)
assert all((ROOT / f).is_file() for f in files)
subprocess.run(['git', 'add', '--', *files], cwd=ROOT, check=True)
assert set(git('diff', '--cached', '--name-only').splitlines()) == set(files)
subprocess.run(['git', '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check'], cwd=ROOT, check=True)
subprocess.run(['git', 'commit', '-m', 'docs(nav): draft decision-level budget acceptance protocol'], cwd=ROOT, check=True)
result = {'commit': git('rev-parse', 'HEAD'), 'files': files, 'plan_append_records': 2, 'completed_tasks': sum(t['status'] for t in after['tasks']), 'plan_validation': 'PASS', 'preparation_checks': 272, 'protocol_activated': False, 'new_episodes': 0, 'optimizer_updates': 0, 'production_rule_integrated': False, 'push_performed': False, 'remaining_tracked_diff': git('diff', '--name-only')}
with (HERE / 'commit.json').open('x') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(json.dumps(result, ensure_ascii=False, indent=2))
