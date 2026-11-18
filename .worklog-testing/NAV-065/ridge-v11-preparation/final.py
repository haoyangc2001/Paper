"""Check final append-only plan and evidence bindings; do not fit any model."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
HERE = Path(__file__).resolve().parent
NAV = ROOT / 'CaMRA/navigation'
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
read = lambda p: json.loads(p.read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
baseline, plan = read(HERE / 'baseline.json'), read(PLAN)
old = baseline['plan']
assert {k: v for k, v in plan.items() if k != 'tasks'} == {k: v for k, v in old.items() if k != 'tasks'}
assert len(plan['tasks']) == len(old['tasks']) == 67
for before, after in zip(old['tasks'], plan['tasks']):
    if before['id'] == 'NAV-065':
        assert {k: v for k, v in before.items() if k != 'work_records'} == {k: v for k, v in after.items() if k != 'work_records'}
        assert after['work_records'][:len(before['work_records'])] == before['work_records']
        assert len(after['work_records']) == len(before['work_records']) + 2
        for record in after['work_records'][-2:]:
            assert set(record) == {'timestamp', 'summary', 'actions', 'files_changed', 'commands', 'results', 'evidence', 'remaining_work', 'blockers'}
    else:
        assert before == after
assert sum(t['status'] for t in plan['tasks']) == 52
states = {t['id']: t['status'] for t in plan['tasks']}
current = next(t for t in plan['tasks'] if not t['status'] and all(states[i] for i in t['depends_on']))
assert current['id'] == 'NAV-065' and not states['NAV-066']
review_path = NAV / 'reports/nav065_ridge_probe_v11_preparation_review.json'
review = read(review_path)
assert review['preparation_pass'] and review['basic_tests_passed'] == 38
assert 'VALID' in (HERE / 'validation-after.log').read_text()
for name, digest in review['evidence'].items():
    assert sha(ROOT / name) == digest, name
for name, digest in review['bound_input_hashes'].items():
    assert sha(NAV / name) == digest, name
proposal = read(NAV / 'config/nav065_ridge_probe_proposal_v11.json')
assert not proposal['approved']
assert not (NAV / 'config/nav065_ridge_probe_prereg_v11.json').exists()
assert not (Path('/pub/data/caohy/work/camra_navigation') / proposal['output_name']).exists()
for path in ('AGENTS.md', 'CaMRA/navigation/README.md', 'CaMRA/navigation/docs/implementation_handoff.md'):
    assert 'v11' in (ROOT / path).read_text()
changed = subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT, text=True).splitlines()
assert sorted(changed) == sorted(['AGENTS.md', 'CaMRA/navigation/README.md', 'CaMRA/navigation/docs/implementation_handoff.md', 'doc/plan/20261109_camra_navigation_implementation.json'])
for name in ('.worklog-testing/NAV-065/ridge-v11-preparation/context.log', '.worklog-testing/NAV-065/ridge-v11-preparation/check-attempt1.log'):
    assert (ROOT / name).exists()
report = {
    'timestamp': datetime.now().astimezone().isoformat(),
    'host': socket.getfqdn(), 'tmux': os.environ['TMUX'], 'python': sys.executable,
    'verification_by': 'main agent using create-plan-worklog-legacy; no test subagent',
    'preparation_pass': True, 'legacy_validator': 'VALID', 'basic_tests_passed': 38,
    'old_plan_fields_tasks_and_records_preserved': True, 'new_NAV065_work_records': 2,
    'completed_tasks': 52, 'NAV065': False, 'NAV066_started': False,
    'formal_head_solves': 0, 'formal_optimizer_steps': 0, 'new_rollouts': 0, 'development_scoring_rounds': 0,
    'proposal_approved': False, 'new_head_solves_requested': 2,
    'evidence_hashes_current': True, 'preparation_review_sha256': sha(review_path),
    'plan_sha256': sha(PLAN), 'base_git': baseline['git'],
    'scientific_pass': False, 'original_G4': 'FAIL',
    'closing_evidence': {p.name: sha(p) for p in (HERE / 'close.py', HERE / 'close.log', HERE / 'validation-after.log', Path(__file__))}
}
with (HERE / 'final.json').open('x') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(json.dumps(report, ensure_ascii=False))
