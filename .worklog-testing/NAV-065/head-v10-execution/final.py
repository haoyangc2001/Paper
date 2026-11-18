"""Validate append-only history, approved bindings, and delivery scope."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
HERE = Path(__file__).resolve().parent
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
read = lambda p: json.loads(p.read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
before, after = read(HERE / 'baseline.json')['plan'], read(PLAN)
assert {k: v for k, v in before.items() if k != 'tasks'} == {k: v for k, v in after.items() if k != 'tasks'}
for old, current in zip(before['tasks'], after['tasks']):
    if old['id'] != 'NAV-065':
        assert old == current
        continue
    assert {k: v for k, v in old.items() if k != 'work_records'} == {k: v for k, v in current.items() if k != 'work_records'}
    assert current['work_records'][:len(old['work_records'])] == old['work_records']
    assert len(current['work_records']) == len(old['work_records']) + 2
    for row in current['work_records'][-2:]:
        assert set(row) == {'timestamp', 'summary', 'actions', 'files_changed', 'commands', 'results', 'evidence', 'remaining_work', 'blockers'}
assert len(after['tasks']) == 67 and sum(t['status'] for t in after['tasks']) == 52
review = read(NAV / 'reports/nav065_head_probe_v10_review.json')
assert review['integrity_pass'] and review['formal_candidate_head_solves'] == 2 and review['audit_head_solves'] == 0
assert not review['scientific_pass'] and not review['full_train_after']['pass'] and review['local_after']['local_pass']
for name, digest in review['evidence']['implementation_hashes'].items():
    assert sha(NAV / name) == digest
for name, digest in review['evidence']['output_hashes'].items():
    assert sha(Path(review['evidence']['output']) / name) == digest
with (HERE / 'validation-after.log').open('x') as f:
    result = subprocess.run([sys.executable, '-B', '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', str(PLAN), '--mode', 'update'], stdout=f, stderr=subprocess.STDOUT)
assert result.returncode == 0
subprocess.run(['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--check'], cwd=ROOT, check=True)
changes = subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT, text=True).splitlines()
assert set(changes) == {'AGENTS.md', 'CaMRA/navigation/README.md', 'CaMRA/navigation/docs/implementation_handoff.md', str(PLAN.relative_to(ROOT))}
state = {t['id']: t['status'] for t in after['tasks']}
assert next(t['id'] for t in after['tasks'] if not t['status'] and all(state[k] for k in t['depends_on'])) == 'NAV-065'
report = {'timestamp': datetime.now().astimezone().isoformat(), 'delivery_integrity_pass': True, 'legacy_validator_pass': True,
    'old_fields_states_records_preserved': True, 'appended_records': 2, 'formal_candidate_head_solves': 2,
    'audit_head_solves': 0, 'optimizer_steps': 0, 'new_rollouts': 0, 'development_scoring_rounds': 0,
    'local_pass': True, 'full_train_pass': False, 'scientific_pass': False, 'NAV065': False,
    'original_G4': 'FAIL', 'completed_tasks': 52, 'NAV066_started': False,
    'changed_tracked_files': changes, 'evidence_hashes': {str(p.relative_to(ROOT)): sha(p) for p in (
        NAV / 'reports/nav065_head_probe_v10_review.json', NAV / 'docs/nav065_head_probe_v10_review_20261118.md',
        HERE / 'audit.py', HERE / 'audit-attempt1.log', HERE / 'execution.json')}}
with (HERE / 'final.json').open('x') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps(report, ensure_ascii=False), flush=True)
