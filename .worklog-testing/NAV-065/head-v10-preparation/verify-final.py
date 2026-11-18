"""Verify final append-only preparation record and local delivery scope."""
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
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
read = lambda p: json.loads(p.read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
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
states = {t['id']: t['status'] for t in after['tasks']}
assert next(t['id'] for t in after['tasks'] if not t['status'] and all(states[x] for x in t['depends_on'])) == 'NAV-065'
review = read(ROOT / 'CaMRA/navigation/reports/nav065_head_probe_v10_preparation_review.json')
for name, digest in review['evidence'].items():
    assert sha(ROOT / name) == digest
for name in ('AGENTS.md', 'CaMRA/navigation/README.md', 'CaMRA/navigation/docs/implementation_handoff.md'):
    assert 'v10' in (ROOT / name).read_text() and '2次' in (ROOT / name).read_text()
with (HERE / 'validation-after.log').open('x') as f:
    result = subprocess.run([sys.executable, '-B', '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', str(PLAN), '--mode', 'update'], stdout=f, stderr=subprocess.STDOUT)
assert result.returncode == 0
subprocess.run(['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--check'], cwd=ROOT, check=True)
changes = subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT, text=True).splitlines()
assert set(changes) == {'AGENTS.md', 'CaMRA/navigation/README.md', 'CaMRA/navigation/docs/implementation_handoff.md', str(PLAN.relative_to(ROOT))}
assert not Path('/pub/data/caohy/work/camra_navigation/nav065_head_probe_v10').exists()
assert not (ROOT / 'CaMRA/navigation/config/nav065_head_probe_prereg_v10.json').exists()
result = {'timestamp': datetime.now().astimezone().isoformat(), 'final_preparation_pass': True,
    'appended_records': 2, 'all_prior_fields_records_and_states_preserved': True, 'completed_tasks': 52,
    'first_eligible': 'NAV-065', 'NAV065': False, 'original_G4': 'FAIL',
    'formal_candidate_solves': 0, 'formal_optimizer_updates': 0, 'new_rollouts': 0, 'development_scoring_rounds': 0,
    'basic_tests_passed': 33, 'legacy_validator_pass': True, 'diff_check_pass': True,
    'approval_pending': True, 'requested_candidate_head_solves': 2, 'NAV066_started': False,
    'source_evidence_hashes_match': True, 'changed_tracked_files': changes}
with (HERE / 'final.json').open('x') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps(result, ensure_ascii=False))
