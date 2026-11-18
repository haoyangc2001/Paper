"""Bind the final basic tests, source snapshots, failed attempts and plan history."""
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
OUT = ROOT / '.worklog-testing/NAV-065/adaptive-v5'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
env = {**os.environ, 'PYTHONPATH': 'src:scripts', 'PYTHONDONTWRITEBYTECODE': '1',
       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
files = [p for folder in ('src', 'scripts', 'config', 'tests') for p in (NAV / folder).rglob('*')
         if p.is_file() and '__pycache__' not in p.parts]
before = {str(p.relative_to(NAV)): sha(p) for p in files}
snapshot = OUT / 'tested_source'
snapshot.mkdir()
for name in ('scripts/refit_nav065_adaptive.py', 'tests/test_nav065_adaptive_refit.py',
             'config/nav065_adaptive_refit_proposal_v5.json'):
    path = snapshot / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write((NAV / name).read_bytes())
commands = []
tests = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
         'tests/test_nav065_adaptive_refit.py', 'tests/test_nav065_guarded_refit.py',
         'tests/test_nav065_boundary_refit.py', 'tests/test_nav065_saved_refit.py']
with (OUT / 'tests-final.log').open('x') as log:
    result = subprocess.run(tests, cwd=NAV, env=env, stdout=log, stderr=subprocess.STDOUT)
commands.append({'command': tests, 'cwd': str(NAV), 'returncode': result.returncode})
assert result.returncode == 0
work = Path('/pub/data/caohy/work/camra_navigation/nav065_adaptive_refit_v5')
assert not work.exists()
refusal = [sys.executable, '-u', 'scripts/refit_nav065_adaptive.py', '--registration',
           'config/nav065_adaptive_refit_proposal_v5.json']
with (OUT / 'unapproved-refusal.log').open('x') as log:
    result = subprocess.run(refusal, cwd=NAV, env=env, stdout=log, stderr=subprocess.STDOUT)
commands.append({'command': refusal, 'cwd': str(NAV), 'returncode': result.returncode})
assert result.returncode != 0 and not work.exists()
assert 'require explicit approval' in (OUT / 'unapproved-refusal.log').read_text()
after = {str(p.relative_to(NAV)): sha(p) for p in files}
assert before == after, 'source changed during final verification'
original = json.loads((OUT / 'before-plan.json').read_text())
current = json.loads((ROOT / 'doc/plan/20261109_camra_navigation_implementation.json').read_text())
assert current.keys() == original.keys()
assert [t['id'] for t in current['tasks']] == [t['id'] for t in original['tasks']]
for old, new in zip(original['tasks'], current['tasks']):
    assert {k: v for k, v in old.items() if k != 'work_records'} == {
            k: v for k, v in new.items() if k != 'work_records'}
    assert new['work_records'][:len(old['work_records'])] == old['work_records']
    if old['id'] != 'NAV-065':
        assert new == old
status = {t['id']: t['status'] for t in current['tasks']}
assert sum(status.values()) == 52 and not status['NAV-065'] and not status['NAV-066']
first = next(t['id'] for t in current['tasks'] if not t['status'] and all(status[d] for d in t['depends_on']))
assert first == 'NAV-065'
coverage = json.loads((NAV / 'reports/nav065_train_coverage_v5_attempt3.json').read_text())
assert (coverage['train_episodes'], coverage['train_steps'], coverage['new_rollouts'],
        coverage['optimizer_steps'], coverage['development_scoring_rounds']) == (540, 207298, 0, 0, 0)
assert len(coverage['candidates']) == 6 and all(c['original_train_pass'] for c in coverage['candidates'])
assert all(len(c['maps']) == 18 for c in coverage['candidates'])
near = sum(m['starts_abs_W_le_0_01'] for c in coverage['candidates'] for m in c['maps'])
assert near == 49
review = json.loads((NAV / 'reports/nav065_guarded_refit_v4_review.json').read_text())
for name, digest in review['evidence']['output_hashes'].items():
    assert sha(Path(review['evidence']['output']) / name) == digest
changed = subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT, text=True).splitlines()
assert all(not (p.startswith(('manuscripts/', 'RC-PPO/', 'HiRA/', 'CaMRA/calibration/',
    'CaMRA/locomotion/')) or p.endswith(('.tex', '.bib', '.pdf'))) for p in changed)
report = {'schema': 'camra-nav065-adaptive-v5-preparation-review', 'task': 'NAV-065',
          'timestamp': datetime.now().astimezone().isoformat(), 'host': socket.getfqdn(),
          'tmux': os.environ['TMUX'], 'python': sys.executable,
          'basic_checks': 48, 'basic_verification_pass': True, 'independent_tester': False,
          'legacy_workflow': True, 'commands': commands, 'tested_source_sha256': before,
          'source_snapshot': str(snapshot), 'diagnosis_sha256': sha(NAV / 'reports/nav065_train_coverage_v5_attempt3.json'),
          'original_train_episodes': 540, 'original_train_steps': 207298,
          'near_boundary_train_starts': near, 'new_formal_rollouts': 0,
          'new_formal_optimizer_trials': 0, 'development_scoring_rounds': 0,
          'formal_output_absent': True, 'proposal_approved': False,
          'plan_history_preserved': True, 'completed_tasks': 52, 'next_task': first,
          'original_G4': 'FAIL', 'NAV_065_status': False, 'NAV_066_started': False,
          'engineering_failures_preserved': ['diagnosis-attempt1.log', 'tests-attempt2.log'],
          'limits': ['Synthetic checks and read-only train diagnosis only; no new fitting result.',
                     'Original development viewed repeatedly; no final or budget-selection access.',
                     'New resource limits need explicit approval; no old quota reused.']}
with (NAV / 'reports/nav065_adaptive_refit_v5_preparation_review.json').open('x') as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print('PASS: 48 basic checks, refusal, source/old output identity and append-only plan history', flush=True)
