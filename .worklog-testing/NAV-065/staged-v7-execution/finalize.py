"""Validate append-only closure and commit only this run's reviewed files."""
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
BASELINE = '3a4362a0c506a789f6932ef87cf8e7352af81e89'
VALIDATOR = '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py'
LAUNCH = Path('/pub/data/caohy/work/camra_navigation/nav065_staged_refit_launch_20261118_v7')
read = lambda p: json.loads(p.read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
r = read(NAV / 'reports/nav065_staged_refit_v7_review.json')
assert r['execution_integrity_pass'] and not r['scientific_pass'] and r['run_status'] == 'HELD_TRAIN_GATE'
assert read(LAUNCH / 'review.json') == r
assert not (LAUNCH / 'ERROR.json').exists() and read(LAUNCH / 'EXIT.json')['source_unchanged']
for name, digest in read(LAUNCH / 'source_before.json').items():
    assert sha(NAV / name) == digest
for name, digest in read(LAUNCH / 'resources.json')['v2_verified_files'].items():
    assert sha(Path(name)) == digest
for name, digest in r['evidence']['output_hashes'].items():
    assert sha(Path(r['evidence']['output']) / name) == digest
baseline = json.loads(subprocess.check_output(['git', 'show', BASELINE + ':doc/plan/20261109_camra_navigation_implementation.json'], cwd=ROOT))
current = read(PLAN)
assert {k:v for k,v in baseline.items() if k != 'tasks'} == {k:v for k,v in current.items() if k != 'tasks'}
for old, now in zip(baseline['tasks'], current['tasks'], strict=True):
    if old['id'] == 'NAV-065':
        assert {k:v for k,v in old.items() if k != 'work_records'} == {k:v for k,v in now.items() if k != 'work_records'}
        assert now['work_records'][:-1] == old['work_records']
        assert 'v7执行收尾' in now['work_records'][-1]['summary']
    else:
        assert old == now
assert sum(t['status'] for t in current['tasks']) == 52
assert not subprocess.check_output(['git', 'diff', '--name-only', BASELINE, '--',
    'manuscripts', 'RC-PPO', 'HiRA', 'CaMRA/locomotion', 'CaMRA/calibration'], cwd=ROOT)
subprocess.run([sys.executable, VALIDATOR, str(PLAN), '--mode', 'update'], check=True)
subprocess.run(['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--check'], cwd=ROOT, check=True)
files = ['AGENTS.md', 'CaMRA/navigation/README.md', 'CaMRA/navigation/docs/implementation_handoff.md',
         str(PLAN.relative_to(ROOT)), 'CaMRA/navigation/docs/nav065_staged_refit_v7_review_20261118.md',
         'CaMRA/navigation/reports/nav065_staged_refit_v7_review.json']
files += [str((HERE / name).relative_to(ROOT)) for name in (
    'await_review.py', 'review.py', 'progress.py', 'close.py', 'finalize.py',
    'review-attempt1.log', 'close.log')]
assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT)
subprocess.run(['git', 'add', '-f', '--', *files], cwd=ROOT, check=True)
assert set(subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).splitlines()) == set(files)
subprocess.run(['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check'], cwd=ROOT, check=True)
subprocess.run(['git', 'diff', '--cached', '--stat'], cwd=ROOT, check=True)
subprocess.run(['git', 'commit', '-m', 'Preserve NAV-065 staged critic refit v7 results and audit'], cwd=ROOT, check=True)
assert not subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT)
assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT)
print(json.dumps({'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'tracked_tree_clean': True, 'nav065_status': False, 'nav066_started': False,
                  'scientific_pass': False, 'original_G4': 'FAIL', 'completed_tasks': 52}))
