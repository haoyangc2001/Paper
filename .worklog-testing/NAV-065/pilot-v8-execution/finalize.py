"""Validate the append-only v8 result and commit its exact local evidence set."""
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
LAUNCH = Path('/pub/data/caohy/work/camra_navigation/nav065_pilot_refit_launch_20261118_v8')
read = lambda p: json.loads(p.read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
assert not (LAUNCH / 'result_commit.json').exists(), 'inspect previous receipt; never recommit blindly'
r = read(NAV / 'reports/nav065_pilot_refit_v8_review.json')
assert r['execution_integrity_pass'] and not r['scientific_nav065_pass'] and r['development_scoring_rounds'] == 0
assert read(LAUNCH / 'review.json') == r and read(LAUNCH / 'EXIT.json')['source_unchanged']
assert not (LAUNCH / 'ERROR.json').exists()
for name, digest in read(LAUNCH / 'source_before.json').items():
    assert sha(NAV / name) == digest
for name, digest in read(LAUNCH / 'resources.json')['v2_verified_files'].items():
    assert sha(Path(name)) == digest
for name, digest in r['evidence']['output_hashes'].items():
    assert sha(Path(r['evidence']['output']) / name) == digest
baseline_commit = read(LAUNCH / 'training_start.json')['git']
baseline = json.loads(subprocess.check_output(['git', 'show', baseline_commit + ':doc/plan/20261109_camra_navigation_implementation.json'], cwd=ROOT))
current = read(PLAN)
assert {k: v for k, v in baseline.items() if k != 'tasks'} == {k: v for k, v in current.items() if k != 'tasks'}
for old, now in zip(baseline['tasks'], current['tasks'], strict=True):
    if old['id'] == 'NAV-065':
        assert {k: v for k, v in old.items() if k != 'work_records'} == {k: v for k, v in now.items() if k != 'work_records'}
        assert now['work_records'][:-1] == old['work_records'] and 'v8执行收尾' in now['work_records'][-1]['summary']
    else:
        assert old == now
assert sum(t['status'] for t in current['tasks']) == 52
assert not subprocess.check_output(['git', 'diff', '--name-only', baseline_commit, '--',
    'manuscripts', 'RC-PPO', 'HiRA', 'CaMRA/locomotion', 'CaMRA/calibration'], cwd=ROOT)
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT / 'HiRA')
subprocess.run([sys.executable, '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',
    str(PLAN), '--mode', 'update'], cwd=ROOT, check=True)
files = ['AGENTS.md', 'CaMRA/navigation/README.md', 'CaMRA/navigation/docs/implementation_handoff.md',
    str(PLAN.relative_to(ROOT)), 'CaMRA/navigation/docs/nav065_pilot_refit_v8_review_20261118.md',
    'CaMRA/navigation/reports/nav065_pilot_refit_v8_review.json']
files += [str((HERE / name).relative_to(ROOT)) for name in (
    'await_review.py', 'review.py', 'progress.py', 'close.py', 'finalize.py',
    'launch-attempt1.log', 'review-attempt1.log', 'close-attempt1.log', 'progress-1.json', 'progress-1.log')]
assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT)
subprocess.run(['git', 'add', '-f', '--', *files], cwd=ROOT, check=True)
assert set(subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).splitlines()) == set(files)
subprocess.run(['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check'], cwd=ROOT, check=True)
subprocess.run(['git', 'diff', '--cached', '--stat'], cwd=ROOT, check=True)
subprocess.run(['git', 'commit', '-m', 'Preserve NAV-065 single-candidate pilot v8 results and audit'], cwd=ROOT, check=True)
assert not subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT)
assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT)
receipt = {'timestamp': datetime.now().astimezone().isoformat(), 'host': socket.getfqdn(),
    'tmux': os.environ['TMUX'], 'python': sys.executable,
    'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'files': files, 'push': False, 'tracked_tree_clean': True, 'execution_integrity_pass': True,
    'scientific_nav065_pass': False, 'completed_tasks': 52, 'NAV-065': False, 'NAV-066_started': False,
    'optimizer_trials': r['optimizer_trials'], 'stop_reason': r['stop_reason']}
with (LAUNCH / 'result_commit.json').open('x') as stream:
    json.dump(receipt, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps(receipt, ensure_ascii=False, indent=2))
