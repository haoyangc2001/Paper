"""Save only the reviewed v6 preparation; no fitting or remote operations."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
HERE = ROOT / '.worklog-testing/NAV-065/block-v6-preparation'
PLAN = 'doc/plan/20261109_camra_navigation_implementation.json'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()

assert git('rev-parse', '--short=7', 'HEAD') == 'd9f8bb6'
assert not git('diff', '--cached', '--name-only'), 'Preserve any preexisting index'
current = json.loads((ROOT / PLAN).read_bytes())
baseline = json.loads(git('show', 'HEAD:' + PLAN))
assert {k: v for k, v in current.items() if k != 'tasks'} == {k: v for k, v in baseline.items() if k != 'tasks'}
for old, new in zip(baseline['tasks'], current['tasks'], strict=True):
    if old['id'] == 'NAV-065':
        assert new['work_records'][:len(old['work_records'])] == old['work_records']
        assert len(new['work_records']) == len(old['work_records']) + 2
        assert {k: v for k, v in new.items() if k != 'work_records'} == {k: v for k, v in old.items() if k != 'work_records'}
    else:
        assert new == old
report = json.loads((ROOT / 'CaMRA/navigation/reports/nav065_block_refit_v6_preparation_review.json').read_bytes())
for path, digest in report['evidence'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
subprocess.run([sys.executable, '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', str(ROOT / PLAN), '--mode', 'update'], check=True)
subprocess.run(['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--check'], cwd=ROOT, check=True)
files = [
    'AGENTS.md', 'CaMRA/navigation/README.md',
    'CaMRA/navigation/docs/implementation_handoff.md', PLAN,
    'CaMRA/navigation/scripts/refit_nav065_blocks.py',
    'CaMRA/navigation/tests/test_nav065_block_refit.py',
    'CaMRA/navigation/config/nav065_block_refit_proposal_v6.json',
    'CaMRA/navigation/docs/nav065_block_refit_v6_20261118.md',
    'CaMRA/navigation/reports/nav065_block_refit_preflight_v6.json',
    'CaMRA/navigation/reports/nav065_block_refit_v6_preparation_review.json',
]
evidence = ['preflight.py', 'preflight.log', 'preflight-attempt2.log',
    'basic-attempt1.log', 'diagnose-attempt1.log', 'review.py',
    'review-attempt1.log', 'resources.log', 'commit.py']
files += [str((HERE / name).relative_to(ROOT)) for name in evidence]
assert all((ROOT / name).is_file() for name in files)
assert not any(name.endswith(('.tex', '.bib')) or name.startswith(('manuscripts/', 'RC-PPO/', 'HiRA/')) for name in files)
subprocess.run(['git', 'add', '-f', '--', *files], cwd=ROOT, check=True)
assert set(git('diff', '--cached', '--name-only').splitlines()) == set(files)
subprocess.run(['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check'], cwd=ROOT, check=True)
subprocess.run(['git', 'commit', '-m', 'Prepare NAV-065 stratified critic block refit v6'], cwd=ROOT, check=True)
assert not git('diff', '--cached', '--name-only')
print(json.dumps({'commit': git('rev-parse', 'HEAD'), 'files': len(files),
    'formal_updates': 0, 'new_rollouts': 0, 'pushed': False}))
