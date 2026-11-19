"""Finalize proposal-only evidence and make the authorized local commit."""
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = 'bcdfa8d31759d5985b736d6c55909bffde0dd97a'
PLAN = 'doc/plan/20261109_camra_navigation_implementation.json'
PYTHON = '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
assert os.environ.get('TMUX_PANE') == '%600'
assert socket.gethostname().split('.')[0] == 'tiger11'
assert Path(sys.executable).resolve() == Path(PYTHON).resolve()
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == BASE
verification = json.loads((HERE / 'verification.json').read_bytes())
assert verification['status'] == 'PROPOSAL_OFFLINE_CHECK_PASS_NOT_EXECUTION_APPROVAL'
proposal = 'CaMRA/navigation/config/nav067_remaining_validation_proposal_v1.json'
assert sha256((ROOT / proposal).read_bytes()).hexdigest() == verification['proposal_sha256']
new = json.loads((ROOT / PLAN).read_bytes())
old = json.loads(subprocess.check_output(['git', 'show', BASE + ':' + PLAN], cwd=ROOT))
assert {k: v for k, v in old.items() if k != 'tasks'} == {k: v for k, v in new.items() if k != 'tasks'}
for a, b in zip(old['tasks'], new['tasks'], strict=True):
    assert {k: v for k, v in a.items() if k != 'work_records'} == {k: v for k, v in b.items() if k != 'work_records'}
    assert b['work_records'][:len(a['work_records'])] == a['work_records']
    if a['id'] == 'NAV-067':
        assert len(b['work_records']) == len(a['work_records']) + 2
    else:
        assert a == b
assert b'\r\n' not in (ROOT / PLAN).read_bytes()
with (HERE / 'final-plan-check.log').open('x', encoding='utf-8') as stream:
    subprocess.run([PYTHON, '-B', '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', str(ROOT / PLAN), '--mode', 'update'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
paths = [PLAN, 'AGENTS.md', 'CaMRA/navigation/docs/implementation_handoff.md', proposal,
         'CaMRA/navigation/docs/nav067_remaining_validation_proposal_20261119.md']
paths += [str((HERE / name).relative_to(ROOT)) for name in
          ('review_proposal.py', 'verification.json', 'review.log', 'close_commit.py', 'final-plan-check.log')]
assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).strip(), 'preexisting staging must not be changed'
assert set(subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT, text=True).splitlines()).issubset(set(paths))
for path in paths:
    assert (ROOT / path).is_file()
subprocess.run(['git', 'add', '-f', '--', *paths], cwd=ROOT, check=True)
assert set(subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).splitlines()) == set(paths)
subprocess.run(['git', 'diff', '--cached', '--check'], cwd=ROOT, check=True)
for path in paths:
    assert subprocess.check_output(['git', 'show', ':' + path], cwd=ROOT) == (ROOT / path).read_bytes()
subprocess.run(['git', 'commit', '-m', 'docs(nav): propose bounded NAV-067 controlled and repeatability checks'], cwd=ROOT, check=True)
print(json.dumps({'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'files_committed': len(paths), 'new_physical_episodes': 0,
                  'NAV_067_status': False, 'remote_push': False}, ensure_ascii=False, indent=2), flush=True)
