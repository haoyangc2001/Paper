"""Commit the authorized v10 outcome and evidence only; no push."""
import json
import os
from pathlib import Path
import socket
import subprocess

ROOT = Path('/home/caohy/repositories/Paper')
HERE = Path(__file__).resolve().parent
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert json.loads((HERE / 'final.json').read_bytes())['delivery_integrity_pass']
def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
assert git('rev-parse', 'HEAD') == 'eea3945d56f362c6097dc2f8a7a40a0ee7ce907a'
assert not git('diff', '--cached', '--name-only')
files = ['AGENTS.md', 'CaMRA/navigation/README.md', 'CaMRA/navigation/docs/implementation_handoff.md',
    'doc/plan/20261109_camra_navigation_implementation.json',
    'CaMRA/navigation/config/nav065_head_probe_prereg_v10.json',
    'CaMRA/navigation/config/nav065_head_probe_prereg_v10.json.sha256',
    'CaMRA/navigation/reports/nav065_head_probe_v10_review.json',
    'CaMRA/navigation/docs/nav065_head_probe_v10_review_20261118.md']
files += [str(HERE.relative_to(ROOT) / name) for name in (
    'context.py', 'launch.py', 'audit.py', 'close.py', 'final.py', 'commit.py',
    'context.log', 'validation-before.log', 'validation-approved.log', 'run-attempt1.log',
    'execution.json', 'audit-attempt1.log', 'close-attempt1.log', 'validation-after.log',
    'final-attempt1.log', 'final.json')]
for name in files:
    assert (ROOT / name).is_file(), name
    assert not name.startswith(('manuscripts/', 'RC-PPO/', 'HiRA/', 'CaMRA/locomotion/', 'CaMRA/calibration/'))
    assert not name.endswith(('.tex', '.bib'))
git('add', '-f', '--', *files)
assert set(git('diff', '--cached', '--name-only').splitlines()) == set(files)
git('-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check')
print(git('diff', '--cached', '--stat'), flush=True)
print(git('commit', '-m', 'Record NAV-065 v10 local solve and full-train regression'), flush=True)
assert not git('diff', '--cached', '--name-only') and not git('diff', '--name-only')
report = {'commit': git('rev-parse', 'HEAD'), 'files': files, 'pushed': False,
    'formal_candidate_head_solves': 2, 'audit_head_solves': 0,
    'local_pass': True, 'full_train_pass': False, 'scientific_pass': False, 'NAV065': False}
with (HERE / 'commit.json').open('x') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps(report, ensure_ascii=False), flush=True)
