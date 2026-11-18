"""Commit only the reviewed v9 preparation under existing local-commit authority."""
import json
import os
from pathlib import Path
import socket
import subprocess

ROOT = Path('/home/caohy/repositories/Paper')
HERE = Path(__file__).resolve().parent
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
base = json.loads((HERE/'closure.json').read_bytes())['base_git']
assert git('rev-parse', 'HEAD') == base
assert not git('diff', '--cached', '--name-only'), 'preserve preexisting index'
files = ['AGENTS.md', 'CaMRA/navigation/README.md', 'CaMRA/navigation/docs/implementation_handoff.md',
    'doc/plan/20261109_camra_navigation_implementation.json',
    'CaMRA/navigation/scripts/probe_nav065_local.py', 'CaMRA/navigation/tests/test_nav065_local_probe.py',
    'CaMRA/navigation/config/nav065_local_probe_proposal_v9.json',
    'CaMRA/navigation/docs/nav065_local_probe_v9_20261118.md',
    'CaMRA/navigation/reports/nav065_local_probe_preflight_v9.json',
    'CaMRA/navigation/reports/nav065_local_probe_v9_preparation_review.json']
files += [str((HERE/name).relative_to(ROOT)) for name in ('context.py', 'check.py', 'review.py', 'close.py',
    'commit.py', 'basic-attempt1.log', 'preflight-attempt1.log', 'validation.log', 'closure.json')]
subprocess.run(['git', 'add', '-f', '--', *files], cwd=ROOT, check=True)
assert set(git('diff', '--cached', '--name-only').splitlines()) == set(files)
subprocess.run(['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check'], cwd=ROOT, check=True)
subprocess.run(['git', 'commit', '-m', 'Prepare NAV-065 local critic fitting diagnosis v9'], cwd=ROOT, check=True)
assert set(git('diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD').splitlines()) == set(files)
receipt = {'commit':git('rev-parse','HEAD'), 'parent':base, 'files':files, 'pushed':False}
with (HERE/'commit.json').open('x') as f:
    json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'commit':receipt['commit'], 'files':len(files), 'pushed':False}))
