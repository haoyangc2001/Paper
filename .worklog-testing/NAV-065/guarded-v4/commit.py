"""Commit the authorized NAV-065 source/evidence set locally, without pushing."""
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
LAUNCH = Path('/pub/data/caohy/work/camra_navigation/nav065_guarded_refit_launch_20261118_v4')
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()

assert not git('diff', '--cached', '--name-only'), 'preserve existing staged changes'
assert not (LAUNCH / 'commit.json').exists(), 'inspect existing commit; do not repeat'
subprocess.run([sys.executable, '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',
                'doc/plan/20261109_camra_navigation_implementation.json', '--mode', 'update'], cwd=ROOT, check=True)
report = json.loads((LAUNCH / 'finalize.json').read_text())
assert report['legacy_validator_pass'] and report['g4_verdict'] == 'FAIL'
files = ['AGENTS.md', 'CaMRA/navigation/README.md', 'CaMRA/navigation/docs/implementation_handoff.md',
         'doc/plan/20261109_camra_navigation_implementation.json',
         'CaMRA/navigation/src/camra_navigation/algorithm/deployment_oracle.py',
         'CaMRA/navigation/tests/test_deployment_oracle.py']
untracked = git('ls-files', '--others', '--exclude-standard', 'CaMRA/navigation').splitlines()
files += [p for p in untracked if 'nav065' in Path(p).name]
files += ['.worklog-testing/NAV-065/guarded-v4/' + name for name in ('launch.py', 'review.py', 'finalize.py')]
files = sorted(set(files))
assert all((ROOT / p).is_file() for p in files)
assert all(not p.startswith(('manuscripts/', 'RC-PPO/', 'HiRA/', 'docs/paper', 'submission'))
           and not p.endswith(('.tex', '.bib')) for p in files)
subprocess.run(['git', 'add', '--', *files], cwd=ROOT, check=True)
assert set(git('diff', '--cached', '--name-only').splitlines()) == set(files)
subprocess.run(['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check'], cwd=ROOT, check=True)
subprocess.run(['git', 'commit', '-m', 'nav: preserve NAV-065 critic refits and failed deployment gate'], cwd=ROOT, check=True)
commit = git('rev-parse', 'HEAD')
assert set(git('diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD').splitlines()) == set(files)
assert not git('diff', '--cached', '--name-only')
assert not git('diff', '--name-only'), 'unexpected tracked changes after commit'
result = {'timestamp': datetime.now().astimezone().isoformat(), 'host': socket.getfqdn(),
          'tmux': os.environ['TMUX'], 'commit': commit, 'files': files, 'push': False,
          'remaining_untracked_count': len(git('ls-files', '--others', '--exclude-standard').splitlines()),
          'g4_verdict': 'FAIL', 'NAV-065': False}
with (LAUNCH / 'commit.json').open('x') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
