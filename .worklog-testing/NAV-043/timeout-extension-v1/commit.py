"""Commit only this authorized diagnostic preparation after saved preflight."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/home/caohy/repositories/Paper')
EV = ROOT/'.worklog-testing/NAV-043/timeout-extension-v1'
assert os.environ.get('TMUX')
assert (EV/'basic_v2.exit').read_text().strip() == '0'
assert (EV/'preflight.exit').read_text().strip() == '0'
assert 'TIMEOUT_PREFLIGHT_PASS' in (EV/'preflight.log').read_text()
def git(*args):
    return subprocess.check_output(['git',*args], cwd=ROOT, text=True).strip()
assert not git('diff','--cached','--name-only'), 'unexpected staged work'
paths = ['CaMRA/navigation/scripts/nav043_timeout.py',
    'CaMRA/navigation/scripts/nav043_timeout_simulation.py',
    'CaMRA/navigation/scripts/run_nav043_timeout.py',
    'CaMRA/navigation/scripts/audit_nav043_timeout.py',
    'CaMRA/navigation/config/nav043_timeout_extension_v1.json',
    'CaMRA/navigation/config/nav043_timeout_extension_v1.json.sha256',
    'CaMRA/navigation/docs/nav043_timeout_extension_20261120.md',
    'doc/plan/20261109_camra_navigation_implementation.json', 'AGENTS.md']
paths += [str(p.relative_to(ROOT)) for p in EV.iterdir() if p.is_file()]
subprocess.run(['git','add','--',*paths], cwd=ROOT, check=True)
staged = git('diff','--cached','--name-only').splitlines()
assert set(staged) <= set(paths)
assert not any(p.startswith(('manuscripts/','RC-PPO/','HiRA/')) or p.endswith(('.tex','.bib')) for p in staged)
nonlogs = [p for p in staged if not p.endswith('.log')]
subprocess.run(['git','diff','--cached','--check','--',*nonlogs], cwd=ROOT, check=True)
subprocess.run(['git','commit','-m','Prepare authorized 90s execution diagnostic for frozen navigation policy'], cwd=ROOT, check=True)
proof = {'at':datetime.now().astimezone().isoformat(), 'commit':git('rev-parse','HEAD'),
    'preflight_passed':True, 'paths':staged, 'pushed':False}
with (EV/'prelaunch-commit.json').open('x') as stream:
    json.dump(proof, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print('TIMEOUT_PRELAUNCH_COMMITTED', proof['commit'], flush=True)
