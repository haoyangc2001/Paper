"""Commit the saved start snapshot without changing the running sources."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/home/caohy/repositories/Paper')
EV = ROOT/'.worklog-testing/NAV-043/timeout-extension-v1'
PYTHON = '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
assert os.environ.get('TMUX')
subprocess.run([PYTHON,'/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',
    'doc/plan/20261109_camra_navigation_implementation.json','--mode','update'],cwd=ROOT,check=True)
def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
assert not git('diff','--cached','--name-only')
files = ['AGENTS.md','CaMRA/navigation/docs/nav043_timeout_extension_20261120.md',
    'doc/plan/20261109_camra_navigation_implementation.json']
files += [str(p.relative_to(ROOT)) for p in EV.iterdir() if p.is_file()
    and p.name not in ('run.log','audit.log','progress-commit.log')]
subprocess.run(['git','add','-f','--',*files],cwd=ROOT,check=True)
staged = git('diff','--cached','--name-only').splitlines()
assert set(staged) <= set(files)
subprocess.run(['git','diff','--cached','--check','--',
    *[p for p in staged if not p.endswith(('.log','.patch'))]],cwd=ROOT,check=True)
subprocess.run(['git','commit','-m','Record started 90s navigation timeout diagnostic and saved checks'],cwd=ROOT,check=True)
print('TIMEOUT_PROGRESS_COMMITTED',git('rev-parse','HEAD'),flush=True)
