"""Commit the approved registration and its bounded launcher before simulation."""
import ast
from pathlib import Path
import os
import subprocess

ROOT = Path('/home/caohy/repositories/Paper')
HERE = Path(__file__).resolve().parent
assert os.environ.get('TMUX_PANE') == '%587'
files = [
    'CaMRA/navigation/config/nav067_startup_recovery_v3.json',
    'CaMRA/navigation/config/nav067_startup_recovery_v3.json.sha256',
    'doc/plan/20261109_camra_navigation_implementation.json',
    *[str((HERE/n).relative_to(ROOT)) for n in ('prepare.py','prepare.log','preflight.json','plan-check.log','launch.py','commit_prelaunch.py')]]
for file in files:
    assert (ROOT/file).is_file(), file
    if file.endswith('.py'): ast.parse((ROOT/file).read_text())
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
dirty = subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines()
assert dirty == ['doc/plan/20261109_camra_navigation_implementation.json'], dirty
subprocess.run(['git','diff','--check','--',*files],cwd=ROOT,check=True)
subprocess.run(['git','add','--',*files],cwd=ROOT,check=True)
assert set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()) == set(files)
subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
subprocess.run(['git','commit','-m','chore(nav): register approved one-time NAV-067 startup recovery'],cwd=ROOT,check=True)
print(subprocess.check_output(['git','log','-1','--format=%H %s'],cwd=ROOT,text=True),flush=True)
