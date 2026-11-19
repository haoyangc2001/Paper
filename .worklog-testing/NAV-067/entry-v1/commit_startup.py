"""Commit only the startup-attempt checkpoint; no push or experiment actions."""
from pathlib import Path
import subprocess

root = Path('/home/caohy/repositories/Paper')
entry = Path('.worklog-testing/NAV-067/entry-v1')
paths = ['CaMRA/navigation/docs/implementation_handoff.md',
         'doc/plan/20261109_camra_navigation_implementation.json']
paths += [str(entry/name) for name in (
    'launch.py','diagnose.py','diagnostic-1.json','checkpoint_startup.py',
    'commit_startup.py','gpu-before-pilot.csv','host-before-pilot.txt',
    'startup-checkpoint.log','startup-plan-check.log','startup-commit.log')]
paths += [str(p.relative_to(root)) for p in sorted((root/entry/'startup-attempt1').iterdir())]
assert all((root/p).is_file() for p in paths)
staged = subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root,text=True).splitlines()
assert set(staged) <= set(paths), staged
changed = subprocess.check_output(['git','diff','--name-only'],cwd=root,text=True).splitlines()
assert set(changed) <= set(paths), changed
check = ['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol']
# Raw stdout and terminal captures retain their exact original whitespace.
audit_paths = [p for p in paths if not p.endswith(('.log','.txt','.csv'))]
subprocess.run([*check,'diff','--check','--',*audit_paths],cwd=root,check=True)
subprocess.run(['git','add','-f','--',*paths],cwd=root,check=True)
assert set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root,text=True).splitlines()) == set(paths)
subprocess.run([*check,'diff','--cached','--check','--',*audit_paths],cwd=root,check=True)
subprocess.run(['git','commit','-m','docs(nav): preserve NAV-067 startup attempt and keep gate held'],cwd=root,check=True)
print(subprocess.check_output(['git','log','-1','--format=%H %s'],cwd=root,text=True))
assert not subprocess.check_output(['git','diff','--name-only'],cwd=root,text=True).strip()
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root,text=True).strip()
