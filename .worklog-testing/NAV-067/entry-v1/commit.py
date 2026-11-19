from pathlib import Path
import subprocess

root=Path('/home/caohy/repositories/Paper')
paths=['CaMRA/navigation/scripts/nav067_sources.py',
       'CaMRA/navigation/scripts/evaluate_navigation_v2.py',
       'CaMRA/navigation/tests/test_nav067_binding.py',
       'CaMRA/navigation/reports/nav067_pilot_binding_v1.json',
       'CaMRA/navigation/docs/implementation_handoff.md',
       'doc/plan/20261109_camra_navigation_implementation.json',
       '.worklog-testing/NAV-067/entry-v1/basic-tests.log',
       '.worklog-testing/NAV-067/entry-v1/prepare.log',
       '.worklog-testing/NAV-067/entry-v1/legacy-format-failure.log',
       '.worklog-testing/NAV-067/entry-v1/review.py',
       '.worklog-testing/NAV-067/entry-v1/review.log',
       '.worklog-testing/NAV-067/entry-v1/plan-check.log']
staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root,text=True).splitlines()
assert set(staged)<=set(paths), staged
changes=subprocess.check_output(['git','diff','--name-only'],cwd=root,text=True).splitlines()
assert all(p in paths for p in changes), changes
whitespace=['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol']
subprocess.run([*whitespace,'diff','--check'],cwd=root,check=True)
subprocess.run(['git','add','-f','--',*paths],cwd=root,check=True)
assert set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root,text=True).splitlines())==set(paths)
subprocess.run([*whitespace,'diff','--cached','--check'],cwd=root,check=True)
subprocess.run(['git','commit','-m','feat(nav): bind selected models and seal NAV-067 clean pilot entry'],cwd=root,check=True)
print(subprocess.check_output(['git','log','-1','--format=%H %s'],cwd=root,text=True))
