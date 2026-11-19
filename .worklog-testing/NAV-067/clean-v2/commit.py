import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

ROOT=Path('/home/caohy/repositories/Paper')
OUT=ROOT/'.worklog-testing/NAV-067/clean-v2'
os.chdir(ROOT)
assert socket.gethostname().split('.')[0]=='tiger11' and os.environ.get('TMUX')
python='/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
subprocess.run([python,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',
               'doc/plan/20261109_camra_navigation_implementation.json','--mode','update'],check=True)
subprocess.run(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--check'],check=True)
report=json.loads((ROOT/'CaMRA/navigation/reports/nav067_substep_preparation_v1.json').read_bytes())
for p,h in report['sources'].items():
    assert hashlib.sha256((ROOT/'CaMRA/navigation'/p).read_bytes()).hexdigest()==h,p
plan=json.loads((ROOT/'doc/plan/20261109_camra_navigation_implementation.json').read_bytes())
assert sum(t['status'] for t in plan['tasks'])==54
assert not next(t for t in plan['tasks'] if t['id']=='NAV-067')['status']
files=['CaMRA/navigation/src/camra_navigation/envs/physical_v2.py',
       'CaMRA/navigation/src/camra_navigation/envs/clean_v2.py',
       'CaMRA/navigation/tests/test_physical_v2.py',
       'CaMRA/navigation/reports/nav067_substep_preparation_v1.json',
       'CaMRA/navigation/docs/implementation_handoff.md',
       'doc/plan/20261109_camra_navigation_implementation.json',
       '.worklog-testing/NAV-067/clean-v2/baseline.json',
       '.worklog-testing/NAV-067/clean-v2/basic-tests.log']
assert not any(p.startswith(('manuscripts/','RC-PPO/','HiRA/','CaMRA/locomotion/','CaMRA/calibration/')) or p.endswith(('.tex','.bib','.pdf')) for p in files)
previously_staged=subprocess.check_output(['git','diff','--cached','--name-only'],text=True).splitlines()
assert set(previously_staged).issubset(files),previously_staged
subprocess.run(['git','add','-f','--',*files],check=True)
staged=subprocess.check_output(['git','diff','--cached','--name-only'],text=True).splitlines()
assert sorted(staged)==sorted(files),staged
subprocess.run(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'],check=True)
subprocess.run(['git','commit','-m','wip(nav): preserve NAV-067 substep accounting and clean adapter foundation'],check=True)
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
print('COMMITTED',head)
with (OUT/'commit.json').open('x') as f:
    json.dump({'head':head,'files':files,'pushed':False,'NAV-067':False},f,indent=2)
