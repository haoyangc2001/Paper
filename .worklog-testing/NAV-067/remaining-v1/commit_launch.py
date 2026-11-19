"""Commit approved sealed preparation, then launch once with declared environment."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
NAV=ROOT/'CaMRA/navigation'
sys.path.insert(0,str(NAV/'scripts'))
from nav067_sources import load, write_new, sha
from run_nav067_remaining_validation import EXECUTION, EXPECTED_ROOT, context

assert os.environ['TMUX_PANE']=='%600'
assert not EXPECTED_ROOT.exists()
execution=load(EXECUTION)
environment=os.environ.copy()
environment.update(execution['fixed_environment'])
os.environ.update(execution['fixed_environment'])
context()
check=[sys.executable,'-B',str(NAV/'scripts/run_nav067_remaining_validation.py'),'--check']
completed=subprocess.run(check,cwd=ROOT,env=environment,capture_output=True,text=True)
(HERE/'sealed-check.log').write_text(completed.stdout+completed.stderr)
assert completed.returncode==0,completed.stdout+completed.stderr
validator=[sys.executable,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',str(ROOT/'doc/plan/20261109_camra_navigation_implementation.json'),'--mode','update']
subprocess.run(validator,cwd=ROOT,check=True)
protected=subprocess.check_output(['git','diff','--name-only','--','manuscripts','*.tex','*.bib','RC-PPO','HiRA'],cwd=ROOT,text=True)
assert not protected.strip(),protected
paths=['AGENTS.md','doc/plan/20261109_camra_navigation_implementation.json','CaMRA/navigation/docs/implementation_handoff.md',
 'CaMRA/navigation/config/nav067_remaining_validation_approval_v1.json','CaMRA/navigation/config/nav067_remaining_validation_approval_v1.json.sha256',
 'CaMRA/navigation/config/nav067_remaining_validation_execution_v1.json','CaMRA/navigation/config/nav067_remaining_validation_execution_v1.json.sha256',
 'CaMRA/navigation/scripts/nav067_controlled_simulation.py','CaMRA/navigation/scripts/nav067_collider_readback.py',
 'CaMRA/navigation/scripts/nav067_remaining_audit.py','CaMRA/navigation/scripts/run_nav067_remaining_validation.py',
 'CaMRA/navigation/src/camra_navigation/envs/controlled_cases_v1.py']
paths.extend(str(p.relative_to(ROOT)) for p in HERE.iterdir() if p.is_file() and not p.name.startswith('commit-launch'))
staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
assert set(staged)<=set(paths),'unrelated staged files must be preserved separately'
subprocess.run(['git','add','-f','--',*paths],cwd=ROOT,check=True)
subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
subprocess.run(['git','commit','-m','feat(nav): seal approved NAV-067 controlled and repeatability validation'],cwd=ROOT,check=True)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
write_new(HERE/'prelaunch-commit.json',{'at':datetime.now().astimezone().isoformat(),'commit':commit,
    'execution_sha256':sha(EXECUTION),'new_applications_before_launch':0,'push':False})
print('PREPARATION_COMMITTED',commit,flush=True)
assert not EXPECTED_ROOT.exists()
print('LAUNCH_APPROVED_SINGLE_COORDINATOR',execution['command'],flush=True)
raise SystemExit(subprocess.call(execution['command'],cwd=ROOT,env=environment))
