"""Run exactly the newly approved v9 probe once; never restart or score development."""
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT/'CaMRA/navigation'
HERE = Path(__file__).resolve().parent
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
sys.path[:0] = [str(NAV/'scripts'),str(NAV/'src')]
import probe_nav065_local as L

context = json.loads((HERE/'context.json').read_bytes())
registration = NAV/'config/nav065_local_probe_prereg_v9.json'
protocol,pin = L.approved_registration(registration)
assert pin == context['approval_pin']
assert not (L.R.WORK/protocol['output_name']).exists()
with (HERE/'validation-open.log').open('x') as f:
    valid = subprocess.run([sys.executable,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',str(ROOT/'doc/plan/20261109_camra_navigation_implementation.json'),'--mode','update'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
assert valid.returncode == 0
snapshot = HERE/'source'
snapshot.mkdir(exist_ok=False)
for name in (*L.BOUND_INPUTS,'config/nav065_local_probe_prereg_v9.json','config/nav065_local_probe_prereg_v9.json.sha256'):
    target=snapshot/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(NAV/name,target)
command = [sys.executable,'-B','-u','scripts/probe_nav065_local.py','--registration','config/nav065_local_probe_prereg_v9.json']
with (HERE/'started.json').open('x') as f:
    json.dump({'timestamp':datetime.now().astimezone().isoformat(),'command':command,'cwd':str(NAV),'pin':pin,'tmux':os.environ['TMUX']},f,ensure_ascii=False,indent=2)
print('V9_PROBE_STARTED',pin,flush=True)
start=time.monotonic()
with (HERE/'run.log').open('x') as f:
    result=subprocess.run(command,cwd=NAV,env={**os.environ,'PYTHONPATH':str(NAV/'src'),'PYTHONDONTWRITEBYTECODE':'1'},stdout=f,stderr=subprocess.STDOUT)
receipt={'returncode':result.returncode,'elapsed_seconds':time.monotonic()-start,
    'output':str(L.R.WORK/protocol['output_name']),
    'finished_exists':(L.R.WORK/protocol['output_name']/'PROBE_FINISHED.json').exists(),
    'automatic_restart_allowed':False}
with (HERE/'process.json').open('x') as f:
    json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt),flush=True)
assert result.returncode == 0 and receipt['finished_exists'], 'inspect preserved log and output; do not restart'
