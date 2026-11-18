"""Run the approved two-solve entry once; preserve outputs on any failure."""
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
HERE = Path(__file__).resolve().parent
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
sys.path.insert(0, str(NAV / 'scripts'))
import probe_nav065_head as H
protocol, pin = H.approved_registration(NAV / 'config/nav065_head_probe_prereg_v10.json')
assert not (H.R.WORK / protocol['output_name']).exists()
with (HERE / 'validation-approved.log').open('x') as f:
    subprocess.run([sys.executable, '-B', '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', str(ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'), '--mode', 'update'], stdout=f, stderr=subprocess.STDOUT, check=True)
command = [sys.executable, '-B', '-u', str(NAV / 'scripts/probe_nav065_head.py'), '--registration', str(NAV / 'config/nav065_head_probe_prereg_v10.json')]
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
started = datetime.now().astimezone().isoformat()
result = subprocess.run(command, cwd=NAV, env=env)
metadata = {'started': started, 'finished': datetime.now().astimezone().isoformat(), 'host': socket.getfqdn(), 'tmux': os.environ['TMUX'], 'cwd': str(NAV), 'command': command, 'environment_overrides': {k: env[k] for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}, 'pin': pin, 'exit_code': result.returncode, 'automatic_restart_allowed': False}
with (HERE / 'execution.json').open('x') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps(metadata, ensure_ascii=False), flush=True)
sys.exit(result.returncode)
