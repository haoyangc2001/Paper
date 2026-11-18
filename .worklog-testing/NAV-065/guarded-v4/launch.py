"""Launch one approved v4 attempt from tiger11/tmux; no automatic retries."""
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
LAUNCH = Path('/pub/data/caohy/work/camra_navigation/nav065_guarded_refit_launch_20261118_v4')
PYTHON = '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
sys.path[:0] = [str(NAV / 'scripts'), str(NAV / 'src')]
import refit_nav065_guarded as G

def write(name, data):
    with (LAUNCH / name).open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')

def stamp():
    return datetime.now().astimezone().isoformat()

assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == PYTHON
assert not (LAUNCH / 'START.json').exists()
p, pin = G.approved_registration(G.REGISTRATION)
out = G.R.WORK / p['output_name']
assert not out.exists()
for name, sha in G.R.load(LAUNCH / 'source_before.json').items():
    G.R.verify_hash(NAV / name, sha)
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        args = (proc / 'cmdline').read_bytes().decode().split('\0')
    except (OSError, UnicodeDecodeError):
        continue
    if args and 'python' in Path(args[0]).name and '--registration' in args:
        assert not any(Path(a).name in ('refit_nav065_guarded.py', 'refit_nav065_boundary.py',
                                       'refit_nav065_saved.py') for a in args), 'concurrent refit'
env = {**os.environ, 'PYTHONPATH': 'src:scripts', 'PYTHONDONTWRITEBYTECODE': '1',
       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
validator = [PYTHON, str(Path('/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py')),
             str(ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'), '--mode', 'update']
with (LAUNCH / 'validate-before.log').open('x') as f:
    subprocess.run(validator, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
write('resources.json', {'timestamp': stamp(), 'host': socket.getfqdn(), 'python': sys.executable,
    'python_version': sys.version, 'cwd': str(NAV), 'tmux': os.environ['TMUX'],
    'window': subprocess.check_output(['tmux','display-message','-p','-t',os.environ['TMUX_PANE'],
                                       '#{session_name}:#{window_name} #{pane_id}'],text=True).strip(),
    'gpu': subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total',
                                   '--format=csv'],text=True),
    'git': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'registration_pin': pin})
cmd = [PYTHON, '-u', 'scripts/refit_nav065_guarded.py', '--registration', str(G.REGISTRATION)]
with (LAUNCH / 'training.log').open('x') as log:
    child = subprocess.Popen(cmd, cwd=NAV, env=env, stdout=log, stderr=subprocess.STDOUT)
    write('START.json', {'timestamp': stamp(), 'pid': child.pid, 'host': socket.getfqdn(),
                        'command': cmd, 'registration_pin': pin, 'output': str(out)})
    code = child.wait()
status = 'INFRASTRUCTURE_OR_INTEGRITY_ERROR' if code else (
    'HELD_TRAIN_GATE' if (out / 'HELD_TRAIN_GATE.json').exists() else
    G.R.load(out / 'gate_G4_alignment_v2.json')['verdict'] if (out / 'gate_G4_alignment_v2.json').exists()
    else 'INCOMPLETE_NO_VERDICT')
write('EXIT.json', {'timestamp': stamp(), 'exit_code': code, 'status': status, 'registration_pin': pin})
print(code, status, flush=True)
