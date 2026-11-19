"""One approved attempt; capture actual resources and keep all outputs."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
WORK = Path('/pub/data/caohy/work/camra_navigation')
LAUNCH = WORK / 'nav065_saved_refit_launch_20261118_v1'
OUT = WORK / 'nav065_saved_refit_v2'
REG = NAV / 'config/nav065_saved_refit_prereg_v2.json'
EXPECTED_PYTHON = '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
assert sys.executable == EXPECTED_PYTHON
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert not OUT.exists(), 'inspect existing output; never relaunch the attempt'
LAUNCH.mkdir(exist_ok=False)
os.chdir(NAV)

def write(name, obj):
    with (LAUNCH / name).open('x') as f:
        json.dump(obj, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())

def digest(p):
    return hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()

env = dict(os.environ, PYTHONPATH='src:scripts', PYTHONDONTWRITEBYTECODE='1',
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.free', '--format=csv,noheader'],
                     capture_output=True, text=True, check=True)
old = subprocess.run(['ps', '-o', 'pid,stat,pcpu,wchan,cmd', '-p',
                      '444562,444563,444564,444566,444568,444569'], capture_output=True, text=True)
write('resources.json', {'time': datetime.now().astimezone().isoformat(),
    'host': socket.getfqdn(), 'tmux': os.environ['TMUX'], 'cwd': str(NAV),
    'python': sys.executable, 'python_version': sys.version,
    'GPU': gpu.stdout.strip(), 'old_attempt_processes': old.stdout,
    'launcher_sha256': digest(Path(__file__)),
    'git_head': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
sys.path[:0] = [str(NAV/'scripts'),str(NAV/'src')]
import refit_nav065_saved as S
protocol, pin = S.approved_registration(REG)
write('registration.json', {'pin': pin, 'registration': protocol})
upstream, _, _, inputs, _, _, _ = S.R.preflight()
source = S.R.load(WORK / protocol['source_run'] / 'fitted_candidates.json')
guarded = {str(WORK / protocol['source_run'] / 'fitted_candidates.json'): protocol['source_manifest_sha256'],
           str(WORK / protocol['source_run'] / 'train_episodes.jsonl'): protocol['source_train_journal_sha256']}
guarded.update({c['checkpoint']: c['checkpoint_sha256'] for c in source['candidates']})
guarded.update({str(NAV/f): h for f,h in protocol['implementation_hashes'].items()})
guarded[str(REG)] = pin
for p,h in guarded.items():
    assert digest(Path(p)) == h, p
write('source_before.json',guarded)
with (LAUNCH/'preflight.log').open('x') as f:
    subprocess.run([sys.executable,'-u','scripts/refit_nav065_saved.py','--preflight'],
                   env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
command=[sys.executable,'-u','scripts/refit_nav065_saved.py','--registration',str(REG)]
start=time.monotonic()
with (LAUNCH/'training.log').open('x') as log:
    child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    write('START.json',{'time':datetime.now().astimezone().isoformat(),'pid':child.pid,
        'command':command,'cwd':str(NAV),'registration_pin':pin,'output':str(OUT),
        'host':socket.getfqdn(),'tmux':os.environ['TMUX'],
        'max_updates_total':1800,'new_rollouts':0})
    print('STARTED',child.pid,'pin',pin,flush=True)
    code=child.wait()
after={p:digest(Path(p)) for p in guarded}
write('source_after.json',after)
unchanged=after==guarded
if code!=0:
    state='ERROR_KEEP_OUTPUTS'
elif (OUT/'HELD_TRAIN_GATE.json').exists():
    state='HELD_TRAIN_GATE'
elif (OUT/'gate_G4_alignment_v2.json').exists():
    state='G4_'+S.R.load(OUT/'gate_G4_alignment_v2.json')['verdict']
else:
    state='INCOMPLETE_KEEP_OUTPUTS'
write('EXIT.json',{'time':datetime.now().astimezone().isoformat(),'exit_code':code,
                  'state':state,'seconds':time.monotonic()-start,
                  'guarded_inputs_unchanged':unchanged,'registration_pin':pin})
print('ENDED',state,'exit',code,'unchanged',unchanged,flush=True)
assert unchanged
assert code==0
