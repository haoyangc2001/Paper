"""One bounded saved-data diagnostic; this wrapper never retries."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path('/home/caohy/repositories/Paper')
HERE = ROOT/'.worklog-testing/NAV-043/horizon90-v1'
PYTHON = '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
SOURCE = ROOT/'CaMRA/navigation/scripts/nav043_motion_response.py'
OUTPUT = Path('/pub/data/caohy/work/camra_navigation/nav043_motion_response_probe_v1')


def save(name, value):
    with (HERE/name).open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2)
        stream.write('\n')


if socket.gethostname().split('.')[0]!='tiger11' or not os.environ.get('TMUX'):
    raise ValueError('requires registered host/tmux')
if OUTPUT.exists():
    raise ValueError('output already exists; do not rerun')
# Opening this marker exclusively prevents a duplicate launcher even before the
# probe finishes its initial source hashing and creates its own output root.
with (HERE/'response-launch.lock').open('x') as stream:
    stream.write(str(os.getpid())+'\n')
environment = os.environ.copy()
environment.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                   PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
command = [PYTHON,'-B',str(SOURCE),'--run']
started = time.monotonic()
with (HERE/'response-run.log').open('x') as log:
    process = subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
    save('response-launch.json',{'at':datetime.now().astimezone().isoformat(),
         'host':socket.gethostname(),'pane':os.environ.get('TMUX_PANE'),
         'wrapper_pid':os.getpid(),'probe_pid':process.pid,'cwd':str(ROOT),'command':command,
         'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
         'thread_limits':{k:environment[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
         'GPU_visible_devices':'','output_root':str(OUTPUT)})
    try:
        code = process.wait(timeout=3700)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        code = -9
    save('response-exit.json',{'at':datetime.now().astimezone().isoformat(),'exit_code':code,
         'wrapper_wall_s':time.monotonic()-started,'finished':(OUTPUT/'FINISHED.json').exists(),
         'held':(OUTPUT/'HELD.json').exists(),'retry_performed':False})
raise SystemExit(code)
