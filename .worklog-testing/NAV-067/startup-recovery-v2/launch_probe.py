"""Launch the one zero-physics probe in the current approved tmux pane."""
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

root = Path('/home/caohy/repositories/Paper')
out = Path(__file__).resolve().parent
assert socket.gethostname().split('.')[0]=='tiger11' and os.environ.get('TMUX')
assert not (out/'probe-attempt1').exists()
env = os.environ.copy()
env.update({'CUDA_VISIBLE_DEVICES':'6','OMNI_KIT_ACCEPT_EULA':'YES',
    'TMPDIR':'/pub/data/caohy/tmp','XDG_DATA_HOME':'/pub/data/caohy/xdgdata',
    'XDG_CACHE_HOME':'/pub/data/caohy/xdgcache','XDG_CONFIG_HOME':'/pub/data/caohy/xdgconfig',
    'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1',
    'LD_LIBRARY_PATH':'/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')})
cmd=[sys.executable,'-B','-u',str(out/'probe.py')]
with (out/'probe-console.log').open('x') as stream:
    process = subprocess.Popen(cmd,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
    with (out/'probe-process.json').open('x') as f:
        json.dump({'pid':process.pid,'started_at':datetime.now().astimezone().isoformat(),
            'cmd':cmd,'cwd':str(root),'tmux':subprocess.check_output(['tmux','display-message',
            '-t',os.environ['TMUX_PANE'],'-p','#{session_name}:#{window_name} #{pane_id}'],text=True).strip()},f,indent=2)
    code = process.wait()
with (out/'probe-exit.json').open('x') as f:
    json.dump({'pid':process.pid,'returncode':code,'ended_at':datetime.now().astimezone().isoformat()},f,indent=2)
raise SystemExit(code)
