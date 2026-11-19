"""Read-only NAV-067 startup diagnosis; no Kit app, environment or rollout."""
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from nav067_sources import NAV, ROOT, OUTPUT, read_manifest, sha, write_new

manifest, _, _ = read_manifest()
plan_path = ROOT/'doc/plan/20261109_camra_navigation_implementation.json'
plan = json.loads(plan_path.read_bytes())
task = next(t for t in plan['tasks'] if t['id']=='NAV-067')
launch = json.loads((OUTPUT/'launch-L1_empty.json').read_bytes())
try:
    os.kill(launch['pid'],0)
except ProcessLookupError:
    live = False
else:
    live = True
assert not live, 'Inspect existing process before doing anything else'
modules = {}
for name in ('unitree_rl_lab','isaaclab','isaaclab_tasks','isaacsim','gymnasium'):
    spec = importlib.util.find_spec(name)
    modules[name] = {'found':spec is not None,
                     'origin':str(spec.origin) if spec else None,
                     'locations':list(spec.submodule_search_locations or []) if spec else []}
out = Path(__file__).resolve().parent
result = {
    'timestamp':datetime.now().astimezone().isoformat(), 'host':socket.getfqdn(),
    'python':sys.executable, 'python_version':sys.version.split()[0],
    'tmux':subprocess.check_output(['tmux','display-message','-t',os.environ['TMUX_PANE'],
        '-p','#{session_name}:#{window_name} #{window_id} #{pane_id}'],text=True).strip(),
    'cwd':str(Path.cwd()), 'sys_path':sys.path, 'modules':modules,
    'local_unitree_source':str(NAV.parent/'locomotion/source/unitree_rl_lab'),
    'local_unitree_init_exists':(NAV.parent/'locomotion/source/unitree_rl_lab/unitree_rl_lab/__init__.py').is_file(),
    'plan_sha256':sha(plan_path),'plan_version':plan['document']['version'],
    'tasks_fully_parsed':len(plan['tasks']),
    'records_fully_parsed':sum(len(t['work_records']) for t in plan['tasks']),
    'current_task':task, 'binding':manifest['binding'],
    'source_hashes':manifest['new_sources'],
    'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'submodule_status':subprocess.check_output(['git','submodule','status'],cwd=ROOT,text=True).splitlines(),
    'gpu6':subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=index,uuid,name,memory.free','--format=csv,noheader'],text=True).strip(),
    'previous_pilot_pid_live':live,'new_navigation_episodes':0,'new_physics_steps':0,
    'new_training_updates':0,'new_W_queries':0}
write_new(out/'preflight.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ('current_task','sys_path','submodule_status')},ensure_ascii=False,indent=2))
