"""Read-only v8 real train diagnosis; invoke on tiger11 inside tmux."""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
HERE = Path(__file__).resolve().parent
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
sys.path.insert(0, str(NAV / 'scripts'))
sys.path.insert(0, str(NAV / 'src'))
import refit_nav065_pilot as P

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
v2 = read(NAV / 'reports/nav064_terminal_tail_training_release_v1.json')['v2_verified_files']
for name, digest in v2.items():
    assert sha(Path(name)) == digest
processes = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
assert not any(re.search(r'python(?:\d+(?:\.\d+)*)?\s+(?:-u\s+)?(?:\S*/)?refit_nav065_\S+\.py\s+--registration', line)
               for line in processes.splitlines())
resources = {'host': socket.getfqdn(), 'tmux': os.environ['TMUX'], 'cwd': str(NAV),
    'python': sys.executable, 'version': sys.version, 'v2_verified_files': len(v2),
    'active_nav065_refit_processes': 0,
    'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.free', '--format=csv,noheader'], text=True, timeout=20),
    'ssh_exception': 'Existing user authorization permits direct same-host tiger11/tmux execution.'}
with (HERE / 'resources.json').open('x') as stream:
    json.dump(resources, stream, indent=2)
P.torch.set_num_threads(1)
_, reg, inputs, prepared = P.sources()
report = P.diagnose(reg, inputs, prepared)
assert (report['train_episodes'], report['train_steps']) == (972, 372471)
assert len(report['candidates']) == 6
assert sum(s['classification'] == 'false_feasible' for c in report['candidates'] for s in c['starts']) == 19
assert sum(s['classification'] == 'false_infeasible' for c in report['candidates'] for s in c['starts']) == 64
with (NAV / 'reports/nav065_pilot_refit_preflight_v8.json').open('x') as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps({'train_episodes': 972, 'train_steps': 372471, 'optimizer_steps': 0,
    'new_rollouts': 0, 'development_scoring_rounds': 0, 'v2_files': len(v2),
    'source_gate_match': True, 'false_feasible': 19, 'false_infeasible': 64}))
