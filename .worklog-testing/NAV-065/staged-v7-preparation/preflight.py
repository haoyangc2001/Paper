"""Read-only real train preflight; no rollout, optimizer or development scoring."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
HERE = ROOT / '.worklog-testing/NAV-065/staged-v7-preparation'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
v2 = json.loads((NAV / 'reports/nav064_terminal_tail_training_release_v1.json').read_bytes())['v2_verified_files']
for name, digest in v2.items():
    assert sha(Path(name)) == digest
resources = {'host': socket.getfqdn(), 'tmux': os.environ['TMUX'], 'cwd': str(NAV),
    'python': sys.executable, 'version': sys.version, 'v2_verified_files': len(v2),
    'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.free', '--format=csv,noheader'], text=True, timeout=20)}
processes = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
assert not any(('scripts/refit_nav065_' in line and '--registration' in line and 'python' in line)
               for line in processes.splitlines())
resources['active_nav065_refit_processes'] = 0
with (HERE / 'resources.json').open('x') as stream:
    json.dump(resources, stream, indent=2)
import refit_nav065_staged as S
S.torch.set_num_threads(1)
upstream, reg, inputs, prepared = S.sources()
report = S.diagnose(reg, inputs, prepared)
assert (report['train_episodes'], report['train_steps']) == (972, 372471)
assert len(report['candidates']) == 6
stored = S.R.load(S.R.WORK / 'nav065_block_refit_v6/train_gate.json')
by_id = {c['run_id']: c for c in stored['candidates']}
for row, (c, model, old, new) in zip(report['candidates'], prepared, strict=True):
    assert all(by_id[row['run_id']][key] == val for key, val in row['gate'].items())
    prediction = S.A.classify(model, old + new, inputs.value['quantities']['Z_REF_J']['value'])
    assert list(prediction) == [s['predicted_feasible'] for s in row['starts']]
    assert [s['actual_feasible'] for s in row['starts']] == [bool(t.feasible) for _, t in old + new]
    assert S.R.digests(model) == c['state_dict_sha256']
report['saved_gate_exact_match'] = True
with (NAV / 'reports/nav065_staged_refit_preflight_v7.json').open('x') as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
print(json.dumps({'train_episodes': 972, 'train_steps': 372471, 'optimizer_steps': 0,
    'new_rollouts': 0, 'development_scoring_rounds': 0, 'v2_files': len(v2),
    'source_gate_match': True, 'false_feasible': sum(s['classification'] == 'false_feasible' for c in report['candidates'] for s in c['starts'])}))
