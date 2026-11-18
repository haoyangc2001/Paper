"""Legacy main-agent audit: reload and evaluate, never solve another head."""
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
HERE = Path(__file__).resolve().parent
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
sys.path.insert(0, str(NAV / 'scripts'))
import numpy as np
import torch
import probe_nav065_head as H

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
read = lambda p: json.loads(p.read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
protocol, pin = H.approved_registration(NAV / 'config/nav065_head_probe_prereg_v10.json')
execution = read(HERE / 'execution.json')
assert execution['exit_code'] == 0 and execution['pin'] == pin
out = H.R.WORK / protocol['output_name']
result, preflight = read(out / 'HEAD_FINISHED.json'), read(out / 'preflight.json')
assert result['pin'] == pin and not (out / 'STOPPED.json').exists()
attempts = [json.loads(line) for line in (out / 'attempts.jsonl').read_text().splitlines()]
assert attempts == [{'kind': 'head_solve_attempt', 'role': role} for role in ('critic_cost', 'critic_reach')]
assert sum(x['head_solves'] for x in result['head_solves'].values()) == 2
assert result['optimizer_steps'] == result['new_rollouts'] == result['development_scoring_rounds'] == 0
assert not result['deployment_allowed'] and not result['scientific_nav065_pass'] and not result['automatic_expansion_allowed']
upstream, reg, inputs, prepared = H.L.sources()
source_hashes = {c['run_id']: H.R.digests(m) for c, m, _, _ in prepared}
assert source_hashes == read(NAV / 'reports/nav065_local_probe_v9_review.json')['source_state_dict_hashes']
c, source, old, new = next(row for row in prepared if row[0]['run_id'] == H.L.P.TARGET)
assert c['checkpoint_sha256'] == protocol['source_checkpoint_sha256']
checkpoint = out / 'diagnostic-head.pt'
assert sha(checkpoint) == result['checkpoint_sha256']
model = H.R.load_network(inputs, c, checkpoint)
allowed = {f'{role}.head.{name}' for role in H.R.SCORE_ROLES for name in ('weight', 'bias')}
changed = [name for name, tensor in source.state_dict().items() if not torch.equal(tensor, model.state_dict()[name])]
assert set(changed) == allowed
selection = read(Path('/pub/data/caohy/work/camra_navigation/nav065_local_probe_v9/selection.json'))
assert preflight['local_keys'] == selection['local_keys'] and len(selection['local_keys']) == 25
batch = H.L.local_batch(old + new, [tuple(k) for k in selection['local_keys']])
for i, key in ((1, 'budget_J'), (2, 'cost_norm'), (3, 'reach')):
    assert np.array_equal(batch[i], selection['local_targets'][key])
scale = inputs.value['quantities']['Z_REF_J']['value']
local_before = H.L.evaluate_local(source, batch, reg, scale)
local_after = H.L.evaluate_local(model, batch, reg, scale)
assert local_before == preflight['source_local'] and local_after == result['local']
settings = read(H.L.P.PROPOSAL)['fit']
before = H.L.P.A.train_gate(source, old, new, reg, settings, scale)
after = H.L.P.A.train_gate(model, old, new, reg, settings, scale)
assert before == preflight['source_full_train_gate'] == result['full_train_before']
assert after == result['full_train_after']
assert before['original_grid']['policy_actual_feasible_by_budget'] == after['original_grid']['policy_actual_feasible_by_budget']
precision = {}
for role, i, name in (('critic_cost', 2, 'vc_norm'), ('critic_reach', 3, 'vh')):
    critic = getattr(model, role)
    with torch.no_grad():
        pred = critic(torch.as_tensor(batch[0], dtype=torch.float32)).numpy()
    assert pred.tolist() == result['head_solves'][role]['production_predictions']
    assert pred.tolist() == result['local']['predictions'][name]
    error = float(np.abs(pred - batch[i]).max())
    assert error == result['head_solves'][role]['production_float32_residual_linf']
    norm = float(np.linalg.norm(H.head_vector(critic) - H.head_vector(getattr(source, role))))
    assert np.isclose(norm, result['head_solves'][role]['correction_norm'], rtol=1e-5, atol=1e-6)
    precision[role] = {'saved_float32_correction_norm': norm, 'float32_residual_linf': error,
        'recorded_float64_residual_linf': result['head_solves'][role]['double_design_residual_linf']}
vc, vh = (np.array(local_after['predictions'][name]) for name in ('vc_norm', 'vh'))
truth = np.maximum(batch[3], batch[2] - batch[1] / scale) <= 0
predicted = np.maximum(vh, vc - batch[1] / scale) <= 0
assert int(np.count_nonzero(~truth & predicted)) == local_after['false_feasible']
assert int(np.count_nonzero(truth & ~predicted)) == local_after['false_infeasible']
starts = {stage: H.L.P.S.start_snapshot(net, old, new, scale, reg['primary_k'])
          for stage, net in (('before', source), ('after', model))}
confusion = {stage: dict(Counter(row['classification'] for row in rows)) for stage, rows in starts.items()}
assert all(len(rows) == 162 for rows in starts.values())
for name, digest in read(HERE / 'baseline.json')['v2_verified_files'].items():
    assert sha(Path(name)) == digest, name
assert source_hashes == {candidate['run_id']: H.R.digests(net) for candidate, net, _, _ in prepared}
assert '33 passed' in (ROOT / '.worklog-testing/NAV-065/head-v10-preparation/basic-attempt2.log').read_text()
report = {'schema': 'camra-nav065-head-probe-review-v10', 'timestamp': datetime.now().astimezone().isoformat(),
    'host': socket.getfqdn(), 'tmux': os.environ['TMUX'], 'python': sys.executable,
    'base_git': read(HERE / 'baseline.json')['git'], 'approval_pin': pin, 'integrity_pass': True,
    'candidate': c['run_id'], 'source_checkpoint_sha256': c['checkpoint_sha256'],
    'available_train_episodes': len(old) + len(new), 'available_train_steps': sum(len(t.observations) for _, t in old + new),
    'fitted_train_starts': len(batch[0]), 'formal_candidate_head_solves': 2, 'audit_head_solves': 0,
    'optimizer_steps': 0, 'new_rollouts': 0, 'development_scoring_rounds': 0,
    'basic_tests_passed': 33, 'basic_test_code_hashes_match_approved_registration': True,
    'source_models_unchanged': True, 'all_six_actors_unchanged': True, 'diagnostic_actor_and_bodies_unchanged': True,
    'changed_diagnostic_parameters': changed, 'source_state_dict_hashes': source_hashes,
    'diagnostic_state_dict_hashes': H.R.digests(model), 'v2_files_verified': 33,
    'checkpoint_sha256': sha(checkpoint), 'precision': precision,
    'local_before': local_before, 'local_after': local_after,
    'full_train_before': before, 'full_train_after': after, 'all_train_start_confusion': confusion,
    'all_train_starts': starts, 'scientific_pass': False, 'original_G4': 'FAIL',
    'NAV065': False, 'NAV066_started': False, 'completed_tasks': 52, 'deployment_allowed': False,
    'evidence': {'output': str(out), 'output_hashes': {p.name: sha(p) for p in out.iterdir() if p.is_file()},
        'registration_sha256': pin, 'implementation_hashes': protocol['implementation_hashes'],
        'execution_metadata': str(HERE / 'execution.json')},
    'limitations': ['The 25 selected train starts can be fitted in the actual float32 network with these fixed bodies.',
        'This finite-sample result does not establish whole-task representation or generalization.',
        'The diagnostic copy is not deployed; actor behavior and saved rollout outcomes are unchanged.',
        'No further solve, optimizer step, development evaluation or downstream task is authorized.']}
with (NAV / 'reports/nav065_head_probe_v10_review.json').open('x') as f:
    json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    f.write('\n')
print(json.dumps({k: report[k] for k in ('integrity_pass', 'formal_candidate_head_solves', 'audit_head_solves', 'fitted_train_starts', 'basic_tests_passed', 'all_train_start_confusion', 'scientific_pass', 'NAV065')}, ensure_ascii=False))
print(json.dumps({'local_before': {k: local_before[k] for k in ('local_pass', 'cost_sign_errors', 'reach_sign_errors', 'false_feasible', 'false_infeasible')}, 'local_after': {k: local_after[k] for k in ('local_pass', 'cost_sign_errors', 'reach_sign_errors', 'false_feasible', 'false_infeasible')}, 'full_before': before, 'full_after': after}, ensure_ascii=False))
