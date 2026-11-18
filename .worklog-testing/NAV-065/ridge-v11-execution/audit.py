"""Legacy main-agent audit of the saved result; no additional fit or rollout."""
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
import probe_nav065_ridge as G

def forbidden(*args, **kwargs):
    raise AssertionError('audit cannot solve, update, sample, or score development')

G.solve_head = G.H.solve_head = np.linalg.lstsq = forbidden
torch.optim.Adam.step = torch.optim.SGD.step = forbidden
G.R.evaluate = forbidden
if hasattr(G.R, 'rollout'):
    G.R.rollout = forbidden
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
read = lambda p: json.loads(p.read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
protocol, pin = G.approved_registration(NAV / 'config/nav065_ridge_probe_prereg_v11.json')
execution = read(HERE / 'execution.json')
assert execution['exit_code'] == 0 and execution['pin'] == pin
out = G.R.WORK / protocol['output_name']
result, preflight = read(out / 'RIDGE_FINISHED.json'), read(out / 'preflight.json')
assert result['pin'] == pin and not (out / 'STOPPED.json').exists()
attempts = [json.loads(line) for line in (out / 'attempts.jsonl').read_text().splitlines()]
assert attempts == [{'kind': 'head_solve_attempt', 'role': role} for role in ('critic_cost', 'critic_reach')]
assert sum(row['head_solves'] for row in result['head_solves'].values()) == 2
assert result['optimizer_steps'] == result['new_rollouts'] == result['development_scoring_rounds'] == 0
assert not result['deployment_allowed'] and not result['scientific_nav065_pass'] and not result['automatic_expansion_allowed']
upstream, reg, inputs, prepared = G.L.sources()
source_hashes = {c['run_id']: G.R.digests(m) for c, m, _, _ in prepared}
assert source_hashes == read(NAV / 'reports/nav065_local_probe_v9_review.json')['source_state_dict_hashes']
c, source, old, new = G.selected_source(prepared, protocol)
checkpoint = out / 'diagnostic-ridge.pt'
assert sha(checkpoint) == result['checkpoint_sha256']
model = G.R.load_network(inputs, c, checkpoint)
allowed = {f'{role}.head.{name}' for role in G.R.SCORE_ROLES for name in ('weight', 'bias')}
changed = [name for name, tensor in source.state_dict().items() if not torch.equal(tensor, model.state_dict()[name])]
assert set(changed) <= allowed and len(changed) == 4
settings = read(G.L.P.PROPOSAL)['fit']
batch, weights, groups = G.training_batch(old, new, settings)
assert [row['rows'] for row in groups] == [33223, 26654, 162, 866]
assert len(weights) == 60905 and len(old) + len(new) == 162
raw_hashes = [hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for x in (*batch, weights)]
assert raw_hashes == preflight['raw_target_weight_sha256']
assert raw_hashes == read(NAV / 'reports/nav065_ridge_preflight_v11.json')['raw_target_weight_sha256']
scale = inputs.value['quantities']['Z_REF_J']['value']
local_batch = G.L.local_batch(old + new, [tuple(k) for k in preflight['local_keys']])
local_before, local_after = (G.L.evaluate_local(net, local_batch, reg, scale) for net in (source, model))
assert local_after == result['local']
before = G.L.P.A.train_gate(source, old, new, reg, settings, scale)
after = G.L.P.A.train_gate(model, old, new, reg, settings, scale)
assert before == preflight['source_full_train_gate'] == result['full_train_before']
assert after == result['full_train_after']
assert before['original_grid']['policy_actual_feasible_by_budget'] == after['original_grid']['policy_actual_feasible_by_budget']
before_groups, after_groups = (G.group_metrics(net, batch, weights, groups, scale) for net in (source, model))
assert before_groups == preflight['groups_before'] == result['groups_before']
assert after_groups == result['groups_after']
precision = {}
for role, i in (('critic_cost', 2), ('critic_reach', 3)):
    a, y, p, w, design = G.weighted_design(getattr(source, role), batch[0], batch[i], weights, protocol['solve'])
    assert design == preflight['ridge_designs'][role]
    fitted = G.H.head_vector(getattr(model, role))
    delta = fitted - w
    recorded = result['head_solves'][role]
    norm = float(np.linalg.norm(delta))
    assert norm == recorded['actual_float32_correction_norm']
    objective = float(np.dot(p, (a @ fitted - y) ** 2) / design['target_rms'] ** 2 + design['ridge_coefficient'] * np.dot(delta, delta))
    assert objective == recorded['regularized_objective_after_float32_cast']
    assert objective <= design['source_objective'] + 1e-6 * max(1., design['source_objective'])
    assert norm <= np.sqrt(objective / design['ridge_coefficient']) + 1e-10
    gradient = a.T @ (p * (a @ fitted - y)) / design['target_rms'] ** 2 + design['ridge_coefficient'] * delta
    precision[role] = {'source_objective': design['source_objective'], 'fitted_objective': objective,
        'actual_float32_correction_norm': norm,
        'conditional_source_objective_correction_bound': design['correction_norm_bound_for_objective_no_worse_than_source'],
        'float32_stationarity_residual_linf': float(np.abs(gradient).max()),
        'ridge_rank': recorded['rank'], 'design_columns': design['columns']}
starts = {stage: G.L.P.S.start_snapshot(net, old, new, scale, reg['primary_k'])
          for stage, net in (('before', source), ('after', model))}
assert result['starts_after'] == starts['after']
confusion = {stage: dict(Counter(row['classification'] for row in rows)) for stage, rows in starts.items()}
assert all(len(rows) == 162 for rows in starts.values())
for name, digest in read(HERE / 'baseline.json')['v2_verified_files'].items():
    assert sha(Path(name)) == digest, name
assert source_hashes == {candidate['run_id']: G.R.digests(net) for candidate, net, _, _ in prepared}
assert '38 passed' in (ROOT / '.worklog-testing/NAV-065/ridge-v11-preparation/basic-attempt1.log').read_text()
report = {'schema': 'camra-nav065-ridge-probe-review-v11', 'timestamp': datetime.now().astimezone().isoformat(),
    'host': socket.getfqdn(), 'tmux': os.environ['TMUX'], 'python': sys.executable,
    'base_git': read(HERE / 'baseline.json')['git'], 'run_git': execution['git'], 'approval_pin': pin, 'integrity_pass': True,
    'candidate': c['run_id'], 'source_checkpoint_sha256': c['checkpoint_sha256'],
    'available_train_episodes': len(old) + len(new), 'available_train_steps': sum(len(t.observations) for _, t in old + new),
    'weighted_group_rows': len(weights), 'formal_candidate_head_solves': 2, 'audit_head_solves': 0,
    'optimizer_steps': 0, 'new_rollouts': 0, 'development_scoring_rounds': 0,
    'basic_tests_passed': 38, 'basic_test_code_hashes_match_approved_registration': True,
    'source_models_unchanged': True, 'all_six_actors_unchanged': True, 'diagnostic_actor_and_bodies_unchanged': True,
    'changed_diagnostic_parameters': changed, 'source_state_dict_hashes': source_hashes,
    'diagnostic_state_dict_hashes': G.R.digests(model), 'v2_files_verified': 33, 'checkpoint_sha256': sha(checkpoint),
    'precision': precision, 'groups_before': before_groups, 'groups_after': after_groups,
    'local_before': local_before, 'local_after': local_after,
    'full_train_before': before, 'full_train_after': after, 'all_train_start_confusion': confusion,
    'all_train_starts': starts, 'scientific_pass': False, 'original_G4': 'FAIL',
    'NAV065': False, 'NAV066_started': False, 'completed_tasks': 52, 'deployment_allowed': False,
    'evidence': {'output': str(out), 'output_hashes': {p.name: sha(p) for p in out.iterdir() if p.is_file()},
        'registration_sha256': pin, 'implementation_hashes': protocol['implementation_hashes'],
        'execution_metadata': str(HERE / 'execution.json')},
    'limitations': ['Full saved train coverage and source-relative ridge do not guarantee each sign or generalization.',
        'This result concerns this fixed v8 body, one regularization value and the reviewed weighting; it cannot establish architecture impossibility.',
        'The diagnostic copy is not deployed; actor behavior and saved rollout outcomes are unchanged.',
        'No further solve, optimizer step, development evaluation or downstream task is authorized.']}
with (NAV / 'reports/nav065_ridge_probe_v11_review.json').open('x') as f:
    json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    f.write('\n')
print(json.dumps({k: report[k] for k in ('integrity_pass', 'formal_candidate_head_solves', 'audit_head_solves', 'basic_tests_passed', 'all_train_start_confusion', 'precision', 'scientific_pass', 'NAV065')}, ensure_ascii=False))
print(json.dumps({'full_before': before, 'full_after': after, 'groups_before': before_groups, 'groups_after': after_groups}, ensure_ascii=False))
