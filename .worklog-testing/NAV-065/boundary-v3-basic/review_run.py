"""Main-agent result checks for the approved single v3 attempt."""

from datetime import datetime
import json
import math
import os
from pathlib import Path
import socket
import sys

import numpy as np
import torch

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
LAUNCH = Path('/pub/data/caohy/work/camra_navigation/nav065_boundary_refit_launch_20261118_v3')
OUT = Path('/pub/data/caohy/work/camra_navigation/nav065_boundary_refit_v3')
sys.path[:0] = [str(NAV / 'scripts'), str(NAV / 'src')]
import refit_nav065_boundary as B


def read(path):
    return json.loads(path.read_text())


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
torch.set_num_threads(1)
exit_record = read(LAUNCH / 'EXIT.json')
assert exit_record['exit_code'] == 0
protocol, pin = B.approved_registration(NAV / 'config/nav065_boundary_refit_prereg_v3.json')
assert pin == exit_record['registration_pin']
for name, digest in read(LAUNCH / 'source_before.json').items():
    B.R.verify_hash(NAV / name, digest)
upstream, reg, inputs, maps, source, train, raw = B.load_sources(protocol)
manifest = read(OUT / 'fitted_candidates.json')
gate = read(OUT / 'train_gate.json')
assert manifest['refit_pin'] == pin and manifest['new_rollouts'] == 0
assert (manifest['train_episodes'], manifest['train_steps']) == (540, 207298)
assert len(manifest['candidates']) == len(gate['candidates']) == 6
source_by_id = {c['run_id']: c for c in source['candidates']}
gate_by_id = {g['run_id']: g for g in gate['candidates']}
design = B.R.E.M.load_design(B.R.E.DESIGN)
specs = {m['map_id']: B.R.E.M.build(m['level'], m['seed'], design) for m in maps}
scale = inputs.value['quantities']['Z_REF_J']['value']
candidates, all_cases, total_updates, total_steps = [], [], 0, 0
for c in manifest['candidates']:
    rid = c['run_id']
    B.R.verify_hash(Path(c['checkpoint']), c['checkpoint_sha256'])
    model = B.R.load_network(inputs, c, c['checkpoint'])
    tensors = B.R.digests(model)
    before = source_by_id[rid]['state_dict_sha256']
    assert tensors == c['state_dict_sha256']
    assert tensors['policy'] == before['policy']
    journal = records(OUT / (rid + '.jsonl'))
    assert journal[0]['kind'] == 'start' and journal[0]['registration_pin'] == pin
    updates = [r for r in journal if r['kind'] == 'update']
    checks = [r for r in journal if r['kind'] == 'train_gate']
    n = len(updates)
    assert n == c['boundary_refit_updates'] == gate_by_id[rid]['updates']
    assert [r['update'] for r in updates] == list(range(1, n + 1))
    assert all(math.isfinite(r[k]) for r in updates for k in ('loss', 'regression_loss', 'sign_loss'))
    assert [r['update'] for r in checks] == list(range(0, n + 1, 10))
    assert checks[-1]['pass'] == gate_by_id[rid]['pass']
    if n:
        assert rid in protocol['resource_limit']['update_candidates']
        assert not any(r['pass'] for r in checks[:-1])
        assert checks[-1]['pass'] or n == 300
        assert all(tensors[k] != before[k] for k in B.R.SCORE_ROLES)
    else:
        assert checks[0]['pass'] and tensors == before
    total_updates += n
    truth_predictions, confusion = [], {}
    absolute_cost_error, absolute_h_error, targets_c, targets_h = [], [], [], []
    for m in maps:
        for k in reg['grid_k']:
            ep = raw[rid][m['map_id'], k]
            t = B.R.reconstruct(ep, specs[m['map_id']], inputs)
            vc, vh = B.S.predict(model, t.observations)
            w = np.maximum(vh, vc - t.budget_J / scale)
            actual, predicted = t.feasible, bool(w[0] <= 0)
            assert not t.budget_feature_clipped.any()
            assert actual == ep['feasible'] == bool(t.path_residual[0] <= 0)
            counts = confusion.setdefault(str(k), dict(total=0, true_feasible=0,
                  false_feasible=0, false_infeasible=0, true_infeasible=0, path_sign_mismatches=0))
            counts['total'] += 1
            classification = ('true_feasible' if predicted else 'false_infeasible') if actual else (
                'false_feasible' if predicted else 'true_infeasible')
            counts[classification] += 1
            counts['path_sign_mismatches'] += int(np.count_nonzero((w <= 0) != (t.path_residual <= 0)))
            all_cases.append({'run_id': rid, 'map_id': m['map_id'], 'k': k,
                'actual_feasible': actual, 'predicted_feasible': predicted, 'label': ep['label'],
                'true_W': float(t.path_residual[0]), 'predicted_W': float(w[0]),
                'true_cost_target_J': float(t.cost_target[0] * scale),
                'predicted_cost_J': float(vc[0] * scale), 'budget_J': float(t.budget_J[0]),
                'true_reach': float(t.reach_target[0]), 'predicted_reach': float(vh[0]),
                'cost_model_J': ep['cost_model_J'], 'final_budget_J': ep['z_J_end']})
            absolute_cost_error.extend(np.abs(vc - t.cost_target))
            absolute_h_error.extend(np.abs(vh - t.reach_target))
            targets_c.extend(np.abs(t.cost_target))
            targets_h.extend(np.abs(t.reach_target))
            total_steps += len(t.observations)
    g = gate_by_id[rid]
    assert confusion == g['by_budget']
    assert np.isclose(np.mean(absolute_cost_error), g['metrics']['vc_norm']['mae'], atol=1e-12)
    assert np.isclose(np.mean(absolute_h_error), g['metrics']['vh']['mae'], atol=1e-12)
    assert np.isclose(np.mean(absolute_h_error)/np.mean(targets_h),
                      g['metrics']['vh']['relative_mae'], atol=1e-12)
    primary = confusion[str(reg['primary_k'])]
    assert primary['true_feasible'] + primary['false_infeasible'] == g['primary_actual_feasible']
    candidates.append({'run_id': rid, 'updates': n, 'actor_unchanged': True,
        'critics_unchanged': tensors == before, 'train_pass': g['pass'], 'failures': g['failures'],
        'primary_actual_feasible': g['primary_actual_feasible'], 'primary_total': g['primary_total'],
        'primary_correct_accepts': primary['true_feasible'],
        'false_feasible': sum(b['false_feasible'] for b in confusion.values()),
        'false_infeasible': sum(b['false_infeasible'] for b in confusion.values()),
        'cost_relative_mae': g['metrics']['vc_norm']['relative_mae'],
        'reach_relative_mae': g['metrics']['vh']['relative_mae'],
        'first_loss': updates[0]['loss'] if updates else None,
        'last_loss': updates[-1]['loss'] if updates else None,
        'first_sign_loss': updates[0]['sign_loss'] if updates else None,
        'last_sign_loss': updates[-1]['sign_loss'] if updates else None})

assert total_steps == 207298
assert total_updates == manifest['additional_updates'] <= 600
all_pass = all(c['train_pass'] for c in candidates)
assert gate['pass'] == all_pass
development_rounds = 0
if not all_pass:
    assert read(OUT / 'HELD_TRAIN_GATE.json')['development_scoring_rounds'] == 0
    for name in ('development_scores.jsonl', 'gate_G4_alignment_v2.json',
                 'deployment_candidates_v2.json', 'deployment_critic_validation_v2.json'):
        assert not (OUT / name).exists(), name
else:
    dev = records(OUT / 'development_scores.jsonl')
    assert len(dev) == 180
    assert len({(r['run_id'], r['map_id'], r['k']) for r in dev}) == 180
    score = read(OUT / 'deployment_critic_validation_v2.json')
    assert score['development_episodes'] == 180 and score['new_episodes'] == 0
    assert score['refit_pin'] == pin and score['eval_pin'] == protocol['evaluation_pin']
    g4 = read(OUT / 'gate_G4_alignment_v2.json')
    assert (OUT / 'deployment_candidates_v2.json').exists() == (g4['verdict'] == 'PASS')
    development_rounds = 1

baseline = read(NAV / 'reports/nav065_boundary_refit_preflight_v3.json')
baseline_ff = sum(b['false_feasible'] for c in baseline['candidates'] for b in c['by_budget'].values())
report = {'task': 'NAV-065', 'reviewer': 'main_agent', 'independent': False,
    'skill': 'create-plan-worklog-legacy', 'timestamp': datetime.now().astimezone().isoformat(),
    'host': socket.getfqdn(), 'registration_pin': pin, 'execution_integrity_pass': True,
    'scientific_pass': all_pass and read(OUT / 'gate_G4_alignment_v2.json')['verdict'] == 'PASS',
    'run_status': exit_record['status'], 'additional_updates': total_updates,
    'train_episodes': 540, 'train_steps': total_steps, 'new_rollouts': 0,
    'development_scoring_rounds': development_rounds,
    'train_pass_candidates': sum(c['train_pass'] for c in candidates),
    'train_false_feasible_before': baseline_ff,
    'train_false_feasible_after': sum(c['false_feasible'] for c in candidates),
    'train_false_infeasible_after': sum(c['false_infeasible'] for c in candidates),
    'actor_unchanged_candidates': 6, 'zero_update_preserved_candidates': sum(c['updates'] == 0 for c in candidates),
    'candidates': candidates,
    'evidence': {'output': str(OUT), 'launch': str(LAUNCH),
                 'source_hashes': str(LAUNCH / 'source_before.json'),
                 'saved_tensor_manifest_sha256': B.R.file_sha(OUT / 'fitted_candidates.json'),
                 'train_gate_sha256': B.R.file_sha(OUT / 'train_gate.json')}}
with (LAUNCH / 'review.json').open('x') as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
with (LAUNCH / 'train_cases.json').open('x') as stream:
    json.dump(all_cases, stream, ensure_ascii=False, indent=1)
    stream.write('\n')
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
