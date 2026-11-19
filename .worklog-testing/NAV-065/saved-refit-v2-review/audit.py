"""Post-run, read-only verification of saved-refit-v2; never score development."""
from collections import Counter
from datetime import datetime
import copy
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
sys.path[:0] = [str(NAV / 'scripts'), str(NAV / 'src')]
import refit_nav065_saved as S
from analyze_nav065_boundary import case, predict, summarize
R = S.R
WORK = R.WORK
OUT = WORK / 'nav065_saved_refit_v2'
LAUNCH = WORK / 'nav065_saved_refit_launch_20261118_v1'
HERE = Path(__file__).parent
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
torch.set_num_threads(1)

def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def raw_check(ep, truth, spec, inputs):
    """Brute-force suffix argmin from the raw budget/distance/safety fields."""
    q = inputs.value['quantities']
    zref = q['Z_REF_J']['value']
    steps = ep['trace']
    z = np.asarray([ep['z0_J']] + [r['z_J'] for r in steps])
    y = np.asarray([0.] + [r['safety_y'] for r in steps])
    d = np.asarray([math.dist(spec.start, spec.goal)] + [r['goal_distance_m'] for r in steps])
    g = np.where(d <= inputs.prereg['environment']['r_goal_m'], q['g_norm']['G_INSIDE'],
                 (d - inputs.prereg['environment']['r_goal_m']) / q['g_norm']['L_REF_m'])
    h = np.where(y > 0, np.maximum(g, q['C']['value'] * y), g)
    hard = ep['label'] in ('collision', 'out_of_bounds')
    if hard:
        h[-1] = inputs.value['bootstrap']['FAIL_PENALTY']['value']
    score = np.maximum(h, -z / zref)
    stops = np.asarray([len(steps) if hard else t + np.argmin(score[t:]) for t in range(len(steps))])
    np.testing.assert_array_equal(stops, truth.stop_index)
    np.testing.assert_allclose(h[stops], truth.reach_target, atol=1e-12, rtol=0)
    np.testing.assert_allclose(score[stops], truth.path_residual, atol=1e-12, rtol=0)
    actual = ep['label'] == 'goal_reached' and all(r['safety_y'] == 0 for r in steps) and z[-1] >= 0
    assert actual == truth.feasible == ep['feasible']
    assert abs(z[0] - z[-1] - ep['cost_model_J']) < 1e-8
    return actual

protocol, pin = S.approved_registration(NAV / 'config/nav065_saved_refit_prereg_v2.json')
before, after = (R.load(LAUNCH / name) for name in ('source_before.json', 'source_after.json'))
assert before == after
for p, digest in before.items():
    R.verify_hash(Path(p), digest)
exit_info = R.load(LAUNCH / 'EXIT.json')
assert exit_info['exit_code'] == 0 and exit_info['state'] == 'HELD_TRAIN_GATE'
assert exit_info['guarded_inputs_unchanged'] is True and exit_info['registration_pin'] == pin
held = R.load(OUT / 'HELD_TRAIN_GATE.json')
assert held['new_rollouts'] == held['development_scoring_rounds'] == 0
_, _, reg, inputs, maps, _, _ = R.preflight()
old_root = WORK / protocol['source_run']
old_manifest = R.load(old_root / 'fitted_candidates.json')
raw = S.saved_train(inputs, maps, reg['grid_k'], old_root, old_manifest)
manifest = R.load(OUT / 'fitted_candidates.json')
gate = R.load(OUT / 'train_gate.json')
assert manifest['refit_pin'] == pin and manifest['new_rollouts'] == 0
assert len(manifest['candidates']) == len(gate['candidates']) == 6
expected_ids = {c['run_id'] for c in inputs.candidates}
assert {c['run_id'] for c in manifest['candidates']} == expected_ids
assert {c['run_id'] for c in gate['candidates']} == expected_ids
old_by_id = {c['run_id']: c for c in old_manifest['candidates']}
gate_by_id = {g['run_id']: g for g in gate['candidates']}
design = R.E.M.load_design(R.E.DESIGN)
specs = {m['map_id']: R.E.M.build(m['level'], m['seed'], design) for m in maps}
scale = inputs.value['quantities']['Z_REF_J']['value']
all_rows, old_rows, candidates = [], [], []
for c in manifest['candidates']:
    rid = c['run_id']
    old = old_by_id[rid]
    R.verify_hash(Path(c['checkpoint']), c['checkpoint_sha256'])
    assert c['parent_refit_checkpoint_sha256'] == old['checkpoint_sha256']
    new_model = R.load_network(inputs, c, c['checkpoint'])
    old_model = R.load_network(inputs, old, old['checkpoint'])
    assert R.digests(new_model) == c['state_dict_sha256']
    for key, tensor in old_model.policy.state_dict().items():
        assert torch.equal(tensor, new_model.policy.state_dict()[key]), (rid, key)
    assert all(R.digests(new_model)[role] != R.digests(old_model)[role] for role in R.SCORE_ROLES)
    journal = [json.loads(line) for line in (OUT / (rid + '.jsonl')).open()]
    assert journal[0]['kind'] == 'start' and journal[0]['registration_pin'] == pin
    assert journal[0]['host'].split('.')[0] == 'tiger11' and journal[0]['tmux']
    updates = [r for r in journal if r['kind'] == 'update']
    checks = [r for r in journal if r['kind'] == 'train_gate']
    n = c['additional_refit_updates']
    assert [r['update'] for r in updates] == list(range(1, n + 1))
    assert [r['update'] for r in checks] == list(range(10, n + 1, 10))
    assert len(journal) == 1 + len(updates) + len(checks)
    assert all(math.isfinite(r['loss']) for r in updates) and n <= 300
    assert not any(r['pass'] for r in checks[:-1])
    assert checks[-1]['pass'] or n == 300
    rows, records, current_cases = [], [], []
    for m in maps:
        mid = m['map_id']
        for k in reg['grid_k']:
            ep = raw[rid][mid, k]
            truth = R.reconstruct(ep, specs[mid], inputs)
            raw_check(ep, truth, specs[mid], inputs)
            records.append(((mid, k), truth))
            vc, vh = predict(new_model, truth.observations)
            rows.append({'map_id': mid, 'k': k, **R.value_comparison(truth, vc, vh, scale)})
            detail = {'run_id': rid, 'variant': c['variant'], 'seed': c['seed'],
                      'map_id': mid, 'level': m['level'], 'k': k,
                      'final_goal_distance_m': ep['trace'][-1]['goal_distance_m'],
                      **case(ep, truth, vc, vh, scale)}
            all_rows.append(detail)
            current_cases.append(detail)
            old_vc, old_vh = predict(old_model, truth.observations)
            old_rows.append({'run_id': rid, 'map_id': mid, 'k': k,
                             **case(ep, truth, old_vc, old_vh, scale)})
    train_reg = copy.deepcopy(reg)
    train_reg['criteria']['min_primary_true_feasible'] = 18
    recalculated = R.summarize(rows, set(specs), train_reg)
    assert {'run_id': rid, 'updates': n, **recalculated} == gate_by_id[rid]
    assert {k: v for k, v in checks[-1].items() if k not in ('kind', 'update')} == recalculated
    primary = [r for r in current_cases if r['k'] == reg['primary_k']]
    candidates.append({'run_id': rid, 'updates': n, 'train_pass': recalculated['pass'],
        'failures': recalculated['failures'], 'actor_tensors_unchanged': True,
        'both_critics_updated': True, 'cpu_reload_verified': True,
        'train_episodes': len(current_cases), 'by_budget': recalculated['by_budget'],
        'primary_actual_feasible': sum(r['actual_feasible'] for r in primary),
        'primary_required': 18, 'metrics': recalculated['metrics']})
    print(rid, n, recalculated['pass'], flush=True)

assert len(all_rows) == manifest['train_episodes'] == 540
assert sum(c['updates'] for c in candidates) == 640
assert sum(c['train_pass'] for c in candidates) == 4 and not gate['pass']
allowed = {'fitted_candidates.json', 'train_gate.json', 'HELD_TRAIN_GATE.json'}
allowed |= {c['run_id'] + suffix for c in candidates for suffix in ('.pt', '.jsonl')}
assert {p.name for p in OUT.iterdir()} == allowed, 'unexpected output / development scoring'
report = {'schema': 'camra-nav065-saved-refit-v2-review-v1', 'task': 'NAV-065',
    'timestamp': datetime.now().astimezone().isoformat(), 'host': socket.getfqdn(),
    'tmux': os.environ['TMUX'], 'cwd': os.getcwd(), 'python': sys.executable,
    'reviewer': {'agent_id': '/root', 'independent': False,
                 'reason': 'User explicitly waived dedicated test subagents.'},
    'integrity_verification': 'PASS', 'scientific_verdict': 'HELD_TRAIN_GATE',
    'registration_pin': pin, 'output_directory': str(OUT), 'launch_exit': exit_info,
    'new_rollouts': 0, 'additional_updates': 640, 'development_scoring_rounds': 0,
    'g4_created': False, 'deployment_bundle_created': False,
    'candidates': candidates, 'train_before': summarize(old_rows),
    'train_after': summarize(all_rows),
    'false_feasible_cases': [r for r in all_rows if r['false_feasible']],
    'primary_actual_failures': [r for r in all_rows if r['k'] == reg['primary_k'] and not r['actual_feasible']],
    'labels': dict(Counter(r['label'] for r in all_rows)),
    'guarded_inputs_verified': len(before),
    'input_hashes': before,
    'output_hashes': {p.name: R.file_sha(p) for p in sorted(OUT.iterdir())},
    'audit_script_sha256': R.file_sha(Path(__file__)),
    'limitations': ['Train replay uses the existing checked full-trajectory oracle, cross-checked by brute-force suffix argmin and raw J/event ledgers; no claim of independent tester identity.',
        'Four train passes are not four G4/deployment passes; development was not rescored.',
        'The train primary requirement 18/18 cannot be met by critic fitting alone where fixed actor actual feasibility is 17/18; the approved threshold is preserved.',
        'Per-candidate caps are exhausted for both failed candidates; unused aggregate allowance cannot be reassigned.']}
save(HERE / 'train_cases.json', all_rows)
save(NAV / 'reports/nav065_saved_refit_v2_review.json', report)
print(json.dumps({k: report[k] for k in ('integrity_verification', 'scientific_verdict', 'additional_updates', 'train_after', 'primary_actual_failures')}, ensure_ascii=False), flush=True)
