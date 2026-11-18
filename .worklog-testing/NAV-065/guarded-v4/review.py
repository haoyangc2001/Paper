"""Main-agent verification of v4 saved results; no optimization or new rollouts."""
from datetime import datetime
import json
from pathlib import Path
import os
import socket
import sys

import numpy as np
import torch

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
LAUNCH = Path('/pub/data/caohy/work/camra_navigation/nav065_guarded_refit_launch_20261118_v4')
OUT = Path('/pub/data/caohy/work/camra_navigation/nav065_guarded_refit_v4')
sys.path[:0] = [str(NAV / 'scripts'), str(NAV / 'src')]
import refit_nav065_guarded as G

def read(p):
    return json.loads(p.read_text())

def rows(p):
    return [json.loads(l) for l in p.read_text().splitlines()]

def check_truth(ep, t, spec, inputs):
    z = np.asarray([ep['z0_J'], *(s['z_J'] for s in ep['trace'])])
    distance = np.asarray([np.linalg.norm(np.asarray(spec.start) - spec.goal),
                           *(s['goal_distance_m'] for s in ep['trace'])])
    y = np.asarray([0., *(s['safety_y'] for s in ep['trace'])])
    q = inputs.value['quantities']
    radius = inputs.prereg['environment']['r_goal_m']
    h = np.where(distance <= radius, q['g_norm']['G_INSIDE'],
                 (distance - radius) / q['g_norm']['L_REF_m'])
    h[y > 0] = np.maximum(h[y > 0], y[y > 0] * q['C']['value'])
    hard_failure = ep['label'] in ('collision', 'out_of_bounds')
    if hard_failure:
        h[-1] = inputs.value['bootstrap']['FAIL_PENALTY']['value']
    score = np.maximum(h, -z / q['Z_REF_J']['value'])
    stop = np.full(len(ep['trace']), len(ep['trace']), dtype=int) if hard_failure else np.asarray([
        i + np.argmin(score[i:]) for i in range(len(ep['trace']))])
    assert np.array_equal(stop, t.stop_index)
    assert np.allclose(h[stop], t.reach_target, rtol=0, atol=1e-12)
    assert np.allclose(score[stop], t.path_residual, rtol=0, atol=1e-12)
    actual = ep['label'] == 'goal_reached' and y[-1] == 0 and z[-1] >= 0
    assert actual == ep['feasible'] == t.feasible
    assert np.isclose(z[0] - z[-1], ep['cost_model_J'], rtol=0, atol=1e-8)

assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
torch.set_num_threads(1)
p, pin = G.approved_registration(G.REGISTRATION)
assert read(LAUNCH / 'EXIT.json')['exit_code'] == 0
for name, sha in read(LAUNCH / 'source_before.json').items():
    G.R.verify_hash(NAV / name, sha)
upstream, reg, inputs, source, train, prepared = G.sources()
manifest = read(OUT / 'fitted_candidates.json')
train_gate = read(OUT / 'train_gate.json')
source_by = {c['run_id']: c for c in source['candidates']}
current_by = {c['run_id']: c for c in manifest['candidates']}
gate_by = {c['run_id']: c for c in train_gate['candidates']}
assert manifest['refit_pin'] == pin and manifest['new_rollouts'] == 0
assert train_gate['pass'] and len(current_by) == len(gate_by) == 6
train_cases, updates, trials, preserved = [], 0, 0, 0
design = G.R.E.M.load_design(G.R.E.DESIGN)
train_maps = G.R.load(Path(G.R.E.SPLIT))['splits']['train']['maps']
specs = {m['map_id']: G.R.E.M.build(m['level'], m['seed'], design) for m in train_maps}
raw_train = G.B.S.saved_train(inputs, train_maps, reg['grid_k'], G.R.WORK / 'nav065_refit_v1', train)
for old, _, records, _ in prepared:
    rid = old['run_id']
    c = current_by[rid]
    G.R.verify_hash(Path(c['checkpoint']), c['checkpoint_sha256'])
    model = G.R.load_network(inputs, c, c['checkpoint'])
    tensors = G.R.digests(model)
    assert tensors == c['state_dict_sha256']
    assert tensors['policy'] == old['state_dict_sha256']['policy']
    log = rows(OUT / (rid + '.jsonl'))
    changes = [r for r in log if r['kind'] == 'trial']
    assert [r['trial'] for r in changes] == list(range(1, len(changes) + 1))
    assert len(changes) <= 300
    n = sum(r['accepted'] for r in changes)
    assert n == c['guarded_refit_updates'] == gate_by[rid]['updates']
    updates += n
    trials += len(changes)
    if rid != G.TARGET:
        assert n == 0 and tensors == old['state_dict_sha256']
        preserved += 1
    else:
        assert n == 1 and len(changes) == 1
        assert all(tensors[k] != old['state_dict_sha256'][k] for k in G.R.SCORE_ROLES)
    gate = G.B.train_summary(model, records, reg, 550.)
    assert gate == {k: v for k, v in gate_by[rid].items() if k not in ('run_id', 'updates')}
    for (map_id, k), t in records:
        ep = raw_train[rid][map_id, k]
        check_truth(ep, t, specs[map_id], inputs)
        vc, vh = G.B.S.predict(model, t.observations)
        train_cases.append({'run_id': rid, 'map_id': map_id, 'k': k,
            'actual_feasible': t.feasible, 'predicted_feasible': bool(max(vh[0], vc[0]-t.budget_J[0]/550.) <= 0),
            'true_W': float(t.path_residual[0]), 'predicted_W': float(max(vh[0], vc[0]-t.budget_J[0]/550.)),
            'cost_model_J': ep['cost_model_J'], 'budget_J': ep['z0_J']})
assert len(train_cases) == 540 and trials == updates == manifest['additional_updates'] == 1
assert not any(not c['actual_feasible'] and c['predicted_feasible'] for c in train_cases)

raw_dev, old_scores = G.R.fixed_dev(inputs, reg['grid_k'], G.R.WORK / 'nav065_deployment_v1')
dev = rows(OUT / 'development_scores.jsonl')
assert len(dev) == len({(r['run_id'], r['map_id'], r['k']) for r in dev}) == 180
dev_specs = {m['map_id']: G.R.E.M.build(m['level'], m['seed'], design) for m in inputs.maps}
false_cases = []
for r in dev:
    rid, mid, k = r['run_id'], r['map_id'], r['k']
    ep = raw_dev[rid][mid, k]
    t = G.R.reconstruct(ep, dev_specs[mid], inputs)
    check_truth(ep, t, dev_specs[mid], inputs)
    assert t.feasible == r['actual_feasible'] == old_scores[rid, mid, k]
    for name, arr in [('shared_stop', t.stop_index), ('vc_target_norm', t.cost_target),
                      ('vh_target', t.reach_target), ('path_W', t.path_residual)]:
        assert np.allclose(r['rows'][name], arr, rtol=0, atol=1e-12)
    w = np.maximum(r['rows']['vh'], np.asarray(r['rows']['vc_norm']) - t.budget_J / 550.)
    assert np.allclose(w, r['rows']['W'], rtol=0, atol=1e-12)
    assert bool(w[0] <= 0) == r['predicted_feasible']
    if not t.feasible and r['predicted_feasible']:
        false_cases.append({'run_id': rid, 'map_id': mid, 'k': k, 'label': ep['label'],
            'budget_J': ep['z0_J'], 'cost_model_J': ep['cost_model_J'],
            'overspend_J': -ep['z_J_end'], 'true_W': float(t.path_residual[0]),
            'predicted_W': float(w[0]), 'true_common_stop_cost_J': float(t.cost_target[0]*550.),
            'predicted_common_stop_cost_J': r['rows']['vc_norm'][0]*550.,
            'true_reach': float(t.reach_target[0]), 'predicted_reach': r['rows']['vh'][0]})
score = read(OUT / 'deployment_critic_validation_v2.json')
for summary in score['candidates']:
    rid = summary['run_id']
    assert G.R.summarize([r for r in dev if r['run_id'] == rid], set(dev_specs), reg) == {
        k: v for k, v in summary.items() if k != 'run_id'}
assert len(false_cases) == 4
assert all(r['actual_feasible'] and r['predicted_feasible'] for r in dev if r['k'] == 1.15)
g4 = read(OUT / 'gate_G4_alignment_v2.json')
assert g4['verdict'] == 'FAIL' and not (OUT / 'deployment_candidates_v2.json').exists()
assert not (OUT / 'HELD_TRAIN_GATE.json').exists()
report = {'task':'NAV-065','reviewer':'main_agent','independent':False,
    'skill':'create-plan-worklog-legacy','timestamp':datetime.now().astimezone().isoformat(),
    'host':socket.getfqdn(),'registration_pin':pin,'execution_integrity_pass':True,'scientific_pass':False,
    'run_status':'FAIL_DEVELOPMENT','train_episodes':540,'train_steps':207298,
    'new_rollouts':0,'optimizer_step_trials':trials,'accepted_updates':updates,
    'actor_unchanged_candidates':6,'other_candidates_preserved':preserved,
    'train_pass_candidates':6,'train_false_feasible':0,
    'development_scoring_rounds':1,'development_episodes':180,
    'development_pass_candidates':sum(c['pass'] for c in score['candidates']),
    'development_primary_correct':36,'development_false_feasible':4,
    'false_feasible_cases':false_cases,'train_candidates':train_gate['candidates'],
    'development_candidates':score['candidates'],
    'evidence':{'launch':str(LAUNCH),'output':str(OUT),'source_hashes':str(LAUNCH/'source_before.json'),
        'output_hashes':{f.name:G.R.file_sha(f) for f in OUT.iterdir() if f.is_file()}}}
for path, content in [(LAUNCH/'review.json', report),(LAUNCH/'train_cases.json',train_cases),
    (NAV/'reports/nav065_guarded_refit_v4_review.json',report)]:
    with path.open('x') as f:
        json.dump(content,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('train_candidates','development_candidates')},
                 ensure_ascii=False,indent=2))
