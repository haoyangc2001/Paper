"""Main-agent read-only audit of the single v5 attempt; no rollouts or updates."""
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
import torch

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
LAUNCH = Path('/pub/data/caohy/work/camra_navigation/nav065_adaptive_refit_launch_20261118_v5')
OUT = Path('/pub/data/caohy/work/camra_navigation/nav065_adaptive_refit_v5')
sys.path[:0] = [str(NAV / 'scripts'), str(NAV / 'src')]
import refit_nav065_adaptive as A
R = A.R

def read(path):
    return json.loads(path.read_text())

def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]

def truth_check(ep, truth, spec, inputs):
    z = np.asarray([ep['z0_J'], *(s['z_J'] for s in ep['trace'])])
    distance = np.asarray([np.linalg.norm(np.asarray(spec.start) - spec.goal),
                           *(s['goal_distance_m'] for s in ep['trace'])])
    y = np.asarray([0., *(s['safety_y'] for s in ep['trace'])])
    q = inputs.value['quantities']
    radius = inputs.prereg['environment']['r_goal_m']
    h = np.where(distance <= radius, q['g_norm']['G_INSIDE'],
                 (distance - radius) / q['g_norm']['L_REF_m'])
    h[y > 0] = np.maximum(h[y > 0], y[y > 0] * q['C']['value'])
    hard = ep['label'] in ('collision', 'out_of_bounds')
    if hard:
        h[-1] = inputs.value['bootstrap']['FAIL_PENALTY']['value']
    score = np.maximum(h, -z / q['Z_REF_J']['value'])
    n = len(ep['trace'])
    stops = np.full(n, n, dtype=int) if hard else np.asarray([
        i + np.argmin(score[i:]) for i in range(n)])
    cost = ((z[:-1] - z[-2]) / q['Z_REF_J']['value'] + inputs.value['bootstrap']['FAIL_PENALTY']['value']
            if hard else (z[:-1] - z[stops]) / q['Z_REF_J']['value'])
    assert np.array_equal(stops, truth.stop_index)
    for actual, expected in ((h[stops], truth.reach_target), (cost, truth.cost_target),
                             (score[stops], truth.path_residual)):
        assert np.allclose(actual, expected, rtol=0, atol=1e-12)
    feasible = ep['label'] == 'goal_reached' and y[-1] == 0 and z[-1] >= 0
    assert feasible == ep['feasible'] == truth.feasible
    assert np.isclose(z[0] - z[-1], ep['cost_model_J'], rtol=0, atol=1e-8)

assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
torch.set_num_threads(1)
p, pin = A.approved_registration(NAV / 'config/nav065_adaptive_refit_prereg_v5.json')
assert read(LAUNCH / 'EXIT.json')['returncode'] == 0
for name, digest in read(LAUNCH / 'source_before.json').items():
    R.verify_hash(NAV / name, digest)
upstream, reg, inputs, train, maps, refs, prepared = A.sources()
manifest, gate = read(OUT / 'fitted_candidates.json'), read(OUT / 'train_gate.json')
assert manifest['refit_pin'] == pin
assert manifest['train_journal_sha256'] == R.file_sha(OUT / 'train_episodes.jsonl')
scale = inputs.value['quantities']['Z_REF_J']['value']
sources = {c['run_id']: (c, model, old) for c, model, old, _ in prepared}
ctx = {c['variant']: R.E.build_rollout_ctx(c['variant']) for c, *_ in prepared}
specs = {(variant, m['map_id']): R.E.M.build(m['level'], m['seed'], context['design'])
         for variant, context in ctx.items() for m in maps}
raw_old = A.B.S.saved_train(inputs, maps, reg['grid_k'], R.WORK / 'nav065_refit_v1', train)
new, new_raw, points = {}, {}, {}
for rid, (candidate, model, old) in sources.items():
    new[rid], new_raw[rid] = [], {}
    for (mid, k), t in old:
        points.setdefault((rid, mid), {})[k] = t.feasible
pending, n_steps, n_episodes, reversals = None, 0, 0, 0
for row in rows(OUT / 'train_episodes.jsonl'):
    kind = row['kind']
    if kind == 'start':
        assert row['registration_pin'] == pin
    elif kind == 'rollout_start':
        assert pending is None and row['split'] == 'train'
        rid, mid, k = row['run_id'], row['map_id'], row['k']
        expected_k, interval = A.next_budget(points[rid, mid])
        assert expected_k == k and interval == row['interval'] and k not in points[rid, mid]
        pending = (rid, mid, k)
    elif kind == 'rollout_result':
        rid, mid, k = row['run_id'], row['map_id'], row['k']
        assert pending == (rid, mid, k)
        pending = None
        candidate, model, _ = sources[rid]
        ep = row['episode']
        assert np.isclose(ep['z0_J'], k * R.consistent_e_ref(refs[mid], candidate['variant']), rtol=1e-12)
        t = R.reconstruct(ep, specs[candidate['variant'], mid], inputs)
        truth_check(ep, t, specs[candidate['variant'], mid], inputs)
        with torch.inference_mode():
            actions = model.act(torch.as_tensor(t.observations, dtype=torch.float32), deterministic=True)[0].numpy()
        assert np.allclose(actions, [s['action'] for s in ep['trace']], atol=1e-6, rtol=0)
        assert not t.budget_feature_clipped.any()
        new[rid].append(((mid, k), t))
        new_raw[rid][mid, k] = ep
        points[rid, mid][k] = t.feasible
        n_steps += ep['n_steps']
        n_episodes += 1
    elif kind == 'map_sampling_finished':
        intervals = A.adjacent_intervals(points[row['run_id'], row['map_id']])
        assert row['intervals'] == intervals
        reversals += sum(i['reversal'] for i in intervals)
    else:
        raise AssertionError('unexpected journal row')
assert pending is None and n_episodes == 432 and n_steps <= 194400
assert all(len(v) == 9 for v in points.values()) and len(points) == 108
assert all(len(v) == 72 for v in new.values())
by_gate = {g['run_id']: g for g in gate['candidates']}
audit_candidates, train_cases, trials, updates = [], [], 0, 0
assert len(manifest['candidates']) == len(by_gate) == 6
for current in manifest['candidates']:
    rid = current['run_id']
    parent, _, old = sources[rid]
    R.verify_hash(Path(current['checkpoint']), current['checkpoint_sha256'])
    model = R.load_network(inputs, current, current['checkpoint'])
    assert R.digests(model) == current['state_dict_sha256']
    assert R.digests(model)['policy'] == parent['state_dict_sha256']['policy']
    log = rows(OUT / (rid + '.jsonl'))
    trial_rows = [r for r in log if r['kind'] == 'trial']
    assert [r['trial'] for r in trial_rows] == list(range(1, len(trial_rows) + 1))
    assert len(trial_rows) <= 300
    accepted = [r for r in trial_rows if r['accepted']]
    trials += len(trial_rows)
    updates += len(accepted)
    declared = by_gate[rid]
    assert declared['trials'] == len(trial_rows) and declared['updates'] == len(accepted)
    recalculated = A.train_gate(model, old, new[rid], reg, p['fit'], scale)
    assert recalculated == {k:v for k,v in declared.items() if k not in ('run_id','trials','updates')}
    if not trial_rows:
        assert R.digests(model) == parent['state_dict_sha256']
    if recalculated['pass'] and trial_rows:
        assert accepted[-1]['trial'] == trial_rows[-1]['trial']
        assert sum(r['report']['pass'] for r in accepted) == 1
    batch = A.selected_rows(old + new[rid], p['fit'])
    vc, vh = A.B.S.predict(model, batch[0])
    actual = np.maximum(batch[3], batch[2] - batch[1] / scale) <= 0
    predicted = np.maximum(vh, vc - batch[1] / scale) <= 0
    selected_fp = int(np.count_nonzero(predicted & ~actual))
    selected_fn = int(np.count_nonzero(~predicted & actual))
    for origin, records, raw in [('old', old, raw_old[rid]), ('new', new[rid], new_raw[rid])]:
        for (mid, k), t in records:
            ep = raw[mid, k]
            truth_check(ep, t, specs[parent['variant'], mid], inputs)
            vc, vh = A.B.S.predict(model, t.observations[:1])
            w = float(max(vh[0], vc[0] - t.budget_J[0] / scale))
            train_cases.append({'run_id':rid,'map_id':mid,'k':k,'origin':origin,
                'actual_feasible':t.feasible,'predicted_feasible':w <= 0,'true_W':float(t.path_residual[0]),
                'predicted_W':w,'label':ep['label'],'budget_J':ep['z0_J'],'cost_model_J':ep['cost_model_J']})
    audit_candidates.append({**declared, 'selected_false_feasible':selected_fp,
        'selected_false_infeasible':selected_fn, 'initial_gate':log[0],
        'final_gate':log[-1]})
assert len(train_cases) == 972
assert trials <= 1800 and manifest['resource_usage'] == {
    'episodes':n_episodes,'steps':n_steps,'trials':trials,'updates':updates}
assert manifest['train_episodes'] == 972 and manifest['train_steps'] == train['train_steps'] + n_steps
assert gate['pass'] == all(g['pass'] for g in gate['candidates'])
report = {'task':'NAV-065','reviewer':'main_agent','independent':False,'skill':'create-plan-worklog-legacy',
    'host':socket.getfqdn(),'tmux':os.environ['TMUX'],'python':sys.executable,
    'timestamp':datetime.now().astimezone().isoformat(),'registration_pin':pin,
    'execution_integrity_pass':True,'scientific_pass':False,'run_status':'HELD_TRAIN_GATE',
    'old_train_episodes':540,'new_train_episodes':n_episodes,'new_train_steps':n_steps,
    'train_steps_total':manifest['train_steps'],'optimizer_trials':trials,'accepted_updates':updates,
    'actor_unchanged_candidates':6,'train_pass_candidates':sum(g['pass'] for g in gate['candidates']),
    'train_false_feasible':sum(not c['actual_feasible'] and c['predicted_feasible'] for c in train_cases),
    'observed_budget_reversals':int(reversals),'development_scoring_rounds':0,
    'train_candidates':audit_candidates,'original_gate_verdict':'FAIL',
    'evidence':{'launch':str(LAUNCH),'output':str(OUT),
        'output_hashes':{f.name:R.file_sha(f) for f in OUT.iterdir() if f.is_file()}}}
if not gate['pass']:
    held = read(OUT / 'HELD_TRAIN_GATE.json')
    assert held['development_scoring_rounds'] == 0 and held['resource_usage'] == manifest['resource_usage']
    for name in ('development_scores.jsonl','gate_G4_alignment_v2.json','deployment_candidates_v2.json'):
        assert not (OUT / name).exists()
else:
    dev = rows(OUT / 'development_scores.jsonl')
    assert len(dev) == len({(r['run_id'],r['map_id'],r['k']) for r in dev}) == 180
    score, g4 = read(OUT / 'deployment_critic_validation_v2.json'), read(OUT / 'gate_G4_alignment_v2.json')
    for summary in score['candidates']:
        assert R.summarize([r for r in dev if r['run_id']==summary['run_id']], {m['map_id'] for m in inputs.maps}, reg) == {
            k:v for k,v in summary.items() if k!='run_id'}
    report.update(development_scoring_rounds=1,development_episodes=180,
        development_pass_candidates=sum(s['pass'] for s in score['candidates']),
        development_false_feasible=sum(not r['actual_feasible'] and r['predicted_feasible'] for r in dev),
        scientific_pass=g4['verdict']=='PASS',original_gate_verdict=g4['verdict'],
        run_status='PASS' if g4['verdict']=='PASS' else 'FAIL_DEVELOPMENT')
    assert (OUT / 'deployment_candidates_v2.json').exists() == report['scientific_pass']
for path,payload in [(LAUNCH/'review.json',report),(LAUNCH/'train_cases.json',train_cases),
                     (NAV/'reports/nav065_adaptive_refit_v5_review.json',report)]:
    with path.open('x') as stream:
        json.dump(payload,stream,ensure_ascii=False,indent=2);stream.write('\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('train_candidates','evidence')},ensure_ascii=False,indent=2))
