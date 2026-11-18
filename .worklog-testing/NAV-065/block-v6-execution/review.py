"""Main-agent read-only review of the approved v6; never fit or collect."""
from collections import Counter
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
LAUNCH = Path('/pub/data/caohy/work/camra_navigation/nav065_block_refit_launch_20261118_v6')
OUT = Path('/pub/data/caohy/work/camra_navigation/nav065_block_refit_v6')
sys.path[:0] = [str(NAV / 'scripts'), str(NAV / 'src')]
import refit_nav065_blocks as V
A, R = V.A, V.R
read = lambda path: json.loads(path.read_bytes())

def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]

def write(path, payload):
    with path.open('x') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def truth_check(ep, truth, spec, inputs):
    z = np.asarray([ep['z0_J'], *(s['z_J'] for s in ep['trace'])])
    distance = np.asarray([np.linalg.norm(np.asarray(spec.start) - spec.goal), *(s['goal_distance_m'] for s in ep['trace'])])
    y = np.asarray([0., *(s['safety_y'] for s in ep['trace'])])
    q = inputs.value['quantities']
    radius = inputs.prereg['environment']['r_goal_m']
    h = np.where(distance <= radius, q['g_norm']['G_INSIDE'], (distance - radius) / q['g_norm']['L_REF_m'])
    h[y > 0] = np.maximum(h[y > 0], y[y > 0] * q['C']['value'])
    hard = ep['label'] in ('collision', 'out_of_bounds')
    if hard:
        h[-1] = inputs.value['bootstrap']['FAIL_PENALTY']['value']
    score = np.maximum(h, -z / q['Z_REF_J']['value'])
    n = len(ep['trace'])
    stops = np.full(n, n, dtype=int) if hard else np.asarray([i + np.argmin(score[i:]) for i in range(n)])
    cost = ((z[:-1] - z[-2]) / q['Z_REF_J']['value'] + inputs.value['bootstrap']['FAIL_PENALTY']['value'] if hard else (z[:-1] - z[stops]) / q['Z_REF_J']['value'])
    assert np.array_equal(stops, truth.stop_index)
    for actual, expected in ((h[stops], truth.reach_target), (cost, truth.cost_target), (score[stops], truth.path_residual)):
        assert np.allclose(actual, expected, rtol=0, atol=1e-12)
    feasible = ep['label'] == 'goal_reached' and y[-1] == 0 and z[-1] >= 0
    assert feasible == ep['feasible'] == truth.feasible
    assert np.isclose(z[0] - z[-1], ep['cost_model_J'], rtol=0, atol=1e-8)

assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
torch.set_num_threads(1)
p, pin = V.approved_registration(NAV / 'config/nav065_block_refit_prereg_v6.json')
assert read(LAUNCH / 'EXIT.json')['returncode'] == 0
assert '63 passed' in (LAUNCH / 'tests.log').read_text()
for name, digest in read(LAUNCH / 'source_before.json').items():
    R.verify_hash(NAV / name, digest)
for path, digest in read(LAUNCH / 'registration.json')['v2_verified_files'].items():
    R.verify_hash(Path(path), digest)
upstream, reg, inputs, prepared = V.sources()
manifest, gates = read(OUT / 'fitted_candidates.json'), read(OUT / 'train_gate.json')
assert manifest['refit_pin'] == pin
assert (manifest['train_episodes'], manifest['train_steps']) == (972, 372471)
scale = inputs.value['quantities']['Z_REF_J']['value']
sources = {c['run_id']: (c, model, old, new) for c, model, old, new in prepared}
maps, _ = R.train_grid(inputs)
specs = {m['map_id']: R.E.M.build(m['level'], m['seed'], R.E.M.load_design(R.E.DESIGN)) for m in maps}
train_manifest = read(R.WORK / 'nav065_refit_v1/fitted_candidates.json')
raw_old = A.B.S.saved_train(inputs, maps, reg['grid_k'], R.WORK / 'nav065_refit_v1', train_manifest)
raw_new = {rid: {} for rid in sources}
for row in rows(R.WORK / 'nav065_adaptive_refit_v5/train_episodes.jsonl'):
    if row['kind'] == 'rollout_result':
        raw_new[row['run_id']][row['map_id'], row['k']] = row['episode']
assert sum(len(v) for v in raw_new.values()) == 432
by_gate = {g['run_id']: g for g in gates['candidates']}
assert len(manifest['candidates']) == len(by_gate) == 6
audited, train_cases = [], []
trials_total = updates_total = accepted_blocks_total = 0
for candidate in manifest['candidates']:
    rid = candidate['run_id']
    parent, source, old, new = sources[rid]
    initial = A.train_gate(source, old, new, reg, p['fit'], scale)
    R.verify_hash(Path(candidate['checkpoint']), candidate['checkpoint_sha256'])
    model = R.load_network(inputs, candidate, candidate['checkpoint'])
    assert R.digests(model) == candidate['state_dict_sha256']
    assert R.digests(model)['policy'] == parent['state_dict_sha256']['policy']
    actual = np.asarray([t.feasible for _, t in old + new])
    initial_correct = A.classify(source, old + new, scale) == actual
    final_prediction = A.classify(model, old + new, scale)
    assert np.all(final_prediction[initial_correct] == actual[initial_correct])
    log = rows(OUT / (rid + '.jsonl'))
    assert log[0] == {'kind': 'train_gate', 'trials': 0, **initial}
    trial_rows = [r for r in log if r['kind'] == 'trial']
    assert [r['trial'] for r in trial_rows] == list(range(1, len(trial_rows) + 1))
    assert len(trial_rows) <= p['resource_limit']['optimizer_trials_per_candidate_max']
    accepted, pending, current, accepted_blocks = 0, [], initial, 0
    rejected = Counter()
    for row in log[1:]:
        if row['kind'] == 'trial':
            assert np.isclose(row['learning_rate'], p['fit']['learning_rate'] * p['fit']['backtrack_factor'] ** row['backtrack'], rtol=0, atol=1e-16)
            assert 0 <= row['backtrack'] < p['fit']['block_backtracks_max']
            assert np.isfinite(row['regression_loss_before'])
            pending.append(row)
            assert len(pending) <= p['fit']['steps_per_block']
        elif row['kind'] == 'block_decision':
            assert len(pending) == row['steps'] and row['trials'] == pending[-1]['trial']
            assert all(t['backtrack'] == row['backtrack'] and not t['report']['pass'] for t in pending)
            if row['accepted']:
                # Recheck all report-level guards; intermediate per-state lock
                # protection is exercised by basic tests and final source locks.
                allowed, reason = V.block_allowed(current, pending[-1]['report'], initial,
                    np.array([], bool), np.array([], bool), np.array([], bool), reg, p['fit'])
                assert allowed and reason == row['reason']
                current = pending[-1]['report']
                accepted += len(pending)
                accepted_blocks += 1
            else:
                rejected[row['reason']] += 1
            pending = []
        elif row['kind'] == 'fit_finished':
            assert row['trials'] == len(trial_rows)
            if pending:
                assert pending[-1]['report']['pass']
                assert not any(t['report']['pass'] for t in pending[:-1])
                accepted += len(pending)
                current = pending[-1]['report']
                pending = []
            assert row['accepted_updates'] == accepted
        else:
            raise AssertionError('unexpected optimizer journal kind')
    assert not pending and log[-1]['kind'] == 'fit_finished'
    declared = by_gate[rid]
    assert (declared['trials'], declared['updates']) == (len(trial_rows), accepted)
    recalculated = A.train_gate(model, old, new, reg, p['fit'], scale)
    assert recalculated == current == {k: v for k, v in declared.items() if k not in ('run_id', 'trials', 'updates')}
    if not accepted:
        assert R.digests(model) == R.digests(source)
    batch = A.selected_rows(old + new, p['fit'])
    vc, vh = A.B.S.predict(model, batch[0])
    truth_sign = np.maximum(batch[3], batch[2] - batch[1] / scale) <= 0
    predicted_sign = np.maximum(vh, vc - batch[1] / scale) <= 0
    for origin, records, raw in [('old', old, raw_old[rid]), ('new', new, raw_new[rid])]:
        for (mid, k), t in records:
            ep = raw[mid, k]
            truth_check(ep, t, specs[mid], inputs)
            vc0, vh0 = A.B.S.predict(model, t.observations[:1])
            w = float(max(vh0[0], vc0[0] - t.budget_J[0] / scale))
            train_cases.append({'run_id': rid, 'map_id': mid, 'k': k, 'origin': origin,
                'actual_feasible': t.feasible, 'predicted_feasible': w <= 0,
                'true_W': float(t.path_residual[0]), 'predicted_W': w,
                'label': ep['label'], 'budget_J': ep['z0_J'], 'cost_model_J': ep['cost_model_J']})
    audited.append({**declared, 'initial_gate': initial, 'accepted_blocks': accepted_blocks,
        'rejected_block_reasons': dict(rejected),
        'selected_false_feasible': int(np.count_nonzero(predicted_sign & ~truth_sign)),
        'selected_false_infeasible': int(np.count_nonzero(~predicted_sign & truth_sign))})
    trials_total += len(trial_rows)
    updates_total += accepted
    accepted_blocks_total += accepted_blocks
assert len(train_cases) == 972
assert trials_total == manifest['optimizer_trials'] <= p['resource_limit']['optimizer_trials_total_max']
assert gates['pass'] == all(g['pass'] for g in gates['candidates'])
report = {'task': 'NAV-065', 'reviewer': 'main agent, legacy; no test subagents', 'independent': False,
    'timestamp': datetime.now().astimezone().isoformat(), 'host': socket.getfqdn(),
    'tmux': os.environ['TMUX'], 'python': sys.executable, 'registration_pin': pin,
    'execution_integrity_pass': True, 'scientific_pass': False, 'run_status': 'HELD_TRAIN_GATE',
    'basic_tests': 63, 'saved_train_episodes': 972, 'saved_train_steps': 372471,
    'new_train_episodes': 0, 'new_development_episodes': 0,
    'optimizer_trials': trials_total, 'accepted_updates': updates_total, 'accepted_blocks': accepted_blocks_total,
    'actor_unchanged_candidates': 6, 'train_pass_candidates': sum(g['pass'] for g in gates['candidates']),
    'initial_train_false_feasible': sum(g['initial_gate']['additional_false_feasible'] for g in audited),
    'train_false_feasible': sum(not r['actual_feasible'] and r['predicted_feasible'] for r in train_cases),
    'development_scoring_rounds': 0, 'original_gate_verdict': 'FAIL', 'train_candidates': audited,
    'limitations': ['Intermediate tensor/RNG rollback is verified with basic controlled tests; no second optimizer replay or new sampling was run.',
        'Repeated development is diagnostic, not a new independent final evaluation.'],
    'evidence': {'launch': str(LAUNCH), 'output': str(OUT),
        'output_hashes': {f.name: R.file_sha(f) for f in OUT.iterdir() if f.is_file()}}}
if not gates['pass']:
    held = read(OUT / 'HELD_TRAIN_GATE.json')
    assert held == {'development_scoring_rounds': 0, 'optimizer_trials': trials_total}
    for name in ('development_scores.jsonl', 'gate_G4_alignment_v2.json', 'deployment_candidates_v2.json'):
        assert not (OUT / name).exists()
else:
    dev = rows(OUT / 'development_scores.jsonl')
    assert len(dev) == len({(r['run_id'], r['map_id'], r['k']) for r in dev}) == 180
    score, g4 = read(OUT / 'deployment_critic_validation_v2.json'), read(OUT / 'gate_G4_alignment_v2.json')
    for summary in score['candidates']:
        assert R.summarize([r for r in dev if r['run_id'] == summary['run_id']], {m['map_id'] for m in inputs.maps}, reg) == {k: v for k, v in summary.items() if k != 'run_id'}
    report.update(development_scoring_rounds=1, development_episodes=180,
        development_pass_candidates=sum(s['pass'] for s in score['candidates']),
        development_false_feasible=sum(not r['actual_feasible'] and r['predicted_feasible'] for r in dev),
        scientific_pass=g4['verdict'] == 'PASS', original_gate_verdict=g4['verdict'],
        run_status='PASS' if g4['verdict'] == 'PASS' else 'FAIL_DEVELOPMENT')
    assert (OUT / 'deployment_candidates_v2.json').exists() == report['scientific_pass']
for path, payload in [(LAUNCH / 'review.json', report), (LAUNCH / 'train_cases.json', train_cases),
                      (NAV / 'reports/nav065_block_refit_v6_review.json', report)]:
    write(path, payload)
print(json.dumps({k: v for k, v in report.items() if k not in ('train_candidates', 'evidence')}, ensure_ascii=False, indent=2))
