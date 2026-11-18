"""NAV-065 v11: all-train head fitting with a source-relative ridge penalty.

Diagnosis never solves for candidate weights. A probe needs a new pinned
approval, fits two copied heads once, and reports the unchanged production gate.
No development scoring, actor/body training, new rollout or deployment.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np
import torch

import probe_nav065_head as H

L, R = H.L, H.R
PROPOSAL = R.NAV / 'config/nav065_ridge_probe_proposal_v11.json'
BOUND_INPUTS = tuple(dict.fromkeys((*H.BOUND_INPUTS,
    'config/nav065_head_probe_prereg_v10.json',
    'reports/nav065_head_probe_v10_review.json',
    'scripts/probe_nav065_ridge.py', 'tests/test_nav065_ridge_probe.py',
    'config/nav065_ridge_probe_proposal_v11.json')))
GROUPS = ('original_all_rows', 'additional_all_rows', 'all_starts', 'original_gate_boundary_rows')


def approved_registration(path):
    protocol = R.load(path)
    if (protocol.get('approved') is not True or protocol.get('status') != 'APPROVED'
            or not protocol.get('authorization_reference')):
        raise ValueError('new all-train head fitting requires explicit approval')
    pin = path.with_suffix('.json.sha256').read_text().strip()
    R.verify_hash(path, pin)
    omit = {'approved', 'status', 'authorization_reference', 'implementation_hashes'}
    if {k: v for k, v in protocol.items() if k not in omit} != {
            k: v for k, v in R.load(PROPOSAL).items() if k not in omit}:
        raise ValueError('registration differs from reviewed v11 proposal')
    if set(protocol.get('implementation_hashes', {})) != set(BOUND_INPUTS):
        raise ValueError('incomplete all-train implementation binding')
    for name, digest in protocol['implementation_hashes'].items():
        R.verify_hash(R.NAV / name, digest)
    return protocol, pin


def training_batch(old, new, settings):
    """All saved steps remain present; extra group appearances are intentional."""
    records = old + new
    if not old or not new or len({key for key, _ in records}) != len(records):
        raise ValueError('nonempty unique original/additional train records required')
    fields = ('observations', 'budget_J', 'cost_target', 'reach_target')
    groups = [tuple(np.concatenate([getattr(t, f) for _, t in rows]) for f in fields)
              for rows in (old, new)]
    groups.append(tuple(np.asarray([getattr(t, f)[0] for _, t in records]) for f in fields))
    groups.append(L.P.A.selected_rows(records, settings))
    weights, meta = [], []
    offset = 0
    for name, group in zip(GROUPS, groups):
        n = len(group[0])
        if not n or any(len(x) != n or not np.isfinite(x).all() for x in group):
            raise ValueError('invalid finite nonempty group ' + name)
        positive = group[3] > 0
        strata = [positive == flag for flag in (False, True) if np.any(positive == flag)]
        mass = np.zeros(n, dtype=np.float64)
        for mask in strata:
            mass[mask] = 1.0 / (len(GROUPS) * len(strata) * mask.sum())
        weights.append(mass)
        meta.append({'name': name, 'start': offset, 'stop': offset + n, 'rows': n,
                     'reach_positive_rows': int(positive.sum()), 'mass': float(mass.sum())})
        offset += n
    arrays = tuple(np.concatenate([g[i] for g in groups]) for i in range(4))
    return arrays, np.concatenate(weights), meta


def weighted_design(critic, observations, target, weights, settings):
    n = len(observations)
    y, p = np.asarray(target, np.float64), np.asarray(weights, np.float64)
    strength, target_floor, head_floor, block = (settings[k] for k in
        ('ridge_strength', 'target_rms_floor', 'head_norm_floor', 'feature_batch_rows'))
    if (n == 0 or y.shape != (n,) or p.shape != (n,)
            or not np.isfinite(y).all() or not np.isfinite(p).all()
            or np.any(p <= 0) or not np.isclose(p.sum(), 1., atol=1e-12, rtol=0.)
            or not all(np.isfinite(x) and x > 0 for x in (strength, target_floor, head_floor))
            or type(block) is not int or block <= 0
            or not np.isfinite(settings['rcond']) or not 0 < settings['rcond'] < 1):
        raise ValueError('invalid finite ridge data or settings')
    a = np.concatenate([H.features(critic, observations[i:i + block]) for i in range(0, n, block)])
    w = H.head_vector(critic)
    if not np.isfinite(w).all() or w.shape != (a.shape[1],):
        raise ValueError('invalid finite source head')
    rms = max(float(np.sqrt(np.dot(p, y * y))), target_floor)
    head_scale = max(float(np.linalg.norm(w)), head_floor)
    residual = y - a @ w
    source_objective = float(np.dot(p, residual * residual) / rms ** 2)
    ridge = strength / head_scale ** 2
    if not np.isfinite([rms, head_scale, source_objective, ridge]).all() or ridge <= 0:
        raise ValueError('nonfinite ridge scales')
    report = {'rows': n, 'columns': a.shape[1], 'target_rms': rms, 'head_scale': head_scale,
        'source_head_norm': float(np.linalg.norm(w)), 'ridge_coefficient': ridge,
        'source_objective': source_objective,
        'correction_norm_bound_for_objective_no_worse_than_source': float(np.sqrt(source_objective / ridge)),
        'candidate_weights_solved': False, 'candidate_parameters_assigned': 0}
    return a, y, p, w, report


def solve_head(critic, observations, target, weights, settings):
    """One full augmented solve, not normal equations or a parameter sweep."""
    a, y, p, w, report = weighted_design(critic, observations, target, weights, settings)
    multiplier = np.sqrt(p) / report['target_rms']
    augmented = np.vstack((a * multiplier[:, None], np.eye(len(w)) * np.sqrt(report['ridge_coefficient'])))
    rhs = np.r_[(y - a @ w) * multiplier, np.zeros(len(w))]
    delta, _, rank, singular = np.linalg.lstsq(augmented, rhs, rcond=settings['rcond'])
    fitted = (w + delta).astype(np.float32)
    actual_delta = fitted.astype(np.float64) - w
    objective = float(np.dot(p, (a @ fitted.astype(np.float64) - y) ** 2) / report['target_rms'] ** 2
                      + report['ridge_coefficient'] * np.dot(actual_delta, actual_delta))
    limit = report['source_objective'] + 1e-6 * max(1., report['source_objective'])
    if not np.isfinite(fitted).all() or not np.isfinite(objective) or objective > limit:
        raise ValueError('float32 head violates finite regularized objective guard')
    with torch.no_grad():
        critic.head.weight.copy_(torch.as_tensor(fitted[:-1], dtype=critic.head.weight.dtype).reshape_as(critic.head.weight))
        critic.head.bias.copy_(torch.as_tensor(fitted[-1:], dtype=critic.head.bias.dtype))
    return {**report, 'head_solves': 1, 'rank': int(rank), 'singular_values': singular.tolist(),
        'candidate_weights_solved': True, 'candidate_parameters_assigned': len(w),
        'regularized_objective_after_float32_cast': objective,
        'actual_float32_correction_norm': float(np.linalg.norm(actual_delta)),
        'actual_float32_head_norm': float(np.linalg.norm(fitted.astype(np.float64))),
        'objective_guard_pass': True}


def group_metrics(model, batch, weights, groups, scale):
    vc, vh = L.P.A.B.S.predict(model, batch[0])
    result = []
    for group in groups:
        s = slice(group['start'], group['stop'])
        tc, th, z = batch[2][s], batch[3][s], batch[1][s] / scale
        actual, predicted = np.maximum(th, tc - z) <= 0, np.maximum(vh[s], vc[s] - z) <= 0
        result.append({**group, 'cost_mae': float(np.abs(vc[s] - tc).mean()),
            'reach_mae': float(np.abs(vh[s] - th).mean()),
            'cost_sign_errors': int(np.count_nonzero((vc[s] - z <= 0) != (tc - z <= 0))),
            'reach_sign_errors': int(np.count_nonzero((vh[s] <= 0) != (th <= 0))),
            'joint_sign_errors': int(np.count_nonzero(predicted != actual)),
            'false_feasible': int(np.count_nonzero(~actual & predicted)),
            'false_infeasible': int(np.count_nonzero(actual & ~predicted))})
    return result


def selected_source(prepared, protocol):
    rows = [r for r in prepared if r[0]['run_id'] == L.P.TARGET]
    if len(rows) != 1:
        raise ValueError('single source candidate required')
    c, model, old, new = rows[0]
    expected = protocol['expected_train_episodes']
    if (c['checkpoint_sha256'] != protocol['source_checkpoint_sha256']
            or len(old) != expected['original'] or len(new) != expected['additional']
            or sum(len(t.observations) for _, t in old + new) != expected['total_steps']):
        raise ValueError('source identity or complete train coverage differs')
    return c, model, old, new


def diagnose(reg, inputs, prepared):
    protocol = R.load(PROPOSAL)
    c, model, old, new = selected_source(prepared, protocol)
    before = {x['run_id']: R.digests(net) for x, net, _, _ in prepared}
    source, keys = L.diagnose(reg, inputs, prepared)
    batch, weights, groups = training_batch(old, new, R.load(L.P.PROPOSAL)['fit'])
    scale = inputs.value['quantities']['Z_REF_J']['value']
    designs = {}
    for role, i in (('critic_cost', 2), ('critic_reach', 3)):
        designs[role] = weighted_design(getattr(model, role), batch[0], batch[i], weights, protocol['solve'])[-1]
    assert before == {x['run_id']: R.digests(net) for x, net, _, _ in prepared}
    return {'schema': 'camra-nav065-ridge-preflight-v11', 'host': socket.getfqdn(), 'tmux': os.environ.get('TMUX'),
        'candidate': c['run_id'], 'source_checkpoint_sha256': c['checkpoint_sha256'],
        'available_train_episodes': len(old) + len(new),
        'available_train_steps': sum(len(t.observations) for _, t in old + new),
        'weighted_group_rows': len(weights), 'weight_sum': float(weights.sum()),
        'raw_target_weight_sha256': [hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
                                     for x in (*batch, weights)],
        'groups_before': group_metrics(model, batch, weights, groups, scale), 'ridge_designs': designs,
        'local_keys': keys, 'source_full_train_gate': source['full_train_gate'],
        'source_local': source['local_source_evaluation'],
        'formal_head_solves': 0, 'formal_optimizer_steps': 0, 'new_rollouts': 0,
        'development_scoring_rounds': 0, 'source_models_unchanged': True,
        'scientific_pass': False, 'deployment_allowed': False,
        'limitations': ['A ridge displacement bound only controls the aggregate fitting objective, not each state or sign.',
            'No candidate weights are solved or tested by this diagnosis; original production train/G4 criteria remain unchanged.',
            'A failure of this fixed feature/regularization attempt cannot establish whole-task architecture impossibility.']}


def run(protocol, pin, reg, inputs, prepared):
    canonical = R.load(PROPOSAL)
    if (protocol['resource_limit'] != canonical['resource_limit'] or protocol['solve'] != canonical['solve']
            or protocol['fit_split'] != 'train' or protocol['candidate'] != L.P.TARGET
            or protocol['output_name'] != canonical['output_name']):
        raise ValueError('all-train scope, quota or fixed solve settings differ')
    out = R.WORK / protocol['output_name']
    if out.exists():
        raise ValueError('output exists; inspect, never overwrite or restart')
    c, source, old, new = selected_source(prepared, protocol)
    preflight = diagnose(reg, inputs, prepared)
    batch, weights, groups = training_batch(old, new, R.load(L.P.PROPOSAL)['fit'])
    scale = inputs.value['quantities']['Z_REF_J']['value']
    before = {x['run_id']: R.digests(net) for x, net, _, _ in prepared}
    model = copy.deepcopy(source)
    out.mkdir(exist_ok=False)
    with (out / 'preflight.json').open('x') as f:
        json.dump(preflight, f, ensure_ascii=False, indent=2, allow_nan=False)
    solves = {}
    try:
        for role, i in (('critic_cost', 2), ('critic_reach', 3)):
            with (out / 'attempts.jsonl').open('a') as f:
                R.write_record(f, {'kind': 'head_solve_attempt', 'role': role})
            solves[role] = solve_head(getattr(model, role), batch[0], batch[i], weights, protocol['solve'])
        allowed = {f'{role}.head.{name}' for role in R.SCORE_ROLES for name in ('weight', 'bias')}
        if any(not torch.equal(original, model.state_dict()[name]) for name, original in source.state_dict().items() if name not in allowed):
            raise ValueError('actor or critic body changed')
        full = L.P.A.train_gate(model, old, new, reg, R.load(L.P.PROPOSAL)['fit'], scale)
        local = L.evaluate_local(model, L.local_batch(old + new, [tuple(k) for k in preflight['local_keys']]), reg, scale)
        starts = L.P.S.start_snapshot(model, old, new, scale, reg['primary_k'])
        after_groups = group_metrics(model, batch, weights, groups, scale)
        checkpoint = out / 'diagnostic-ridge.pt'
        torch.save({'model': model.state_dict(), 'diagnostic_only': True, 'pin': pin}, checkpoint)
        if R.digests(R.load_network(inputs, c, checkpoint)) != R.digests(model):
            raise ValueError('ridge checkpoint CPU reload differs')
        if before != {x['run_id']: R.digests(net) for x, net, _, _ in prepared}:
            raise ValueError('original model changed')
        result = {'pin': pin, 'head_solves': solves, 'local': local, 'starts_after': starts,
            'full_train_before': preflight['source_full_train_gate'], 'full_train_after': full,
            'groups_before': preflight['groups_before'], 'groups_after': after_groups,
            'source_models_unchanged': True, 'actor_and_bodies_unchanged': True,
            'checkpoint_sha256': R.file_sha(checkpoint), 'optimizer_steps': 0, 'new_rollouts': 0,
            'development_scoring_rounds': 0, 'scientific_nav065_pass': False,
            'deployment_allowed': False, 'automatic_expansion_allowed': False}
        with (out / 'RIDGE_FINISHED.json').open('x') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    except Exception as exc:
        torch.save({'model': model.state_dict(), 'diagnostic_only': True, 'pin': pin}, out / 'failed-ridge.pt')
        with (out / 'STOPPED.json').open('x') as f:
            json.dump({'error_type': type(exc).__name__, 'error': str(exc), 'automatic_restart_allowed': False}, f)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--diagnose', type=Path)
    modes.add_argument('--registration', type=Path)
    args = parser.parse_args()
    if socket.gethostname().split('.')[0] != 'tiger11' or not os.environ.get('TMUX'):
        parser.error('requires tiger11 and tmux')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.registration:
        protocol, pin = approved_registration(args.registration)
        run(protocol, pin, *L.sources()[1:])
        return
    if args.diagnose.exists() or args.diagnose.resolve().parent != (R.NAV / 'reports').resolve():
        parser.error('diagnosis requires an exclusive new navigation report')
    # Validate the completed v10 evidence while retaining its unfitted v8 source.
    H.approved_registration(R.NAV / 'config/nav065_head_probe_prereg_v10.json')
    review = R.load(R.NAV / 'reports/nav065_head_probe_v10_review.json')
    for name, digest in review['evidence']['output_hashes'].items():
        R.verify_hash(Path(review['evidence']['output']) / name, digest)
    report = diagnose(*L.sources()[1:])
    with args.diagnose.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: report[k] for k in ('available_train_episodes', 'available_train_steps', 'weighted_group_rows', 'formal_head_solves', 'scientific_pass')}))


if __name__ == '__main__':
    main()
