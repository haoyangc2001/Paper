"""Train-only local fitting diagnosis, distinct from deployment acceptance.

--diagnose only measures current errors and feature separation. A probe fits a
deep copy on a fixed small start set; it needs a new approved registration and
never changes the original six models, labels, architecture or scientific gate.
"""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import socket

import numpy as np
import torch

import refit_nav065_pilot as P
from evaluate_nav065_deployment import error_metrics

R = P.R
PROPOSAL = R.NAV / 'config/nav065_local_probe_proposal_v9.json'
BOUND_INPUTS = tuple(dict.fromkeys((*P.BOUND_INPUTS,
    'config/nav065_pilot_refit_prereg_v8.json', 'reports/nav065_pilot_refit_v8_review.json',
    'scripts/probe_nav065_local.py', 'tests/test_nav065_local_probe.py',
    'config/nav065_local_probe_proposal_v9.json')))


def approved_registration(path):
    protocol = R.load(path)
    if (protocol.get('approved') is not True or protocol.get('status') != 'APPROVED'
            or not protocol.get('authorization_reference')):
        raise ValueError('new local optimizer quota requires explicit approval')
    pin = path.with_suffix('.json.sha256').read_text().strip()
    R.verify_hash(path, pin)
    omit = {'approved', 'status', 'authorization_reference', 'implementation_hashes'}
    if {k: v for k, v in protocol.items() if k not in omit} != {
            k: v for k, v in R.load(PROPOSAL).items() if k not in omit}:
        raise ValueError('registration differs from reviewed v9 proposal')
    if set(protocol.get('implementation_hashes', {})) != set(BOUND_INPUTS):
        raise ValueError('incomplete implementation binding')
    for name, digest in protocol['implementation_hashes'].items():
        R.verify_hash(R.NAV / name, digest)
    if protocol['resource_limit']['update_candidates'] != [P.TARGET]:
        raise ValueError('local probe is single-candidate only')
    return protocol, pin


def sources():
    upstream, reg, inputs, prepared = P.sources()
    proposal = R.load(PROPOSAL)
    root = R.WORK / proposal['source_run']
    review = R.load(R.NAV / 'reports/nav065_pilot_refit_v8_review.json')
    for name, digest in review['evidence']['output_hashes'].items():
        R.verify_hash(root / name, digest)
    R.verify_hash(root / 'PILOT_FINISHED.json', proposal['source_finished_sha256'])
    result = R.load(root / 'PILOT_FINISHED.json')
    _, pin = P.approved_registration(R.NAV / 'config/nav065_pilot_refit_prereg_v8.json')
    if result['pilot_pin'] != pin or result['candidate']['run_id'] != P.TARGET:
        raise ValueError('v8 source identity differs')
    candidate = result['candidate']
    if candidate['checkpoint_sha256'] != proposal['source_checkpoint_sha256']:
        raise ValueError('v8 source checkpoint differs')
    R.verify_hash(Path(candidate['checkpoint']), proposal['source_checkpoint_sha256'])
    source = R.load_network(inputs, candidate, candidate['checkpoint'])
    if R.digests(source) != candidate['state_dict_sha256']:
        raise ValueError('v8 source tensors differ')
    for i, (parent, model, old, new) in enumerate(prepared):
        if parent['run_id'] == P.TARGET:
            if R.digests(model)['policy'] != R.digests(source)['policy']:
                raise ValueError('source actor differs')
            prepared[i] = candidate, source, old, new
    return upstream, reg, inputs, prepared


def select_local(starts):
    """Ground-truth pairs; nearest feasible can be on either budget side."""
    by_key = {(r['map_id'], r['k']): r for r in starts}
    if len(by_key) != len(starts) or any(not np.isfinite(r['budget_J']) for r in starts):
        raise ValueError('duplicate or invalid start key')
    selected, pairs = set(), []
    for key, row in sorted(by_key.items()):
        if row['classification'] != 'false_feasible':
            continue
        feasible = [r for r in starts if r['map_id'] == row['map_id'] and r['actual_feasible']]
        if not feasible:
            raise ValueError('no actually feasible saved counterpart')
        other = min(feasible, key=lambda r: (abs(r['budget_J'] - row['budget_J']), r['k']))
        mate = other['map_id'], other['k']
        selected.update((key, mate))
        pairs.append({'infeasible': list(key), 'feasible': list(mate)})
    if not selected:
        raise ValueError('no false-feasible starts to diagnose')
    return sorted(selected), pairs


def local_batch(records, keys):
    table = {key: t for key, t in records}
    if len(table) != len(records) or len(set(keys)) != len(keys) or any(k not in table for k in keys):
        raise ValueError('nonunique or unknown local train key')
    return tuple(np.asarray([getattr(table[key], field)[0] for key in keys])
                 for field in ('observations', 'budget_J', 'cost_target', 'reach_target'))


def evaluate_local(model, batch, reg, scale):
    obs, budget, tc, th = batch
    vc, vh = P.A.B.S.predict(model, obs)
    actual = np.maximum(th, tc - budget / scale) <= 0
    predicted = np.maximum(vh, vc - budget / scale) <= 0
    metrics = {name: error_metrics(pred, target, reg['criteria']['variance_ratio_min'])
               for name, pred, target in (('vc_norm', vc, tc), ('vh', vh, th))}
    failures = P.A.numeric_failures(metrics, reg)
    cost_errors = int(np.count_nonzero((vc - budget / scale <= 0) != (tc - budget / scale <= 0)))
    reach_errors = int(np.count_nonzero((vh <= 0) != (th <= 0)))
    joint_errors = int(np.count_nonzero(actual != predicted))
    if cost_errors or reach_errors or joint_errors:
        failures.append('local component or joint signs')
    return {'local_pass': not failures, 'failures': failures, 'metrics': metrics,
        'cost_sign_errors': cost_errors, 'reach_sign_errors': reach_errors,
        'joint_sign_errors': joint_errors,
        'false_feasible': int(np.count_nonzero(~actual & predicted)),
        'false_infeasible': int(np.count_nonzero(actual & ~predicted)),
        'predictions': {'vc_norm': vc.tolist(), 'vh': vh.tolist(),
            'W': np.maximum(vh, vc - budget / scale).tolist()}}


def local_loss(model, batch):
    obs, _, tc, th = (torch.as_tensor(x, dtype=torch.float32) for x in batch)
    vc, vh = model.values(obs)
    terms = []
    for positive in (False, True):
        mask = (th > 0) == positive
        if mask.any():
            terms.append(((vc[mask] - tc[mask]).square().mean()
                          + (vh[mask] - th[mask]).square().mean()) / 2)
    return torch.stack(terms).mean()


def feature_separation(model, batch, keys, pairs):
    """Read-only geometry, not a fitted head or a capacity impossibility proof."""
    x = torch.as_tensor(batch[0], dtype=torch.float32)
    index = {tuple(key): i for i, key in enumerate(keys)}
    result = {}
    for role, target in (('critic_cost', batch[2]), ('critic_reach', batch[3])):
        critic = getattr(model, role)
        with torch.no_grad():
            features = critic.body(x).numpy().astype(np.float64)
        singular = np.linalg.svd(np.c_[features, np.ones(len(features))], compute_uv=False)
        tolerance = np.finfo(np.float32).eps * max(features.shape) * singular[0]
        rows = []
        for pair in pairs:
            a, b = [index[tuple(pair[k])] for k in ('infeasible', 'feasible')]
            delta = float(np.linalg.norm(features[b] - features[a]))
            jump = float(abs(target[b] - target[a]))
            rows.append({**pair, 'budget_gap_J': float(abs(batch[1][b] - batch[1][a])),
                'input_linf': float(np.abs(batch[0][b] - batch[0][a]).max()),
                'feature_l2': delta, 'target_jump': jump,
                'frozen_body_head_norm_lower_bound': jump / delta if delta else None})
        result[role] = {'pairs': rows, 'feature_rank_at_float32_tolerance': int((singular > tolerance).sum()),
            'singular_values': singular.tolist(), 'current_head_weight_norm': float(critic.head.weight.detach().norm())}
    return result


def diagnose(reg, inputs, prepared):
    candidate, model, old, new = next(r for r in prepared if r[0]['run_id'] == P.TARGET)
    before = {c['run_id']: R.digests(net) for c, net, _, _ in prepared}
    scale = inputs.value['quantities']['Z_REF_J']['value']
    starts = P.S.start_snapshot(model, old, new, scale, reg['primary_k'])
    gate = P.A.train_gate(model, old, new, reg, R.load(P.PROPOSAL)['fit'], scale)
    saved = R.load(R.WORK / R.load(PROPOSAL)['source_run'] / 'PILOT_FINISHED.json')
    if gate != saved['train_gate']:
        raise ValueError('source full train gate differs')
    keys, pairs = select_local(starts)
    batch = local_batch(old + new, keys)
    false = [r for r in starts if r['classification'] == 'false_feasible']
    report = {'schema': 'camra-nav065-local-preflight-v9', 'host': socket.getfqdn(), 'tmux': os.environ['TMUX'],
        'candidate': candidate['run_id'], 'source_checkpoint_sha256': candidate['checkpoint_sha256'],
        'source_train_gate_exact_match': True, 'full_train_gate': gate,
        'available_train_episodes': len(old) + len(new),
        'available_train_steps': sum(len(t.observations) for _, t in old + new),
        'starts': starts, 'start_confusion': dict(Counter(r['classification'] for r in starts)),
        'false_feasible_breakdown': {
            'total': len(false), 'reach_target_positive': sum(r['target_reach'] > 0 for r in false),
            'cost_only_target_block': sum(r['target_reach'] <= 0 for r in false)},
        'local_keys': keys, 'pairs': pairs, 'local_rows': len(keys),
        'local_targets': {'budget_J': batch[1].tolist(), 'cost_norm': batch[2].tolist(), 'reach': batch[3].tolist()},
        'local_source_evaluation': evaluate_local(model, batch, reg, scale),
        'feature_separation': feature_separation(model, batch, keys, pairs),
        'optimizer_steps': 0, 'new_rollouts': 0, 'development_scoring_rounds': 0,
        'scientific_pass': False, 'source_models_unchanged': True,
        'limitations': ['Pair selection uses already seen train mistakes; local success cannot establish generalization.',
            'Feature/head bounds hold with the body fixed; they do not prove the architecture cannot fit after updating it.',
            'No local optimizer or alternative architecture has been tested by this read-only report.']}
    assert before == {c['run_id']: R.digests(net) for c, net, _, _ in prepared}
    return report, keys


def fit_local(model, batch, protocol, reg, scale, journal):
    model.policy.requires_grad_(False)
    actor = R.digests(model)['policy']
    report = evaluate_local(model, batch, reg, scale)
    R.write_record(journal, {'kind': 'local_evaluation', 'trial': 0, **report})
    if report['local_pass']:
        return 0, report
    params = [p for role in R.SCORE_ROLES for p in model.parameters_of(role)]
    optimizer = torch.optim.Adam(params, lr=protocol['fit']['learning_rate'])
    for trial in range(1, protocol['resource_limit']['optimizer_trials_total_max'] + 1):
        optimizer.zero_grad()
        loss = local_loss(model, batch)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite local loss')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
            raise ValueError('nonfinite local gradient')
        R.write_record(journal, {'kind': 'optimizer_attempt', 'trial': trial})
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in params):
            raise ValueError('nonfinite local parameter')
        report = evaluate_local(model, batch, reg, scale)
        R.write_record(journal, {'kind': 'local_evaluation', 'trial': trial,
            'loss_before': float(loss.detach()), **report})
        if R.digests(model)['policy'] != actor:
            raise ValueError('actor changed')
        if report['local_pass']:
            return trial, report
    return trial, report


def run(protocol, pin, upstream, reg, inputs, prepared):
    if protocol['resource_limit']['update_candidates'] != [P.TARGET]:
        raise ValueError('local probe is single-candidate only')
    before = {c['run_id']: R.digests(net) for c, net, _, _ in prepared}
    diagnostic, keys = diagnose(reg, inputs, prepared)
    c, original, old, new = next(r for r in prepared if r[0]['run_id'] == P.TARGET)
    model = copy.deepcopy(original)
    batch = local_batch(old + new, keys)
    scale = inputs.value['quantities']['Z_REF_J']['value']
    out = R.WORK / protocol['output_name']
    out.mkdir(exist_ok=False)
    with (out / 'selection.json').open('x') as f:
        json.dump(diagnostic, f, ensure_ascii=False, indent=2, allow_nan=False)
    try:
        with (out / 'local.jsonl').open('x') as journal:
            trials, local = fit_local(model, batch, protocol, reg, scale, journal)
    except Exception as exc:
        torch.save({'model': model.state_dict(), 'diagnostic_only': True, 'pin': pin}, out / 'failed-probe.pt')
        with (out / 'STOPPED.json').open('x') as f:
            json.dump({'error_type': type(exc).__name__, 'error': str(exc), 'automatic_restart_allowed': False}, f)
        raise
    full = P.A.train_gate(model, old, new, reg, R.load(P.PROPOSAL)['fit'], scale)
    checkpoint = out / 'diagnostic-probe.pt'
    torch.save({'model': model.state_dict(), 'diagnostic_only': True, 'pin': pin}, checkpoint)
    if R.digests(R.load_network(inputs, c, checkpoint)) != R.digests(model):
        raise ValueError('diagnostic CPU reload differs')
    if R.digests(model)['policy'] != before[P.TARGET]['policy']:
        raise ValueError('diagnostic actor differs')
    if before != {c['run_id']: R.digests(net) for c, net, _, _ in prepared}:
        raise ValueError('original model changed')
    with (out / 'PROBE_FINISHED.json').open('x') as f:
        json.dump({'pin': pin, 'optimizer_trials': trials, 'local': local,
            'full_train_before': diagnostic['full_train_gate'], 'full_train_after': full,
            'diagnostic_checkpoint_sha256': R.file_sha(checkpoint), 'source_models_unchanged': True,
            'new_rollouts': 0, 'development_scoring_rounds': 0,
            'scientific_nav065_pass': False, 'deployment_allowed': False,
            'automatic_expansion_allowed': False}, f, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--diagnose', type=Path)
    mode.add_argument('--registration', type=Path)
    args = parser.parse_args()
    if socket.gethostname().split('.')[0] != 'tiger11' or not os.environ.get('TMUX'):
        parser.error('requires tiger11 and tmux')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.registration:
        protocol, pin = approved_registration(args.registration)
        run(protocol, pin, *sources())
        return
    if args.diagnose.exists() or args.diagnose.resolve().parent != (R.NAV / 'reports').resolve():
        parser.error('diagnosis needs an exclusive new navigation report')
    report, _ = diagnose(*sources()[1:])
    with args.diagnose.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: report[k] for k in ('local_rows', 'start_confusion', 'optimizer_steps', 'scientific_pass')}))


if __name__ == '__main__':
    main()
