"""NAV-065 v8: train-only diagnosis and a separately approved single-actor pilot.

The pilot preserves the original targets, network, actor and final train gate.
No development evaluator is called, even if the pilot passes. Diagnosis never
steps an optimizer; proposed optimizer updates require a new pinned approval.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import socket

import numpy as np
import torch

import refit_nav065_staged as S

A, V, R = S.A, S.V, S.R
PROPOSAL = R.NAV / 'config/nav065_pilot_refit_proposal_v8.json'
TARGET = 'nav064_terminal_tail_v1_full_s0'
BOUND_INPUTS = tuple(dict.fromkeys((*S.BOUND_INPUTS,
    'config/nav065_staged_refit_prereg_v7.json',
    'reports/nav065_staged_refit_v7_review.json',
    'scripts/refit_nav065_pilot.py', 'tests/test_nav065_pilot_refit.py',
    'config/nav065_pilot_refit_proposal_v8.json')))


def approved_registration(path):
    p = R.load(path)
    if p.get('approved') is not True or p.get('status') != 'APPROVED' or not p.get('authorization_reference'):
        raise ValueError('new pilot process and optimizer quota require explicit approval')
    pin = path.with_suffix('.json.sha256').read_text().strip()
    R.verify_hash(path, pin)
    omit = {'approved', 'status', 'authorization_reference', 'implementation_hashes'}
    if {k: v for k, v in p.items() if k not in omit} != {
            k: v for k, v in R.load(PROPOSAL).items() if k not in omit}:
        raise ValueError('registration differs from reviewed v8 proposal')
    if set(p.get('implementation_hashes', {})) != set(BOUND_INPUTS):
        raise ValueError('incomplete implementation binding')
    for name, digest in p['implementation_hashes'].items():
        R.verify_hash(R.NAV / name, digest)
    if p['resource_limit']['update_candidates'] != [TARGET]:
        raise ValueError('single-candidate scope differs')
    return p, pin


def sources():
    upstream, reg, inputs, previous = S.sources()
    p = R.load(PROPOSAL)
    _, pin7 = S.approved_registration(R.NAV / 'config/nav065_staged_refit_prereg_v7.json')
    review = R.load(R.NAV / 'reports/nav065_staged_refit_v7_review.json')
    root = R.WORK / p['source_run']
    for name, digest in review['evidence']['output_hashes'].items():
        R.verify_hash(root / name, digest)
    R.verify_hash(root / 'fitted_candidates.json', p['source_manifest_sha256'])
    manifest = R.load(root / 'fitted_candidates.json')
    if manifest['refit_pin'] != pin7:
        raise ValueError('v7 source pin differs')
    by_id = {c['run_id']: (c, old, new) for c, _, old, new in previous}
    if len(manifest['candidates']) != 6 or {c['run_id'] for c in manifest['candidates']} != set(by_id):
        raise ValueError('v7 candidate set differs')
    prepared = []
    for c in manifest['candidates']:
        R.verify_hash(Path(c['checkpoint']), c['checkpoint_sha256'])
        model = R.load_network(inputs, c, c['checkpoint'])
        parent, old, new = by_id[c['run_id']]
        if R.digests(model) != c['state_dict_sha256'] or R.digests(model)['policy'] != parent['state_dict_sha256']['policy']:
            raise ValueError('source tensors or actor differ')
        prepared.append((c, model, old, new))
    return upstream, reg, inputs, prepared


def compose_direction(regression_gradient, sign_gradient, max_fraction):
    """Keep regression; remove opposing sign component, then cap without amplification.

    For a plain SGD step, dot(r, direction) >= ||r||^2 in exact arithmetic.
    This is a first-order property of the sampled MSE, not a guarantee about
    finite-step MAE, Pearson, classification, held-out data or convergence.
    """
    r, s = regression_gradient, sign_gradient
    if not 0 <= max_fraction <= 1 or not torch.isfinite(r).all() or not torch.isfinite(s).all():
        raise ValueError('invalid finite gradients or correction fraction')
    rr, ss, dot = torch.dot(r, r), torch.dot(s, s), torch.dot(r, s)
    if rr.item() == 0 or ss.item() == 0 or max_fraction == 0:
        return r.clone(), {'sign_multiplier': 0., 'projected_sign_norm': 0., 'correction_norm': 0.,
            'regression_dot_direction': float(rr)}
    projected = s - torch.minimum(dot, dot.new_zeros(())) / rr * r
    norm = projected.norm()
    multiplier = min(1., max_fraction * float(rr.sqrt()) / float(norm)) if norm.item() else 0.
    correction = multiplier * projected
    direction = r + correction
    return direction, {'sign_multiplier': multiplier, 'projected_sign_norm': float(norm),
        'correction_norm': float(correction.norm()), 'regression_dot_direction': float(torch.dot(r, direction))}


def regression_first_gradients(regression, signs, groups, max_fraction):
    stats = {}
    for role, sign in zip(R.SCORE_ROLES, signs):
        params = groups[role]
        r, s = [torch.cat([g.reshape(-1) for g in torch.autograd.grad(
            term, params, retain_graph=True)]) for term in (regression, sign)]
        rn, sn = float(r.norm()), float(s.norm())
        direction, detail = compose_direction(r, s, max_fraction)
        offset = 0
        for param in params:
            param.grad = direction[offset:offset + param.numel()].reshape_as(param).clone()
            offset += param.numel()
        stats[role] = {**detail, 'regression_gradient_norm': rn, 'sign_gradient_norm': sn,
            'gradient_cosine': float(torch.dot(r, s)) / (rn * sn) if rn and sn else None,
            'direction_norm': float(direction.norm())}
    return stats


def observation_aliases(records):
    """Exact float32 input collisions within one fixed actor's full train set."""
    obs = np.concatenate([t.observations for _, t in records]).astype(np.float32)
    _, inverse, counts = np.unique(obs, axis=0, return_inverse=True, return_counts=True)
    result = {'rows': len(obs), 'unique_inputs': len(counts), 'duplicate_input_groups': int((counts > 1).sum())}
    for name in ('cost_target', 'reach_target', 'path_residual'):
        values = np.concatenate([getattr(t, name) for _, t in records])
        lo, hi = np.full(len(counts), np.inf), np.full(len(counts), -np.inf)
        np.minimum.at(lo, inverse, values)
        np.maximum.at(hi, inverse, values)
        result[name] = {'different_target_groups_above_1e_5': int(((hi - lo) > 1e-5).sum()),
            'opposite_sign_groups': int(((lo <= 0) & (hi > 0)).sum()),
            'max_target_range': float((hi - lo).max())}
    return result


def adjacent_target_pairs(records):
    """Largest reach-target jump for each map, between adjacent sampled budgets."""
    maps = defaultdict(list)
    for (mid, k), t in records:
        maps[mid].append((k, t))
    examples = []
    for mid, items in sorted(maps.items()):
        items.sort(key=lambda x: x[0])
        pairs = []
        for (ka, a), (kb, b) in zip(items, items[1:]):
            pairs.append({'map_id': mid, 'k': [ka, kb],
                'budget_J': [float(a.budget_J[0]), float(b.budget_J[0])],
                'reach_target': [float(a.reach_target[0]), float(b.reach_target[0])],
                'true_W': [float(a.path_residual[0]), float(b.path_residual[0])],
                'actual_feasible': [bool(a.feasible), bool(b.feasible)],
                'input_linf': float(np.abs(a.observations[0].astype(np.float32) - b.observations[0].astype(np.float32)).max())})
        if pairs:
            examples.append(max(pairs, key=lambda p: abs(p['reach_target'][1] - p['reach_target'][0])))
    return examples


def journal_diagnosis(path):
    stages, gradients = {}, defaultdict(list)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row['kind'] == 'train_gate' and row['trial'] == 0:
            stages['initial'] = row
        elif row['kind'] == 'trial':
            stages[row['stage']] = row
            for role, detail in (row['gradient_diagnostics'] or {}).items():
                cosine = detail['gradient_cosine']
                if cosine is not None:
                    gradients[role].append(cosine)
    result = {}
    for stage, row in stages.items():
        result[stage] = {'starts': dict(Counter(s['classification'] for s in row['starts'])),
            'additional_metrics': row['report']['additional_metrics']}
    result['boundary_gradients'] = {role: {'steps': len(values),
        'opposing_steps': sum(v < 0 for v in values),
        'old_direction_can_oppose_regression_steps': sum(v < (1 - math.sqrt(5)) / 2 for v in values),
        'mean_cosine': sum(values) / len(values)} for role, values in gradients.items()}
    return result


def diagnose(reg, inputs, prepared):
    p, rows = R.load(PROPOSAL), []
    scale = inputs.value['quantities']['Z_REF_J']['value']
    saved = {c['run_id']: c for c in R.load(R.WORK / p['source_run'] / 'train_gate.json')['candidates']}
    for c, model, old, new in prepared:
        before = R.digests(model)
        gate = A.train_gate(model, old, new, reg, p['fit'], scale)
        assert gate == {k: v for k, v in saved[c['run_id']].items() if k not in ('run_id', 'trials')}
        rows.append({'run_id': c['run_id'], 'gate': gate,
            'starts': S.start_snapshot(model, old, new, scale, reg['primary_k']),
            'input_aliases': observation_aliases(old + new),
            'adjacent_budget_examples': adjacent_target_pairs(old + new),
            'v7_journal': journal_diagnosis(R.WORK / p['source_run'] / (c['run_id'] + '.jsonl'))})
        assert R.digests(model) == before
        print('diagnosed', c['run_id'], flush=True)
    return {'schema': 'camra-nav065-pilot-preflight-v8', 'host': socket.getfqdn(),
        'tmux': os.environ['TMUX'], 'train_episodes': sum(len(o) + len(n) for _, _, o, n in prepared),
        'train_steps': sum(len(t.observations) for _, _, o, n in prepared for _, t in o + n),
        'optimizer_steps': 0, 'new_rollouts': 0, 'development_scoring_rounds': 0,
        'source_gate_exact_match': True, 'candidates': rows,
        'limitations': ['No exact collision is not a proof of sufficient representation or generalization.',
            'Adjacent-budget examples can follow different budget-conditioned actions; they do not prove an optimal value discontinuity.',
            'Gradient geometry concerns sampled regression only; finite-step train and all original scientific gates still decide acceptance.']}


def fit(model, old, new, protocol, reg, scale, seed, journal, stage_checkpoint=None):
    model.policy.requires_grad_(False)
    actor, settings, trials = R.digests(model)['policy'], protocol['fit'], 0
    groups = {role: model.parameters_of(role) for role in R.SCORE_ROLES}
    params = [p for ps in groups.values() for p in ps]
    rng = np.random.default_rng(settings['sampling_seed'] + seed)
    batch = A.selected_rows(old + new, settings)
    gate = A.train_gate(model, old, new, reg, settings, scale)
    starts = S.start_snapshot(model, old, new, scale, reg['primary_k'])
    R.write_record(journal, {'kind': 'train_gate', 'trial': 0, 'report': gate, 'starts': starts})
    if gate['pass']:
        return 0, gate, 'train_pass'
    for stage in ('numeric', 'boundary'):
        if stage == 'numeric' and S.numeric_pass(gate, reg):
            continue
        if stage == 'boundary' and not S.numeric_pass(gate, reg):
            R.write_record(journal, {'kind': 'fit_finished', 'trials': trials,
                'reason': 'numeric_gate_not_met', 'pass': False})
            return trials, gate, 'numeric_gate_not_met'
        optimizer = (torch.optim.Adam(params, lr=settings['numeric_learning_rate']) if stage == 'numeric'
            else torch.optim.SGD(params, lr=settings['boundary_learning_rate'], momentum=0.))
        R.write_record(journal, {'kind': 'stage_started', 'stage': stage, 'trial': trials,
            'optimizer': 'Adam' if stage == 'numeric' else 'SGD_no_momentum'})
        for step in range(1, settings[stage + '_steps_max'] + 1):
            regression, strata = V.regression_loss(model, old, new, settings, rng)
            if not torch.isfinite(regression):
                raise ValueError('nonfinite pilot regression loss')
            optimizer.zero_grad()
            stats = None
            if stage == 'numeric':
                regression.backward()
            else:
                signs = A.sign_losses(model, batch, settings, scale)
                if any(not torch.isfinite(s) for s in signs):
                    raise ValueError('nonfinite pilot sign loss')
                stats = regression_first_gradients(regression, signs, groups, settings['correction_fraction_max'])
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
                raise ValueError('nonfinite pilot gradient')
            trials += 1
            if trials > protocol['resource_limit']['optimizer_trials_total_max']:
                raise ValueError('pilot quota exceeded')
            R.write_record(journal, {'kind': 'optimizer_attempt', 'trial': trials, 'stage': stage, 'stage_step': step})
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in params):
                raise ValueError('nonfinite pilot parameter')
            gate = A.train_gate(model, old, new, reg, settings, scale)
            current = S.start_snapshot(model, old, new, scale, reg['primary_k'])
            R.write_record(journal, {'kind': 'trial', 'trial': trials, 'stage': stage, 'stage_step': step,
                'regression_loss_before': float(regression.detach()), 'strata': strata,
                'gradient_diagnostics': stats, 'report': gate, 'starts': current,
                'start_changes': S.changes(starts, current)})
            starts = current
            if R.digests(model)['policy'] != actor:
                raise ValueError('actor changed')
            if gate['pass']:
                return trials, gate, 'train_pass'
            if stage == 'numeric' and S.numeric_pass(gate, reg):
                break
            if stage == 'boundary' and step % settings['numeric_review_interval'] == 0 and not S.numeric_pass(gate, reg):
                return trials, gate, 'boundary_numeric_regression'
        if stage_checkpoint:
            stage_checkpoint(stage, trials, model)
    return trials, gate, 'trial_cap_reached'


def run(protocol, pin, upstream, reg, inputs, prepared):
    if protocol['resource_limit']['update_candidates'] != [TARGET] or sum(c['run_id'] == TARGET for c, _, _, _ in prepared) != 1:
        raise ValueError('pilot requires exactly the registered candidate')
    out = R.WORK / protocol['output_name']
    out.mkdir(exist_ok=False)
    torch.use_deterministic_algorithms(True)
    c, model, old, new = next(row for row in prepared if row[0]['run_id'] == TARGET)
    actor = R.digests(model)['policy']
    def save_stage(stage, trials, net):
        with (out / f'{TARGET}.{stage}-{trials}.pt').open('xb') as stream:
            torch.save({'model': net.state_dict(), 'pilot_pin': pin, 'stage': stage, 'trial': trials}, stream)
    with (out / (TARGET + '.jsonl')).open('x') as journal:
        try:
            trials, gate, reason = fit(model, old, new, protocol, reg,
                inputs.value['quantities']['Z_REF_J']['value'], c['seed'], journal, save_stage)
        except Exception as exc:
            R.write_record(journal, {'kind': 'run_stopped', 'error_type': type(exc).__name__, 'error': str(exc)})
            with (out / 'STOPPED.json').open('x') as stream:
                R.write_record(stream, {'reason': str(exc), 'automatic_restart_allowed': False,
                    'development_scoring_rounds': 0})
            raise
        R.write_record(journal, {'kind': 'pilot_finished', 'trials': trials, 'reason': reason, 'pass': gate['pass']})
    path = out / (TARGET + '.pt')
    with path.open('xb') as stream:
        torch.save({'model': model.state_dict(), 'pilot_pin': pin}, stream)
    digest = R.digests(model)
    if digest['policy'] != actor or R.digests(R.load_network(inputs, c, path)) != digest:
        raise ValueError('actor or CPU reload differs')
    untouched = []
    for other, net, _, _ in prepared:
        if other['run_id'] != TARGET:
            if R.digests(net) != other['state_dict_sha256']:
                raise ValueError('nonpilot model changed')
            untouched.append(other['run_id'])
    result = {'pilot_pin': pin, 'candidate': {**c, 'parent_pilot_checkpoint': c['checkpoint'],
        'checkpoint': str(path), 'checkpoint_sha256': R.file_sha(path), 'state_dict_sha256': digest},
        'optimizer_trials': trials, 'train_gate': gate, 'stop_reason': reason,
        'untouched_candidates': untouched, 'development_scoring_rounds': 0,
        'scientific_nav065_pass': False, 'automatic_expansion_allowed': False}
    with (out / 'PILOT_FINISHED.json').open('x') as stream:
        R.write_record(stream, result)
    # A one-candidate pilot never authorizes development, a six-model bundle or NAV-066.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--diagnose', type=Path)
    modes.add_argument('--registration', type=Path)
    args = parser.parse_args()
    if socket.gethostname().split('.')[0] != 'tiger11' or not os.environ.get('TMUX'):
        parser.error('requires tiger11 and tmux')
    torch.set_num_threads(1)
    if args.registration:
        protocol, pin = approved_registration(args.registration)
        run(protocol, pin, *sources())
        return
    if args.diagnose.exists() or args.diagnose.resolve().parent != (R.NAV / 'reports').resolve():
        parser.error('diagnosis requires a new report in navigation/reports')
    _, reg, inputs, prepared = sources()
    with args.diagnose.open('x') as stream:
        json.dump(diagnose(reg, inputs, prepared), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
