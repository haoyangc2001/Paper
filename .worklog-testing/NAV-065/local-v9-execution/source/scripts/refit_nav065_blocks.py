"""Train-only phase-II critic regression with bounded blocks (proposed v6).

No new trajectories. Diagnosis never steps an optimizer or scores development.
Working block guards are distinct from the unchanged final v5/G4 criteria.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import socket

import numpy as np
import torch

import refit_nav065_adaptive as A

R = A.R
PROPOSAL = R.NAV / 'config/nav065_block_refit_proposal_v6.json'
BOUND_INPUTS = (*A.BOUND_INPUTS, 'config/nav065_adaptive_refit_prereg_v5.json',
                'reports/nav065_adaptive_refit_v5_review.json',
                'scripts/refit_nav065_blocks.py', 'tests/test_nav065_block_refit.py',
                'config/nav065_block_refit_proposal_v6.json')


def approved_registration(path):
    p = R.load(path)
    if p.get('approved') is not True or p.get('status') != 'APPROVED' or not p.get('authorization_reference'):
        raise ValueError('block guards and new optimizer quota require explicit approval')
    pin = path.with_suffix('.json.sha256').read_text().strip()
    R.verify_hash(path, pin)
    omit = {'approved', 'status', 'authorization_reference', 'implementation_hashes'}
    if {k: v for k, v in p.items() if k not in omit} != {
            k: v for k, v in R.load(PROPOSAL).items() if k not in omit}:
        raise ValueError('registration differs from reviewed v6 proposal')
    if set(p.get('implementation_hashes', {})) != set(BOUND_INPUTS):
        raise ValueError('incomplete implementation binding')
    for name, digest in p['implementation_hashes'].items():
        R.verify_hash(R.NAV / name, digest)
    return p, pin


def regression_loss(model, old, new, settings, rng):
    """Balance discrete stop branches without smoothing or changing any labels."""
    losses, counts = [], {}
    for origin, records in (('old', old), ('new', new)):
        if not records:
            continue
        for kind in ('start', 'interior', 'boundary'):
            if kind == 'boundary':
                obs, _, tc, th = A.selected_rows(records, settings)
            else:
                samples = []
                for _, t in records:
                    n = len(t.observations)
                    idx = np.array([0]) if kind == 'start' else rng.choice(
                        np.arange(1, n) if n > 1 else [0],
                        size=settings['interior_samples_per_episode'],
                        replace=n - 1 < settings['interior_samples_per_episode'])
                    samples.append((t, idx))
                obs, tc, th = (np.concatenate([getattr(t, field)[idx] for t, idx in samples])
                               for field in ('observations', 'cost_target', 'reach_target'))
            for positive in (False, True):
                mask = (th > 0) == positive
                counts[f'{origin}/{kind}/reach_positive={positive}'] = int(mask.sum())
                if mask.any():
                    vc, vh = model.values(torch.as_tensor(obs[mask], dtype=torch.float32))
                    targets = [torch.as_tensor(x[mask], dtype=torch.float32) for x in (tc, th)]
                    losses.append(((vc - targets[0]).square().mean() +
                                   (vh - targets[1]).square().mean()) / 2)
    if not losses:
        raise ValueError('no train regression rows')
    return torch.stack(losses).mean(), counts


def block_allowed(previous, trial, source, prediction, actual, locked, reg, settings):
    """Commit guards; temporary in-block values cannot become final criteria."""
    if not trial['original_grid']['pass']:
        return False, 'original_grid'
    if np.any(prediction[locked] != actual[locked]):
        return False, 'previously_correct_start'
    for name in ('additional_false_feasible', 'selected_sign_mismatches'):
        if trial[name] > previous[name]:
            return False, name
    for name, metric in trial['additional_metrics'].items():
        anchor = source['additional_metrics'][name]
        if anchor['relative_mae'] is None or metric['relative_mae'] is None:
            return False, name + '_undefined_mae'
        limit = max(anchor['relative_mae'], reg['criteria']['rel_mae_max'])
        if not np.isfinite(metric['relative_mae']) or metric['relative_mae'] > limit * (1 + settings['working_relative_mae_slack']):
            return False, name + '_working_mae'
        if metric['pearson_required']:
            baseline = anchor['pearson'] if anchor['pearson'] is not None else -1.0
            floor = min(baseline, reg['criteria']['pearson_min']) - settings['working_pearson_slack']
            if metric['pearson'] is None or not np.isfinite(metric['pearson']) or metric['pearson'] < floor:
                return False, name + '_working_pearson'
    progress = any(trial[n] < previous[n] for n in ('additional_false_feasible', 'selected_sign_mismatches'))
    progress |= trial['additional_metrics']['vh']['relative_mae'] < previous['additional_metrics']['vh']['relative_mae'] - 1e-10
    return bool(progress), 'progress' if progress else 'no_progress'


def fit(model, old, new, protocol, reg, scale, seed, journal):
    model.policy.requires_grad_(False)
    actor = R.digests(model)['policy']
    settings, records = protocol['fit'], old + new
    gate = A.train_gate(model, old, new, reg, settings, scale)
    R.write_record(journal, {'kind': 'train_gate', 'trials': 0, **gate})
    if gate['pass']:
        return 0, 0, gate
    source = copy.deepcopy(gate)
    groups = {role: model.parameters_of(role) for role in R.SCORE_ROLES}
    optimizer = torch.optim.Adam([p for ps in groups.values() for p in ps], lr=settings['learning_rate'])
    rng = np.random.default_rng(settings['sampling_seed'] + seed)
    actual = np.array([t.feasible for _, t in records])
    locked = A.classify(model, records, scale) == actual
    batch = A.selected_rows(records, settings)
    trials = accepted = 0
    cap = protocol['resource_limit']['optimizer_trials_per_candidate_max']
    while trials < cap and not gate['pass']:
        tensors, optim, random_state = (copy.deepcopy(model.state_dict()),
            copy.deepcopy(optimizer.state_dict()), copy.deepcopy(rng.bit_generator.state))
        committed = False
        for backtrack in range(settings['block_backtracks_max']):
            model.load_state_dict(tensors)
            optimizer.load_state_dict(optim)
            rng.bit_generator.state = copy.deepcopy(random_state)
            for group in optimizer.param_groups:
                group['lr'] = settings['learning_rate'] * settings['backtrack_factor'] ** backtrack
            steps = min(settings['steps_per_block'], cap - trials)
            if not steps:
                break
            for _ in range(steps):
                regression, counts = regression_loss(model, old, new, settings, rng)
                signs = A.sign_losses(model, batch, settings, scale)
                optimizer.zero_grad()
                A.G.balanced_gradients(regression, signs, groups)
                if any(not torch.isfinite(p.grad).all() for ps in groups.values() for p in ps):
                    raise ValueError('nonfinite block gradient')
                trials += 1  # Every attempted step counts, including rejected blocks.
                optimizer.step()
                if any(not torch.isfinite(p).all() for ps in groups.values() for p in ps):
                    raise ValueError('nonfinite block parameter')
                trial = A.train_gate(model, old, new, reg, settings, scale)
                R.write_record(journal, {'kind': 'trial', 'trial': trials, 'backtrack': backtrack,
                    'learning_rate': optimizer.param_groups[0]['lr'], 'strata': counts,
                    'regression_loss_before': float(regression.detach()), 'report': trial})
                if trial['pass']:
                    accepted += _ + 1
                    if R.digests(model)['policy'] != actor:
                        raise ValueError('actor changed')
                    R.write_record(journal, {'kind': 'fit_finished', 'trials': trials,
                        'accepted_updates': accepted, 'pass': True})
                    return trials, accepted, trial
            prediction = A.classify(model, records, scale)
            allowed, reason = block_allowed(gate, trial, source, prediction, actual, locked, reg, settings)
            R.write_record(journal, {'kind': 'block_decision', 'trials': trials,
                'steps': steps, 'accepted': allowed, 'reason': reason, 'backtrack': backtrack})
            if allowed:
                accepted += steps
                gate, committed = trial, True
                locked |= prediction == actual
                break
        if not committed:
            model.load_state_dict(tensors)
            optimizer.load_state_dict(optim)
            rng.bit_generator.state = copy.deepcopy(random_state)
            break
    if R.digests(model)['policy'] != actor:
        raise ValueError('actor changed')
    R.write_record(journal, {'kind': 'fit_finished', 'trials': trials,
        'accepted_updates': accepted, **gate})
    return trials, accepted, gate


def sources():
    """Verify all old/new train records and load v5 final models, with no fitting."""
    proposal = R.load(PROPOSAL)
    _, pin5 = A.approved_registration(R.NAV / 'config/nav065_adaptive_refit_prereg_v5.json')
    upstream, reg, inputs, _, maps, refs, previous = A.sources()
    root = R.WORK / proposal['source_run']
    review = R.load(R.NAV / 'reports/nav065_adaptive_refit_v5_review.json')
    for name, digest in review['evidence']['output_hashes'].items():
        R.verify_hash(root / name, digest)
    R.verify_hash(root / 'fitted_candidates.json', proposal['source_manifest_sha256'])
    manifest = R.load(root / 'fitted_candidates.json')
    if manifest['refit_pin'] != pin5 or manifest['train_journal_sha256'] != R.file_sha(root / 'train_episodes.jsonl'):
        raise ValueError('v5 source pin or journal differs')
    by_id = {c['run_id']: (c, old) for c, _, old, _ in previous}
    current = {c['run_id']: c for c in manifest['candidates']}
    if len(manifest['candidates']) != 6 or set(current) != set(by_id):
        raise ValueError('v5 candidates differ')
    design = R.E.M.load_design(R.E.DESIGN)
    specs = {m['map_id']: R.E.M.build(m['level'], m['seed'], design) for m in maps}
    new = {rid: [] for rid in by_id}
    actions = {rid: [] for rid in by_id}
    points = {rid: {(mid, k): t.feasible for (mid, k), t in old} for rid, (_, old) in by_id.items()}
    pending = None
    with (root / 'train_episodes.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['kind'] not in ('rollout_start', 'rollout_result'):
                continue
            rid, mid, k = row['run_id'], row['map_id'], row['k']
            if rid not in by_id or mid not in specs or row['split'] != 'train':
                raise ValueError('non-train or unknown saved rollout')
            if row['kind'] == 'rollout_start':
                expected, interval = A.next_budget({v: f for (m, v), f in points[rid].items() if m == mid})
                if pending or k != expected or row['interval'] != interval:
                    raise ValueError('saved adaptive order differs')
                pending = (rid, mid, k)
            else:
                if pending != (rid, mid, k) or (mid, k) in points[rid]:
                    raise ValueError('duplicate or unmatched saved rollout')
                episode = row['episode']
                expected_budget = k * R.consistent_e_ref(refs[mid], current[rid]['variant'])
                if not np.isclose(episode['z0_J'], expected_budget, rtol=1e-12, atol=1e-9):
                    raise ValueError('saved train initial budget differs')
                t = R.reconstruct(episode, specs[mid], inputs)
                if t.budget_feature_clipped.any():
                    raise ValueError('clipped saved budgets')
                new[rid].append(((mid, k), t))
                actions[rid].append(np.asarray([s['action'] for s in episode['trace']]))
                points[rid][mid, k] = t.feasible
                pending = None
    if pending or any(len(new[rid]) != 72 or any(sum(m == mid for (m, _), _ in new[rid]) != 4 for mid in specs) for rid in by_id):
        raise ValueError('incomplete saved train coverage')
    prepared = []
    for rid, c in current.items():
        R.verify_hash(Path(c['checkpoint']), c['checkpoint_sha256'])
        model = R.load_network(inputs, c, c['checkpoint'])
        if R.digests(model) != c['state_dict_sha256'] or R.digests(model)['policy'] != by_id[rid][0]['state_dict_sha256']['policy']:
            raise ValueError('v5 tensors or actor differ')
        with torch.inference_mode():
            for (_, truth), recorded in zip(new[rid], actions[rid]):
                predicted = model.act(torch.as_tensor(truth.observations, dtype=torch.float32), deterministic=True)[0].numpy()
                if not np.allclose(predicted, recorded, rtol=0, atol=1e-6):
                    raise ValueError('saved new train actor actions differ')
        prepared.append((c, model, by_id[rid][1], new[rid]))
    limits = proposal['resource_limit']
    if (sum(len(old) + len(new) for _, _, old, new in prepared) != limits['existing_train_episodes'] or
            sum(len(t.observations) for _, _, old, new in prepared for _, t in old + new) != limits['existing_train_steps']):
        raise ValueError('saved train totals differ')
    return upstream, reg, inputs, prepared


def diagnose(reg, inputs, prepared):
    p, candidates = R.load(PROPOSAL), []
    scale = inputs.value['quantities']['Z_REF_J']['value']
    for c, model, old, new in prepared:
        before = R.digests(model)
        loss, counts = regression_loss(model, old, new, p['fit'], np.random.default_rng(p['fit']['sampling_seed'] + c['seed']))
        signs = A.sign_losses(model, A.selected_rows(old + new, p['fit']), p['fit'], scale)
        gradients = A.G.balanced_gradients(loss, signs, {role: model.parameters_of(role) for role in R.SCORE_ROLES})
        pairs = []
        for mid in sorted({m for (m, _), _ in old}):
            ordered = sorted((k, t) for (m, k), t in old + new if m == mid)
            crossing = [(a, b) for a, b in zip(ordered, ordered[1:]) if a[1].feasible != b[1].feasible]
            a, b = min(crossing, key=lambda ab: ab[1][0] - ab[0][0])
            pairs.append({'map_id': mid, 'k_lo': a[0], 'k_hi': b[0],
                'reach_lo': float(a[1].reach_target[0]), 'reach_hi': float(b[1].reach_target[0]),
                'budget_difference_J': float(b[1].budget_J[0] - a[1].budget_J[0])})
        if R.digests(model) != before:
            raise ValueError('diagnosis changed model tensors')
        candidates.append({'run_id': c['run_id'], 'strata': counts, 'gradient_diagnostics': gradients,
            'boundary_pairs': pairs, 'gate': A.train_gate(model, old, new, reg, p['fit'], scale)})
        model.zero_grad(set_to_none=True)
    return {'schema': 'camra-nav065-block-preflight-v6', 'host': socket.getfqdn(), 'tmux': os.environ['TMUX'],
        'train_episodes': 972, 'train_steps': 372471, 'new_rollouts': 0, 'optimizer_steps': 0,
        'development_scoring_rounds': 0, 'candidates': candidates}


def run(protocol, pin, upstream, reg, inputs, prepared):
    out = R.WORK / protocol['output_name']
    out.mkdir(exist_ok=False)
    fitted, gates, total = [], [], 0
    torch.use_deterministic_algorithms(True)
    for c, model, old, new in prepared:
        with (out / (c['run_id'] + '.jsonl')).open('x') as journal:
            trials, updates, gate = fit(model, old, new, protocol, reg,
                inputs.value['quantities']['Z_REF_J']['value'], c['seed'], journal)
        total += trials
        if total > protocol['resource_limit']['optimizer_trials_total_max']:
            raise ValueError('optimizer total exceeded')
        path = out / (c['run_id'] + '.pt')
        with path.open('xb') as stream:
            torch.save({'model': model.state_dict(), 'block_refit_pin': pin,
                'prereg_sha256': inputs.train_pin, 'implementation_versions': inputs.prereg['implementation_versions'],
                'variant': c['variant'], 'seed': c['seed']}, stream)
        if R.digests(R.load_network(inputs, c, path)) != R.digests(model):
            raise ValueError('saved CPU reload differs')
        fitted.append({**c, 'parent_block_checkpoint': c['checkpoint'], 'checkpoint': str(path),
            'checkpoint_sha256': R.file_sha(path), 'state_dict_sha256': R.digests(model)})
        gates.append({'run_id': c['run_id'], 'trials': trials, 'updates': updates, **gate})
    for name, payload in (('fitted_candidates.json', {'refit_pin': pin, 'candidates': fitted,
            'train_episodes': 972, 'train_steps': 372471, 'optimizer_trials': total}),
            ('train_gate.json', {'candidates': gates, 'pass': all(g['pass'] for g in gates)})):
        with (out / name).open('x') as stream:
            R.write_record(stream, payload)
    if not all(g['pass'] for g in gates):
        with (out / 'HELD_TRAIN_GATE.json').open('x') as stream:
            R.write_record(stream, {'development_scoring_rounds': 0, 'optimizer_trials': total})
        return
    R.evaluate(out, upstream, pin, reg, inputs, fitted, R.WORK / 'nav065_deployment_v1', 972, 372471)


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
    payload = diagnose(reg, inputs, prepared)
    with args.diagnose.open('x') as stream:
        R.write_record(stream, payload)
    print('972 saved train episodes verified; zero optimizer steps, new rollouts or development scoring', flush=True)


if __name__ == '__main__':
    main()
