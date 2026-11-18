"""NAV-065 v7: separate offline numerical fitting from boundary correction.

Diagnosis uses saved train only and never steps an optimizer. A new approved,
hash-pinned registration is mandatory for the staged run. Final train/G4 gates
are inherited unchanged; process freedom never authorizes deployment.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket

import numpy as np
import torch

import refit_nav065_blocks as V

A, R = V.A, V.R
PROPOSAL = R.NAV / 'config/nav065_staged_refit_proposal_v7.json'
BOUND_INPUTS = (*V.BOUND_INPUTS, 'config/nav065_block_refit_prereg_v6.json',
                'reports/nav065_block_refit_v6_review.json',
                'scripts/refit_nav065_staged.py', 'tests/test_nav065_staged_refit.py',
                'config/nav065_staged_refit_proposal_v7.json')


def approved_registration(path):
    p = R.load(path)
    if p.get('approved') is not True or p.get('status') != 'APPROVED' or not p.get('authorization_reference'):
        raise ValueError('staged process guards and new optimizer quota require explicit approval')
    pin = path.with_suffix('.json.sha256').read_text().strip()
    R.verify_hash(path, pin)
    omit = {'approved', 'status', 'authorization_reference', 'implementation_hashes'}
    if {k: v for k, v in p.items() if k not in omit} != {
            k: v for k, v in R.load(PROPOSAL).items() if k not in omit}:
        raise ValueError('registration differs from reviewed v7 proposal')
    if set(p.get('implementation_hashes', {})) != set(BOUND_INPUTS):
        raise ValueError('incomplete implementation binding')
    for name, digest in p['implementation_hashes'].items():
        R.verify_hash(R.NAV / name, digest)
    settings, limits = p['fit'], p['resource_limit']
    if settings['numeric_steps_max'] + settings['boundary_steps_max'] != limits['optimizer_trials_per_candidate_max']:
        raise ValueError('stage caps differ from approved candidate cap')
    return p, pin


def sources():
    upstream, reg, inputs, previous = V.sources()
    p = R.load(PROPOSAL)
    _, pin6 = V.approved_registration(R.NAV / 'config/nav065_block_refit_prereg_v6.json')
    root = R.WORK / p['source_run']
    review = R.load(R.NAV / 'reports/nav065_block_refit_v6_review.json')
    for name, digest in review['evidence']['output_hashes'].items():
        R.verify_hash(root / name, digest)
    R.verify_hash(root / 'fitted_candidates.json', p['source_manifest_sha256'])
    manifest = R.load(root / 'fitted_candidates.json')
    if manifest['refit_pin'] != pin6:
        raise ValueError('v6 source pin differs')
    by_id = {c['run_id']: (c, old, new) for c, _, old, new in previous}
    if len(manifest['candidates']) != 6 or {c['run_id'] for c in manifest['candidates']} != set(by_id):
        raise ValueError('v6 candidate set differs')
    prepared = []
    for c in manifest['candidates']:
        R.verify_hash(Path(c['checkpoint']), c['checkpoint_sha256'])
        model = R.load_network(inputs, c, c['checkpoint'])
        original, old, new = by_id[c['run_id']]
        if R.digests(model) != c['state_dict_sha256'] or R.digests(model)['policy'] != original['state_dict_sha256']['policy']:
            raise ValueError('v6 tensors or actor differ')
        prepared.append((c, model, old, new))
    return upstream, reg, inputs, prepared


def numeric_pass(gate, reg):
    """A stage transition only; all final sign/recognition checks still apply."""
    metrics = (gate['original_grid']['metrics'], gate['additional_metrics'])
    return all(not A.numeric_failures(m, reg) and all(
        v['relative_mae'] is not None and np.isfinite(v['relative_mae']) and
        (not v['pearson_required'] or (v['pearson'] is not None and np.isfinite(v['pearson'])))
        for v in m.values()) for m in metrics)


def start_snapshot(model, old, new, scale, primary_k):
    records = old + new
    obs, budgets, _, _ = A.G.start_tensors(records, scale)
    with torch.inference_mode():
        vc, vh = model.values(obs)
        w = torch.maximum(vh, vc - budgets)
    rows = []
    for i, ((mid, k), truth) in enumerate(records):
        predicted, actual = bool(w[i] <= 0), bool(truth.feasible)
        if not torch.isfinite(w[i]) or not torch.isfinite(vc[i]) or not torch.isfinite(vh[i]):
            raise ValueError('nonfinite start prediction')
        rows.append({'origin': 'old' if i < len(old) else 'new', 'map_id': mid, 'k': k,
            'primary': k == primary_k, 'actual_feasible': actual, 'predicted_feasible': predicted,
            'classification': ('true_feasible' if actual else 'false_feasible') if predicted else
                              ('false_infeasible' if actual else 'true_infeasible'),
            'vc_norm': float(vc[i]), 'vh': float(vh[i]), 'W': float(w[i]),
            'target_cost_norm': float(truth.cost_target[0]), 'target_reach': float(truth.reach_target[0]),
            'true_W': float(truth.path_residual[0]), 'budget_J': float(truth.budget_J[0])})
    return rows


def changes(before, after):
    """Identify corrections and regressions, distinguishing dangerous rejections."""
    result = []
    for a, b in zip(before, after, strict=True):
        if any(a[k] != b[k] for k in ('origin', 'map_id', 'k', 'actual_feasible')):
            raise ValueError('start identities differ')
        if a['classification'] != b['classification']:
            result.append({**b, 'previous_classification': a['classification'], 'previous_W': a['W']})
    return result


def fit(model, old, new, protocol, reg, scale, seed, journal, stage_checkpoint=None):
    model.policy.requires_grad_(False)
    actor = R.digests(model)['policy']
    settings, trials = protocol['fit'], 0
    gate = A.train_gate(model, old, new, reg, settings, scale)
    starts = start_snapshot(model, old, new, scale, reg['primary_k'])
    R.write_record(journal, {'kind': 'train_gate', 'trial': 0, 'report': gate, 'starts': starts})
    if gate['pass']:
        return 0, gate
    groups = {role: model.parameters_of(role) for role in R.SCORE_ROLES}
    params = [p for ps in groups.values() for p in ps]
    rng = np.random.default_rng(settings['sampling_seed'] + seed)
    batch = A.selected_rows(old + new, settings)
    for stage in ('numeric', 'boundary'):
        if stage == 'numeric' and numeric_pass(gate, reg):
            R.write_record(journal, {'kind': 'stage_skipped', 'stage': stage, 'trial': trials, 'reason': 'numeric_gate_pass'})
            continue
        optimizer = torch.optim.Adam(params, lr=settings[stage + '_learning_rate'])
        R.write_record(journal, {'kind': 'stage_started', 'stage': stage, 'trial': trials,
            'learning_rate': settings[stage + '_learning_rate'], 'fresh_adam': True})
        for step in range(1, settings[stage + '_steps_max'] + 1):
            regression, strata = V.regression_loss(model, old, new, settings, rng)
            if not torch.isfinite(regression):
                raise ValueError('nonfinite regression loss')
            optimizer.zero_grad()
            if stage == 'numeric':
                regression.backward()
                gradient_stats = None
            else:
                signs = A.sign_losses(model, batch, settings, scale)
                if any(not torch.isfinite(s) for s in signs):
                    raise ValueError('nonfinite sign loss')
                gradient_stats = A.G.balanced_gradients(regression, signs, groups)
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
                raise ValueError('nonfinite or missing staged gradient')
            trials += 1
            if trials > protocol['resource_limit']['optimizer_trials_per_candidate_max']:
                raise ValueError('candidate trial cap exceeded')
            R.write_record(journal, {'kind': 'optimizer_attempt', 'trial': trials, 'stage': stage, 'stage_step': step})
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in params):
                raise ValueError('nonfinite staged parameter')
            gate = A.train_gate(model, old, new, reg, settings, scale)
            current = start_snapshot(model, old, new, scale, reg['primary_k'])
            R.write_record(journal, {'kind': 'trial', 'trial': trials, 'stage': stage, 'stage_step': step,
                'regression_loss_before': float(regression.detach()), 'strata': strata,
                'gradient_diagnostics': gradient_stats, 'report': gate,
                'starts': current, 'start_changes': changes(starts, current)})
            starts = current
            if R.digests(model)['policy'] != actor:
                raise ValueError('actor changed')
            if gate['pass']:
                R.write_record(journal, {'kind': 'fit_finished', 'trials': trials, 'pass': True})
                return trials, gate
            if stage == 'numeric' and numeric_pass(gate, reg):
                break
        if stage_checkpoint:
            stage_checkpoint(stage, trials, model)
        R.write_record(journal, {'kind': 'stage_finished', 'stage': stage, 'trial': trials,
            'numeric_pass': numeric_pass(gate, reg), 'full_pass': gate['pass']})
    R.write_record(journal, {'kind': 'fit_finished', 'trials': trials, 'pass': gate['pass']})
    return trials, gate


def diagnose(reg, inputs, prepared):
    settings, candidates = R.load(PROPOSAL)['fit'], []
    scale = inputs.value['quantities']['Z_REF_J']['value']
    for c, model, old, new in prepared:
        before = R.digests(model)
        reasons, rejected = Counter(), []
        for line in (R.WORK / R.load(PROPOSAL)['source_run'] / (c['run_id'] + '.jsonl')).read_text().splitlines():
            row = json.loads(line)
            if row['kind'] == 'block_decision' and not row['accepted']:
                reasons[row['reason']] += 1
                rejected.append({'trial': row['trials'], 'reason': row['reason']})
        gate = A.train_gate(model, old, new, reg, settings, scale)
        starts = start_snapshot(model, old, new, scale, reg['primary_k'])
        assert R.digests(model) == before
        candidates.append({'run_id': c['run_id'], 'gate': gate, 'starts': starts,
            'numeric_stage_required': not numeric_pass(gate, reg),
            'v6_rejected_reasons': dict(reasons), 'v6_rejected_blocks': rejected})
    return {'schema': 'camra-nav065-staged-preflight-v7', 'host': socket.getfqdn(), 'tmux': os.environ['TMUX'],
        'train_episodes': sum(len(old) + len(new) for _, _, old, new in prepared),
        'train_steps': sum(len(t.observations) for _, _, old, new in prepared for _, t in old + new),
        'new_rollouts': 0, 'optimizer_steps': 0, 'development_scoring_rounds': 0, 'candidates': candidates,
        'limitations': ['v6 did not save intermediate per-start predictions or rejected checkpoints; exact old-to-wrong types cannot be recovered without new optimizer steps. No optimizer replay was performed.',
            'v7 stages are a bounded engineering hypothesis, not an RC-PPO theorem or a guarantee of finite-network zero error.']}


def run(protocol, pin, upstream, reg, inputs, prepared):
    out = R.WORK / protocol['output_name']
    out.mkdir(exist_ok=False)
    fitted, gates, total = [], [], 0
    torch.use_deterministic_algorithms(True)
    for c, model, old, new in prepared:
        actor = R.digests(model)['policy']
        def save_stage(stage, trial, net):
            with (out / (c['run_id'] + f'.{stage}-{trial}.pt')).open('xb') as stream:
                torch.save({'model': net.state_dict(), 'staged_refit_pin': pin, 'trial': trial, 'stage': stage}, stream)
        with (out / (c['run_id'] + '.jsonl')).open('x') as journal:
            try:
                trials, gate = fit(model, old, new, protocol, reg,
                    inputs.value['quantities']['Z_REF_J']['value'], c['seed'], journal, save_stage)
            except Exception as exc:
                R.write_record(journal, {'kind': 'run_stopped', 'error_type': type(exc).__name__, 'error': str(exc)})
                with (out / 'STOPPED.json').open('x') as stream:
                    R.write_record(stream, {'run_id': c['run_id'], 'development_scoring_rounds': 0,
                        'reason': str(exc), 'automatic_restart_allowed': False})
                raise
        total += trials
        if total > protocol['resource_limit']['optimizer_trials_total_max'] or R.digests(model)['policy'] != actor:
            raise ValueError('total quota exceeded or actor changed')
        path = out / (c['run_id'] + '.pt')
        with path.open('xb') as stream:
            torch.save({'model': model.state_dict(), 'staged_refit_pin': pin,
                'prereg_sha256': inputs.train_pin, 'implementation_versions': inputs.prereg['implementation_versions'],
                'variant': c['variant'], 'seed': c['seed']}, stream)
        if R.digests(R.load_network(inputs, c, path)) != R.digests(model):
            raise ValueError('CPU reload differs')
        fitted.append({**c, 'parent_staged_checkpoint': c['checkpoint'], 'checkpoint': str(path),
            'checkpoint_sha256': R.file_sha(path), 'state_dict_sha256': R.digests(model)})
        gates.append({'run_id': c['run_id'], 'trials': trials, **gate})
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
    with args.diagnose.open('x') as stream:
        R.write_record(stream, diagnose(reg, inputs, prepared))
    print('972 saved train episodes verified; zero optimizer steps, new rollouts or development scoring', flush=True)


if __name__ == '__main__':
    main()
