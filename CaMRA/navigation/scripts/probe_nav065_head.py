"""NAV-065 v10: read-only head geometry, then a separately approved head probe.

The diagnosis computes ranks and necessary correction norms without solving for
or assigning candidate weights. A registered probe solves each original linear
head once on a deep copy, then uses the production local/full-train checks.
Neither mode scores development or authorizes deployment.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import socket

import numpy as np
import torch

import probe_nav065_local as L

R = L.R
PROPOSAL = R.NAV / 'config/nav065_head_probe_proposal_v10.json'
BOUND_INPUTS = tuple(dict.fromkeys((*L.BOUND_INPUTS,
    'config/nav065_local_probe_prereg_v9.json',
    'reports/nav065_local_probe_v9_review.json',
    'scripts/probe_nav065_head.py', 'tests/test_nav065_head_probe.py',
    'config/nav065_head_probe_proposal_v10.json')))


def approved_registration(path):
    protocol = R.load(path)
    if (protocol.get('approved') is not True or protocol.get('status') != 'APPROVED'
            or not protocol.get('authorization_reference')):
        raise ValueError('candidate head fitting needs a new explicit approval')
    pin = path.with_suffix('.json.sha256').read_text().strip()
    R.verify_hash(path, pin)
    omit = {'approved', 'status', 'authorization_reference', 'implementation_hashes'}
    if {k: v for k, v in protocol.items() if k not in omit} != {
            k: v for k, v in R.load(PROPOSAL).items() if k not in omit}:
        raise ValueError('head registration differs from reviewed proposal')
    if set(protocol.get('implementation_hashes', {})) != set(BOUND_INPUTS):
        raise ValueError('incomplete head implementation binding')
    for name, digest in protocol['implementation_hashes'].items():
        R.verify_hash(R.NAV / name, digest)
    return protocol, pin


def features(critic, observations):
    with torch.no_grad():
        z = critic.body(torch.as_tensor(observations, dtype=torch.float32)).cpu().numpy()
    if z.ndim != 2 or not np.isfinite(z).all():
        raise ValueError('invalid finite head features')
    return np.c_[z.astype(np.float64), np.ones(len(z))]


def head_vector(critic):
    return np.r_[critic.head.weight.detach().cpu().numpy().reshape(-1),
                 critic.head.bias.detach().cpu().numpy()].astype(np.float64)


def spectrum(design, target, current, rcond):
    """Projection residual/norm bound only; do not construct fitted weights."""
    a, y, w = (np.asarray(x, dtype=np.float64) for x in (design, target, current))
    if (a.ndim != 2 or y.shape != (len(a),) or w.shape != (a.shape[1],)
            or not 0 < rcond < 1 or not all(np.isfinite(x).all() for x in (a, y, w))):
        raise ValueError('invalid spectral diagnostic inputs')
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    keep = s > rcond * s[0] if len(s) else np.zeros(0, bool)
    residual = y - a @ w
    coefficients = u[:, keep].T @ residual
    unreachable = residual - u[:, keep] @ coefficients
    rank32 = int(np.count_nonzero(s > np.finfo(np.float32).eps * max(a.shape) * s[0]))
    return {'rows': len(a), 'columns': a.shape[1], 'rank_at_solve_rcond': int(keep.sum()),
        'rank_at_float32_tolerance': rank32, 'singular_values': s.tolist(),
        'condition_number': float(s[0] / s[-1]) if s[-1] > 0 else None,
        'target_residual_outside_retained_span_linf': float(np.abs(unreachable).max()),
        'minimum_correction_norm_for_retained_projection': float(np.linalg.norm(coefficients / s[keep])),
        'current_head_norm': float(np.linalg.norm(w)), 'current_prediction_mae': float(np.abs(residual).mean()),
        'candidate_weights_solved': False, 'candidate_parameters_assigned': 0}


def verified_v9(inputs, candidate):
    review = R.load(R.NAV / 'reports/nav065_local_probe_v9_review.json')
    root = Path(review['evidence']['output'])
    for name, digest in review['evidence']['output_hashes'].items():
        R.verify_hash(root / name, digest)
    _, pin = L.approved_registration(R.NAV / 'config/nav065_local_probe_prereg_v9.json')
    result = R.load(root / 'PROBE_FINISHED.json')
    if result['pin'] != pin or result['optimizer_trials'] != 400:
        raise ValueError('v9 completion identity differs')
    model = R.load_network(inputs, candidate, root / 'diagnostic-probe.pt')
    if R.digests(model) != review['diagnostic_state_dict_hashes']:
        raise ValueError('v9 tensor identity differs')
    rows = [json.loads(line) for line in (root / 'local.jsonl').read_text().splitlines()]
    ev = [r for r in rows if r['kind'] == 'local_evaluation']
    attempts = [r['trial'] for r in rows if r['kind'] == 'optimizer_attempt']
    if [r['trial'] for r in ev] != list(range(401)) or attempts != list(range(1, 401)):
        raise ValueError('v9 evaluation sequence differs')
    return model, ev, R.load(root / 'selection.json')


def diagnose(reg, inputs, prepared):
    before = {c['run_id']: R.digests(m) for c, m, _, _ in prepared}
    d, keys = L.diagnose(reg, inputs, prepared)
    c, source, old, new = next(row for row in prepared if row[0]['run_id'] == L.P.TARGET)
    after, ev, selection = verified_v9(inputs, c)
    if [list(k) for k in keys] != selection['local_keys']:
        raise ValueError('v9 fixed local selection differs')
    batch = L.local_batch(old + new, keys)
    if not all(np.array_equal(batch[i], selection['local_targets'][name]) for i, name in
               ((1, 'budget_J'), (2, 'cost_norm'), (3, 'reach'))):
        raise ValueError('v9 local truth differs')
    proposal = R.load(PROPOSAL)
    if c['checkpoint_sha256'] != proposal['source_checkpoint_sha256']:
        raise ValueError('fixed v8 source differs')
    spectra = {}
    for stage, model in (('v8_source', source), ('v9_final', after)):
        spectra[stage] = {role: spectrum(features(getattr(model, role), batch[0]),
            batch[i], head_vector(getattr(model, role)), proposal['solve']['rcond'])
            for role, i in (('critic_cost', 2), ('critic_reach', 3))}
    curve = [{k: e[k] for k in ('trial', 'local_pass', 'cost_sign_errors', 'reach_sign_errors',
                               'joint_sign_errors', 'false_feasible', 'false_infeasible', 'metrics')}
             for e in ev if e['trial'] in (0, 1, 10, 50, 100, 200, 300, 400)]
    sign_groups = []
    for positive in (False, True):
        mask = (batch[3] > 0) == positive
        truth, pred = batch[3][mask], np.array(ev[-1]['predictions']['vh'])[mask]
        sign_groups.append({'positive_target': positive, 'n': int(mask.sum()),
            'target_min': float(truth.min()), 'target_max': float(truth.max()),
            'prediction_min': float(pred.min()), 'prediction_max': float(pred.max())})
    assert before == {c['run_id']: R.digests(m) for c, m, _, _ in prepared}
    return {'schema': 'camra-nav065-head-preflight-v10', 'host': socket.getfqdn(),
        'tmux': os.environ.get('TMUX'), 'source_checkpoint_sha256': c['checkpoint_sha256'],
        'source_gate_exact_match': True, 'source_full_train_gate': d['full_train_gate'],
        'available_train_episodes': len(old) + len(new), 'available_train_steps': sum(len(t.observations) for _, t in old + new),
        'local_keys': keys, 'local_rows': len(keys), 'source_local': d['local_source_evaluation'],
        'spectra': spectra, 'v9_curve': curve, 'v9_final_reach_sign_groups': sign_groups,
        'v9_joint_errors_min': min(e['joint_sign_errors'] for e in ev),
        'v9_all_local_checks_ever_passed': any(e['local_pass'] for e in ev),
        'formal_optimizer_steps': 0, 'formal_head_solves': 0, 'new_rollouts': 0,
        'development_scoring_rounds': 0, 'source_models_unchanged': True,
        'scientific_pass': False, 'deployment_allowed': False,
        'limitations': ['Spectral residual and norm bounds do not assign a candidate head or prove float32 fitting success.',
            'These selected train starts do not establish full-task representation or generalization.',
            'A local solve remains model fitting and requires a separately approved run, even with zero optimizer steps.']}


def solve_head(critic, observations, target, rcond):
    """One minimum-displacement least-squares solve; only call for an approved copy."""
    a = features(critic, observations)
    w = head_vector(critic)
    y = np.asarray(target, dtype=np.float64)
    if y.shape != (len(a),) or not np.isfinite(y).all() or not 0 < rcond < 1:
        raise ValueError('invalid head solve target or tolerance')
    delta, _, rank, singular = np.linalg.lstsq(a, y - a @ w, rcond=rcond)
    fitted = w + delta
    if not np.isfinite(fitted).all() or not np.isfinite(fitted.astype(np.float32)).all():
        raise ValueError('nonfinite head solution')
    with torch.no_grad():
        critic.head.weight.copy_(torch.as_tensor(fitted[:-1], dtype=critic.head.weight.dtype).reshape_as(critic.head.weight))
        critic.head.bias.copy_(torch.as_tensor(fitted[-1:], dtype=critic.head.bias.dtype))
        actual = critic(torch.as_tensor(observations, dtype=torch.float32)).numpy()
    return {'rank': int(rank), 'singular_values': singular.tolist(), 'correction_norm': float(np.linalg.norm(delta)),
        'double_design_residual_linf': float(np.abs(a @ fitted - y).max()),
        'production_float32_residual_linf': float(np.abs(actual - y).max()),
        'production_predictions': actual.tolist(), 'head_solves': 1}


def run(protocol, pin, reg, inputs, prepared):
    limits = protocol['resource_limit']
    if (limits['update_candidates'] != [L.P.TARGET]
            or limits['head_linear_solves_total_max'] != 2
            or limits['solves_per_original_critic_head'] != 1
            or limits['optimizer_steps_total_max'] != 0
            or limits['new_train_episodes_max'] != 0
            or limits['development_scoring_rounds_max'] != 0):
        raise ValueError('head probe scope or quota differs')
    out = R.WORK / protocol['output_name']
    if out.exists():
        raise ValueError('head probe output exists; inspect, never overwrite or restart')
    diagnostic = diagnose(reg, inputs, prepared)
    c, source, old, new = next(row for row in prepared if row[0]['run_id'] == L.P.TARGET)
    before = {x['run_id']: R.digests(m) for x, m, _, _ in prepared}
    copy_model = copy.deepcopy(source)
    batch = L.local_batch(old + new, [tuple(k) for k in diagnostic['local_keys']])
    scale = inputs.value['quantities']['Z_REF_J']['value']
    out.mkdir(exist_ok=False)
    with (out / 'preflight.json').open('x') as f:
        json.dump(diagnostic, f, ensure_ascii=False, indent=2, allow_nan=False)
    solves = {}
    try:
        for role, i in (('critic_cost', 2), ('critic_reach', 3)):
            with (out / 'attempts.jsonl').open('a') as f:
                R.write_record(f, {'kind': 'head_solve_attempt', 'role': role})
            solves[role] = solve_head(getattr(copy_model, role), batch[0], batch[i], protocol['solve']['rcond'])
        allowed = {f'{role}.head.{name}' for role in R.SCORE_ROLES for name in ('weight', 'bias')}
        for name, original in source.state_dict().items():
            if name not in allowed and not torch.equal(original, copy_model.state_dict()[name]):
                raise ValueError('non-head parameter changed: ' + name)
        local = L.evaluate_local(copy_model, batch, reg, scale)
        full = L.P.A.train_gate(copy_model, old, new, reg, R.load(L.P.PROPOSAL)['fit'], scale)
        checkpoint = out / 'diagnostic-head.pt'
        torch.save({'model': copy_model.state_dict(), 'diagnostic_only': True, 'pin': pin}, checkpoint)
        if R.digests(R.load_network(inputs, c, checkpoint)) != R.digests(copy_model):
            raise ValueError('head probe CPU reload differs')
        assert before == {x['run_id']: R.digests(m) for x, m, _, _ in prepared}
        result = {'pin': pin, 'head_solves': solves, 'local': local,
            'full_train_before': diagnostic['source_full_train_gate'], 'full_train_after': full,
            'source_models_unchanged': True, 'actor_and_bodies_unchanged': True,
            'checkpoint_sha256': R.file_sha(checkpoint), 'optimizer_steps': 0, 'new_rollouts': 0,
            'development_scoring_rounds': 0, 'scientific_nav065_pass': False, 'deployment_allowed': False,
            'automatic_expansion_allowed': False}
        with (out / 'HEAD_FINISHED.json').open('x') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    except Exception as exc:
        torch.save({'model': copy_model.state_dict(), 'diagnostic_only': True, 'pin': pin}, out / 'failed-head.pt')
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
    report = diagnose(*L.sources()[1:])
    with args.diagnose.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: report[k] for k in ('local_rows', 'formal_head_solves', 'formal_optimizer_steps', 'scientific_pass')}))


if __name__ == '__main__':
    main()
