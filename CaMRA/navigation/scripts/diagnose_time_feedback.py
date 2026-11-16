"""Read existing feedback train/dev artifacts; no rollout or optimization."""
from __future__ import annotations

import argparse
import dataclasses as dc
import json
from pathlib import Path
import sys

import numpy as np
import torch

NAV = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(NAV / 'scripts'), str(NAV / 'src'), str(NAV.parent / 'calibration/src')]
from camra_navigation.algorithm import storage as S, networks as NW
from camra_navigation.algorithm.validation import file_sha


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--work-root', default='/pub/data/caohy/work/camra_navigation')
    args = ap.parse_args()
    out = Path(args.out)
    assert not out.exists()
    torch.set_num_threads(1)
    root = Path(args.work_root)
    prereg = NAV / 'config/training_loop_prereg_feedback_v1.json'
    p = json.loads(prereg.read_bytes())
    cfg = dc.replace(S.ObjectiveConfig.from_contracts(
        str(NAV / p['binding']['value_contract']['path']), r_goal_m=.5),
        joint_min_recursion=True, h_semantics='v2_deep_well')
    nav = json.loads((NAV / p['binding']['navigation_contract']['path']).read_bytes())
    devpath = NAV / 'reports/nav064_feedback_v1_development.json'
    dev = json.loads(devpath.read_bytes())
    train = []
    references = []
    dev_rows = []
    for run in dev['runs']:
        rid = run['run_id']
        index = root / 'nav064_feedback_v1_audit_u600' / (rid + '_canonical_index.jsonl')
        references.append({'path': str(index), 'sha256': file_sha(index)})
        entries = [json.loads(x) for x in index.read_text().splitlines()]
        for update in (99, 349, 599):
            counts = {'samples': 0, 'first_crossings': 0, 'first_crossings_hidden_by_distance': 0,
                      'saturated_history_nonfailure': 0, 'saturated_stop_now': 0,
                      'saturated_target_one': 0}
            changes = {str(lam): [] for lam in (.90, .95, .98, .995)}
            crossing_changes = {k: [] for k in changes}
            for e in entries:
                if e['global_step_start']//227 != update:
                    continue
                path = Path(e['store']) / e['file']
                assert file_sha(path) == e['file_sha256']
                with np.load(path) as f:
                    a = {k: f[k] for k in f.files}
                references.append({'path': str(path), 'sha256': e['file_sha256']})
                mask = (a['safety_y'] == 0) & (a['safety_y_next'] > 0)
                gnext = np.where(a['goal_distance_m_next'] <= .5, -3.,
                                 (a['goal_distance_m_next']-.5)/15.)
                counts['samples'] += len(mask)
                counts['first_crossings'] += int(mask.sum())
                counts['first_crossings_hidden_by_distance'] += int(np.sum(mask & (gnext >= a['safety_y_next'])))
                sat = (a['safety_y'] == 1) & (e['terminal_kind'] != 'failure')
                counts['saturated_history_nonfailure'] += int(sat.sum())
                counts['saturated_stop_now'] += int(np.sum(a['obj_stop_now'][sat]))
                counts['saturated_target_one'] += int(np.sum(np.isclose(a['obj_target_total'][sat], 1.)))
                for lam in (.90, .95, .98, .995):
                    rebuilt = S.compute_objectives(a, dc.replace(cfg, gamma_h=e.get('gamma_h',
                        .99999 if update >= 300 else .999 + (.99999-.999)*update/300),
                        gae_lambda=lam), terminal_kind=e['terminal_kind'])
                    if lam == .95:
                        for k in ('obj_target_cost', 'obj_target_reach', 'obj_target_total'):
                            assert np.allclose(rebuilt[k], a[k], atol=1e-10, rtol=1e-9), (rid, update, k)
                    delta = rebuilt['obj_adv_total'] - a['obj_adv_total']
                    changes[str(lam)].extend(np.abs(delta).tolist())
                    crossing_changes[str(lam)].extend(delta[mask].tolist())
            assert counts['samples'] == 4086
            sensitivity = {k: {'mean_absolute_advantage_change': float(np.mean(v)),
                'crossing_mean_signed_change': float(np.mean(crossing_changes[k]))
                    if crossing_changes[k] else None} for k, v in changes.items()}
            train.append({'run_id': rid, 'update': update, 'counts': counts,
                          'lambda_sensitivity': sensitivity})
        checkpoint = root / rid / 'state.pt'
        assert file_sha(checkpoint) == run['state_sha256']
        st = torch.load(checkpoint, map_location='cpu', weights_only=False)
        ini = p['action_init_fix']
        net = NW.build(nav, action_bounded=True, initial_forward_speed=ini['initial_forward_speed'],
            log_std_min=ini['log_std_min'], log_std_init=ini['log_std_init'],
            steering_gain=p['steering_prior']['gain'])
        net.load_state_dict(st['model']); net.eval()
        references.append({'path': str(checkpoint), 'sha256': run['state_sha256']})
        for e in run['episodes']:
            if e['k'] != 1.15:
                continue
            x = torch.tensor([s['observation'] for s in e['trace']], dtype=torch.float32)
            with torch.no_grad():
                vc, vh = net.values(x)
                actions = net.policy(x).mean.numpy()
            replay_error = float(np.max(np.abs(actions-np.array([s['action'] for s in e['trace']]))))
            assert replay_error < 2e-6
            residual = np.maximum(vh.numpy(), vc.numpy()-x[:, -2].numpy())
            dev_rows.append({'run_id': rid, 'variant': run['variant'], 'seed': run['seed'],
                'map_id': e['map_id'], 'feasible': e['feasible'], 'label': e['label'],
                'danger_steps': e['danger_steps'], 'remaining_budget_J': e['z_J_end'],
                'predicted_feasible_fraction': float(np.mean(residual <= 0)),
                'initial_W': float(residual[0]), 'final_W': float(residual[-1]),
                'action_replay_error': replay_error})
    full = [t['counts'] for t in train if '_full_' in t['run_id']]
    summary = {k: sum(x[k] for x in full) for k in full[0]}
    index_out = out.with_suffix('.sources.json')
    write_new(index_out, {'files': references})
    write_new(out, {'schema': 'camra-time-feedback-diagnosis-v1', 'owner': 'NAV-064',
        'data_access': {'train_updates': [99, 349, 599], 'train_maps_per_run': 18,
            'existing_development_only': True, 'new_rollouts': 0, 'training_updates': 0,
            'budget_selection_or_final_read': False},
        'inputs': {str(prereg): file_sha(prereg), str(devpath): file_sha(devpath)},
        'source_index': {'path': str(index_out), 'sha256': file_sha(index_out)},
        'full_sample_summary': summary, 'train': train, 'development_replay': dev_rows,
        'limitations': [
            'lambda changes existing targets only; no causal performance comparison or lambda selection.',
            'W is trained for a stochastic policy; one deterministic trajectory cannot certify an expectation error.',
            'Unsafe plateaus follow irreversible history; removing history would change the objective.',
            'Raw 16-bin obstacle features do not directly encode inflated clearance; additional features remain a separate candidate.',
            'Old obs-v5 time clipping cannot be inverted to recover the missing first15 seconds.']})
    print(json.dumps({'full_sample_summary': summary, 'dev_episodes': len(dev_rows),
                      'sources': len(references), 'new_rollouts': 0}))


if __name__ == '__main__':
    main()
