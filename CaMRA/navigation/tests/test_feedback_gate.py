"""NAV-064: independent boundary, feedback, updater and train-replay evidence.

This is a CPU numerical gate. It does not launch new navigation training or
evaluate a candidate on development, selection or final data.
"""
from __future__ import annotations

import argparse
import ast
import copy
import dataclasses as dc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

NAV = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(NAV / 'scripts'), str(NAV / 'tests'), str(NAV / 'src'),
                str(NAV.parent / 'calibration/src')]
import train_formal as F
from camra_navigation.algorithm import storage as S, targets as T, ppo as P, networks as NW
from camra_navigation.algorithm.validation import verify_training_bindings
from test_r14_repair_gate import oracle_min, oracle_cost, episode

CHECKS = []
CFG = dc.replace(S.ObjectiveConfig.from_contracts(
    str(NAV / 'config/value_objective_contract_feedback_v1.json'), r_goal_m=.5),
    h_semantics='v2_deep_well', joint_min_recursion=True)
LEGACY = dc.replace(CFG, estimator_version='rcppo-r14-v3')
OBJECTIVES = ('obj_target_cost', 'obj_target_reach', 'obj_target_total')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(a, b):
    return np.allclose(a, b, atol=1e-12, rtol=1e-10)


def check(name, ok, detail=None):
    CHECKS.append({'check': name, 'ok': bool(ok), 'detail': detail})
    print(('PASS ' if ok else 'FAIL ') + name, flush=True)


def fixture(dist=(4., 3., 2.), y=None, event='timeout', **kw):
    kind = 'truncation' if event == 'none' else S.terminal_kind_for_event(event, CFG)
    a = episode(list(dist), y, kind=kind, **kw)
    a['event_code'] = np.zeros(len(dist)-1, dtype=np.int32)
    a['event_code'][-1] = S.EVENT_CODES[event]
    return a, kind


def calc(a, kind):
    return S.compute_objectives(a, CFG, terminal_kind=kind)


def boundaries():
    F.obj_cfg = CFG
    for event in ('timeout', 'stagnation', 'goal_reached', 'collision', 'out_of_bounds'):
        a, kind = fixture(event=event)
        x, used = F.close_segment(SimpleNamespace(arrays=lambda: copy.deepcopy(a)),
            'truncation', .7, -.9, env_done=True, label=event)
        z, _ = F.close_segment(SimpleNamespace(arrays=lambda: copy.deepcopy(a)),
            'truncation', 99., -99., env_done=True, label=event)
        check('production_ending_' + event, used == kind and x['terminated'][-1]
              and not x['truncated'][-1] and all(same(x[k], z[k]) for k in OBJECTIVES))
    check('task_deadline_no_learned_future', all(c['ok'] for c in CHECKS
          if c['check'] in ('production_ending_timeout', 'production_ending_stagnation')))
    a, kind = fixture(event='none', bh=-.1)
    x = calc(a, kind)
    b = copy.deepcopy(a); b['bootstrap_value_reach'][-1] = -.6
    z = calc(b, kind)
    check('window_consumes_actual_reach', not same(x[OBJECTIVES[1]], z[OBJECTIVES[1]])
          and not same(x[OBJECTIVES[2]], z[OBJECTIVES[2]]))
    b = copy.deepcopy(a); b['bootstrap_value_cost'][-1] = 3.
    z = calc(b, kind)
    check('window_consumes_actual_cost', not same(x[OBJECTIVES[0]], z[OBJECTIVES[0]])
          and not same(x[OBJECTIVES[2]], z[OBJECTIVES[2]]))
    # Wrong event identity or irreversible history must not enter production.
    for name, mutate in (
        ('deadline_as_slice', lambda b: b['truncated'].__setitem__(-1, True)),
        ('reversed_history', lambda b: (b['safety_y'].__setitem__(0, .8),
                                        b['safety_y_next'].__setitem__(0, .2)))):
        b, _ = fixture(); mutate(b)
        try: calc(b, 'task_end'); rejected = False
        except S.StorageError: rejected = True
        check('reject_' + name, rejected)
    for name, ys, z0 in (('unsafe_arrival', [0., .2, .2], 1.),
                          ('overspent_arrival', None, -.1)):
        a, kind = fixture(dist=(1.2, .7, .2), y=ys, z0=z0, event='goal_reached')
        check(name + '_infeasible', not calc(a, kind)['obj_feasible'].any())


def independent_enumeration():
    rng = np.random.default_rng(6404)
    ok = True; cases = 0
    for n in (1, 2, 5):
        for stop in range(n+1):
            for lam in (0., .5, .95):
                y = np.maximum.accumulate(rng.choice([0., .2, .6, 1.], n+1))
                h = np.maximum(y, rng.uniform(-1, 2, n+1))
                raw = rng.uniform(-2, 2, n+1)
                # Independent projection, without calling the production helper.
                v = np.array([max(yy, min(vv, hh)) if yy > 0 else min(vv, hh)
                              for vv, hh, yy in zip(raw, h, y)])
                bounded = T.bounded_reach_continuation(raw, h, y, 1.)
                ok &= same(v, bounded)
                d = np.zeros(n, dtype=bool)
                if stop < n: d[stop] = True
                for gamma in (.999, 1.):
                    ok &= same(T.min_gae_return(h, bounded, d, gamma, lam),
                               oracle_min(h, v, d, gamma, lam))
                    cases += 1
    check('independent_bounded_n_step_enumeration', ok, {'cases': cases})
    a, kind = fixture(y=[.2, .6, 1.], event='none', bh=-2.)
    a['value_reach'][:] = -2.
    out = calc(a, kind)
    check('unsafe_targets_respect_irreversible_bound',
          np.all(out['obj_target_reach'] >= a['safety_y'] * CFG.C - 1e-12)
          and np.all(out['obj_target_total'] >= a['safety_y'] * CFG.C - 1e-12))
    check('absorbing_failure_has_no_stopping_upper_bound',
          same(T.bounded_reach_continuation([1.], [.2], [.2], 1., allow_stop=False), [1.]))
    # Raw prediction is the regression consumer, not its clipped continuation.
    raw = torch.nn.Parameter(torch.tensor([-.9]))
    loss, _ = P.value_loss(raw, raw.detach(), torch.tensor([.6]), P.PPOConfig())
    loss.backward()
    check('raw_reach_regression_has_correct_gradient', raw.grad.item() < 0.)
    d = np.array([False]); q = np.array([.2, 0.]); v = q.copy()
    exact = float(T.min_gae_return(q, v, d, 1., .95)[0])
    approx = float(T.min_gae_return(q, v, d, .999, .95)[0])
    check('joint_gamma_finite_zero_threshold', exact == 0. and approx > 0.,
          {'gamma1': exact, 'gamma0999': approx, 'mechanical_cost_discount': 1.})
    # Full production objectives, independently composed from enumerated n-step targets.
    for event in ('none', 'timeout', 'goal_reached', 'collision'):
        a, kind = fixture(event=event, y=[0., .2, .6], bh=-.7)
        o = calc(a, kind)
        h = np.r_[o['obj_h'], o['obj_h_terminal'][-1]]
        upper = h[:-1] if kind != 'failure' else np.full(len(h)-1, np.inf)
        vh = np.maximum(np.where(a['safety_y'] > 0, a['safety_y'], -np.inf),
                        np.minimum(a['value_reach'], upper))
        bh = h[-1] if kind != 'truncation' else max(a['safety_y_next'][-1],
                         min(h[-1], a['bootstrap_value_reach'][-1]))
        bc = a['bootstrap_value_cost'][-1] if kind == 'truncation' else 0.
        c = a['cost_norm'].copy()
        if kind == 'failure': c[-1] = CFG.fail_penalty
        d = o['obj_stop_now']
        vtot = np.r_[np.maximum(vh, a['value_cost']-a['z_norm']),
                     max(bh, bc-a['z_norm_next'][-1])]
        q = np.r_[o['obj_g_train'], o['obj_g_terminal'][-1]]
        expected = [oracle_cost(c, np.r_[a['value_cost'], bc], d, 1., .95),
                    oracle_min(h, np.r_[vh, bh], d, CFG.gamma_h, .95, kind == 'failure'),
                    oracle_min(q, vtot, d, 1., .95, kind == 'failure')]
        check('production_three_estimators_oracle_' + event,
              all(same(o[key], val) for key, val in zip(OBJECTIVES, expected)))


def feedback_update():
    a, kind = fixture(event='none', y=[0., .8, .8], bh=-.9)
    b, _ = fixture(event='none', y=[0., 0., 0.], bh=-.9)
    bad, good = calc(a, kind), calc(b, kind)
    adv = torch.tensor([bad['obj_adv_total'][0], good['obj_adv_total'][0]], dtype=torch.float64)
    logits = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    opt = torch.optim.SGD([logits], lr=.1)
    old = torch.log_softmax(logits.detach(), 0).clone()
    norm, _ = P.normalize_advantage(adv)
    for _ in range(3):
        opt.zero_grad()
        loss, _ = P.policy_loss(torch.log_softmax(logits, 0), old, norm,
                                torch.zeros(2), dc.replace(P.PPOConfig(), ent_coef=0.))
        loss.backward(); opt.step()
    probability = float(torch.softmax(logits.detach(), 0)[0])
    check('crossing_action_bad_probability_decreases', adv[0] > adv[1] and probability < .5,
          {'bad_advantage': adv[0].item(), 'safe_advantage': adv[1].item(),
           'probability_before': .5, 'probability_after': probability})
    prereg = {'dead_step_mask': {'enabled': False}}
    ys = np.array([0., .2, .6, 1.])
    masks = F.training_sample_masks(ys, prereg, CFG)
    nav = json.loads((NAV / 'config/navigation_contract_r14_repair_v2.json').read_bytes())
    torch.manual_seed(6404)
    net = NW.build(nav, action_bounded=True)
    obs = torch.zeros((4, nav['observation_spec']['total_dim']))
    with torch.no_grad():
        actions, logp, _ = net.policy.act(obs)
        vc, vh = net.values(obs)
    batch = dict(obs=obs, action=actions, log_prob_old=logp.clone(),
                 adv_total=torch.tensor([-1., -.2, .6, 2.]),
                 value_cost_old=vc.clone(), target_cost=vc+.1,
                 value_reach_old=vh.clone(), target_reach=torch.tensor([-.4, .2, .6, 1.]),
                 **masks)
    updater = P.MinimizationUpdater(net, dc.replace(P.PPOConfig(), max_grad_norm=2.5))
    with mock.patch.object(P, 'normalize_advantage', wraps=P.normalize_advantage) as norm_spy, \
         mock.patch.object(P, 'value_loss', wraps=P.value_loss) as value_spy:
        stats = updater.update(batch)
        all_rows = norm_spy.call_args.args[0].shape[0] == 4
        all_rows &= norm_spy.call_args.kwargs.get('mask') is None
        all_rows &= all(c.args[0].shape[0] == 4 and c.kwargs.get('mask') is None
                        for c in value_spy.call_args_list)
    check('all_graded_states_in_loss', masks == {} and all_rows
          and abs(stats['adv_mean']-.35) < 1e-6 and stats['grad_norm_reach'] > 0.)
    legacy = F.training_sample_masks(ys, {'dead_step_mask': {'enabled': True}}, LEGACY)
    check('legacy_mask_preserved', same(legacy['live_policy'].numpy(), [1, 1, 0, 0]))


def replay_train(out):
    root = Path('/pub/data/caohy/work/camra_navigation')
    evidence = []; ok = True; old_ok = True
    for method in ('uncalibrated', 'full'):
        for seed in range(3):
            rid = f'nav064_r14a2_{method}_s{seed}'
            rows = []; selected = []
            # Every train environment, at fixed predeclared update positions.
            for env in range(18):
                folder = root / rid / f'rollout_env{env}' / 'rollout'
                manifest = folder / 'manifest.jsonl'
                for line in manifest.read_text().splitlines():
                    e = json.loads(line)
                    u = e['global_step_start']//227
                    if u not in (0, 300, 599): continue
                    path = folder / e['file']
                    with np.load(path) as f: a = {k: f[k] for k in f.files}
                    assert sha(path) == e['file_sha256'] and S.content_hash(a) == e['content_sha256']
                    old_cfg = dc.replace(LEGACY, gamma_h=e['gamma_h'])
                    old = S.compute_objectives(a, old_cfg, terminal_kind=e['terminal_kind'])
                    old_ok &= all(same(old[k], a[k]) for k in old)
                    event = S.EVENT_NAMES[int(a['event_code'][-1])]
                    kind = 'truncation' if event == 'none' else S.terminal_kind_for_event(event, CFG)
                    b = copy.deepcopy(a)
                    b['terminated'][-1] = kind != 'truncation'
                    b['truncated'][-1] = kind == 'truncation'
                    fresh_cfg = dc.replace(CFG, gamma_h=e['gamma_h'])
                    fresh = S.compute_objectives(b, fresh_cfg, terminal_kind=kind)
                    safety = np.asarray(b['safety_y'], dtype=np.float64)
                    unsafe = safety > 0
                    # Promote before subtracting: float32(1)-1e-12 rounds to 1.
                    ok &= np.all(fresh['obj_target_reach'][unsafe] >= safety[unsafe]-1e-12)
                    ok &= np.all(fresh['obj_target_total'][unsafe] >= safety[unsafe]-1e-12)
                    ok &= all(np.isfinite(fresh[k]).all() for k in OBJECTIVES)
                    rows.append({'samples': len(a['cost_norm']), 'unsafe': int(unsafe.sum()),
                                 'event': event, 'kind': kind,
                                 'changed_total': int(np.sum(np.abs(fresh['obj_target_total']-a['obj_target_total']) > 1e-10)),
                                 'max_total_change': float(np.max(np.abs(fresh['obj_target_total']-a['obj_target_total'])))})
                    selected.append({'path': str(path), 'sha256': e['file_sha256'],
                                     'content_sha256': e['content_sha256']})
            record = {'run_id': rid, 'manifest_chunks': len(rows),
                      'samples': sum(r['samples'] for r in rows),
                      'unsafe_samples': sum(r['unsafe'] for r in rows),
                      'changed_total_samples': sum(r['changed_total'] for r in rows),
                      'max_total_change': max(r['max_total_change'] for r in rows),
                      'task_ends': sum(r['kind'] == 'task_end' for r in rows),
                      'selected_chunks': selected}
            ok &= record['samples'] == 3*4086
            evidence.append(record)
            print('REPLAY', rid, record['samples'], 'samples', flush=True)
    replay = out.with_suffix('.replay.json')
    with replay.open('x') as f: json.dump({'split': 'train', 'updates': [0, 300, 599],
        'runs': evidence, 'legacy_ok': bool(old_ok), 'feedback_bounds_ok': bool(ok)}, f, indent=2)
    check('legacy_r14a2_stored_targets_unchanged', old_ok)
    check('train_replay_feedback_bounds', ok,
          {'runs': 6, 'samples': sum(r['samples'] for r in evidence),
           'report': str(replay), 'sha256': sha(replay)})


def startup(report, out):
    old = json.loads((NAV / 'config/training_loop_prereg_r14_repair_v2.json').read_bytes())
    p = copy.deepcopy(old)
    p['implementation_versions'] = report['implementation_versions']
    p['dead_step_mask'] = {'enabled': False}
    p['feedback_policy'] = 'finite_horizon_all_states'
    p['authorization']['training_approved'] = True  # In-memory fixture only; never written as an approval.
    p['binding']['navigation_contract'] = {'path': 'config/navigation_contract_feedback_v1.json',
        'sha256': sha(NAV / 'config/navigation_contract_feedback_v1.json')}
    p['binding']['value_contract'] = {'path': 'config/value_objective_contract_feedback_v1.json',
                                     'sha256': sha(NAV / 'config/value_objective_contract_feedback_v1.json')}
    p['binding']['numerical_gate'] = {'path': str(out.resolve()), 'sha256': sha(out)}
    for e in p['binding'].values(): e['sha256'] = sha(NAV / e['path'])
    verify_training_bindings(p, NAV)
    for name in ('approval', 'mask', 'policy', 'source'):
        bad = copy.deepcopy(p)
        if name == 'approval': bad['authorization']['training_approved'] = False
        if name == 'mask': bad['dead_step_mask']['enabled'] = True
        if name == 'policy': bad['feedback_policy'] = 'legacy'
        if name == 'source': bad['binding']['value_contract']['sha256'] = '0'*64
        try: verify_training_bindings(bad, NAV); rejected = False
        except ValueError: rejected = True
        check('startup_reject_feedback_' + name, rejected)


def coordinator_interfaces():
    import run_feedback_round as runner
    rows = [{'variant': v, 'seed': s, 'run_id': f'nav064_feedback_v1_{v}_s{s}'}
            for v in ('uncalibrated', 'full') for s in range(3)]
    p = {'runtime_contract': {'run_prefix': 'nav064_feedback_v1'}}
    paths = runner.attempt_paths(p)
    commands = [runner.training_command(row, 'approved-feedback.json') for row in rows]
    check('feedback_runner_current_registry_and_prereg', len(commands) == 6
          and all(row['run_id'] in cmd and cmd[-1] == 'approved-feedback.json'
                  for row, cmd in zip(rows, commands))
          and len(set(paths.values())) == len(paths))
    cmd = runner.audit_command(100, 'approved-feedback.json', paths)
    tree = ast.parse((NAV / 'scripts/audit_r14a2.py').read_text())
    check('feedback_auditor_consumes_selected_prereg', cmd[cmd.index('--prereg')+1] == 'approved-feedback.json'
          and any(isinstance(x, ast.Call) and ast.unparse(x.func) == 'F.verify_prereg'
                  and ast.unparse(x.args[0]) == 'args.prereg' for x in ast.walk(tree)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
    args = ap.parse_args(); out = Path(args.out).resolve()
    assert not out.exists()
    torch.set_num_threads(1)
    component = out.with_suffix('.legacy.json')
    subprocess.run([sys.executable, str(NAV / 'tests/test_r14a2_gate.py'),
                    '--out', str(component)], check=True)
    base = json.loads(component.read_bytes())
    CHECKS.extend(base['checks'])
    boundaries(); independent_enumeration(); feedback_update(); replay_train(out); coordinator_interfaces()
    files = set(base['code_files']) | {'tests/test_feedback_gate.py',
             'config/value_objective_contract_feedback_v1.json',
             'config/navigation_contract_feedback_v1.json', 'scripts/run_feedback_round.py'}
    report = dict(base, schema='camra-feedback-gate-v1',
                  implementation_versions={'observation': 'obs-v5', 'estimator': 'rcppo-feedback-v4'},
                  checks=CHECKS, verdict='PASS' if all(c['ok'] for c in CHECKS) else 'FAIL',
                  code_files={f: sha(NAV / f) for f in sorted(files)},
                  protocol={'path': 'config/feedback_numerical_protocol_v1.json',
                            'sha256': sha(NAV / 'config/feedback_numerical_protocol_v1.json')},
                  component_reports={str(component): sha(component)},
                  limits=['CPU controlled updates and existing train replay only; no new performance evidence.',
                          'Safety continuation bounds are a versioned Go2 adaptation, not literal RC-PPO source.',
                          'NAV-064 stays false until authorized training and development acceptance pass.'])
    # Probe a provisional report without rewriting the final report/old evidence.
    probe = out.with_suffix('.startup.json')
    with probe.open('x') as f: json.dump(report, f, indent=2)
    if report['verdict'] == 'PASS': startup(report, probe)
    report['checks'] = CHECKS
    report['verdict'] = 'PASS' if all(c['ok'] for c in CHECKS) else 'FAIL'
    with out.open('x') as f: json.dump(report, f, ensure_ascii=False, indent=2); f.write('\n')
    print(report['verdict'], len(CHECKS), 'checks', flush=True)
    return int(report['verdict'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
