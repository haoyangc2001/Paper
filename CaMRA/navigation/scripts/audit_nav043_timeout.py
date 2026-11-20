"""Read saved timeout diagnostic evidence; no model inference or physical work."""
from datetime import datetime
import os
from pathlib import Path
import socket

import numpy as np

from nav067_sources import E, NAV, ROOT, LEVELS, load, sha
from nav067_runtime import persist
from nav067_remaining_audit import audit_episode
from audit_nav043_exploration import review_runtime
from prepare_nav065_sources import REGISTRATION
from train_formal import verify_prereg
from camra_navigation.budget.accounting import consistent_e_ref
from camra_navigation.budget.guarded import select_guarded_budget
from camra_navigation.envs.physical_v2 import PhysicalState, SubstepNavigation
from nav043_timeout import TimeoutNavigation, diagnostic_outcome

OUTPUT = Path('/pub/data/caohy/work/camra_navigation/nav043_timeout_extension_v1')
REPORT = NAV/'reports/nav043_timeout_extension_review_v1.json'


def main():
    assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
    protocol_path = NAV/'config/nav043_timeout_extension_v1.json'
    protocol = load(protocol_path)
    assert sha(protocol_path) == Path(str(protocol_path)+'.sha256').read_text().strip()
    terminal = OUTPUT/'FINISHED.json'
    if not terminal.exists():
        terminal = OUTPUT/'HELD.json'
    assert terminal.exists(), 'allocation is still active'
    for name, digest in protocol['sealed_hashes'].items():
        assert sha(name) == digest, name
    pair = load(NAV/'config/nav066_decision_prereg_v2.json')
    protected = {str(NAV/k):v for k,v in pair['implementation_hashes'].items()}
    protected.update(pair['protected_hashes'])
    protected.update({c['checkpoint']:c['checkpoint_sha256'] for c in pair['six_candidates']})
    for name, digest in protected.items():
        assert sha(name) == digest, name
    prereg, _ = verify_prereg(REGISTRATION)
    manifest = load(protocol['pilot_manifest'])
    reports, incomplete = [], []
    steps = queries = 0
    for level in LEVELS:
        app = OUTPUT/level
        if not (app/'FINISHED.json').exists():
            incomplete.append({'level':level, 'status':'application not finished',
                'error':load(app/'ERROR.json') if (app/'ERROR.json').exists() else None})
            continue
        assert load(app/'EXIT.json')['exit_code'] == 0 and not (app/'ERROR.json').exists()
        calls = load(app/'physical_step_accounting.json')['calls']
        assert calls <= 20000
        steps += calls
        row = next(r for r in manifest['maps'] if r['level'] == level)
        review_runtime(load(app/'runtime.json'), row)
        trial = app/'full'
        saved = load(trial/'decision.json')
        assert sha(trial/'decision.json') == load(trial/'decision.pin.json')['sha256']
        assert saved['query_count'] == 5 and not saved['physical_actions_started']
        queries += saved['query_count']
        ctx = E.build_rollout_ctx('full')
        spec = E.M.build(level, row['seed'], ctx['design'])
        assert spec.to_dict() == row['spec']
        initial = PhysicalState(**saved['initial_state'])
        qiter = iter(saved['decision']['trace'])
        def query(z):
            q = next(qiter)
            assert z == q['budget_J']
            return (float('nan') if q['vc_norm'] is None else q['vc_norm'],
                    float('nan') if q['vh'] is None else q['vh'])
        repeated = select_guarded_budget(query, consistent_e_ref(row['z0_J'], 'full'))
        assert repeated == saved['decision']
        for q, obs in zip(repeated['trace'], saved['query_observations']):
            mon = SubstepNavigation(spec, ctx['geom'], ctx['cm'], [initial], [q['budget_J']], prereg)
            np.testing.assert_array_equal(np.asarray(mon.observation(0), dtype=np.float32), obs)
        result = load(trial/'result.json')
        if repeated['selected_budget_J'] is None:
            assert result['rejected'] and not (trial/'NAVIGATION_STARTED.json').exists()
            reports.append({'level':level, 'rejected':True})
            continue
        z = repeated['selected_budget_J']
        assert result['z0_J'] == z
        assert load(trial/'NAVIGATION_STARTED.json')['decision_sha256'] == sha(trial/'decision.json')
        audited = audit_episode(trial)
        rows = load(trial/'substeps.json')
        navigation = load(trial/'navigation.json')
        assert len(rows) <= 18000 and len(navigation) <= 900
        mon = TimeoutNavigation(spec, ctx['geom'], ctx['cm'], [initial], [z], prereg)
        np.testing.assert_array_equal(navigation[0]['observation'],
            saved['query_observations'][repeated['selected_index']])
        for index, r in enumerate(rows):
            if index % 20 == 0:
                nav = navigation[index//20]
                assert nav['physics_tick'] == index
                np.testing.assert_array_equal(
                    np.asarray(mon.observation(0), dtype=np.float32), nav['observation'])
            mon.set_commands([r['applied']])
            state = r['state']
            mon.advance([PhysicalState(*state[:7], tuple(state[7:]))], physics_tick=r['physics_tick'])
            assert mon.rows[-1] == r, (level, r['physics_tick'])
        metrics = diagnostic_outcome(result['label'], result['safety_y'], result['t_s'],
            result['measured_joint_work_J'], result['internal_model_cost_J'], z)
        assert all(result[k] == v for k,v in metrics.items())
        cutoff = min(len(rows), 9000)
        with np.load(trial/'energy.npz', allow_pickle=False) as archive:
            work45 = float((np.maximum(archive['tau_applied'][:cutoff].astype(float)
                *archive['qdot_post'][:cutoff].astype(float), 0.).sum(axis=-1)
                *.005*archive['valid_mask'][:cutoff]).sum())
        old = load(Path(protocol['old_exploration_root'])/level/'full/result.json')
        reports.append({'level':level, 'rejected':False, **audited, **metrics,
            'selected_k':repeated['selected_k'], 'saved_event_observation_replay':True,
            'prefix_45_or_terminal':{'t_s':rows[cutoff-1]['t_s'],
                'goal_distance_m':rows[cutoff-1]['goal_distance_m'],
                'actual_work_J':work45, 'z_J':rows[cutoff-1]['z_J'],
                'safety_y':rows[cutoff-1]['safety_y']},
            'old_45s_case':{k:old[k] for k in ('label','t_s','z0_J',
                'measured_joint_work_J','safe_timely_measured_budget_success')},
            'old_45s_goal_distance_m':old['terminal']['goal_distance_m']})
        print('SAVED_TIMEOUT_CASE_REVIEWED', level, result['label'], flush=True)
    ledger = load(terminal)
    assert steps <= 60000 and queries <= 15 and ledger['gpu_wall_s'] <= 10800
    accepted = [r for r in reports if not r['rejected']]
    sources = {str(p):sha(p) for p in OUTPUT.rglob('*') if p.is_file()}
    persist(REPORT, {'at':datetime.now().astimezone().isoformat(),
        'status':'SAVED_TIMEOUT_DIAGNOSTIC_REVIEW_NOT_G5', 'allocation_terminal':terminal.name,
        'protocol_sha256':sha(protocol_path), 'protected_hashes_checked':len(protected),
        'episodes':reports, 'incomplete':incomplete, 'verified_explicit_calls':steps,
        'verified_initial_W_queries':queries, 'allocation_ledger':ledger,
        'summary':{'planned_decisions':3, 'completed_decisions':len(reports),
            'rejected':sum(r['rejected'] for r in reports), 'physical_episodes':len(accepted),
            'safe_arrivals':sum(r['timely_goal'] and r['safe'] for r in accepted),
            'safe_actual_budget_arrivals':sum(r['safe_timely_measured_budget_success'] for r in accepted),
            'arrivals_after_45s':sum(r['timely_goal'] and not r['old_45s_timely_goal'] for r in accepted)},
        'limitations':['exposed integration cases, once each; no generalization estimate',
            'old 45s actor observation retained and clamped after 45s',
            'old value predictions and margins are not validated for a 90s objective',
            'no training, calibration or formal G5 acceptance'],
        'raw_sources':sources, 'new_applications':0, 'new_model_inference':0,
        'NAV_043_pass':False, 'G5_pass':False})
    print('SAVED_TIMEOUT_REVIEW_COMPLETE', str(REPORT), flush=True)


if __name__ == '__main__':
    main()
