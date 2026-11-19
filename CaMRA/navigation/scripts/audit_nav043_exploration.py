"""Read saved exploratory decisions and traces; no model inference or Isaac."""
from datetime import datetime
import os
from pathlib import Path
import socket

import numpy as np

from nav067_sources import E, NAV, ROOT, LEVELS, load, sha
from nav067_runtime import persist
from nav067_remaining_audit import audit_episode
from prepare_nav065_sources import REGISTRATION
from train_formal import verify_prereg
from nav043_budget import outcome
from camra_navigation.budget.accounting import consistent_e_ref
from camra_navigation.budget.guarded import select_guarded_budget
from camra_navigation.envs.physical_v2 import PhysicalState, SubstepNavigation

OUTPUT=Path('/pub/data/caohy/work/camra_navigation/nav043_budget_exploration_v1')


def review_runtime(runtime,row):
    c=runtime['config']
    assert c['profile_id']=='clean' and not c['obs_corruption_enabled'] and c['obs_noise'] is None
    assert set(c['removed_event_terms'])=={'physics_material','add_base_mass','base_external_force_torque','push_robot'}
    assert c['command_resampling_disabled'] and c['automatic_reset_disabled']
    for key in ('reset_base_pose_range','reset_base_velocity_range'):
        assert all(v==[0.,0.] for v in c[key].values())
    assert c['reset_joint_scale_range']==[1.,1.] and c['reset_joint_velocity_range']==[0.,0.]
    assert runtime['physics_dt_s']==.005 and runtime['decimation']==4
    assert runtime['event_modes']==['reset']
    assert runtime['event_terms']=={'reset':['reset_base','reset_robot_joints']}
    assert runtime['friction']['ok'] and runtime['mass']['ok']
    assert runtime['ground_material']=={'static':1.,'dynamic':1.,'restitution':0.}
    assert runtime['ground_material_usd_readback']
    assert all(m['static_dynamic_restitution']==[1.,1.,0.] for m in runtime['ground_material_usd_readback'])
    assert runtime['scene']['map_id']==row['map_id'] and len(runtime['actual_origins'])==1
    assert all(c['position_error_m']<=1e-4 and c['enabled_collision_prims'] for c in runtime['colliders'])


def main():
    assert socket.gethostname().split('.')[0]=='tiger11' and os.environ.get('TMUX')
    terminal=OUTPUT/'FINISHED.json'
    if not terminal.exists():terminal=OUTPUT/'HELD.json'
    if not terminal.exists():raise ValueError('allocation still active or terminal evidence absent')
    protocol_path=NAV/'config/nav043_exploration_v1.json'
    protocol=load(protocol_path)
    assert sha(protocol_path)==Path(str(protocol_path)+'.sha256').read_text().strip()
    assert load(OUTPUT/'OWNER.json')['protocol_sha256']==sha(protocol_path)
    sources={str(p):sha(p) for p in (protocol_path,OUTPUT/'OWNER.json',terminal,Path(__file__))}
    for path,digest in protocol['sealed_hashes'].items():assert sha(path)==digest,path
    pair=load(NAV/'config/nav066_decision_prereg_v2.json')
    protected={str(NAV/k):v for k,v in pair['implementation_hashes'].items()}
    protected.update(pair['protected_hashes'])
    protected.update({c['checkpoint']:c['checkpoint_sha256'] for c in pair['six_candidates']})
    for path,digest in protected.items():assert sha(path)==digest,path
    prereg,_=verify_prereg(REGISTRATION)
    manifest=load(protocol['pilot_manifest'])
    reports=[];missing=[];queries=0;physical=0;nominal=0;steps=0;apps=0;wall=0.
    for level in LEVELS:
        app=OUTPUT/level
        if not app.exists():
            missing.append({'level':level,'status':'not started'})
            continue
        apps+=1
        for path in app.rglob('*'):
            if path.is_file():sources[str(path)]=sha(path)
        if not (app/'FINISHED.json').exists():
            missing.append({'level':level,'status':'application did not finish','ERROR':load(app/'ERROR.json') if (app/'ERROR.json').exists() else None})
            continue
        assert load(app/'EXIT.json')['exit_code']==0 and not (app/'ERROR.json').exists()
        wall=load(app/'EXIT.json')['cumulative_gpu_wall_s']
        calls=load(app/'physical_step_accounting.json')['calls']
        assert calls<=20000
        steps+=calls
        row=next(r for r in manifest['maps'] if r['level']==level)
        review_runtime(load(app/'runtime.json'),row)
        for method in ('uncalibrated','full'):
            trial=app/method
            saved=load(trial/'decision.json')
            assert sha(trial/'decision.json')==load(trial/'decision.pin.json')['sha256']
            queries+=saved['query_count']
            assert saved['query_count']==5 and saved['physical_actions_started'] is False
            ctx=E.build_rollout_ctx(method)
            spec=E.M.build(level,row['seed'],ctx['design'])
            assert spec.to_dict()==row['spec']
            initial=PhysicalState(**saved['initial_state'])
            qiter=iter(saved['decision']['trace'])
            def query(z):
                q=next(qiter)
                assert z==q['budget_J']
                return float('nan') if q['vc_norm'] is None else q['vc_norm'],float('nan') if q['vh'] is None else q['vh']
            repeated=select_guarded_budget(query,consistent_e_ref(row['z0_J'],method))
            assert repeated==saved['decision']
            for q,obs in zip(repeated['trace'],saved['query_observations']):
                mon=SubstepNavigation(spec,ctx['geom'],ctx['cm'],[initial],[q['budget_J']],prereg)
                np.testing.assert_array_equal(np.asarray(mon.observation(0),dtype=np.float32),obs)
            result=load(trial/'result.json')
            if repeated['selected_budget_J'] is None:
                assert result['rejected'] and not (trial/'NAVIGATION_STARTED.json').exists()
                reports.append({'level':level,'method':method,'rejected':True,'reason':repeated['reason']})
                continue
            physical+=1;nominal+=1
            z=repeated['selected_budget_J']
            assert result['z0_J']==z
            assert load(trial/'NAVIGATION_STARTED.json')['decision_sha256']==sha(trial/'decision.json')
            traced=audit_episode(trial)
            rows=load(trial/'substeps.json')
            mon=SubstepNavigation(spec,ctx['geom'],ctx['cm'],[initial],[z],prereg)
            for r in rows:
                mon.set_commands([r['applied']])
                state=r['state']
                mon.advance([PhysicalState(*state[:7],tuple(state[7:]))],physics_tick=r['physics_tick'])
                assert mon.rows[-1]==r,(level,method,r['physics_tick'])
            metrics=outcome(result['label'],result['safety_y'],result['t_s'],
                traced['actual_work_J'],result['internal_model_cost_J'],z)
            assert all(result[k]==v for k,v in metrics.items())
            nom=load(trial/'nominal.json')
            assert nom['initial_state']==saved['initial_state'] and nom['z0_J']==z and len(nom['trace'])<=450
            chosen=repeated['selected_index']
            np.testing.assert_array_equal(np.asarray(nom['trace'][0]['observation'],dtype=np.float32),saved['query_observations'][chosen])
            np.testing.assert_array_equal(load(trial/'navigation.json')[0]['observation'][0],saved['query_observations'][chosen])
            assert abs(nom['cost_model_J']-(z-nom['trace'][-1]['z_J']))<1e-8
            assert nom['feasible']==(nom['label']=='goal_reached' and nom['trace'][-1]['safety_y']==0. and nom['trace'][-1]['z_J']>=0.)
            reports.append({'level':level,'method':method,'rejected':False,'selected_k':repeated['selected_k'],
                **traced,**metrics,'nominal_label':nom['label'],'nominal_feasible':nom['feasible'],
                'nominal_t_s':nom['t_s'],'nominal_cost_J':nom['cost_model_J'],
                'max_same_command_deviation_m':max(r['deviation_m'] for r in rows),
                'complete_saved_physics_event_replay':True})
    assert apps<=3 and queries<=30 and physical<=6 and nominal<=6 and steps<=60000 and wall<=10800
    methods={}
    for method in ('uncalibrated','full'):
        rows=[r for r in reports if r['method']==method];accepted=[r for r in rows if not r['rejected']]
        methods[method]={'decisions':len(rows),'accepted':len(accepted),'rejected':len(rows)-len(accepted),
            'physical_safe_timely_budget_success':sum(r['safe_timely_measured_budget_success'] for r in accepted),
            'internal_safe_budget_success':sum(r['internal_safe_budget_success'] for r in accepted),
            'nominal_feasible':sum(r['nominal_feasible'] for r in accepted)}
    report={'at':datetime.now().astimezone().isoformat(),'status':'SAVED_EVIDENCE_REVIEW_NOT_G5',
        'allocation_terminal':terminal.name,'allocation_resource_ledger':load(terminal),
        'protocol_sha256':sha(protocol_path),
        'protected_hashes_checked':len(protected),'methods':methods,'episodes':reports,'incomplete':missing,
        'applications':apps,'completed_initial_W_queries':queries,'completed_physical_episodes':physical,
        'completed_nominal_episodes':nominal,'observed_explicit_step_calls_finished_apps':steps,
        'gpu_wall_s_finished_apps':wall,'raw_sources':sources,'NAV_043_pass':False,'G5_pass':False,
        'limitations':['already exposed integration maps; one run/configuration',
            'old delta and <=30s vy=0 applicability gap unresolved',
            'no grid trajectory sweep: budget minimum efficiency not established',
            'any incomplete application is retained; completed counts are not inferred totals'],
        'new_applications':0,'new_model_inference':0,'new_rollouts':0,'training_updates':0}
    persist(NAV/'reports/nav043_exploration_review_v1.json',report)
    print({k:v for k,v in report.items() if k not in ('raw_sources','episodes')},flush=True)


if __name__=='__main__':main()
