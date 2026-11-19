"""Main-agent review of actual saved evidence. Never starts a simulator."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
NAV=ROOT/'CaMRA/navigation'
sys.path[:0]=[str(NAV/'scripts'),str(NAV/'src'),str(ROOT/'CaMRA/calibration/src')]
from nav067_sources import load, sha, write_new, E
import run_nav067_remaining_validation as V1
import run_nav067_remaining_recovery_v2 as V2
from nav067_remaining_audit import audit_episode, compare
from camra_navigation.envs.controlled_cases_v1 import fixture_spec, review_actual_case
from camra_navigation.envs.physical_v2 import SubstepNavigation, PhysicalState
from camra_navigation.evaluation.recorder import offline_recompute

def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()

def check_runtime(runtime):
    assert runtime['config']['profile_id']=='clean'
    assert runtime['config']['obs_corruption_enabled'] is False
    assert runtime['config']['command_resampling_disabled'] is True
    assert runtime['config']['automatic_reset_disabled'] is True
    assert runtime['event_modes']==['reset']
    assert runtime['event_terms']=={'reset':['reset_base','reset_robot_joints']}
    assert runtime['physics_dt_s']==.005 and runtime['decimation']==4
    assert runtime['friction']['ok'] and runtime['mass']['ok']
    for ground in runtime['ground']:
        assert ground['values']==[1.,1.,0.]
    for collider in runtime['colliders']:
        assert collider['enabled']
        assert max(collider['errors'].values())<=1.e-4

def check_repetition_runtime(runtime):
    # Repetitions use the unmodified production runtime schema, whereas the
    # isolated controlled fixture has its own richer collider/ground records.
    config=runtime['config']
    assert config['profile_id']=='clean'
    assert config['obs_corruption_enabled'] is False and config['obs_noise'] is None
    assert config['command_resampling_disabled'] is True
    assert config['automatic_reset_disabled'] is True
    assert runtime['event_modes']==['reset']
    assert runtime['event_terms']=={'reset':['reset_base','reset_robot_joints']}
    assert runtime['friction']['ok'] and runtime['mass']['ok']
    assert runtime['physics_dt_s']==.005 and runtime['decimation']==4
    for bounds in config['reset_base_pose_range'].values():
        assert bounds==[0.,0.]
    for bounds in config['reset_base_velocity_range'].values():
        assert bounds==[0.,0.]
    assert config['reset_joint_scale_range']==[1.,1.]
    assert config['reset_joint_velocity_range']==[0.,0.]
    for collider in runtime['colliders']:
        assert collider['enabled_collision_prims']
        assert collider['position_error_m']<=1.e-4
    # Match the prior main-agent readback review rather than fabricating the
    # ground material schema from the controlled-only fixture.
    assert runtime['ground_material_usd_readback']
    for ground in runtime['ground_material_usd_readback']:
        assert ground['static_dynamic_restitution']==[1.,1.,0.]

def controlled_review():
    proposal,execution,manifest,prereg,nets=V2.context(initial=False)
    ctx=E.build_rollout_ctx('full')
    summaries=[]
    for cid in V1.SLOTS[:6]:
        out=V2.output_for(cid)
        assert not (out/'ERROR.json').exists()
        assert load(out/'EXIT.json')['exit_code']==0
        finished=load(out/'FINISHED.json')
        fixture=execution['fixtures'][cid]
        runtime=load(out/'runtime.json')
        check_runtime(runtime)
        window=load(out/'window.json')
        assert window['warmup']['physics_ticks']==200 and window['warmup']['policy_calls']==50
        assert window['reset']['previous_action_max']==0.
        assert window['environment_ticks_with_warmup']<=V1.slot_bounds(execution,cid)
        assert window['total_physics_ticks_with_warmup']<=400
        raw=load(out/'raw_actual.json')
        commands=load(out/'commands.json')
        initial=load(out/'pose_writes.json')[0]['immediate_actual_readback']
        monitor=SubstepNavigation(fixture_spec(fixture),ctx['geom'],ctx['cm'],[PhysicalState(**s) for s in initial],[1000.]*fixture['num_envs'],prereg)
        monitor.set_commands(fixture['commands'])
        for row in raw:
            event=monitor.advance([PhysicalState(**s) for s in row['actual_states']],physics_tick=row['physics_tick'])
            assert event['active_before']==row['active_before'] and event['active_after']==row['active_after']
            assert monitor.times==row['monitor_times_s'] and monitor.y==row['monitor_safety_y']
            assert [b.spent_J for b in monitor.budgets]==row['monitor_spent_J']
            assert monitor.labels==row['monitor_labels']
        assert monitor.rows==load(out/'substeps.json')
        energy=dict(window['recorder_metadata'])
        with np.load(out/'energy.npz',allow_pickle=False) as archive:
            energy.update({key:archive[key] for key in archive.files})
        review=review_actual_case(fixture,ctx['geom'],SimpleNamespace(monitor=monitor,raw=raw),energy,commands)
        for key,value in review.items():
            assert finished[key]==value,(cid,key)
        audits=[offline_recompute(energy,i) for i in range(fixture['num_envs'])]
        assert all(a['steps_match'] and a['rel_diff']<=1.e-10 for a in audits)
        origin=load(out/'origin_readback.json')
        if fixture['num_envs']==2:
            assert np.linalg.norm(np.array(origin['actual_origins'][0][:2])-origin['actual_origins'][1][:2])>=origin['required_spacing_m']-1.e-4
        forces=[row['contact']['robot_obstacle_pair_force_norm_N'] for row in raw if row['contact'] is not None]
        summaries.append({'case':cid,'status':'PASS_ACTUAL_READBACK_AND_OFFLINE_REPLAY','environment_ticks':window['environment_ticks_with_warmup'],
            'measurement_ticks':len(raw),'labels':monitor.labels,'env_active_steps':energy['env_active_steps'],
            'native_filtered_contact_max_N':max(forces) if forces else None,
            'evidence_kind':finished['evidence_kind'],'performance_evidence':False,'raw_sha256':sha(out/'raw_actual.json'),
            'finished_sha256':sha(out/'FINISHED.json'),'offline_energy_relative_error_max':max(a['rel_diff'] for a in audits)})
    return summaries

def full_review(controlled):
    root=V1.EXPECTED_ROOT
    final=load(root/'RECOVERY_V2_FINISHED.json')
    assert not (root/'RECOVERY_V2_HELD.json').exists()
    assert final['completed_slots']==list(V1.SLOTS)
    assert final['kit_starts_including_failure']==10
    assert final['reserved_environment_ticks_including_failure']==57830
    assert final['cumulative_gpu_wall_s']<=14400
    _,_,manifest,_,_=V2.context(initial=False)
    source=load(NAV/'config/nav067_remaining_validation_proposal_v1.json')
    old=Path(source['inputs']['manifest']['path']).parent
    repeats=[]
    runtimes=[]
    for row in manifest['maps']:
        level=row['level']
        out=root/level
        assert not (out/'ERROR.json').exists() and load(out/'EXIT.json')['exit_code']==0
        assert (out/'FINISHED.json').is_file()
        runtime=load(out/'runtime.json')
        check_repetition_runtime(runtime)
        runtimes.append({'level':level,'sha256':sha(out/'runtime.json'),'runtime_keys':list(runtime)})
        baseline=old/next(s for s in source['repetitions']['slots'] if s['level']==level)['baseline_relative_directory']
        for method in ('uncalibrated','full'):
            comparison=compare(baseline/method,out/method)
            repeats.append({'level':level,'method':method,**comparison})
        assert all(p['enabled'] and max(p['errors'].values())<=1.e-4 for p in load(out/'collider_geometry.json'))
    artifacts=[]
    for slot in V1.SLOTS:
        for path in sorted(V2.output_for(slot).rglob('*')):
            if path.is_file():
                artifacts.append({'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)})
    report={'schema':'camra-nav067-remaining-recovery-review-v2','at':now(),'task':'NAV-067',
        'status':'ACTUAL_CONTROLLED_AND_REPEAT_TRACE_REVIEW_PASS_PENDING_TASK_CLOSURE',
        'controlled':controlled,'paired_repetitions':repeats,'repetition_runtime_sources':runtimes,
        'resources':final,'saved_artifacts':artifacts,'review_source_sha256':sha(Path(__file__)),
        'original_failed_attempt_retained':True,'original_failure_sha256':sha(root/'HELD.json'),
        'approval_sha256':sha(V2.APPROVAL),'proposal_sha256':sha(V2.PROPOSAL),
        'principle_and_frozen_identity_context':'V2.context original source/evidence/295 protected identities pass',
        'new_training':0,'W_queries':0,'final_data_access':False,'NAV_043_executed':False,
        'performance_claim':'Common-budget wiring pilots and descriptive repetition only; no formal physical budget-selected performance or unknown-scene guarantee.',
        'historical_scientific_FAIL_preserved':True}
    write_new(HERE/'actual-full-review.json',report)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--full',action='store_true')
    args=parser.parse_args()
    assert os.environ.get('TMUX_PANE')=='%599'
    controlled=controlled_review()
    if args.full:
        report=full_review(controlled)
        print(json.dumps({'status':report['status'],'controlled':len(controlled),'repetition_pairs':len(report['paired_repetitions']),'resources':report['resources']},ensure_ascii=False),flush=True)
    else:
        write_new(HERE/'actual-controlled-review.json',{'at':now(),'controlled':controlled,'status':'PASS_6_ACTUAL_CASES_OFFLINE_REPLAY_NOT_NAVIGATION_PERFORMANCE'})
        print(json.dumps({'controlled':controlled,'status':'PASS_6_ACTUAL_CASES_OFFLINE_REPLAY'},ensure_ascii=False),flush=True)
