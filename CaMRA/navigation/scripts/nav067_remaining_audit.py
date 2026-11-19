"""Saved-trace and paired-repeat review. No simulator or model inference."""
import json
from pathlib import Path
import numpy as np


def load(path):
    return json.loads(Path(path).read_bytes())


def audit_episode(trial):
    trial=Path(trial)
    result=load(trial/'result.json')
    steps=load(trial/'substeps.json')
    commands=load(trial/'commands.json')
    navigation=load(trial/'navigation.json')
    with np.load(trial/'energy.npz',allow_pickle=False) as archive:
        energy={k:archive[k] for k in archive.files}
    n=result['physics_steps']
    assert n>0 and len(steps)==n
    assert [r['physics_tick'] for r in steps]==list(range(1,n+1))
    np.testing.assert_allclose([r['t_s'] for r in steps],np.arange(1,n+1)*.005,atol=1.e-8,rtol=0.)
    assert steps[-1]==result['terminal'] and steps[-1]['label']==result['label']
    assert all(s['label'] is None for s in steps[:-1])
    assert result['warmup']['physics_steps']==200 and result['warmup']['policy_calls']==50
    assert result['reset']['previous_action_max']==0.
    assert commands[0]['tick_before']==200 and len(commands)==(n+3)//4
    assert len(navigation)==(n+19)//20 and sum(c['substeps'] for c in commands)==n
    for j,c in enumerate(commands):
        assert 1<=c['substeps']<=4 and c['tick_after']-c['tick_before']==c['substeps']
        assert not j or commands[j-1]['tick_after']==c['tick_before']
        np.testing.assert_allclose(c['command_written'],c['command_seen'],atol=1.e-6,rtol=0.)
        np.testing.assert_allclose(c['command_written'],navigation[j//5]['command'],atol=1.e-7,rtol=0.)
        assert c['max_error']<=1.e-6
        assert abs(c['sim_time_after']-c['sim_time_before']-.005*c['substeps'])<=1.e-7
    valid=energy['valid_mask'].astype(bool)
    assert valid.shape==(n,1) and valid.all()
    actual=float((np.maximum(energy['tau_applied'].astype(float)*energy['qdot_post'].astype(float),0.).sum(axis=-1)*.005*valid).sum())
    assert abs(actual-result['measured_joint_work_J'])<=1.e-9*max(1.,actual)
    assert abs(result['t_s']-n*.005)<1.e-8
    assert abs(result['z0_J']-steps[-1]['z_J']-result['internal_model_cost_J'])<1.e-8
    assert max(s['safety_y'] for s in steps)==result['safety_y']
    assert not result['performance_evidence']
    return {'status':'PASS_SAVED_TRACE_ONLY','trial':str(trial),'method':result['method'],
        'label':result['label'],'safety_y':result['safety_y'],'t_s':result['t_s'],
        'actual_work_J':actual,'internal_model_cost_J':result['internal_model_cost_J'],
        'budget_J':result['z0_J'],'goal_distance_m':steps[-1]['goal_distance_m'],
        'events':result['terminal']['events'],'physics_ticks':n,'performance_evidence':False}


def compare(baseline,repeat):
    a,b=audit_episode(baseline),audit_episode(repeat)
    ar,br=load(Path(baseline)/'result.json'),load(Path(repeat)/'result.json')
    assert ar['candidate']==br['candidate'] and ar['z0_J']==br['z0_J'] and ar['map_id']==br['map_id']
    asteps,bsteps=load(Path(baseline)/'substeps.json'),load(Path(repeat)/'substeps.json')
    n=min(len(asteps),len(bsteps))
    difference=np.array([s['state'][:2] for s in bsteps[:n]])-np.array([s['state'][:2] for s in asteps[:n]])
    distances=np.linalg.norm(difference,axis=1)
    return {'baseline':a,'repeat':b,'runs_per_configuration':2,
        'delta_repeat_minus_baseline':{k:b[k]-a[k] for k in ('t_s','actual_work_J','internal_model_cost_J','goal_distance_m')},
        'common_active_physics_ticks':n,'position_difference_max_m':float(distances.max()),
        'position_difference_RMS_m':float(np.sqrt(np.mean(distances**2))),
        'position_alignment_after_start':False,'general_reliability_claim':False,
        'description':'Descriptive paired differences; all timeouts and over-budget outcomes retained.'}
