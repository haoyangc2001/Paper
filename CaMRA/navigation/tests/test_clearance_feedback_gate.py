"""NAV-064 signed-clearance gate; controlled CPU checks, no performance trial."""
from __future__ import annotations

import argparse
import copy
import dataclasses as dc
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from unittest import mock

import numpy as np
import torch

NAV = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(NAV/'scripts'), str(NAV/'tests'), str(NAV/'src'),
               str(NAV.parent/'calibration/src')]
import train_formal as F
import eval_r14_repair as E
import run_feedback_round as R
import audit_r14a2 as A
import test_feedback_gate as B
from camra_calibration.geometry import Polygon
from camra_navigation import observation as O
from camra_navigation.algorithm import networks as NW, ppo as P, storage as S
from camra_navigation.algorithm.validation import file_sha, verify_training_bindings
from camra_navigation.envs.nominal import NominalNavEnv

CHECKS = []
VERSION = 'obs-v7'


def check(name, ok, detail=None):
    CHECKS.append(dict(check=name, ok=bool(ok), detail=detail))
    print(('PASS ' if ok else 'FAIL ') + name, flush=True)


def enc(ego, obstacles=(), radius=3., rem=45., y=.27, version=VERSION):
    return np.asarray(O.encode_observation(ego, obstacles, O.MapExtent(-15.,15.,-15.,15.),
        O.TaskState(8.,0.,y,420.,rem), version=version, inflation_radius_m=radius))


def geometry_checks():
    sl = O.field_slices(VERSION)
    poly = Polygon(np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]]))
    ob = O.obstacle_from_polygon(poly.vertices)
    good = True; cases = 0
    for radius in (0., .5055013879783843, 3.119367799506481):
        for px,py in ((3.,0.), (0.,0.), (1.,0.), (1.,1.), (3.,3.), (-5.,0.)):
            for yaw in (0., .73, -math.pi, math.pi/2):
                ego=O.EgoState(px,py,yaw,.1,-.03,.2)
                for shape,signed in ((ob,float(poly.distance([px,py])[0])),
                                     (O.Obstacle(0.,0.,1.),math.hypot(px,py)-1.)):
                    x=enc(ego,[shape],radius)
                    expected=np.clip((signed-radius)/6.,-1.,1.)
                    good &= np.isclose(x[sl['safety_margin_bins']].min(),expected,atol=1e-12)
                    good &= np.isclose(O.signed_surface_distance(ego,shape),signed,atol=1e-12)
                    cases+=1
    check('clearance_independent_signed_geometry', good, {'cases':cases})
    empty=enc(O.EgoState(0.,0.,0.,0.,0.,0.))
    check('clearance_empty_bin_positive_sentinel', np.array_equal(empty[sl['safety_margin_bins']],np.ones(16)))
    far=enc(O.EgoState(0.,0.,0.,0.,0.,0.),[O.Obstacle(7.2,0.,1.)])
    check('clearance_preserves_raw_sensing_range', np.array_equal(far[sl['safety_margin_bins']],np.ones(16)))
    e=O.EgoState(0.,0.,.41,0.,0.,0.)
    shapes=[ob,O.Obstacle(4.,2.,.2)]
    check('clearance_obstacle_permutation',np.array_equal(enc(e,shapes),enc(e,shapes[::-1])))
    good=True
    for radius in (.5055013879783843,3.119367799506481):
        for wall in range(4):
            for margin in (-.05,0.,.05):
                pos=[0.,0.]; axis=wall//2
                pos[axis]=(-15.+radius+margin if wall%2==0 else 15.-radius-margin)
                v=enc(O.EgoState(*pos,0.,0.,0.,0.),radius=radius)
                good &= np.isclose(v[sl['wall_margin_n']][wall],margin/6.,atol=1e-12)
    check('clearance_four_wall_boundary_signs',good)
    for r in (None,float('nan'),float('inf'),-1.):
        try: enc(e,radius=r); rejected=False
        except ValueError: rejected=True
        check('clearance_reject_invalid_radius_'+str(r),rejected)
    a=enc(e,[O.Obstacle(0.,0.,5.)],radius=4.)
    check('clearance_deep_interior_clips_negative',a[sl['safety_margin_bins']].min()==-1.)
    one=enc(O.EgoState(0.,0.,.17,0.,0.,0.),[O.Obstacle(4.,0.,1.)])
    turn=enc(O.EgoState(0.,0.,.17+math.pi/2,0.,0.,0.),[O.Obstacle(4.,0.,1.)])
    check('clearance_body_bearing_rotation',np.array_equal(np.roll(one[sl['safety_margin_bins']],-4),turn[sl['safety_margin_bins']] ))
    task=O.TaskState(8.,0.,.27,420.,45.)
    physical=O.encode_from_physical(base_pos_xy=[e.px,e.py],base_quat_wxyz=[math.cos(e.yaw/2),0.,0.,math.sin(e.yaw/2)],
        lin_vel_body=[0.,0.,0.],ang_vel_body=[0.,0.,0.],obstacles=shapes,
        extent=O.MapExtent(-15.,15.,-15.,15.),task=task,version=VERSION,inflation_radius_m=3.)
    check('clearance_physical_encoder_identity',np.array_equal(physical,enc(e,shapes)))
    check('clearance_time_and_history_retained',empty[-1]==1. and empty[-3]==.27 and sl['safety_y'].start==58)


def legacy_checks():
    root=Path('/pub/data/caohy/work/camra_navigation/nav064_clearance_feedback_repair_20261116_v1')
    snap=root/'source_before/CaMRA/navigation/src/camra_navigation/observation.py'
    manifest=json.loads((root/'preflight.json').read_bytes())
    assert file_sha(snap)==manifest['files']['CaMRA/navigation/src/camra_navigation/observation.py']
    name='camra_navigation._before_clearance'
    spec=importlib.util.spec_from_file_location(name,snap); old=importlib.util.module_from_spec(spec)
    sys.modules[name]=old; spec.loader.exec_module(old)
    rng=np.random.default_rng(6407); ok=True; projection=True; cases=0
    for _ in range(50):
        ego=O.EgoState(*rng.uniform(-2.,2.,6)); radius=float(rng.uniform(.1,4.))
        shapes=[O.Obstacle(4.,2.,.7),O.PolygonObstacle(((1.,1.),(2.,1.),(2.,3.),(1.,3.)))]
        task=O.TaskState(8.,0.,float(rng.uniform()),float(rng.uniform(-2200,2200)),float(rng.uniform(0,45)))
        for version in ('obs-v2','obs-v3','obs-v4','obs-v5','obs-v6'):
            oldshapes=[old.Obstacle(4.,2.,.7),old.PolygonObstacle(shapes[1].vertices)]
            before=old.encode_observation(old.EgoState(**vars(ego)),oldshapes,old.MapExtent(-15.,15.,-15.,15.),old.TaskState(**vars(task)),version=version)
            after=O.encode_observation(ego,shapes,O.MapExtent(-15.,15.,-15.,15.),task,version=version,inflation_radius_m=radius)
            ok &= np.array_equal(before,after); cases+=1
        v6=O.encode_observation(ego,shapes,O.MapExtent(-15.,15.,-15.,15.),task,version='obs-v6')
        v7=O.encode_observation(ego,shapes,O.MapExtent(-15.,15.,-15.,15.),task,version=VERSION,inflation_radius_m=radius)
        projection &= np.array_equal(v6,v7[:38]+v7[-3:])
    check('clearance_legacy_bitwise_replay',ok,{'cases':cases,'snapshot':str(snap),'sha256':file_sha(snap)})
    check('clearance_original41_named_values_preserved',projection)
    check('clearance_default_unchanged',O.OBS_VERSION=='obs-v5' and O.OBS_DIM==41 and O.observation_dim()==41 and O.observation_dim(VERSION)==61)


def production_checks(p,nav):
    F.NAV_CONTRACT=str(NAV/p['binding']['navigation_contract']['path'])
    F.VALUE_CONTRACT=str(NAV/p['binding']['value_contract']['path'])
    groups={}; ok=True; same=True; audited=0
    for variant in p['method']['variants']:
        envs,_,ids=F.build_envs(p,variant); groups[variant]=envs
        ok &= len(envs)==len(ids)==18
        for e in envs:
            same &= e.observation.shape==(1,61) and e.observation_version==VERSION
            scalar=NominalNavEnv(e.spec,e.geometry,copy.deepcopy(e._cost_models[0]),z0_J=e.z_J(0),t_max_s=45.,
                observation_version=VERSION,safety_severity_enabled=True,stagnation_min_displacement_m=.02)
            same &= np.array_equal(scalar.observation(),e.observation[0])
            A.validate_clearance_observations(e.observation.astype(np.float32),e); audited+=1
            rec=e.step(np.array([[.3,0.,0.]]))
            A.validate_clearance_observations(rec.obs.astype(np.float32),e); audited+=1
    sl=O.field_slices(VERSION); full=groups['full'][0]; uncal=groups['uncalibrated'][0]
    radius_ok=full.geometry.inflation_radius_m>uncal.geometry.inflation_radius_m
    for e in (full,uncal):
        px,py,_=e.pose[0];xmin,xmax,ymin,ymax=e.spec.extent
        raw=np.array([px-xmin,xmax-px,py-ymin,ymax-py])
        radius_ok &= np.allclose(e.observation[0,sl['wall_margin_n']],
                                np.clip((raw-e.geometry.inflation_radius_m)/6.,-1.,1.))
    radius_ok &= not np.array_equal(full.observation[0,sl['wall_margin_n']],uncal.observation[0,sl['wall_margin_n']])
    check('clearance_actual_method_radius',radius_ok, {k:v[0].geometry.inflation_radius_m for k,v in groups.items()})
    captured=[]; real_env=E.E.VectorNominalNavEnv
    def constructor(*a,**kw):
        env=real_env(*a,**kw);captured.append((env.observation_version,env.observation.shape,env.geometry.inflation_radius_m));env._done[0]=True;return env
    maps=json.loads(Path(F.SPLIT).read_bytes())['splits']['development']['maps']
    for variant in p['method']['variants']:
        ctx=E.E.build_rollout_ctx(variant)
        for m in maps:
            spec=E.E.M.build(m['level'],m['seed'],ctx['design'])
            with mock.patch.object(E.E,'VectorNominalNavEnv',constructor): E.rollout(None,spec,ctx,420.,p)
    check('production_train_dev_clearance_v7',ok and same and len(captured)==12
          and all(v==VERSION and shape==(1,61) for v,shape,r in captured),{'train_constructors':36,'development_constructors':12,'development_steps':0})
    check('production_auditor_clearance_v7',audited==72)
    for name in ('wrong_radius','wrong_sign','old_dimension'):
        x=full.observation.copy()
        if name=='wrong_radius': x[0,sl['wall_margin_n']]+=.1
        if name=='wrong_sign': x[0,sl['wall_margin_n']]*=-1.
        if name=='old_dimension': x=x[:,:41]
        try: A.validate_clearance_observations(x,full); rejected=False
        except AssertionError: rejected=True
        check('auditor_reject_clearance_'+name,rejected)
    paths=R.attempt_paths(p)
    check('production_coordinator_clearance_v7',paths['coordinator'].name=='nav064_clearance_feedback_v1'
          and all('nav064_clearance_feedback_v1' in str(v) for v in paths.values())
          and all(row['run_id'] in R.training_command(row,'proposal.json') for row in p['run_registry']))
    check('clearance_outputs_absent',not any(path.exists() for path in paths.values())
          and all(not (R.ROOT/row['run_id']).exists() for row in p['run_registry']))
    torch.manual_seed(6407)
    net=NW.build(nav,action_bounded=True,initial_forward_speed=.2,log_std_init=-1.)
    # Two controlled single steps through the actual nominal safety/event path.
    # The base begins 2.5cm inside the right safety boundary in an empty map.
    branches=[]
    for sign in (1.,-1.):
        e=copy.deepcopy(full);e.reset()
        e._pose[0]=[e.spec.extent[1]-e.geometry.inflation_radius_m-.025,0.,0.]
        e._refresh_obs();before_obs=e.observation.copy()
        distance=float(np.linalg.norm(np.asarray(e.spec.goal)-e.pose[0,:2]))
        rec=e.step(np.array([[sign*.5,0.,0.]]))
        branches.append((before_obs,distance,rec))
    check('clearance_actual_pre_crossing_action',np.array_equal(branches[0][0],branches[1][0])
          and np.isclose(branches[0][0][0,sl['wall_margin_n']][1],.025/6.)
          and branches[0][2].safety_y[0]>0 and branches[1][2].safety_y[0]==0
          and branches[0][2].obs[0,sl['wall_margin_n']][1]<0
          and branches[1][2].obs[0,sl['wall_margin_n']][1]>0)
    x=torch.tensor(branches[0][0],dtype=torch.float32,requires_grad=True)
    responses=[]
    for fn in (lambda z:net.policy(z).base.base_dist.loc.sum(),lambda z:net.critic_cost(z).sum(),lambda z:net.critic_reach(z).sum()):
        grad=torch.autograd.grad(fn(x),x)[0]
        responses.append(bool(torch.isfinite(grad).all() and (grad[:,38:58].abs()>0).all()))
    check('production_three_networks_clearance_v7',net.cfg.obs_dim==61 and all(responses)
          and set(net.layout.map_slices()) >= {'safety_margin_bins','wall_margin_n'})
    # Production minPPO and both value regressions on two actions at the same
    # pre-crossing state. This controlled fixture is not a navigation rollout.
    before,distance,bad_rec=branches[0];_,_,good_rec=branches[1]
    a,kind=B.fixture(event='none',dist=(distance,float(bad_rec.goal_distance_m[0])),
                     y=[0.,float(bad_rec.safety_y[0])],bh=-.9)
    b,_=B.fixture(event='none',dist=(distance,float(good_rec.goal_distance_m[0])),
                  y=[0.,float(good_rec.safety_y[0])],bh=-.9)
    F.obj_cfg=B.CFG
    bad,_=F.close_segment(type('Segment',(),{'arrays':lambda self:copy.deepcopy(a)})(),kind,.0,-.9,env_done=False,label='none')
    good,_=F.close_segment(type('Segment',(),{'arrays':lambda self:copy.deepcopy(b)})(),kind,.0,-.9,env_done=False,label='none')
    adv=torch.tensor([bad['obj_adv_total'][0],good['obj_adv_total'][0]],dtype=torch.float32)
    xx=x.detach().repeat(2,1); actions=torch.tensor([[.5,0.,0.],[-.5,0.,0.]])
    with torch.no_grad(): old=net.policy(xx).log_prob(actions).clone();vc,vh=net.values(xx)
    batch=dict(obs=xx,action=actions,log_prob_old=old.clone(),adv_total=adv,
        value_cost_old=vc.clone(),target_cost=vc+.1,value_reach_old=vh.clone(),target_reach=vh+.1)
    updater=P.MinimizationUpdater(net,dc.replace(P.PPOConfig(),ent_coef=0.,max_grad_norm=2.5))
    before=float(torch.softmax(old,0)[0])
    for _ in range(3): stats=updater.update(batch)
    with torch.no_grad(): after=float(torch.softmax(net.policy(xx).log_prob(actions),0)[0])
    check('crossing_v7_probability_decreases',adv[0]>adv[1] and after<before
          and torch.equal(old,batch['log_prob_old']) and stats['grad_norm_policy']>0,
          {'crossing_advantage':adv[0].item(),'safe_advantage':adv[1].item(),'relative_density_before':before,'relative_density_after':after,
           'limit':'Relative density of two bounded continuous actions, not integrated event probability.'})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--base');args=ap.parse_args()
    out=Path(args.out).resolve();assert not out.exists();torch.set_num_threads(1)
    basepath=Path(args.base).resolve() if args.base else out.with_suffix('.time.json')
    if not args.base: subprocess.run([sys.executable,str(NAV/'tests/test_time_feedback_gate.py'),'--out',str(basepath)],check=True)
    base=json.loads(basepath.read_bytes());assert base['verdict']=='PASS' and all(c['ok'] for c in base['checks'])
    assert all(file_sha(NAV/f)==h for f,h in base['code_files'].items())
    assert file_sha(NAV/base['protocol']['path'])==base['protocol']['sha256']
    CHECKS.extend(base['checks'])
    p=json.loads((NAV/'config/training_loop_clearance_feedback_proposal_v1.json').read_bytes())
    nav=json.loads((NAV/p['binding']['navigation_contract']['path']).read_bytes())
    geometry_checks();legacy_checks();production_checks(p,nav)
    files=set(base['code_files'])|{'tests/test_clearance_feedback_gate.py','config/navigation_contract_clearance_feedback_v1.json'}
    report=dict(base,schema='camra-clearance-feedback-gate-v1',implementation_versions=p['implementation_versions'],checks=CHECKS,
        verdict='PASS' if all(c['ok'] for c in CHECKS) else 'FAIL',code_files={f:file_sha(NAV/f) for f in sorted(files)},
        protocol={'path':'config/clearance_feedback_numerical_protocol_v1.json','sha256':file_sha(NAV/'config/clearance_feedback_numerical_protocol_v1.json')},component_reports={str(basepath):file_sha(basepath)})
    probe=out.with_suffix('.startup.json');probe.write_text(json.dumps(report,indent=2))
    fixture=copy.deepcopy(p);fixture['authorization']['training_approved']=True
    for key,entry in fixture['binding'].items():
        if key=='numerical_gate': entry['path']=str(probe)
        entry['sha256']=file_sha(NAV/entry['path'])
    verify_training_bindings(fixture,NAV)
    check('startup_v7_full_chain_accepted_in_memory',True)
    for name in ('unapproved','old_prefix','missing_check','wrong_radius_contract','wrong_layout'):
        bad=copy.deepcopy(fixture)
        if name=='unapproved': bad['authorization']['training_approved']=False
        if name=='old_prefix': bad['runtime_contract']['run_prefix']='nav064_time_feedback_v1'
        if name=='missing_check':
            g=copy.deepcopy(report);g['checks']=[c for c in CHECKS if c['check']!='production_auditor_clearance_v7']
            path=out.with_suffix('.missing_check.json');path.write_text(json.dumps(g));bad['binding']['numerical_gate']={'path':str(path),'sha256':file_sha(path)}
        if name in ('wrong_radius_contract','wrong_layout'):
            n=copy.deepcopy(nav)
            if name=='wrong_radius_contract': n['observation_spec']['safety_clearance']['radius_source']='global_full_default'
            else: n['observation_spec']['layout'][-4]['dim']=3
            path=out.with_suffix('.'+name+'.json');path.write_text(json.dumps(n));bad['binding']['navigation_contract']={'path':str(path),'sha256':file_sha(path)}
        try: verify_training_bindings(bad,NAV);rejected=False
        except ValueError: rejected=True
        check('startup_v7_reject_'+name,rejected)
    report['checks']=CHECKS;report['verdict']='PASS' if all(c['ok'] for c in CHECKS) else 'FAIL'
    report['limits']=base['limits']+['Feature-only candidate; gamma/lambda/objectives and fixedv2 unchanged. No new performance result.','Physical shared-encoder identity is synthetic; actual physical obs-v7 rollout remains downstream work.']
    with out.open('x') as f: json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print(report['verdict'],len(CHECKS),'checks');return int(report['verdict']!='PASS')


if __name__=='__main__': raise SystemExit(main())
