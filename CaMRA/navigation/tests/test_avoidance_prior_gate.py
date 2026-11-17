"""Finite actor prior gate. Synthetic cases and36 on-policy mechanical episodes.

No development performance rollout, no oracle demonstrations in PPO, no
formal allocation, no checkpoint or heuristic selection from these fixtures.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
from unittest import mock

import numpy as np
import torch

NAV = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(NAV/'scripts'), str(NAV/'tests'), str(NAV/'src'), str(NAV.parent/'calibration/src')]
import train_formal as F
import eval_r14_repair as E
import audit_r14a2 as A
import run_feedback_round as R
import check_detour_exploration as EX
import test_episode_feedback_gate as EP
from camra_navigation import observation as O
from camra_navigation.algorithm import networks as NW
from camra_navigation.algorithm.policy_prior import PRIOR_CONFIG
from camra_navigation.algorithm.validation import file_sha, verify_training_bindings
from camra_navigation.algorithm.storage import EVENT_NAMES

CHECKS = []


def check(name, ok, detail=None):
    CHECKS.append(dict(check=name, ok=bool(ok), detail=detail))
    print(('PASS ' if ok else 'FAIL ') + name, flush=True)


def build(p, nav):
    return E.build_registered_network(nav, p)


def synthetic(nav, p):
    snapshot=Path('/pub/data/caohy/work/camra_navigation/nav064_avoidance_prior_repair_20261117_v1/source_before/src/camra_navigation/algorithm/networks.py')
    spec=importlib.util.spec_from_file_location('camra_navigation.algorithm._prior_parent',snapshot)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    ini=p['action_init_fix'];kw=dict(action_bounded=True,initial_forward_speed=ini['initial_forward_speed'],log_std_min=ini['log_std_min'],log_std_init=ini['log_std_init'],steering_gain=p['steering_prior']['gain'])
    torch.manual_seed(6408);before=old.build(nav,**kw)
    torch.manual_seed(6408);after=NW.build(nav,**kw)
    x=torch.randn(32,61);a=before.policy(x).sample()
    check('prior_default_bitwise',list(before.state_dict())==list(after.state_dict())
          and all(torch.equal(v,after.state_dict()[k]) for k,v in before.state_dict().items())
          and torch.equal(before.policy(x).log_prob(a),after.policy(x).log_prob(a))
          and all(torch.equal(v,w) for v,w in zip(before.values(x),after.values(x))),
          dict(snapshot=str(snapshot),sha256=file_sha(snapshot)))
    torch.manual_seed(6408);net=build(p,nav)
    with torch.no_grad():net.policy.head.weight.zero_()
    sl=O.field_slices('obs-v7')
    def obs(obstacles=(),radius=3.119367799506481,px=0.,py=0.,yaw=0.,goal=(10.,0.)):
        return torch.tensor([O.encode_observation(O.EgoState(px,py,yaw,0.,0.,0.),obstacles,
            O.MapExtent(-15.,15.,-15.,15.),O.TaskState(*goal,0.,420.,45.),version='obs-v7',inflation_radius_m=radius)],dtype=torch.float32)
    cases={
        'clear':obs(),
        'obstacle_left':obs([O.Obstacle(4.,.8,.2)]),
        'obstacle_right':obs([O.Obstacle(4.,-.8,.2)]),
        'behind':obs([O.Obstacle(-4.,0.,.2)]),
        'beyond_goal':obs([O.Obstacle(4.,0.,.2)],goal=(1.,0.)),
        'right_wall':obs(px=11.7),
        'right_wall_rotated':obs(px=11.7,yaw=math.pi/2),
    }
    offsets={k:net.policy.avoidance_prior(v).detach().numpy()[0].tolist() for k,v in cases.items()}
    ok=(np.array_equal(offsets['clear'],[0.,0.,0.]) and np.array_equal(offsets['behind'],[0.,0.,0.])
        and np.array_equal(offsets['beyond_goal'],[0.,0.,0.])
        and offsets['obstacle_left'][1]<0 and offsets['obstacle_left'][2]<0
        and offsets['obstacle_right'][1]>0 and offsets['obstacle_right'][2]>0
        and offsets['right_wall'][0]<0 and offsets['right_wall_rotated'][1]>0)
    check('prior_direction_and_walls',ok,offsets)
    # Actual per-method radius is present in signed margins; no hidden map or
    # method lookup is permitted in this observation-only field.
    radii={}
    for variant in p['method']['variants']:
        env=F.build_envs(p,variant)[0][0]
        v=obs([O.Obstacle(4.,.8,.2)],radius=env.geometry.inflation_radius_m)
        radii[variant]=dict(radius=env.geometry.inflation_radius_m,offset=net.policy.avoidance_prior(v).detach().numpy()[0].tolist())
    check('prior_actual_method_clearance',np.linalg.norm(radii['full']['offset'])>0 and np.array_equal(radii['uncalibrated']['offset'],[0.,0.,0.]),radii)
    xx=torch.cat(list(cases.values()))
    # Changing task history/budget/time does not inject routing state. Current
    # geometry, yaw and goal are the only inputs used by the prior itself.
    changed=xx.clone();changed[:,sl['z_ratio']]=-1.;changed[:,sl['safety_y']]=1.;changed[:,sl['time_remaining_n']]=.01
    altered=net.policy.avoidance_prior(changed)
    check('prior_observation_only',torch.equal(altered,net.policy.avoidance_prior(xx))
          and not list(net.policy.avoidance_prior.parameters()) and not net.policy.avoidance_prior.state_dict(),
          dict(fields=list(net.policy.avoidance_prior.slices),hidden_state=False,oracle_route=False))
    dist=net.policy(xx);action=dist.sample();u=dist._atanh(action)
    manual=torch.distributions.Normal(dist.base.base_dist.loc,dist.stddev).log_prob(u).sum(-1)-torch.log1p(-action.square()+dist.eps).sum(-1)
    finite=torch.isfinite(dist.log_prob(action)).all() and torch.allclose(manual,dist.log_prob(action),atol=2e-5,rtol=1e-5)
    net.zero_grad();loss=(dist.log_prob(action)*torch.linspace(-1.,1.,len(xx))).mean();loss.backward()
    finite=bool(finite and net.policy.head.bias.grad.isfinite().all() and net.policy.head.bias.grad.abs().max()>0.)
    branch=copy.deepcopy(net);one=cases['obstacle_left']
    with torch.no_grad():
        branch.policy.head.bias.sub_(branch.policy.avoidance_prior(one)[0])
    parent=copy.deepcopy(branch);parent.policy.avoidance_prior=None
    parent.policy.head.bias.data.copy_(net.policy.head.bias.data)
    cancelled=torch.allclose(branch.policy(one).mean,parent.policy(one).mean,atol=1e-7,rtol=1e-6)
    check('prior_probability_and_residual',finite and cancelled,
          dict(max_log_prob_error=float((manual-dist.log_prob(action)).abs().max()),residual_cancels=cancelled,finite_gradient=finite))
    rebuilt=build(p,nav);rebuilt.load_state_dict(net.state_dict());rebuilt.eval()
    check('prior_actual_development_build',torch.equal(rebuilt.policy(xx).mean,net.policy(xx).mean)
          and torch.equal(rebuilt.policy(xx).log_prob(action),net.policy(xx).log_prob(action)),
          'Actual factory called by development main; no development trajectories evaluated.')
    without=copy.deepcopy(rebuilt);without.policy.avoidance_prior=None
    replay_mismatch=not torch.allclose(without.policy(xx).log_prob(action),dist.log_prob(action),atol=2e-5,rtol=1e-5)
    check('prior_omitted_density_detected',replay_mismatch)
    return dict(radii=radii,offsets=offsets)


def startup(p,report,out):
    probe=out.with_suffix('.startup.json');probe.write_text(json.dumps(report,indent=2))
    fp=copy.deepcopy(p);fp['authorization']['training_approved']=True
    for name,e in fp['binding'].items():
        if name=='numerical_gate':e['path']=str(probe)
        e['sha256']=file_sha(NAV/e['path'])
    verify_training_bindings(fp,NAV);check('prior_startup_accepted_in_memory',True)
    for case in ('unapproved','omitted_prior','wrong_prior','old_namespace','missing_production_check','wrong_gate_prior','missing_exploration_gate'):
        bad=copy.deepcopy(fp)
        if case=='unapproved':bad['authorization']['training_approved']=False
        if case=='omitted_prior':bad.pop('avoidance_prior')
        if case=='wrong_prior':bad['avoidance_prior']['lookahead_m']=1.
        if case=='old_namespace':bad['runtime_contract']['run_prefix']='nav064_episode_feedback_v1'
        if case=='missing_exploration_gate':bad.pop('exploration_gate')
        if case in ('missing_production_check','wrong_gate_prior'):
            g=copy.deepcopy(report)
            if case=='missing_production_check':g['checks']=[r for r in g['checks'] if r['check']!='prior_actual_producer']
            else:g['avoidance_prior']['lookahead_m']=1.
            path=out.with_suffix('.'+case+'.json');path.write_text(json.dumps(g));bad['binding']['numerical_gate']={'path':str(path),'sha256':file_sha(path)}
        try:verify_training_bindings(bad,NAV);rejected=False
        except ValueError:rejected=True
        check('prior_startup_reject_'+case,rejected)


def exploration_checks(p, root):
    maps=json.loads((NAV/p['binding']['split']['path']).read_bytes())['splits']['train']['maps']
    selected=[(i,m['map_id']) for i,m in enumerate(maps) if m['map_id'].startswith('l2_')]
    goal=next(k for k,v in EVENT_NAMES.items() if v=='goal_reached')
    fixture=copy.deepcopy(p);fixture['exploration_gate']['after_updates']=2
    outcomes={}
    for case in ('pass','unsafe','incomplete','tampered'):
        work=root/('prefix_fixture_'+case);work.mkdir()
        for row in fixture['run_registry']:
            if row['variant']!='full':continue
            for i,mid in selected:
                store=work/row['run_id']/f'rollout_env{i}'/'rollout';store.mkdir(parents=True)
                entries=[]
                for u in range(2):
                    if case=='incomplete' and row['seed']==1 and u==1:continue
                    path=store/f'{u}.npz'
                    y=1. if case=='unsafe' and row['seed']==1 else 0.
                    np.savez(path,event_code=np.array([goal]),safety_y=np.array([0.]),safety_y_next=np.array([y]),z_norm_next=np.array([.1]))
                    entries.append(dict(update_idx=u,file=path.name,file_sha256=('0'*64 if case=='tampered' and row['seed']==1 else file_sha(path)),map_id=mid))
                (store/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in entries))
        written=[]
        try:
            R.enforce_exploration(fixture,work,2,lambda name,value:written.append((name,value)))
            result='PASS'
        except RuntimeError:result='HELD'
        except ValueError:result='REJECTED'
        outcomes[case]=dict(result=result,reports=len(written))
    check('prior_exploration_prefix_gate',outcomes['pass']['result']=='PASS'
          and outcomes['unsafe']['result']=='HELD' and outcomes['unsafe']['reports']==1
          and all(outcomes[c]['result']=='REJECTED' for c in ('incomplete','tampered'))
          and p['exploration_gate']==EX.EXPLORATION_GATE,
          dict(cases=outcomes,limit='Synthetic raw fixtures; no new formal rollout. Real coordinator enforces this after50-update audit and before resuming.'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--work',required=True);ap.add_argument('--base',required=True);args=ap.parse_args()
    out=Path(args.out).resolve();root=Path(args.work).resolve();assert not out.exists() and not root.exists();root.mkdir()
    torch.set_num_threads(1)
    basepath=Path(args.base).resolve();base=json.loads(basepath.read_bytes())
    assert base['verdict']=='PASS' and all(file_sha(NAV/f)==h for f,h in base['code_files'].items())
    CHECKS.extend(base['checks'])
    p=json.loads((NAV/'config/training_loop_avoidance_prior_proposal_v1.json').read_bytes())
    assert p['avoidance_prior']==PRIOR_CONFIG
    nav=json.loads((NAV/p['binding']['navigation_contract']['path']).read_bytes())
    synthetic_result=synthetic(nav,p)
    EP.CHECKS.clear();production=EP.production(p,root)
    CHECKS.extend(EP.CHECKS)
    check('prior_actual_producer',all(c['ok'] for c in EP.CHECKS),production)
    check('prior_actual_auditor',production['audit']['checks_pass'])
    # Verify corrupt runtime metadata is caught by the real raw auditor.
    real_lines=A.lines
    def corrupt(path):
        rows=copy.deepcopy(real_lines(path))
        if Path(path).name=='train_log.jsonl':rows[0]['avoidance_prior']=None
        return rows
    dest=root/'reject_prior';dest.mkdir()
    with mock.patch.object(A,'lines',corrupt):
        try:A.run_audit(Path(production['run']).name,'full',0,dict(p,window_semantics=dict(p['window_semantics'],max_updates=2)),'mechanical-fixture-not-formal-registration',2,dest);rejected=False
        except AssertionError:rejected=True
    check('prior_corrupt_replay_rejected',rejected)
    exploration_checks(p,root)
    files=set(base['code_files'])|{'src/camra_navigation/algorithm/policy_prior.py','tests/test_avoidance_prior_gate.py','scripts/eval_training_round.py','scripts/check_detour_exploration.py'}
    report=dict(base,schema='camra-avoidance-prior-gate-v1',checks=CHECKS,
        avoidance_prior=p['avoidance_prior'],verdict='PASS' if all(c['ok'] for c in CHECKS) else 'FAIL',
        code_files={f:file_sha(NAV/f) for f in sorted(files)},
        protocol={'path':'config/avoidance_prior_numerical_protocol_v1.json','sha256':file_sha(NAV/'config/avoidance_prior_numerical_protocol_v1.json')},
        component_reports={str(basepath):file_sha(basepath)},synthetic=synthetic_result,production_fixture=production,
        limits=['Synthetic cases and36 train-map on-policy mechanical episodes; no performance claim or formal training allocation.','Reactive tangent prior may fail; NAV-064=false until separately approved training passes all original criteria.'])
    startup(p,report,out);report['checks']=CHECKS;report['verdict']='PASS' if all(c['ok'] for c in CHECKS) else 'FAIL'
    with out.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print(report['verdict'],len(CHECKS),'checks',flush=True);return int(report['verdict']!='PASS')


if __name__=='__main__':raise SystemExit(main())
