"""Independent finite-task lambda weights and real producer/auditor gate.

Only train data and finite mechanical fixtures are used. No formal training,
development rollouts, budget selection, or final evaluation is authorized.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses as dc
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

NAV=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(NAV/'src'),str(NAV/'scripts'),str(NAV/'tests'),str(NAV.parent/'calibration/src')]
from camra_navigation.algorithm import storage as S, targets as T, ppo as P
from camra_navigation.algorithm.validation import file_sha, verify_training_bindings
from camra_navigation.algorithm.stability import TERMINAL_TAIL_STABILITY_GATE
import train_formal as F
import run_feedback_round as R
import test_feedback_gate as B
import test_episode_feedback_gate as EP
from test_r14_repair_gate import oracle_min

CHECKS=[]
def check(name,ok,detail=None):
    CHECKS.append(dict(check=name,ok=bool(ok),detail=detail))
    print(('PASS ' if ok else 'FAIL ')+name,flush=True)

def candidate():
    p=json.loads((NAV/'config/training_loop_prereg_avoidance_prior_v2.json').read_bytes())
    p['authorization']['training_approved']=False
    p['implementation_versions']['estimator']=S.TERMINAL_TAIL_ESTIMATOR
    p['terminal_tail_policy']='true_endings_only'
    p['learning_stability_gate']=copy.deepcopy(TERMINAL_TAIL_STABILITY_GATE)
    p['runtime_contract']['run_prefix']='nav064_terminal_tail_v1'
    for row in p['run_registry']:row['run_id']=f'nav064_terminal_tail_v1_{row["variant"]}_s{row["seed"]}'
    for key,path in [('navigation_contract','config/navigation_contract_terminal_tail_v1.json'),('value_contract','config/value_objective_contract_terminal_tail_v1.json')]:
        p['binding'][key]={'path':path,'sha256':file_sha(NAV/path)}
    return p

def cfg_for(p):
    return dc.replace(S.ObjectiveConfig.from_contracts(str(NAV/p['binding']['value_contract']['path']),r_goal_m=p['environment']['r_goal_m']),h_semantics='v2_deep_well',joint_min_recursion=True)

def oracle(h,v,d,gamma,lam,tail,force):
    """Enumerate each length separately, with an explicit geometric remainder."""
    result=[];n=len(d)
    for t in range(n):
        if d[t]:result.append(h[t]);continue
        stops=[j for j in range(t+1,n) if d[j]]
        if stops or not tail:
            # Existing shared internal-stop/window oracle is independently
            # enumerated, never the rolling production implementation.
            result.append(oracle_min(h[t:],v[t:],d[t:],gamma,lam,force)[0]);continue
        values=[];weights=[];length=n-t
        for k in range(1,length+1):
            value=v[t+k]
            for j in range(t+k-1,t-1,-1):
                continuation=(1-gamma)*h[j]+gamma*value
                value=continuation if force else min(h[j],continuation)
            values.append(value)
            weights.append(lam**(k-1) if k==length else (1-lam)*lam**(k-1))
        assert abs(sum(weights)-1)<1e-12
        result.append(float(np.dot(values,weights)))
    return np.array(result)

def numerical():
    rng=np.random.default_rng(6451);cases=0;err=0.
    for n in (1,2,5):
        for stop in range(n+1):
            for lam in (0.,.5,.95,.999):
                h=rng.uniform(-1.,2.,n+1);v=rng.uniform(-2.,2.,n+1)
                d=np.zeros(n,bool)
                if stop<n:d[stop]=True
                for gamma in (.999,1.):
                    for force in (False,True):
                        for tail in (False,True):
                            expected=oracle(h,v,d,gamma,lam,tail,force)
                            actual=T.min_gae_return(h,v,d,gamma,lam,force_continue=force,terminal_tail=tail)
                            err=max(err,float(np.max(abs(actual-expected))));cases+=1
    check('terminal_tail_independent_oracle',err<1e-11,dict(cases=cases,max_error=err))
    lam=.999;n=450;h=np.ones(n+1);v=np.full(n+1,-.3);v[-1]=.05;d=np.zeros(n,bool)
    a=T.min_gae_return(h,v,d,1.,lam,terminal_tail=True)
    b=v.copy();b[-1]-=1e-4
    derivative=(a[0]-T.min_gae_return(h,b,d,1.,lam,terminal_tail=True)[0])/1e-4
    old=T.min_gae_return(h,v,d,1.,lam)
    old_derivative=(old[0]-T.min_gae_return(h,b,d,1.,lam)[0])/1e-4
    check('terminal_tail_450_weight',abs(derivative-lam**449)<1e-10,
          dict(old_direct_weight=old_derivative,new_direct_weight=derivative,expected=lam**449))
    gamma_cases=[]
    for gamma in (1.,.999):
        value=T.min_gae_return(np.array([.2,0.]),np.array([.2,0.]),np.zeros(1,bool),gamma,lam,terminal_tail=True)[0]
        gamma_cases.append(dict(gamma=gamma,zero_boundary_target=float(value)))
    check('terminal_tail_gamma_comparison',gamma_cases[0]['zero_boundary_target']==0. and gamma_cases[1]['zero_boundary_target']>0.,gamma_cases)

def endpoints(p):
    cfg=cfg_for(p);old=dc.replace(cfg,estimator_version='rcppo-feedback-v4')
    outcomes=[];slice_unchanged=True
    F.obj_cfg=cfg
    for event in ('goal_reached','timeout','stagnation','collision','out_of_bounds','none'):
        a,kind=B.fixture(dist=(1.,.7,.01) if event=='goal_reached' else (4.,3.,2.),event=event)
        x,used=F.close_segment(SimpleNamespace(arrays=lambda:copy.deepcopy(a)),'truncation',.7,-.9,env_done=event!='none',label=event if event!='none' else None)
        z,_=F.close_segment(SimpleNamespace(arrays=lambda:copy.deepcopy(a)),'truncation',99.,-99.,env_done=event!='none',label=event if event!='none' else None)
        expected=S.terminal_kind_for_event(event,cfg) if event!='none' else 'truncation'
        o=S.compute_objectives(a,cfg,terminal_kind=expected)
        previous=S.compute_objectives(a,old,terminal_kind=expected)
        keys=('obj_target_cost','obj_target_reach','obj_target_total')
        if event=='none':
            slice_unchanged &= all(np.array_equal(o[k],previous[k]) for k in keys)
            outcomes.append(used==expected and not np.allclose(x[keys[1]],z[keys[1]]))
        else:
            outcomes.append(used==expected and all(np.array_equal(x[k],z[k]) for k in keys))
        outcomes.append(np.array_equal(o['obj_stop_index'],previous['obj_stop_index']) and np.array_equal(o['obj_target_cost'],previous['obj_target_cost']))
    check('terminal_tail_true_endings',all(outcomes),dict(events=['success','deadline','stagnation','collision','out_of_bounds','sampling_slice'],cost_estimator_unchanged=True))
    check('terminal_tail_slice_legacy',slice_unchanged)
    cases=[]
    for name,y,z in [('safe',[0.,0.,0.],1.),('unsafe',[0.,.2,.2],1.),('overspent',[0.,0.,0.],-.1)]:
        a,kind=B.fixture(dist=(1.,.7,.01),y=y,z0=z,event='goal_reached')
        o=S.compute_objectives(a,cfg,terminal_kind=kind)
        cases.append(bool(o['obj_feasible'][0])==(name=='safe'))
    check('terminal_tail_scoring_unchanged',all(cases))
    # Same starting state, two possible future action sequences, same optimistic
    # interior critic: finite-horizon targets must rank arrival before timeout.
    def target(n,terminal):
        h=np.r_[np.full(n,.2),terminal];v=np.r_[np.full(n,-.3),terminal]
        return T.min_gae_return(h,v,np.zeros(n,bool),1.,.999,terminal_tail=True)[0]+.3
    advantages=torch.tensor([target(400,-.1),target(450,.05)],dtype=torch.float64)
    adv,_=P.normalize_advantage(advantages)
    logits=torch.nn.Parameter(torch.zeros(2,dtype=torch.float64));optimizer=torch.optim.SGD([logits],lr=.1)
    baseline=torch.log_softmax(logits.detach(),0).clone()
    for _ in range(3):
        optimizer.zero_grad();loss,_=P.policy_loss(torch.log_softmax(logits,0),baseline,adv,torch.zeros(2,dtype=torch.float64),dc.replace(P.PPOConfig(),ent_coef=0.));loss.backward();optimizer.step()
    probability=torch.softmax(logits,0).detach().tolist()
    check('terminal_tail_production_updater',advantages[0]<advantages[1] and probability[1]<.5,dict(raw_advantages=advantages.tolist(),probability_arrival_timeout=probability))

def replay(p,out):
    cfg=cfg_for(p);old=dc.replace(cfg,estimator_version='rcppo-feedback-v4')
    work=Path('/pub/data/caohy/work/camra_navigation');rows=[];legacy_error=0.
    for seed in range(3):
        run=work/f'nav064_avoidance_prior_v2_full_s{seed}';entries=[]
        for env in range(6,12):
            manifest=[json.loads(line) for line in (run/f'rollout_env{env}/rollout/manifest.jsonl').read_text().splitlines()]
            for meta in manifest:
                if meta['episode_id']<250 or meta['episode_k']<1.15 or meta['terminal_kind']!='task_end':continue
                path=run/f'rollout_env{env}/rollout/chunk_{meta["episode_id"]:05d}.npz'
                with np.load(path) as raw:a={k:raw[k].copy() for k in raw.files}
                if a['event_code'][-1]!=6:continue
                local_old=dc.replace(old,gamma_h=float(meta['gamma_h']))
                local_new=dc.replace(cfg,gamma_h=float(meta['gamma_h']))
                before=S.compute_objectives(a,local_old,terminal_kind='task_end')
                after=S.compute_objectives(a,local_new,terminal_kind='task_end')
                for key in before:
                    legacy_error=max(legacy_error,float(np.max(abs(before[key].astype(float)-a[key].astype(float)))))
                assert np.array_equal(before['obj_stop_index'],after['obj_stop_index'])
                assert np.array_equal(before['obj_target_cost'],after['obj_target_cost'])
                entries.append(dict(path=str(path),sha256=file_sha(path),true=float(before['obj_stop_residual'][0]),old=float(before['obj_target_total'][0]),new=float(after['obj_target_total'][0])))
        rows.append(dict(seed=seed,count=len(entries),negative_old=sum(e['old']<0 for e in entries),negative_new=sum(e['new']<0 for e in entries),mean={key:float(np.mean([e[key] for e in entries])) for key in ('true','old','new')},episodes=entries))
    report=dict(owner='NAV-064',split='train',selection='predeclared last50 full L2 timeout k>=1.15',legacy_max_error=legacy_error,seeds=rows,
                limits='Read-only reweighting with old critics, not on-policy v5 performance. Remaining optimistic targets require new learning verification; a single failed stochastic sample does not establish critic bias.')
    with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    check('terminal_tail_existing_train_replay',legacy_error<1e-6 and sum(r['count'] for r in rows)==157 and all(abs(r['mean']['new']-r['mean']['true'])<abs(r['mean']['old']-r['mean']['true']) for r in rows),dict(report=str(out),sha256=file_sha(out),summary=[{k:v for k,v in r.items() if k!='episodes'} for r in rows]))

def startup(p,report,out):
    probe=out.with_suffix('.startup.json')
    with probe.open('x') as f:json.dump(report,f,indent=2)
    fixture=copy.deepcopy(p);fixture['authorization']['training_approved']=True
    fixture['binding']['numerical_gate']={'path':str(probe),'sha256':file_sha(probe)}
    for entry in fixture['binding'].values():entry['sha256']=file_sha(NAV/entry['path'])
    verify_training_bindings(fixture,NAV)
    rejected=[]
    for case in ('unapproved','missing_policy','old_namespace','missing_tail_check','wrong_value_version'):
        bad=copy.deepcopy(fixture)
        if case=='unapproved':bad['authorization']['training_approved']=False
        if case=='missing_policy':bad.pop('terminal_tail_policy')
        if case=='old_namespace':bad['runtime_contract']['run_prefix']='nav064_avoidance_prior_v2'
        if case=='wrong_value_version':bad['binding']['value_contract']={'path':'config/value_objective_contract_episode_feedback_v1.json','sha256':file_sha(NAV/'config/value_objective_contract_episode_feedback_v1.json')}
        if case=='missing_tail_check':
            gate=copy.deepcopy(report);gate['checks']=[c for c in gate['checks'] if c['check']!='terminal_tail_450_weight']
            path=out.with_suffix('.missing_tail.json')
            with path.open('x') as f:json.dump(gate,f)
            bad['binding']['numerical_gate']={'path':str(path),'sha256':file_sha(path)}
        try:verify_training_bindings(bad,NAV);rejected.append(False)
        except ValueError:rejected.append(True)
    check('terminal_tail_startup_rejections',all(rejected),dict(cases=list(zip(('unapproved','missing_policy','old_namespace','missing_tail_check','wrong_value_version'),rejected))))

def stability(p,root):
    fixture=copy.deepcopy(p)
    fixture['learning_stability_gate'].update(after_updates=4,baseline_interval=[0,2],recent_interval=[2,4],minimum_episodes_per_seed_interval=2)
    maps=json.loads((NAV/p['binding']['split']['path']).read_bytes())['splits']['train']['maps']
    results={}
    for case in ('pass','regressed','tampered'):
        directory=root/('stability_'+case);directory.mkdir()
        for row in fixture['run_registry']:
            if row['variant']!='full':continue
            for env,m in enumerate(maps):
                if not m['map_id'].startswith('l2_'):continue
                store=directory/row['run_id']/f'rollout_env{env}'/'rollout';store.mkdir(parents=True)
                entries=[]
                for u in range(4):
                    failed=case=='regressed' and u>=2 and row['seed']==1
                    path=store/f'{u}.npz'
                    np.savez(path,event_code=np.array([0,6 if failed else 3]),safety_y=np.zeros(2),safety_y_next=np.zeros(2),z_norm_next=np.ones(2),action_applied=np.array([[.4,0,0],[.4,0,0]]))
                    entries.append(dict(update_idx=u,map_id=m['map_id'],file=path.name,file_sha256=file_sha(path),episode_k=1.2))
                with (store/'manifest.jsonl').open('x') as f:
                    for entry in entries:f.write(json.dumps(entry)+'\n')
                if case=='tampered' and row['seed']==0 and env==6:path.write_bytes(b'broken raw fixture')
        written=[]
        try:
            R.enforce_learning_stability(fixture,directory,4,lambda name,value:written.append((name,value)))
            results[case]='PASS'
        except RuntimeError:results[case]='HELD' if written and written[0][1]['verdict']=='FAIL_LEARNING_STABILITY' else 'WRONG'
        except ValueError:results[case]='REJECTED'
    check('terminal_tail_learning_stability_hold',results=={'pass':'PASS','regressed':'HELD','tampered':'REJECTED'},results)

def coordinator(p,root):
    """Run real coordinator control flow with no subprocess or new sampling."""
    outcomes={}
    for regressed in (False,True):
        directory=root/('coordinator_hold' if regressed else 'coordinator_pass');directory.mkdir()
        work=directory/'coordinator';events=[];finished=[False];processes=[]
        paths={key:directory/key for key in R.attempt_paths(p)};paths['coordinator']=work
        class Process:
            def __init__(self,cmd,**kwargs):
                self.pid=100000+len(processes);self.returncode=None;processes.append(self)
                rid=cmd[cmd.index('--run-id')+1];store=directory/rid;store.mkdir()
                (store/'train_log.jsonl').write_text((json.dumps({'kind':'update'})+'\n')*150)
            def poll(self):
                if finished[0]:self.returncode=0;return 0
                return None
        def run(cmd,**kwargs):
            if 'scripts/audit_r14a2.py' in cmd:
                events.append(('audit',int(cmd[cmd.index('--updates')+1])))
            elif 'scripts/eval_r14_repair.py' in cmd:
                events.append(('development',))
                paths['development'].write_text(json.dumps({'performance_pass':True,'summary':{}}))
            else:raise AssertionError(cmd)
        def guard(prereg,base,updates,write):
            events.append(('stability',updates))
            write('learning_stability_u150.json',{'verdict':'FAIL_LEARNING_STABILITY' if regressed else 'PASS'})
            if regressed:raise RuntimeError('fixture learning regression')
        def advance(_):
            for row in p['run_registry']:
                (directory/row['run_id']/'train_log.jsonl').write_text((json.dumps({'kind':'update'})+'\n')*300)
            finished[0]=True
        with (mock.patch.object(F,'verify_prereg',return_value=(p,'mechanical-control-flow')),
             mock.patch.object(sys,'argv',['runner','--prereg','in-memory-control-flow']),
             mock.patch.object(R,'ROOT',directory),mock.patch.object(R,'attempt_paths',return_value=paths),
             mock.patch.object(R.subprocess,'Popen',Process),mock.patch.object(R.subprocess,'run',run),
             mock.patch.object(R.subprocess,'check_output',return_value='fixture-head'),
             mock.patch.object(R.os,'kill',side_effect=lambda pid,sig:events.append(('signal',int(sig)))),
             mock.patch.object(R.time,'sleep',advance),mock.patch.object(R,'enforce_exploration',return_value=None),
             mock.patch.object(R,'enforce_learning_stability',guard)):
            try:code=R.main();held=False
            except RuntimeError:code=None;held=True
        audits=[e[1] for e in events if e[0]=='audit']
        after=events[events.index(('stability',150))+1:]
        if regressed:
            passed=(held and (work/'HELD.json').is_file() and audits==[50,150]
                    and not (work/'FINISHED.json').exists() and not paths['development'].exists()
                    and all(e[0]=='signal' and e[1]==int(R.signal.SIGSTOP) for e in after))
        else:passed=(code==0 and not held and audits==[50,150,300] and (work/'FINISHED.json').is_file())
        outcomes['held' if regressed else 'passed']=dict(ok=passed,audits=audits,events=events)
    check('terminal_tail_actual_coordinator',all(v['ok'] for v in outcomes.values()),outcomes)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--work',required=True);ap.add_argument('--base',required=True);args=ap.parse_args()
    out=Path(args.out).resolve();root=Path(args.work).resolve();assert not out.exists() and not root.exists();root.mkdir()
    torch.set_num_threads(1)
    basepath=Path(args.base);base=json.loads(basepath.read_bytes())
    assert base['verdict']=='PASS' and all(file_sha(NAV/rel)==digest for rel,digest in base['code_files'].items())
    CHECKS.extend(base['checks']);p=candidate()
    numerical();endpoints(p);replay(p,root/'existing_train_replay.json');stability(p,root);coordinator(p,root)
    EP.CHECKS.clear();production=EP.production(p,root);CHECKS.extend(EP.CHECKS)
    check('terminal_tail_producer_auditor',all(c['ok'] for c in EP.CHECKS) and production['audit']['checks_pass'],production)
    check('terminal_tail_candidate_startup',R.allocation(p)==(50,300) and all('nav064_terminal_tail_v1' in str(v) for v in R.attempt_paths(p).values()))
    files=set(base['code_files'])|{'tests/test_terminal_tail_gate.py','src/camra_navigation/algorithm/stability.py','config/value_objective_contract_terminal_tail_v1.json','config/navigation_contract_terminal_tail_v1.json'}
    report=dict(base,schema='camra-terminal-tail-gate-v1',implementation_versions=p['implementation_versions'],checks=CHECKS,
                verdict='PASS' if all(c['ok'] for c in CHECKS) else 'FAIL',production_fixture=production,
                code_files={rel:file_sha(NAV/rel) for rel in sorted(files)},
                protocol={'path':'config/terminal_tail_numerical_protocol_v1.json','sha256':file_sha(NAV/'config/terminal_tail_numerical_protocol_v1.json')},
                component_reports={str(basepath):file_sha(basepath)},
                limits=['Numerical and mechanical evidence only; no formal training approval.','Old stochastic critic optimism is reduced but remains; v5 learned performance is unknown.','NAV-064=false; deployment/search/physics downstream remain blocked.'])
    startup(p,report,out);report['checks']=CHECKS;report['verdict']='PASS' if all(c['ok'] for c in CHECKS) else 'FAIL'
    with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(report['verdict'],len(CHECKS),'checks',flush=True)
    return int(report['verdict']!='PASS')

if __name__=='__main__':raise SystemExit(main())
