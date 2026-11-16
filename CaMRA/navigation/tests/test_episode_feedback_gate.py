"""NAV-064 complete-episode production gate; mechanical evidence only.

The two-update fixture samples the real current policy on all18 train maps.
Its allocation/registration is injected in memory for mechanics, never saved
as an approved formal registration or used as performance/selection evidence.
"""
from __future__ import annotations
import argparse
import copy
import dataclasses as dc
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

NAV=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(NAV/'scripts'),str(NAV/'tests'),str(NAV/'src'),str(NAV.parent/'calibration/src')]
import train_formal as F
import run_feedback_round as R
import audit_r14a2 as A
import test_feedback_gate as B
from camra_navigation.algorithm import storage as S, networks as NW, ppo as P
from camra_navigation.algorithm.validation import file_sha, verify_training_bindings

CHECKS=[]
def check(name,ok,detail=None):
    CHECKS.append(dict(check=name,ok=bool(ok),detail=detail))
    print(('PASS ' if ok else 'FAIL ')+name,flush=True)

def production(p,root):
    fp=copy.deepcopy(p)
    fp['window_semantics']['max_updates']=2
    rid=root.name+'_onpolicy_mechanical'
    fp['run_registry'][3]['run_id']=rid
    # This is a narrow in-memory registration injection, not a saved approval.
    # The actual formal startup must still reject the unapproved proposal.
    pin='mechanical-fixture-not-formal-registration'
    built=[]; snapshots={}; closed=[]; mb_checks=[]
    real_build=NW.build; real_close=F.close_segment; real_update=P.MinimizationUpdater.update
    def build(*args,**kw):
        net=real_build(*args,**kw);built.append(net);return net
    def closer(seg,*args,**kw):
        u=len(closed)//18
        weights={k:v.detach().clone() for k,v in built[0].state_dict().items()}
        if u not in snapshots:snapshots[u]=weights
        assert all(torch.equal(v,snapshots[u][k]) for k,v in weights.items()),'policy changed during collection'
        arrays,kind=real_close(seg,*args,**kw)
        closed.append((u,kind,copy.deepcopy(arrays),kw['env_done']))
        return arrays,kind
    def update(self,batch):
        before={k:v.clone() for k,v in batch.items()}
        result=real_update(self,batch)
        mb_checks.append(all(torch.equal(v,batch[k]) for k,v in before.items()))
        return result
    args=['train_formal','--run-id',rid,'--variant','full','--seed','0','--work-root',str(R.ROOT),'--prereg-name','in-memory-mechanical-fixture']
    with mock.patch.object(F,'verify_prereg',return_value=(fp,pin)), \
         mock.patch.object(sys,'argv',args),mock.patch.object(NW,'build',build), \
         mock.patch.object(F,'close_segment',closer),mock.patch.object(P.MinimizationUpdater,'update',update):
        assert F.main()==0
    run=R.ROOT/rid;logs=A.lines(run/'train_log.jsonl')
    updates=[r for r in logs if r['kind']=='update']
    counts=[r['n_samples'] for r in updates]
    check('episode_production_complete_batches',len(closed)==36 and len(snapshots)==2
          and all(done and kind!='truncation' and not a['truncated'].any()
                  and a['t'][0]==0 and a['event_code'][-1]!=0 for u,kind,a,done in closed)
          and all(r['n_segments']==18 for r in updates)
          and sum(len(a['t']) for u,kind,a,done in closed)==sum(counts)<16200,
          dict(run=str(run),updates=2,episodes=36,samples=counts,limit='Mechanical current-policy fixture, not performance/selection evidence.'))
    independent=True; maxerror=0.
    for u,kind,a,done in closed:
        net=copy.deepcopy(built[0]);net.load_state_dict(snapshots[u]);net.eval()
        with torch.no_grad():
            x=torch.tensor(a['obs'],dtype=torch.float32)
            lp=net.policy(x).log_prob(torch.tensor(a['action_raw'],dtype=torch.float32)).numpy()
            vc,vh=net.values(x)
        err=float(np.max(np.abs(lp-a['log_prob'])));maxerror=max(maxerror,err)
        independent &= np.allclose(lp,a['log_prob'],atol=2e-5,rtol=1e-5)
        independent &= np.allclose(vc.numpy(),a['value_cost'],atol=2e-6,rtol=1e-5)
        independent &= np.allclose(vh.numpy(),a['value_reach'],atol=2e-6,rtol=1e-5)
    check('episode_old_log_prob_fixed',independent and bool(mb_checks) and all(mb_checks),
          dict(minibatches=len(mb_checks),max_log_prob_error=maxerror,critics_replayed=True))
    auditdir=root/'audit';auditdir.mkdir()
    result=A.run_audit(rid,'full',0,fp,pin,2,auditdir)
    check('episode_auditor_variable_samples',result['checks_pass'] and result['counts']['completed_episodes']==36
          and result['counts']['samples']==sum(counts) and result['counts']['chunks']==36
          and min(counts)<8100, result)
    # Corrupt copied in-memory logs, never raw evidence. Auditor must reject
    # incomplete episode identity or inaccurate variable sample counts.
    real_lines=A.lines
    for corruption in ('sample_count','missing_episode','stale_update'):
        d=root/('reject_'+corruption);d.mkdir()
        def changed(path):
            rows=copy.deepcopy(real_lines(path))
            if corruption=='sample_count' and Path(path).name=='train_log.jsonl':
                next(r for r in rows if r['kind']=='update')['n_samples']-=1
            if Path(path).name=='manifest.jsonl':
                if corruption=='missing_episode': rows=rows[:-1]
                if corruption=='stale_update': rows[0]['update_idx']=1
            return rows
        with mock.patch.object(A,'lines',changed):
            try:A.run_audit(rid,'full',0,fp,pin,2,d);rejected=False
            except AssertionError:rejected=True
        check('episode_auditor_reject_'+corruption,rejected)
    return dict(run=str(run),log_sha256=file_sha(run/'train_log.jsonl'),audit=result)

def endings_and_horizon(p):
    cfg=dc.replace(S.ObjectiveConfig.from_contracts(str(NAV/p['binding']['value_contract']['path']),r_goal_m=.5),h_semantics='v2_deep_well',joint_min_recursion=True)
    F.obj_cfg=cfg;events={}
    for label in ('goal_reached','collision','out_of_bounds','budget_exhausted','timeout','stagnation'):
        a,kind=B.fixture(event=label,dist=(4.,3.,.2) if label=='goal_reached' else (4.,3.,2.))
        out,used=F.close_segment(SimpleNamespace(arrays=lambda:copy.deepcopy(a)),kind,.3,.8,env_done=True,label=label)
        events[label]=dict(kind=used,finite=all(np.isfinite(v).all() for v in out.values() if v.dtype.kind=='f'))
    check('episode_all_terminal_kinds_retained',len(events)==6 and all(v['finite'] for v in events.values())
          and events['goal_reached']['kind']=='success' and events['timeout']['kind']=='task_end',events)
    hp=NAV/p['binding']['horizon_diagnosis']['path'];h=json.loads(hp.read_bytes())
    check('episode_horizon_feedback_18_of_18',file_sha(hp)==p['binding']['horizon_diagnosis']['sha256']
          and any(r['lambda_']==.999 and r['cap'] is None and r['comparisons']==18
                  and r['safe_lower_than_both']==r['safe_density_improves_against_both']==18 for r in h['summary']),
          dict(path=str(hp),sha256=file_sha(hp),limit='Controlled interventions with frozen parent critics; no exploration/learning claim.'))
    check('episode_three_estimators_lambda',cfg.gae_lambda==.999 and cfg.gamma_cost==cfg.gamma_joint==1.)
    paths=R.attempt_paths(p)
    check('episode_coordinator_allocation',R.allocation(p)==(50,300) and 'audit50' in paths and 'audit300' in paths
          and '--updates' in R.audit_command(50,'unapproved-proposal',paths)
          and all('nav064_episode_feedback_v1' in str(x) for x in paths.values()))
    check('episode_formal_outputs_absent',not any(x.exists() for x in paths.values())
          and all(not (R.ROOT/r['run_id']).exists() for r in p['run_registry']))

def startup(p,report,out):
    probe=out.with_suffix('.startup.json');probe.write_text(json.dumps(report,indent=2))
    fixture=copy.deepcopy(p);fixture['authorization']['training_approved']=True
    for key,entry in fixture['binding'].items():
        if key=='numerical_gate':entry['path']=str(probe)
        entry['sha256']=file_sha(NAV/entry['path'])
    verify_training_bindings(fixture,NAV)
    check('episode_startup_candidate_accepted_in_memory',True)
    for name in ('unapproved','old_collection','old_calls','old_updates','missing_gate','old_lambda'):
        bad=copy.deepcopy(fixture)
        if name=='unapproved':bad['authorization']['training_approved']=False
        if name=='old_collection':bad['window_semantics']['collection']='window_v1'
        if name=='old_calls':bad['window_semantics']['calls_per_update']=227
        if name=='old_updates':bad['window_semantics']['max_updates']=600
        if name=='missing_gate':
            g=copy.deepcopy(report);g['checks']=[c for c in g['checks'] if c['check']!='episode_production_complete_batches']
            path=out.with_suffix('.missing.json');path.write_text(json.dumps(g));bad['binding']['numerical_gate']={'path':str(path),'sha256':file_sha(path)}
        if name=='old_lambda':
            v=json.loads((NAV/bad['binding']['value_contract']['path']).read_bytes());v['estimator']['gae_lambda']=.95
            path=out.with_suffix('.lambda.json');path.write_text(json.dumps(v));bad['binding']['value_contract']={'path':str(path),'sha256':file_sha(path)}
        try:verify_training_bindings(bad,NAV);rejected=False
        except ValueError:rejected=True
        check('episode_startup_reject_'+name,rejected)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--work',required=True);ap.add_argument('--base');args=ap.parse_args()
    out=Path(args.out).resolve();root=Path(args.work).resolve();assert not out.exists() and not root.exists();root.mkdir()
    torch.set_num_threads(1)
    basepath=Path(args.base) if args.base else out.with_suffix('.clearance.json')
    if not args.base:subprocess.run([sys.executable,str(NAV/'tests/test_clearance_feedback_gate.py'),'--out',str(basepath)],check=True)
    base=json.loads(basepath.read_bytes());assert base['verdict']=='PASS'
    assert all(file_sha(NAV/f)==h for f,h in base['code_files'].items())
    CHECKS.extend(base['checks'])
    p=json.loads((NAV/'config/training_loop_episode_feedback_proposal_v1.json').read_bytes())
    production_result=production(p,root);endings_and_horizon(p)
    files=set(base['code_files'])|{'tests/test_episode_feedback_gate.py','config/navigation_contract_episode_feedback_v1.json','config/value_objective_contract_episode_feedback_v1.json','scripts/diagnose_detour_feedback.py','scripts/compare_detour_horizon.py'}
    report=dict(base,schema='camra-episode-feedback-gate-v1',checks=CHECKS,implementation_versions=p['implementation_versions'],
        verdict='PASS' if all(c['ok'] for c in CHECKS) else 'FAIL',code_files={f:file_sha(NAV/f) for f in sorted(files)},
        protocol={'path':'config/episode_feedback_numerical_protocol_v1.json','sha256':file_sha(NAV/'config/episode_feedback_numerical_protocol_v1.json')},
        component_reports={str(basepath):file_sha(basepath)},production_fixture=production_result,
        limits=['Two current-policy updates on train maps test mechanics only; no formal allocation/selection/performance result.','Controlled detour comparisons do not prove exploration or learning success.','NAV-064 remains false until separately approved training passes development.'])
    startup(p,report,out);report['checks']=CHECKS;report['verdict']='PASS' if all(c['ok'] for c in CHECKS) else 'FAIL'
    with out.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print(report['verdict'],len(CHECKS),'checks',flush=True)
    return int(report['verdict']!='PASS')

if __name__=='__main__':raise SystemExit(main())
