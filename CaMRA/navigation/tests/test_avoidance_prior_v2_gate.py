"""Finite repair evidence, separate from formal learning/evaluation allocations."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

NAV = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(NAV/'scripts'), str(NAV/'tests'), str(NAV/'src'),
                str(NAV.parent/'calibration/src')]
import train_formal as F
import eval_r14_repair as E
import test_episode_feedback_gate as EP
import test_avoidance_prior_gate as V1
import run_feedback_round as R
from camra_navigation import observation as O
from camra_navigation.algorithm.policy_prior import (
    PRIOR_CONFIG, PRIOR_CONFIG_V2, ClearanceTangentPrior)
from camra_navigation.algorithm.validation import file_sha, verify_training_bindings

WORK = Path('/pub/data/caohy/work/camra_navigation')
CHECKS = []


def check(name, ok, detail=None):
    CHECKS.append(dict(check=name, ok=bool(ok), detail=detail))
    print(('PASS ' if ok else 'FAIL ') + name, flush=True)


def proposal():
    p = json.loads((NAV/'config/training_loop_avoidance_prior_proposal_v1.json').read_bytes())
    p['avoidance_prior'] = copy.deepcopy(PRIOR_CONFIG_V2)
    p['runtime_contract']['run_prefix'] = 'nav064_avoidance_prior_v2'
    for r in p['run_registry']:
        r['run_id'] = r['run_id'].replace('prior_v1_', 'prior_v2_')
    return p


def mechanics(p, root):
    nav = json.loads((NAV/p['binding']['navigation_contract']['path']).read_bytes())
    torch.manual_seed(0)
    net = E.build_registered_network(nav, p).eval()
    prior = net.policy.avoidance_prior
    # Retained v1 source is independent of the modified implementation.
    oldpath = WORK/'nav064_prior_v2_repair_20261117_v1/source_before/src/camra_navigation/algorithm/policy_prior.py'
    spec = importlib.util.spec_from_file_location('retained_prior_v1', oldpath)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    before = old.ClearanceTangentPrior(PRIOR_CONFIG, net.policy.layout, .465, .2)
    replay = ClearanceTangentPrior(PRIOR_CONFIG, net.policy.layout, .465, .2)
    torch.manual_seed(6417); xx = torch.randn(128, 61)
    check('prior_v1_bitwise_replay', torch.equal(before(xx), replay(xx)),
          dict(snapshot=str(oldpath), sha256=file_sha(oldpath)))
    sl = O.field_slices('obs-v7')
    walls = []
    # Independent world normals: xmin/xmax/ymin/ymax -> +x/-x/+y/-y.
    for i, inward in enumerate(((1.,0.),(-1.,0.),(0.,1.),(0.,-1.))):
        for yaw in np.linspace(-math.pi, math.pi, 16, endpoint=False):
            x = torch.zeros(1, 61); x[0, sl['sin_yaw']] = math.sin(yaw)
            x[0, sl['cos_yaw']] = math.cos(yaw)
            x[0, sl['goal_dir_x_body']] = 1.; x[0, sl['goal_dist_n']] = 1.
            x[0, sl['safety_margin_bins']] = 1.; x[0, sl['wall_margin_n']] = 1.
            x[0, sl['wall_margin_n'].start+i] = 0.
            with torch.no_grad():
                mu = prior(x)[0] + torch.tensor([math.atanh(.465),0.,0.])
            body = np.tanh(mu.numpy())[:2]*[1.,.4]
            c,s = math.cos(yaw), math.sin(yaw)
            world = np.array([[c,-s],[s,c]])@body
            walls.append(float(world@np.asarray(inward)))
    check('prior_v2_all_wall_normals', len(walls)==64 and min(walls)>0.,
          dict(cases=64, minimum_inward_velocity_m_s=min(walls)))
    counts = []
    theta = (np.arange(16)+.5)*math.pi/8-math.pi
    n = np.stack((np.cos(theta),np.sin(theta)),-1)
    sources = {}
    for seed in range(3):
        for env in range(6,12):
            folder=WORK/f'nav064_avoidance_prior_v1_full_s{seed}'/f'rollout_env{env}'/'rollout'
            manifest=folder/'manifest.jsonl'; sources[str(manifest)]=file_sha(manifest)
            for e in map(json.loads, manifest.read_text().splitlines()):
                if e['update_idx']>=50: continue
                path=folder/e['file']; assert file_sha(path)==e['file_sha256']
                with np.load(path) as a: x=a['obs']
                raw=(1-x[:,10:26])*6; margin=x[:,38:54]*6; goal=x[:,7:9]
                dot=goal@n.T; cross=goal[:,0,None]*n[:,1]-goal[:,1,None]*n[:,0]
                threat=(x[:,38:54]<1)&(dot>0)&(raw*dot<x[:,9,None]*15+.25)&(raw*abs(cross)<np.maximum(raw-margin,0)+.25)
                w=np.where(threat,np.clip((2.5-margin)/2.5,0,1),0)
                idx=w.argmax(-1); active=w[np.arange(len(x)),idx]>.1
                cr=cross[np.arange(len(x)),idx]
                oldside=np.where(cr>=0,-1,1)
                newside=np.where(cr < -math.sin(math.pi/8),1,-1)
                pair=active[1:]&active[:-1]
                counts.append((int(sum((oldside[1:]!=oldside[:-1])&pair)),
                               int(sum((newside[1:]!=newside[:-1])&pair))))
                sources[str(path)]=e['file_sha256']
    count=np.asarray(counts)
    check('prior_v2_quantized_side_replay', len(count)==900 and
          int(count[:,1].sum())<int(count[:,0].sum()) and np.median(count[:,1])==0,
          dict(episodes=len(count),old_total=int(count[:,0].sum()),
               new_total=int(count[:,1].sum()),old_median=float(np.median(count[:,0])),
               new_median=float(np.median(count[:,1])),
               limit='Same recorded observations; no counterfactual rollout.'))
    traces=[]; summaries=[]
    for variant in p['method']['variants']:
        torch.manual_seed(0); actor=E.build_registered_network(nav,p).eval()
        envs, refs, ids = F.build_envs(p,variant)
        for env, mid in zip(envs,ids):
            # Fixed diagnostic budget point, never selected from outcomes.
            env.reset(z0_J=1.15*refs[mid]); trace=[]
            while not env.done[0]:
                x=env.observation.copy()
                with torch.no_grad():
                    action=actor.policy(torch.tensor(x,dtype=torch.float32)).mean.numpy()
                rec=env.step(action)
                trace.append(dict(obs=x[0].tolist(),action=action[0].tolist(),
                                  next_obs=rec.obs[0].tolist(),
                                  y=float(rec.safety_y[0]),z_J=float(rec.z_J[0])))
                assert len(trace)<=450
            res=env.result(0)
            row=dict(variant=variant,map_id=mid,label=env._label[0],
                     steps=len(trace),safe=all(r['y']==0 for r in trace),
                     remaining_J=float(res.z_J),distance_m=float(res.goal_distance_m))
            summaries.append(row);traces.append(dict(summary=row,trace=trace))
            print('MECHANISM',json.dumps(row),flush=True)
    with (root/'train_mechanism_traces.json').open('x') as f: json.dump(traces,f)
    with (root/'replay_sources.json').open('x') as f: json.dump(sources,f,indent=2)
    check('prior_v2_train_closed_loop', len(summaries)==36 and
          all(r['label']=='goal_reached' and r['safe'] for r in summaries),
          dict(episodes=36,rows=summaries,
               limits='Fresh deterministic seed0 train-only mechanism fixture. No optimization, parameter search or formal performance claim; budget failures retained.'))
    # Probability and evaluator use exactly the same repaired latent mean.
    x=torch.tensor([t['trace'][0]['obs'] for t in traces],dtype=torch.float32)
    dist=net.policy(x); a=dist.sample(); u=dist._atanh(a)
    manual=torch.distributions.Normal(dist.base.base_dist.loc,dist.stddev).log_prob(u).sum(-1)-torch.log1p(-a.square()+dist.eps).sum(-1)
    rebuilt=E.build_registered_network(nav,p);rebuilt.load_state_dict(net.state_dict())
    check('prior_v2_density_and_factory',torch.allclose(manual,dist.log_prob(a),atol=2e-5,rtol=1e-5)
          and torch.equal(rebuilt.policy(x).mean,dist.mean))
    return summaries


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True)
    ap.add_argument('--work',required=True); ap.add_argument('--base')
    ap.add_argument('--mechanics', help='Reuse unchanged finite mechanics; never repeat its36 episodes.')
    args=ap.parse_args(); root=Path(args.work); out=Path(args.out)
    assert not root.exists() and not out.exists(); root.mkdir()
    torch.set_num_threads(1); p=proposal()
    paths=R.attempt_paths(p)
    check('prior_v2_coordinator',R.allocation(p)==(50,300) and
          all('nav064_avoidance_prior_v2' in str(v) for v in paths.values()),
          {k:str(v) for k,v in paths.items()})
    if args.mechanics:
        saved=json.loads(Path(args.mechanics).read_bytes())
        assert saved['verdict']=='PASS'
        rel='src/camra_navigation/algorithm/policy_prior.py'
        assert saved['code_files'][rel]==file_sha(NAV/rel)
        CHECKS.extend(saved['checks']);summaries=saved['mechanism']
        assert len(summaries)==36
    else:
        summaries=mechanics(p,root)
    if args.base and all(c['ok'] for c in CHECKS):
        base=json.loads(Path(args.base).read_bytes())
        assert base['verdict']=='PASS'
        assert all(file_sha(NAV/f)==h for f,h in base['code_files'].items())
        own=list(CHECKS); CHECKS[:]=base['checks']+own
        EP.CHECKS.clear(); prod=EP.production(p,root)
        CHECKS.extend(EP.CHECKS)
        check('prior_actual_producer',all(c['ok'] for c in EP.CHECKS),prod)
        check('prior_actual_auditor',prod['audit']['checks_pass'])
        report=dict(base,checks=CHECKS,avoidance_prior=p['avoidance_prior'])
        files=set(base['code_files'])|{'tests/test_avoidance_prior_v2_gate.py'}
        report['code_files']={f:file_sha(NAV/f) for f in sorted(files)}
        report['protocol']={'path':'config/avoidance_prior_numerical_protocol_v2.json',
                            'sha256':file_sha(NAV/'config/avoidance_prior_numerical_protocol_v2.json')}
        report['verdict']='PASS'
        V1.CHECKS.clear(); V1.startup(p,report,out); CHECKS.extend(V1.CHECKS)
        # Missing repair evidence must fail the actual startup validator.
        bad=copy.deepcopy(p);bad['authorization']['training_approved']=True
        gate=copy.deepcopy(report);gate['checks']=[c for c in CHECKS if c['check']!='prior_v2_all_wall_normals']
        broken=root/'missing_wall_gate.json';broken.write_text(json.dumps(gate))
        for name,e in bad['binding'].items():
            if name=='numerical_gate':e['path']=str(broken)
            e['sha256']=file_sha(NAV/e['path'])
        try:verify_training_bindings(bad,NAV);rejected=False
        except ValueError:rejected=True
        check('prior_v2_missing_wall_gate_rejected',rejected)
    else:
        report=dict(code_files={'src/camra_navigation/algorithm/policy_prior.py':file_sha(NAV/'src/camra_navigation/algorithm/policy_prior.py')})
    report.update(schema='camra-avoidance-prior-v2-gate',owner='NAV-064',
                  checks=CHECKS,verdict='PASS' if all(c['ok'] for c in CHECKS) else 'FAIL',
                  mechanism=summaries,scope='Finite train-only repair evidence; no formal allocation.')
    with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(report['verdict'],len(CHECKS),'checks',flush=True)
    return int(report['verdict']!='PASS')


if __name__=='__main__': raise SystemExit(main())
