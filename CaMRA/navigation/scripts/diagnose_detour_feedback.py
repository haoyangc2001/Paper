"""Pinned train replay and complete, controlled train-map counterexamples.

Reference commands are diagnostic interventions, never on-policy PPO samples,
demonstration training data, or evidence of learned navigation performance.
All outputs are new; the selected immutable preregistration is explicit.
"""
from __future__ import annotations

import argparse
import collections
import copy
import dataclasses as dc
import datetime
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

NAV = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(NAV/'scripts'), str(NAV/'src'), str(NAV.parent/'calibration/src')]
import train_formal as F
import eval_training_round as E
from camra_navigation.algorithm import storage as S, networks as NW, ppo as P
from camra_navigation.algorithm.validation import file_sha as _file_sha
from camra_navigation.budget.accounting import consistent_e_ref
from camra_navigation.maps.feasibility import check_map


def file_sha(path):
    return _file_sha(Path(path))


def write_new(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2); f.write('\n')


def load_chunk(path, entry, sources):
    assert file_sha(path) == entry['file_sha256'], path
    with np.load(path) as f: a = {k: f[k] for k in f.files}
    assert S.content_hash(a) == entry['content_sha256'], path
    sources[str(path)] = {'sha256': entry['file_sha256'], 'content_sha256': entry['content_sha256']}
    return a


def train_replay(root, p, cfg, sources):
    rows = []
    for r in p['run_registry']:
        if r['variant'] != 'full': continue
        counts = collections.Counter(); goals = collections.Counter()
        for env in range(18):
            folder = root/r['run_id']/f'rollout_env{env}'/'rollout'
            manifest = folder/'manifest.jsonl'
            sources[str(manifest)] = {'sha256': file_sha(manifest)}
            entries = [json.loads(x) for x in manifest.read_text().splitlines()]
            for e in entries:
                selected = e['update_idx'] in (99, 349, 599)
                arrival = (e['map_id'].startswith('l2_') and e['env_done']
                           and e['terminal_kind'] == 'success')
                if not selected and not arrival: continue
                a = load_chunk(folder/e['file'], e, sources)
                if arrival:
                    assert S.EVENT_NAMES[int(a['event_code'][-1])] == 'goal_reached'
                    safe = a['safety_y_next'][-1] == 0.
                    goals.update(arrivals=1, safe_arrivals=int(safe),
                                 feasible_arrivals=int(safe and a['z_norm_next'][-1] >= 0.))
                if not selected: continue
                rebuilt = S.compute_objectives(a, dc.replace(cfg, gamma_h=e['gamma_h']),
                                                terminal_kind=e['terminal_kind'])
                for k in ('obj_target_cost', 'obj_target_reach', 'obj_target_total'):
                    assert np.allclose(a[k], rebuilt[k], atol=1e-10, rtol=1e-9), (r['run_id'], k)
                first = (a['safety_y'] == 0.) & (a['safety_y_next'] > 0.)
                hidden = first & (S.g_norm_from_distance(a['goal_distance_m_next'], cfg)
                                  >= cfg.C*a['safety_y_next'])
                plateau = (a['safety_y'] == 1.) & (e['terminal_kind'] != 'failure')
                counts.update(samples=len(first), first_crossing=int(first.sum()),
                    distance_masks_immediate_crossing=int(hidden.sum()),
                    saturated_nonfailure=int(plateau.sum()),
                    plateau_stop_now=int((plateau & a['obj_stop_now']).sum()),
                    plateau_target_one=int((plateau & np.isclose(a['obj_target_total'], 1., atol=1e-10, rtol=0)).sum()))
        assert counts['samples'] == 12258
        rows.append(dict(run_id=r['run_id'], selected_counts=dict(counts), l2_all_updates=dict(goals)))
        print('REPLAY', r['run_id'], dict(goals), flush=True)
    return rows


def build_net(p, state=None):
    nav = json.loads((NAV/p['binding']['navigation_contract']['path']).read_bytes())
    ini = p['action_init_fix']
    net = NW.build(nav, action_bounded=p['action_bounded'],
        initial_forward_speed=ini['initial_forward_speed'], log_std_min=ini['log_std_min'],
        log_std_init=ini['log_std_init'], steering_gain=p['steering_prior']['gain'])
    if state is not None: net.load_state_dict(state['model'])
    return net


def controlled_trajectory(spec, ctx, p, z0, points, mode):
    sev=p['safety_continuity']; stag=p['stagnation']
    env=E.VectorNominalNavEnv(spec,ctx['geom'],ctx['cm'],n_envs=1,z0_J=z0,
        t_max_s=p['environment']['t_max_s'],r_goal_m=p['environment']['r_goal_m'],
        observation_version=p['implementation_versions']['observation'],
        stagnation_window_s=stag['window_s'],stagnation_min_displacement_m=stag['min_displacement_m'],
        safety_severity_enabled=sev['enabled'],safety_severity_floor=sev['severity_floor'],
        safety_severity_ref_depth_m=sev['severity_ref_depth_m'])
    rows=[]; wp=1
    while not env.done[0]:
        pose=env.pose[0]; x=env.observation[0].copy()
        while wp<len(points)-1 and np.linalg.norm(np.asarray(points[wp])-pose[:2])<1e-7: wp+=1
        delta=np.asarray(points[wp])-pose[:2]; dist=np.linalg.norm(delta)
        # Pure translation follows the verified polyline exactly; no rounding a corner.
        velocity=delta*min(10., .4/max(dist,1e-12))
        c,s=np.cos(pose[2]),np.sin(pose[2])
        command=np.array([c*velocity[0]+s*velocity[1],-s*velocity[0]+c*velocity[1],0.])
        action=command/np.array([1.,.4,1.])  # env.step consumes normalized actions.
        if mode=='timeout': action[:]=[.01,0.,0.]
        assert np.max(np.abs(action))<=1.+1e-12
        rec=env.step(action[None,:])
        rows.append(dict(obs=x,action=action,next_obs=rec.obs[0].copy(),
            cost_norm=rec.cost_model_J[0]/550.,z_norm=rec.z_J_before[0]/550.,
            z_norm_next=rec.z_J[0]/550.,z_ratio=rec.z_ratio_before[0],z_ratio_next=rec.z_ratio[0],
            safety_y=rec.safety_y_before[0],safety_y_next=rec.safety_y[0],
            goal_distance_m=rec.goal_distance_before[0],goal_distance_m_next=rec.goal_distance_m[0],
            pose_before=pose.copy(),pose_after=rec.pose_after[0].copy(),
            event_code=S.EVENT_CODES[rec.label[0] or 'none']))
        assert len(rows)<=451
    a={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    n=len(rows)
    for k in ('value_cost','value_reach','bootstrap_value_cost','bootstrap_value_reach'):
        a[k]=np.zeros(n)
    a['terminated']=np.zeros(n,bool);a['truncated']=np.zeros(n,bool)
    # Use the actual environment event, not the intended route label.
    label=env._label[0]
    return a,dict(label=label,steps=n,cost_J=float(env.result(0).cost_model_J),
        remaining_J=float(env.result(0).z_J),safe=bool(a['safety_y_next'][-1]==0.),
        final_distance_m=float(a['goal_distance_m_next'][-1]),
        feasible=bool(label=='goal_reached' and a['safety_y_next'][-1]==0. and a['z_norm_next'][-1]>=0.))


def objectives(a, cfg, label, net=None):
    b=copy.deepcopy(a)
    if net is not None:
        with torch.no_grad():
            vc,vh=net.values(torch.tensor(b['obs'],dtype=torch.float32))
        b['value_cost']=vc.numpy();b['value_reach']=vh.numpy()
    F.obj_cfg=cfg
    return F.close_segment(SimpleNamespace(arrays=lambda:b),'truncation',0.,0.,
                           env_done=True,label=label)[0]


def density_update(net, trajectories, computed):
    # Three interventions at exactly the same observation. Controlled direction
    # check only: these actions are not samples drawn from the source policy.
    model=copy.deepcopy(net)
    x=torch.tensor(np.stack([a['obs'][0] for a in trajectories]),dtype=torch.float32)
    acts=torch.tensor(np.stack([a['action'][0] for a in trajectories]),dtype=torch.float32)
    assert torch.equal(x,x[0].expand_as(x))
    with torch.no_grad():
        old=model.policy(x).log_prob(acts).clone();vc,vh=model.values(x)
    batch=dict(obs=x,action=acts,log_prob_old=old,adv_total=torch.tensor(
        [o['obj_adv_total'][0] for o in computed],dtype=torch.float32),
        value_cost_old=vc,target_cost=torch.tensor([o['obj_target_cost'][0] for o in computed],dtype=torch.float32),
        value_reach_old=vh,target_reach=torch.tensor([o['obj_target_reach'][0] for o in computed],dtype=torch.float32))
    updater=P.MinimizationUpdater(model,dc.replace(P.PPOConfig(),ent_coef=0.,max_grad_norm=2.5))
    before=(old[0]-old[1:]).tolist()
    stats=updater.update(batch)
    with torch.no_grad():after=(model.policy(x).log_prob(acts)[0]-model.policy(x).log_prob(acts)[1:]).tolist()
    return dict(safe_vs_unsafe_and_timeout_log_density_before=before,
                safe_vs_unsafe_and_timeout_log_density_after=after,
                increases=[b>a for a,b in zip(before,after)],grad_norm_policy=stats['grad_norm_policy'])


def counterexamples(root, p, cfg, outdir, sources):
    ctx=E.build_rollout_ctx('full')
    split=json.loads((NAV/'config/split_manifest_v1.json').read_bytes())
    maps=[m for m in split['splits']['train']['maps'] if m['level']=='L2_single_detour']
    refpath=NAV/p['binding']['ref_energy_t45']['path']
    refs={m['map_id']:m['e_ref_J'] for m in json.loads(refpath.read_bytes())['maps']}
    sources[str(refpath)]={'sha256':file_sha(refpath)}
    nets=[]
    for seed in p['method']['seeds']:
        rid=next(r['run_id'] for r in p['run_registry'] if r['variant']=='full' and r['seed']==seed)
        path=root/rid/'state.pt';sources[str(path)]={'sha256':file_sha(path)}
        st=torch.load(path,map_location='cpu',weights_only=False)
        assert st['prereg_sha256']==file_sha(NAV/'config/training_loop_prereg_clearance_feedback_v1.json')
        nets.append((seed,build_net(p,st)))
    reports=[]
    for m in maps:
        spec=E.M.build(m['level'],m['seed'],ctx['design'])
        admitted=check_map(spec,ctx['geom'],resolution=.1,t_deadline_s=45.)
        assert admitted['admitted'], m['map_id']
        routes=[admitted['route']]
        shortcut_geom=dc.replace(ctx['geom'],delta_m=ctx['geom'].delta_m/2.)
        shortcut=check_map(spec,shortcut_geom,resolution=.1,t_deadline_s=45.)
        assert shortcut['admitted']
        shortcuts=[shortcut['route']]
        z0=1.15*consistent_e_ref(refs[m['map_id']],'full')
        a=[];summaries=[]
        for mode,points in [('safe',routes[0]['polyline']),('unsafe',shortcuts[0]['polyline']),('timeout',[spec.start,spec.goal])]:
            arr,summary=controlled_trajectory(spec,ctx,p,z0,points,mode)
            path=outdir/(m['map_id']+'_'+mode+'.npz');assert not path.exists()
            np.savez_compressed(path,**arr)
            sources[str(path)]={'sha256':file_sha(path),'content_sha256':S.content_hash(arr)}
            a.append(arr);summaries.append(dict(mode=mode,**summary))
        print('GENERATED',m['map_id'],summaries,flush=True)
        assert summaries[0]['feasible'], 'safe diagnostic route must actually succeed'
        assert summaries[1]['label']=='goal_reached' and not summaries[1]['safe']
        assert summaries[2]['label']=='timeout' and summaries[2]['safe']
        assert all(np.array_equal(a[0]['obs'][0],b['obs'][0]) for b in a)
        results=[]
        for seed,net in [(None,None)]+nets:
            comp=[objectives(b,cfg,su['label'],net) for b,su in zip(a,summaries)]
            rows=[]
            for ob in comp:
                rows.append({k:float(ob[k][0]) for k in ('obj_target_cost','obj_target_reach',
                    'obj_target_total','obj_adv_total','obj_stop_index','obj_g_train')})
            outcome=dict(seed=seed,baseline='zero' if net is None else 'final_sampled_policy_critic',first_state=rows)
            if net is not None:outcome['controlled_first_action_update']=density_update(net,a,comp)
            # Counterfactual boundary influence: erase ONLY the goal deep well
            # in a synthetic copy of this same safe trajectory, not a new task.
            base=a[0].copy();base['goal_distance_m_next']=a[0]['goal_distance_m_next'].copy()
            base['goal_distance_m_next'][-1]=.500001
            neutral=objectives(base,cfg,'goal_reached',net)
            diff=neutral['obj_target_total']-comp[0]['obj_target_total']
            outcome['safe_terminal_influence']={str(t):float(diff[t]) for t in (0,min(99,len(diff)-1),len(diff)-1)}
            outcome['safe_before_u227_steps']=min(227,len(diff))
            results.append(outcome)
        reports.append(dict(map_id=m['map_id'],z0_J=z0,reference_polyline=routes[0]['polyline'],
                            trajectories=summaries,target_comparisons=results))
        print('COUNTEREXAMPLE',m['map_id'],summaries,flush=True)
    return reports


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--work-root',default='/pub/data/caohy/work/camra_navigation')
    ap.add_argument('--trajectory-dir',required=True)
    ap.add_argument('--replay-cache',help='Reuse a completed read-only replay, verifying its source hashes.')
    args=ap.parse_args();out=Path(args.out);assert not out.exists()
    td=Path(args.trajectory_dir);td.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    pr=NAV/'config/training_loop_prereg_clearance_feedback_v1.json';p=json.loads(pr.read_bytes())
    assert file_sha(pr)==pr.with_suffix('.json.sha256').read_text().split()[0]
    assert p['implementation_versions']=={'observation':'obs-v7','estimator':'rcppo-feedback-v4'}
    cfg=dc.replace(S.ObjectiveConfig.from_contracts(str(NAV/p['binding']['value_contract']['path']),r_goal_m=.5),
                   h_semantics='v2_deep_well',joint_min_recursion=True,gamma_h=.99999)
    sources={str(pr):{'sha256':file_sha(pr)}}
    for key in ('split','map_design','geometry_contract','power_model','footprint','margins','navigation_contract','value_contract'):
        e=p['binding'][key];path=(NAV/e['path']).resolve()
        assert file_sha(path)==e['sha256'];sources[str(path)]={'sha256':file_sha(path)}
    root=Path(args.work_root)
    for stage in (100,600):
        path=root/f'nav064_clearance_feedback_v1_audit_u{stage}'/'report.json'
        audit=json.loads(path.read_bytes());assert audit['verdict']=='PASS' and audit['prereg_sha256']==file_sha(pr)
        sources[str(path)]={'sha256':file_sha(path)}
    if args.replay_cache:
        cache=json.loads(Path(args.replay_cache).read_bytes())
        for path,e in cache['sources'].items(): assert file_sha(path)==e['sha256'],path
        sources.update(cache['sources']);train=cache['train']
    else:
        train=train_replay(root,p,cfg,sources)
    write_new(td/'train_replay.json',dict(train=train,sources=sources))
    cases=counterexamples(root,p,cfg,td,sources)
    devpath=NAV/'reports/nav064_clearance_feedback_v1_development.json'
    dev=json.loads(devpath.read_bytes());assert dev['prereg_sha256']==file_sha(pr)
    sources[str(devpath)]={'sha256':file_sha(devpath)}
    primary={}
    for variant in p['method']['variants']:
        eps=[e for r in dev['runs'] if r['variant']==variant for e in r['episodes'] if e['k']==1.15]
        assert len(eps)==18
        primary[variant]=dict(episodes=18,feasible=sum(e['feasible'] for e in eps),
            labels=dict(collections.Counter(e['label'] for e in eps)),
            unsafe_episodes=sum(any(t['safety_y']>0 for t in e['trace']) for e in eps),overspent=sum(e['z_J_end']<0 for e in eps))
    index=out.with_suffix('.sources.json');write_new(index,sources)
    write_new(out,dict(schema='camra-detour-feedback-diagnosis-v1',owner='NAV-064',
        timestamp=datetime.datetime.now().astimezone().isoformat(),verdict='DIAGNOSIS_ONLY',
        prereg_sha256=file_sha(pr),train=train,controlled_train_trajectories=cases,
        development_summary=dev['summary'],primary=primary,
        data_access=dict(existing_train_updates=[99,349,599],l2_goal_events_all_600_updates=True,
            new_controlled_train_trajectories=len(cases)*3,new_training_updates=0,
            new_learned_policy_performance_episodes=0,budget_selection_or_final_read=False),
        sources=dict(path=str(index),sha256=file_sha(index)),script_sha256=file_sha(__file__),
        limitations=['Complete reference routes are diagnostic interventions, not samples of the learned policy.',
            'A conditional outcome advantage with a sampled-policy critic is not an unbiased comparison of three policies.',
            'First-action density comparisons use actual MinimizationUpdater on three controlled rows; they are not integrated event probabilities.',
            'Terminal-well ablation is a target sensitivity check; it is not an executable alternate episode.',
            'No change to safety constants, lambda, stopping, observations, training data or frozen resources.']))
    print('DIAGNOSIS WRITTEN',out,flush=True)


if __name__=='__main__':main()
