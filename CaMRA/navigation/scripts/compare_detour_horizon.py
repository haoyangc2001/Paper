"""Compare min-GAE temporal influence on retained controlled train traces only.

No new environments, policy training, development or selection evaluations.
Lambda changes here are diagnostic counterfactuals, not production settings.
"""
from __future__ import annotations
import argparse
import copy
import dataclasses as dc
import json
from pathlib import Path
import numpy as np
import torch
import diagnose_detour_feedback as D


def compute(a, cfg, label, net, cap):
    if cap is None or len(a['cost_norm']) <= cap:
        return D.objectives(a, cfg, label, net)
    b={k:v[:cap].copy() for k,v in a.items()}
    if net is not None:
        with torch.no_grad():
            vc,vh=net.values(torch.tensor(b['obs'],dtype=torch.float32))
            bc,bh=net.values(torch.tensor(b['next_obs'][-1:],dtype=torch.float32))
        b['value_cost']=vc.numpy();b['value_reach']=vh.numpy()
        bc,bh=float(bc[0]),float(bh[0])
    else:bc=bh=0.
    D.F.obj_cfg=cfg
    return D.F.close_segment(D.SimpleNamespace(arrays=lambda:b),'truncation',bc,bh,
                              env_done=False,label=None)[0]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--diagnosis',required=True)
    ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out);assert not out.exists();torch.set_num_threads(1)
    dp=Path(args.diagnosis);d=json.loads(dp.read_bytes())
    si=json.loads(Path(d['sources']['path']).read_bytes())
    assert D.file_sha(d['sources']['path'])==d['sources']['sha256']
    ppath=D.NAV/'config/training_loop_prereg_clearance_feedback_v1.json'
    assert D.file_sha(ppath)==d['prereg_sha256']
    p=json.loads(ppath.read_bytes())
    cfg=dc.replace(D.S.ObjectiveConfig.from_contracts(str(D.NAV/p['binding']['value_contract']['path']),r_goal_m=.5),
                   h_semantics='v2_deep_well',joint_min_recursion=True,gamma_h=.99999)
    nets=[(None,None)]
    for r in p['run_registry']:
        if r['variant']!='full':continue
        path=Path('/pub/data/caohy/work/camra_navigation')/r['run_id']/'state.pt'
        assert D.file_sha(path)==si[str(path)]['sha256']
        st=torch.load(path,map_location='cpu',weights_only=False)
        nets.append((r['seed'],D.build_net(p,st)))
    rows=[]
    for m in d['controlled_train_trajectories']:
        arrays=[]
        for mode in ('safe','unsafe','timeout'):
            paths=[Path(x) for x in si if x.endswith('/'+m['map_id']+'_'+mode+'.npz')]
            assert len(paths)==1 and D.file_sha(paths[0])==si[str(paths[0])]['sha256']
            with np.load(paths[0]) as f:a={k:f[k] for k in f.files}
            assert D.S.content_hash(a)==si[str(paths[0])]['content_sha256']
            arrays.append(a)
        labels=[s['label'] for s in m['trajectories']]
        for lam in (.95,.98,.995,.999):
            lc=dc.replace(cfg,gae_lambda=lam)
            for seed,net in nets:
                for cap in (227,None):
                    o=[compute(a,lc,label,net,cap) for a,label in zip(arrays,labels)]
                    vals=[float(x['obj_target_total'][0]) for x in o]
                    # Boundary sensitivity without additional env steps.
                    b=copy.deepcopy(arrays[0]);b['goal_distance_m_next'][-1]=.500001
                    neutral=compute(b,lc,labels[0],net,cap)
                    terminal=float(neutral['obj_target_total'][0]-vals[0])
                    row=dict(map_id=m['map_id'],seed=seed,lambda_=lam,cap=cap,
                        targets=vals,advantages=[float(x['obj_adv_total'][0]) for x in o],
                        safe_lower_than_both=vals[0]<min(vals[1:]),terminal_influence=terminal)
                    if net is not None:
                        row['update']=D.density_update(net,arrays,o)
                    rows.append(row)
        print('COMPARED',m['map_id'],flush=True)
    summary=[]
    for lam in (.95,.98,.995,.999):
        for cap in (227,None):
            selected=[r for r in rows if r['lambda_']==lam and r['cap']==cap and r['seed'] is not None]
            influence=[r['terminal_influence'] for r in selected]
            summary.append(dict(lambda_=lam,cap=cap,comparisons=len(selected),
                safe_lower_than_both=sum(r['safe_lower_than_both'] for r in selected),
                safe_density_improves_against_both=sum(all(r['update']['increases']) for r in selected),
                initial_terminal_influence_range=[min(influence),max(influence)]))
    D.write_new(out,dict(schema='camra-detour-horizon-comparison-v1',owner='NAV-064',
        diagnosis=dict(path=str(dp),sha256=D.file_sha(dp)),script_sha256=D.file_sha(__file__),
        summary=summary,rows=rows,new_rollouts=0,new_training_updates=0,
        limits=['Fixed teacher trajectories with frozen sampled-policy critics; no learned performance claim.',
                'An actual actor-critic on-policy gate remains necessary before performance conclusions.',
                'Changing lambda or collection length in production requires an independently versioned protocol.']))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
