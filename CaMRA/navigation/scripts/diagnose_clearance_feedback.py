"""Reproduce the latest time-feedback diagnosis from retained train/dev only."""
from __future__ import annotations

import argparse
import collections
import datetime
import json
from pathlib import Path
import sys

import numpy as np

NAV=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(NAV/'src'),str(NAV.parent/'calibration/src')]
from camra_navigation.algorithm import storage as S
from camra_navigation.algorithm.validation import file_sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out);assert not out.exists()
    root=Path('/pub/data/caohy/work/camra_navigation')
    parent=NAV/'config/training_loop_prereg_time_feedback_v1.json'
    p=json.loads(parent.read_bytes());pin=file_sha(parent)
    assert pin==(parent.with_suffix('.json.sha256')).read_text().split()[0]
    assert p['implementation_versions']=={'observation':'obs-v6','estimator':'rcppo-feedback-v4'}
    completed=root/'nav064_time_feedback_v1/FINISHED.json'
    closure=json.loads(completed.read_bytes());assert closure['prereg_sha256']==pin
    auditpath=root/'nav064_time_feedback_v1_audit_u600/report.json'
    audit=json.loads(auditpath.read_bytes());assert audit['verdict']=='PASS' and audit['prereg_sha256']==pin
    cfg=S.ObjectiveConfig.from_contracts(str(NAV/p['binding']['value_contract']['path']),r_goal_m=.5)
    sources=[];rows=[]
    for seed in p['method']['seeds']:
        rid=next(r['run_id'] for r in p['run_registry'] if r['variant']=='full' and r['seed']==seed)
        counts=collections.Counter()
        for env in range(18):
            folder=root/rid/f'rollout_env{env}'/'rollout';manifest=folder/'manifest.jsonl'
            sources.append(dict(path=str(manifest),sha256=file_sha(manifest)))
            for line in manifest.read_text().splitlines():
                entry=json.loads(line)
                if entry['global_step_start']//227 not in (99,349,599):continue
                path=folder/entry['file'];assert file_sha(path)==entry['file_sha256']
                with np.load(path) as f:a={k:f[k] for k in f.files}
                assert S.content_hash(a)==entry['content_sha256']
                crossing=(a['safety_y']==0)&(a['safety_y_next']>0)
                g_next=S.g_norm_from_distance(a['goal_distance_m_next'],cfg)
                hidden=crossing&(g_next>=cfg.C*a['safety_y_next'])
                plateau=(a['safety_y']==1)&(entry['terminal_kind']!='failure')
                counts.update(samples=len(a['safety_y']),first_crossing=int(crossing.sum()),immediate_distance_dominates=int(hidden.sum()),
                    saturated_history_nonfailure=int(plateau.sum()),saturated_stop_now=int((plateau&a['obj_stop_now']).sum()),
                    saturated_target_one=int((plateau&np.isclose(a['obj_target_total'],1.,atol=1e-10,rtol=0)).sum()))
                sources.append(dict(path=str(path),sha256=entry['file_sha256'],content_sha256=entry['content_sha256']))
        assert counts['samples']==3*4086
        rows.append(dict(run_id=rid,**counts))
    devpath=NAV/'reports/nav064_time_feedback_v1_development.json'
    dev=json.loads(devpath.read_bytes());assert dev['prereg_sha256']==pin
    development={}
    for variant in p['method']['variants']:
        primary=[e for r in dev['runs'] if r['variant']==variant for e in r['episodes'] if e['k']==1.15]
        assert len(primary)==18
        development[variant]=dict(episodes=18,labels=dict(collections.Counter(e['label'] for e in primary)),
            feasible=sum(e['feasible'] for e in primary),overspent=sum(e['z_J_end']<0 for e in primary),
            dangerous_episodes=sum(any(t['safety_y']>0 for t in e['trace']) for e in primary),
            danger_steps=sum(e['danger_steps'] for e in primary))
    index=out.with_suffix('.sources.json');assert not index.exists()
    index.write_text(json.dumps(dict(split='train',updates=[99,349,599],sources=sources),indent=2)+'\n')
    report=dict(schema='camra-clearance-feedback-diagnosis-v1',owner='NAV-064',timestamp=datetime.datetime.now().astimezone().isoformat(),
        verdict='DIAGNOSIS_ONLY_NOT_PERFORMANCE_ACCEPTANCE',new_training=False,new_performance_rollouts=0,
        parent_prereg=dict(path=str(parent),sha256=pin),closure=dict(path=str(completed),sha256=file_sha(completed),verdict=closure['verdict']),
        full_store_audit=dict(path=str(auditpath),sha256=file_sha(auditpath),verdict=audit['verdict']),
        development_source=dict(path=str(devpath),sha256=file_sha(devpath)),development_summary=dev['summary'],development_primary=development,
        train_selection=dict(updates=[99,349,599],runs=rows,totals=dict(sum((collections.Counter({k:v for k,v in row.items() if k!='run_id'}) for row in rows),collections.Counter()))),
        sources=dict(path=str(index),sha256=file_sha(index)),script=dict(path=str(Path(__file__)),sha256=file_sha(Path(__file__))),
        interpretation=['Distance dominance at first crossing describes the immediate max signal only; bounded future continuation can still penalize the crossing action.',
            'Saturated irreversible history can create equal targets; do not clear safety history or mask unsafe samples to hide this.',
            'Timeout and unsafe routes dominate latest development failures; only one full primary episode overspent.',
            'Explicit clearance is an information-interface hypothesis; numerical gates cannot prove it will fix safety or timeout.',
            'This feature-only candidate leaves gamma/lambda and the learned stochastic-vs-deterministic deployment issue unchanged; refitting deployment critics belongs to NAV-065.'])
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('development_primary','train_selection')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
