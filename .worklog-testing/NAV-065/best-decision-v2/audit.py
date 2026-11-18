from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import socket
import sys

import numpy as np
from camra_navigation.algorithm.deployment_oracle import reconstruct_truth
from camra_navigation.budget.guarded import GRID, summarize_decisions

ROOT=Path('/home/caohy/repositories/Paper')
NAV=ROOT/'CaMRA/navigation'
HERE=Path(__file__).resolve().parent
OUT=Path('/pub/data/caohy/work/camra_navigation/nav066_budget_decision_v2_best')
load=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
reg=load(NAV/'config/nav066_decision_prereg_v2.json')
pin=sha(NAV/'config/nav066_decision_prereg_v2.json')
assert pin==(NAV/'config/nav066_decision_prereg_v2.json.sha256').read_text().strip()
assert all(sha(p)==h for p,h in reg['protected_hashes'].items())
assert all(sha(NAV/p)==h for p,h in reg['implementation_hashes'].items())
assert all(sha(c['checkpoint'])==c['checkpoint_sha256'] for c in reg['six_candidates'])
saved=load(OUT/'decisions.json')
assert sha(OUT/'decisions.json')==(OUT/'decisions.json.sha256').read_text().strip()
manifest=load(OUT/'manifest.json')
result=load(OUT/'decision_validation.json')
finished=load(OUT/'FINISHED.json')
assert finished['report_sha256']==sha(OUT/'decision_validation.json')
assert saved['registration_pin']==pin and saved['manifest_sha256']==sha(OUT/'manifest.json')
maps={m['map_id']:m for m in manifest['maps']}
journal=[json.loads(line) for line in (OUT/'episodes.jsonl').read_text().splitlines()]
assert len(journal)==180
starts=journal[::2]
rows=[{k:v for k,v in r.items() if k!='kind'} for r in journal[1::2]]
assert all(r['kind']=='episode_start' for r in starts)
assert all(r['kind']=='episode_result' for r in journal[1::2])
assert [s['number'] for s in starts]==list(range(1,91))
assert len({(r['method'],r['map_id'],r['k']) for r in rows})==90
assert (OUT/'decisions.json').stat().st_mtime_ns < (OUT/'episodes.jsonl').stat().st_mtime_ns
prereg=load(NAV/'config/training_loop_prereg_terminal_tail_v1.json')
value=load(NAV/prereg['binding']['value_contract']['path'])
decisions={(d['method'],d['map_id']):d for d in saved['decisions']}
assert len(decisions)==18 and saved['query_count']==90 and saved['rollouts_started']==0
for start,row in zip(starts,rows):
 assert all(start[k]==row[k] for k in ('method','map_id','k'))
 ep=row['episode'];m=maps[row['map_id']]
 assert 0<ep['n_steps']<=450 and ep['t_s']<=45+1e-9
 truth=reconstruct_truth(ep,initial_goal_distance_m=math.dist(m['spec']['start'],m['spec']['goal']),value_contract=value,
   horizon_s=45,goal_radius_m=.5,step_s=.1)
 assert truth.feasible==row['actual_feasible']==ep['feasible']
 assert math.isclose(ep['z0_J'],row['k']*m['e_ref_consistent_J'][row['method']],rel_tol=0,abs_tol=1e-10)
 d=decisions[row['method'],row['map_id']]
 i=GRID.index(row['k'])
 assert ep['trace'][0]['observation']==d['query_observations'][i]
 assert row['selected']==(i==d['decision']['selected_index'])
for key,d in decisions.items():
 scores=[q['W'] for q in d['decision']['trace']]
 assert len(scores)==5
 for q in d['decision']['trace']:
  assert q['W']==max(q['vh'],q['vc_norm']-q['budget_J']/550)
  assert q['raw_feasible']==(q['W']<=0)
 flags=[w<0 for w in scores]
 choice=d['decision']['selected_index']
 if choice is not None:
  assert flags.index(True)+1==choice and flags[choice]
  assert not any(flags[i] and not flags[i+1] for i in range(4))
  assert 0 not in scores
 selected_rows=[r for r in rows if (r['method'],r['map_id'])==key]
 if choice is not None:assert selected_rows[0]['k']==GRID[choice]
again=summarize_decisions(saved['decisions'],rows,manifest['maps'])
assert all(again[k]==result[k] for k in again)
steps=sum(r['episode']['n_steps'] for r in rows)
assert steps==result['steps'] and steps<=40500 and len(rows)==90
baseline=load(HERE/'baseline.json')
plan=load(ROOT/'doc/plan/20261109_camra_navigation_implementation.json')
assert [t['id'] for t in plan['tasks']]==[t['id'] for t in baseline['plan']['tasks']]
assert all(new['work_records'][:len(old['work_records'])]==old['work_records'] for old,new in zip(baseline['plan']['tasks'],plan['tasks']))
grid=[]
for r in rows:
 e=r['episode'];m=r['method'];ref=maps[r['map_id']]['e_ref_nominal_J']
 ratio=maps[r['map_id']]['e_ref_consistent_J'][m]/ref
 grid.append({k:r[k] for k in ('method','map_id','k','selected','actual_feasible')}|{
  'label':e['label'],'t_s':e['t_s'],'danger_steps':e['danger_steps'],'budget_J':e['z0_J'],
  'cost_model_J':e['cost_model_J'],'cost_nominal_J':e['cost_model_J']/ratio,
  'overspend_J':max(e['cost_model_J']-e['z0_J'],0),
  'overspend_fraction':max(e['cost_model_J']/e['z0_J']-1,0),
  'utilization':e['cost_model_J']/e['z0_J']})
audited={**result,'grid_rows':grid,'preregistration':'config/nav066_decision_prereg_v2.json',
 'selected_candidates':reg['selected_candidates'],
 'readiness':'reports/nav065_budget_decision_readiness_v2.json',
 'raw_evidence_directory':str(OUT),'source_commit':'0696350a60b289e713da39dbe5484c0cbfedf004',
 'integrity_pass':True,'old_G4':'FAIL',
 'interpretation':'各方法9个实际预算决策，45网格行仅用于召回/效率诊断。未训练或反馈调参；有限名义研究结果，不证明物理、未知布局、多种子或总体风险。'}
(NAV/'reports/nav066_budget_decision_validation_v2.json').write_text(json.dumps(audited,ensure_ascii=False,indent=2)+'\n')
summary={'integrity_pass':True,'scientific_gate':result['pass'],'audit_time':datetime.now().astimezone().isoformat(),
 'episodes':90,'unique_keys':90,'steps':steps,'queries':90,'decisions':18,
 'registration_pin':pin,'model_identity_pass':True,'protected_identity_pass':True,
 'decision_seal_pass':True,'history_prefix_preserved':True,'final_results_opened':False,
 'updates':0,'physical_episodes':0,'host':socket.getfqdn(),'python':sys.executable,
 'source_hashes':{str(p):sha(p) for p in (OUT/'manifest.json',OUT/'decisions.json',OUT/'episodes.jsonl',OUT/'decision_validation.json')}}
(HERE/'final-audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({m:{k:v for k,v in s.items() if k!='details'} for m,s in result['methods'].items()},ensure_ascii=False,indent=2))
print('INTEGRITY_PASS',steps,'steps; scientific_gate',result['pass'],flush=True)
