import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path('/home/caohy/repositories/Paper')
NAV=ROOT/'CaMRA/navigation'
HERE=Path(__file__).resolve().parent
load=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
base=load(HERE/'baseline.json')
plan=load(ROOT/'doc/plan/20261109_camra_navigation_implementation.json')
tasks={t['id']:t for t in plan['tasks']}
accept=load(NAV/'reports/nav066_efficiency_limitation_acceptance_v1.json')
assert len(plan['tasks'])==67 and sum(t['status'] for t in plan['tasks'])==54
assert plan['document']['version']=='1.4.1'
assert next(t['id'] for t in plan['tasks'] if not t['status'] and all(tasks[d]['status'] for d in t['depends_on']))=='NAV-067'
for old,new in zip(base['plan']['tasks'],plan['tasks']):
 assert old['id']==new['id'] and old['depends_on']==new['depends_on']
 assert new['work_records'][:len(old['work_records'])]==old['work_records']
 if old['id']!='NAV-066':assert old==new
 else:
  assert new['status'] and not old['status']
  assert new['completion_basis']['disposition']=='ACCEPTED_WITH_LIMITATION'
  assert new['completion_basis']['scientific_status']=='FAIL_BUDGET_EFFICIENCY'
  assert new['acceptance_definition_history'][:len(old['acceptance_definition_history'])]==old['acceptance_definition_history']
  for k in ('goal','constraints_and_notes','acceptance_criteria','planned_outputs','status','verification_result'):
   assert new['acceptance_definition_history'][-1][k]==old[k]
assert all(sha(p)==h for p,h in base['original_evidence_hashes'].items())
assert all(sha(p)==h for p,h in base['protected_hashes'].items())
assert all(sha(NAV/p)==h for p,h in base['implementation_hashes'].items())
reg=load(NAV/'config/nav066_decision_prereg_v2.json')
assert all(sha(c['checkpoint'])==c['checkpoint_sha256'] for c in reg['six_candidates'])
original=load(NAV/'reports/nav066_budget_decision_validation_v2.json')
assert not original['pass'] and original['methods']['full']['efficiency_failures']==1
assert accept['post_result_user_decision'] and accept['scientific_verdict']=='FAIL_BUDGET_EFFICIENCY'
assert plan['budget_decision_stage_disposition_v1']['sha256']==sha(NAV/'reports/nav066_efficiency_limitation_acceptance_v1.json')
assert all(v==0 for v in accept['this_turn_activity'].values())
assert all(not tasks[i]['status'] for i in ('NAV-067',*[f'NAV-{j:03}' for j in range(43,55)]))
with (HERE/'validation-after.txt').open('w') as out:
 subprocess.run([sys.executable,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',str(ROOT/'doc/plan/20261109_camra_navigation_implementation.json'),'--mode','update'],stdout=out,stderr=subprocess.STDOUT,check=True)
subprocess.run(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--check'],cwd=ROOT,check=True)
changed=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines()
assert all(not p.startswith(('manuscripts/','RC-PPO/','HiRA/','CaMRA/locomotion/','CaMRA/calibration/')) and not p.endswith(('.tex','.bib')) for p in changed)
evidence={'plan_valid':True,'history_prefix_preserved':True,'task_order_and_dependencies_preserved':True,
 'only_task_status_changed':'NAV-066 false -> true (accepted limitation, not scientific PASS)',
 'completed':54,'remaining':13,'next':'NAV-067','original_scientific_failure_preserved':True,
 'model_implementation_and_frozen_inputs_unchanged':True,'original_raw_evidence_unchanged':True,
 'new_queries_training_nominal_physical_episodes':0,'final_results_opened':False,
 'original_report_sha256':sha(NAV/'reports/nav066_budget_decision_validation_v2.json'),
 'acceptance_record_sha256':sha(NAV/'reports/nav066_efficiency_limitation_acceptance_v1.json')}
(HERE/'verification.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(evidence,ensure_ascii=False,indent=2))
