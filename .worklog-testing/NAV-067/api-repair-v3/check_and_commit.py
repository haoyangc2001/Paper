"""Verify immutable identities and append-only plan, then commit this repair."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

from nav067_sources import NAV, ROOT, OUTPUT, read_manifest, sha, write_new

OUT = Path(__file__).resolve().parent
manifest, _, _ = read_manifest()
verification = json.loads((OUT/'verification.json').read_bytes())
assert '37 passed' in verification['tests_output']
for rel,digest in verification['new_source_hashes'].items():
    assert sha(NAV/rel) == digest, rel
proposal_path = NAV/'config/nav067_startup_recovery_proposal_v3.json'
proposal = json.loads(proposal_path.read_bytes())
report = json.loads((NAV/'reports/nav067_api_repair_preparation_v3.json').read_bytes())
assert sha(proposal_path) == report['proposal_sha256']
assert proposal['status'] == 'PROPOSED'
for rel,digest in proposal['source_hashes'].items():
    assert sha(NAV/rel) == digest, rel
for path,digest in proposal['evidence_hashes'].items():
    assert sha(path) == digest, path
for slot in proposal['slots']:
    assert not (OUTPUT/slot['relative_output']).exists()
assert not (NAV/'config/nav067_startup_recovery_v3.json').exists()

plan_rel = 'doc/plan/20261109_camra_navigation_implementation.json'
plan = json.loads((ROOT/plan_rel).read_bytes())
base = json.loads(subprocess.check_output(['git','show',
    'f0bd8c7976fdaa02d16f9be076157076993cf2f8:'+plan_rel],cwd=ROOT))
assert {k:v for k,v in plan.items() if k!='tasks'} == {k:v for k,v in base.items() if k!='tasks'}
appended = []
for old,new in zip(base['tasks'],plan['tasks']):
    assert {k:v for k,v in old.items() if k!='work_records'} == {k:v for k,v in new.items() if k!='work_records'}
    prefix = len(old['work_records'])
    assert old['work_records'] == new['work_records'][:prefix]
    if len(new['work_records']) != prefix:
        appended.append((new['id'],len(new['work_records'])-prefix))
assert len(plan['tasks']) == len(base['tasks']) == 67
assert appended == [('NAV-067',2)]
assert sum(t['status'] for t in plan['tasks']) == 54
validator = [sys.executable,'-B',
    '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',
    str(ROOT/plan_rel),'--mode','update']
with (OUT/'plan-check.log').open('x') as log:
    subprocess.run(validator,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
write_new(OUT/'final-review.json',{
    'status':'PASS_BASIC_REVIEW_ONLY','timestamp':datetime.now().astimezone().isoformat(),
    'plan_sha256':sha(ROOT/plan_rel),'protected_hashes_checked':295,
    'all_task_definitions_statuses_dependencies_old_records_preserved':True,
    'only_appended_records':appended,'completed_tasks':54,
    'original_four_sources_manifest_and_v2_proposal_unchanged':True,
    'new_v3_sources_and_evidence_match':True,'no_approved_registration_or_recovery_outputs':True,
    'basic_tests':37,'validator_command':validator,
    'validator_output':(OUT/'plan-check.log').read_text(),
    'new_apps':0,'new_physics_steps':0,'new_navigation_episodes':0,
    'new_training_updates':0,'new_W_queries':0,'NAV_067_status':False})
paths = ['CaMRA/navigation/src/camra_navigation/envs/clean_v3.py',
    'CaMRA/navigation/scripts/nav067_simulation_v3.py',
    'CaMRA/navigation/scripts/evaluate_navigation_v4.py',
    'CaMRA/navigation/tests/test_clean_v3.py',
    'CaMRA/navigation/config/nav067_startup_recovery_proposal_v3.json',
    'CaMRA/navigation/reports/nav067_api_repair_preparation_v3.json',
    'CaMRA/navigation/docs/implementation_handoff.md',plan_rel]
evidence = ['verify.py','verify.log','verification.json','basic-tests.log','simulator-diff.txt',
    'prepare.py','prepare.log','production-unapproved-refusal.log','plan-check.log',
    'final-review.json','check_and_commit.py']
paths += [str((OUT/name).relative_to(ROOT)) for name in evidence]
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
modified = subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines()
assert set(modified) <= set(paths), modified
for path in paths:
    assert (ROOT/path).is_file(), path
    assert Path(path).suffix.lower() not in ('.tex','.bib','.pdf')
    assert not path.startswith(('manuscripts/','RC-PPO/','HiRA/','submission','docs/paper'))
subprocess.run(['git','add','-f','--',*paths],cwd=ROOT,check=True)
staged = subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
assert set(staged) == set(paths), staged
checked = [p for p in paths if Path(p).suffix not in ('.txt','.log','.jsonl','.csv')]
subprocess.run(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
    'diff','--cached','--check','--',*checked],cwd=ROOT,check=True)
subprocess.run(['git','commit','-m','fix(nav): repair rigid-body readback and direct-loop reset initialization'],cwd=ROOT,check=True)
head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert not subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).strip()
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
result = {'commit':head,'pushed':False,'files':paths,'basic_tests':37,
          'new_physics_steps':0,'new_navigation_episodes':0,'NAV_067_status':False}
write_new(OUT/'commit-result.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2))
