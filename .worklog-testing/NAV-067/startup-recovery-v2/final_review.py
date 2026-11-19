"""Basic final verification only; never create a simulation or pilot attempt."""
import ast
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from nav067_sources import NAV, ROOT, OUTPUT, read_manifest, sha, write_new

OUT = Path(__file__).resolve().parent
assert socket.gethostname().split('.')[0] == 'tiger11'
assert os.environ.get('TMUX') and os.environ.get('TMUX_PANE') == '%587'
assert sys.version_info[:2] == (3, 11)
manifest, _, _ = read_manifest()
assert manifest['binding']['protected_hashes_checked'] == 295
proposal_path = NAV/'config/nav067_startup_recovery_proposal_v2.json'
proposal = json.loads(proposal_path.read_bytes())
assert proposal['status'] == 'PROPOSED'
assert sha(proposal_path) == '241fcff15a15003e9e0f4fb621e8576b9dc441899918b55b385b5bc45cc04436'
assert proposal['original_manifest_sha256'] == sha(OUTPUT/'manifest.json')
for rel, digest in proposal['source_hashes'].items():
    assert sha(NAV/rel) == digest, rel
for path, digest in proposal['evidence_hashes'].items():
    assert sha(path) == digest, path
for name in ('scripts/nav067_runtime.py', 'scripts/evaluate_navigation_v3.py',
             'tests/test_nav067_runtime.py'):
    ast.parse((NAV/name).read_text(), filename=name)
assert not (NAV/'config/nav067_startup_recovery_v2.json').exists()
for slot in proposal['slots']:
    assert not (OUTPUT/slot['relative_output']).exists()

plan_path = ROOT/'doc/plan/20261109_camra_navigation_implementation.json'
plan = json.loads(plan_path.read_bytes())
baseline = json.loads(subprocess.check_output(
    ['git', 'show', '645275d956987dd77ef55976d982f5b8a369aeb2:'+str(plan_path.relative_to(ROOT))], cwd=ROOT))
assert list(plan) == list(baseline)
for key in plan:
    if key != 'tasks':
        assert plan[key] == baseline[key], key
assert len(plan['tasks']) == len(baseline['tasks']) == 67
appended = []
for old, new in zip(baseline['tasks'], plan['tasks']):
    assert {k:v for k,v in old.items() if k!='work_records'} == {
        k:v for k,v in new.items() if k!='work_records'}, new['id']
    records = old['work_records']
    assert new['work_records'][:len(records)] == records, new['id']
    extra = len(new['work_records'])-len(records)
    if extra:
        appended.append((new['id'],extra))
assert appended == [('NAV-067',2)], appended
assert sum(t['status'] for t in plan['tasks']) == 54
assert not next(t for t in plan['tasks'] if t['id']=='NAV-067')['status']
assert not (OUTPUT/'L1_empty/FINISHED.json').exists()

tests = [sys.executable, '-B', '-m', 'pytest', '-q',
         'CaMRA/navigation/tests/test_nav067_runtime.py',
         'CaMRA/navigation/tests/test_nav067_binding.py',
         'CaMRA/navigation/tests/test_physical_v2.py']
with (OUT/'final-basic-tests.log').open('x') as log:
    subprocess.run(tests,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
validator = [sys.executable, '-B',
    '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',
    str(plan_path), '--mode', 'update']
with (OUT/'final-plan-check.log').open('x') as log:
    subprocess.run(validator,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

changed = subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines()
assert set(changed) == {str(plan_path.relative_to(ROOT)),
                       'CaMRA/navigation/docs/implementation_handoff.md'}, changed
result = {
    'status':'PASS_BASIC_REVIEW_ONLY',
    'timestamp':datetime.now().astimezone().isoformat(),
    'host':socket.getfqdn(),'python':sys.executable,'cwd':str(ROOT),
    'tmux':subprocess.check_output(['tmux','display-message','-t',os.environ['TMUX_PANE'],
        '-p','#{session_name}:#{window_name} #{window_id} #{pane_id}'],text=True).strip(),
    'plan_version':plan['document']['version'],'plan_sha256':sha(plan_path),
    'baseline_head':'645275d956987dd77ef55976d982f5b8a369aeb2',
    'fully_parsed_tasks':67,'fully_parsed_records':sum(len(t['work_records']) for t in plan['tasks']),
    'preserved_task_order_status_dependencies_and_old_records':True,
    'only_appended_records':appended,'completed_tasks':54,
    'protected_hashes_checked':295,'original_manifest_and_four_sources_unchanged':True,
    'proposal_sha256':sha(proposal_path),'approved_amendment_absent':True,
    'recovery_output_directories_absent':True,
    'source_hashes':proposal['source_hashes'],
    'tests_command':tests,'tests_output':(OUT/'final-basic-tests.log').read_text(),
    'validator_command':validator,'validator_output':(OUT/'final-plan-check.log').read_text(),
    'new_simulation_apps':0,'new_physics_steps':0,'new_navigation_episodes':0,
    'new_training_updates':0,'new_W_queries':0,
    'limitations':['Import/config probe and basic checks do not validate physical integration.',
        'Original no-automatic-restart policy requires an explicit recovery decision.',
        'NAV-067 remains false; NAV-043 and final_* remain unopened.']}
write_new(OUT/'final-review.json', result)
print(json.dumps(result,ensure_ascii=False,indent=2))
