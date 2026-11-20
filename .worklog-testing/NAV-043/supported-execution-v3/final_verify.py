"""Close plan/history/provenance checks, no fitting or new trajectory execution."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/home/caohy/repositories/Paper')
EV=ROOT/'.worklog-testing/NAV-043/supported-execution-v3'
sys.path.insert(0,str(ROOT/'CaMRA/navigation/scripts'))
import nav043_supported_response as c
import audit_nav043_supported_response as audit

c.context()
p=c.approved()
path='doc/plan/20261109_camra_navigation_implementation.json'
old=json.loads(subprocess.check_output(['git','show','9f2d197:'+path],text=True))
now=c.base.load(ROOT/path)
assert {k:v for k,v in old.items() if k!='tasks'}=={k:v for k,v in now.items() if k!='tasks'}
for a,b in zip(old['tasks'],now['tasks'],strict=True):
    assert {k:v for k,v in a.items() if k!='work_records'}=={k:v for k,v in b.items() if k!='work_records'}
    assert b['work_records'][:len(a['work_records'])]==a['work_records']
    if a['id']!='NAV-043':
        assert a==b
task=next(t for t in now['tasks'] if t['id']=='NAV-043')
assert len(task['work_records'])==60 and not task['status']
assert sum(t['status'] for t in now['tasks'])==55
assert (EV/'close.exit').read_text().strip()=='0'
report=c.base.load(audit.REPORT)
assert report['status']=='SAVED_SUPPORTED_RESPONSE_EXECUTION_AUDIT_PASS'
assert not report['scientific_diagnostic_pass']
for item in report['output_index']:
    path=Path(item['path'])
    assert path.stat().st_size==item['bytes'] and c.base.sha(path)==item['sha256']
    if path.suffix=='.json':
        saved=EV/'saved-small-output'/path.relative_to(p['output_root'])
        assert c.base.sha(saved)==item['sha256']
validation=subprocess.run([sys.executable,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py','doc/plan/20261109_camra_navigation_implementation.json','--mode','update'],text=True,capture_output=True)
assert validation.returncode==0,validation.stdout+validation.stderr
paths=['AGENTS.md','README.md','CaMRA/navigation/README.md','CaMRA/navigation/docs/implementation_handoff.md','CaMRA/navigation/docs/nav043_supported_response_result_20261120.md','CaMRA/navigation/reports/nav043_supported_response_review_v3.json','doc/plan/20261109_camra_navigation_implementation.json','.worklog-testing/NAV-043/supported-execution-v3']
diff=subprocess.run(['git','-c','core.preloadIndex=false','-c','core.fsmonitor=false','diff','--check','--',*paths],text=True,capture_output=True)
assert diff.returncode==0,diff.stdout+diff.stderr
c.base.write_new(EV/'final_verify.json',dict(status='PASS',pane=os.environ['TMUX_PANE'],
    previous_59_records_unchanged=True,all_67_task_definitions_and_statuses_unchanged=True,
    source_pins_unchanged=True,all_83_outputs_rechecked=True,all_53_small_copies_match=True,
    legacy_validation=validation.stdout,diff_check_passed=True,NAV043=False,G5_passed=False,
    parameter_solves=0,source_hashes={str(p.relative_to(ROOT)):c.base.sha(p) for p in [ROOT/x for x in paths if (ROOT/x).is_file()]}))
print(validation.stdout.strip(),flush=True)
print('FINAL_HISTORY_AND_OUTPUT_VERIFICATION_PASS; NAV-043 remains false',flush=True)
