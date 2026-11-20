"""Final source/history and raw-index recheck, no response fitting."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/home/caohy/repositories/Paper')
sys.path.insert(0,str(ROOT/'CaMRA/navigation/scripts'))
import nav043_supported_response as c
import nav043_motion_probe_recovery_guard as g

c.context()
p=c.pin(c.PROPOSAL); ready=c.pin(c.READINESS)
for group in (p['source_hashes'],ready['source_hashes']):
    for path,digest in group.items():assert c.base.sha(path)==digest, path
assert not c.APPROVAL.exists() and not (c.HERE/'launch.lock').exists() and not Path(p['output_root']).exists()
identities=g.protected_sources()
assert identities==dict(protected_identities=295,existing_raw_input_hashes=90,bundle_files=29)
raw=c.base.load(ROOT/'.worklog-testing/NAV-043/motion-recovery-v2/held-raw-output-index.json')
for item in raw['files']:assert c.base.sha(item['path'])==item['sha256']
planpath='doc/plan/20261109_camra_navigation_implementation.json'
old=json.loads(subprocess.check_output(['git','show','3d4db99:'+planpath],cwd=ROOT,text=True))
now=c.base.load(ROOT/planpath)
for a,b in zip(old['tasks'],now['tasks'],strict=True):
    assert {k:v for k,v in a.items() if k!='work_records'}=={k:v for k,v in b.items() if k!='work_records'}
    assert b['work_records'][:len(a['work_records'])]==a['work_records']
    if a['id']!='NAV-043':assert a==b
assert {k:v for k,v in old.items() if k!='tasks'}=={k:v for k,v in now.items() if k!='tasks'}
assert len(next(t for t in now['tasks'] if t['id']=='NAV-043')['work_records'])==55
assert sum(t['status'] for t in now['tasks'])==55
review=c.base.load(c.base.NAV/'reports/nav043_motion_probe_recovery_held_review_v2.json')
assert not review['strict_original_completion_gate_passed'] and review['case_integrity_passed']
c.base.write_new(c.HERE/'close.json',dict(status='PREPARED_ONLY_CLOSE_PASS',
    pane=os.environ['TMUX_PANE'], source_commit='3d4db99', identities=identities,
    raw_files=108, task_definitions_and_statuses_unchanged=67,
    NAV043_history_prefix_records_unchanged=53, NAV043_current_records=55,
    original_HELD_preserved=True, production_enabled=False,
    applications=0, actual_fits=0, W_queries=0, updates=0,
    files={str(path.relative_to(ROOT)):c.base.sha(path) for path in
           [c.PROPOSAL,c.READINESS,ROOT/planpath,c.HERE/'verify.json',
            c.base.NAV/'docs/nav043_supported_response_preparation_20261120.md']}))
print('PREPARED_ONLY_CLOSE_PASS')
