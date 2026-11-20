"""Verify frozen preparation and actual refusal; no real fit or new episodes."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT=Path('/home/caohy/repositories/Paper')
EV=ROOT/'.worklog-testing/NAV-043/coupled-response-v4'
sys.path.insert(0,str(ROOT/'CaMRA/navigation/scripts'))
import run_nav043_coupled_response as r
import nav043_motion_probe_recovery_guard as g
import audit_nav043_coupled_response as audit

r.context()
p=r.core.old.pin(r.PROPOSAL)
ready=r.core.old.pin(r.READINESS)
assert p['status']=='PROPOSED' and ready['status']=='PREPARED_ONLY'
assert ready['proposal_sha256']==r.base.sha(r.PROPOSAL) and p['limits']==r.LIMITS
for group in (p['source_hashes'],ready['source_hashes']):
    for path,digest in group.items():
        assert r.base.sha(path)==digest,path
assert (EV/'support.exit').read_text().strip()=='0'
assert (EV/'final-basic.exit').read_text().strip()=='0'
assert 'Ran 11 tests' in (EV/'final-basic.log').read_text()
assert '\nOK\n' in (EV/'final-basic.log').read_text()
assert not r.APPROVAL.exists() and not Path(p['output_root']).exists()
assert not (EV/'launch.lock').exists() and not audit.REPORT.exists()
identities=g.protected_sources()
assert identities==dict(protected_identities=295,existing_raw_input_hashes=90,bundle_files=29)
cases=r.load_cases()
reference=r.reference_rows()
assert len(cases)==30 and len(reference)==30
before={str(path):r.base.sha(path) for path in EV.iterdir() if path.is_file()}
refusals=[]
for name in ('launch_nav043_coupled_response_once.py','audit_nav043_coupled_response.py'):
    result=subprocess.run([r.base.PYTHON,'-B',str(r.base.NAV/'scripts'/name)],cwd=ROOT,
                          env=os.environ.copy(),text=True,capture_output=True)
    assert result.returncode==1 and 'require separate APPROVED' in result.stderr
    refusals.append(dict(entry=name,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
assert before=={str(path):r.base.sha(path) for path in EV.iterdir() if path.is_file()}
assert not Path(p['output_root']).exists() and not (EV/'launch.lock').exists() and not audit.REPORT.exists()
path='doc/plan/20261109_camra_navigation_implementation.json'
old=json.loads(subprocess.check_output(['git','show','a59b0e2:'+path],text=True))
now=r.base.load(ROOT/path)
assert {k:v for k,v in old.items() if k!='tasks'}=={k:v for k,v in now.items() if k!='tasks'}
for a,b in zip(old['tasks'],now['tasks'],strict=True):
    assert {k:v for k,v in a.items() if k!='work_records'}=={k:v for k,v in b.items() if k!='work_records'}
    assert b['work_records'][:len(a['work_records'])]==a['work_records']
    if a['id']!='NAV-043':
        assert a==b
task=next(t for t in now['tasks'] if t['id']=='NAV-043')
assert not task['status'] and len(task['work_records'])==61
assert sum(t['status'] for t in now['tasks'])==55
consumers=subprocess.run(['rg','-l','nav043_coupled_response',str(r.base.NAV/'src')],text=True,capture_output=True)
assert consumers.returncode==1 and not consumers.stdout
validation=subprocess.run([sys.executable,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',path,'--mode','update'],text=True,capture_output=True)
assert validation.returncode==0,validation.stdout+validation.stderr
r.base.write_new(EV/'verify.json',dict(status='PREPARED_ONLY_VERIFIED',host=socket.gethostname(),
    pane=os.environ['TMUX_PANE'],python=sys.executable,proposal_sha256=r.base.sha(r.PROPOSAL),
    readiness_sha256=r.base.sha(r.READINESS),identities=identities,cases=30,measurement_ticks=182797,
    old_60_records_and_all_67_definitions_statuses_preserved=True,refusals=refusals,
    actual_parameter_solves=0,synthetic_axis_solves_total=6,synthetic_known_solution_witnesses=9,
    main_agent_final_basic_checks=11,production_consumer=False,original_v3_FAIL_and_physical_HELD_preserved=True,
    applications=0,new_episodes=0,W_queries=0,network_updates=0,calibration_fits=0,
    NAV043=False,G5_passed=False,plan_true=55,validation=validation.stdout,
    source_hashes={str(path.relative_to(ROOT)):r.base.sha(path) for path in (EV/'verify.py',EV/'verify.sh')}))
print('PREPARED_ONLY_VERIFIED; actual solves0; exact new fit scope still unapproved',flush=True)
