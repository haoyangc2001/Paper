"""Final read-only evidence/plan audit; no fitting or new trajectory prediction."""
from pathlib import Path
import ast
import json
import os
import subprocess
import sys

ROOT=Path('/home/caohy/repositories/Paper')
sys.path.insert(0,str(ROOT/'CaMRA/navigation/scripts'))
import nav043_matched_support_v5 as m
import run_nav043_coupled_response as old
import nav043_motion_probe_recovery_guard as guard

m.context()
ev=ROOT/'.worklog-testing/NAV-043/v4-support-review'
assert m.base.sha(m.PROPOSAL)=='71d5a31efe3890a713681d3e95600e87541563197e698c3a32446331a0e178d6'
assert m.base.sha(m.READINESS)=='364fe158b8bacedde45681137f8b4bb9e050560ead4ff5983a9962f8f1c0e982'
p,ready=[m.diagonal.pin(path) for path in (m.PROPOSAL,m.READINESS)]
assert p['status']=='PROPOSED' and ready['status']=='PREPARED_ONLY'
assert p['limits']==m.LIMITS and ready['proposal_sha256']==m.base.sha(m.PROPOSAL)
for mapping in (p['source_hashes'],ready['source_hashes']):
    for path,digest in mapping.items():
        assert m.base.sha(path)==digest, path
assert not m.APPROVAL.exists() and not Path(str(m.APPROVAL)+'.sha256').exists()
assert not Path(p['output_root']).exists() and not (m.HERE/'launch.lock').exists() and not m.REVIEW.exists()
for name in ('preflight.json','preflight.log','preflight.exit','launch.json','exit.json','run.log','wrapper.log'):
    assert not (m.HERE/name).exists(),name
assert ready['basic_tests']==11 and ready['coefficient_solves_during_tests']==0
assert len(ready['rank_checks'])==18
assert all(r['rank']==r['columns'] and r['condition']<=1.e8 for r in ready['rank_checks'])
assert all(r['exit_code']==1 and 'separate APPROVED' in r['stderr'] for r in ready['missing_approval_denials'])
for root,name,count in ((ev,'basic',6),(m.HERE,'basic',7),(m.HERE,'launch-basic',4)):
    assert (root/(name+'.exit')).read_text().strip()=='0'
    assert f'Ran {count} tests' in (root/(name+'.log')).read_text()
assert (ev/'analyze.exit').read_text().strip()=='0'
support=m.base.load(m.base.NAV/'reports/nav043_v4_support_review_v1.json')
assert support['status']=='SAVED_SUPPORT_AND_ERROR_ACCOUNTING_PASS' and len(support['cases'])==30
assert all(value==0 for value in support['scope'].values())
assert support['source_code_sha256']==m.base.sha(m.base.NAV/'scripts/analyze_nav043_v4_support.py')
probe=[r for r in support['cases'] if r['cohort']=='probe']
nav=[r for r in support['cases'] if r['cohort']=='navigation']
assert len(probe)==12 and len(nav)==18
assert sum(r['command_support']['outside_axis_box_count'] for r in probe)==600
assert sum(r['command_support']['held_intervals'] for r in probe)==1200
assert sum(r['command_support']['outside_axis_box_count'] for r in nav)==103
assert sum(r['command_support']['held_intervals'] for r in nav)==7921
assert sum(r['command_support']['outside_axis_box_count']>0 for r in probe)==6
assert max(r['response_error']['recurrence_closure_max'] for r in support['cases'])<1.e-11
assert max(abs(r['terminal_yaw_accounting_closure_rad']) for r in support['cases'])<1.e-9
identities=guard.protected_sources()
assert identities==support['identities']==ready['identities']==dict(protected_identities=295,existing_raw_input_hashes=90,bundle_files=29)
old.approved()
source_path=m.base.NAV/'reports/nav043_coupled_response_review_v4.json'
assert m.base.sha(source_path)==support['source_report_sha256']
source=m.base.load(source_path)
assert not source['scientific_diagnostic_pass'] and sum(source['diagnostic_gate']['checks'].values())==1
for item in source['output_index']:
    path=Path(item['path'])
    assert path.stat().st_size==item['bytes'] and m.base.sha(path)==item['sha256']
for key in ('wrapper_pid','child_pid'):
    assert not Path('/proc',str(source['launch'][key])).exists()
path='doc/plan/20261109_camra_navigation_implementation.json'
prior=json.loads(subprocess.check_output(['git','show','f187b6f:'+path],cwd=ROOT,text=True))
plan=m.base.load(ROOT/path)
assert {k:v for k,v in prior.items() if k!='tasks'}=={k:v for k,v in plan.items() if k!='tasks'}
for a,b in zip(prior['tasks'],plan['tasks'],strict=True):
    assert {k:v for k,v in a.items() if k!='work_records'}=={k:v for k,v in b.items() if k!='work_records'}
    assert b['work_records'][:len(a['work_records'])]==a['work_records']
    if a['id']=='NAV-043':
        assert len(a['work_records'])==66 and len(b['work_records'])==68
    else:
        assert a==b
states={t['id']:t['status'] for t in plan['tasks']}
assert len(states)==67 and sum(states.values())==55
assert [t['id'] for t in plan['tasks'] if not t['status'] and all(states[d] for d in t['depends_on'])]==['NAV-043']
validation=subprocess.run([sys.executable,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',path,'--mode','update'],
    cwd=ROOT,text=True,capture_output=True)
assert validation.returncode==0,validation.stdout+validation.stderr
subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
changed=subprocess.check_output(['git','diff','--name-only','f187b6f'],cwd=ROOT,text=True).splitlines()
assert not any(p.startswith(('manuscripts/','RC-PPO/','HiRA/','CaMRA/locomotion/','CaMRA/calibration/')) or p.endswith(('.tex','.bib')) for p in changed)
for folder in (ev,m.HERE):
    for script in folder.glob('*.sh'):
        subprocess.run(['bash','-n',str(script)],check=True,capture_output=True)
    for script in folder.glob('*.py'):
        ast.parse(script.read_text(),filename=str(script))
for script in ('analyze_nav043_v4_support.py','nav043_matched_support_v5.py'):
    ast.parse((m.base.NAV/'scripts'/script).read_text(),filename=script)
# Search existing Python sources. An analyzed file need not mention its own name;
# the requirement is absence of consumers outside the two isolated diagnostics.
all_hits=subprocess.check_output(['rg','-l','nav043_matched_support_v5|analyze_nav043_v4_support',
                                 str(m.base.NAV),'-g','*.py'],cwd=ROOT,text=True).splitlines()
allowed={str(m.base.NAV/'scripts'/name) for name in ('nav043_matched_support_v5.py','analyze_nav043_v4_support.py')}
assert set(all_hits).issubset(allowed)
assert (ev/'final-verify.exit').read_text().strip()=='1'
assert 'AssertionError' in (ev/'final-verify.log').read_text()
assert (ev/'final-verify-v1-failed-source.py').is_file()
m.base.write_new(ev/'final-verify.json',dict(status='SAVED_REVIEW_AND_MATCHED_PREPARATION_PASS',
    pane=os.environ['TMUX_PANE'],identities=identities,original_output_files=83,
    saved_cases=30,basic_checks=17,extra_real_and_synthetic_coefficient_solves=0,
    proposal_sha256=m.base.sha(m.PROPOSAL),readiness_sha256=m.base.sha(m.READINESS),
    proposed_only_axis_solves=18,actual_new_fits=0,formal_output_present=False,APPROVED_present=False,
    original_66_NAV043_records_preserved=True,current_NAV043_records=68,all_other_tasks_unchanged=True,
    plan_true=55,NAV043=False,G5_passed=False,old_diagnostic_fail_preserved=True,
    no_production_consumer=True,validation=validation.stdout,
    original_verifier_false_positive_retained=True,
    original_verifier_source_sha256=m.base.sha(ev/'final-verify-v1-failed-source.py'),
    source_hashes={str(path.relative_to(ROOT)):m.base.sha(path) for path in
        (ev/'final-verify.py',ev/'final-verify-v2.sh',m.base.NAV/'docs/nav043_v4_support_review_20261120.md')}))
print('SAVED_REVIEW_AND_MATCHED_PREPARATION_PASS; 55/67, NAV043=false, new fits 0',flush=True)
