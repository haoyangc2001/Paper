"""Exact approved scope and saved-test reuse, without starting an application."""
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path('/home/caohy/repositories/Paper')
sys.path.insert(0, str(ROOT/'CaMRA/navigation/scripts'))
import nav043_motion_probe_guard as G

p, a, r = G.approved()
context = G.registered_context(p)
G.refuse_conflicts(p, ROOT/'CaMRA/navigation/scripts/run_nav043_motion_probe.py')
identities = G.protected_sources()
assert identities == {'protected_identities':295,'existing_raw_input_hashes':90,'bundle_files':29}
xml = ET.parse(ROOT/r['basic_verification']['evidence']).getroot()
suites = [xml] if xml.tag == 'testsuite' else xml.findall('testsuite')
assert sum(int(s.attrib['tests']) for s in suites) == 54
assert all(int(s.attrib[k]) == 0 for s in suites for k in ('failures','errors','skipped'))
assert (G.EVIDENCE/'basic-v3.exit').read_text().strip() == '0'
old = json.loads(subprocess.check_output(['git','show','1f7d58f:doc/plan/20261109_camra_navigation_implementation.json'],cwd=ROOT))
new = G.load(ROOT/'doc/plan/20261109_camra_navigation_implementation.json')
assert {k:v for k,v in old.items() if k != 'tasks'} == {k:v for k,v in new.items() if k != 'tasks'}
assert len(old['tasks']) == len(new['tasks']) == 67
for before, after in zip(old['tasks'],new['tasks']):
    assert {k:v for k,v in before.items() if k != 'work_records'} == {k:v for k,v in after.items() if k != 'work_records'}
    assert before['work_records'] == after['work_records'][:len(before['work_records'])]
    assert len(after['work_records']) == len(before['work_records']) + int(after['id']=='NAV-043')
gpu = G.gpu_ready(p)
G.persist(G.EVIDENCE/'approval-preflight.json', {
    'status':'APPROVED_PREFLIGHT_PASS','context':context,'identities':identities,'gpu':gpu,
    'proposal_sha256':G.sha(G.PROPOSAL),'approval_sha256':G.sha(G.APPROVAL),
    'readiness_sha256':G.sha(G.READINESS),'saved_CPU_checks_reused':54,
    'all_sealed_sources_unchanged':True,'all_history_preserved':True,
    'applications_started':0,'physical_steps':0,'NAV043_passed':False,'G5_passed':False})
print('APPROVED_PREFLIGHT_PASS',context,identities,gpu,flush=True)
