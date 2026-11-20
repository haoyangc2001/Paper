"""Read-only exact recovery approval preflight; never create an application."""
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path('/home/caohy/repositories/Paper')
sys.path.insert(0,str(ROOT/'CaMRA/navigation/scripts'))
import nav043_motion_probe_recovery_guard as G

p,a,r=G.approved()
context=G.registered_context(p)
G.refuse_conflicts(p,ROOT/'CaMRA/navigation/scripts/run_nav043_motion_probe_recovery_v2.py')
identities=G.protected_sources()
assert identities=={'protected_identities':295,'existing_raw_input_hashes':90,'bundle_files':29}
xml=ET.parse(ROOT/r['basic_verification']['evidence']).getroot()
suites=[xml] if xml.tag=='testsuite' else xml.findall('testsuite')
assert sum(int(s.attrib['tests']) for s in suites)==20
assert all(int(s.attrib[k])==0 for s in suites for k in ('failures','errors','skipped'))
assert (G.EVIDENCE/'basic.exit').read_text().strip()=='0'
failure=G.load(ROOT/'CaMRA/navigation/reports/nav043_motion_probe_first_failure_v1.json')
for item in failure['saved_files']:
    assert Path(item['path']).stat().st_size==item['bytes']
    assert G.sha(item['path'])==item['sha256']
baseline=json.loads(subprocess.check_output(
    ['git','show','2de0b21:doc/plan/20261109_camra_navigation_implementation.json'],cwd=ROOT))
current=G.load(ROOT/'doc/plan/20261109_camra_navigation_implementation.json')
assert {k:v for k,v in baseline.items() if k!='tasks'}=={k:v for k,v in current.items() if k!='tasks'}
assert len(baseline['tasks'])==len(current['tasks'])==67
for before,after in zip(baseline['tasks'],current['tasks']):
    assert {k:v for k,v in before.items() if k!='work_records'}=={k:v for k,v in after.items() if k!='work_records'}
    assert before['work_records']==after['work_records'][:len(before['work_records'])]
    assert len(after['work_records'])==len(before['work_records'])+int(after['id']=='NAV-043')
gpu=G.gpu_ready(p)
report={
    'schema':'nav043-motion-probe-recovery-approved-preflight-v2',
    'status':'APPROVED_PREFLIGHT_PASS',
    'at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    'context':context,'identities':identities,'gpu':gpu,
    'proposal_sha256':G.sha(G.PROPOSAL),'approval_sha256':G.sha(G.APPROVAL),
    'readiness_sha256':G.sha(G.READINESS),'saved_CPU_checks_reused':20,
    'all_sealed_sources_unchanged':True,'all_history_preserved':True,
    'original_saved_files_unchanged':len(failure['saved_files']),
    'applications_started':0,'physical_steps':0,'NAV043_passed':False,'G5_passed':False}
G.persist(G.EVIDENCE/'approval-preflight.json',report)
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
