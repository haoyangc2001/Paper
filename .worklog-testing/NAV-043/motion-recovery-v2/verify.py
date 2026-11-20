"""Saved evidence and preparation checks only; never start a simulator."""
import ast
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT=Path('/home/caohy/repositories/Paper');NAV=ROOT/'CaMRA/navigation'
sys.path.insert(0,str(NAV/'scripts'))
import nav043_motion_probe_guard as old
import nav043_motion_probe_recovery_guard as G
E=G.EVIDENCE
assert os.environ.get('TMUX') and socket.gethostname().split('.')[0]=='tiger11'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert Path(sys.executable).resolve()==Path('/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python').resolve()
p,r=G.pinned(G.PROPOSAL),G.pinned(G.READINESS)
assert p['status']=='PROPOSED' and r['status']=='PREPARED_ONLY'
assert r['proposal_sha256']==G.sha(G.PROPOSAL)
assert not G.APPROVAL.exists() and not (E/'launch.lock').exists()
assert not Path(p['output_root']).exists()
old.approved()
for group in (p['source_hashes'],r['source_hashes']):
    for name,digest in group.items():
        path=ROOT/name
        assert G.sha(path)==digest, name
        if path.suffix=='.py': ast.parse(path.read_text(),filename=str(path))
identities=G.protected_sources()
assert identities=={'protected_identities':295,'existing_raw_input_hashes':90,'bundle_files':29}
failure=G.load(NAV/'reports/nav043_motion_probe_first_failure_v1.json')
for item in failure['saved_files']:
    assert Path(item['path']).stat().st_size==item['bytes']
    assert G.sha(item['path'])==item['sha256']
original=Path(p['recovery']['original_output_root'])
owner,child=G.load(original/'OWNER.json'),G.load(original/'PROCESS.json')
assert all(not (Path('/proc')/str(pid)).exists() for pid in (owner['pid'],child['pid']))
wall_bound=(datetime.fromisoformat(failure['at'])-datetime.fromisoformat(owner['at'])).total_seconds()
assert 0<wall_bound<=900
assert p['limits']['gpu_wall_s']+900==7200
assert p['limits']['all_explicit_simulation_step_calls']+895==32000
assert p['limits']['setup_reset_reserve_step_calls']+26400==31105
refusals=[]
for script,args in [('run_nav043_motion_probe_recovery_v2.py',[]),
                    ('run_nav043_motion_probe_recovery_v2.py',['--child']),
                    ('run_nav043_motion_probe_recovery_v2.py',['--check']),
                    ('audit_nav043_motion_probe_recovery_v2.py',[])]:
    result=subprocess.run([sys.executable,'-B',str(NAV/'scripts'/script),*args],
                          capture_output=True,text=True,timeout=30)
    assert result.returncode!=0 and 'recovery APPROVED registration is missing' in result.stderr
    refusals.append({'script':script,'args':args,'exit_code':result.returncode,
                     'missing_approval_rejected':True})
assert not (E/'launch.lock').exists() and not Path(p['output_root']).exists()
baseline=json.loads(subprocess.check_output(
    ['git','show','05c52f4:doc/plan/20261109_camra_navigation_implementation.json'],cwd=ROOT))
current=G.load(ROOT/'doc/plan/20261109_camra_navigation_implementation.json')
assert {k:v for k,v in baseline.items() if k!='tasks'}=={k:v for k,v in current.items() if k!='tasks'}
for before,after in zip(baseline['tasks'],current['tasks']):
    assert {k:v for k,v in before.items() if k!='work_records'}=={k:v for k,v in after.items() if k!='work_records'}
    assert before['work_records']==after['work_records'][:len(before['work_records'])]
    assert after['id']=='NAV-043' or before['work_records']==after['work_records']
assert len(current['tasks'])==67 and sum(t['status'] for t in current['tasks'])==55
suites=ET.parse(E/'basic.xml').getroot().findall('testsuite')
assert sum(int(s.attrib['tests']) for s in suites)==20
assert all(int(s.attrib[k])==0 for s in suites for k in ('failures','errors','skipped'))
report={'schema':'nav043-motion-probe-recovery-preparation-v2','status':'PREPARED_NOT_APPROVED',
 'at':datetime.now().astimezone().isoformat(),'pane':os.environ.get('TMUX_PANE'),
 'proposal_sha256':G.sha(G.PROPOSAL),'readiness_sha256':G.sha(G.READINESS),
 'main_agent_CPU_checks_passed':20,'missing_approval_entries_rejected':refusals,
 'sealed_support_sources':len(r['source_hashes']),'identities_checked':identities,
 'original_saved_files_unchanged':len(failure['saved_files']),
 'conservative_failed_slot_bound_s':wall_bound,'failed_slot_charge_s':900,
 'history_preserved':True,'original_failure_preserved':True,
 'new_applications':0,'new_physical_steps':0,'new_fits':0,'W_queries':0,
 'network_updates':0,'calibration_fits':0,'recovery_approval_exists':False,
 'actual_recovery_chain_proven':False,'NAV043_passed':False,'G5_passed':False}
G.persist(NAV/'reports/nav043_motion_probe_recovery_preparation_v2.json',report)
print(json.dumps(report,ensure_ascii=False,indent=2))
