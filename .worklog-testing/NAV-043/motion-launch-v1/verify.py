"""Preparation review only: no application, simulation, fitting or GPU scan."""
import ast
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT/'CaMRA/navigation'
HERE = ROOT/'.worklog-testing/NAV-043/motion-launch-v1'
sys.path.insert(0, str(NAV/'scripts'))
import nav043_motion_probe_guard as G

assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert Path(sys.executable).resolve() == Path('/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python').resolve()
p = G.pinned(G.PROPOSAL)
assert G.sha(G.PROPOSAL) == G.PROPOSAL_PIN and p['status'] == 'PROPOSED'
assert not G.APPROVAL.exists() and not Path(p['output_root']).exists()
assert not (HERE/'launch.lock').exists()
r = G.pinned(G.READINESS)
assert r['status'] == 'PREPARED_ONLY' and r['proposal_sha256'] == G.PROPOSAL_PIN
for sources in (p['source_hashes'], r['source_hashes']):
    for name, digest in sources.items():
        assert G.sha(ROOT/name) == digest, name
        if name.endswith('.py'):
            ast.parse((ROOT/name).read_text())
identities = G.protected_sources()
assert identities == {'protected_identities':295, 'existing_raw_input_hashes':90, 'bundle_files':29}
xml = ET.parse(HERE/'basic-v3.xml').getroot()
suites = [xml] if xml.tag == 'testsuite' else xml.findall('testsuite')
assert sum(int(s.attrib['tests']) for s in suites) == 54
assert all(int(s.attrib[k]) == 0 for s in suites for k in ('failures','errors','skipped'))
assert (HERE/'basic-v3.exit').read_text().strip() == '0'
refusals = []
for arguments in (['--check'], ['--child'], []):
    proc = subprocess.run([sys.executable,'-B',str(NAV/'scripts/run_nav043_motion_probe.py'),*arguments],
                          cwd=ROOT,capture_output=True,text=True)
    assert proc.returncode == 1 and 'APPROVED registration is missing' in proc.stderr
    refusals.append({'arguments':arguments,'exit_code':proc.returncode,
                     'stdout':proc.stdout,'stderr':proc.stderr})
assert not G.APPROVAL.exists() and not Path(p['output_root']).exists()
assert not (HERE/'launch.lock').exists()
old = json.loads(subprocess.check_output(['git','show',
    '5ff80ea:doc/plan/20261109_camra_navigation_implementation.json'],cwd=ROOT))
new = G.load(ROOT/'doc/plan/20261109_camra_navigation_implementation.json')
assert {k:v for k,v in old.items() if k != 'tasks'} == {k:v for k,v in new.items() if k != 'tasks'}
assert len(old['tasks']) == len(new['tasks']) == 67
appended = 0
for before, after in zip(old['tasks'],new['tasks']):
    assert {k:v for k,v in before.items() if k != 'work_records'} == {
        k:v for k,v in after.items() if k != 'work_records'}
    count = len(before['work_records'])
    assert before['work_records'] == after['work_records'][:count]
    if after['id'] == 'NAV-043':
        appended = len(after['work_records'])-count
        assert appended == 2 and not after['status']
    else:
        assert before['work_records'] == after['work_records']
states = {t['id']:t['status'] for t in new['tasks']}
assert sum(states.values()) == 55
assert [t['id'] for t in new['tasks'] if not t['status']
        and all(states[d] for d in t['depends_on'])] == ['NAV-043']
G.persist(HERE/'preparation-review.json', {
    'status':'ISOLATED_LAUNCHER_AND_SAVED_REVIEW_PREPARATION_PASS',
    'at':datetime.now().astimezone().isoformat(),'host':socket.gethostname(),
    'pane':os.environ.get('TMUX_PANE'),'python':sys.executable,
    'final_CPU_synthetic_checks':54,'identities_checked':identities,
    'source_files_sealed':len(r['source_hashes']), 'proposal_sha256':G.sha(G.PROPOSAL),
    'readiness_sha256':G.sha(G.READINESS), 'APPROVED_exists':False,
    'actual_output_exists':False, 'persistent_launch_lock_exists':False,
    'actual_missing_approval_entry_refusals':refusals,
    'history_baseline':'5ff80ea', 'NAV043_appended_records':appended,
    'historical_definitions_statuses_records_preserved':True,'historical_true_count':55,
    'actual_new_applications':0,'actual_new_physics_steps':0,'actual_new_fits':0,
    'actual_W_queries':0,'actual_network_updates':0,'actual_calibration_fits':0,
    'GPU_use':False,'actual_installed_physical_collection_verified':False,
    'NAV043_passed':False,'G5_passed':False})
print('PREPARATION_PASS 54 CPU checks; 295/90/29 identities; 3 missing-approval refusals; no physics/GPU')
