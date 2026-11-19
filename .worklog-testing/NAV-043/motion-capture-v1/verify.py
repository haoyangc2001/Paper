"""Preparation-only history, identity and final CPU-check review."""
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
HERE = ROOT/'.worklog-testing/NAV-043/motion-capture-v1'
sys.path.insert(0, str(NAV/'scripts'))
import nav043_affine_response as affine
base = affine.base

assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert Path(sys.executable).resolve() == Path(base.PYTHON).resolve()
affine.approval()
base.verify_protocol()
proposal = NAV/'config/nav043_motion_probe_proposal_v1.json'
assert base.sha(proposal) == '9eb3caf00523a65d4848dc07e11a8fcafab87d8a01fa6c33a0459398ba7444bb'
p = base.load(proposal)
assert p['status'] == 'PROPOSED' and not (ROOT/p['approval_path']).exists()
assert not Path(p['output_root']).exists()
for name, digest in p['source_hashes'].items():
    assert base.sha(ROOT/name) == digest
xml = ET.parse(HERE/'basic-v3.xml').getroot()
suites = [xml] if xml.tag == 'testsuite' else xml.findall('testsuite')
assert sum(int(s.attrib['tests']) for s in suites) == 23
assert all(int(s.attrib[k]) == 0 for s in suites for k in ('failures','errors','skipped'))
assert (HERE/'basic-v3.exit').read_text().strip() == '0'
old = json.loads(subprocess.check_output(['git','show',
    'fa9c2f9:doc/plan/20261109_camra_navigation_implementation.json'], cwd=ROOT))
new = base.load(ROOT/'doc/plan/20261109_camra_navigation_implementation.json')
assert {k:v for k,v in old.items() if k != 'tasks'} == {k:v for k,v in new.items() if k != 'tasks'}
assert len(old['tasks']) == len(new['tasks']) == 67
appended = 0
for before, after in zip(old['tasks'], new['tasks']):
    assert {k:v for k,v in before.items() if k != 'work_records'} == {
        k:v for k,v in after.items() if k != 'work_records'}
    n = len(before['work_records'])
    assert before['work_records'] == after['work_records'][:n]
    if after['id'] == 'NAV-043':
        appended = len(after['work_records'])-n
        assert appended == 2 and not after['status']
    else:
        assert before['work_records'] == after['work_records']
states = {t['id']:t['status'] for t in new['tasks']}
assert sum(states.values()) == 55
assert [t['id'] for t in new['tasks'] if not t['status'] and all(states[d] for d in t['depends_on'])] == ['NAV-043']
consumers = []
for directory in (NAV/'src', NAV/'scripts'):
    for path in directory.rglob('*.py'):
        if path.name != 'motion_capture_v1.py' and 'motion_capture_v1' in path.read_text():
            consumers.append(str(path.relative_to(ROOT)))
assert consumers == []
paths = [NAV/'src/camra_navigation/envs/motion_capture_v1.py', NAV/'tests/test_motion_capture.py', Path(__file__)]
base.write_new(HERE/'preparation-review.json', {
    'status':'ISOLATED_CAPTURE_CORE_PREPARATION_PASS', 'at':datetime.now().astimezone().isoformat(),
    'host':socket.gethostname(), 'pane':os.environ['TMUX_PANE'], 'python':sys.executable,
    'basic_CPU_tests_passed':23, 'production_consumers':consumers,
    'physical_launcher_implemented':False, 'real_controlled_collection_validated':False,
    'plan_baseline':'fa9c2f9', 'NAV043_appended_records':appended,
    'history_definitions_statuses_preserved':True, 'historical_true_count':55,
    'proposal_sha256':base.sha(proposal), 'APPROVED_exists':False, 'experiment_output_exists':False,
    'protected_identities_checked':295, 'existing_raw_input_hashes_checked':90,
    'source_hashes':{str(path.relative_to(ROOT)):base.sha(path) for path in paths},
    'actual_new_applications':0, 'actual_new_physics_steps':0, 'actual_new_fits':0,
    'actual_W_queries':0, 'actual_network_updates':0, 'actual_calibration_fits':0,
    'GPU_use':False, 'NAV043_passed':False, 'G5_passed':False})
print('PASS: 23 CPU checks, 295/90 identities, histories unchanged; no real launcher or new scientific execution.')
