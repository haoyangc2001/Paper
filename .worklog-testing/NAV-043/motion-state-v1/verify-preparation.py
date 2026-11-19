"""Read-only closure checks; no fits, replay, simulator or policy calls."""
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
EVIDENCE = ROOT/'.worklog-testing/NAV-043/motion-state-v1'
sys.path.insert(0, str(NAV/'scripts'))
import nav043_affine_response as affine
from camra_navigation import motion_state as M

base = affine.base
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert Path(sys.executable).resolve() == Path(base.PYTHON).resolve()
affine.approval()
protocol = base.verify_protocol()
before = json.loads(subprocess.check_output(['git','show',
    '10d9259:doc/plan/20261109_camra_navigation_implementation.json'],cwd=ROOT))
current = base.load(ROOT/'doc/plan/20261109_camra_navigation_implementation.json')
assert {k:v for k,v in before.items() if k!='tasks'} == {
    k:v for k,v in current.items() if k!='tasks'}
assert len(current['tasks']) == 67 and sum(t['status'] for t in current['tasks']) == 55
additions = 0
for original, now in zip(before['tasks'], current['tasks']):
    assert {k:v for k,v in original.items() if k!='work_records'} == {
        k:v for k,v in now.items() if k!='work_records'}
    n = len(original['work_records'])
    assert original['work_records'] == now['work_records'][:n]
    if original['id'] == 'NAV-043':
        additions = len(now['work_records'])-n
        assert additions >= 1 and not now['status']
    else:
        assert original['work_records'] == now['work_records']
statuses = {t['id']:t['status'] for t in current['tasks']}
eligible = [t['id'] for t in current['tasks'] if not t['status'] and
            all(statuses[d] for d in t['depends_on'])]
assert eligible == ['NAV-043']

contract_path = NAV/'config/motion_state_candidate_v1.json'
contract = base.load(contract_path)
assert contract['status'] == 'PREPARED_ONLY' and not contract['production_enabled']
assert contract['state_version'] == M.STATE_VERSION
assert contract['observation_version'] == M.OBSERVATION_VERSION
assert contract['time_profile'] == M.PROFILE.profile_id and contract['t_max_s'] == 90
assert base.sha(ROOT/contract['time_profile_path']) == contract['time_profile_sha256']
support = contract['saved_support_review']
assert base.sha(ROOT/support['path']) == support['sha256']
review = base.load(ROOT/support['path'])
assert review['trace_count'] == 18 and review['interval_commands_including_partial_tails'] == 7944
assert review['command_quantiles'][0][0] == support['forward_command_range_m_s'][0]
assert review['command_quantiles'][-1][0] == support['forward_command_range_m_s'][1]
assert review['exact_zero_command_intervals'] == 0
assert review['near_zero_command_intervals_max_abs_001'] == 0
assert review['forward_command_abs_below_01_intervals'] == 0
assert all(v == 0 for v in contract['new_scientific_allocation'].values())
assert not contract['NAV043_passed'] and not contract['G5_passed']

xml = ET.parse(EVIDENCE/'basic-v4.xml').getroot()
suites = [xml] if xml.tag == 'testsuite' else xml.findall('testsuite')
assert sum(int(x.attrib['tests']) for x in suites) == 22
assert all(int(x.attrib[k]) == 0 for x in suites for k in ('failures','errors','skipped'))
assert (EVIDENCE/'basic-v4.exit').read_text().strip() == '0'
assert (EVIDENCE/'support.exit').read_text().strip() == '0'
consumers = []
for directory in (NAV/'src', NAV/'scripts'):
    for path in directory.rglob('*.py'):
        if path == NAV/'src/camra_navigation/motion_state.py':
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                targets = [node.module or ''] + [a.name for a in node.names]
            elif isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            else:
                continue
            if any('motion_state' in name.split('.') for name in targets):
                consumers.append(str(path.relative_to(ROOT)))
assert consumers == [], consumers
subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
sources = [NAV/'src/camra_navigation/motion_state.py', NAV/'tests/test_motion_state.py',
           contract_path, EVIDENCE/'review-affine-support.py', Path(__file__)]
base.write_new(EVIDENCE/'preparation-review.json', {
    'status':'ISOLATED_PREPARATION_CHECKS_PASS',
    'at':datetime.now().astimezone().isoformat(), 'host':socket.gethostname(),
    'pane':os.environ['TMUX_PANE'], 'python':sys.executable,
    'plan_baseline':'10d9259', 'historical_records_preserved':True,
    'task_definitions_statuses_and_top_level_preserved':True,
    'NAV043_appended_records':additions, 'historical_true_count':55,
    'first_eligible_false':'NAV-043', 'basic_tests_passed':22,
    'protected_identities_checked':295, 'raw_input_hashes_checked':90,
    'production_consumers':consumers, 'production_integration_validated':False,
    'source_hashes':{str(p.relative_to(ROOT)):base.sha(p) for p in sources},
    'new_fits':0,'new_replays':0,'new_business_episodes':0,'W_queries':0,
    'network_updates':0,'calibration_fits':0,'GPU_use':False,
    'NAV043_passed':False,'G5_passed':False,
    'remaining':'Real warmup, frozen domain-valid dynamics, reference program/J costs, independent ninety-second residual and downstream revalidation.'})
print('PASS: 22 basic checks; 295 protected / 90 raw hashes; original 67 definitions and histories retained; no production consumer or scientific execution.')
