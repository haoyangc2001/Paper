"""Verify preserved result identities/history without another fit or replay."""
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path('/home/caohy/repositories/Paper')
sys.path.insert(0, str(ROOT/'CaMRA/navigation/scripts'))
import nav043_affine_response as affine
import audit_nav043_affine_response as auditor

base, here = affine.base, auditor.HERE
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert Path(sys.executable).resolve() == Path(base.PYTHON).resolve()
proposal, approval = affine.approval()
protocol = base.verify_protocol()
readiness_path = base.NAV/'config/nav043_affine_response_readiness_v2.json'
assert base.sha(readiness_path) == Path(str(readiness_path)+'.sha256').read_text().strip()
readiness = base.load(readiness_path)
for path, digest in readiness['source_hashes'].items():
    assert base.sha(path) == digest
plan_path = 'doc/plan/20261109_camra_navigation_implementation.json'
old = json.loads(subprocess.check_output(['git', 'show', '0734784:'+plan_path], cwd=ROOT))
new = base.load(ROOT/plan_path)
assert {k:v for k,v in old.items() if k != 'tasks'} == {k:v for k,v in new.items() if k != 'tasks'}
assert len(new['tasks']) == len(old['tasks']) == 67
for before, after in zip(old['tasks'], new['tasks']):
    assert {k:v for k,v in before.items() if k != 'work_records'} == {k:v for k,v in after.items() if k != 'work_records'}
    assert after['work_records'][:len(before['work_records'])] == before['work_records']
    assert len(after['work_records'])-len(before['work_records']) == (1 if before['id'] == 'NAV-043' else 0)
assert sum(t['status'] for t in new['tasks']) == 55
states = {t['id']:t['status'] for t in new['tasks']}
first = next(t for t in new['tasks'] if not t['status'] and all(states[d] for d in t['depends_on']))
assert first['id'] == 'NAV-043' and first['verification_result'] == ''
output = Path(proposal['output_root'])
assert (output/'FINISHED.json').is_file() and not (output/'HELD.json').exists()
result = base.load(output/'FINISHED.json')
report, summary = base.load(auditor.REPORT), base.load(here/'result-summary.json')
assert report['status'] == 'SAVED_AFFINE_EXECUTION_AUDIT_PASS'
assert result['gate'] == report['gate'] and report['gate']['diagnostic_pass']
assert len(result['solves']) == 9 and len(result['rows']) == 18
assert len(report['normal_equation_checks']) == 9 and report['extra_verification_fits'] == 0
assert summary['saved_ticks'] == report['saved_measurement_ticks'] == 158797
assert report['NAV043_passed'] is False and report['G5_passed'] is False
for key in ('new_physical_episodes', 'new_nominal_episodes', 'W_queries', 'actor_critic_updates', 'calibration_fits'):
    assert result[key] == 0
assert (here/'audit-saved.exit').read_text().strip() == '0'
assert (here/'preserve-result.exit').read_text().strip() == '0'
assert report['exit'] == base.load(here/'exit.json')
assert (here/'launch.lock').read_text().strip() == str(report['launch']['wrapper_pid'])
for key in ('wrapper_pid', 'probe_pid'):
    assert not Path('/proc', str(report['launch'][key])).exists(), 'original process remains alive'
assert {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()} == auditor.expected_files(protocol)
assert len(report['output_index']) == len(summary['copied_files']) == 41
for item, copy in zip(report['output_index'], summary['copied_files']):
    assert item['path'] == copy['original'] and item['sha256'] == copy['sha256']
    assert base.sha(item['path']) == base.sha(ROOT/copy['copy']) == item['sha256']
    assert Path(item['path']).stat().st_size == (ROOT/copy['copy']).stat().st_size == item['bytes']
suites = ET.parse(here/'final-basic.xml').getroot()
assert sum(int(s.get('tests', 0)) for s in suites.iter('testsuite')) == 20
assert sum(int(s.get('failures', 0))+int(s.get('errors', 0)) for s in suites.iter('testsuite')) == 0
base.write_new(here/'close-result.json', {
    'at': datetime.now().astimezone().isoformat(), 'host': socket.gethostname(),
    'pane': os.environ['TMUX_PANE'], 'python': sys.executable,
    'status': 'RESULT_IDENTITIES_HISTORY_AND_COPY_CHECK_PASS',
    'base_commit': '0734784', 'protected_identities_checked': 295,
    'raw_input_hashes_checked': 90, 'exact_output_files': 41, 'byte_identical_copies': 41,
    'unchanged_basic_checks_passed': 20, 'actual_axis_solves': 9,
    'new_closing_solves_or_replays': 0, 'old_processes_exited': True,
    'unchanged_task_definitions': 67, 'historical_true': 55,
    'appended_NAV043_records': 1, 'report_sha256': base.sha(auditor.REPORT),
    'output_index': report['output_index'], 'NAV043_passed': False, 'G5_passed': False})
print('295 protected/90 inputs/source pins/history/41 outputs+copies PASS; no fit or replay')
