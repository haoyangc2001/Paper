"""Check immutable raw data, source identities, history and saved verdicts."""
from datetime import datetime
import ast
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

ROOT = Path('/home/caohy/repositories/Paper')
EV = ROOT/'.worklog-testing/NAV-043/motion-recovery-v2'
sys.path.insert(0,str(ROOT/'CaMRA/navigation/scripts'))
import nav043_motion_probe_recovery_guard as G
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
p,a,r = G.approved()
assert G.protected_sources() == {'protected_identities':295,'existing_raw_input_hashes':90,'bundle_files':29}
report_path = G.NAV/'reports/nav043_motion_probe_recovery_held_review_v2.json'
review = G.load(report_path)
assert review['case_integrity_passed'] and not review['strict_original_completion_gate_passed']
assert review['strict_original_gate_still_rejects'] and review['terminal_protocol_state'] == 'HELD'
assert len(review['cases']) == 12 and all(c['integrity_passed'] for c in review['cases'])
assert review['actual_ticks_reviewed'] == 26400 and review['actual_command_proofs_reviewed'] == 6600
assert review['completed_q_intervals_reviewed'] == 1212
assert review['cumulative_applications_started'] == 2 and review['cumulative_explicit_step_calls'] == 27295
assert review['cumulative_charged_wall_s'] < 7200 and review['further_applications_authorized'] == 0
assert not review['NAV043_passed'] and not review['G5_passed']
out = Path(p['output_root'])
index = G.load(EV/'held-raw-output-index.json')
assert len(index['files']) == 108 and index['total_bytes'] == 29543786
paths = {str(path) for path in out.rglob('*') if path.is_file()}
assert paths == {item['path'] for item in index['files']}
for item in index['files']:
    assert Path(item['path']).stat().st_size == item['bytes'] and G.sha(item['path']) == item['sha256']
byrelative = {str(Path(item['path']).relative_to(out)):item for item in index['files']}
for name in review['small_evidence_copies']:
    path = ROOT/name
    relative = str(path.relative_to(EV/'held-saved-small'))
    assert path.stat().st_size == byrelative[relative]['bytes']
    assert G.sha(path) == byrelative[relative]['sha256']
assert all(not (Path('/proc')/str(pid)).exists() for pid in review['registered_processes_absent'])
assert (out/'HELD.json').is_file() and not (out/'ERROR.json').exists()
assert all(not (out/name).exists() for name in review['missing_terminal_records_not_backfilled'])
strict = G.load(EV/'strict-review-retains-held.json')
assert strict['returncode'] == 1 and not strict['original_gate_changed']
assert (EV/'review-held-capture.exit').read_text().strip() == '1'
assert (EV/'review-held-capture-v2.exit').read_text().strip() == '0'
assert (EV/'approval-commit-v2.exit').read_text().strip() == '0'
xml = ET.parse(ROOT/r['basic_verification']['evidence']).getroot()
suites = [xml] if xml.tag == 'testsuite' else xml.findall('testsuite')
assert sum(int(s.attrib['tests']) for s in suites) == 20
assert all(int(s.attrib[k]) == 0 for s in suites for k in ('failures','errors','skipped'))
baseline = json.loads(subprocess.check_output(['git','show',
    'c8d6f71:doc/plan/20261109_camra_navigation_implementation.json'],cwd=ROOT))
current = G.load(ROOT/'doc/plan/20261109_camra_navigation_implementation.json')
assert {k:v for k,v in baseline.items() if k!='tasks'} == {k:v for k,v in current.items() if k!='tasks'}
assert len(current['tasks']) == 67 and sum(t['status'] for t in current['tasks']) == 55
byid = {t['id']:t for t in current['tasks']}
assert [t['id'] for t in current['tasks'] if not t['status'] and
    all(byid[d]['status'] for d in t['depends_on'])][0] == 'NAV-043'
for before,after in zip(baseline['tasks'],current['tasks']):
    assert {k:v for k,v in before.items() if k!='work_records'} == {k:v for k,v in after.items() if k!='work_records'}
    assert before['work_records'] == after['work_records'][:len(before['work_records'])]
    assert len(after['work_records']) == len(before['work_records']) + (2 if after['id']=='NAV-043' else 0)
for path in (EV/'inspect-held.py',EV/'review-held-capture-attempt1.py',EV/'review-held-capture.py',Path(__file__)):
    ast.parse(path.read_text(),filename=str(path))
at = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
result = {'schema':'nav043-motion-recovery-v2-close-review','status':'SAVED_DATA_AND_HISTORY_RECHECK_PASS',
    'at':at,'hostname':socket.gethostname(),'python':sys.executable,'tmux_pane':os.environ['TMUX_PANE'],
    'plan_tasks':67,'plan_historical_true':55,'NAV043_status':False,'G5_passed':False,
    'NAV043_records':len(byid['NAV-043']['work_records']),'history_prefix_preserved':True,
    'all_task_definitions_statuses_and_unknown_fields_preserved':True,
    'original_strict_protocol_status':'HELD','saved_case_integrity':'PASS',
    'original_output_files_rechecked':108,'original_output_bytes':29543786,
    'small_exact_copies_rechecked':len(review['small_evidence_copies']),
    'saved_CPU_tests_reused_without_source_change':20,'new_test_subagents':0,
    'review_report_sha256':G.sha(report_path),'raw_index_sha256':G.sha(EV/'held-raw-output-index.json'),
    'review_source_sha256':G.sha(EV/'review-held-capture.py'),
    'first_review_source_sha256':G.sha(EV/'review-held-capture-attempt1.py'),
    'approval_sha256':G.sha(G.APPROVAL),'identities':G.protected_sources(),
    'original_outputs_unchanged':True,'sealed_sources_unchanged':True,
    'new_applications':0,'new_fits':0,'new_navigation_episodes':0,'new_W_queries':0,
    'new_network_updates':0,'new_calibration_fits':0,'further_applications_authorized':0}
G.persist(EV/'close-review.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
