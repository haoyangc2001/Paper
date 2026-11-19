"""Main-agent basic checks of the real saved-data audit and append-only scope."""
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
NAV = ROOT / 'CaMRA/navigation'
HERE = Path(__file__).resolve().parent
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
load = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
report_path = NAV / 'reports/nav043_saved_applicability_audit_v1.json'
report = load(report_path)
assert report['protected_identities_checked'] == 295
assert report['status'] == 'SAVED_EVIDENCE_DIAGNOSTIC_NOT_APPLICABILITY_ACCEPTANCE'
assert len(report['rows']) == 12
assert not report['NAV043_passed'] and not report['G5_passed']
assert all(report[k] == 0 for k in ('new_applications', 'new_physical_episodes', 'new_nominal_episodes', 'W_queries', 'training_updates'))
assert not report['final_data_access']
source_count = 0
for name, digest in report['source_hashes'].items():
    assert sha(name) == digest, name
    source_count += 1
for row in report['rows']:
    for name, digest in row['source_hashes'].items():
        assert sha(name) == digest, name
        source_count += 1
    assert row['saved_trace_audit'] == 'PASS_SAVED_TRACE_ONLY'
    assert row['reconstructed_reference_max_abs_error'] < 1.e-9
    assert row['budget_mode'] == 'common_budget' and not row['independent_performance_evidence']
    assert row['commands']['outside_frozen_calibration_command_domain_time_share'] == 0.
    assert row['commands']['outside_old_rule_table_domain_time_share'] == 1.
    if row['method'] == 'full':
        assert row['terminal']['cost_covered_by_original_alpha']
        assert not row['terminal']['position_covered_by_original_delta_on_10Hz']
        assert all(p['cost_covered_by_original_alpha'] for p in row['prefixes_s'].values() if p['complete'])
    if row['level'] in ('L1_empty', 'L3_multi_path') and row['method'] == 'full':
        assert row['prefixes_s']['45']['complete'] is False
        assert 'eps_c' not in row['prefixes_s']['45']
for method in ('uncalibrated', 'full'):
    groups = [r for r in report['rows'] if r['method'] == method and r['trial_kind'] == 'baseline']
    assert len(groups) == 3 and len({r['level'] for r in groups}) == 3
    for row in groups:
        repeat = next(r for r in report['rows'] if r['method'] == method and r['level'] == row['level'] and r['trial_kind'] == 'repeat')
        assert row['terminal'] == repeat['terminal'] and row['prefixes_s'] == repeat['prefixes_s']
        assert row['commands'] == repeat['commands']
resource = report['resources']
assert resource['latest_retained_report_episode_count_sum'] == 99
assert resource['available_NAV043_episodes'] is None and resource['available_NAV043_gpu_wall_hours'] is None
assert not resource['new_execution_authorized_by_this_audit']
script = NAV / 'scripts/audit_nav043_saved_applicability.py'
ast.parse(script.read_text())
refusal = subprocess.run([sys.executable, '-B', str(script), '--out', str(report_path)], cwd=ROOT, text=True, capture_output=True)
assert refusal.returncode != 0 and 'AssertionError' in refusal.stderr
assert sha(report_path) == hashlib.sha256(report_path.read_bytes()).hexdigest()
plan_path = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
plan = load(plan_path)
base = json.loads(subprocess.check_output(['git', 'show', 'df7b64a:doc/plan/20261109_camra_navigation_implementation.json'], cwd=ROOT))
assert {k: v for k, v in plan.items() if k != 'tasks'} == {k: v for k, v in base.items() if k != 'tasks'}
for old, new in zip(base['tasks'], plan['tasks']):
    if old['id'] == 'NAV-043':
        assert old['work_records'] == new['work_records'][:len(old['work_records'])]
        assert {k: v for k, v in old.items() if k != 'work_records'} == {k: v for k, v in new.items() if k != 'work_records'}
    else:
        assert old == new
assert sum(t['status'] for t in plan['tasks']) == 55
assert not subprocess.check_output(['git', 'diff', 'df7b64a', '--name-only', '--', 'manuscripts', '*.tex', '*.bib', 'RC-PPO', 'HiRA', 'CaMRA/locomotion', 'CaMRA/calibration'], cwd=ROOT, text=True).strip()
validation = subprocess.run([sys.executable, '-B', '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', str(plan_path), '--mode', 'update'], cwd=ROOT, text=True, capture_output=True)
assert validation.returncode == 0, validation.stdout + validation.stderr
out = {'at': datetime.now().astimezone().isoformat(), 'status': 'PASS_BASIC_SAVED_AUDIT_CHECKS_NOT_NAV043_ACCEPTANCE',
       'source_hash_entries_checked': source_count, 'saved_trials': 12, 'distinct_configurations_per_method': 3,
       'duplicate_repetitions_not_pooled': True, 'incomplete_45s_prefixes_not_padded': True,
       'existing_output_refused': True, 'old_plan_definitions_and_records_preserved': True,
       'protected_paths_unchanged': True, 'audit_sha256': sha(report_path),
       'verifier_sha256': sha(Path(__file__)), 'validator': validation.stdout,
       'NAV043_status': False, 'completed': 55, 'new_applications': 0, 'W_queries': 0,
       'test_subagent_used': False, 'remote_push': False}
with (HERE / 'saved-audit-verification.json').open('x') as stream:
    json.dump(out, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps(out, ensure_ascii=False, indent=2))
