"""Verify scope preparation only. Never imports a model or starts Isaac."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
NAV = ROOT / 'CaMRA/navigation'
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
assert socket.gethostname().split('.')[0] == 'tiger11'
assert os.environ.get('TMUX_PANE') == '%599'
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
raw = PLAN.read_bytes()
plan = json.loads(raw)
base_raw = subprocess.check_output(['git', 'show', '6125043:doc/plan/20261109_camra_navigation_implementation.json'], cwd=ROOT)
base = json.loads(base_raw)
states = {t['id']: t['status'] for t in plan['tasks']}
assert len(states) == 67 and sum(states.values()) == 55
assert next(t['id'] for t in plan['tasks'] if not t['status'] and all(states[d] for d in t['depends_on'])) == 'NAV-043'
for old, new in zip(base['tasks'], plan['tasks']):
    if new['id'] == 'NAV-043':
        assert not new['status'] and not new['verification_result']
        assert len(new['work_records']) == 2 and old['work_records'] == []
        assert {k: v for k, v in old.items() if k != 'work_records'} == {k: v for k, v in new.items() if k != 'work_records'}
    else:
        assert old == new
assert {k: v for k, v in base.items() if k != 'tasks'} == {k: v for k, v in plan.items() if k != 'tasks'}
fields = set(plan['document']['work_record_fields'])
assert all(fields <= set(r) for t in plan['tasks'] for r in t['work_records'])
report = json.loads((NAV / 'reports/nav067_engineering_acceptance_v1.json').read_bytes())
assert report['status'] == 'PASS_ENGINEERING_INTEGRATION_NOT_NAVIGATION_PERFORMANCE'
assert len(report['controlled_cases']) == 6 and len(report['actual_navigation_results']) == 12
assert all(r['over_budget'] and not r['performance_evidence'] for r in report['actual_navigation_results'])
reg = json.loads((NAV / 'config/nav066_decision_prereg_v2.json').read_bytes())
rule = reg['decision_rule']
assert rule['grid'] == [.7, .85, 1., 1.15, 1.3] and rule['reserve_steps'] == 1 and rule['beta'] == 0
geometry = json.loads((NAV / 'config/geometry_margin_closedloop_v1.json').read_bytes())
assert geometry['validity']['T_range_s'] == [5, 30] and geometry['derivation']['fit_not_validation']
doc = NAV / 'docs/nav043_next_stage_scope_20261119.md'
assert doc.is_file()
assert not subprocess.check_output(['git', 'diff', '6125043', '--name-only', '--', 'manuscripts', '*.tex', '*.bib', 'RC-PPO', 'HiRA', 'CaMRA/locomotion', 'CaMRA/calibration'], cwd=ROOT, text=True).strip()
if base_raw.count(b'\r\n') and base_raw.count(b'\r\n') == base_raw.count(b'\n'):
    PLAN.write_bytes(raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
validation = subprocess.run([sys.executable, '-B', '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', str(PLAN), '--mode', 'update'], cwd=ROOT, capture_output=True, text=True)
assert validation.returncode == 0, validation.stdout + validation.stderr
sources = [doc, NAV / 'reports/nav067_engineering_acceptance_v1.json', NAV / 'config/nav066_decision_prereg_v2.json', NAV / 'config/geometry_margin_closedloop_v1.json', NAV / 'config/resource_budget_v1.json', NAV / 'scripts/nav067_sources.py']
result = {'at': datetime.now().astimezone().isoformat(), 'status': 'PREPARATION_SCOPE_VERIFIED_NOT_EXECUTION_READY', 'completed': 55, 'current_task': 'NAV-043', 'task_status': False, 'old_task_definitions_and_records_preserved': True, 'all_plan_bytes_read': len(raw), 'all_records_read': sum(len(t['work_records']) for t in plan['tasks']), 'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, 'new_simulator_applications': 0, 'new_physical_episodes': 0, 'new_nominal_episodes': 0, 'W_queries': 0, 'training_updates': 0, 'final_data_access': False, 'execution_protocol_approved': False, 'validator': validation.stdout}
with (HERE / 'verification.json').open('x') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
