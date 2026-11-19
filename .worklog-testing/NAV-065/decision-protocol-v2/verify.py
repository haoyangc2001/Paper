"""Basic document/integrity audit only; never loads models or runs rollouts."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
HERE = Path(__file__).resolve().parent
PYTHON = '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
REL = 'CaMRA/navigation/config/nav065_budget_decision_protocol_v2.json'
REPORT = ROOT / 'CaMRA/navigation/reports/nav065_budget_decision_protocol_v2_preparation_review.json'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == PYTHON
baseline = json.loads((HERE / 'baseline.json').read_bytes())
p = json.loads((ROOT / REL).read_bytes())
checks = []

def check(name, value):
    assert value, name
    checks.append(name)

check('draft_not_active', p['authorization']['status'] == 'DRAFT_NOT_ACTIVATED' and not p['authorization']['execution_enabled'])
check('no_new_run_authorized', not p['authorization']['training_authorized'] and not p['authorization']['evaluation_authorized'])
for name in p['sources']:
    check('source_exists:' + name, (ROOT / name).is_file())
source = json.loads((ROOT / 'CaMRA/navigation/reports/nav065_staged_refit_v7_review.json').read_bytes())
bindings = {}
for candidate in source['train_candidates']:
    ckpt = candidate['stage_checkpoints'][-1]
    actual = hashlib.sha256(Path(ckpt['path']).read_bytes()).hexdigest()
    check('six_candidate_source:' + candidate['run_id'], actual == ckpt['sha256'] and ckpt['stage'] == 'boundary' and ckpt['trial'] == 300)
    bindings[candidate['run_id']] = {'path': ckpt['path'], 'sha256': actual}
check('six_bound_two_proposed', len(bindings) == 6 and len(p['candidate_proposal']['selected_candidates']) == 2)
for item in p['candidate_proposal']['selected_candidates']:
    check('selected_hash:' + item['method'], bindings[item['candidate_id']] == {k: item[k] for k in ('path', 'sha256')} and item['candidate_id'].endswith('_s0'))
split = json.loads((ROOT / 'CaMRA/navigation/config/split_manifest_v1.json').read_bytes())
old_seeds = {seed for spec in split['splits'].values() for seed in spec['seeds']}
data = p['independent_data_proposal']
check('metadata_seed_disjoint_only', not old_seeds.intersection(data['seed_family']))
check('fresh_manifest_not_claimed_created', not data['manifest_created'])
check('nine_map_spec', len(data['seed_family']) * len(data['levels']) == data['n_maps'] == 9)
rule = p['decision_rule']
grid = rule['k_grid']
check('original_grid_retained', grid == [0.7, 0.85, 1.0, 1.15, 1.3])
check('one_step_reserve', rule['reserve_steps'] == 1 and all(abs(grid[i+1]-grid[i]-rule['reserve_k']) < 1e-12 for i in range(4)))
check('production_gap_disclosed', not rule['implemented_in_production'])
ev = p['evaluation_proposal']
check('ninety_grid_rows', ev['new_nominal_episodes_max'] == 9 * 2 * len(grid) == 90)
check('eighteen_decisions_not_ninety', ev['max_actual_decisions'] == 2 * ev['decisions_per_method'] == 18)
check('ninety_value_queries', ev['max_W_queries'] == ev['max_actual_decisions'] * rule['max_W_queries_per_decision'] == 90)
check('finite_step_cap', ev['max_nominal_steps'] == ev['new_nominal_episodes_max'] * ev['max_steps_per_episode'] == 27000)
check('zero_training_physics', ev['new_training_episodes'] == ev['optimizer_updates'] == ev['physical_episodes'] == 0)
gate = p['proposed_acceptance']
use = gate['usefulness']
check('safety_budget_not_redefined', gate['safety_time_budget']['accepted_failure_count_max'] == 0 and gate['safety_time_budget']['permitted_overspend_J'] == 0)
check('nonvacuous_coverage', use['minimum_accepted_per_method'] == 6 and use['minimum_accepted_per_level_per_method'] == 2)
check('reject_limit_consistent', use['feasible_map_recall_min'] == 0.8 and use['false_rejection_rate_among_feasible_max'] == 0.2)
recall_counts = {str(n): next(a for a in range(n+1) if 5*a >= 4*n) for n in range(1, 10)}
check('recall_rounds_up', recall_counts == {'1': 1, '2': 2, '3': 3, '4': 4, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8})
check('efficiency_matches_reserve', gate['budget_efficiency']['max_grid_steps_above_actual_minimum'] == rule['reserve_steps'])
for name, digest in baseline['v2_files'].items():
    check('v2:' + name, hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest)
changed_docs = {'AGENTS.md', 'CaMRA/navigation/README.md', 'CaMRA/navigation/docs/implementation_handoff.md', 'doc/plan/20261109_camra_navigation_implementation.json'}
for name, digest in baseline['hashes'].items():
    if name not in changed_docs:
        check('old_evidence_unchanged:' + name, hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest)
plan_path = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
now = json.loads(plan_path.read_bytes())
old = baseline['plan']
check('plan_top_level_unchanged', {k:v for k,v in now.items() if k != 'tasks'} == {k:v for k,v in old.items() if k != 'tasks'})
for before, after in zip(old['tasks'], now['tasks'], strict=True):
    if before['id'] == 'NAV-065':
        check('NAV065_old_fields_unchanged', {k:v for k,v in before.items() if k != 'work_records'} == {k:v for k,v in after.items() if k != 'work_records'})
        check('NAV065_history_prefix', after['work_records'][:len(before['work_records'])] == before['work_records'])
        check('NAV065_only_append', len(after['work_records']) in (len(before['work_records'])+1, len(before['work_records'])+2))
    else:
        check('other_task_unchanged:' + before['id'], after == before)
states = {t['id']:t['status'] for t in now['tasks']}
check('52_of_67_NAV065_NAV066_false', len(states) == 67 and sum(states.values()) == 52 and not states['NAV-065'] and not states['NAV-066'])
validator = subprocess.run([PYTHON, '-B', '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', str(plan_path), '--mode', 'update'], cwd=ROOT, capture_output=True, text=True, check=True)
(HERE / 'validation-after.log').write_text(validator.stdout + validator.stderr)
check('legacy_plan_validation', validator.returncode == 0)
git_diff = subprocess.run(['git', '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--check'], cwd=ROOT, capture_output=True, text=True)
check('git_diff_check', git_diff.returncode == 0)
changed = subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT, text=True).splitlines()
check('tracked_changes_scope', set(changed) <= changed_docs)
report = {'schema': 'camra-navigation-protocol-preparation-review-v2', 'timestamp': datetime.now().astimezone().isoformat(), 'task': 'NAV-065', 'plan_version': now['document']['version'], 'host': socket.getfqdn(), 'tmux': os.environ['TMUX'], 'python': PYTHON, 'reviewer': '/root', 'independent_test_subagent': False, 'preparation_pass': True, 'check_count': len(checks), 'checks': checks, 'protocol_sha256': hashlib.sha256((ROOT / REL).read_bytes()).hexdigest(), 'source_bindings_verified_without_loading': bindings, 'metadata_seed_disjoint_only': True, 'fresh_geometry_manifest_verified': False, 'recall_minimum_correct_counts': recall_counts, 'actual_training_updates': 0, 'actual_new_episodes': 0, 'production_rule_integrated': False, 'protocol_activated': False, 'scientific_validation_performed': False, 'NAV065': False, 'NAV066_started': False, 'completed_tasks': 52, 'original_G4': 'FAIL', 'limitations': ['仅文档/身份/完整性基础检查；未验证选择效果。', '新实例几何/暴露账本及新版生产调用链须在激活后核验。', '建议阈值、来源、数据与90回合额度待明确批准。']}
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('checks','source_bindings_verified_without_loading')}, ensure_ascii=False, indent=2))
