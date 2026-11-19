"""Check completed saved reports and plan provenance; never rerun reviewers."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
HERE = Path(__file__).resolve().parent
BASE = '268c2efe906bb89c54c2846a87002719f728919d'
OUT = HERE / 'exploration-close-basic.json'
assert socket.gethostname().split('.')[0] == 'tiger11'
assert os.environ.get('TMUX') and Path.cwd() == ROOT
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
assert not OUT.exists()


def load(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


plan_path = Path('doc/plan/20261109_camra_navigation_implementation.json')
before = json.loads(subprocess.check_output(['git', 'show', f'{BASE}:{plan_path}']))
after = load(plan_path)
record_delta = 0
for old, new in zip(before['tasks'], after['tasks'], strict=True):
    if old['id'] == 'NAV-043':
        assert old['status'] is False and new['status'] is False
        n = len(old['work_records'])
        assert new['work_records'][:n] == old['work_records']
        record_delta = len(new['work_records']) - n
        assert record_delta >= 2
        old['work_records'] = new['work_records']
    assert old == new, old['id']
assert before == after
assert sum(t['status'] for t in after['tasks']) == 55

review_path = NAV / 'reports/nav043_exploration_review_v1.json'
margin_path = NAV / 'reports/nav043_exploration_applicability_v1.json'
review, margin = load(review_path), load(margin_path)
assert review['status'] == 'SAVED_EVIDENCE_REVIEW_NOT_G5'
assert margin['status'] == 'DESCRIPTIVE_BUDGET_SELECTED_MARGIN_REVIEW_NOT_G5'
assert not review['NAV_043_pass'] and not review['G5_pass']
assert not margin['NAV043_passed'] and not margin['G5_passed']
assert review['allocation_terminal'] == 'FINISHED.json'
assert not review['incomplete'] and not margin['incomplete'] and not margin['refusals']
assert len(review['episodes']) == len(margin['rows']) == 6
assert review['applications'] == 3
assert review['verified_initial_W_queries_in_finished_apps'] == 30
assert review['completed_initial_W_queries_all_saved_decisions'] == 30
for key in ('verified_physical_episodes_in_finished_apps',
            'verified_nominal_episodes_in_finished_apps', 'physical_navigation_starts',
            'nominal_starts', 'saved_physical_results', 'saved_nominal_results'):
    assert review[key] == 6, key
assert review['observed_explicit_step_calls_finished_apps'] <= 60000
assert review['gpu_wall_s_finished_apps'] <= 10800
assert review['protected_hashes_checked'] == margin['protected_identities_checked'] == 295
assert all(review[k] == 0 for k in ('new_applications', 'new_model_inference',
                                  'new_rollouts', 'training_updates'))
assert all(margin[k] == 0 for k in ('new_applications', 'new_physical_episodes',
                                  'new_nominal_episodes', 'W_queries',
                                  'training_updates', 'calibration_fits'))
assert margin['final_data_access'] is False
source_count = 0
for index in (review['raw_sources'], margin['source_hashes']):
    for path, digest in index.items():
        assert sha(path) == digest, path
        source_count += 1
protocol_path = NAV / 'config/nav043_exploration_v1.json'
protocol = load(protocol_path)
assert sha(protocol_path) == Path(str(protocol_path) + '.sha256').read_text().strip()
assert sha(protocol_path) == review['protocol_sha256']
for path, digest in protocol['sealed_hashes'].items():
    assert sha(path) == digest, path
pair = load(NAV / 'config/nav066_decision_prereg_v2.json')
protected = {str(NAV / k): v for k, v in pair['implementation_hashes'].items()}
protected.update(pair['protected_hashes'])
protected.update({c['checkpoint']: c['checkpoint_sha256'] for c in pair['six_candidates']})
assert len(protected) == 295
for path, digest in protected.items():
    assert sha(path) == digest, path

totals = {m: {'decisions': 0, 'accepted': 0, 'rejected': 0,
              'physical_safe_timely_budget_success': 0,
              'internal_safe_budget_success': 0, 'nominal_feasible': 0}
          for m in ('uncalibrated', 'full')}
rows = []
for episode in review['episodes']:
    trial = Path(episode['trial'])
    result, nominal = load(trial / 'result.json'), load(trial / 'nominal.json')
    assert episode['complete_saved_physics_event_replay']
    assert episode['nominal_saved_actions_replayed'] == len(nominal['trace'])
    assert not result['rejected']
    metrics = totals[result['method']]
    metrics['decisions'] += 1
    metrics['accepted'] += 1
    metrics['physical_safe_timely_budget_success'] += result['safe_timely_measured_budget_success']
    metrics['internal_safe_budget_success'] += result['internal_safe_budget_success']
    metrics['nominal_feasible'] += nominal['feasible']
    m = next(r for r in margin['rows'] if r['trial'] == str(trial))
    assert m['physics_ticks'] == result['physics_steps']
    assert m['source_hashes'][str(trial / 'result.json')] == sha(trial / 'result.json')
    rows.append({'level': episode['level'], 'method': result['method'],
                 'selected_budget_J': result['z0_J'], 'physical_label': result['label'],
                 'physical_t_s': result['t_s'], 'measured_J': result['measured_joint_work_J'],
                 'internal_J': result['internal_model_cost_J'],
                 'safe_timely_measured_budget_success': result['safe_timely_measured_budget_success'],
                 'nominal_feasible': nominal['feasible'], 'nominal_t_s': nominal['t_s'],
                 'max_position_error_200Hz_m': m['terminal']['max_position_error_200Hz_m'],
                 'cost_error_vs_same_command_nominal': m['terminal']['eps_c']})
assert totals == review['methods']
assert margin['original_allocation_ledger'] == review['allocation_resource_ledger']
followup = load(HERE / 'applicability-followup-finished.json')
assert followup['review_sha256'] == sha(review_path)
assert followup['output_sha256'] == sha(margin_path) and followup['returncode'] == 0
assert not (HERE / 'applicability-followup-error.json').exists()
protected_diff = subprocess.check_output(
    ['git', 'diff', BASE, '--name-only', '--ignore-submodules=all', '--',
     'manuscripts', '*.tex', '*.bib', 'RC-PPO', 'HiRA', 'CaMRA/locomotion',
     'CaMRA/calibration'], text=True)
assert not protected_diff.strip(), protected_diff
report = {'at': datetime.now().astimezone().isoformat(), 'host': socket.gethostname(),
          'pane': os.environ.get('TMUX_PANE'), 'status': 'PASS_SAVED_EXPLORATION_CLOSURE_NOT_G5',
          'baseline_commit': BASE, 'plan_records_appended': record_delta, 'tasks_true': 55,
          'verification_sha256': sha(__file__), 'review_sha256': sha(review_path),
          'applicability_sha256': sha(margin_path), 'source_hash_entries_checked': source_count,
          'protected_hashes_checked': len(protected), 'sealed_hashes_checked': len(protocol['sealed_hashes']),
          'original_allocation_gpu_wall_s': review['gpu_wall_s_finished_apps'],
          'original_allocation_explicit_steps': review['observed_explicit_step_calls_finished_apps'],
          'methods': totals, 'rows': rows, 'NAV043_passed': False, 'G5_passed': False,
          'new_applications': 0, 'new_episodes': 0, 'new_W_queries': 0, 'new_fits': 0}
with OUT.open('x') as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write('\n')
    stream.flush()
    os.fsync(stream.fileno())
print(json.dumps(report, ensure_ascii=False), flush=True)
