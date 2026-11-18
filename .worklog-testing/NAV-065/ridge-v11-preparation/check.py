"""Main-agent legacy audit: preparation integrity, original gates and protected inputs."""
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
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
read = lambda p: json.loads(p.read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
baseline, plan = read(HERE / 'baseline.json'), read(PLAN)
old = baseline['plan']
assert {k: v for k, v in plan.items() if k != 'tasks'} == {k: v for k, v in old.items() if k != 'tasks'}
for before, after in zip(old['tasks'], plan['tasks']):
    if before['id'] == 'NAV-065':
        assert {k: v for k, v in before.items() if k != 'work_records'} == {k: v for k, v in after.items() if k != 'work_records'}
        assert after['work_records'][:len(before['work_records'])] == before['work_records']
        assert len(after['work_records']) == len(before['work_records']) + 1
    else:
        assert before == after
assert len(plan['tasks']) == 67 and sum(t['status'] for t in plan['tasks']) == 52
assert '38 passed' in (HERE / 'basic-attempt1.log').read_text()
assert 'READONLY_PREFLIGHT_FINISHED' in (HERE / 'preflight-attempt1.log').read_text()
sys.path.insert(0, str(NAV / 'scripts'))
import probe_nav065_ridge as G
proposal = read(G.PROPOSAL)
assert not proposal['approved'] and proposal['status'] == 'PROPOSED_NOT_EXECUTED'
assert not (NAV / 'config/nav065_ridge_probe_prereg_v11.json').exists()
assert not (G.R.WORK / proposal['output_name']).exists()
assert proposal['resource_limit']['head_linear_solves_total_max'] == 2
assert proposal['resource_limit']['optimizer_steps_total_max'] == proposal['resource_limit']['new_train_episodes_max'] == proposal['resource_limit']['development_scoring_rounds_max'] == 0
d = read(NAV / 'reports/nav065_ridge_preflight_v11.json')
previous = read(NAV / 'reports/nav065_local_probe_preflight_v9.json')
assert d['source_full_train_gate'] == previous['full_train_gate'] and not d['source_full_train_gate']['pass']
assert d['local_keys'] == previous['local_keys'] and len(d['local_keys']) == 25
assert d['available_train_episodes'] == 162 and d['available_train_steps'] == 59877
assert d['weighted_group_rows'] == 60905 and abs(d['weight_sum'] - 1.) < 1e-12
assert [g['name'] for g in d['groups_before']] == list(G.GROUPS)
assert [g['rows'] for g in d['groups_before']] == [33223, 26654, 162, 866]
assert sum(g['rows'] for g in d['groups_before'][:2]) == d['available_train_steps']
assert all(abs(g['mass'] - .25) < 1e-12 for g in d['groups_before'])
starts = d['groups_before'][2]
assert starts['false_feasible'] == 13 and starts['false_infeasible'] == 1
assert d['formal_head_solves'] == d['formal_optimizer_steps'] == d['new_rollouts'] == d['development_scoring_rounds'] == 0
for role, design in d['ridge_designs'].items():
    assert design['rows'] == 60905 and design['columns'] == 257
    assert design['candidate_parameters_assigned'] == 0 and not design['candidate_weights_solved']
    bound = (design['source_objective'] / design['ridge_coefficient']) ** .5
    assert abs(bound - design['correction_norm_bound_for_objective_no_worse_than_source']) < 1e-12
_, v10pin = G.H.approved_registration(NAV / 'config/nav065_head_probe_prereg_v10.json')
v10 = read(NAV / 'reports/nav065_head_probe_v10_review.json')
assert v10pin == v10['approval_pin']
for name, digest in v10['evidence']['output_hashes'].items():
    assert sha(Path(v10['evidence']['output']) / name) == digest
v2 = read(NAV / 'reports/nav064_terminal_tail_training_release_v1.json')['v2_verified_files']
for name, digest in v2.items():
    assert sha(Path(name)) == digest
manifest = read(G.R.WORK / 'nav065_staged_refit_v7/fitted_candidates.json')
assert len(manifest['candidates']) == 6
for candidate in manifest['candidates']:
    assert sha(Path(candidate['checkpoint'])) == candidate['checkpoint_sha256']
v8 = read(G.R.WORK / 'nav065_pilot_refit_v8/PILOT_FINISHED.json')
assert sha(Path(v8['candidate']['checkpoint'])) == proposal['source_checkpoint_sha256']
changed = subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT, text=True).splitlines()
assert not any(p.startswith(('manuscripts/', 'RC-PPO/', 'HiRA/', 'CaMRA/locomotion/', 'CaMRA/calibration/')) or p.endswith(('.tex', '.bib')) for p in changed)
for line in baseline['git_status'].splitlines():
    if line.startswith('?? '):
        assert (ROOT / line[3:]).exists(), line
files = ['CaMRA/navigation/scripts/probe_nav065_ridge.py', 'CaMRA/navigation/tests/test_nav065_ridge_probe.py',
    'CaMRA/navigation/config/nav065_ridge_probe_proposal_v11.json',
    'CaMRA/navigation/docs/nav065_ridge_probe_v11_20261118.md',
    'CaMRA/navigation/reports/nav065_ridge_preflight_v11.json',
    '.worklog-testing/NAV-065/ridge-v11-preparation/context.py',
    '.worklog-testing/NAV-065/ridge-v11-preparation/open.py',
    '.worklog-testing/NAV-065/ridge-v11-preparation/preflight.py',
    '.worklog-testing/NAV-065/ridge-v11-preparation/check.py',
    '.worklog-testing/NAV-065/ridge-v11-preparation/context.log',
    '.worklog-testing/NAV-065/ridge-v11-preparation/basic-attempt1.log',
    '.worklog-testing/NAV-065/ridge-v11-preparation/preflight-attempt1.log']
report = {'schema': 'camra-nav065-ridge-preparation-review-v11',
    'timestamp': datetime.now().astimezone().isoformat(), 'host': socket.getfqdn(),
    'tmux': os.environ['TMUX'], 'python': sys.executable, 'base_git': baseline['git'],
    'preparation_pass': True, 'basic_tests_passed': 38, 'verification_by': 'main agent, legacy; no test subagent',
    'source_gate_exact_match': True, 'complete_original_and_additional_train_coverage': True,
    'available_train_episodes': 162, 'available_train_steps': 59877, 'weighted_group_rows': 60905,
    'formal_head_solves': 0, 'formal_optimizer_steps': 0, 'new_rollouts': 0, 'development_scoring_rounds': 0,
    'candidate_models_and_old_implementation_hashes_unchanged': True, 'v2_files_verified': len(v2),
    'synthetic_tests_use_frozen_candidates': False, 'proposal_approved': False, 'requested_head_solves': 2,
    'scientific_pass': False, 'original_G4': 'FAIL', 'NAV065': False, 'NAV066_started': False, 'completed_tasks': 52,
    'bound_input_hashes': {name: sha(NAV / name) for name in G.BOUND_INPUTS},
    'evidence': {name: sha(ROOT / name) for name in files},
    'limitations': d['limitations'] + ['Fixed ridge strength is an engineering diagnostic choice, not a theorem-derived optimum.',
        'Complete train coverage and a displacement bound do not guarantee zero false feasible decisions or held-out accuracy.']}
with (NAV / 'reports/nav065_ridge_probe_v11_preparation_review.json').open('x') as f:
    json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    f.write('\n')
print(json.dumps({k: v for k, v in report.items() if k not in ('evidence', 'bound_input_hashes', 'limitations')}, ensure_ascii=False))
