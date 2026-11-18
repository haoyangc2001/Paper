"""Main-agent preparation audit; no solving or optimizing a real candidate."""
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
baseline = read(HERE / 'baseline.json')
plan = read(PLAN)
old = baseline['plan']
assert {k: v for k, v in plan.items() if k != 'tasks'} == {k: v for k, v in old.items() if k != 'tasks'}
for before, after in zip(old['tasks'], plan['tasks']):
    if before['id'] == 'NAV-065':
        assert {k: v for k, v in before.items() if k != 'work_records'} == {k: v for k, v in after.items() if k != 'work_records'}
        assert after['work_records'][:len(before['work_records'])] == before['work_records']
    else:
        assert before == after
assert len(plan['tasks']) == 67 and sum(t['status'] for t in plan['tasks']) == 52
assert '33 passed' in (HERE / 'basic-attempt2.log').read_text()
d = read(NAV / 'reports/nav065_head_preflight_v10.json')
old_d = read(NAV / 'reports/nav065_local_probe_preflight_v9.json')
assert d['source_full_train_gate'] == old_d['full_train_gate']
assert d['local_keys'] == old_d['local_keys'] and d['local_rows'] == 25
assert d['available_train_episodes'] == 162 and d['available_train_steps'] == 59877
assert not d['v9_all_local_checks_ever_passed'] and d['v9_joint_errors_min'] == 1
assert d['formal_head_solves'] == d['formal_optimizer_steps'] == d['new_rollouts'] == d['development_scoring_rounds'] == 0
for stages in d['spectra'].values():
    for values in stages.values():
        assert values['rank_at_solve_rcond'] == 25
        assert values['candidate_parameters_assigned'] == 0 and not values['candidate_weights_solved']
        assert values['target_residual_outside_retained_span_linf'] < 1e-10
proposal = read(NAV / 'config/nav065_head_probe_proposal_v10.json')
assert proposal['approved'] is False and proposal['resource_limit']['head_linear_solves_total_max'] == 2
assert not (NAV / 'config/nav065_head_probe_prereg_v10.json').exists()
work = Path('/pub/data/caohy/work/camra_navigation')
assert not (work / proposal['output_name']).exists()
v9 = read(NAV / 'reports/nav065_local_probe_v9_review.json')
for name, digest in v9['evidence']['output_hashes'].items():
    assert sha(Path(v9['evidence']['output']) / name) == digest
v2 = read(NAV / 'reports/nav064_terminal_tail_training_release_v1.json')['v2_verified_files']
for name, digest in v2.items():
    assert sha(Path(name)) == digest
for registration in ('nav065_pilot_refit_prereg_v8.json', 'nav065_local_probe_prereg_v9.json'):
    protocol = read(NAV / 'config' / registration)
    for name, digest in protocol['implementation_hashes'].items():
        assert sha(NAV / name) == digest
    assert sha(NAV / 'config' / registration) == (NAV / 'config' / registration).with_suffix('.json.sha256').read_text().strip()
manifest = read(work / 'nav065_staged_refit_v7/fitted_candidates.json')
for candidate in manifest['candidates']:
    assert sha(Path(candidate['checkpoint'])) == candidate['checkpoint_sha256']
assert sha(Path(work / 'nav065_pilot_refit_v8/PILOT_FINISHED.json')) == read(NAV / 'config/nav065_local_probe_proposal_v9.json')['source_finished_sha256']
protected = ('manuscripts/', 'RC-PPO/', 'HiRA/', 'CaMRA/locomotion/', 'CaMRA/calibration/')
changed = subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT, text=True).splitlines()
assert not any(name.startswith(protected) or name.endswith(('.tex', '.bib')) for name in changed)
evidence_files = ['CaMRA/navigation/scripts/probe_nav065_head.py', 'CaMRA/navigation/tests/test_nav065_head_probe.py',
    'CaMRA/navigation/config/nav065_head_probe_proposal_v10.json', 'CaMRA/navigation/reports/nav065_head_preflight_v10.json',
    'CaMRA/navigation/docs/nav065_head_probe_v10_20261118.md',
    '.worklog-testing/NAV-065/head-v10-preparation/context.py', '.worklog-testing/NAV-065/head-v10-preparation/check.py',
    '.worklog-testing/NAV-065/head-v10-preparation/basic-attempt1.log', '.worklog-testing/NAV-065/head-v10-preparation/basic-attempt2.log',
    '.worklog-testing/NAV-065/head-v10-preparation/preflight-attempt1.log']
report = {'schema': 'camra-nav065-head-preparation-review-v10', 'timestamp': datetime.now().astimezone().isoformat(),
    'host': socket.getfqdn(), 'tmux': os.environ['TMUX'], 'python': sys.executable, 'base_git': baseline['git'],
    'preparation_pass': True, 'basic_tests_passed': 33, 'source_gate_exact_match': True,
    'available_train_episodes': 162, 'available_train_steps': 59877, 'local_rows': 25,
    'formal_head_solves': 0, 'formal_optimizer_steps': 0, 'new_rollouts': 0, 'development_scoring_rounds': 0,
    'candidate_models_and_old_implementation_hashes_unchanged': True, 'v2_files_verified': len(v2),
    'synthetic_tests_use_frozen_candidates': False, 'proposal_approved': False, 'requested_head_solves': 2,
    'requested_optimizer_steps': 0, 'scientific_pass': False, 'original_G4': 'FAIL', 'NAV065': False,
    'NAV066_started': False, 'completed_tasks': 52,
    'evidence': {name: sha(ROOT / name) for name in evidence_files},
    'limitations': d['limitations'] + ['Exact-interpolation norm is not the norm required merely to pass the original error/sign checks.',
        'Read-only diagnosis cannot establish candidate head solve, full training or development success.']}
with (NAV / 'reports/nav065_head_probe_v10_preparation_review.json').open('x') as f:
    json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    f.write('\n')
print(json.dumps({k: v for k, v in report.items() if k not in ('evidence', 'limitations')}, ensure_ascii=False))
