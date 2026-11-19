"""Main-agent offline proposal audit. Does not import Isaac, Torch or runners."""
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLAN = 'doc/plan/20261109_camra_navigation_implementation.json'
PROPOSAL = 'CaMRA/navigation/config/nav067_remaining_validation_proposal_v1.json'
BASE = 'bcdfa8d31759d5985b736d6c55909bffde0dd97a'
EXPECTED_PYTHON = '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'


def main():
    assert os.environ.get('TMUX_PANE') == '%600'
    assert socket.gethostname().split('.')[0] == 'tiger11'
    assert Path(sys.executable).resolve() == Path(EXPECTED_PYTHON).resolve()
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == BASE
    p = json.loads((ROOT / PROPOSAL).read_bytes())
    assert p['status'] == 'PROPOSED' and p['approved'] is False
    assert p['implementation_readiness']['controlled_physical_runner_implemented'] is False
    assert p['implementation_readiness']['executable_entry'] is None
    checked = []
    for name, item in p['inputs'].items():
        path = Path(item['path'])
        if not path.is_absolute():
            path = ROOT / path
        assert sha256(path.read_bytes()).hexdigest() == item['sha256'], name
        checked.append(str(path))
    for name, expected in p['sealed_sources'].items():
        assert sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
        checked.append(name)
    manifest = json.loads(Path(p['inputs']['manifest']['path']).read_bytes())
    manifest_text = json.dumps(manifest, sort_keys=True)
    for expected in p['selected_model_sha256'].values():
        assert expected in manifest_text, 'selected-model identity is absent from the original manifest'
    old_review = json.loads((ROOT / p['inputs']['completed_execution_review']['path']).read_bytes())
    assert old_review['status'] == 'PILOT_TRACE_PASS_NAV067_INCOMPLETE'
    assert old_review['NAV_067_status'] is False and old_review['performance_evidence'] is False
    assert not old_review['main_agent_saved_episode_review_failures']
    assert not old_review['main_agent_saved_runtime_review_failures']
    e, r, c, bound = (p[k] for k in ('execution', 'repetitions', 'controlled_cases', 'resource_upper_bound'))
    assert (e['host'], e['ip'], e['python']) == ('tiger11', '192.168.20.11', EXPECTED_PYTHON)
    assert e['max_concurrent_gpu_processes'] == e['max_allocated_gpus'] == 1
    assert e['gpu_wall_clock_cap_s'] == 4 * 3600
    assert e['automatic_retries'] == e['new_training_updates'] == e['new_W_queries'] == e['new_nominal_episodes'] == 0
    assert e['final_data_access'] is e['NAV_043_execution'] is e['remote_push'] is False
    assert not Path(e['new_output_root']).exists(), 'proposed execution output must not already exist'
    assert (r['horizon_s'], r['warmup_s'], r['seed'], r['simulator_seed']) == (45, 1, 3001, 100)
    assert (r['physics_hz'], r['low_level_hz'], r['high_level_hz']) == (200, 50, 10)
    assert [s['level'] for s in r['slots']] == ['L1_empty', 'L2_single_detour', 'L3_multi_path']
    assert all(s['methods_in_order'] == ['uncalibrated', 'full'] for s in r['slots'])
    episodes = sum(len(s['methods_in_order']) for s in r['slots'])
    assert episodes == r['new_navigation_episodes'] == 6
    assert r['new_repetitions_per_existing_configuration'] == 1
    assert r['total_runs_per_configuration_including_baseline'] == 2
    assert [case['id'] for case in c['cases_in_order']] == ['C01', 'C02', 'C03', 'C04', 'C05', 'C06']
    assert c['cases'] == len(c['cases_in_order']) == 6
    assert c['warmup_s'] + c['max_measurement_s_per_case'] == c['max_total_s_per_case_including_warmup'] == 2
    repeat_seconds = episodes * (r['horizon_s'] + r['warmup_s'])
    controlled_seconds = c['cases'] * c['max_environments_per_case'] * c['max_total_s_per_case_including_warmup']
    assert repeat_seconds == bound['repeat_environment_seconds_including_warmup'] == 276
    assert controlled_seconds == bound['controlled_environment_seconds_including_warmup'] == 24
    assert repeat_seconds + controlled_seconds == bound['total_environment_seconds_including_warmup'] == 300
    assert 300 * r['physics_hz'] == bound['total_environment_ticks_including_warmup'] == 60000
    assert c['physics_hz'] == r['physics_hz']
    assert e['max_kit_starts'] == c['cases'] + len(r['slots']) == 9
    assert bound['quota_transfer_allowed'] is False
    plan = json.loads((ROOT / PLAN).read_bytes())
    old = json.loads(subprocess.check_output(['git', 'show', BASE + ':' + PLAN], cwd=ROOT))
    assert {k: v for k, v in plan.items() if k != 'tasks'} == {k: v for k, v in old.items() if k != 'tasks'}
    for before, after in zip(old['tasks'], plan['tasks'], strict=True):
        assert {k: v for k, v in before.items() if k != 'work_records'} == {k: v for k, v in after.items() if k != 'work_records'}
        assert after['work_records'][:len(before['work_records'])] == before['work_records']
        if before['id'] != 'NAV-067':
            assert before == after
    assert sum(t['status'] for t in plan['tasks']) == 54
    ready = [t['id'] for t in plan['tasks'] if not t['status'] and all(next(u for u in plan['tasks'] if u['id'] == dep)['status'] for dep in t['depends_on'])]
    assert ready == ['NAV-067']
    assert b'\r\n' not in (ROOT / PLAN).read_bytes()
    assert not subprocess.check_output(['git', 'diff', BASE, '--name-only', '--', 'manuscripts', '*.tex', '*.bib', 'RC-PPO', 'HiRA'], cwd=ROOT, text=True).strip()
    command = [EXPECTED_PYTHON, '-B', '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', str(ROOT / PLAN), '--mode', 'update']
    subprocess.run(command, check=True, cwd=ROOT)
    result = {
        'status': 'PROPOSAL_OFFLINE_CHECK_PASS_NOT_EXECUTION_APPROVAL',
        'at': datetime.now().astimezone().isoformat(),
        'host': socket.gethostname(), 'tmux_pane': os.environ['TMUX_PANE'],
        'python': sys.executable, 'baseline_git_head': BASE,
        'proposal_sha256': sha256((ROOT / PROPOSAL).read_bytes()).hexdigest(),
        'checked_input_and_source_identities': checked,
        'quota_math': {'new_navigation_episodes': episodes, 'short_controlled_cases': 6, 'max_environment_seconds': 300, 'max_environment_ticks': 60000, 'max_gpu_wall_clock_hours': 4, 'max_kit_starts': 9},
        'historic_plan_records_and_definitions_preserved': True,
        'NAV_067_status': False, 'tasks_recorded_complete': 54,
        'new_app_starts': 0, 'new_physical_episodes': 0, 'new_training_updates': 0, 'new_W_queries': 0,
        'remaining_work': ['Obtain explicit approval for new quotas, then implement/seal/check isolated harness before its first actual simulation.'],
        'limitations': ['No controlled runner implementation or actual-case/repeatability evidence is claimed.', 'This audit checks the proposal and protected input bindings, not physical performance.']
    }
    target = HERE / 'verification.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
