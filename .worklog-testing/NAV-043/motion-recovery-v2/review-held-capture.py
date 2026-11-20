"""Saved-only case review, separate from the retained failed v2 closure gate.

Never writes the physical output root, imports Isaac, fits a model, queries W,
or changes the sealed coordinator/reviewer. Terminal HELD stays HELD.
"""
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path('/home/caohy/repositories/Paper')
EV = ROOT/'.worklog-testing/NAV-043/motion-recovery-v2'
sys.path.insert(0, str(ROOT/'CaMRA/navigation/scripts'))
import nav043_motion_probe_recovery_guard as G
import audit_nav043_motion_probe_recovery_v2 as A

assert socket.gethostname().split('.')[0] == 'tiger11'
assert os.environ.get('TMUX') and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert str(Path(sys.executable).absolute()) == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
p, approval, readiness = G.approved()
out = Path(p['output_root'])
identities = G.protected_sources()
assert identities == {'protected_identities':295,'existing_raw_input_hashes':90,'bundle_files':29}

def manifest(directory):
    return [{'path':str(path), 'bytes':path.stat().st_size, 'sha256':G.sha(path)}
            for path in sorted(directory.rglob('*')) if path.is_file()]

before = manifest(out)
print('SAVED_FILES', len(before), sum(r['bytes'] for r in before), flush=True)
owner, process = G.load(out/'OWNER.json'), G.load(out/'PROCESS.json')
assert owner == G.load(EV/'launch.lock')
assert owner['git_head'] == 'c8d6f7158e9f9f8dc44c93743ad5e05519ce069c'
assert owner['proposal_sha256'] == G.sha(G.PROPOSAL)
assert owner['approval_sha256'] == G.sha(G.APPROVAL)
assert owner['readiness_sha256'] == G.sha(G.READINESS)
assert owner['gpu']['registered_gpu'][:2] == ['6','GPU-ae8422a0-8130-a384-1ec6-8ba59fe59d3a']
pids = [owner['pid'], process['pid']]
assert all(not (Path('/proc')/str(pid)).exists() for pid in pids)
assert (out/'CAPTURE_FINISHED.json').is_file() and (out/'HELD.json').is_file()
missing = ['child_step_accounting.json','execution_ledger.json','FINISHED.json']
assert all(not (out/name).exists() for name in missing)
assert not (out/'ERROR.json').exists() and not (out/'child_cleanup_error.json').exists()
held, cleanup, exit_info = [G.load(path) for path in
    (out/'HELD.json', out/'process_cleanup.json', EV/'exit.json')]
assert held['error'] == "FileNotFoundError(2, 'No such file or directory')"
assert held['primary_application_error'] is None and held['applications_started'] == 1
assert held['exit_code'] == 0 and not held['timeout'] and not held['cleanup_pending']
assert held['no_automatic_retry'] and not exit_info['retry_performed']
assert cleanup['returncode'] == exit_info['exit_code'] == 0
assert cleanup['cleanup_confirmed'] and not cleanup['errors']
assert (EV/'launch.exit').read_text().strip() == '1'

runtime = G.load(out/'runtime.json')
assert runtime['num_envs'] == 1 and runtime['physics_dt_s'] == .005 and runtime['decimation'] == 4
assert runtime['terrain']['terrain_type'] == 'plane' and runtime['terrain']['terrain_generator'] is None
assert runtime['actual_origins'] == [[0.,0.,0.]]
assert runtime['actual_isaaclab_module'] == '/pub/data/caohy/IsaacLab_v0472/source/isaaclab/isaaclab/__init__.py'
assert runtime['event_modes'] == ['reset']
assert runtime['event_terms'] == {'reset':['reset_base','reset_robot_joints']}
assert not runtime['observation_corruption_enabled']
clean = runtime['clean_configuration']
assert not clean['obs_corruption_enabled'] and clean['obs_noise'] is None
assert clean['command_resampling_disabled'] and clean['automatic_reset_disabled']
assert all(v == [0.,0.] for v in clean['reset_base_pose_range'].values())
assert all(v == [0.,0.] for v in clean['reset_base_velocity_range'].values())
assert clean['reset_joint_scale_range'] == [1.,1.] and clean['reset_joint_velocity_range'] == [0.,0.]
assert len(runtime['ground']) == 1 and runtime['ground'][0]['values'] == [1.,1.,0.]
assert runtime['robot_usd_path'] == '/pub/data/caohy/unitree_model/Go2/usd/go2.usd'
assert runtime['policy_sha256'] == '8e3f14ad6aca0cb9f48148375657b2eabef6788fa6ed6f0fa3a0a8ebd30d57d4'
act = runtime['actuators']['GO2HV']
assert act['stiffness'] == 25. and act['damping'] == .5 and act['friction'] == .01

steps = A.review_steps(out/'step_calls.jsonl')
assert steps == {'explicit_step_calls':26400,'returned_calls':26400,'reserve_calls':0,'completed_cases':12}
pre = G.load(out/'pre_cleanup_step_accounting.json')
assert all(pre[k] == v for k,v in steps.items())
assert pre['case_id'] is None and pre['case_ticks'] == 0 and not pre['in_flight']
assert pre['native_internal_initialization_steps'] == 'unobservable; not declared zero'
captured = G.load(out/'CAPTURE_FINISHED.json')
assert len(captured['cases']) == 12
rows = []
for index, case in enumerate(p['schedule']['cases'], 1):
    trial = out/case['id']
    result = G.load(trial/'result.json')
    assert captured['cases'][index-1] == result
    assert result['ticks'] == 2200 and result['completed_q_count'] == 101
    assert result['diagnostic_only'] and not result['physical_navigation_success']
    ledger = result['explicit_step_accounting']
    assert ledger['explicit_step_calls'] == ledger['returned_calls'] == 2200*index
    assert ledger['completed_cases'] == index and ledger['reserve_calls'] == 0
    assert ledger['case_id'] is None and ledger['case_ticks'] == 0 and not ledger['in_flight']
    reset = G.load(trial/'reset.json')
    assert reset['actual_origins'] == [[0.,0.,0.]]
    assert reset['commands_seen'] == [[0.,0.,0.]] and reset['previous_action_max'] == 0.
    # clean_v3.ensure_direct_reset_buffer reports creation once per env, not
    # whether every later reset succeeded. Require the actual cleared buffer
    # on every case and the creation flag only on the first case.
    assert reset['direct_reset_buffer_initialized'] == (index == 1)
    assert reset['reset_buffer_after_reset'] == [False]
    reviewed = A.review_case(trial, case)
    rows.append(reviewed)
    print('CASE_PASS', case['id'], case['command'], reviewed['last_two_seconds_mean_completed_q'], flush=True)

strict = subprocess.run([sys.executable,'-B',str(G.NAV/'scripts/audit_nav043_motion_probe_recovery_v2.py')],
    cwd=ROOT, capture_output=True, text=True, check=False)
assert strict.returncode == 1 and 'no finished physical allocation; do not relaunch' in strict.stderr
G.persist(EV/'strict-review-retains-held.json', {'returncode':strict.returncode,
    'stdout':strict.stdout,'stderr':strict.stderr,'original_gate_changed':False})

first_failure = G.load(G.NAV/'reports/nav043_motion_probe_first_failure_v1.json')
for item in first_failure['saved_files']:
    assert Path(item['path']).stat().st_size == item['bytes'] and G.sha(item['path']) == item['sha256']
assert len(first_failure['saved_files']) == 17
assert manifest(out) == before
assert G.protected_sources() == identities
G.approved()
assert all(not (Path('/proc')/str(pid)).exists() for pid in pids)

source = Path('/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/lib/python3.11/site-packages/isaacsim/exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py')
source_lines = source.read_text().splitlines()
G.persist(EV/'shutdown-source-review.json', {'path':str(source),'sha256':G.sha(source),
    'excerpt_start_line':788,'excerpt_end_line':850,'excerpt':source_lines[787:850],
    'conclusion':'The installed close calls native shutdown_and_release_framework. Saved pre-close evidence exists but the post-close Python ledger does not. Exact native termination mechanism is not established.',
    'installed_source_changed':False,'sealed_capture_source_changed':False,
    'CPU_fake_close_is_not_proof_of_actual_framework_shutdown':True})
G.persist(EV/'held-raw-output-index.json', {'output_root':str(out),'files':before,
    'total_files':len(before),'total_bytes':sum(r['bytes'] for r in before),
    'originals_retained':True,'unchanged_during_review':True})
copies = EV/'held-saved-small'
copies.mkdir(exist_ok=False)
copied = []
keep = {'OWNER.json','PROCESS.json','ENTRY.json','LOCAL_SOURCE.json','APP_READY.json','runtime.json',
    'CAPTURE_FINISHED.json','pre_cleanup_step_accounting.json','HELD.json','process_cleanup.json',
    'reset.json','result.json','energy_metadata.json'}
for item in before:
    path = Path(item['path'])
    if path.name not in keep:
        continue
    target = copies/path.relative_to(out)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as stream:
        stream.write(path.read_bytes())
    assert G.sha(target) == item['sha256']
    copied.append(str(target.relative_to(ROOT)))

report = {'schema':'nav043-motion-probe-held-saved-case-review-v2',
    'status':'SAVED_12_CASE_INTEGRITY_PASS_WITH_TERMINAL_PROTOCOL_HELD',
    'at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    'output_root':str(out),'context':{'hostname':socket.gethostname(),'python':sys.executable,
        'tmux_pane_from_process_environment':os.environ['TMUX_PANE'],
        'tmux':subprocess.check_output(['tmux','display-message','-p','-t',os.environ['TMUX_PANE'],
            '#{session_name} #{window_id} #{pane_id}'],text=True).strip()},
    'proposal_sha256':G.sha(G.PROPOSAL),'approval_sha256':G.sha(G.APPROVAL),'readiness_sha256':G.sha(G.READINESS),
    'identities_checked':identities,'original_failed_files_unchanged':17,
    'case_integrity_passed':True,'strict_original_completion_gate_passed':False,
    'saved_review_first_attempt':{'exit_code':1,'cases_reviewed_before_stop':1,
        'reason':'Reviewer assumed reset buffer creation occurs every reset; clean_v3 creates it only once per env.',
        'source_and_log_retained':True,'physical_data_changed':False,'sealed_gate_changed':False},
    'strict_original_gate_still_rejects':True,'terminal_protocol_state':'HELD',
    'missing_terminal_records_not_backfilled':missing,'application_error_record_exists':False,
    'application_exit_code':0,'coordinator_wrapper_exit_code':1,'process_cleanup':cleanup,
    'registered_processes_absent':pids,'step_accounting_from_journal':steps,
    'pre_cleanup_ledger_matches_journal':True,'native_internal_initialization_steps':'unobservable; not declared zero',
    'actual_ticks_reviewed':26400,'warmup_ticks_reviewed':2400,'measurement_ticks_reviewed':24000,
    'actual_command_proofs_reviewed':6600,'completed_q_intervals_reviewed':1212,
    'current_application_wall_s':exit_info['wall_s'],'failed_slot_conservative_wall_charge_s':900,
    'cumulative_charged_wall_s':900+exit_info['wall_s'],'cumulative_explicit_step_calls':895+26400,
    'cumulative_applications_started':2,'further_applications_authorized':0,
    'raw_file_index':str((EV/'held-raw-output-index.json').relative_to(ROOT)),
    'small_evidence_copies':copied,'all_output_files_unchanged_during_review':True,
    'clean_runtime_readback_review_passed':True,'cases':rows,
    'scope':'Twelve exposed controlled constant-command cases, one attempt each, 10 s measurement; not 90 s navigation or calibration.',
    'independent_unseen_validation':False,'steady_state_or_generalization_claimed':False,
    'new_review_applications':0,'new_response_fits':0,'new_navigation_episodes':0,'new_W_queries':0,
    'new_actor_critic_updates':0,'new_calibration_fits':0,'NAV043_passed':False,'G5_passed':False}
G.persist(G.NAV/'reports/nav043_motion_probe_recovery_held_review_v2.json', report)
print(report['status'], 'cases',len(rows), 'wall_s',report['cumulative_charged_wall_s'], flush=True)
