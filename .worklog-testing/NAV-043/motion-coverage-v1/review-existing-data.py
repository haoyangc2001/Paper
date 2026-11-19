"""Inspect frozen training-command metadata and saved summaries; never fit/run."""
from collections import Counter
from datetime import datetime
import os
from pathlib import Path
import socket
import sys

import numpy as np

ROOT = Path('/home/caohy/repositories/Paper')
sys.path.insert(0,str(ROOT/'CaMRA/navigation/scripts'))
import nav043_affine_response as affine
from camra_calibration.commands import build_program

base = affine.base
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert Path(sys.executable).resolve() == Path(base.PYTHON).resolve()
affine.approval()
base.verify_protocol()
lock_path = ROOT/'CaMRA/calibration/config/data_lock_v1.json'
lock = base.load(lock_path)
entry = lock['trial_lists']['files']['baseline_fit']
metadata_path = ROOT/'CaMRA/calibration/experiments/hira_go2_18500/v2_lists/baseline_fit.jsonl'
assert base.sha(metadata_path) == entry['sha256']
power_path = ROOT/'CaMRA/calibration/config/power_model_v1.json'
assert base.load(power_path)['train']['list_sha256'] == entry['sha256']
source = ROOT/'CaMRA/calibration/src/camra_calibration/commands.py'
assert base.sha(source) == lock['code']['files']['src/camra_calibration/commands.py']
specs = [__import__('json').loads(row) for row in metadata_path.read_text().splitlines()]
assert len(specs) == entry['count'] == 240 and {r['split'] for r in specs} == {'baseline_fit'}
commands = []
for spec in specs:
    assert spec['profile'] == 'clean'
    program = build_program(spec['family'],spec['seed'],duration_s=spec['duration_s'],
                            command_dt=spec['command_dt'])
    assert program.sha256() == spec['program_sha256']
    commands.append(np.asarray(program.commands[:-1],dtype=float))
command = np.concatenate(commands)
assert command.shape == (72000,3)
summary_path = base.NAV/'reports/low_speed_energy_executability.json'
summary = base.load(summary_path)
assert summary['task'] == 'NAV-059' and len(summary['attempts']) == 51
saved_fields = sorted(set().union(*(row.keys() for row in summary['attempts'])))
assert not {'substeps','root_pos_w','root_quat_w','poses','state_series'} & set(saved_fields)
existing = {
    'declared_training_list': {'path':entry['path'],'exists':Path(entry['path']).exists()},
    'declared_training_trials_root': {'path':'/pub/data/caohy/logs/camra/trials/baseline_fit',
        'exists':Path('/pub/data/caohy/logs/camra/trials/baseline_fit').exists()}}
assert all(not row['exists'] for row in existing.values())
base.verify_protocol()
result = {
    'status':'READ_ONLY_DATA_AVAILABILITY_AND_TEMPLATE_COVERAGE',
    'at':datetime.now().astimezone().isoformat(),'host':socket.gethostname(),
    'pane':os.environ['TMUX_PANE'],'python':sys.executable,
    'source_hashes':{str(path.relative_to(ROOT)):base.sha(path) for path in
        (lock_path,metadata_path,power_path,source,summary_path,Path(__file__))},
    'training_command_metadata':{
        'exact_original_list_sha256':entry['sha256'],'trial_specs':len(specs),
        'families':dict(Counter(row['family'] for row in specs)),
        'deterministically_rebuilt_templates':240,
        'actual_interval_commands_excluding_unused_endpoints':len(command),
        'minimum_command':command.min(axis=0).tolist(),
        'maximum_command':command.max(axis=0).tolist(),
        'zero_command_intervals':int(np.all(command==0.,axis=1).sum()),
        'forward_abs_below_01_intervals':int((np.abs(command[:,0])<.1).sum()),
        'meaning':'Exact frozen v2 training-command templates only; no actual v2 state trajectory or additional simulator observation.'},
    'authoritative_paths_checked':existing,
    'availability_scope':'Only the declared original training paths were checked. No assertion that copies cannot exist elsewhere; no other host or heldout/final data inspected.',
    'saved_low_speed_report':{'attempts':51,'saved_per_attempt_fields':saved_fields,
        'contains_full_stamped_pose_sequence':False,
        'suffices_to_reconstruct_completed_q':False,
        'collector_source':'CaMRA/navigation/scripts/check_low_speed_energy.py',
        'collector_source_sha256':base.sha(base.NAV/'scripts/check_low_speed_energy.py'),
        'meaning':'Saved aggregate report cannot reconstruct 0.1-second SE(2) increments or earlier warmup endpoints.'},
    'protected_identities_checked':295,'raw_navigation_hashes_checked':90,
    'new_fits':0,'new_state_replays':0,'new_business_episodes':0,'W_queries':0,
    'network_updates':0,'new_physics_steps':0,'GPU_use':False,
    'NAV043_passed':False,'G5_passed':False}
base.write_new(base.NAV/'reports/nav043_motion_data_availability_v1.json',result)
print(result['training_command_metadata'])
print(existing)
print('PASS: metadata/templates only; no usable new stamped physical trajectories, no fitting or simulation.')
