"""Describe saved L1 pilot evidence without inference, fitting or simulation."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUT = Path('/pub/data/caohy/work/camra_navigation/nav067_clean_pilot_v1/L1_empty/attempt2')
assert os.environ.get('TMUX_PANE')
rows = []
for method in ('uncalibrated', 'full'):
    trial = OUTPUT / method
    result = json.loads((trial / 'result.json').read_bytes())
    steps = json.loads((trial / 'substeps.json').read_bytes())
    state = np.asarray([s['state'] for s in steps], dtype=np.float64)
    commands = np.asarray([s['applied'] for s in steps], dtype=np.float64)
    duration, budget = result['t_s'], result['z0_J']
    assert abs(duration - len(steps) * .005) < 1e-8
    initial = result['warmup']['state_at_window_start'][0]
    yaw = np.unwrap(np.r_[initial['yaw'], state[:, 2]])
    actual_work = result['measured_joint_work_J']
    internal = result['internal_model_cost_J']
    rows.append({
        'method': method, 'budget_mode': result['budget_mode'],
        'label': result['label'], 'duration_s': duration, 'safety_y': result['safety_y'],
        'budget_J': budget, 'actual_work_J': actual_work,
        'actual_overrun_J': actual_work - budget,
        'actual_overrun_fraction': actual_work / budget - 1.,
        'internal_cost_J': internal, 'internal_overrun_J': internal - budget,
        'actual_mean_positive_joint_power_W': actual_work / duration,
        'reference_start_yaw_rad': initial['yaw'],
        'reference_end_yaw_rad': steps[-1]['ref_pose'][2],
        'actual_unwrapped_yaw_change_rad': float(yaw[-1] - yaw[0]),
        'commanded_angular_integral_rad': float(commands[:, 2].sum() * .005),
        'actual_body_wz_integral_rad': float(state[:, 5].sum() * .005),
        'mean_command_body_velocity': commands.mean(axis=0).tolist(),
        'mean_actual_body_velocity': state[:, 3:6].mean(axis=0).tolist(),
        'terminal_goal_distance_m': steps[-1]['goal_distance_m'],
        'maximum_reference_position_deviation_m': max(s['deviation_m'] for s in steps),
        'input_sha256': {name: hashlib.sha256((trial / name).read_bytes()).hexdigest()
                         for name in ('result.json', 'substeps.json')},
    })
report = {
    'at': datetime.now().astimezone().isoformat(),
    'status': 'OBSERVATIONAL_DIAGNOSTICS_ONLY',
    'tmux': subprocess.check_output(['tmux', 'display-message', '-t', os.environ['TMUX_PANE'],
        '-p', '#{session_name}:#{window_name} #{window_id}/#{pane_id}'], text=True).strip(),
    'episodes': rows,
    'new_simulation_starts': 0, 'training_updates': 0, 'W_queries': 0,
    'interpretation_limits': [
        'Pilot common budgets are not budgets selected by the frozen W rule.',
        'Two episodes do not estimate method success rates or establish a cause.',
        'Yaw change and body angular velocity integral use different coordinate quantities.',
        'A budget overrun does not prove that every policy is infeasible at that budget.',
        'Reference drift is descriptive and does not replace the frozen-margin applicability gate.',
    ],
}
with (HERE / 'L1-observational-diagnostics.json').open('x') as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
