"""Read already saved evidence; do not start or alter the running simulator."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = Path('/pub/data/caohy/work/camra_navigation/nav067_clean_pilot_v1')
assert os.environ.get('TMUX_PANE')
assert len(sys.argv) == 4
level, attempt, method = sys.argv[1:]
assert level in ('L1_empty','L2_single_detour','L3_multi_path')
assert attempt in ('attempt2','direct') and method in ('uncalibrated','full')
trial = OUTPUT/level
if attempt != 'direct': trial = trial/attempt
trial = trial/method
result = json.loads((trial/'result.json').read_bytes())
steps = json.loads((trial/'substeps.json').read_bytes())
commands = json.loads((trial/'commands.json').read_bytes())
navigation = json.loads((trial/'navigation.json').read_bytes())
with np.load(trial/'energy.npz',allow_pickle=False) as archive:
    energy = {k:archive[k] for k in archive.files}
n = result['physics_steps']
assert n == len(steps) and n > 0
assert [r['physics_tick'] for r in steps] == list(range(1,n+1))
np.testing.assert_allclose([r['t_s'] for r in steps],np.arange(1,n+1)*.005,atol=1e-8,rtol=0)
assert steps[-1] == result['terminal']
assert steps[-1]['label'] == result['label'] and all(r['label'] is None for r in steps[:-1])
assert commands[0]['tick_before'] == 200
assert result['warmup']['physics_steps'] == 200 and result['warmup']['policy_calls'] == 50
assert result['reset']['previous_action_max'] == 0.
assert len(commands) == (n+3)//4 and len(navigation) == (n+19)//20
assert sum(c['substeps'] for c in commands) == n
for j,c in enumerate(commands):
    assert 1 <= c['substeps'] <= 4
    assert c['tick_after']-c['tick_before'] == c['substeps']
    if j: assert commands[j-1]['tick_after'] == c['tick_before']
    np.testing.assert_allclose(c['command_written'],c['command_seen'],atol=1e-6,rtol=0)
    assert c['max_error'] <= 1e-6
    assert abs(c['sim_time_after']-c['sim_time_before']-.005*c['substeps']) <= 1e-7
    nav = navigation[j//5]
    np.testing.assert_allclose(nav['command'],c['command_written'],atol=1e-7,rtol=0)
tau = energy['tau_applied'].astype(np.float64)
velocity = energy['qdot_post'].astype(np.float64)
valid = energy['valid_mask'].astype(bool)
assert tau.shape[0] == n and valid.shape == (n,1) and valid.all()
actual = float((np.maximum(tau*velocity,0).sum(axis=-1)*.005*valid).sum())
assert abs(actual-result['measured_joint_work_J']) < 1e-9*max(1.,actual)
assert abs(result['z0_J']-steps[-1]['z_J']-result['internal_model_cost_J']) < 1e-8
assert max(s['safety_y'] for s in steps) == result['safety_y']
assert bool(result['physical_BVR']) == (actual > result['z0_J'])
assert not result['performance_evidence']
state = np.asarray([s['state'] for s in steps])
applied = np.asarray([s['applied'] for s in steps])
report = {'at':datetime.now().astimezone().isoformat(),
    'status':'PASS_SAVED_EPISODE_TRACE_ONLY','level':level,'method':method,'trial':str(trial),
    'tmux':subprocess.check_output(['tmux','display-message','-t',os.environ['TMUX_PANE'],
        '-p','#{session_name}:#{window_name} #{window_id} #{pane_id}'],text=True).strip(),
    'python':sys.executable,'physics_steps':n,'policy_steps':len(commands),'navigation_steps':len(navigation),
    'label':result['label'],'safety_y':result['safety_y'],'duration_s':result['t_s'],
    'budget_J':result['z0_J'],'measured_joint_work_J':actual,'internal_cost_J':result['internal_model_cost_J'],
    'mechanical_work_recompute_difference_J':actual-result['measured_joint_work_J'],
    'terminal_goal_distance_m':steps[-1]['goal_distance_m'],
    'mean_command_body_velocity':applied.mean(axis=0).tolist(),
    'mean_actual_body_velocity':state[:,3:6].mean(axis=0).tolist(),
    'body_velocity_tracking_RMSE':np.sqrt(((applied-state[:,3:6])**2).mean(axis=0)).tolist(),
    'max_reference_deviation_m':max(s['deviation_m'] for s in steps),
    'new_simulation_starts':0,'training_updates':0,'W_queries':0,'NAV_067_status':False,
    'limitations':['One pilot episode only; no method performance acceptance or causal attribution.',
        'Actual controlled counterexamples and same-configuration repeatability remain unverified.']}
name = HERE/('main-agent-'+level+'-'+method+'.json')
with name.open('x') as stream: json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
