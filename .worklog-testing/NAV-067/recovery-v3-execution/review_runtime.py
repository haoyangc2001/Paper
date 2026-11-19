"""Audit saved runtime readbacks only; never import or launch a simulator."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
OUTPUT = Path('/pub/data/caohy/work/camra_navigation/nav067_clean_pilot_v1')
assert os.environ.get('TMUX_PANE')
level = sys.argv[1]
manifest = json.loads((OUTPUT/'manifest.json').read_bytes())
row = next(r for r in manifest['maps'] if r['level'] == level)
trial = OUTPUT/level
if level == 'L1_empty': trial = trial/'attempt2'
source = trial/'runtime.json'
raw = source.read_bytes()
r = json.loads(raw)
c = r['config']
assert c['profile_id'] == 'clean'
assert c['obs_corruption_enabled'] is False and c['obs_noise'] is None
assert set(c['removed_event_terms']) == {'physics_material','add_base_mass','base_external_force_torque','push_robot'}
for key in ('reset_base_pose_range','reset_base_velocity_range'):
    assert set(c[key]) == {'x','y','z','roll','pitch','yaw'}
    assert all(v == [0.,0.] for v in c[key].values())
assert c['reset_joint_scale_range'] == [1.,1.] and c['reset_joint_velocity_range'] == [0.,0.]
assert c['command_resampling_disabled'] is True and c['automatic_reset_disabled'] is True
assert r['event_modes'] == ['reset']
assert r['event_terms'] == {'reset':['reset_base','reset_robot_joints']}
assert r['physics_dt_s'] == .005 and r['decimation'] == 4
for field in ('friction','mass'):
    assert r[field]['ok'] is True and r[field]['checked_envs'] == 1
for field in ('static_match','dynamic_match','restitution_match'):
    assert r['friction'][field] is True
assert r['mass']['mass_match'] is True and r['mass']['inertia_match'] is True
assert r['mass']['delta_kg'] == 0.
assert r['ground_material'] == {'static':1.,'dynamic':1.,'restitution':0.}
assert r['ground_material_usd_readback']
assert all(m['static_dynamic_restitution'] == [1.,1.,0.] for m in r['ground_material_usd_readback'])
assert r['scene']['map_id'] == row['spec']['map_id']
assert r['scene']['level'] == level and r['scene']['num_envs'] == 1
assert len(r['actual_origins']) == 1
expected = r['scene']['wall_boxes'] + r['scene']['island_boxes']
assert len(expected) == len(r['colliders'])
for box,readback in zip(expected,r['colliders']):
    assert box == readback['box']
    assert 0. <= readback['position_error_m'] <= 1e-4
    assert readback['enabled_collision_prims']
report = {'status':'PASS_SAVED_RUNTIME_READBACK_ONLY','at':datetime.now().astimezone().isoformat(),
    'level':level,'runtime':str(source),'runtime_sha256':hashlib.sha256(raw).hexdigest(),
    'tmux':subprocess.check_output(['tmux','display-message','-t',os.environ['TMUX_PANE'],
        '-p','#{session_name}:#{window_name} #{window_id} #{pane_id}'],text=True).strip(),
    'python':sys.executable,'enabled_colliders':len(expected),'event_terms':r['event_terms'],
    'new_simulation_starts':0,'training_updates':0,'W_queries':0,
    'limitations':['Saved config plus runtime material/mass and collider pose/enabled readbacks only.',
        'Collider size and actual controlled collision responses are not independently measured by this check.',
        'This is not repeatability or NAV-043 physical applicability acceptance.']}
with (HERE/('runtime-review-'+level+'.json')).open('x') as stream:
    json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
