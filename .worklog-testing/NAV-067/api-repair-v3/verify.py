"""Static installed-API and basic regression verification; no Kit or rollout."""
import ast
from datetime import datetime
import difflib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from nav067_sources import NAV, ROOT, OUTPUT, read_manifest, sha, write_new

OUT = Path(__file__).resolve().parent
ISAAC = Path('/pub/data/caohy/IsaacLab_v0472/source/isaaclab/isaaclab')
assert socket.gethostname().split('.')[0] == 'tiger11'
assert os.environ.get('TMUX_PANE') == '%587'
manifest, _, _ = read_manifest()
plan = json.loads((ROOT/'doc/plan/20261109_camra_navigation_implementation.json').read_bytes())
state = {t['id']:t['status'] for t in plan['tasks']}
ready = next(t['id'] for t in plan['tasks'] if not t['status'] and all(state[d] for d in t['depends_on']))
assert ready == 'NAV-067'
assert sum(state.values()) == 54

api_paths = {
    'rigid_object':'assets/rigid_object/rigid_object.py',
    'rigid_data':'assets/rigid_object/rigid_object_data.py',
    'manager_env':'envs/manager_based_env.py',
    'manager_rl_env':'envs/manager_based_rl_env.py',
    'observation_manager':'managers/observation_manager.py',
    'action_manager':'managers/action_manager.py'}
sources = {k:(ISAAC/v).read_text() for k,v in api_paths.items()}
assert 'def root_physx_view(self) -> physx.RigidBodyView:' in sources['rigid_object']
assert 'self._root_physx_view.get_transforms().clone()' in sources['rigid_data']
assert 'get_root_transforms' not in sources['rigid_data']
rl_tree = ast.parse(sources['manager_rl_env'])
buf_writes = []
for node in ast.walk(rl_tree):
    if isinstance(node, ast.FunctionDef):
        for child in ast.walk(node):
            if isinstance(child, ast.Assign) and any(
                isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                and t.value.id == 'self' and t.attr == 'reset_buf' for t in child.targets):
                buf_writes.append({'function':node.name,'line':child.lineno})
assert buf_writes == [{'function':'step','line':205}], buf_writes
assert 'self.reset_buf' not in sources['manager_env']
assert 'def compute(self, update_history: bool = False)' in sources['observation_manager']
assert 'def apply_action(self)' in sources['action_manager']

# The new simulator must differ only at the two explicitly reviewed API sites.
old = (NAV/'scripts/evaluate_navigation_v2.py').read_text()
old_sim = old[old.index('def simulate('):old.index("\n\nif __name__=='__main__':")].rstrip()
new = (NAV/'scripts/nav067_simulation_v3.py').read_text()
new_sim = new[new.index('def simulate('):].rstrip()
expected = old_sim.replace(
    'from camra_navigation.envs.clean_v2 import configure_clean, IsaacSubstepAdapter',
    'from camra_navigation.envs.clean_v3 import configure_clean, IsaacSubstepAdapter, rigid_body_transform').replace(
    'p = body.root_physx_view.get_root_transforms()[0].detach().cpu().numpy()',
    'p = rigid_body_transform(body)')
assert new_sim == expected
with (OUT/'simulator-diff.txt').open('x') as stream:
    stream.write(''.join(difflib.unified_diff(old_sim.splitlines(True), new_sim.splitlines(True),
                                            fromfile='sealed-v2',tofile='api-repair-v3')))
paths = ('scripts/evaluate_navigation_v4.py','scripts/nav067_simulation_v3.py',
         'src/camra_navigation/envs/clean_v3.py','tests/test_clean_v3.py')
for rel in paths:
    ast.parse((NAV/rel).read_text(),filename=rel)

tests = [sys.executable,'-B','-m','pytest','-q',
    'CaMRA/navigation/tests/test_clean_v3.py',
    'CaMRA/navigation/tests/test_nav067_runtime.py',
    'CaMRA/navigation/tests/test_nav067_binding.py',
    'CaMRA/navigation/tests/test_physical_v2.py']
with (OUT/'basic-tests.log').open('x') as stream:
    subprocess.run(tests,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)

old_proposal = NAV/'config/nav067_startup_recovery_proposal_v2.json'
assert sha(old_proposal) == '241fcff15a15003e9e0f4fb621e8576b9dc441899918b55b385b5bc45cc04436'
oldp = json.loads(old_proposal.read_bytes())
for rel,digest in oldp['source_hashes'].items():
    assert sha(NAV/rel) == digest, rel
for slot in oldp['slots']:
    assert not (OUTPUT/slot['relative_output']).exists()
assert not (NAV/'config/nav067_startup_recovery_v2.json').exists()
assert not (NAV/'config/nav067_startup_recovery_v3.json').exists()
result = {
    'status':'PASS_STATIC_AND_BASIC_ONLY','timestamp':datetime.now().astimezone().isoformat(),
    'host':socket.getfqdn(),'python':sys.executable,'cwd':str(ROOT),
    'tmux':subprocess.check_output(['tmux','display-message','-t',os.environ['TMUX_PANE'],
        '-p','#{session_name}:#{window_name} #{window_id} #{pane_id}'],text=True).strip(),
    'plan_version':plan['document']['version'],'fully_parsed_tasks':len(plan['tasks']),
    'fully_parsed_records':sum(len(t['work_records']) for t in plan['tasks']),
    'protected_hashes_checked':295,'completed_tasks':54,'current_task':ready,
    'installed_api_source_hashes':{str(ISAAC/v):sha(ISAAC/v) for v in api_paths.values()},
    'manager_reset_buffer_assignments':buf_writes,
    'rigid_body_backend_api':'RigidBodyView.get_transforms (native xyz/xyzw)',
    'changed_simulator_sites':['clean_v3 adapter import','rigid-body transform read'],
    'original_four_sources_manifest_and_v2_proposal_unchanged':True,
    'new_source_hashes':{rel:sha(NAV/rel) for rel in paths},
    'tests_command':tests,'tests_output':(OUT/'basic-tests.log').read_text(),
    'new_apps':0,'new_physics_steps':0,'new_navigation_episodes':0,
    'new_training_updates':0,'new_W_queries':0,
    'limitations':['Static installed-source and basic tests only; no actual robot validation.',
        'Original attempt traceback is absent; these defects are not a reconstruction of it.',
        'Explicit startup recovery decision and controlled/repetition quota are still pending.']}
write_new(OUT/'verification.json', result)
print(json.dumps(result,ensure_ascii=False,indent=2))
