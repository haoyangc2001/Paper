"""Main-agent saved-failure review and installed-source CPU compatibility checks."""
import ast
from copy import deepcopy
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
NAV=ROOT/'CaMRA/navigation'
sys.path.insert(0,str(NAV/'scripts'))
from nav067_sources import E, load, sha, write_new, read_manifest
import run_nav067_remaining_validation as V1
import run_nav067_remaining_recovery_v2 as V2
from camra_navigation.envs.controlled_terrain_v2 import configure_fixture_plane

assert os.environ.get('TMUX_PANE')=='%599'
assert Path(sys.executable).resolve()==Path(V1.PYTHON).resolve()
torch.set_num_threads(1)
root=V1.EXPECTED_ROOT
error=load(root/'C01/ERROR.json')
held=load(root/'HELD.json')
assert error['message']=='actual clone origins are not separated'
assert held['kit_starts']==1 and held['reserved_environment_ticks']==408
assert not Path('/proc/'+str(load(root/'C01/PROCESS.json')['pid'])).exists()
assert not (root/'C01/FINISHED.json').exists() and not (root/'C01/runtime.json').exists()
assert not (root/'C01/window.json').exists()
assert not V2.APPROVAL.exists() and not (root/'C01/attempt2').exists()
assert all(not (root/s).exists() for s in V1.SLOTS[1:])
checks=[]
def check(name,condition=True):
    assert condition,name
    checks.append(name)

isaac=importlib.util.find_spec('isaaclab')
base=Path(isaac.origin).parent
actual=base/'terrains/terrain_importer.py'
tree=ast.parse(actual.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TerrainImporter')
names=('_compute_env_origins_grid','_compute_env_origins_curriculum','configure_env_origins')
methods=[deepcopy(n) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
check('actual installed source has all three origin methods',len(methods)==3)
namespace={'np':np,'torch':torch}
exec(compile(ast.Module(body=methods,type_ignores=[]),str(actual),'exec'),namespace)
holder=SimpleNamespace(device='cpu',cfg=SimpleNamespace(num_envs=2,env_spacing=63.,max_init_terrain_level=1))
for name in names:
    setattr(holder,name,namespace[name].__get__(holder))
torch.manual_seed(123)
holder.configure_env_origins(np.array([[[7.,11.,0.]]],dtype=np.float32))
old_origins=holder.env_origins.numpy().copy()
check('installed generator single-cell origin reproduces overlapping environments',np.array_equal(old_origins[0],old_origins[1]))
holder.configure_env_origins()
grid=holder.env_origins.numpy().copy()
check('installed grid source returns separated two-env origins',abs(float(np.linalg.norm(grid[0]-grid[1]))-63.)<1.e-6)
for n in (1,2):
    cfg=SimpleNamespace(scene=SimpleNamespace(env_spacing=2.5,terrain=SimpleNamespace(
        terrain_type='generator',terrain_generator=object(),env_spacing=2.5)))
    report=configure_fixture_plane(cfg,63.)
    check('isolated plane config N'+str(n),cfg.scene.terrain.terrain_type=='plane' and cfg.scene.terrain.terrain_generator is None and cfg.scene.env_spacing==63.)
    holder.cfg.num_envs=n
    holder.configure_env_origins()
    check('actual grid method finite N'+str(n),holder.env_origins.shape==(n,3) and torch.isfinite(holder.env_origins).all())
for bad in (0.,-1.,float('nan'),float('inf')):
    try:
        configure_fixture_plane(cfg,bad)
    except ValueError:
        check('invalid spacing rejected '+str(bad))
    else:
        raise AssertionError('invalid spacing accepted')
check('installed plane branch selects grid origins',"self.import_ground_plane(\"terrain\")\n            # configure the origins in a grid\n            self.configure_env_origins()" in actual.read_text())
check('newer option absent in installed importer','use_terrain_origins' not in actual.read_text())
reference=Path('/pub/data/caohy/IsaacLab/source/isaaclab/isaaclab/terrains/terrain_importer.py')
check('other source tree differs and contains the unused option',sha(reference)!=sha(actual) and 'use_terrain_origins' in reference.read_text())
new_source=NAV/'scripts/nav067_controlled_simulation_v2.py'
old_source=NAV/'scripts/nav067_controlled_simulation.py'
old_lines=old_source.read_text().splitlines()
new_lines=new_source.read_text().splitlines()
import difflib
diff='\n'.join(difflib.unified_diff(old_lines,new_lines,fromfile=str(old_source),tofile=str(new_source),lineterm=''))+'\n'
(HERE/'controlled-terrain-recovery.patch').write_text(diff)
for needle in ('low.policy_step(fixture[\'commands\'],monitor=scheduled,recorder=recorder)',
               'review_actual_case(fixture,ctx[\'geom\'],scheduled,energy,low.evidence)',
               'monitor.set_commands(fixture[\'commands\'])'):
    check('production stepping/review retained '+needle,needle in new_source.read_text() and needle in old_source.read_text())
check('origin proof saved before rejecting',new_source.read_text().index("out/'origin_readback.json'")<new_source.read_text().index("raise ValueError('actual clone origins are not separated')"))
new_paths=['CaMRA/navigation/src/camra_navigation/envs/controlled_terrain_v2.py',
           'CaMRA/navigation/scripts/nav067_controlled_simulation_v2.py',
           'CaMRA/navigation/scripts/run_nav067_remaining_recovery_v2.py']
for name in new_paths:
    ast.parse((ROOT/name).read_text())
    check('source syntax '+name)
with tempfile.TemporaryDirectory(prefix='nav067-recovery-basic-') as directory:
    path=Path(directory)/'no-approval.json'
    command=[sys.executable,'-B',str(NAV/'scripts/run_nav067_remaining_recovery_v2.py'),'--check','--approval',str(path)]
    environment=os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES']='6'
    refused=subprocess.run(command,cwd=ROOT,env=environment,capture_output=True,text=True)
    check('production recovery refuses missing approval before AppLauncher',refused.returncode!=0 and 'required approval/execution file absent' in refused.stderr)
    write_new(HERE/'recovery-missing-approval.json',{'command':command,'exit_code':refused.returncode,'stdout':refused.stdout,'stderr':refused.stderr})
manifest,prereg,nets=read_manifest()
execution=load(V1.EXECUTION)
check('original preparation/source identities unchanged',all(sha(ROOT/p)==h for p,h in execution['source_hashes'].items()))
check('all original fixtures unchanged',load(HERE/'fixtures.json')==execution['fixtures'])
check('no new recovery output or application',not V2.APPROVAL.exists() and not (root/'RECOVERY_V2_OWNER.json').exists())
check('explicit proposed cumulative bounds',1+len(V1.SLOTS)==10 and 408+execution['reserved_environment_ticks_max']==57830 and 57830<=60000)
evidence_paths=sorted(p for p in root.rglob('*') if p.is_file())
failure_evidence={str(p):sha(p) for p in evidence_paths}
failure={'status':'HELD_ACTUAL_CLONE_ORIGIN_CONFIGURATION_FAILURE','at':datetime.now().astimezone().isoformat(),
    'task':'NAV-067','plan_version':'1.4.1','NAV_067_status':False,'completed_tasks':54,
    'preparation_commit':'a70214edab165e5d3bb553c1c83932521d574490','original_execution_sha256':sha(V1.EXECUTION),
    'applications_started':1,'controlled_cases_passed':0,'new_navigation_episodes':0,
    'explicit_warmup_policy_calls_reached':0,'controlled_measurement_ticks_reached':0,
    'unobserved_initialization_physics_steps':'not recorded; do not claim zero total simulator steps',
    'error':error,'terminal_coordinator':held,'child_exit':load(root/'C01/EXIT.json'),
    'exit_zero_is_not_success':True,'stopped_on_saved_error':True,'all_attempts_preserved':True,
    'failure_category':'fixture/environment wiring; no inference about policy performance',
    'root_cause':'Preparation used /pub/data/caohy/IsaacLab source instead of the imported /pub/data/caohy/IsaacLab_v0472 installation. The installed generator always samples the single terrain origin and ignores the added newer use_terrain_origins field.',
    'actual_origin_numeric_values':'not saved before the guard raised; exact values unavailable',
    'CPU_source_check_only':{'single_cell_example_origins':old_origins.tolist(),'grid_example_origins':grid.tolist(),
        'not_new_physical_evidence':True},'actual_isaaclab_origin':isaac.origin,'actual_terrain_importer_sha256':sha(actual),
    'failed_cleanup':load(HERE/'failed-child-stop.json'),'failure_evidence_hashes':failure_evidence,
    'unchanged':['selected pair','v2 complete frozen system','old six pilot traces','RC-PPO objective','all benchmark maps/budgets',
                 'sealed v1 sources/fixtures/approval/execution','old scientific FAIL','NAV-043 unexecuted','final_* sealed'],
    'next':'Prepared isolated plane/grid recovery requires explicit one-time approval; no automatic retry or quota reuse.'}
write_new(NAV/'reports/nav067_remaining_validation_v1_execution_review.json',failure)
compatibility={'status':'RECOVERY_BASIC_CPU_PASS_NOT_PHYSICAL_ACCEPTANCE','at':datetime.now().astimezone().isoformat(),
    'python':sys.executable,'tmux':os.environ['TMUX'],'pane':os.environ['TMUX_PANE'],
    'checks':checks,'count':len(checks),'installed_source':str(actual),'installed_sha256':sha(actual),
    'CPU_grid_origins':grid.tolist(),'new_applications':0,'new_physical_ticks':0,
    'source_hashes':{p:sha(ROOT/p) for p in new_paths}}
write_new(HERE/'recovery-verification.json',compatibility)
installed_paths=[base/'terrains/terrain_importer.py',base/'terrains/terrain_importer_cfg.py',
    base/'scene/interactive_scene.py',base/'sensors/contact_sensor/contact_sensor.py',
    base/'sensors/contact_sensor/contact_sensor_cfg.py']
proposal={'schema':'camra-nav067-remaining-recovery-proposal-v2','task':'NAV-067','status':'PROPOSED',
    'automatic_execution_authorized':False,'requested_exception':'Exactly one C01 recovery in its own attempt2 after actual origin-configuration failure; original failure remains immutable.',
    'original_execution_sha256':sha(V1.EXECUTION),'actual_isaaclab_origin':isaac.origin,
    'source_hashes':compatibility['source_hashes'],
    'installed_source_hashes':{str(p):sha(p) for p in installed_paths},
    'evidence_hashes':{**failure_evidence,str((HERE/'recovery-verification.json').relative_to(ROOT)):sha(HERE/'recovery-verification.json'),
        'CaMRA/navigation/reports/nav067_remaining_validation_v1_execution_review.json':sha(NAV/'reports/nav067_remaining_validation_v1_execution_review.json')},
    'previous_gpu_wall_s':held['cumulative_gpu_wall_s'],'limits':{
        'additional_kit_starts_max':9,'cumulative_kit_starts_including_failed_attempt_max':10,
        'cumulative_gpu_wall_s_including_failure_max':14400,'cumulative_environment_ticks_including_failed_reserved_slot_max':60000,
        'exact_cumulative_reserved_environment_ticks':57830,'new_controlled_runs':6,'additional_navigation_episodes':6,
        'controlled_max_envs':2,'controlled_max_s_including_warmup':2.,'repeat_max_s_including_warmup':46.,
        'gpus':1,'concurrent_processes':1,'automatic_retries_after_this_exception':0,
        'training_updates':0,'W_queries':0,'nominal_episodes':0},
    'fixture_change':'Only isolated controlled ground uses the installed plane -> grid API; exact maps/targets/schedules/caps/monitors/actions/energy and all repetition configurations unchanged.',
    'output':'Original root retained: C01/attempt2, then original unstarted C02-C06 and L1/L2/L3 slots. No replacement root.',
    'execution_order':list(V1.SLOTS),'fixed_environment':execution['fixed_environment'],
    'command':[sys.executable,'-B',str(NAV/'scripts/run_nav067_remaining_recovery_v2.py')],
    'failure_policy':'Save and stop on first actual evidence/entry/runtime/trace error. Cleanup a failed application; no automatic fixture edits, retries, replacements or quota transfers.',
    'not_authorized':['NAV-043','final_*','new training','shared Isaac/framework changes','frozen v2 changes','Git push'],
    'acceptance':'All original NAV-067 criteria remain; basic CPU checks do not pass actual controlled/repeatability acceptance.'}
text=json.dumps(proposal,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
proposal_path=NAV/'config/nav067_remaining_validation_recovery_proposal_v2.json'
assert not proposal_path.exists()
patch='*** Begin Patch\n*** Add File: '+str(proposal_path)+'\n'+''.join('+'+line+'\n' for line in text.splitlines())+'*** End Patch\n'
subprocess.run(['apply_patch'],cwd=ROOT,input=patch,text=True,check=True)
print(json.dumps({'status':'RECOVERY_PREPARATION_PASS_NOT_APPROVED','checks':len(checks),
    'proposal_sha256':sha(proposal_path),'new_applications':0,'NAV_067_status':False},ensure_ascii=False),flush=True)
