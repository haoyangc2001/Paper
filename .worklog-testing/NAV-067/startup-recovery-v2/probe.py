"""One Kit/import/config probe, with no environment, reset or physics step."""
from datetime import datetime
import importlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback

from nav067_sources import NAV, ROOT, read_manifest, sha, write_new

manifest, prereg, _ = read_manifest()
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
out = Path(__file__).resolve().parent/'probe-attempt1'
out.mkdir(exist_ok=False)
stages = []


def record(stage, **values):
    item = {'stage':stage,'timestamp':datetime.now().astimezone().isoformat(),**values}
    stages.append(item)
    with (out/'stages.jsonl').open('a') as stream:
        stream.write(json.dumps(item,ensure_ascii=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(item,ensure_ascii=False),flush=True)


app = None
code = 1
record('STARTED', pid=os.getpid(), python=sys.executable,
       git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       source_sha256=sha(__file__), binding=manifest['binding'])
try:
    source = (NAV.parent/'locomotion/source/unitree_rl_lab').resolve()
    assert (source/'unitree_rl_lab/__init__.py').is_file()
    sys.path.insert(0,str(source))
    spec = importlib.util.find_spec('unitree_rl_lab')
    assert spec and Path(spec.origin).resolve()==source/'unitree_rl_lab/__init__.py'
    record('LOCAL_SOURCE_BOUND', origin=spec.origin)
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True,device='cuda:0')
    app = launcher.app
    record('APP_READY')
    # Imports mirror the production path, while no gym environment is created.
    for name in ('gymnasium','isaaclab_tasks','unitree_rl_lab.tasks','isaaclab.sim',
                 'unitree_rl_lab.utils.external_command','unitree_rl_lab.utils.parser_cfg',
                 'camra_navigation.envs.clean_v2','camra_navigation.envs.physical_v2',
                 'camra_navigation.evaluation.recorder','camra_navigation.maps.scene_adapter'):
        importlib.import_module(name)
        record('IMPORTED', module=name)
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
    from camra_calibration.profiles import sample_applied
    from camra_navigation.envs.clean_v2 import configure_clean
    sys.path.insert(0,str(NAV.parent/'calibration/sim'))
    from profiles_sim import configure_env_for_profile
    cfg = parse_env_cfg('CaMRA-Go2-Velocity',device='cuda:0',num_envs=1,
                        use_fabric=True,entry_point_key='env_cfg_entry_point')
    applied = sample_applied('clean',100,duration_s=45.)
    config_proof = configure_clean(cfg,configure_env_for_profile,applied)
    record('CONFIG_READY', clean_proof=config_proof,
           physics_dt_s=cfg.sim.dt,decimation=cfg.decimation,
           robot_usd=cfg.scene.robot.spawn.usd_path)
    code = 0
    result = {'status':'PASS_IMPORT_AND_CONFIG_ONLY','stages':stages,
              'new_navigation_episodes':0,'new_physics_steps':0,
              'new_training_updates':0,'new_W_queries':0,
              'limits':'No robot environment or physical acceptance; original pilot slots retained.'}
except BaseException as error:
    detail = traceback.format_exc()
    print(detail,file=sys.stderr,flush=True)
    record('FAILED', error_type=type(error).__name__,message=str(error),traceback=detail)
    result = {'status':'FAIL_IMPORT_OR_CONFIG','stages':stages,
              'new_navigation_episodes':0,'new_physics_steps':0,
              'new_training_updates':0,'new_W_queries':0}
finally:
    # Kit shutdown can terminate Python, so persist every exception first.
    write_new(out/'RESULT.json',result)
    record('RESULT_PERSISTED', intended_exit_code=code)
    if app is not None:
        app.close()
raise SystemExit(code)
