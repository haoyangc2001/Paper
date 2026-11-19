"""Exploratory NAV-043 actual-state W budgets; derived from pinned NAV-067 v3."""
import sys
from nav067_sources import E, NAV, write_new, sha
from evaluate_navigation_v2 import next_command
from nav043_budget import decide, nominal, outcome


def simulate(row, out, manifest, prereg, nets, device):
    import gymnasium as gym
    import numpy as np
    import torch
    import isaaclab_tasks  # noqa: F401
    import unitree_rl_lab.tasks  # noqa: F401
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
    from unitree_rl_lab.utils.external_command import DEFAULT_LIMITS, ExternalCommandWrapper
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
    from camra_calibration.profiles import sample_applied
    # The existing clean profile implementation stays byte-for-byte unchanged.
    sys.path.insert(0, str(NAV.parent/'calibration/sim'))
    from profiles_sim import configure_env_for_profile, apply_friction, apply_mass
    from camra_navigation.envs.clean_v3 import configure_clean, IsaacSubstepAdapter, rigid_body_transform
    from camra_navigation.envs.physical_v2 import SubstepNavigation
    from camra_navigation.evaluation.recorder import NavEnergyRecorder, offline_recompute
    from camra_navigation.maps.scene_adapter import build_scene
    from pxr import Usd, UsdPhysics, UsdShade

    ctx = E.build_rollout_ctx('full')
    spec = E.M.build(row['level'],row['seed'],ctx['design'])
    if spec.geometry_hash() != row['geometry_hash'] or spec.to_dict() != row['spec']:
        raise ValueError('runtime pilot map identity mismatch')
    layout = build_scene(spec,0,1)
    boxes = [*layout.wall_boxes,*layout.island_boxes]
    cfg = parse_env_cfg('CaMRA-Go2-Velocity',device=device,num_envs=1,
                        use_fabric=True,entry_point_key='env_cfg_entry_point')
    cfg.seed = 100
    cfg.episode_length_s = 46.
    applied = sample_applied('clean',100,duration_s=45.)
    config_proof = configure_clean(cfg,configure_env_for_profile,applied)
    tile = float(layout.env_spacing_m)
    gen = cfg.scene.terrain.terrain_generator
    gen.size,gen.num_rows,gen.num_cols,gen.border_width = (tile,tile),1,1,1.
    gen.sub_terrains = {'flat':MeshPlaneTerrainCfg(size=(tile,tile))}
    cfg.scene.terrain.terrain_type = 'generator'
    cfg.scene.env_spacing = tile
    material = cfg.scene.terrain.physics_material
    material.static_friction,material.dynamic_friction,material.restitution = 1.,1.,0.
    for j,box in enumerate(boxes):
        setattr(cfg.scene,'nav_box_'+str(j),RigidObjectCfg(
            prim_path='{ENV_REGEX_NS}/nav_box_'+str(j),
            spawn=sim_utils.CuboidCfg(size=(box.size_x,box.size_y,box.height),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True)),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(box.center_x,box.center_y,box.z_center),rot=box.quat_wxyz)))
    raw = gym.make('CaMRA-Go2-Velocity',cfg=cfg)
    env = ExternalCommandWrapper(raw,limits=DEFAULT_LIMITS,enforce_limits=True,on_reset='zero')
    u = raw.unwrapped
    robot = u.scene['robot']
    v2 = torch.jit.load(manifest['binding']['controller']['bundle']+'/exported/policy.pt',map_location=device).eval()
    low = IsaacSubstepAdapter(env,v2,map_center=layout.map_center)
    env.reset()
    friction = apply_friction(robot,[0],applied.friction,applied.restitution)
    mass = apply_mass(robot,[0],applied.added_mass_kg)
    if not friction['ok'] or not mass['ok']:
        raise ValueError('clean material/mass readback failed')
    collider_proof=[]
    stage = u.sim.stage
    for j,box in enumerate(boxes):
        body = u.scene['nav_box_'+str(j)]
        p = rigid_body_transform(body)
        target=np.array([box.center_x+low.origins[0,0],box.center_y+low.origins[0,1],box.z_center])
        error=float(np.max(np.abs(p[:3]-target)))
        prim=stage.GetPrimAtPath('/World/envs/env_0/nav_box_'+str(j))
        enabled=[str(node.GetPath()) for node in Usd.PrimRange(prim)
                 if node.HasAPI(UsdPhysics.CollisionAPI) and
                 UsdPhysics.CollisionAPI(node).GetCollisionEnabledAttr().Get()]
        if error>1e-4 or not enabled:
            raise ValueError('actual map collider pose/enabled mismatch')
        collider_proof.append({'box':box.as_dict(),'position_error_m':error,'enabled_collision_prims':enabled})
    ground_materials=[]
    ground=stage.GetPrimAtPath(cfg.scene.terrain.prim_path)
    for node in Usd.PrimRange(ground):
        if not node.HasAPI(UsdPhysics.CollisionAPI):
            continue
        bound,_=UsdShade.MaterialBindingAPI(node).ComputeBoundMaterial(materialPurpose='physics')
        if bound:
            api=UsdPhysics.MaterialAPI(bound.GetPrim())
            actual=[api.GetStaticFrictionAttr().Get(),api.GetDynamicFrictionAttr().Get(),
                    api.GetRestitutionAttr().Get()]
            if actual != [1.,1.,0.]:
                raise ValueError('actual ground material differs from fixed clean profile')
            ground_materials.append({'collision_prim':str(node.GetPath()),
                'material':str(bound.GetPath()),'static_dynamic_restitution':actual})
    if not ground_materials:
        raise ValueError('actual ground material evidence missing')
    runtime={'config':config_proof,'event_modes':u.event_manager.available_modes,
        'event_terms':u.event_manager.active_terms,'friction':friction,'mass':mass,
        'scene':layout.as_dict(),'actual_origins':low.origins.tolist(),'colliders':collider_proof,
        'physics_dt_s':u.sim.get_physics_dt(),'decimation':u.cfg.decimation,
        'ground_material':{'static':material.static_friction,'dynamic':material.dynamic_friction,
                           'restitution':material.restitution},
        'ground_material_usd_readback':ground_materials}
    write_new(out/'runtime.json',runtime)
    reports=[]
    for method in ('uncalibrated','full'):
        trial=out/method
        trial.mkdir()
        reset=low.reset(start=spec.start,yaw=layout.robot_spawn_yaw)
        for _ in range(50):
            low.policy_step([[0.,0.,0.]])
        warmup={'policy_calls':low.policy_calls,'physics_steps':low.tick,
                'state_at_window_start':[vars(s) for s in low.states()]}
        low.evidence=[]
        ctx=E.build_rollout_ctx(method)
        initial=low.states()[0]
        write_new(trial/'warmup.json',{'reset':reset,'warmup':warmup})
        decision=decide(nets[method],spec,ctx,initial,row['z0_J'],prereg,trial)
        if decision['selected_budget_J'] is None:
            result={'method':method,'map_id':spec.map_id,'rejected':True,'decision_reason':decision['reason'],
                    'physical_episode_started':False,'nominal_episode_started':False}
            write_new(trial/'result.json',result)
            reports.append(result)
            continue
        z0=decision['selected_budget_J']
        write_new(trial/'NAVIGATION_STARTED.json',{'decision_sha256':sha(trial/'decision.json'),
            'z0_J':z0,'initial_state':vars(initial)})
        monitor=SubstepNavigation(spec,ctx['geom'],ctx['cm'],[initial],[z0],prereg)
        recorder=NavEnergyRecorder(9000,1,device=device,expected_steps=9000,label=method)
        recorder.begin_window()
        nav_rows=[]
        try:
            while any(monitor.active):
                obs,action,command=next_command(nets[method],monitor)
                nav_rows.append({'navigation_step':len(nav_rows),'physics_tick':monitor.tick,
                    'observation':obs.tolist(),'normalized_action':action.tolist(),'command':command.tolist()})
                for _ in range(5):
                    if not any(monitor.active): break
                    low.policy_step(command,monitor=monitor,recorder=recorder)
        except BaseException:
            write_new(trial/'PARTIAL_substeps.json',monitor.rows)
            write_new(trial/'PARTIAL_commands.json',low.evidence)
            write_new(trial/'PARTIAL_navigation.json',nav_rows)
            raise
        energy=recorder.finish()
        audit=offline_recompute(energy,0)
        np.savez_compressed(trial/'energy.npz',**{k:v for k,v in energy.items() if isinstance(v,np.ndarray)})
        write_new(trial/'substeps.json',monitor.rows)
        write_new(trial/'commands.json',low.evidence)
        write_new(trial/'navigation.json',nav_rows)
        if energy['problems'] or not audit['steps_match'] or audit['rel_diff']>1e-10:
            write_new(trial/'HELD_energy.json',{'problems':energy['problems'],'audit':audit})
            raise ValueError('actual 200Hz energy audit failed')
        result={'method':method,'map_id':spec.map_id,'candidate':next(c for c in manifest['candidates'] if c['variant']==method),
            'reset':reset,'warmup':warmup,'budget_mode':'actual_state_guarded_W','z0_J':z0,'rejected':False,
            'label':monitor.labels[0],'safety_y':monitor.y[0],'t_s':monitor.times[0],
            'z_J_end':monitor.budgets[0].z_J,'internal_model_cost_J':z0-monitor.budgets[0].z_J,
            'measured_joint_work_J':float(energy['total_j'][0]),'energy_audit':audit,
            'terminal':monitor.terminal[0], 'navigation_steps':len(nav_rows),
            'policy_steps':len(low.evidence),'physics_steps':monitor.tick,
            'physical_SR':monitor.labels[0]=='goal_reached',
            'physical_BVR':float(energy['total_j'][0])>z0,
            'internal_safe_and_budget_feasible':monitor.labels[0]=='goal_reached' and monitor.y[0]==0. and monitor.budgets[0].z_J>=0.,
            'performance_evidence':False}
        result.update(outcome(result['label'],result['safety_y'],result['t_s'],
            result['measured_joint_work_J'],result['internal_model_cost_J'],z0))
        write_new(trial/'result.json',result)
        write_new(trial/'NOMINAL_STARTED.json',{'z0_J':z0,'initial_state':vars(initial)})
        nominal_result=nominal(nets[method],spec,ctx,initial,z0,prereg)
        write_new(trial/'nominal.json',nominal_result)
        reports.append(result)
        print('EPISODE',method,spec.map_id,result['label'],result['t_s'],result['measured_joint_work_J'],flush=True)
    write_new(out/'FINISHED.json',{'episodes':reports,'scientific_acceptance':False,'purpose':'exploratory actual-state W budget execution',
        'repeatability_measured':False,'controlled_actual_cases_measured':False})
    env.close()
    return 0

