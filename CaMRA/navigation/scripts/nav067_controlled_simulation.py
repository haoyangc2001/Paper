"""Bounded synthetic event/contact fixtures using the unchanged v2 physical loop."""
import sys
import numpy as np
from nav067_sources import NAV, E, write_new
from nav067_collider_readback import read_box, read_ground, TOLERANCES


def simulate(fixture, out, manifest, prereg, device):
    import torch
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import unitree_rl_lab.tasks  # noqa: F401
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
    from unitree_rl_lab.utils.external_command import DEFAULT_LIMITS, ExternalCommandWrapper
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
    from camra_calibration.profiles import sample_applied
    sys.path.insert(0,str(NAV.parent/'calibration/sim'))
    from profiles_sim import configure_env_for_profile, apply_friction, apply_mass
    from camra_navigation.envs.clean_v3 import configure_clean, IsaacSubstepAdapter
    from camra_navigation.envs.physical_v2 import SubstepNavigation
    from camra_navigation.envs.controlled_cases_v1 import fixture_spec, ScheduledMonitor, review_actual_case
    from camra_navigation.evaluation.recorder import NavEnergyRecorder, offline_recompute
    from camra_navigation.maps.scene_adapter import build_scene

    n=fixture['num_envs']
    spec=fixture_spec(fixture)
    layout=build_scene(spec,0,1)  # pure env-local boxes; clone origins are added once
    boxes=[*layout.wall_boxes,*layout.island_boxes]
    cfg=parse_env_cfg('CaMRA-Go2-Velocity',device=device,num_envs=n,
                      use_fabric=True,entry_point_key='env_cfg_entry_point')
    cfg.seed=100
    cfg.episode_length_s=2.
    applied=sample_applied('clean',100,duration_s=1.)
    proof=configure_clean(cfg,configure_env_for_profile,applied)
    tile=float(layout.env_spacing_m)
    gen=cfg.scene.terrain.terrain_generator
    # A flat mesh covering all isolated clones, with grid origins instead of
    # sampling both robots onto the same single generated terrain origin.
    ground_size=tile*n
    gen.size,gen.num_rows,gen.num_cols,gen.border_width=(ground_size,ground_size),1,1,1.
    gen.sub_terrains={'flat':MeshPlaneTerrainCfg(size=(ground_size,ground_size))}
    cfg.scene.terrain.terrain_type='generator'
    cfg.scene.terrain.use_terrain_origins=False
    cfg.scene.terrain.env_spacing=tile
    cfg.scene.env_spacing=tile
    material=cfg.scene.terrain.physics_material
    material.static_friction,material.dynamic_friction,material.restitution=1.,1.,0.
    for j,box in enumerate(boxes):
        setattr(cfg.scene,'nav_box_'+str(j),RigidObjectCfg(
            prim_path='{ENV_REGEX_NS}/nav_box_'+str(j),
            spawn=sim_utils.CuboidCfg(size=(box.size_x,box.size_y,box.height),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(box.center_x,box.center_y,box.z_center),rot=box.quat_wxyz)))
    if fixture['case_id']=='C06':
        # A single base body with a specific obstacle filter. Ground net force is not used.
        cfg.scene.fixture_contact=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/base',
            filter_prim_paths_expr=['{ENV_REGEX_NS}/nav_box_4'],update_period=.005,history_length=1)
    raw=gym.make('CaMRA-Go2-Velocity',cfg=cfg)
    env=ExternalCommandWrapper(raw,limits=DEFAULT_LIMITS,enforce_limits=True,on_reset='zero')
    u=raw.unwrapped
    robot=u.scene['robot']
    policy=torch.jit.load(manifest['binding']['controller']['bundle']+'/exported/policy.pt',map_location=device).eval()
    low=IsaacSubstepAdapter(env,policy,map_center=layout.map_center)
    env.reset()
    ids=list(range(n))
    friction=apply_friction(robot,ids,applied.friction,applied.restitution)
    mass=apply_mass(robot,ids,applied.added_mass_kg)
    if not friction['ok'] or not mass['ok']:
        raise ValueError('controlled clean material/mass readback failed')
    if n>1 and np.linalg.norm(low.origins[0,:2]-low.origins[1,:2])<tile-1.e-4:
        raise ValueError('actual clone origins are not separated')
    runtime={'config':proof,'friction':friction,'mass':mass,'event_terms':u.event_manager.active_terms,
             'event_modes':u.event_manager.available_modes,'physics_dt_s':u.sim.get_physics_dt(),
             'decimation':u.cfg.decimation,'actual_origins':low.origins.tolist(),'scene':layout.as_dict(),
             'ground':read_ground(u.sim.stage,cfg.scene.terrain.prim_path),
             'colliders':[read_box(u.sim.stage,u.scene['nav_box_'+str(j)],b,low.origins,i)
                          for i in range(n) for j,b in enumerate(boxes)],'tolerances':TOLERANCES}
    write_new(out/'runtime.json',runtime)
    reset=low.reset(start=spec.start,yaw=spec.start_yaw)
    for _ in range(50):
        low.policy_step([[0.,0.,0.]]*n)
    warmup={'physics_ticks':low.tick,'policy_calls':low.policy_calls,'actual_states':[vars(s) for s in low.states()]}
    writes=[]
    def place(poses):
        # Test input only. Actual post-physics readbacks are saved independently.
        pose=robot.data.root_state_w[:,:7].clone()
        for i,(x,y,yaw) in enumerate(poses):
            pose[i,0]=float(x-low.center[0]+low.origins[i,0])
            pose[i,1]=float(y-low.center[1]+low.origins[i,1])
            pose[i,3:7]=torch.tensor([np.cos(yaw/2),0.,0.,np.sin(yaw/2)],device=device)
        robot.write_root_pose_to_sim(pose)
        robot.write_root_velocity_to_sim(torch.zeros((n,6),device=device))
        robot.update(0.)
        actual=low.states()
        for i,s in enumerate(actual):
            if max(abs(s.x-poses[i][0]),abs(s.y-poses[i][1]))>TOLERANCES['pose_roundtrip_m']:
                raise ValueError('actual world/map pose roundtrip failed')
        writes.append({'requested_map_pose':poses,'written_world_pose':pose.detach().cpu().tolist(),
                       'immediate_actual_readback':[vars(s) for s in actual]})
    place(fixture['initial'])
    ctx=E.build_rollout_ctx('full')
    monitor=SubstepNavigation(spec,ctx['geom'],ctx['cm'],low.states(),[1000.]*n,prereg)
    monitor.set_commands(fixture['commands'])
    def contact():
        sensor=u.scene['fixture_contact']
        forces=sensor.data.force_matrix_w
        if forces is None or tuple(forces.shape[:2])!=(n,1) or not torch.isfinite(forces).all():
            raise ValueError('actual filtered native contact tensor is absent or malformed')
        return {'sensor_body':'Robot/base','filter':'nav_box_4','shape':list(forces.shape),
                'force_matrix_w_N':forces.detach().cpu().tolist(),
                'robot_obstacle_pair_force_norm_N':float(torch.linalg.vector_norm(forces[0]).item())}
    scheduled=ScheduledMonitor(monitor,fixture,place,contact if fixture['case_id']=='C06' else None)
    recorder=NavEnergyRecorder(fixture['measurement_tick_cap'],n,device=device,label=fixture['case_id'])
    recorder.begin_window()
    low.evidence=[]
    scheduled.prepare_tick(1)
    error=None
    try:
        while any(scheduled.active) and monitor.tick<fixture['measurement_tick_cap']:
            # All caps are full cycles unless asynchronous termination stops the cycle.
            low.policy_step(fixture['commands'],monitor=scheduled,recorder=recorder)
    except BaseException as exc:
        error=exc
    finally:
        energy=recorder.finish()
        np.savez_compressed(out/'energy.npz',**{k:v for k,v in energy.items() if isinstance(v,np.ndarray)})
        write_new(out/'raw_actual.json',scheduled.raw)
        write_new(out/'substeps.json',monitor.rows)
        write_new(out/'pose_writes.json',writes)
        write_new(out/'commands.json',low.evidence)
        write_new(out/'window.json',{'reset':reset,'warmup':warmup,'requested_schedule':scheduled.requested_writes,
            'measurement_ticks':monitor.tick,'total_physics_ticks_with_warmup':low.tick,
            'environment_ticks_with_warmup':low.tick*n,
            'recorder_metadata':{k:v for k,v in energy.items() if not isinstance(v,np.ndarray)}})
    if error is not None:
        raise error
    audits=[offline_recompute(energy,i) for i in range(n)]
    if any(not a['steps_match'] or a['rel_diff']>1.e-10 for a in audits):
        raise ValueError('actual controlled mechanical-work audit failed')
    review=review_actual_case(fixture,ctx['geom'],scheduled,energy,low.evidence)
    review.update(energy_audits=audits,total_environment_ticks=low.tick*n)
    write_new(out/'FINISHED.json',review)
    env.close()
    return 0
