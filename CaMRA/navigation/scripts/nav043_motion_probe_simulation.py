"""Single clean plane, frozen low-level controller and fixed 12-case capture.

This module is imported only inside the approved application after the
installed explicit step hook is active. No upper actor/critic or W is used.
"""
from dataclasses import asdict
import sys

from nav043_motion_probe_guard import BUNDLE, NAV, persist, sha


def simulate(p, out, budget):
    import gymnasium as gym
    import numpy as np
    import torch
    import isaaclab_tasks  # noqa: F401
    import unitree_rl_lab.tasks  # noqa: F401
    from unitree_rl_lab.utils.external_command import DEFAULT_LIMITS, ExternalCommandWrapper
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
    from camra_calibration.profiles import sample_applied
    sys.path.insert(0, str(NAV.parent/'calibration/sim'))
    from profiles_sim import configure_env_for_profile, apply_friction, apply_mass
    from camra_navigation.envs.clean_v3 import configure_clean, IsaacSubstepAdapter
    from camra_navigation.envs.controlled_terrain_v2 import configure_fixture_plane
    from camra_navigation.envs.motion_capture_v1 import capture_case
    from camra_navigation.evaluation.recorder import NavEnergyRecorder
    from nav067_collider_readback import read_ground

    cfg = parse_env_cfg('CaMRA-Go2-Velocity', device='cuda:0', num_envs=1,
                        use_fabric=True, entry_point_key='env_cfg_entry_point')
    cfg.seed = p['environment']['seed']
    cfg.episode_length_s = 12.
    applied = sample_applied('clean', cfg.seed, duration_s=11.)
    clean = configure_clean(cfg, configure_env_for_profile, applied)
    plane = configure_fixture_plane(cfg, 30.)
    material = cfg.scene.terrain.physics_material
    material.static_friction, material.dynamic_friction, material.restitution = 1., 1., 0.
    raw = gym.make('CaMRA-Go2-Velocity', cfg=cfg)
    env = ExternalCommandWrapper(raw, limits=DEFAULT_LIMITS, enforce_limits=True, on_reset='zero')
    u, robot = raw.unwrapped, raw.unwrapped.scene['robot']
    policy = torch.jit.load(str(BUNDLE/'exported/policy.pt'), map_location='cuda:0').eval()
    low = IsaacSubstepAdapter(env, policy, map_center=(0.,0.))
    runtime = {'clean_configuration':clean, 'terrain':plane,
        'actual_origins':low.origins.tolist(), 'actual_isaaclab_module':__import__('isaaclab').__file__,
        'physics_dt_s':u.sim.get_physics_dt(), 'decimation':u.cfg.decimation,
        'num_envs':u.num_envs, 'event_terms':u.event_manager.active_terms,
        'event_modes':u.event_manager.available_modes,
        'observation_corruption_enabled':u.cfg.observations.policy.enable_corruption,
        'ground':read_ground(u.sim.stage, cfg.scene.terrain.prim_path),
        'robot_usd_path':str(u.cfg.scene.robot.spawn.usd_path),
        'robot_joint_names':list(robot.joint_names),
        'policy_sha256':sha(BUNDLE/'exported/policy.pt'),
        'actuators':{name:{'class':type(act).__module__+'.'+type(act).__name__,
                           'stiffness':act.cfg.stiffness, 'damping':act.cfg.damping,
                           'friction':act.cfg.friction}
                     for name, act in robot.actuators.items()},
        'purpose':'exploratory fixed commands, no navigation or calibration acceptance'}
    persist(out/'runtime.json', runtime)
    if (u.num_envs != 1 or np.asarray(low.origins).shape != (1,3)
            or not np.isfinite(low.origins).all()
            or abs(u.sim.get_physics_dt()-.005) > 1.e-12 or u.cfg.decimation != 4
            or u.cfg.observations.policy.enable_corruption
            or runtime['actual_isaaclab_module'] != p['environment']['isaaclab_origin']):
        raise ValueError('actual clean timing/environment/module readback failed')
    results = []
    try:
        for case in p['schedule']['cases']:
            trial = out/case['id']
            trial.mkdir(exist_ok=False)
            persist(trial/'STARTED.json', case)
            reset = low.reset(start=(0.,0.), yaw=0.)
            friction = apply_friction(robot, [0], applied.friction, applied.restitution)
            mass = apply_mass(robot, [0], applied.added_mass_kg)
            actual = low.states()
            reset.update(actual_states=[asdict(s) for s in actual],
                actual_clock_s=float(u.sim.current_time), actual_origins=low.origins.tolist(),
                friction=friction, mass=mass, explicit_step_accounting=budget.snapshot())
            persist(trial/'reset.json', reset)
            if not friction['ok'] or not mass['ok']:
                raise ValueError('actual clean robot material/mass readback failed')
            if max(abs(actual[0].x), abs(actual[0].y), abs(actual[0].yaw)) > 1.e-4:
                raise ValueError('actual reset planar origin/yaw roundtrip failed')
            budget.begin_case(case['id'])
            recorder = NavEnergyRecorder(2200, 1, device='cuda:0',
                                        expected_steps=2200, label=case['id'])
            result = capture_case(low, case['command'], trial, lambda:float(u.sim.current_time), recorder)
            budget.finish_case()
            result.update(case_id=case['id'], requested_command=case['command'],
                          explicit_step_accounting=budget.snapshot())
            persist(trial/'result.json', result)
            results.append(result)
            print('CONTROLLED_CAPTURE_FINISHED', case['id'], result['measurement_J'], flush=True)
        persist(out/'CAPTURE_FINISHED.json', {'cases':results, 'NAV043_passed':False,
                    'G5_passed':False, 'requires_saved_evidence_review':True})
    finally:
        env.close()
    return 0
