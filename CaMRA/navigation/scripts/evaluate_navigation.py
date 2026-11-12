"""NAV-042: run the online closed-loop navigation on the physical simulator.

Per process (gym.make may only run once): one map, both methods, one episode
each -> 3 processes x 2 = 6 episodes, exactly the authorised
``integration_pilot`` quota.  These episodes are **wiring evidence**: they never
enter a performance denominator, and the budget is the *common-budget* anchor
``z0 = k * E_ref(map)`` with ``k`` stated on the command line -- not a NAV-041
selection result (that route returned ``no_feasible_budget`` for both methods and
is reported separately).

What is real here:

* the upper layer reads the **simulator's** base pose/velocity every 0.1 s,
* the low level re-observes and re-infers 5 times per navigation step, 4 physics
  substeps each (200 Hz), through the same ExternalCommandWrapper the frozen
  policy was validated with,
* the internal budget is charged with ``(1+alpha)*c_nom`` while the measured
  mechanical work is accumulated at 200 Hz by ``NavEnergyRecorder``,
* the nominal reference integrates the applied commands on its own, never
  re-anchored to the actual pose.

Usage (one level per process):

    <frozen python> CaMRA/navigation/scripts/evaluate_navigation.py \
        --level L1_empty --methods uncalibrated,full --k 1.0 \
        --run-id nav042_L1 --json CaMRA/navigation/reports/nav042_L1_empty.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(CALIB, "src"))

BUNDLE = "/pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2"
WORK_ROOT = "/pub/data/caohy/work/camra_navigation"
MAP_SEED = 3001
WARMUP_SECONDS = 1.0
FALL_HEIGHT_M = 0.15
FALL_TILT_DEG = 60.0
DECIMATION = 4
POLICY_DT_S = 0.02


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(q) -> float:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def projected_gravity(quat) -> tuple[float, float, float]:
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    return (-2.0 * (x * z + w * y), -2.0 * (y * z - w * x), -(1.0 - 2.0 * (x * x + y * y)))


def state_dict_sha(sd) -> str:
    import hashlib

    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("ascii"))
        h.update(str(t.dtype).encode("ascii"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


class IsaacLowLevel:
    """The :class:`~camra_navigation.envs.physical.LowLevel` protocol on Isaac Lab."""

    def __init__(self, env, robot, sim, v2_policy, *, num_envs: int,
                 recorder, device: str, spawn: tuple[float, float] | None = None) -> None:
        self.env = env
        self.robot = robot
        self.sim = sim
        self.v2 = v2_policy
        self.num_envs = num_envs
        self.recorder = recorder
        self.device = device
        self._spawn = spawn
        self._tick = 0
        self._obs = None
        self.falls = 0

    # ---- protocol ----
    def reset(self) -> None:
        import torch

        self.env.reset()
        if self._spawn is not None:
            for e in range(self.num_envs):
                root = torch.zeros(1, 7, device=self.device)
                root[0, 0] = self._spawn[0]
                root[0, 1] = self._spawn[1]
                root[0, 2] = 0.4
                qw = math.cos(self._spawn[2] / 2.0)
                qz = math.sin(self._spawn[2] / 2.0)
                root[0, 3], root[0, 6] = qw, qz
                idx = torch.tensor([e], device=self.device)
                self.robot.write_root_pose_to_sim(root, env_ids=idx)
                self.robot.write_root_velocity_to_sim(
                    torch.zeros(1, 6, device=self.device), env_ids=idx)
                djp = self.robot.data.default_joint_pos.detach()[e:e + 1]
                self.robot.write_joint_state_to_sim(djp, torch.zeros_like(djp), env_ids=idx)
        self.robot.update(self.sim.get_physics_dt())
        am = self.env.unwrapped.action_manager
        prev = getattr(am, "prev_action", getattr(am, "_prev_action"))
        prev.zero_()
        self.robot.update(self.sim.get_physics_dt())
        self._tick = 0
        self._obs = self.env.unwrapped.observation_manager.compute()

    def write_commands(self, rows) -> None:
        import torch

        for e, row in enumerate(rows):
            self.env.bridge.set_command(row, env_ids=torch.tensor([e], device=self.device))

    def step_policy(self) -> dict[str, Any]:
        u = self.env.unwrapped
        # fresh observation AFTER the command was written, then infer: this is
        # the 50 Hz re-observation the contract asks for
        obs = u.observation_manager.compute()
        act = self.v2(obs["policy"])
        self._obs, _, _, _, _ = self.env.step(act)
        self._tick += 1
        events = {}
        for e in range(self.num_envs):
            h = float(self.robot.data.root_pos_w[e, 2])
            gx, gy, gz = projected_gravity(self.robot.data.root_quat_w[e].detach().cpu().numpy())
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, -gz))))
            if h < FALL_HEIGHT_M or tilt > FALL_TILT_DEG:
                events[e] = "fall"
                self.falls += 1
        return {
            "obs_command": [obs["policy"][e, 6:9].detach().cpu().numpy().tolist()
                            for e in range(self.num_envs)],
            "substeps": DECIMATION,
            "sim_time": self._tick * POLICY_DT_S,
            "events": events,
            "obs_fingerprint": obs["policy"][0, 6:9].detach().cpu().numpy().tolist(),
        }

    def zero_clock(self) -> None:
        """Start the navigation clock at zero after the gait warm-up."""
        self._tick = 0

    def step_time(self) -> float:
        return self._tick * POLICY_DT_S

    def true_state(self, env_id: int):
        pos = self.robot.data.root_pos_w[env_id, :2].detach().cpu().numpy()
        quat = self.robot.data.root_quat_w[env_id].detach().cpu().numpy()
        yaw = yaw_from_quat(quat)
        lin = self.robot.data.root_lin_vel_b[env_id].detach().cpu().numpy()
        ang = self.robot.data.root_ang_vel_b[env_id].detach().cpu().numpy()
        origin = self.origin_of(env_id)
        return (float(pos[0] - origin[0]), float(pos[1] - origin[1]), yaw,
                float(lin[0]), float(lin[1]), float(ang[2]))

    def origin_of(self, env_id: int) -> tuple[float, float]:
        o = self.env.unwrapped.scene.env_origins[env_id].detach().cpu().numpy()
        return (float(o[0]), float(o[1]))

    def measured_J(self) -> list[float]:
        if self.recorder is None:
            return [0.0] * self.num_envs
        return [float(v) for v in self.recorder.total_j.detach().cpu().numpy()]


def load_candidate(method: str, device: str = "cpu"):
    import torch

    from camra_navigation.algorithm import networks as NW

    bundle = json.load(open(os.path.join(NAV_ROOT, "delivery",
                                         "deployment_candidates_v1.json"), encoding="utf-8"))
    cands = sorted((c for c in bundle["candidates"] if c["method"] == method),
                   key=lambda c: c["seed"])
    if not cands:
        raise SystemExit(f"no deployment candidate for {method}")
    c = cands[0]
    path = os.path.join(WORK_ROOT, c["source_run_id"], "deploy_refit", "state.pt")
    if not os.path.exists(path):
        path = os.path.join(WORK_ROOT, c["source_run_id"], "state.pt")
    state = torch.load(path, map_location="cpu", weights_only=False)
    sd = state["model"]
    got = state_dict_sha({k: v for k, v in sd.items() if k.startswith("policy.")})
    if got != c["weights"]["actor_sha256"]:
        raise SystemExit(f"actor hash mismatch for {c['candidate_id']}")
    nav_contract = json.load(open(os.path.join(NAV_ROOT, "config",
                                              "navigation_contract_v1.json"), encoding="utf-8"))
    net = NW.build(nav_contract)
    net.load_state_dict(sd)
    net.eval().to(device)
    return c, net


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="L1_empty")
    ap.add_argument("--methods", default="uncalibrated,full")
    ap.add_argument("--k", type=float, default=1.0, help="common-budget anchor")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--trace-dir", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-nav-steps", type=int, default=310)
    args = ap.parse_args()

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    sim_app = launcher.app  # noqa: F841

    import gymnasium as gym
    import numpy as np
    import torch

    import isaaclab_tasks  # noqa: F401
    import unitree_rl_lab.tasks  # noqa: F401
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from unitree_rl_lab.utils.external_command import DEFAULT_LIMITS, ExternalCommandWrapper
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

    from camra_navigation.costs import JCostModel
    from camra_navigation.envs.nominal import DT_NAV_S
    from camra_navigation.envs.physical import PhysicalConfig, PhysicalNavEnv
    from camra_navigation.evaluation.recorder import NavEnergyRecorder
    from camra_navigation.maps import generator as M
    from camra_navigation.maps.geometry import SafetyGeometry
    from camra_navigation.maps.scene_adapter import build_scene

    design = json.load(open(os.path.join(NAV_ROOT, "config", "map_design_v1.json"),
                            encoding="utf-8"))
    spec = M.build(args.level, MAP_SEED, design)
    e_ref = {r["map_id"]: float(r["e_ref_J"]) for r in
             json.load(open(os.path.join(NAV_ROOT, "reports",
                                         "budget_selection_reference_energy.json"),
                           encoding="utf-8"))["maps"]}
    if spec.map_id not in e_ref:
        raise SystemExit(f"no E_ref for {spec.map_id}")
    z0 = float(args.k) * e_ref[spec.map_id]

    N = 1
    layout = build_scene(spec, env_index=0, num_envs=N)
    boxes = list(layout.wall_boxes) + list(layout.island_boxes)
    print(f"PROGRESS map={spec.map_id} boxes={len(boxes)} z0={z0:.1f} "
          f"spacing={layout.env_spacing_m:.2f}", flush=True)

    device = args.device
    env_cfg = parse_env_cfg("CaMRA-Go2-Velocity", device=device, num_envs=N,
                            use_fabric=True, entry_point_key="env_cfg_entry_point")
    env_cfg.seed = 100
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.commands.base_velocity.ranges = env_cfg.commands.base_velocity.limit_ranges
    env_cfg.episode_length_s = WARMUP_SECONDS + 40.0
    env_cfg.curriculum = {}
    from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg

    tile = float(layout.env_spacing_m)
    gen = env_cfg.scene.terrain.terrain_generator
    gen.size = (tile, tile)
    gen.num_rows = 1
    gen.num_cols = N
    gen.border_width = 1.0
    gen.sub_terrains = {"flat": MeshPlaneTerrainCfg(size=(tile, tile))}
    env_cfg.scene.terrain.terrain_type = "generator"
    env_cfg.scene.terrain.terrain_generator = gen
    env_cfg.scene.env_spacing = tile

    for j, box in enumerate(boxes):
        name = f"nav_box_{j}"
        setattr(env_cfg.scene, name, RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.CuboidCfg(
                size=(abs(box.size_x), abs(box.size_y), box.height),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg()),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(box.center_x, box.center_y, box.z_center))))

    raw = gym.make("CaMRA-Go2-Velocity", cfg=env_cfg)
    env = ExternalCommandWrapper(raw, limits=DEFAULT_LIMITS, enforce_limits=True,
                                 on_reset="zero")
    u = raw.unwrapped
    robot, sim = u.scene["robot"], u.sim
    sim_dt = sim.get_physics_dt()
    env.reset()
    sim_utils.spawn_cuboid(
        "/World/nav_ground",
        sim_utils.CuboidCfg(size=(600.0, 600.0, 0.2),
                            collision_props=sim_utils.CollisionPropertiesCfg(),
                            physics_material=sim_utils.RigidBodyMaterialCfg(
                                static_friction=1.0, dynamic_friction=1.0)),
        translation=(0.0, 0.0, -0.1))
    origins = u.scene.env_origins.detach().cpu().numpy().copy()
    for k, name in enumerate([f"nav_box_{j}" for j in range(len(boxes))]):
        box = boxes[k]
        poses = torch.zeros(N, 7, device=device)
        poses[0, 0] = float(origins[0, 0]) + (box.center_x - layout.env_origin[0])
        poses[0, 1] = float(origins[0, 1]) + (box.center_y - layout.env_origin[1])
        poses[0, 2] = box.z_center
        qw, qx_, qy_, qz_ = box.quat_wxyz
        poses[0, 3], poses[0, 4], poses[0, 5], poses[0, 6] = qw, qx_, qy_, qz_
        u.scene[name].write_root_pose_to_sim(poses)
    print("PROGRESS obstacles_placed", flush=True)

    mc = layout.map_center
    spawn = (float(origins[0, 0]) + float(spec.start[0] - mc[0]),
             float(origins[0, 1]) + float(spec.start[1] - mc[1]),
             float(layout.robot_spawn_yaw))

    # ---- energy recorder hooked into the 200 Hz physics step ----
    n_ctrl = int(round(40.0 / POLICY_DT_S))
    state: dict[str, Any] = {"rec": None}
    orig_step = sim.step

    def patched_step(*a, **kw):
        out = orig_step(*a, **kw)
        rec = state["rec"]
        if rec is not None:
            tau = robot.data.applied_torque.detach()
            robot.update(sim_dt)
            rec.record(tau_applied_lab=tau, qdot_post_lab=robot.data.joint_vel.detach(),
                       root_pos_lab=robot.data.root_pos_w[:, :3].detach())
        return out

    sim.step = patched_step

    def v2_callable(policy_graph):
        def call(obs_tensor):
            with torch.no_grad():
                return policy_graph(obs_tensor)
        return call

    v2 = torch.jit.load(os.path.join(BUNDLE, "exported", "policy.pt"), map_location=device)
    v2.eval()

    reports = []
    t_start_wall = time.time()
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        # the navigation actor is tiny (one inference per 0.1 s): keep it on CPU
        # so the observation vector built in numpy does not need a device copy;
        # Isaac and the 50 Hz locomotion graph stay on the GPU
        cand, net = load_candidate(method, "cpu")
        geometry = SafetyGeometry.from_frozen(
            os.path.join(CALIB, "config", "footprint_v1.json"),
            os.path.join(CALIB, "config", "margins_v1.json"),
            os.path.join(NAV_ROOT, "config", "geometry_contract_v1.json"))
        if method == "uncalibrated":
            import dataclasses
            geometry = dataclasses.replace(geometry, delta_m=0.0)
        cm = JCostModel.from_frozen(os.path.join(CALIB, "config", "power_model_v1.json"),
                                    os.path.join(CALIB, "config", "margins_v1.json"),
                                    variant=method)
        recorder = NavEnergyRecorder(capacity_steps=n_ctrl * DECIMATION + 400,
                                     num_envs=N, device=device,
                                     label=f"nav042_{args.level}_{method}",
                                     expected_steps=n_ctrl * DECIMATION)
        low = IsaacLowLevel(env, robot, sim, v2_callable(v2), num_envs=N,
                            recorder=recorder, device=device, spawn=spawn)
        state["rec"] = recorder
        recorder.begin_warmup()
        policy = lambda vec: [float(v) for v in  # noqa: E731
                              net.act(torch.as_tensor([vec], dtype=torch.float32),
                                      deterministic=True)[0][0]]
        nav = PhysicalNavEnv(spec, geometry, cm, low, policy,
                             candidate_id=cand["candidate_id"], cfg=PhysicalConfig())
        # warm-up first: zero command for 1 s so the gait settles, THEN open the
        # episode from the settled pose (the clock and the trace start together)
        low.reset()
        low.write_commands([[0.0, 0.0, 0.0]])
        for _ in range(int(round(WARMUP_SECONDS / POLICY_DT_S))):
            low.step_policy()
        low.zero_clock()
        recorder.begin_window()
        trace = nav.reset(z0_J=z0, reset_low=False)
        t0 = time.time()
        steps = 0
        while not all(nav.done) and steps < args.max_nav_steps:
            nav.step()
            steps += 1
        recorder.mark_event(list(range(N)), nav.label[0] or "timeout", detail="nav episode")
        res = recorder.finish()
        wall = time.time() - t0
        problems = recorder.problems(res)
        state["rec"] = None
        print(f"PROGRESS {method} label={nav.label[0]} steps={steps} wall={wall:.1f}s "
              f"measured_J={res['total_j'][0]:.1f} problems={problems}", flush=True)

        payload = trace.as_dict(include_steps=True)
        payload["method"] = method
        payload["k_anchor"] = args.k
        payload["geometry"] = {"delta_m": geometry.delta_m,
                               "inflation_radius_m": geometry.inflation_radius_m}
        payload["cost_model"] = {"alpha": cm.alpha, "variant": cm.variant}
        payload["recorder"] = {"total_j": float(res["total_j"][0]),
                               "substeps": int(res["valid_mask"][:, 0].sum()),
                               "problems": problems}
        if args.trace_dir:
            os.makedirs(args.trace_dir, exist_ok=True)
            path = os.path.join(args.trace_dir, f"{args.run_id}_{method}_{spec.map_id}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            print(f"PROGRESS trace_written {path}", flush=True)

        # structural consistency of the trace itself
        recs = trace.records
        ids_ok = all(r.episode_id == trace.episode_id for r in recs)
        clock_ok = all(abs(r.t_start_s - i * DT_NAV_S) < 1e-9
                       and abs(r.t_end_s - (i + 1) * DT_NAV_S) < 1e-9
                       for i, r in enumerate(recs))
        chain_ok = all(len(r.policy_steps) == 5 and all(p.substeps == DECIMATION
                                                        for p in r.policy_steps)
                       for r in recs)
        current_ok = all(p.command_current for r in recs for p in r.policy_steps)
        max_cmd_diff = max((p.max_command_diff for r in recs for p in r.policy_steps), default=0.0)
        reports.append({
            "method": method,
            "candidate_id": cand["candidate_id"],
            "actor_sha256": cand["weights"]["actor_sha256"],
            "map_id": spec.map_id,
            "z0_J": round(z0, 3),
            "summary": trace.summary(),
            "trace_checks": {
                "episode_ids_consistent": ids_ok,
                "nav_clock_on_the_dt_grid": clock_ok,
                "five_policy_steps_x4_substeps": chain_ok,
                "command_current_every_policy_step": current_ok,
                "max_command_diff": max_cmd_diff,
            },
            "recorder_problems": problems,
            "measured_J": float(res["total_j"][0]),
            "wall_seconds": round(wall, 1),
        })

    report = {
        "task": "NAV-042",
        "schema": "camra-online-navigation-pilot-v1",
        "run_id": args.run_id,
        "level": args.level,
        "map_id": spec.map_id,
        "split": "integration_pilot",
        "budget_anchor": {"rule": "z0 = k * E_ref(map), common-budget anchor",
                          "k": args.k, "e_ref_J": e_ref[spec.map_id],
                          "z0_J": round(z0, 3),
                          "not_a_selection_result":
                              "NAV-041 recorded no_feasible_budget for both methods on the "
                              "budget_selection split; this pilot uses the common-budget "
                              "anchor purely to exercise the loop and must not be read as a "
                              "selected budget"},
        "rate_structure": {"nav_hz": 10, "policy_hz": 50, "physics_hz": 200,
                           "policy_steps_per_nav": 5, "substeps_per_policy": DECIMATION,
                           "physics_dt_s": sim_dt},
        "reporting": "integration_pilot: wiring evidence, never a performance denominator",
        "episodes": reports,
        "wall_seconds_total": round(time.time() - t_start_wall, 1),
    }
    out = args.json or os.path.join(NAV_ROOT, "reports", f"nav042_{args.level}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({"written": out, "episodes": [
        {"method": e["method"], "label": e["summary"]["label"],
         "checks": e["trace_checks"]} for e in reports]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
