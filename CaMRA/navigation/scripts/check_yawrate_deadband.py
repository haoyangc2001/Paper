"""NAV-060 辅助：实测冻结 v2 底层的小角速率死区。

背景：NAV-060 pilot 发现闭环指令里 |wz|≈0.03-0.05 rad/s 持续 30 s 几乎不被执行，
导致名义参考（按完美跟踪积分）自己转出约 1 rad、ε_d 虚高到 6.5 m。
在改导航层之前，必须先把"底层到底能执行多小的角速率"量出来，而不是凭猜。

方法（预声明，跑之前写死在这里）：
  * 每个 env 一个固定 |wz| 档位，5 s 恒速闭环；取后 4 s 的均值作为该档实测角速率；
  * 正负号各测一轮（检查对称性）；
  * 两个不同的量，必须分开报：
      - deadband_no_response：升序扫档位，**最后一个** ratio < 0.05 的档位
        （此档及以下底层基本不转，命令可视为无效）；
      - usable_threshold：第一个 ratio >= 0.60 的档位（能用的最小角速率）。
    第一版只报后者的"阈值"，把"增益不足"误当"死区"，已改正。
  * ratio 用正负两轮的平均。

物理量与口径：
  * 角速率取机身系 z 分量（四元数把世界角速度转到机身系），与命令同系；
  * 另记 5 s 内的净航向变化（∫wz dt），作为比值之外的第二证据。

用法：
    python check_yawrate_deadband.py --json CaMRA/navigation/reports/yawrate_deadband.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(CALIB, "src"))
sys.path.insert(0, CALIB)          # calibration 侧用 `sim.profiles_sim` 导入

WARMUP_SECONDS = 1.0
WAVE_SECONDS = 5.0
MEASURE_TAIL_S = 4.0
CTRL_DT = 0.02
WZ_LEVELS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.20, 0.30, 0.50]
RATIO_ACCEPT = 0.60
TILE_M = 60.0


def progress(msg: str) -> None:
    print(f"PROGRESS {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--reanalyze", action="store_true",
                    help="只按已落盘的 raw 重算两个阈值，不启动仿真")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.reanalyze:
        import numpy as np

        doc = json.load(open(args.json, encoding="utf-8"))
        curve = []
        for lv in doc["method"]["wz_levels"]:
            rows = [r for r in doc["raw"] if abs(abs(r["wz_cmd"]) - lv) < 1e-12]
            curve.append({"abs_wz_cmd": lv,
                          "ratio_pos_neg_mean": float(np.mean([r["ratio"] for r in rows])),
                          "achieved_mean": float(np.mean([r["wz_achieved_mean"] for r in rows])),
                          "net_ratio_mean": float(np.mean([r["net_ratio"] for r in rows]))})
        deadband = 0.0
        for row in curve:
            if row["ratio_pos_neg_mean"] < 0.05:
                deadband = row["abs_wz_cmd"]
            else:
                break
        usable = next((r["abs_wz_cmd"] for r in curve
                       if r["ratio_pos_neg_mean"] >= RATIO_ACCEPT), None)
        doc["curve"] = curve
        doc["deadband_no_response_rad_s"] = deadband
        doc["usable_threshold_rad_s"] = usable if usable is not None else max(doc["method"]["wz_levels"])
        doc["rule_met"] = usable is not None
        doc["method"]["deadband_rule"] = "ascending scan, LAST level with pos/neg mean ratio < 0.05"
        doc["method"]["usable_rule"] = "ascending scan, FIRST level with pos/neg mean ratio >= 0.60"
        doc.pop("threshold_rad_s", None)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=2, default=float)
            fh.write("\n")
        print(json.dumps({"deadband_no_response_rad_s": deadband,
                          "usable_threshold_rad_s": doc["usable_threshold_rad_s"],
                          "curve": [(c["abs_wz_cmd"], round(c["ratio_pos_neg_mean"], 3))
                                    for c in curve]}, ensure_ascii=False))
        return 0

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    sim_app = launcher.app  # noqa: F841

    import gymnasium as gym
    import numpy as np
    import torch

    import isaaclab_tasks  # noqa: F401
    import unitree_rl_lab.tasks  # noqa: F401
    import isaaclab.sim as sim_utils
    from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
    from unitree_rl_lab.utils.external_command import DEFAULT_LIMITS, ExternalCommandWrapper
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

    from camra_calibration.profiles import sample_applied
    from sim.profiles_sim import configure_env_for_profile

    BUNDLE = "/pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2"
    policy = torch.jit.load(os.path.join(BUNDLE, "exported", "policy.pt")).eval()

    N = len(WZ_LEVELS)
    device = args.device
    env_cfg = parse_env_cfg("CaMRA-Go2-Velocity", device=device, num_envs=N,
                            use_fabric=True, entry_point_key="env_cfg_entry_point")
    env_cfg.seed = 100
    configure_env_for_profile(env_cfg, sample_applied("clean", 0))
    env_cfg.commands.base_velocity.ranges = env_cfg.commands.base_velocity.limit_ranges
    env_cfg.episode_length_s = 120.0
    env_cfg.curriculum = {}

    gen = env_cfg.scene.terrain.terrain_generator
    gen.size = (TILE_M, TILE_M)
    gen.num_rows = 1
    gen.num_cols = N
    gen.border_width = 1.0
    gen.sub_terrains = {"flat": MeshPlaneTerrainCfg(size=(TILE_M, TILE_M))}
    env_cfg.scene.terrain.terrain_type = "generator"
    env_cfg.scene.terrain.terrain_generator = gen
    env_cfg.scene.env_spacing = TILE_M

    progress("gym_make_start")
    raw = gym.make("CaMRA-Go2-Velocity", cfg=env_cfg)
    env = ExternalCommandWrapper(raw, limits=DEFAULT_LIMITS, enforce_limits=True, on_reset="zero")
    u = raw.unwrapped
    robot = u.scene["robot"]
    sim = u.sim
    sim_dt = sim.get_physics_dt()
    env.reset()
    sim_utils.spawn_cuboid(
        "/World/nav_ground",
        sim_utils.CuboidCfg(size=(2000.0, 2000.0, 0.2),
                            collision_props=sim_utils.CollisionPropertiesCfg(),
                            physics_material=sim_utils.RigidBodyMaterialCfg(
                                static_friction=1.0, dynamic_friction=1.0)),
        translation=(0.0, 0.0, -0.1))
    progress("gym_make_done")

    orig = u.scene.env_origins.detach().cpu().numpy().copy()

    def yaw_of(q) -> float:
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def wz_body(quat, ang_w) -> float:
        w, x, y, z = [float(v) for v in quat]
        r02 = 2.0 * (x * z + w * y)
        r12 = 2.0 * (y * z - w * x)
        r22 = 1.0 - 2.0 * (x * x + y * y)
        ax, ay, az = [float(v) for v in ang_w]
        return r02 * ax + r12 * ay + r22 * az

    def write_cmd(row) -> None:
        # 纯转向命令：vx=vy=0，wz 放第三个槽位（第一版把 wz 填进了 vx 位，实测全为 0）
        for e in range(N):
            env.bridge.set_command([0.0, 0.0, float(row[e])],
                                   env_ids=torch.tensor([e], device=device))

    def reset_all() -> None:
        env.reset()
        for e in range(N):
            root = torch.zeros(1, 7, device=device)
            root[0, 0] = float(orig[e, 0])
            root[0, 1] = float(orig[e, 1])
            root[0, 2] = 0.4
            root[0, 3] = 1.0
            robot.write_root_pose_to_sim(root, env_ids=torch.tensor([e], device=device))
            robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device),
                                             env_ids=torch.tensor([e], device=device))
            djp = robot.data.default_joint_pos.detach()[e:e + 1]
            robot.write_joint_state_to_sim(djp, torch.zeros_like(djp),
                                           env_ids=torch.tensor([e], device=device))
        robot.update(sim_dt)
        am = u.action_manager
        prev = getattr(am, "prev_action", getattr(am, "_prev_action"))
        prev.zero_()
        robot.update(sim_dt)

    results = []
    t0 = time.time()
    for sign, label in ((1.0, "positive"), (-1.0, "negative")):
        reset_all()
        obs = u.observation_manager.compute()
        # warmup：零速闭环 1.0 s
        write_cmd([0.0] * N)
        for _ in range(int(round(WARMUP_SECONDS / CTRL_DT))):
            act = policy(obs["policy"])
            obs, _, _, _, _ = env.step(act)
        # 测量波：恒定 wz 5 s
        n_ctrl = int(round(WAVE_SECONDS / CTRL_DT))
        samples = [[] for _ in range(N)]
        yaw_start = [yaw_of(robot.data.root_quat_w[e].detach().cpu().numpy()) for e in range(N)]
        for i in range(n_ctrl):
            write_cmd([sign * lv for lv in WZ_LEVELS])
            act = policy(obs["policy"])
            obs, _, _, _, _ = env.step(act)
            if (i + 1) * CTRL_DT >= WAVE_SECONDS - MEASURE_TAIL_S:
                quat = robot.data.root_quat_w.detach().cpu().numpy()
                w = robot.data.root_ang_vel_w.detach().cpu().numpy()
                for e in range(N):
                    samples[e].append(wz_body(quat[e], w[e]))
        yaw_end = [yaw_of(robot.data.root_quat_w[e].detach().cpu().numpy()) for e in range(N)]
        for e, lv in enumerate(WZ_LEVELS):
            cmd = sign * lv
            ach = float(np.mean(samples[e])) if samples[e] else float("nan")
            dyaw = (yaw_end[e] - yaw_start[e] + math.pi) % (2 * math.pi) - math.pi
            results.append({
                "wz_cmd": cmd, "sign": label, "wave_s": WAVE_SECONDS,
                "wz_achieved_mean": ach,
                "ratio": (ach / cmd) if cmd else None,
                "net_dyaw_rad": dyaw,
                "net_ratio": (dyaw / (cmd * WAVE_SECONDS)) if cmd else None,
            })
        progress(f"wave_{label}_done")

    # 正负平均
    curve = []
    for lv in WZ_LEVELS:
        rows = [r for r in results if abs(abs(r["wz_cmd"]) - lv) < 1e-12]
        ratio = float(np.mean([r["ratio"] for r in rows]))
        curve.append({"abs_wz_cmd": lv, "ratio_pos_neg_mean": ratio,
                      "achieved_mean": float(np.mean([r["wz_achieved_mean"] for r in rows])),
                      "net_ratio_mean": float(np.mean([r["net_ratio"] for r in rows]))})

    deadband = 0.0
    for row in curve:                                   # 升序：最后一个"基本不转"的档位
        if row["ratio_pos_neg_mean"] < 0.05:
            deadband = row["abs_wz_cmd"]
        else:
            break
    usable = None
    for row in curve:                                   # 升序：第一个达标的档位
        if row["ratio_pos_neg_mean"] >= RATIO_ACCEPT:
            usable = row["abs_wz_cmd"]
            break
    rule_hit = usable is not None
    threshold = usable if usable is not None else max(WZ_LEVELS)

    report = {
        "task": "NAV-060", "purpose": "measure the frozen v2 yaw-rate deadband",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": {"wave_s": WAVE_SECONDS, "measure_tail_s": MEASURE_TAIL_S,
                   "ctrl_dt_s": CTRL_DT, "wz_levels": WZ_LEVELS,
                   "ratio_accept": RATIO_ACCEPT,
                   "deadband_rule": "ascending scan, LAST level with pos/neg mean ratio < 0.05",
                   "usable_rule": "ascending scan, FIRST level with pos/neg mean ratio >= 0.60"},
        "deadband_no_response_rad_s": deadband,
        "usable_threshold_rad_s": threshold, "rule_met": rule_hit,
        "curve": curve, "raw": results,
        "wallclock_s": round(time.time() - t0, 1),
        "note": "本测量只改命令、不改任何冻结物；结果用于导航层死区规避（NAV-060 处置 C）",
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2, default=float)
            fh.write("\n")
    print(json.dumps({"deadband_no_response_rad_s": deadband,
                      "usable_threshold_rad_s": threshold, "rule_met": rule_hit,
                      "curve": [(c["abs_wz_cmd"], round(c["ratio_pos_neg_mean"], 3)) for c in curve]},
                     ensure_ascii=False))
    print("PROGRESS hard_exit_skip_close", flush=True)
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
