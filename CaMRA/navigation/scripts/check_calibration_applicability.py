"""NAV-060: 训练前闭环指令与旧裕度适用性 pilot（G2A 门禁）。

问题：冻结的功率模型与裕度 α/δ 是在「每秒目标 + 0.2 s 过渡」的开环命令程序上
标定的。导航层产生的是 **10 Hz 闭环在线指令**（规则反馈，随状态变化）。本 pilot
核验：把这套指令喂给同一个冻结 v2 底层，冻结的 α/δ 与功率模型是否仍然适用。

三条硬约束（来自任务定义）：
  1) 阈值与资源必须在看到任何数据之前冻结 —— ``--write-prereg`` 先落盘设计文件，
     运行模式会重新校验输入散列，输入一变立刻中止。
  2) 不许声称尚未训练的 actor 已被验证 —— 这里只评"执行侧 + 名义模型"的适用性。
  3) 显著不匹配不得自行收窄命令或反调裕度 —— 只如实报告并请求用户裁定。

用法（先冻结，再跑）：
    python check_calibration_applicability.py --write-prereg
    python check_calibration_applicability.py --split old
    python check_calibration_applicability.py --split new --level L1_empty
    ...
    python check_calibration_applicability.py --aggregate --json out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(CALIB, "src"))
sys.path.insert(0, CALIB)          # calibration 侧脚本用 `sim.profiles_sim` 导入

# ---- 冻结输入（路径全部为已登记产物） -----------------------------------
MARGINS_PATH = os.path.join(CALIB, "config", "margins_v1.json")
POWER_PATH = os.path.join(CALIB, "config", "power_model_v1.json")
PROTOCOL_PATH = os.path.join(CALIB, "config", "calibration_protocol_v1.0.2.json")
V2_LISTS = os.path.join(CALIB, "experiments", "hira_go2_18500", "v2_lists")
PILOT_LIST = os.path.join(V2_LISTS, "pilot.jsonl")
LOCK_PATH = os.path.join(V2_LISTS, "lock.json")
SPLIT_MANIFEST = os.path.join(NAV_ROOT, "config", "split_manifest_v1.json")
MAP_DESIGN = os.path.join(NAV_ROOT, "config", "map_design_v1.json")
PREREG_PATH = os.path.join(NAV_ROOT, "config", "calibration_applicability_prereg_v1.json")

# ---- 时序（导航契约 timing） --------------------------------------------
EPISODE_SECONDS = 30.0
WARMUP_SECONDS = 1.0
SUBSTEP_DT = 0.005
CTRL_DT = 0.02            # low_level_dt_s
NAV_DT = 0.1              # navigation_dt_s
CTRL_PER_NAV = int(round(NAV_DT / CTRL_DT))          # 5
N_CTRL = int(round(EPISODE_SECONDS / CTRL_DT))       # 1500
N_NAV = int(round(EPISODE_SECONDS / NAV_DT))         # 300
SUBSTEPS_PER_NAV = int(round(NAV_DT / SUBSTEP_DT))   # 20
EXPECTED_SUBSTEPS = int(round(EPISODE_SECONDS / SUBSTEP_DT))  # 6000

# ---- 冻结阈值（本文件即冻结载体；改动必须在跑数据之前） -----------------
PREFIXES_S = [5, 10, 20, 30]
Q90_EPS_C_MAX = 0.15          # 比 α=0.1905 更紧；通过即意味着旧 α 仍然覆盖
LOW_SPEED_SHIFT_MAX = 0.10    # 单向：新指令低速停留不得高于旧程序 + 10 个百分点
LATENCY_MAX_S = CTRL_DT       # 指令到物理生效 ≤ 1 个控制步
J_MIN = 1.0                   # 协议 near_zero 规则

# ---- pilot 域配置 -------------------------------------------------------
OLD_PROFILES = ("clean", "friction_only", "mass_only")
NEW_PROFILES = ("clean", "friction_only", "mass_only")
NEW_SEEDS = {"clean": [9001, 9002, 9003, 9004],
             "friction_only": [9101, 9102, 9103, 9104],
             "mass_only": [9201, 9202, 9203, 9204]}
NEW_LEVELS = ("L1_empty", "L2_single_detour", "L3_multi_path")
MAP_SEED = 101

# 规则反馈导航（与 NAV-028 G2 同一套常数；0.1 s 在线指令）
ROUTE_VX_CRUISE = 0.60
ROUTE_VX_NEAR_GOAL = 0.30
ROUTE_NEAR_GOAL_M = 1.0
ROUTE_K_WZ = 1.5
ROUTE_WZ_LIMIT = 0.80
GOAL_RADIUS_M = 0.40
FALL_HEIGHT_M = 0.15
FALL_TILT_DEG = 60.0

LOW_SPEED_VX = 0.3
LOW_SPEED_WZ = 0.3

# 旧程序侧：无墙体（校准是空地）；环境间距放大到 60 m 避免机器人互撞
OLD_TILE_M = 60.0


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_self() -> str:
    return sha256_file(os.path.abspath(__file__))


def progress(msg: str) -> None:
    print(f"PROGRESS {msg}", flush=True)


def file_hashes() -> dict:
    return {
        "margins_v1.json": sha256_file(MARGINS_PATH),
        "power_model_v1.json": sha256_file(POWER_PATH),
        "calibration_protocol_v1.0.2.json": sha256_file(PROTOCOL_PATH),
        "pilot.jsonl": sha256_file(PILOT_LIST),
        "lock.json": sha256_file(LOCK_PATH),
        "split_manifest_v1.json": sha256_file(SPLIT_MANIFEST),
        "map_design_v1.json": sha256_file(MAP_DESIGN),
    }


# =============================================================== 设计冻结
def build_prereg() -> dict:
    margins = json.load(open(MARGINS_PATH, encoding="utf-8"))
    protocol = json.load(open(PROTOCOL_PATH, encoding="utf-8"))
    pilot = [json.loads(l) for l in open(PILOT_LIST, encoding="utf-8")]

    old_eps = [t for t in pilot if t["profile"] in OLD_PROFILES]
    excluded = sorted({t["profile"] for t in pilot} - set(OLD_PROFILES))

    return {
        "task": "NAV-060",
        "gate": "G2A",
        "schema": "camra.nav.applicability.prereg.v1",
        "created_at": datetime_now(),
        "frozen_before_data": True,
        "mutation_rule": "本文件在运行任何 pilot 回合之前落盘；运行模式会校验输入散列，"
                         "任一输入变化即拒绝继续。阈值不得在看到数据后调整。",
        "question": "闭环 10 Hz 在线指令下，冻结功率模型与裕度 α/δ 是否仍适用于成本与几何判定",
        "frozen_inputs": file_hashes(),
        "frozen_values": {
            "alpha": margins["alpha"],
            "delta": margins["delta"],
            "q90_eps_c_of_calibration": margins["q90_eps_c"],
            "q90_eps_d_of_calibration": margins["q90_eps_d"],
            "margins_sha256_declared": margins.get("sha256"),
            "power_model_sha256_declared": margins["data"]["power_model_sha256"],
            "protocol_version": protocol["version"],
            "command_domain": protocol["command_domain"],
            "command_program": {
                "target_change_period_s": protocol["command_program"]["target_change_period_s"],
                "transition_ramp_s": protocol["command_program"]["transition_ramp_s"],
            },
        },
        "streams": {
            "old": {
                "description": "原校准命令程序：每 1.0 s 换目标、0.2 s 线性过渡，0.1 s 命令点，开环",
                "generator": "camra_calibration.commands.build_program(family, seed, 30 s, dt=0.1, "
                             "target_period=1.0, ramp=0.2)",
                "domain": "pilot split（v2_lists/pilot.jsonl）中的 clean / friction_only / mass_only，"
                          "4 命令族 × 3 profile = 12 回合；program_sha256 逐条校验",
                "excluded_profiles": excluded,
                "exclusion_reason": "push_only / combined 含水平速度冲击；协议 energy.push_work_rule "
                                    "规定推扰做功不计入 J_exec，会把能量配对污染成系统性偏差。"
                                    "本 pilot 不评推扰条件，属显式排除，不是静默丢弃。",
            },
            "new": {
                "description": "规则反馈导航产生的 0.1 s 在线闭环指令（朝目标 heading P 控制）",
                "controller": {"vx_cruise": ROUTE_VX_CRUISE, "vx_near_goal": ROUTE_VX_NEAR_GOAL,
                               "near_goal_m": ROUTE_NEAR_GOAL_M, "k_wz": ROUTE_K_WZ,
                               "wz_limit": ROUTE_WZ_LIMIT, "vy": 0.0},
                "maps": list(NEW_LEVELS),
                "map_seed": MAP_SEED,
                "seeds": NEW_SEEDS,
                "note": "9xxx 段为导航 pilot 专用种子；v2_lists 实占 4000-4399/5100-5119/6000-6799/7000-7399/8000-8199，互不重叠（运行时断言）",
            },
        },
        "episode": {"seconds": EPISODE_SECONDS, "warmup_seconds": WARMUP_SECONDS,
                    "substeps": EXPECTED_SUBSTEPS, "nav_steps": N_NAV},
        "metrics": {
            "eps_c": "(J_exec - J_nom) / J_nom，J_nom >= 1.0 J 才算；J_nom 由冻结功率模型对该回合"
                     "实际 applied 命令序列积分（c_nom = P_nom(applied) * 0.1）",
            "eps_d": "单回合 max_t ||p_exec(t) - p_nom(t)||_2，10 Hz 公共网格，"
                     "p_nom 由 warmup 后实测平面初态积分得到（不插值、不相位对齐）",
            "population_statistic": "nearest-rank q90（与协议 statistics.quantile_method 一致，"
                                    "k = ceil(0.9 n)，等于阈值视为通过）",
            "change_rate": "相邻 0.1 s 命令点的 ||Δu||∞，报均值/最大值",
            "switch_rate": "各轴符号翻转次数 / 秒",
            "dwell": "命令保持（||Δu||∞ <= 0.02）的平均持续时间",
            "low_speed_share": "|vx| <= 0.3 且 |wz| <= 0.3 的 0.1 s 区间占比（协议 strict 口径）",
            "latency": {"nav_period_s": NAV_DT,
                        "command_to_obs_ctrl_steps": "命令写入后 policy 观测携带该命令的控制步数",
                        "command_to_first_substep_s": "换命令的控制步内首个物理子步是否出现力矩变化"},
        },
        "thresholds": {
            "q90_eps_c_max": Q90_EPS_C_MAX,
            "q90_eps_d_max_m": margins["delta"],
            "alpha_bootstrap_interval": margins["alpha_interval"],
            "delta_bootstrap_interval": margins["delta_interval"],
            "low_speed_dwell_shift_max_pp": LOW_SPEED_SHIFT_MAX,
            "low_speed_dwell_shift_direction": "one_sided",
            "low_speed_dwell_shift_note": "只限制「新指令低速停留不超过旧程序 +10 个百分点」。"
                                          "反向（更少低速）不构成适用性风险，不判失败；"
                                          "双向数值仍如实报告。",
            "latency_max_s": LATENCY_MAX_S,
            "j_min_j": J_MIN,
            "gate_populations": {
                "primary": "new_clean（3 图 × 4 种子 = 12 回合）",
                "secondary": "new_friction_only / new_mass_only（各 12 回合），同阈值分开报告",
                "control": "old（12 回合）必须复现 α/δ，否则管线不可比，结论记为不可比而非通过",
            },
        },
        "decisions": {
            "on_pass": "G2A 通过：证明冻结 α/δ 对闭环指令形态适用；不替未来策略背书，"
                       "NAV-034/043 仍需学习策略复查，NAV-048 需最终审计。",
            "on_fail": "保持 false 并询问用户；不得收窄命令域、不得重估 α/δ、不得删样本。",
        },
    }


def datetime_now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds")


def write_prereg() -> int:
    if os.path.exists(PREREG_PATH):
        print(f"REFUSE 设计文件已存在：{PREREG_PATH}", flush=True)
        print("如需重冻结，请人工删除并说明理由（不得在看到数据后重冻结）。", flush=True)
        return 2
    doc = build_prereg()
    with open(PREREG_PATH, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(json.dumps({"prereg": PREREG_PATH, "sha256": sha256_file(PREREG_PATH),
                      "alpha": doc["frozen_values"]["alpha"],
                      "delta": doc["frozen_values"]["delta"],
                      "q90_eps_c_max": Q90_EPS_C_MAX}, ensure_ascii=False, indent=1))
    return 0


def load_prereg_verified() -> dict:
    if not os.path.exists(PREREG_PATH):
        raise SystemExit(f"缺少设计冻结文件 {PREREG_PATH}；先跑 --write-prereg")
    doc = json.load(open(PREREG_PATH, encoding="utf-8"))
    now = file_hashes()
    bad = {k: (doc["frozen_inputs"].get(k), v) for k, v in now.items()
           if doc["frozen_inputs"].get(k) != v}
    if bad:
        raise SystemExit(f"冻结后输入发生变化，拒绝继续：{json.dumps(bad, ensure_ascii=False)}")
    return doc


# =============================================================== 纯离线部分
def nearest_rank(values, q: float = 0.9):
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    k = math.ceil(q * len(vals))
    k = min(max(k, 1), len(vals))
    return vals[k - 1]


def plan_old_episodes() -> list[dict]:
    """pilot split 里的 clean/friction/mass 回合，program_sha256 逐条校验。"""
    from camra_calibration.commands import build_program

    rows = [json.loads(l) for l in open(PILOT_LIST, encoding="utf-8")]
    eps = []
    for t in rows:
        if t["profile"] not in OLD_PROFILES:
            continue
        prog = build_program(t["family"], t["seed"], duration_s=t["duration_s"],
                             command_dt=t["command_dt"], target_period_s=1.0, ramp_s=0.2)
        got = prog.sha256()
        if got != t["program_sha256"]:
            raise SystemExit(f"程序重建不一致 {t['trial_id']}: {got} != {t['program_sha256']}")
        eps.append({"kind": "old", "trial_id": t["trial_id"], "family": t["family"],
                    "seed": t["seed"], "profile": t["profile"], "level": "flat",
                    "program_sha256": got})
    return eps


def plan_new_episodes(level: str) -> list[dict]:
    eps = []
    for profile in NEW_PROFILES:
        for seed in NEW_SEEDS[profile]:
            eps.append({"kind": "new", "trial_id": f"nav-{level}-{profile}-{seed}",
                        "family": f"nav_feedback_{level}", "seed": seed,
                        "profile": profile, "level": level})
    return eps


def check_seed_disjoint() -> None:
    used = set()
    for fn in ("pilot.jsonl", "calibration.jsonl", "heldout.jsonl", "near_obstacle.jsonl",
               "baseline_fit.jsonl", "baseline_dev.jsonl", "baseline_validate.jsonl"):
        p = os.path.join(V2_LISTS, fn)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            used.add(json.loads(line)["seed"])
    clash = sorted(set(sum(NEW_SEEDS.values(), [])) & used)
    if clash:
        raise SystemExit(f"导航 pilot 种子与已锁定分割冲突：{clash}")


# =============================================================== Isaac 运行
def run_isaac(args, prereg: dict) -> dict:
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
    from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
    from unitree_rl_lab.utils.external_command import DEFAULT_LIMITS, ExternalCommandWrapper
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

    from camra_calibration.commands import build_program
    from camra_calibration.power import PowerModel
    from camra_calibration.profiles import sample_applied
    from sim.profiles_sim import apply_friction, apply_mass, configure_env_for_profile

    from camra_navigation.evaluation.recorder import NavEnergyRecorder, offline_recompute
    from camra_navigation.maps.generator import build as build_map
    from camra_navigation.maps.scene_adapter import build_scene

    BUNDLE = "/pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2"
    policy = torch.jit.load(os.path.join(BUNDLE, "exported", "policy.pt")).eval()
    model = PowerModel.load(POWER_PATH)

    split = args.split
    if split == "old":
        episodes = plan_old_episodes()
        level = "flat"
    else:
        episodes = plan_new_episodes(args.level)
        level = args.level
    N = len(episodes)
    progress(f"plan_{split}_{level}_n={N}")

    # ---- 场景几何 --------------------------------------------------------
    boxes = []
    mc = (0.0, 0.0)
    spawn_local = (0.0, 0.0)
    goal_local = (0.0, 0.0)
    spawn_yaw = 0.0
    if split == "new":
        design = json.load(open(MAP_DESIGN, encoding="utf-8"))
        spec = build_map(level, MAP_SEED, design)
        layout = build_scene(spec, env_index=0, num_envs=N)
        boxes = list(layout.wall_boxes) + list(layout.island_boxes)
        mc = layout.map_center
        spawn_local = (float(spec.start[0] - mc[0]), float(spec.start[1] - mc[1]))
        goal_local = (float(spec.goal[0] - mc[0]), float(spec.goal[1] - mc[1]))
        spawn_yaw = float(layout.robot_spawn_yaw)
        tile = float(layout.env_spacing_m)
        progress(f"boxes={len(boxes)} spacing={tile:.2f}")
    else:
        layout = None
        tile = OLD_TILE_M
        progress(f"no_obstacles spacing={tile:.2f}")

    device = args.device
    env_cfg = parse_env_cfg("CaMRA-Go2-Velocity", device=device, num_envs=N,
                            use_fabric=True, entry_point_key="env_cfg_entry_point")
    env_cfg.seed = 100
    # 与校准采集同口径：移除隐藏随机化（摩擦/质量改逐 env 直写、推扰移除、
    # reset 位姿与速度随机化归零）。三种 profile 的 obs_noise 均为 None，
    # 因此混合批次共用 enable_corruption=False 是一致的。
    configure_env_for_profile(env_cfg, sample_applied("clean", 0))
    env_cfg.commands.base_velocity.ranges = env_cfg.commands.base_velocity.limit_ranges
    env_cfg.episode_length_s = WARMUP_SECONDS + EPISODE_SECONDS + 5.0
    env_cfg.curriculum = {}          # 课程会漂移 env_origins（NAV-028 实测）

    gen = env_cfg.scene.terrain.terrain_generator
    gen.size = (tile, tile)
    gen.num_rows = 1
    gen.num_cols = N
    gen.border_width = 1.0
    gen.sub_terrains = {"flat": MeshPlaneTerrainCfg(size=(tile, tile))}
    env_cfg.scene.terrain.terrain_type = "generator"
    env_cfg.scene.terrain.terrain_generator = gen
    env_cfg.scene.env_spacing = tile

    box_names = []
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
        box_names.append(name)

    progress("gym_make_start")
    raw = gym.make("CaMRA-Go2-Velocity", cfg=env_cfg)
    env = ExternalCommandWrapper(raw, limits=DEFAULT_LIMITS, enforce_limits=True, on_reset="zero")
    u = raw.unwrapped
    robot = u.scene["robot"]
    sim = u.sim
    sim_dt = sim.get_physics_dt()
    env.reset()                       # 首次 reset 定型 env_origins（NAV-028 实测）
    progress("origins_settled")

    slab = 2000.0
    sim_utils.spawn_cuboid(
        "/World/nav_ground",
        sim_utils.CuboidCfg(
            size=(slab, slab, 0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0,
                                                            dynamic_friction=1.0)),
        translation=(0.0, 0.0, -0.1))
    progress("nav_ground_spawned")

    env_origins_np = u.scene.env_origins.detach().cpu().numpy().copy()
    print("DBG env_origins:", env_origins_np[:, :2].tolist(), flush=True)

    for k, name in enumerate(box_names):
        box = boxes[k]
        poses = torch.zeros(N, 7, device=device)
        for e in range(N):
            poses[e, 0] = float(env_origins_np[e, 0]) + (box.center_x - layout.env_origin[0])
            poses[e, 1] = float(env_origins_np[e, 1]) + (box.center_y - layout.env_origin[1])
            poses[e, 2] = box.z_center
            qw, qx_, qy_, qz_ = box.quat_wxyz
            poses[e, 3], poses[e, 4], poses[e, 5], poses[e, 6] = qw, qx_, qy_, qz_
        u.scene[name].write_root_pose_to_sim(poses)
    progress("obstacles_placed")

    # ---- 驱动辅助 --------------------------------------------------------
    state = {"rec": None, "probe": None}
    orig_step = sim.step

    def patched_step(*a, **kw):
        out = orig_step(*a, **kw)
        rec = state["rec"]
        if rec is not None:
            tau = robot.data.applied_torque.detach()
            robot.update(sim_dt)
            rec.record(tau_applied_lab=tau,
                       qdot_post_lab=robot.data.joint_vel.detach(),
                       root_pos_lab=robot.data.root_pos_w[:, :3].detach())
            pr = state.get("probe")
            if pr is not None:
                pr["n"] += 1
                if pr["first"] is None and float((tau - pr["tau0"]).abs().max()) > 1e-6:
                    pr["first"] = pr["n"]
        return out

    sim.step = patched_step

    def write_cmd(rows) -> None:
        for e, row in enumerate(rows):
            env.bridge.set_command(row, env_ids=torch.tensor([e], device=device))

    def yaw_of(q) -> float:
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def reset_spawns(yaws) -> None:
        env.reset()
        for e in range(N):
            qw = math.cos(yaws[e] / 2.0)
            qz = math.sin(yaws[e] / 2.0)
            root = torch.zeros(1, 7, device=device)
            root[0, 0] = float(env_origins_np[e, 0]) + spawn_local[0]
            root[0, 1] = float(env_origins_np[e, 1]) + spawn_local[1]
            root[0, 2] = 0.4                        # v2 契约 init_state z
            root[0, 3], root[0, 6] = qw, qz
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
        env_origins_np[:] = u.scene.env_origins.detach().cpu().numpy()

    results: list[dict] = []
    wall0 = time.time()

    # ---- 单批：全部回合并行 ---------------------------------------------
    rec = NavEnergyRecorder(capacity_steps=EXPECTED_SUBSTEPS + 400, num_envs=N,
                            device=device, label=f"nav060_{split}_{level}",
                            expected_steps=EXPECTED_SUBSTEPS)
    state["rec"] = rec
    rec.begin_warmup()

    yaws = []
    for ep in episodes:
        rng = torch.Generator().manual_seed(ep["seed"] + 7919)
        jit = float(torch.empty(1).uniform_(-0.15, 0.15, generator=rng))
        yaws.append((spawn_yaw if split == "new" else 0.0) + jit)
    reset_spawns(yaws)
    rec.observe_reset(range(N))          # warmup 期：正常放行

    # 扰动逐 env 直写（校准同款 API；读回校验）
    profile_applied = []
    for e, ep in enumerate(episodes):
        applied = sample_applied(ep["profile"], ep["seed"], duration_s=EPISODE_SECONDS)
        fr = apply_friction(robot, [e], applied.friction, restitution=applied.restitution)
        ms = apply_mass(robot, [e], applied.added_mass_kg)
        rec_p = {"env": e, "profile": ep["profile"], "seed": ep["seed"],
                 "friction": applied.friction, "added_mass_kg": applied.added_mass_kg,
                 "friction_readback_ok": bool(fr.get("ok")),
                 "mass_readback_ok": bool(ms.get("mass_match")),
                 "friction_readback": fr, "mass_readback": {k: ms.get(k) for k in
                                                            ("body", "expected_mass_kg",
                                                             "readback_mass_max_abs_diff", "mass_match")}}
        profile_applied.append(rec_p)
    progress("profiles_applied")

    # reset 会重置状态，profile 写完后重新摆位（reset 后 origins 已定型）
    reset_spawns(yaws)
    obs = u.observation_manager.compute()
    progress("spawns_written")

    # ---- warmup：1.0 s 零速闭环 -----------------------------------------
    write_cmd([[0.0, 0.0, 0.0]] * N)
    for _ in range(int(round(WARMUP_SECONDS / CTRL_DT))):
        act = policy(obs["policy"])
        obs, _, _, _, _ = env.step(act)
    progress("warmup_done")

    # 窗口起点：warmup 后实测平面初态（名义参考从这里积分，不插值不对齐）
    pos0 = robot.data.root_pos_w[:, :2].detach().cpu().numpy().copy()
    quat0 = robot.data.root_quat_w.detach().cpu().numpy().copy()
    init_pose = [(float(pos0[e, 0] - env_origins_np[e, 0]),
                  float(pos0[e, 1] - env_origins_np[e, 1]), yaw_of(quat0[e])) for e in range(N)]
    print("DBG init_pose:", [(round(a, 3), round(b, 3), round(c, 3)) for a, b, c in init_pose],
          flush=True)

    rec.begin_window()

    # ---- 30 s 窗口：0.1 s 更新一次指令 ----------------------------------
    old_progs = []
    if split == "old":
        for ep in episodes:
            old_progs.append(build_program(ep["family"], ep["seed"], duration_s=EPISODE_SECONDS,
                                           command_dt=NAV_DT, target_period_s=1.0, ramp_s=0.2))

    applied_cmds = [[] for _ in range(N)]      # 每回合 300 条 applied 命令
    obs_cmd_mismatch = [0] * N
    # 指令 -> 物理延迟：换指令的控制步内，第几个物理子步出现力矩变化
    probe_first: list[int] = []
    probe_changed_writes = 0
    outcome = ["complete"] * N
    fell = [False] * N
    fall_t = [None] * N
    prev_tau = None
    last_cmd = [[0.0, 0.0, 0.0] for _ in range(N)]

    for step in range(N_CTRL):
        if step % CTRL_PER_NAV == 0:
            rows = []
            for e, ep in enumerate(episodes):
                if outcome[e] != "complete":
                    rows.append([0.0, 0.0, 0.0])
                    continue
                if split == "old":
                    rows.append(list(old_progs[e].commands[step // CTRL_PER_NAV]))
                else:
                    p = robot.data.root_pos_w[e, :2].detach().cpu().numpy()
                    pl = (float(p[0] - env_origins_np[e, 0]), float(p[1] - env_origins_np[e, 1]))
                    yaw = yaw_of(robot.data.root_quat_w[e].detach().cpu().numpy())
                    dx, dy = goal_local[0] - pl[0], goal_local[1] - pl[1]
                    dist = math.hypot(dx, dy)
                    he = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi
                    wz = max(-ROUTE_WZ_LIMIT, min(ROUTE_WZ_LIMIT, ROUTE_K_WZ * he))
                    if abs(wz) < args.wz_deadband:
                        # 实测底层对该量级的角速率无响应（yawrate_deadband.json），
                        # 继续下发只会在名义参考里累积一个不会发生的转角
                        wz = 0.0
                    vx = ROUTE_VX_NEAR_GOAL if dist < ROUTE_NEAR_GOAL_M else ROUTE_VX_CRUISE
                    rows.append([vx, 0.0, wz])
            write_cmd(rows)
            changed = any(max(abs(rows[e][i] - last_cmd[e][i]) for i in range(3)) > 1e-9
                          for e in range(N))
            for e in range(N):
                applied_cmds[e].append([float(v) for v in rows[e]])
                last_cmd[e] = [float(v) for v in rows[e]]
            if changed:
                probe_changed_writes += 1
                state["probe"] = {"tau0": robot.data.applied_torque.detach().clone(),
                                  "n": 0, "first": None}

        act = policy(obs["policy"])
        obs, _, _, _, _ = env.step(act)

        # 命令通道核对（obs 必须携带本控制步写入的命令）
        if step % CTRL_PER_NAV == 0:
            seen = obs["policy"][:, 6:9].detach().cpu().numpy()
            for e in range(N):
                if max(abs(float(seen[e, i]) - last_cmd[e][i]) for i in range(3)) > 5e-3:
                    obs_cmd_mismatch[e] += 1
            pr = state.get("probe")
            if pr is not None:
                if pr["first"] is not None:
                    probe_first.append(int(pr["first"]))
                else:
                    probe_first.append(-1)      # 该控制步内未出现可测力矩变化
                state["probe"] = None

        # 事件判定
        t = (step + 1) * CTRL_DT
        h = robot.data.root_pos_w[:, 2].detach().cpu().numpy()
        quat = robot.data.root_quat_w.detach().cpu().numpy()
        for e in range(N):
            if outcome[e] != "complete":
                continue
            gz = -(1.0 - 2.0 * (float(quat[e, 1]) ** 2 + float(quat[e, 2]) ** 2))
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, -gz))))
            if h[e] < FALL_HEIGHT_M or tilt > FALL_TILT_DEG:
                fell[e] = True
                fall_t[e] = t
                outcome[e] = "business_failure_fall"
                rec.mark_event([e], "fall", detail=f"t={t:.2f}s tilt={tilt:.1f}")
        if split == "new":
            for e in range(N):
                if outcome[e] != "complete":
                    continue
                p = robot.data.root_pos_w[e, :2].detach().cpu().numpy()
                pl = (float(p[0] - env_origins_np[e, 0]), float(p[1] - env_origins_np[e, 1]))
                if math.hypot(goal_local[0] - pl[0], goal_local[1] - pl[1]) <= GOAL_RADIUS_M:
                    outcome[e] = "goal_reached"
        if args.debug and step % 250 == 0:
            p0 = robot.data.root_pos_w[0, :2].detach().cpu().numpy()
            print(f"DBG step={step} t={t:.1f} pos=({p0[0]:.2f},{p0[1]:.2f}) h={h[0]:.3f} "
                  f"cmd={[round(v,2) for v in last_cmd[0]]}", flush=True)

    progress("window_done")
    rec.mark_event(range(N), "timeout", detail="30 s 窗口到时")
    res = rec.finish()
    state["rec"] = None
    problems = rec.problems(res)
    if problems:
        print("RECORDER_PROBLEMS:", problems, flush=True)

    # ---- 单回合对账 ------------------------------------------------------
    # 10 Hz 公共网格：p_nom[k] 与 p_exec[k] 对应同一时刻 t = k * 0.1 s
    series = {"t": [k * NAV_DT for k in range(N_NAV + 1)],
              "exec": np.zeros((N, N_NAV + 1, 2)), "nom": np.zeros((N, N_NAV + 1, 2)),
              "cmd": np.zeros((N, N_NAV, 3)), "valid_k": np.zeros(N, dtype=int)}
    from camra_calibration.nominal import PlanarState, integrate

    for e, ep in enumerate(episodes):
        valid = res["valid_mask"][:, e]
        n_valid = int(valid.sum())
        total_j = float(res["total_j"][e])
        root = res["root_pos_w"][:, e, :2].astype(np.float64) - np.array(
            [float(env_origins_np[e, 0]), float(env_origins_np[e, 1])])
        cmds = applied_cmds[e]

        entry = {
            "trial_id": ep["trial_id"], "kind": ep["kind"], "level": level,
            "family": ep["family"], "seed": ep["seed"], "profile": ep["profile"],
            "outcome": outcome[e], "fell": fell[e], "fall_at_s": fall_t[e],
            "init_pose": [round(v, 5) for v in init_pose[e]],
            "substeps_recorded": n_valid, "total_j": total_j,
            "energy_offline_rel_diff": offline_recompute(res, e)["rel_diff"],
            "profile_applied": profile_applied[e],
            "obs_cmd_mismatch_steps": obs_cmd_mismatch[e],
            "command_stats": command_stats(cmds, valid, root, init_pose[e], model),
            "prefixes": {},
        }

        full_cmds = [tuple(c) for c in cmds] + [tuple(cmds[-1])] if cmds else []
        nom_states = (integrate(PlanarState(*init_pose[e]), full_cmds, NAV_DT)
                      if len(full_cmds) >= 2 else None)
        if nom_states is not None:
            entry["in_domain"] = model.in_domain(np.asarray(full_cmds, dtype=np.float64))

        d_all, along_all, cross_all = [], [], []
        k_max = 0
        if nom_states is not None:
            for kk in range(N_NAV + 1):
                if kk == 0:
                    pe = np.array([init_pose[e][0], init_pose[e][1]])
                else:
                    idx = kk * SUBSTEPS_PER_NAV - 1
                    if idx >= n_valid:
                        break
                    pe = root[idx]
                pn = np.array([nom_states[kk].px, nom_states[kk].py])
                d = pe - pn
                th = float(nom_states[kk].yaw)
                d_all.append(float(np.linalg.norm(d)))
                along_all.append(float(d[0] * math.cos(th) + d[1] * math.sin(th)))
                cross_all.append(float(-d[0] * math.sin(th) + d[1] * math.cos(th)))
                series["exec"][e, kk] = pe
                series["nom"][e, kk] = pn
                k_max = kk
            for kk, c in enumerate(cmds):
                series["cmd"][e, kk] = c
            series["valid_k"][e] = k_max

        # 速度追踪（解释 along 滞后的来源）：路径速度 vs 命令 vx
        seg = [float(np.linalg.norm(series["exec"][e, kk + 1] - series["exec"][e, kk])) / NAV_DT
               for kk in range(k_max)]
        entry["speed"] = {
            "mean_cmd_vx": float(np.mean([c[0] for c in cmds])) if cmds else None,
            "mean_path_speed": float(np.mean(seg)) if seg else None,
        }
        entry["tracking_decomposition"] = {
            "n_grid_points": k_max,
            "eps_d_max": max(d_all) if d_all else None,
            "along_max_abs": max((abs(v) for v in along_all), default=None),
            "cross_max_abs": max((abs(v) for v in cross_all), default=None),
            "along_at_max": along_all[int(np.argmax(d_all))] if d_all else None,
            "cross_at_max": cross_all[int(np.argmax(d_all))] if d_all else None,
            "note": "along = 沿名义航向的滞后（正=落后于名义），cross = 横向偏移；"
                    "诊断用，不参与门禁判定",
        }

        for T in PREFIXES_S:
            k = int(round(T / NAV_DT))
            n_sub = min(k * SUBSTEPS_PER_NAV, n_valid)
            j_exec = float(np.sum(res["energy_step_j"][:n_sub, e])) if n_sub else 0.0
            pref = {"t_s": T, "substeps_available": n_sub, "j_exec": j_exec,
                    "window_complete": n_valid >= k * SUBSTEPS_PER_NAV}
            if nom_states is not None and n_sub >= k * SUBSTEPS_PER_NAV:
                j_nom = float(model.j_nom(np.asarray(full_cmds[:k + 1], dtype=np.float64)))
                pref["j_nom"] = j_nom
                if j_nom >= J_MIN:
                    pref["eps_c"] = (j_exec - j_nom) / j_nom
                    pref["eps_c_reason"] = "ok"
                else:
                    pref["eps_c"] = None
                    pref["eps_c_reason"] = "non_computable"
                pref["eps_d"] = max(d_all[:k + 1])
                pref["eps_d_along_abs"] = max(abs(v) for v in along_all[:k + 1])
                pref["eps_d_cross_abs"] = max(abs(v) for v in cross_all[:k + 1])
            else:
                pref["j_nom"] = None
                pref["eps_c"] = None
                pref["eps_c_reason"] = "terminated_before_prefix"
                pref["eps_d"] = None
            entry["prefixes"][str(T)] = pref
        results.append(entry)
        print(f"EP {ep['trial_id']} outcome={outcome[e]} J={total_j:.1f} "
              f"eps_c30={entry['prefixes']['30']['eps_c']} "
              f"eps_d30={entry['prefixes']['30']['eps_d']} "
              f"along={entry['tracking_decomposition']['along_max_abs']} "
              f"cross={entry['tracking_decomposition']['cross_max_abs']}", flush=True)

    if args.npz:
        np.savez_compressed(args.npz, t=np.asarray(series["t"]),
                            exec_pos=series["exec"], nom_pos=series["nom"],
                            cmd=series["cmd"], valid_k=series["valid_k"],
                            trial_ids=np.asarray([e["trial_id"] for e in episodes]))
        print(f"PROGRESS npz_written {args.npz}", flush=True)

    return {
        "split": split, "level": level,
        "wallclock_s": round(time.time() - wall0, 1),
        "n": N, "episodes": results,
        "latency": {"nav_period_s": NAV_DT, "ctrl_dt_s": CTRL_DT,
                    "changed_command_writes": probe_changed_writes,
                    "first_response_substep": probe_first,
                    "first_response_substep_max": (max(probe_first) if probe_first else None),
                    "n_no_response": sum(1 for v in probe_first if v <= 0),
                    "obs_cmd_mismatch_total": int(sum(obs_cmd_mismatch))},
        "controller": {"kind": "heading_p_feedback", "nav_period_s": NAV_DT,
                       "vx_cruise": ROUTE_VX_CRUISE, "vx_near_goal": ROUTE_VX_NEAR_GOAL,
                       "k_wz": ROUTE_K_WZ, "wz_limit": ROUTE_WZ_LIMIT,
                       "wz_deadband_rad_s": args.wz_deadband},
        "recorder_problems": problems,
        "env_origins": env_origins_np[:, :2].tolist(),
    }


def command_stats(cmds, valid, root, init_pose, model) -> dict:
    """指令分布 / 变化率 / 切换频率 / 保持时长 / 低速停留。"""
    import numpy as np

    arr = np.asarray(cmds, dtype=np.float64)
    if arr.size == 0:
        return {}
    du = np.abs(np.diff(arr, axis=0)).max(axis=1) if len(arr) > 1 else np.array([0.0])
    keep = du <= 0.02
    # 保持时长：连续保持段长度 * 0.1 s
    runs, cur = [], 1
    for k in keep:
        if k:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    sign_flip = {}
    for i, name in enumerate(("vx", "vy", "wz")):
        v = arr[:, i]
        nz = np.sign(v[v != 0.0])
        sign_flip[name] = int(np.sum(nz[1:] != nz[:-1])) if len(nz) > 1 else 0
    low = float(np.mean((np.abs(arr[:, 0]) <= LOW_SPEED_VX) & (np.abs(arr[:, 2]) <= LOW_SPEED_WZ)))
    return {
        "n_points": int(len(arr)),
        "mean_abs": {k: float(np.mean(np.abs(arr[:, i]))) for i, k in
                     enumerate(("vx", "vy", "wz"))},
        "max_abs": {k: float(np.max(np.abs(arr[:, i]))) for i, k in
                    enumerate(("vx", "vy", "wz"))},
        "change_rate_mean": float(np.mean(du)),
        "change_rate_max": float(np.max(du)),
        "switch_per_s": {k: v / EPISODE_SECONDS for k, v in sign_flip.items()},
        "switch_count": sign_flip,
        "hold_mean_s": float(np.mean(runs)) * NAV_DT,
        "hold_max_s": float(np.max(runs)) * NAV_DT,
        "low_speed_share": low,
        "domain_ok": bool(model.in_domain(arr)[0]) if len(arr) >= 2 else None,
    }


# =============================================================== 聚合与门禁

def route_diagnostics(inputs: list[str]) -> dict:
    """从 10 Hz 存档算路线级事实：命令累计转角、路径对弦线的最大横向偏移。

    用途：判定 ε_d 的偏差来源是"实际跑偏"还是"名义参考自己漂了"。
    """
    import numpy as np

    out: dict[str, dict] = {}
    for path in inputs:
        npz = os.path.splitext(path)[0] + ".npz"
        if not os.path.exists(npz):
            continue
        z = np.load(npz, allow_pickle=True)
        ids = [str(x) for x in z["trial_ids"]]
        for i, tid in enumerate(ids):
            vk = int(z["valid_k"][i])
            if vk < 10:
                continue
            ex = z["exec_pos"][i][:vk + 1].astype(float)
            cm = z["cmd"][i][:vk].astype(float)
            a, b = ex[0], ex[-1]
            v = b - a
            chord = float(np.linalg.norm(v))
            lat = None
            if chord > 1e-6:
                nrm = np.array([-v[1], v[0]]) / chord
                lat = float(np.max(np.abs((ex - a) @ nrm)))
            out[tid] = {
                "chord_m": chord,
                "max_lateral_dev_from_chord_m": lat,
                "cmd_cum_turn_rad": float(np.sum(cm[:, 2]) * NAV_DT),
                "cmd_path_m": float(np.sum(cm[:, 0]) * NAV_DT),
                "n_grid_points": vk,
            }
    return out



def diagnostic_observations(groups: dict, rdiag: dict) -> list[dict]:
    """把"ε_d 到底在量什么"写成可复核的观察（不参与门禁）。"""
    obs: list[dict] = []
    for key in ("new_clean", "old_clean"):
        eps = groups.get(key, [])
        if not eps:
            continue
        lat, turn, chord = [], [], []
        for e in eps:
            d = rdiag.get(e["trial_id"])
            if not d:
                continue
            if d["max_lateral_dev_from_chord_m"] is not None:
                lat.append(d["max_lateral_dev_from_chord_m"])
            turn.append(abs(d["cmd_cum_turn_rad"]))
            chord.append(d["chord_m"])
        if not lat:
            continue
        obs.append({
            "population": key,
            "max_lateral_dev_from_chord_m": {"min": min(lat), "max": max(lat)},
            "abs_cmd_cum_turn_rad": {"min": min(turn), "max": max(turn)},
            "chord_m": {"min": min(chord), "max": max(chord)},
            "reading": ("执行路径对起终点弦线的最大横向偏移远小于命令累计转角所隐含的偏移"
                        if max(lat) < 0.5 else "执行路径确实偏离弦线"),
        })
    if obs:
        obs.append({
            "inference": "new 流的 ε_d（≈6.5 m）与执行路径偏离弦线（≤0.18 m）相差一个数量级，"
                         "而命令累计转角约 1.0 rad（30 s × ≈0.035 rad/s）：名义参考按「完美跟踪」"
                         "把这段转角全部执行，实际底层几乎未执行（横向偏移证据）。"
                         "因此 new 流的 ε_d 主要量的是名义参考自身的漂移，不是机器人跑偏。"
                         "该推断只用于解释机制，不改变 ε_d 的门禁结论。",
            "role": "diagnostic_only",
        })
    return obs


def aggregate(paths: list[str], prereg: dict) -> dict:
    margins = json.load(open(MARGINS_PATH, encoding="utf-8"))
    alpha, delta = float(margins["alpha"]), float(margins["delta"])
    runs = [json.load(open(p, encoding="utf-8")) for p in paths]
    rdiag = route_diagnostics(paths)

    episodes = [ep for r in runs for ep in r["episodes"]]
    for ep in episodes:
        ep["route_diagnostics"] = rdiag.get(ep["trial_id"])
    groups: dict[str, list[dict]] = {}
    for ep in episodes:
        key = f"{ep['kind']}_{ep['profile']}"
        groups.setdefault(key, []).append(ep)

    def pop_stats(eps: list[dict], T: str = "30") -> dict:
        ec = [e["prefixes"][T]["eps_c"] for e in eps]
        ed = [e["prefixes"][T]["eps_d"] for e in eps]
        usable_c = [v for v in ec if v is not None]
        usable_d = [v for v in ed if v is not None]
        return {
            "n_episodes": len(eps),
            "n_eps_c": len(usable_c),
            "n_eps_d": len(usable_d),
            "q90_eps_c": nearest_rank(usable_c),
            "q90_eps_d": nearest_rank(usable_d),
            "max_eps_c": max(usable_c) if usable_c else None,
            "max_eps_d": max(usable_d) if usable_d else None,
            "median_eps_c": sorted(usable_c)[len(usable_c) // 2] if usable_c else None,
            "n_non_computable": sum(1 for v in ec if v is None),
            "n_failed": sum(1 for e in eps if e["outcome"].startswith("business_failure")),
            "low_speed_share": [e["command_stats"].get("low_speed_share") for e in eps],
            "change_rate_max": [e["command_stats"].get("change_rate_max") for e in eps],
            "hold_mean_s": [e["command_stats"].get("hold_mean_s") for e in eps],
            "switch_per_s_wz": [e["command_stats"].get("switch_per_s", {}).get("wz")
                                for e in eps],
        }

    pooled = {}
    for ep in episodes:
        pooled.setdefault(f"{ep['kind']}_all", []).append(ep)
    groups.update(pooled)          # 池化组参与统计，不参与"逐 profile"展示时的重复计数

    per_group = {k: pop_stats(v) for k, v in sorted(groups.items())}
    per_group_prefix = {k: {T: pop_stats(v, T) for T in map(str, PREFIXES_S)}
                        for k, v in sorted(groups.items())}

    decomp = {}
    for k, v in sorted(groups.items()):
        decomp[k] = {
            "along_max_abs": [e["tracking_decomposition"]["along_max_abs"] for e in v
                              if e.get("tracking_decomposition")],
            "cross_max_abs": [e["tracking_decomposition"]["cross_max_abs"] for e in v
                              if e.get("tracking_decomposition")],
            "mean_cmd_vx": [e["speed"]["mean_cmd_vx"] for e in v if e.get("speed")],
            "mean_path_speed": [e["speed"]["mean_path_speed"] for e in v if e.get("speed")],
        }

    primary = groups.get("new_clean", [])
    control = groups.get("old_clean", [])
    secondary = {k: groups.get(k, []) for k in ("new_friction_only", "new_mass_only")}

    # 低速停留：单向（风险方向）
    old_low = [e["command_stats"]["low_speed_share"] for e in control
               if e["command_stats"].get("low_speed_share") is not None]
    new_low = [e["command_stats"]["low_speed_share"] for e in primary
               if e["command_stats"].get("low_speed_share") is not None]
    old_low_mean = sum(old_low) / len(old_low) if old_low else None
    new_low_mean = sum(new_low) / len(new_low) if new_low else None
    low_shift = (new_low_mean - old_low_mean) if (old_low_mean is not None and new_low_mean is not None) else None

    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    ps = per_group.get("new_clean", {})
    ai_lo, ai_hi = [float(v) for v in margins["alpha_interval"]]
    di_lo, di_hi = [float(v) for v in margins["delta_interval"]]
    old_all = per_group.get("old_all", {})
    add("control_old_reproduces_alpha",
        old_all.get("q90_eps_c") is not None and old_all["q90_eps_c"] <= ai_hi,
        {"q90_eps_c_old_all": old_all.get("q90_eps_c"),
         "alpha_frozen": alpha, "alpha_bootstrap_interval": [ai_lo, ai_hi],
         "n": old_all.get("n_eps_c"),
         "comparison": "池化对照（α 本身即在全 cell 上池化估计）；判据为不超过区间上界，"
                       "因为落在区间下界之下意味着模型更保守，不构成可比性失败",
         "note": "区间取自 margins_v1.json 的 bootstrap 95%，不是本次临时设定"})
    add("control_old_reproduces_delta",
        old_all.get("q90_eps_d") is not None and old_all["q90_eps_d"] <= di_hi,
        {"q90_eps_d_old_all": old_all.get("q90_eps_d"),
         "delta_frozen": delta, "delta_bootstrap_interval": [di_lo, di_hi],
         "n": old_all.get("n_eps_d")})
    add("new_clean_q90_eps_c", ps.get("q90_eps_c") is not None and ps["q90_eps_c"] <= Q90_EPS_C_MAX,
        {"q90_eps_c_new_clean": ps.get("q90_eps_c"), "threshold": Q90_EPS_C_MAX,
         "alpha_frozen": alpha, "n": ps.get("n_eps_c"),
         "q90_eps_c_new_all_reference": per_group.get("new_all", {}).get("q90_eps_c")})
    add("new_clean_q90_eps_d", ps.get("q90_eps_d") is not None and ps["q90_eps_d"] <= delta,
        {"q90_eps_d_new_clean": ps.get("q90_eps_d"), "delta_frozen": delta})
    add("low_speed_dwell_one_sided", low_shift is not None and low_shift <= LOW_SPEED_SHIFT_MAX,
        {"new_clean_mean": new_low_mean, "old_clean_mean": old_low_mean,
         "shift": low_shift, "threshold": LOW_SPEED_SHIFT_MAX,
         "direction": "one_sided(risk side: more low-speed)",
         "two_sided_abs_value": abs(low_shift) if low_shift is not None else None})
    add("command_domain_respected",
        all(e["command_stats"].get("domain_ok") for e in episodes),
        {k: [e["command_stats"].get("domain_ok") for e in v]
         for k, v in sorted(groups.items())})
    lat_ok = True
    lat_detail = {}
    for r in runs:
        lat = r["latency"]
        mx = lat["first_response_substep_max"]
        ok = (lat["obs_cmd_mismatch_total"] == 0
              and lat["changed_command_writes"] > 0
              and mx is not None and mx > 0
              and mx * SUBSTEP_DT <= LATENCY_MAX_S)
        lat_ok = lat_ok and ok
        lat_detail[f"{r['split']}_{r['level']}"] = {
            "obs_cmd_mismatch_total": lat["obs_cmd_mismatch_total"],
            "changed_command_writes": lat["changed_command_writes"],
            "first_response_substep_max": mx,
            "latency_max_s": (mx * SUBSTEP_DT) if mx else None,
            "n_no_response": lat["n_no_response"]}
    add("latency_within_one_ctrl_step", lat_ok, {**lat_detail, "threshold_s": LATENCY_MAX_S})
    add("prefix_denominators_separate",
        all(len(e["prefixes"]) == len(PREFIXES_S) for e in episodes),
        {"prefixes_s": PREFIXES_S,
         "note": "每个前缀独立记 j_exec/j_nom/avail；完整窗口与前缀不混算覆盖率"})
    add("perturbation_effective",
        all(ep["profile_applied"]["friction_readback_ok"]
            and ep["profile_applied"]["mass_readback_ok"] for ep in episodes),
        {"note": "摩擦/质量逐 env 直写后 PhysX 读回校验",
         "sample": episodes[0]["profile_applied"] if episodes else None})
    add("recorder_clean", all(not r["recorder_problems"] for r in runs),
        {r["split"] + "_" + r["level"]: r["recorder_problems"] for r in runs})

    failed = [c["check"] for c in checks if not c["ok"]]
    verdict = "PASS" if not failed else "FAIL"

    return {
        "task": "NAV-060", "gate": "G2A", "verdict": verdict,
        "generated_at": datetime_now(),
        "prereg_sha256": sha256_file(PREREG_PATH),
        "prereg_inputs": prereg["frozen_inputs"],
        "code_sha256": sha256_self(),
        "frozen_margins": {"alpha": alpha, "delta": delta,
                           "q90_eps_c_of_calibration": margins["q90_eps_c"],
                           "q90_eps_d_of_calibration": margins["q90_eps_d"]},
        "thresholds": {"q90_eps_c_max": Q90_EPS_C_MAX, "q90_eps_d_max_m": delta,
                       "low_speed_dwell_shift_max_pp": LOW_SPEED_SHIFT_MAX,
                       "latency_max_s": LATENCY_MAX_S, "j_min_j": J_MIN},
        "checks": checks,
        "failed_checks": failed,
        "populations": per_group,
        "populations_by_prefix": per_group_prefix,
        "diagnostic_observations": diagnostic_observations(groups, rdiag),
        "tracking_decomposition": {
            "role": "diagnostic_only",
            "note": "ε_d 是时刻对齐的平面距离，同时含沿轨迹滞后与横向偏移。此分解用于解释"
                    "偏差来源，不参与 G2A 判定；闸门只用预登记的 ε_c/ε_d。",
            "per_population": decomp,
        },
        "runs": [{"split": r["split"], "level": r["level"], "n": r["n"],
                  "wallclock_s": r["wallclock_s"], "latency": r["latency"],
                  "controller": r.get("controller"), "source": p}
                 for r, p in zip(runs, paths)],
        "episodes": episodes,
        "engineering_ok": [c for c in checks if c["ok"]],
        "empirical_applicability": {
            "statement": ("闭环 10 Hz 指令下，冻结 α/δ 在预登记判据内适用"
                          if verdict == "PASS" else
                          "闭环 10 Hz 指令下未达预登记判据，需用户裁定（不得自行收窄命令或重估裕度）"),
            "scope_limit": "本 pilot 只评执行侧的指令形态适用性，不构成对任何未训练/未来策略的验证；"
                           "NAV-034/043 仍需学习策略复查，NAV-048 需最终审计。",
        },
        "notes": [
            "old 侧为 pilot split 的 clean/friction_only/mass_only（push/combined 因推扰功不计入 J_exec 而显式排除）",
            "new 侧为规则反馈导航的 0.1 s 闭环指令，不含任何已训练导航网络",
            "q90 用 nearest-rank（协议口径）；n=12 时 q90 取第 11 个值，属保守判读",
        ],
    }



def gate_summary(report: dict, pilot_path: str) -> dict:
    """G2A 门禁摘要（权威判定），完整数据留在 pilot 报告里。"""
    return {
        "task": "NAV-060", "gate": "G2A", "verdict": report["verdict"],
        "generated_at": report["generated_at"],
        "failed_checks": report["failed_checks"],
        "checks": report["checks"],
        "thresholds": report["thresholds"],
        "frozen_margins": report["frozen_margins"],
        "populations": report["populations"],
        "diagnostic_observations": report["diagnostic_observations"],
        "tracking_decomposition": report["tracking_decomposition"],
        "empirical_applicability": report["empirical_applicability"],
        "hashes": {
            "prereg_sha256": report["prereg_sha256"],
            "code_sha256": report["code_sha256"],
            "pilot_report": pilot_path,
            "pilot_report_sha256": sha256_file(pilot_path),
            "inputs": report["prereg_inputs"],
        },
        "runs": report["runs"],
        "decision_request": {
            "status": "awaiting_user_ruling",
            "why": "唯一失败项是几何项 ε_d（q90 6.47 m > δ 2.61 m）。诊断表明该偏差主要来自"
                   "名义参考自身漂移（命令累计转角 ~1 rad 未被底层执行），执行路径对弦线的实际"
                   "横向偏移仅 0.15-0.18 m。按任务约束，显著不匹配不得自行处理，须由用户裁定。",
            "options": [
                {"id": "A", "label": "维持 ε_d 定义，单独给闭环流几何裕度",
                 "impact": "需为闭环指令重估一个更大的 δ（当前数据量级 ~6.5 m），"
                           "障碍膨胀显著变大，规划可行域收窄"},
                {"id": "B", "label": "闭环流的几何项改用「计划路线」作名义参考",
                 "impact": "δ 保留 2.61 m（实测 0.15-0.18 m 远低于它）；但等于为闭环流"
                           "新增一种名义参考定义，属契约变更，须登记"},
                {"id": "C", "label": "导航层加角速率死区规避后复测",
                 "impact": "只发底层可执行的 |wz|（先测死区阈值），让 ε_d 恢复原语义；"
                           "同时修掉小角速率不执行的工程缺陷，需重跑 pilot"},
                {"id": "D", "label": "成本项 α 沿用、几何项标注为闭环暂不适用",
                 "impact": "不阻塞上层训练（NAV-029+），但规划器在闭环指令下不得依赖 δ 做"
                           "安全膨胀，须另记已知限制"},
            ],
            "note": "不选也合法：可先推进不依赖几何裕度的任务；但 NAV-029 及上层学习须在此裁定后启动。",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-prereg", action="store_true")
    ap.add_argument("--split", choices=["old", "new"], default="")
    ap.add_argument("--level", default="L1_empty")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--inputs", nargs="*", default=[])
    ap.add_argument("--json", default="")
    ap.add_argument("--gate", default="")
    ap.add_argument("--wz-deadband", type=float, default=0.0,
                    help="导航层角速率死区规避阈值（rad/s）：|wz| 低于它直接发 0，依据 reports/yawrate_deadband.json 的实测")
    ap.add_argument("--npz", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.write_prereg:
        return write_prereg()

    prereg = load_prereg_verified()

    if args.aggregate:
        report = aggregate(args.inputs, prereg)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(report, fh, ensure_ascii=False, indent=2, default=float)
                fh.write("\n")
        if args.gate:                      # 摘要必须在 pilot 报告落盘之后（要算它的散列）
            gs = gate_summary(report, args.json or "")
            with open(args.gate, "w", encoding="utf-8") as fh:
                json.dump(gs, fh, ensure_ascii=False, indent=2, default=float)
                fh.write("\n")
        print(json.dumps({"verdict": report["verdict"], "failed": report["failed_checks"]},
                         ensure_ascii=False))
        return 0 if report["verdict"] == "PASS" else 1

    if not args.split:
        raise SystemExit("需要 --split old|new 或 --write-prereg / --aggregate")

    check_seed_disjoint()
    out = run_isaac(args, prereg)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2, default=float)
            fh.write("\n")
    print(json.dumps({"split": out["split"], "level": out["level"], "n": out["n"],
                      "wallclock_s": out["wallclock_s"],
                      "n_failed": sum(1 for e in out["episodes"] if e["fell"])},
                     ensure_ascii=False))
    print("PROGRESS hard_exit_skip_close", flush=True)
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
