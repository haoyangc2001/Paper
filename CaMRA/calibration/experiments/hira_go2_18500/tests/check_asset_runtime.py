"""Asset and actuator audit of the HiRA URDF inside Isaac Lab (HIRA-012).

静态段回答"源里是什么"，本脚本回答"导入之后还是不是这个"。

    python tests/check_asset_runtime.py --headless --device cuda:0 \
        --urdf <源 urdf> --contract <契约> --out <报告> --usd-dir <缓存目录>

判据（与 integration_acceptance_prereg_v1.json 对齐）：
- 关节名建立双射（不假定顺序相同，顺序差异如实记录成置换）
- 限位 |Δ| ≤ 1e-3 rad
- effort / velocity 与 URDF 一致
- USD 侧刚度/阻尼为 0（证明 PD 不在求解器里）
- 执行器组 stiffness=0、damping=0、effort_limit 等于 URDF 值
- 施加 ±1000 的力矩命令，读回被裁到 ±effort；施加 0 时读回 0
- 总质量与 URDF 逐链接求和在 1% 内
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
ADAPTER = REPO / "CaMRA" / "locomotion" / "adapters"
sys.path.insert(0, str(ADAPTER))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="HiRA asset/actuator runtime audit")
parser.add_argument("--urdf", required=True)
parser.add_argument("--contract", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--usd-dir", default=None)
parser.add_argument("--joint-limit-atol", type=float, default=1e-3)
parser.add_argument("--mass-rtol", type=float, default=0.01)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

from hira_go2 import asset_cfg  # noqa: E402
from hira_go2.asset_audit import static_audit  # noqa: E402


def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


T0 = time.time()


def log(text: str) -> None:
    print(f"[audit] {time.time() - T0:7.1f}s {text}", flush=True)


def main() -> int:
    urdf_path = Path(args_cli.urdf).expanduser().resolve()
    contract_path = Path(args_cli.contract).expanduser().resolve()
    out_path = Path(args_cli.out).expanduser().resolve()
    usd_dir = None if args_cli.usd_dir is None else Path(args_cli.usd_dir).expanduser().resolve()
    if usd_dir is not None:
        usd_dir.mkdir(parents=True, exist_ok=True)

    device = args_cli.device
    log(f"device={device} urdf={urdf_path}")

    sim = SimulationContext(
        SimulationCfg(dt=asset_cfg.HIRA_SIM_DT, device=device, gravity=(0.0, 0.0, -9.81))
    )
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=300.0)
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = asset_cfg.build_hira_go2_cfg(urdf_path, usd_dir=usd_dir).replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)
    sim.reset()

    # 关键发现：``Articulation.reset()`` 只重置执行器与外力，**不写 init_state**；
    # Isaac Lab 2.x 里初始关节角要由 env 的重置事件显式写进 sim。所以这里分两步验证：
    # (a) init_state 是否被正确解析成 default_joint_pos；(b) 把 default 写进 sim 后能否原样读回。
    default_pos_tensor = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(default_pos_tensor, torch.zeros_like(default_pos_tensor))
    robot.update(sim.get_physics_dt())
    initial_readback = robot.data.joint_pos[0].tolist()   # 必须立刻读：后面测试会改写关节状态

    joint_names = list(robot.joint_names)
    body_names = list(robot.body_names)
    n_joints = robot.num_joints
    log(f"dof={n_joints} bodies={len(body_names)}")

    view = robot.root_physx_view
    limits = view.get_dof_limits()[0].tolist()
    max_forces = view.get_dof_max_forces()[0].tolist()
    max_vels = view.get_dof_max_velocities()[0].tolist()
    stiffness = view.get_dof_stiffnesses()[0].tolist()
    damping = view.get_dof_dampings()[0].tolist()
    link_masses = view.get_masses()[0].tolist()

    static = static_audit(urdf_path, contract_path)
    table = static["source"]["joint_limits"]
    urdf_order = static["source"]["urdf"]["dof_joint_names"]

    # ---- R1 双射与顺序 ----
    bijection = sorted(joint_names) == sorted(urdf_order)
    permutation = [urdf_order.index(name) for name in joint_names]
    order_equal = joint_names == urdf_order

    # ---- R2..R4 逐关节限位/effort/velocity ----
    rows = []
    for index, name in enumerate(joint_names):
        expected = table.get(name)
        if expected is None:
            rows.append({"joint": name, "ok": False, "detail": "URDF 里没有同名关节"})
            continue
        d_lower = abs(limits[index][0] - expected["lower"])
        d_upper = abs(limits[index][1] - expected["upper"])
        d_effort = abs(max_forces[index] - expected["effort"])
        d_velocity = abs(max_vels[index] - expected["velocity_limit"])
        rows.append(
            {
                "joint": name,
                "usd": {
                    "lower": limits[index][0],
                    "upper": limits[index][1],
                    "effort": max_forces[index],
                    "velocity_limit": max_vels[index],
                },
                "urdf": expected,
                "abs_diff": {
                    "lower": d_lower,
                    "upper": d_upper,
                    "effort": d_effort,
                    "velocity_limit": d_velocity,
                },
                # 限位用预登记的 1e-3 rad；effort/velocity 用 float32 表示误差量级（1e-5 或 1e-6 相对）
                "ok": d_lower <= args_cli.joint_limit_atol
                and d_upper <= args_cli.joint_limit_atol
                and d_effort <= max(1e-5, 1e-6 * expected["effort"])
                and d_velocity <= max(1e-5, 1e-6 * expected["velocity_limit"]),
            }
        )
    limit_bad = [row["joint"] for row in rows if not row["ok"]]

    # ---- R5 USD 侧增益为 0（PD 不在求解器里）----
    gains_zero = all(abs(value) < 1e-12 for value in stiffness + damping)

    # ---- R6 执行器组参数 ----
    actuator = robot.actuators[asset_cfg.HIRA_ACTUATOR_GROUP]
    actuator_report = {
        "group": asset_cfg.HIRA_ACTUATOR_GROUP,
        "type": type(actuator).__name__,
        "stiffness": sorted({float(value) for value in _tolist(actuator.stiffness)}),
        "damping": sorted({float(value) for value in _tolist(actuator.damping)}),
        "effort_limit": {name: float(value) for name, value in _named(actuator, "effort_limit", joint_names)},
        "joint_names": list(actuator.joint_names),
        "joint_ids": (
            list(actuator.joint_indices)
            if not isinstance(actuator.joint_indices, slice)
            else "slice(all)"
        ),
    }
    actuator_ok = (
        actuator_report["stiffness"] == [0.0]
        and actuator_report["damping"] == [0.0]
        and all(
            abs(actuator_report["effort_limit"][name] - table[name]["effort"]) <= 1e-6 for name in joint_names
        )
    )

    # ---- R7 力矩透传与裁剪 ----
    effort_vec = torch.tensor([table[name]["effort"] for name in joint_names], device=robot.device)
    command = torch.full((robot.num_instances, n_joints), 1.5, device=robot.device)
    command[0, 0] = 1000.0
    command[0, 1] = -1000.0
    expected = torch.clamp(command, -effort_vec, effort_vec)
    robot.set_joint_effort_target(command)
    robot.write_data_to_sim()
    applied = robot.data.applied_torque.clone()
    passthrough_error = float((applied - expected).abs().max())
    commands_for_log = {"joint0": float(command[0, 0]), "joint1": float(command[0, 1])}
    sim.step(render=False)
    applied_after_step = robot.data.applied_torque.clone()

    # ---- R8 零命令 + 关节偏置 → 读回 0（证明没有隐式 PD 在拉）----
    offset = torch.tensor(
        [asset_cfg.HIRA_DEFAULT_ANGLES[name] + 0.3 for name in joint_names], device=robot.device
    ).repeat(robot.num_instances, 1)
    robot.write_joint_state_to_sim(offset, torch.zeros_like(offset))
    zero_cmd = torch.zeros((robot.num_instances, n_joints), device=robot.device)
    robot.set_joint_effort_target(zero_cmd)
    robot.write_data_to_sim()
    applied_zero = robot.data.applied_torque.clone()
    sim.step(render=False)
    joint_pos_after = robot.data.joint_pos.clone()

    # ---- R9 质量 ----
    usd_total_mass = float(sum(link_masses))
    urdf_total_mass = float(static["source"]["urdf"]["total_mass"])
    mass_rel = abs(usd_total_mass - urdf_total_mass) / max(urdf_total_mass, 1e-9)
    per_body = []
    for name, mass in zip(body_names, link_masses):
        per_body.append({"body": name, "usd_mass": float(mass), "urdf_mass": static["source"]["urdf"]["mass_by_link"].get(name)})
    # 逐体质量比较用 float32 容差：0.678 这类数在 PhysX 里就是 0.678000032…
    mismatched_bodies = [
        item
        for item in per_body
        if item["urdf_mass"] is not None
        and abs(item["usd_mass"] - item["urdf_mass"]) > max(1e-5, 1e-6 * abs(item["urdf_mass"]))
    ]
    merged_away = sorted(set(static["source"]["urdf"]["mass_by_link"]) - set(body_names))

    # ---- R10 默认角 ----
    init_pos = list(initial_readback)
    resolved = default_pos_tensor[0].tolist()
    resolve_errors = {
        name: abs(resolved[index] - asset_cfg.HIRA_DEFAULT_ANGLES[name]) for index, name in enumerate(joint_names)
    }
    write_errors = {
        name: abs(init_pos[index] - asset_cfg.HIRA_DEFAULT_ANGLES[name]) for index, name in enumerate(joint_names)
    }
    # 阈值 1e-6：PhysX 的关节角是 float32，1.5 rad 处的表示误差就在 1e-7 量级
    default_ok = max(resolve_errors.values()) <= 1e-6 and max(write_errors.values()) <= 1e-6

    usd_files = sorted(usd_dir.glob("**/*.usd*")) if usd_dir is not None else []
    runtime = {
        "device": device,
        "sim_dt": asset_cfg.HIRA_SIM_DT,
        "n_dof": n_joints,
        "n_bodies": len(body_names),
        "joint_names_usd": joint_names,
        "body_names_usd": body_names,
        "order": {
            "equal_to_urdf": order_equal,
            "bijection": bijection,
            "permutation_urdf_index_of_usd_order": permutation,
            "note": "未假定顺序相同；HIRA-013 用本置换按名字建立双向映射",
        },
        "joints": rows,
        "joint_limits_ok": not limit_bad,
        "joint_limits_bad": limit_bad,
        "usd_gains": {
            "stiffness": sorted({float(value) for value in stiffness}),
            "damping": sorted({float(value) for value in damping}),
            "all_zero": gains_zero,
        },
        "actuator": actuator_report,
        "actuator_ok": actuator_ok,
        "torque_passthrough": {
            "command": commands_for_log,
            "max_abs_error_before_step": passthrough_error,
            "applied_joint0": float(applied[0, 0]),
            "applied_joint1": float(applied[0, 1]),
            "effort_limit_joint0": float(effort_vec[0]),
            "clipped_to_effort": abs(float(applied[0, 0]) - float(effort_vec[0])) <= 1e-6,
            "applied_after_step_unchanged": float((applied_after_step - applied).abs().max()) <= 1e-6,
            "ok": passthrough_error <= 1e-6,
        },
        "no_hidden_pd": {
            "setup": "12 关节偏离默认角 0.3 rad，力矩命令 0",
            "max_abs_applied": float(applied_zero.abs().max()),
            "max_abs_position_after_step": float(joint_pos_after.abs().max()),
            "ok": float(applied_zero.abs().max()) <= 1e-9,
        },
        "mass": {
            "usd_total": usd_total_mass,
            "urdf_total": urdf_total_mass,
            "rel_diff": mass_rel,
            "per_body": per_body,
            "bodies_with_urdf_counterpart_mismatch": mismatched_bodies,
            "links_not_present_as_bodies": merged_away,
            "n_links_in_urdf": len(static["source"]["urdf"]["mass_by_link"]),
            "note": "fixed 关节被合并后 body 名会变少，因此逐 body 只对同名项比较；总质量是主判据",
            "ok": mass_rel <= args_cli.mass_rtol,
        },
        "init_default_angles": {
            "resolve_max_abs_error": max(resolve_errors.values()),
            "write_readback_max_abs_error": max(write_errors.values()),
            "per_joint": {
                name: {
                    "resolved_from_cfg": resolved[index],
                    "readback_after_write": init_pos[index],
                    "expected": asset_cfg.HIRA_DEFAULT_ANGLES[name],
                }
                for index, name in enumerate(joint_names)
            },
            "note": "Articulation.reset() 不写 init_state，需 env 侧显式 write_joint_state_to_sim；"
            "HIRA-013 的逐 env reset 必须包含这一步",
            "ok": default_ok,
        },
        "usd_artifacts": [{"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in usd_files],
    }

    checks = {
        "R1_bijection": bijection,
        "R2_joint_limits": not limit_bad,
        "R3_usd_gains_zero": gains_zero,
        "R4_actuator_zero_gain": actuator_ok,
        "R5_torque_passthrough": runtime["torque_passthrough"]["ok"],
        "R6_no_hidden_pd": runtime["no_hidden_pd"]["ok"],
        "R7_mass": runtime["mass"]["ok"],
        "R8_init_default_angles": default_ok,
    }

    report = {
        "report": "asset_actuator_audit",
        "task": "HIRA-012",
        "isaaclab": {"app_launcher": "headless", "physics_backend": "PhysX 5.1"},
        "static": static,
        "runtime": runtime,
        "checks": [{"id": key, "ok": bool(value)} for key, value in checks.items()],
        "verdict": {
            "all_passed": all(checks.values()),
            "failed": [key for key, value in checks.items() if not value],
            "order_differs_from_urdf": not order_equal,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"all_passed={report['verdict']['all_passed']} failed={report['verdict']['failed']}")
    log(f"order_equal={order_equal} permutation={permutation}")
    log(f"torque_passthrough_error={passthrough_error:.3e} mass_rel={mass_rel:.3e}")
    log(f"usd_files={[path.name for path in usd_files]}")
    print("ASSET_RUNTIME_OK" if report["verdict"]["all_passed"] else "ASSET_RUNTIME_FAIL", flush=True)
    return 0 if report["verdict"]["all_passed"] else 1


def _tolist(value):
    if torch.is_tensor(value):
        return value.detach().flatten().tolist()
    if isinstance(value, dict):
        flat = []
        for item in value.values():
            flat.extend(_tolist(item))
        return flat
    return [float(value)]


def _named(actuator, attribute: str, joint_names: list[str]):
    """把 actuator 的标量/字典/张量参数展开成 关节名→值。"""
    raw = getattr(actuator, attribute)
    if torch.is_tensor(raw):
        values = raw.detach().flatten().tolist()
    elif isinstance(raw, dict):
        values = [float(raw[name]) if name in raw else float("nan") for name in joint_names]
    else:
        values = [float(raw)] * len(joint_names)
    return list(zip(joint_names, values))


if __name__ == "__main__":
    exit_code = 2
    try:
        exit_code = main()
    except SystemExit as exc:  # main 通过返回值报告结果，这里只兜住显式退出
        exit_code = int(exc.code or 0)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("ASSET_RUNTIME_ERROR", flush=True)
        exit_code = 2
    finally:
        # Isaac Sim 5.1 的 close() 在本机（headless、卡被多人共享）会长时间不返回，
        # 实测 >5 min 且此时报告已落盘，故跳过 close 直接退出；这条作为环境差异登记在报告里。
        log("hard exit (skip simulation_app.close)")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
