"""HIRA-019：验证新执行器的 200 Hz 真实力矩与能量计量。

契约是 D-ENERGY-001：

    J = Σ_k Σ_j max(τ_j[k] · q̇_j[k], 0) · dt        dt = 0.005 s

四个必须自证、缺一不可的东西：

**A. 力矩来源**（E01–E04、E12）
   用的是 ``robot.data.applied_torque``。它不是"我们想施加的力矩"，也不是
   ``computed_torque``：Isaac Lab 的 ``write_data_to_sim()`` 先跑执行器模型得到
   ``applied_effort``，把同一份值同时写进 ``data.applied_torque`` 和
   ``_joint_effort_target_sim``，再交给 ``root_physx_view.set_dof_actuation_forces``。
   本脚本在运行时把这几行源码抓出来（行号 + 文件散列）作为可核查的出处，
   并用"三者逐子步对照 + 强制饱和工况下的能量倍数差"证明它们不能混用。

**B. 相位**（E05、E06）
   力矩在 ``sim.step()`` 之前写入，所以步后读到的是**刚执行完的子步**的力矩；
   速度取 ``robot.update(dt)`` 之后的值，即该子步**结束时刻**的速度——与 v2
   记录器 ``sim/recorder.py`` 的顺序逐字一致。脚本同时按 pre/post 两种口径
   复算，把"选错相位会差多少"直接量出来。

**C. 只读**（E17、E18）
   记录器源码里不允许出现任何 ``set_*`` / ``write_*`` / ``apply_*`` / ``sim.step``
   调用（AST 扫描）；并且开/关记录器的整条轨迹跨进程逐位一致。

**D. 复算与边界**（E07–E11、E13–E16）
   用三套独立实现（在线 torch float64、numpy ``integrate_series``、纯 Python
   ``EnergyMeter``）对照；再测零输入、已知单关节正功、反向力矩（负功必须被丢弃）、
   外力冲击（不算关节正功）、强制饱和、warmup 排除、窗口提前结束、非整策略步结束。

分阶段运行（一个进程只能有一个 InteractiveScene）：

    python tests/check_energy_meter.py --phase exec --device cuda:0
    python tests/check_energy_meter.py --phase readonly --recorder on  --device cuda:0
    python tests/check_energy_meter.py --phase readonly --recorder off --device cuda:0
    python tests/check_energy_meter.py --phase merge
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
ADAPTER = REPO / "CaMRA" / "locomotion" / "adapters"
CAL_SRC = REPO / "CaMRA" / "calibration" / "src"
for path in (str(ADAPTER), str(CAL_SRC), str(EXP / "tests")):
    if path not in sys.path:
        sys.path.insert(0, path)

parser = argparse.ArgumentParser()
parser.add_argument("--phase", choices=("exec", "readonly", "merge"), default="exec")
parser.add_argument("--recorder", choices=("on", "off"), default="on")
parser.add_argument("--urdf", default=str(REPO / "HiRA/legged_gym_go2/resources/robots/go2/urdf/go2.urdf"))
parser.add_argument("--usd-dir", default="/pub/data/caohy/logs/camra_hira/assets/usd_cache")
parser.add_argument("--policy", default=str(EXP / "exported/policy.pt"))
parser.add_argument("--lab-root", default="/pub/data/caohy/IsaacLab")
parser.add_argument("--out-dir", default=str(EXP / "reports"))
# clean 的摩擦按源（URDF/isaacgym）1.0；导入的 USD 默认 0.5（HIRA-017 实测）
parser.add_argument("--friction", type=float, default=1.0)
parser.add_argument("--window", type=float, default=10.0)
# 注意：--device 由 AppLauncher.add_app_launcher_args() 提供，这里不能重复声明

_PRE = parser.parse_known_args()[0]
if _PRE.phase == "merge":
    args_cli = parser.parse_args()
    app_launcher = None
    simulation_app = None
else:
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

import torch  # noqa: E402

from camra_calibration.energy import (  # noqa: E402
    DEFAULT_DT,
    EnergyMeter,
    integrate_series,
    reference_naive_sum_then_clip,
)

T0 = time.time()
CHECKS: list[dict] = []


def log(text: str) -> None:
    print(f"[energy] {time.time() - T0:7.1f}s {text}", flush=True)


def add(cid: str, name: str, ok: bool, detail: str, **extra) -> dict:
    record = {"id": cid, "name": name, "ok": bool(ok), "detail": detail}
    record.update(extra)
    CHECKS.append(record)
    print(f"[{'ok  ' if ok else 'FAIL'}] {cid} {name} :: {detail}", flush=True)
    return record


# --------------------------------------------------------------------------- #
# 工具
def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def rel_err(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_lines(path: Path, needles: dict[str, str]) -> dict:
    """在源码里定位关键行；找不到就是 0 条，会直接被判定为证据不足。"""
    if not path.is_file():
        return {"path": str(path), "sha256": None, "missing_file": True, "hits": {}}
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    hits = {}
    for key, needle in needles.items():
        found = [index + 1 for index, line in enumerate(lines) if needle in line]
        hits[key] = {
            "needle": needle,
            "lines": found,
            "text": [lines[index - 1].strip() for index in found],
        }
    return {"path": str(path), "sha256": sha256_file(path), "hits": hits}


def static_readonly_scan(path: Path) -> dict:
    """AST 扫描：记录器里不允许有任何写仿真/写资产的调用。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = ""
        if isinstance(func, ast.Attribute):
            name = func.attr
        elif isinstance(func, ast.Name):
            name = func.id
        if name.startswith(("set_", "write_", "apply_", "reset_")) or name in (
            "step",
            "step_known_torque",
            "advance",
        ):
            forbidden.append({"call": name, "line": getattr(node, "lineno", None)})
    return {"file": str(path), "n_forbidden_calls": len(forbidden), "forbidden": forbidden}


def build_runner():
    from hira_go2.env_runner import HiraLabRunner, TimingConfig

    timing = TimingConfig(sim_dt=DEFAULT_DT, window_s=args_cli.window)
    runner = HiraLabRunner(
        args_cli.urdf,
        args_cli.usd_dir,
        num_envs=1,
        device=args_cli.device,
        policy_path=args_cli.policy,
        timing=timing,
    )
    view = runner.robot.root_physx_view
    material = view.get_material_properties().clone()
    friction_as_imported = float(material[0, 0, 0])
    material[:, :, 0] = args_cli.friction
    material[:, :, 1] = args_cli.friction
    view.set_material_properties(material, torch.tensor([0], dtype=torch.int64))
    log(f"friction {friction_as_imported:.2f} → {float(view.get_material_properties()[0, 0, 0]):.2f}")
    return runner, friction_as_imported


def new_recorder(runner, capacity: int, label: str, expected: int | None = None):
    from hira_go2.energy_recorder import HiRAEnergyRecorder

    return HiRAEnergyRecorder(
        runner.order,
        capacity_steps=capacity,
        dt=runner.timing.sim_dt,
        num_envs=runner.num_envs,
        n_joints=12,
        device=runner.device,
        label=label,
        expected_steps=expected,
    )


class ConstantPolicy(torch.nn.Module):
    """固定动作策略：用来构造"必然饱和"的工况，不参与任何正常评估。"""

    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.full((obs.shape[0], 12), self.value, dtype=torch.float32, device=obs.device)


def per_env(result: dict, env: int = 0) -> dict:
    out = {
        "total_j": float(result["total_j"][env]),
        "energy_step_j": result["energy_step_j"][:, env],
        "per_joint_j": result["per_joint_j"][env],
    }
    for key in (
        "tau_applied_source",
        "tau_written_source",
        "tau_request_source",
        "tau_computed_source",
        "qdot_pre_source",
        "qdot_post_source",
        "joint_acc_source",
    ):
        # 一律升到 float64：留档本身是 float32，若在 float32 里做"闭式值"对照，
        # 会得到 6e-8 级假误差（第一版 E10 就是这么假失败的）
        out[key] = result[key][:, env, :].astype(np.float64)
    return out


# --------------------------------------------------------------------------- #
def phase_exec() -> int:
    from hira_go2.env_runner import SOURCE_ORDER

    runner, friction_as_imported = build_runner()
    log("runner ready")
    dt = runner.timing.sim_dt
    effort = runner.torque_adapter.effort_limit_source.reshape(-1).detach().cpu().numpy()
    experiments: dict = {}

    # ---------------- E01 力矩来源的出处 ----------------
    articulation = Path(args_cli.lab_root) / "source/isaaclab/isaaclab/assets/articulation/articulation.py"
    data_file = Path(args_cli.lab_root) / "source/isaaclab/isaaclab/assets/articulation/articulation_data.py"
    provenance = {
        "articulation": find_lines(
            articulation,
            {
                "applied_torque_assignment": "self._data.applied_torque[:, actuator.joint_indices] = actuator.applied_effort",
                "effort_from_actuator": "self._joint_effort_target_sim[:, actuator.joint_indices] = control_action.joint_efforts",
                "write_to_physx": "set_dof_actuation_forces(self._joint_effort_target_sim",
                "update_buffers": "def update(self, dt: float):",
            },
        ),
        "articulation_data": find_lines(
            data_file,
            {
                "computed_torque_doc": "Joint torques computed from the actuator model (before clipping)",
                "applied_torque_doc": "Joint torques applied from the actuator model (after clipping)",
            },
        ),
        "actuator_config": str(EXP / "config/hira_source_contract_v1.json"),
    }
    hits = provenance["articulation"].get("hits", {})
    ok_prov = bool(
        hits.get("applied_torque_assignment", {}).get("lines")
        and hits.get("effort_from_actuator", {}).get("lines")
        and hits.get("write_to_physx", {}).get("lines")
    )
    add(
        "E01",
        "力矩来源：applied_torque 与交给 PhysX 的是同一份 applied_effort",
        ok_prov,
        "articulation.py "
        f"L{hits.get('applied_torque_assignment', {}).get('lines')} 赋值 applied_torque、"
        f"L{hits.get('effort_from_actuator', {}).get('lines')} 该值进入 _joint_effort_target_sim、"
        f"L{hits.get('write_to_physx', {}).get('lines')} 写进 PhysX（文件 sha {str(provenance['articulation'].get('sha256'))[:12]}）",
    )

    # ---------------- E02–E07 策略闭环窗口 ----------------
    n_steps = int(round(args_cli.window / dt))
    rec = new_recorder(runner, n_steps, "policy_window", expected=n_steps)
    runner.reset_all()
    runner.set_command(torch.tensor([[0.5, 0.0, 0.0]], dtype=runner.dtype))
    runner.recorder = rec
    rec.begin_window()
    for _ in range(n_steps // runner.timing.decimation):
        runner.policy_step()
    runner.recorder = None
    window = rec.finish()
    env = per_env(window)
    applied = env["tau_applied_source"].astype(np.float64)
    written = env["tau_written_source"].astype(np.float64)
    request = env["tau_request_source"].astype(np.float64)
    computed = env["tau_computed_source"].astype(np.float64)
    q_pre = env["qdot_pre_source"].astype(np.float64)
    q_post = env["qdot_post_source"].astype(np.float64)

    d_aw = float(np.abs(applied - written).max())
    add(
        "E02",
        "实际施加 == 我们写入的限幅力矩（执行器不再改一笔）",
        d_aw <= 1e-4,
        f"{window['steps']} 子步，|applied−written| max={d_aw:.3e}（阈值 1e-4，预登记 torque atol）",
        max_abs_diff=d_aw,
    )
    d_cw = float(np.abs(computed - written).max())
    d_ac = float(np.abs(applied - computed).max())
    add(
        "E03",
        "computed_torque == applied_torque（无隐藏 PD/增益二次加工）",
        d_cw <= 1e-4 and d_ac <= 1e-4,
        f"|computed−written| max={d_cw:.3e}，|applied−computed| max={d_ac:.3e}",
        max_abs_diff=max(d_cw, d_ac),
    )
    over = (np.abs(applied) - effort.reshape(1, -1)).max()
    sat_steps = int(window["saturated_substeps"])
    add(
        "E04",
        "实际施加从不越限幅",
        over <= 1e-6,
        f"max(|applied|−effort)={over:.3e}；窗口内饱和关节-子步数={sat_steps}",
        over_limit=float(over),
    )

    online = env["total_j"]
    offline_post = integrate_series(applied, q_post, dt)
    meter = EnergyMeter(dt=dt, n_joints=12, expected_steps=window["steps"], label="offline-replay")
    meter.start()
    for index in range(applied.shape[0]):
        meter.add_step(applied[index].tolist(), q_post[index].tolist())
    meter_total = meter.total_j
    d_np = rel_err(online, offline_post)
    d_meter = rel_err(online, meter_total)
    add(
        "E05",
        "在线（torch float64）与两套离线实现一致",
        d_np <= 1e-9 and d_meter <= 1e-9,
        f"在线 J={online:.9f}；numpy 复算 {offline_post:.9f}（rel {d_np:.3e}）；"
        f"EnergyMeter 重放 {meter_total:.9f}（rel {d_meter:.3e}）；阈值 rel ≤1e-9（预登记 T-10）",
        online_J=online,
        numpy_J=offline_post,
        meter_J=meter_total,
        rel_vs_numpy=d_np,
        rel_vs_meter=d_meter,
    )
    offline_pre = integrate_series(applied, q_pre, dt)
    d_phase = rel_err(offline_pre, online)
    dq = float(np.abs(q_post - q_pre).max())
    add(
        "E06",
        "相位选择不是可有可无：pre 与 post 口径确实不同",
        (d_phase >= 1e-3) and (d_np <= 1e-9),
        f"同一段数据按子步起始速度算得 J={offline_pre:.6f}，按子步结束速度算得 {online:.6f}，"
        f"相对差 {d_phase * 100:.2f}%（单步 |Δq̇| max={dq:.3e} rad/s）；"
        "契约取后者，与 v2 记录器一致",
        rel_pre_vs_post=d_phase,
        J_pre=offline_pre,
    )
    naive = reference_naive_sum_then_clip(applied, q_post, dt)
    d_naive = rel_err(naive, online)
    add(
        "E07",
        "逐关节截断与先求和再截断确实不同",
        d_naive >= 1e-4,
        f"错误口径（先求和再截断）J={naive:.6f} vs 契约 {online:.6f}，相对差 {d_naive * 100:.2f}%",
        rel_naive=d_naive,
    )
    problems = rec.problems(window)
    add(
        "E08",
        "守恒自检：逐步/逐关节求和 == 总量，全部非负有限",
        not problems,
        "无异常" if not problems else "；".join(problems),
        problems=problems,
    )
    d_req = float(np.abs(request - applied).max())
    experiments["policy_window"] = {
        "command": [0.5, 0.0, 0.0],
        "steps": window["steps"],
        "elapsed_s": window["elapsed_s"],
        "J_online": online,
        "J_pre_phase": offline_pre,
        "J_naive_sum_then_clip": naive,
        "per_joint_J": env["per_joint_j"],
        "saturated_substeps": sat_steps,
        "max_abs_request_vs_applied": d_req,
        "max_abs_tau": float(np.abs(applied).max()),
        "max_abs_qdot": float(np.abs(q_post).max()),
        "root_height_min": float(window["root_pos_w"][:, 0, 2].min()),
        "root_height_max": float(window["root_pos_w"][:, 0, 2].max()),
    }
    log(f"policy window: J={online:.3f} J, sat={sat_steps}, d_aw={d_aw:.2e}")

    # ---------------- E09 零输入 ----------------
    zero = torch.zeros((1, 12), dtype=runner.dtype)
    runner.reset_all()
    rec_zero = new_recorder(runner, 200, "zero_torque", expected=200)
    runner.recorder = rec_zero
    rec_zero.begin_window()
    for _ in range(200):
        runner.step_known_torque(zero)
    runner.recorder = None
    zero_res = rec_zero.finish()
    zero_j = float(zero_res["total_j"][0])
    add(
        "E09",
        "零力矩 → 能量恒为 0（符号与截断的基线）",
        zero_j == 0.0,
        f"200 子步零力矩，J={zero_j!r}（要求逐位 0）；机器人高度 "
        f"{float(zero_res['root_pos_w'][:, 0, 2].min()):.3f}→{float(zero_res['root_pos_w'][-1, 0, 2]):.3f} m",
        J=zero_j,
    )

    # ---------------- E10/E11 已知单关节力矩：正功与负功 ----------------
    calf = SOURCE_ORDER.index("FL_calf_joint")
    runner.reset_all()
    rec_known = new_recorder(runner, 200, "known_torque", expected=200)
    runner.recorder = rec_known
    rec_known.begin_window()
    positive = torch.zeros((1, 12), dtype=runner.dtype)
    positive[0, calf] = 3.0
    for _ in range(100):
        runner.step_known_torque(positive)
    negative = torch.zeros((1, 12), dtype=runner.dtype)
    negative[0, calf] = -3.0
    for _ in range(100):
        runner.step_known_torque(negative)
    runner.recorder = None
    known = rec_known.finish()
    known_env = per_env(known)
    qdot_calf = known_env["qdot_post_source"][:, calf]
    phase_a = float(known_env["energy_step_j"][:100].sum())
    expected_a = float(np.clip(3.0 * qdot_calf[:100], 0.0, None).sum() * dt)
    other_joints = np.delete(known_env["per_joint_j"], calf)
    add(
        "E10",
        "已知单关节 +3 N·m：能量等于闭式值，其余 11 关节贡献恒 0",
        rel_err(phase_a, expected_a) <= 1e-9 and float(np.abs(other_joints).max()) == 0.0,
        f"前 100 子步 J={phase_a:.9f}，闭式值 Σmax(3·q̇,0)dt={expected_a:.9f}"
        f"（rel {rel_err(phase_a, expected_a):.2e}）；非驱动关节 |J| max={float(np.abs(other_joints).max()):.1e}",
        J_phase_a=float(phase_a),
        J_expected=expected_a,
    )
    phase_b_mask = qdot_calf[100:] > 0
    phase_b_energy = known_env["energy_step_j"][100:]
    n_opposing = int(phase_b_mask.sum())
    max_opposing = float(phase_b_energy[phase_b_mask].max()) if n_opposing else 0.0
    add(
        "E11",
        "反向力矩在关节仍正向运动时产生负功 → 被逐关节截断丢弃",
        n_opposing >= 10 and max_opposing == 0.0,
        f"后 100 子步中 q̇_calf>0 的有 {n_opposing} 步，这些步的单步能量 max={max_opposing!r}"
        f"（要求逐位 0；若用 |τ·q̇| 或先求和再截断这里会非零）",
        n_opposing_steps=n_opposing,
    )
    experiments["known_torque"] = {
        "joint": "FL_calf_joint",
        "torque_Nm": [3.0, -3.0],
        "steps_per_phase": 100,
        "J_phase_a": float(phase_a),
        "J_phase_b": float(phase_b_energy.sum()),
        "qdot_max_phase_a": float(qdot_calf[:100].max()),
        "qdot_min_phase_b": float(qdot_calf[100:].min()),
        "n_opposing_steps": n_opposing,
    }

    # ---------------- E12 外力冲击不算关节正功 ----------------
    runner.reset_all()
    rec_push = new_recorder(runner, 100, "push", expected=100)
    runner.recorder = rec_push
    rec_push.begin_window()
    runner.step_known_torque(zero)
    before = runner.robot.data.root_lin_vel_w.clone()
    dv = torch.tensor([[1.0, 0.0, 0.0]], device=runner.device, dtype=runner.dtype)
    runner.robot.write_root_velocity_to_sim(
        torch.cat([before + dv, runner.robot.data.root_ang_vel_w], dim=-1)
    )
    runner.robot.update(dt)
    for _ in range(99):
        runner.step_known_torque(zero)
    runner.recorder = None
    push_res = rec_push.finish()
    add(
        "E12",
        "外部速度冲击（1 m/s）下零关节力矩 → 能量仍为 0",
        float(push_res["total_j"][0]) == 0.0,
        f"注入 Δv=(1,0,0) m/s 后 J={float(push_res['total_j'][0])!r}；"
        "推扰做功不计入关节正功（约束：不把机械功说成电池能耗）",
    )

    # ---------------- E13 强制饱和：请求力矩 ≠ 实际施加 ----------------
    runner.reset_all()
    rec_sat = new_recorder(runner, 200, "forced_saturation", expected=200)
    runner.recorder = rec_sat
    rec_sat.begin_window()
    policy_backup = runner.policy
    runner.policy = ConstantPolicy(100.0)
    for _ in range(50):
        runner.policy_step()
    runner.policy = policy_backup
    runner.recorder = None
    sat = rec_sat.finish()
    sat_env = per_env(sat)
    req_sat = sat_env["tau_request_source"].astype(np.float64)
    app_sat = sat_env["tau_applied_source"].astype(np.float64)
    qp_sat = sat_env["qdot_post_source"].astype(np.float64)
    saturated_mask = np.abs(req_sat) > effort.reshape(1, -1) + 1e-9
    n_sat_entries = int(saturated_mask.sum())
    at_limit = float(np.abs(np.abs(app_sat[saturated_mask]) - np.broadcast_to(effort, req_sat.shape)[saturated_mask]).max()) if n_sat_entries else 0.0
    j_req = integrate_series(req_sat, qp_sat, dt)
    j_app = integrate_series(app_sat, qp_sat, dt)
    ratio = (j_req / j_app) if j_app > 0 else float("inf")
    add(
        "E13",
        "饱和时请求力矩与实际施加差一个量级：不能拿请求值当施加值",
        n_sat_entries > 0 and at_limit <= 1e-6 and ratio >= 5.0,
        f"动作=+100 的 200 子步：{n_sat_entries} 个关节-子步越限，越限处 |applied| 恰为 effort"
        f"（偏差 {at_limit:.2e}）；J(请求)={j_req:.1f} J vs J(施加)={j_app:.2f} J，倍数 {ratio:.1f}×",
        ratio=ratio,
        n_saturated_entries=n_sat_entries,
    )
    experiments["forced_saturation"] = {
        "J_request": float(j_req),
        "J_applied": float(j_app),
        "ratio": float(ratio),
        "n_saturated_entries": n_sat_entries,
        "max_abs_request": float(np.abs(req_sat).max()),
        "max_abs_applied": float(np.abs(app_sat).max()),
    }

    # ---------------- E14 warmup 排除 ----------------
    runner.reset_all()
    rec_warm = new_recorder(runner, 40, "warmup_exclusion", expected=40)
    runner.recorder = rec_warm
    rec_warm.begin_warmup()
    policy_backup = runner.policy
    runner.policy = ConstantPolicy(0.0)
    runner.warmup(200)
    rec_warm.begin_window()
    for _ in range(10):
        runner.policy_step()
    runner.policy = policy_backup
    runner.recorder = None
    warm = rec_warm.finish()
    add(
        "E14",
        "warmup 的 200 个物理步一步都没进窗口",
        warm["warmup_steps"] == 200 and warm["steps"] == 40 and not warm["partial"],
        f"warmup_steps={warm['warmup_steps']}，窗口 steps={warm['steps']}（10 个策略步 = 40 子步），"
        f"窗口 J={float(warm['total_j'][0]):.6f}",
        warmup_steps=warm["warmup_steps"],
        window_steps=warm["steps"],
    )

    # ---------------- E15 窗口提前结束 / 非整策略步结束 ----------------
    runner.reset_all()
    rec_part = new_recorder(runner, 2000, "partial", expected=2000)
    runner.recorder = rec_part
    rec_part.begin_window()
    policy_backup = runner.policy
    runner.policy = ConstantPolicy(0.0)
    for _ in range(336):          # 336 × 4 = 1344
        runner.policy_step()
    for _ in range(3):            # 再 3 个子步：窗口停在策略步中间
        runner.substep()
    runner.policy = policy_backup
    runner.recorder = None
    part = rec_part.finish()
    part_problems = rec_part.problems(part)
    add(
        "E15",
        "提前结束/非整策略步结束时，计量仍然自洽",
        part["steps"] == 1347 and part["partial"] and not part_problems,
        f"steps={part['steps']}（1344+3，不是 4 的倍数）、partial={part['partial']}、"
        f"J={float(part['total_j'][0]):.6f}；守恒问题：{part_problems or '无'}",
        steps=part["steps"],
    )

    # ---------------- E16 记录器的静态只读性 ----------------
    scan = static_readonly_scan(ADAPTER / "hira_go2" / "energy_recorder.py")
    add(
        "E16",
        "记录器源码内没有任何写仿真/写资产的调用（AST 扫描）",
        scan["n_forbidden_calls"] == 0,
        f"{Path(scan['file']).name}：禁用调用 {scan['n_forbidden_calls']} 处"
        + (f" → {scan['forbidden'][:5]}" if scan["n_forbidden_calls"] else "（set_*/write_*/apply_*/step 全部未出现）"),
        forbidden=scan["forbidden"][:10],
    )

    report = {
        "report": "energy_meter_validation",
        "task": "HIRA-019",
        "phase": "exec",
        "contract": {
            "metric_id": "D-ENERGY-001",
            "formula": "J = Σ_k Σ_j max(τ_j[k]·q̇_j[k], 0)·dt",
            "dt_s": dt,
            "unit": "J",
            "joints": 12,
            "clip": "逐关节截断后求和（不先求和再截断）",
            "phase": "τ 属于刚执行完的子步；q̇ 取该子步结束时刻（robot.update 之后）",
        },
        "provenance": provenance,
        "environment": {
            "friction": args_cli.friction,
            "friction_as_imported": friction_as_imported,
            "policy": str(args_cli.policy),
            "policy_sha256": sha256_file(Path(args_cli.policy)),
            "urdf": str(args_cli.urdf),
            "urdf_sha256": sha256_file(Path(args_cli.urdf)),
            "torch": torch.__version__,
            "lab_root": args_cli.lab_root,
        },
        "effort_limits": dict(zip(SOURCE_ORDER, effort.tolist())),
        "checks": CHECKS,
        "experiments": experiments,
    }
    out = Path(args_cli.out_dir) / "energy_meter_validation.exec.json"
    out.write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    failed = [item["id"] for item in CHECKS if not item["ok"]]
    print(f"EXEC_OK failed={failed}" if not failed else f"EXEC_FAIL failed={failed}", flush=True)
    return 0 if not failed else 1


# --------------------------------------------------------------------------- #
def phase_readonly() -> int:
    """同一段固定动作，开/关记录器各跑一次；关掉时也要走完全相同的物理路径。"""
    from hira_go2.env_runner import SOURCE_ORDER

    runner, friction_as_imported = build_runner()
    n_policy_steps = 250
    n_steps = n_policy_steps * runner.timing.decimation
    rec = None
    if args_cli.recorder == "on":
        rec = new_recorder(runner, n_steps, "readonly_on", expected=n_steps)
        runner.recorder = rec
        rec.begin_window()

    runner.reset_all()
    runner.set_command(torch.tensor([[0.5, 0.0, 0.0]], dtype=runner.dtype))
    digest = hashlib.sha256()
    frames = []
    for _ in range(n_policy_steps):
        runner.policy_step()
        frame = torch.cat(
            [
                runner.robot.data.root_pos_w[0],
                runner.robot.data.root_quat_w[0],
                runner.robot.data.joint_pos[0],
                runner.robot.data.joint_vel[0],
                runner.robot.data.applied_torque[0],
            ]
        ).detach().cpu().numpy().astype(np.float32)
        digest.update(frame.tobytes())
        frames.append(frame)
    runner.recorder = None
    energy = None if rec is None else float(rec.finish()["total_j"][0])
    array = np.stack(frames)
    out_npy = Path(args_cli.out_dir) / f"energy_readonly_{args_cli.recorder}.npy"
    np.save(out_npy, array)
    payload = {
        "report": "energy_readonly",
        "task": "HIRA-019",
        "recorder": args_cli.recorder,
        "frames": int(array.shape[0]),
        "frame_width": int(array.shape[1]),
        "fingerprint_sha256": digest.hexdigest(),
        "array_sha256": sha256_file(out_npy),
        "array_file": str(out_npy),
        "J_online": energy,
        "final_root_pos": array[-1, 0:3].tolist(),
        "final_joint_pos": array[-1, 7:19].tolist(),
        "friction": args_cli.friction,
        "friction_as_imported": friction_as_imported,
    }
    out = Path(args_cli.out_dir) / f"energy_readonly_{args_cli.recorder}.json"
    out.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    log(f"readonly[{args_cli.recorder}] fingerprint={payload['fingerprint_sha256'][:16]} J={energy}")
    print(f"READONLY_OK recorder={args_cli.recorder}", flush=True)
    return 0


def phase_merge() -> int:
    out_dir = Path(args_cli.out_dir)
    exec_path = out_dir / "energy_meter_validation.exec.json"
    on_path = out_dir / "energy_readonly_on.json"
    off_path = out_dir / "energy_readonly_off.json"
    for path in (exec_path, on_path, off_path):
        if not path.is_file():
            raise SystemExit(f"缺少输入：{path}")
    exec_report = json.loads(exec_path.read_text(encoding="utf-8"))
    on = json.loads(on_path.read_text(encoding="utf-8"))
    off = json.loads(off_path.read_text(encoding="utf-8"))

    same = on["fingerprint_sha256"] == off["fingerprint_sha256"]
    arr_on = np.load(on["array_file"])
    arr_off = np.load(off["array_file"])
    max_diff = float(np.abs(arr_on.astype(np.float64) - arr_off.astype(np.float64)).max())
    check = {
        "id": "E17",
        "name": "记录器开/关的整条轨迹逐位一致（只读，不改动力学）",
        "ok": bool(same and max_diff == 0.0 and arr_on.shape == arr_off.shape),
        "detail": f"{on['frames']} 个策略步 × {on['frame_width']} 个量（root 位姿/关节 q,q̇/施加力矩）："
        f"散列 {'一致' if same else '不一致'}，逐元素 max|Δ|={max_diff!r}",
        "fingerprint_on": on["fingerprint_sha256"],
        "fingerprint_off": off["fingerprint_sha256"],
        "max_abs_diff": max_diff,
    }
    checks = list(exec_report["checks"]) + [check]
    failed = [item["id"] for item in checks if not item["ok"]]
    report = {
        "report": "hira_energy_meter_validation",
        "task": "HIRA-019",
        "contract": exec_report["contract"],
        "provenance": exec_report["provenance"],
        "environment": exec_report["environment"],
        "effort_limits": exec_report["effort_limits"],
        "experiments": exec_report["experiments"],
        "readonly": {
            "policy_steps": on["frames"],
            "J_online": on["J_online"],
            "fingerprint_on": on["fingerprint_sha256"],
            "fingerprint_off": off["fingerprint_sha256"],
            "max_abs_diff": max_diff,
        },
        "checks": checks,
        "verdict": {
            "n_checks": len(checks),
            "n_passed": len(checks) - len(failed),
            "failed": failed,
            "all_passed": not failed,
        },
        "notes": [
            "J 是关节机械正功（D-ENERGY-001），不是电池能耗；外部推扰功不计入。",
            "τ 取 applied_torque，即执行器模型的 applied_effort，也是交给 PhysX 的那份值；请求力矩与 computed_torque 仅作对照。",
            "q̇ 取子步结束时刻（robot.update 之后），与 v2 记录器相位一致；pre 口径的差异已量化（E06）。",
        ],
    }
    out = out_dir / "hira_energy_meter_validation.json"
    out.write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for item in checks:
        print(f"[{'ok  ' if item['ok'] else 'FAIL'}] {item['id']} {item['name']}", flush=True)
    print(f"VERDICT all_passed={not failed} failed={failed} -> {out}", flush=True)
    return 0 if not failed else 1


# --------------------------------------------------------------------------- #
def main() -> int:
    if args_cli.phase == "exec":
        return phase_exec()
    if args_cli.phase == "readonly":
        return phase_readonly()
    return phase_merge()


if __name__ == "__main__":
    code = 2
    try:
        code = main()
    except SystemExit as exc:
        code = int(exc.code or 0)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("ENERGY_METER_ERROR", flush=True)
        code = 2
    finally:
        # Isaac Sim 5.1 在本机 headless 下 close() 会卡 >5 min；报告已落盘，直接退出。
        log("hard exit (skip simulation_app.close)")
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            import os

            os._exit(code)
        except Exception:  # noqa: BLE001
            raise
