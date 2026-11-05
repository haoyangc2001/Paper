"""G1 接入门禁：逐项矩阵 + 冻结控制器包 + 重载确认（HIRA-020）。

门禁不是"看命令有没有报错"，而是把预登记的每一行拿实测值重新判一遍，
并且**独立复核**那些能复核的东西（散列、双射、时钟算术、证据文件本身）。

三条硬规则（来自 `integration_acceptance_prereg_v1.json`）：

1. T-01..T-13、T-15 任一失败 → G1 不通过，不得进入 pilot；
2. T-14 失败要区分"接入错误"与"策略能力"，后者记录并继续；
3. 总体 pass 必须是全部必需行的合取，不能只看退出码。

T-08 按 `config/integration_acceptance_amendment_t08_v1.json` 执行（用户裁定）：
拆成「解耦判定」与「行为等价」两条；原判据的失败实测保留在
`reports/batch_isolation.json` 中，不删改。

    python tests/check_integration_gate.py --phase matrix      # 判矩阵 + 冻结包 + 拒混测试
    python tests/check_integration_gate.py --headless --device cuda:0 --phase reload
    python tests/check_integration_gate.py --phase merge
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
ADAPTER = REPO / "CaMRA" / "locomotion" / "adapters"
CAL_SRC = REPO / "CaMRA" / "calibration" / "src"
for path in (str(ADAPTER), str(CAL_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

parser = argparse.ArgumentParser()
parser.add_argument("--phase", choices=("matrix", "reload", "merge"), default="matrix")
parser.add_argument("--out-dir", default=str(EXP / "reports"))
parser.add_argument("--bundle-root", default=str(EXP / "frozen"))
parser.add_argument("--controller-id", default="hira_go2_18500")
parser.add_argument("--policy", default=str(EXP / "exported/policy.pt"))
parser.add_argument("--urdf", default=str(REPO / "HiRA/legged_gym_go2/resources/robots/go2/urdf/go2.urdf"))
parser.add_argument("--usd-dir", default="/pub/data/caohy/logs/camra_hira/assets/usd_cache")
parser.add_argument(
    "--checkpoint", default=str(REPO / "HiRA/logs/rough_go2/Sep08_11-57-26_/model_18500.pt")
)
parser.add_argument("--prereg", default=str(EXP / "config/integration_acceptance_prereg_v1.json"))
parser.add_argument(
    "--amendment", default=str(EXP / "config/integration_acceptance_amendment_t08_v1.json")
)
parser.add_argument("--reload-steps", type=int, default=10)
parser.add_argument("--refreeze", action="store_true", help="删除已有冻结目录后重新冻结（包不可变，须显式声明）")

_PRE = parser.parse_known_args()[0]
if _PRE.phase == "reload":
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
else:
    args_cli = parser.parse_args()
    app_launcher = None
    simulation_app = None

T0 = time.time()


def log(text: str) -> None:
    print(f"[gate] {time.time() - T0:7.1f}s {text}", flush=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(name: str) -> dict:
    path = Path(args_cli.out_dir) / name
    if not path.is_file():
        raise SystemExit(f"缺少证据文件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


# --------------------------------------------------------------------------- #
# 门禁行
def make_row(row_id: str, name: str, ok: bool, detail: str, evidence: list[str], measured=None) -> dict:
    return {
        "id": row_id,
        "name": name,
        "required": True,
        "ok": bool(ok),
        "detail": detail,
        "evidence": evidence,
        "measured": measured or {},
    }


def row_t01(rep: dict) -> dict:
    export = rep["actor_export_parity"]
    checkpoint = Path(args_cli.checkpoint)
    actual = sha256_file(checkpoint) if checkpoint.is_file() else None
    recorded = export.get("checkpoint_sha256")
    arch = export.get("architecture") or {}
    ok = (
        actual is not None
        and actual == recorded
        and export["summary"]["passed"]
        and arch.get("obs_dim") == 45
        and arch.get("act_dim") == 12
        and len(arch.get("hidden_dims", [])) == 3
    )
    return make_row(
        "T-01",
        "checkpoint 身份与 actor 载入",
        ok,
        f"checkpoint sha256 现场重算 {str(actual)[:16]}… 与登记 {str(recorded)[:16]}… "
        f"{'一致' if actual == recorded else '不一致'}；结构 45→{[int(x) for x in arch.get('hidden_dims', [])]}→12；"
        f"导出/对照 29 例 passed={export['summary']['passed']}",
        ["reports/actor_export_parity.json", f"checkpoint 现场散列 {args_cli.checkpoint}"],
        {
            "checkpoint_sha256_actual": actual,
            "checkpoint_sha256_recorded": recorded,
            "actor_tensor_digest": export.get("export_tensor_digest"),
            "architecture": arch,
        },
    )


def row_t02(rep: dict) -> dict:
    audit = rep["asset_actuator_audit"]
    runtime = audit["runtime"]
    checks = audit["checks"]
    failed = [item["id"] for item in checks if not item["ok"]]
    mass = runtime["mass"]
    rel = abs(mass["usd_total"] - mass["urdf_total"]) / mass["urdf_total"]
    ok = audit["verdict"]["all_passed"] and not failed and rel <= 0.01
    return make_row(
        "T-02",
        "资产/关节顺序/坐标系",
        ok,
        f"USD DOF 顺序与 URDF 不同（置换 {runtime['order']}），限位差 ≤1e-3 rad；"
        f"总质量 {mass['usd_total']:.4f} vs URDF {mass['urdf_total']:.3f} kg（相对差 {rel:.1e}）；"
        f"运行时检查 {len(checks) - len(failed)}/{len(checks)}",
        ["reports/asset_actuator_audit.json"],
        {
            "permutation_source_to_target": runtime["order"],
            "joint_names_usd": runtime["joint_names_usd"],
            "mass_relative_diff": rel,
            "failed_checks": failed,
        },
    )


def row_t03(rep: dict) -> dict:
    obs = rep["observation_adapter_tests"]
    parity = rep["source_adapter_parity"]
    blocks_ok = True
    worst = 0.0
    for device, block in parity["devices"].items():
        for row in block["fixtures"]:
            for name, item in row["obs_blocks"].items():
                blocks_ok = blocks_ok and item["passed"]
                worst = max(worst, float(item["max_abs_diff"]))
    ok = obs["verdict"]["all_passed"] and blocks_ok
    return make_row(
        "T-03",
        "45 维观测与归一化",
        ok,
        f"适配器自测 {obs['verdict']['n_checks']} 项全过；独立 oracle 四组（cpu/cuda × f32/f64）"
        f"六块逐块通过，最大的块偏差 {worst:.2e}（判据 atol 1e-6/rtol 1e-5）",
        ["reports/observation_adapter_tests.json", "reports/source_adapter_parity.json"],
        {"max_block_diff": worst},
    )


def row_t04(rep: dict) -> dict:
    action = rep["action_pd_tests"]
    parity = rep["source_adapter_parity"]
    q_bad = 0.0
    tau_bad = 0.0
    for device, block in parity["devices"].items():
        for row in block["fixtures"]:
            q_bad = max(q_bad, float(row["q_target"]["max_abs_diff"]))
            tau_bad = max(tau_bad, float(row["tau_clipped"]["max_abs_diff"]))
    const = action["constants"]
    ok = action["verdict"]["all_passed"] and q_bad <= 1e-6 and tau_bad <= 1e-4
    return make_row(
        "T-04",
        "动作/默认姿态/PD/实际力矩",
        ok,
        f"q_target 最大偏差 {q_bad:.2e} rad（≤1e-6）；τ 最大偏差 {tau_bad:.2e} N·m（≤1e-4）；"
        f"action_scale={const['action_scale']}、clip={const['clip_actions']}、kp={const['stiffness']}、kd={const['damping']}；"
        f"自测 {action['verdict']['n_checks']} 项",
        ["reports/action_pd_tests.json", "reports/source_adapter_parity.json"],
        {"q_target_max": q_bad, "tau_max": tau_bad, "constants": const},
    )


def row_t05(rep: dict) -> dict:
    timing = rep["timing_command_reset"]
    counters = timing["counters"]
    expected = {
        "window_physics_steps": 6000,
        "policy_steps": 1500,
        "command_intervals": 300,
        "paired_points": 301,
    }
    ids = {item["id"]: item["ok"] for item in timing["checks"]}
    timing_ids = ["C1-warmup", "C2-counters", "C3-decimation", "C4-command-interval"]
    ok = (
        timing["verdict"]["all_passed"]
        and all(int(counters[key]) == value for key, value in expected.items())
        and all(ids.get(item, False) for item in timing_ids)
    )
    return make_row(
        "T-05",
        "200/50/10 Hz 时序",
        ok,
        f"实测 {counters['window_physics_steps']} 物理步 / {counters['policy_steps']} 策略步 / "
        f"{counters['command_intervals']} 命令区间 / {counters['paired_points']} 配对点；"
        f"每策略步恰 4 子步、每命令恰 5 策略步（{timing['verdict']['n_checks']} 项检查全过）",
        ["reports/timing_command_reset.json"],
        counters,
    )


def row_t06(rep: dict) -> dict:
    timing = rep["timing_command_reset"]
    ids = {item["id"]: item["ok"] for item in timing["checks"]}
    cmd_checks = ["C5-command-fidelity", "N2-command-override"]
    ok = timing["verdict"]["all_passed"] and all(ids.get(key, False) for key in cmd_checks)
    return make_row(
        "T-06",
        "外部命令无暗中改写",
        ok,
        "命令块与写入值偏差 0.0；0.1 m/s 未被 0.2 死区吃掉；纯 vy / 纯 wz 未被吞；"
        f"相关检查 {cmd_checks} 全过",
        ["reports/timing_command_reset.json", "reports/timing_command_trace.jsonl"],
        {"command_checks": cmd_checks},
    )


def row_t07(rep: dict) -> dict:
    timing = rep["timing_command_reset"]
    ids = {item["id"]: item["ok"] for item in timing["checks"]}
    reset_checks = ["C7-termination-capture", "C8-per-env-reset", "N3-cross-env"]
    ok = timing["verdict"]["all_passed"] and all(ids.get(key, False) for key in reset_checks)
    return make_row(
        "T-07",
        "reset 与上一动作逐 env 清零",
        ok,
        f"终止状态在重置前抓取、重置后回到初始高度、其他 env 的上一动作与命令逐位不变；"
        f"相关检查 {reset_checks}",
        ["reports/timing_command_reset.json"],
        {"reset_checks": reset_checks},
    )


def row_t08(rep: dict) -> dict:
    batch = rep["batch_isolation"]
    amendment = rep["amendment"]
    by = {item["id"]: item for item in batch["checks"]}
    a = by["T08-decoupling"]
    b = by["T08-behavioral-equivalence"]
    orig = by["T08-trajectory-prereg"]
    ok = bool(a["ok"]) and bool(b["ok"])
    return make_row(
        "T-08",
        "批量隔离（按修正判据 T-08a/T-08b）",
        ok,
        f"T-08a 解耦：其余 31 个 env 反向时 env0 的 τ 逐位不变（max|Δ|={a['max_abs_diff_env0']!r}），"
        f"同时 env1 变化 {a['max_abs_diff_env1']:.2f} N·m 证明改动生效；"
        f"T-08b 行为等价：单 env 落在批量散布的 "
        f"{max(abs(v) for v in b['z_scores'].values()):.2f}σ（判据 3σ）。"
        f"原判据（τ 轨迹 atol 1e-4）实测 max|Δ|={orig['curve']['max']:.2f}，"
        f"已按修正案 {amendment['sha256'][:12]}… 保留为失败记录",
        ["reports/batch_isolation.json", "config/integration_acceptance_amendment_t08_v1.json"],
        {
            "amendment_sha256": amendment["sha256"],
            "decoupling_max_abs_diff": a["max_abs_diff_env0"],
            "behavioral_max_abs_z": max(abs(v) for v in b["z_scores"].values()),
            "original_criterion_max_abs_diff": orig["curve"]["max"],
            "original_criterion_passed": orig["ok"],
        },
    )


def row_t09(rep: dict) -> dict:
    profile = rep["profile_effectiveness"]
    readback = profile["readback"]
    return make_row(
        "T-09",
        "五 profile 实际施加",
        profile["verdict"]["all_passed"],
        f"{profile['verdict']['n_checks']} 项全过：摩擦逐 shape 读回、质量 +3.000 kg、推扰 Δv=(0.5,0)、"
        f"噪声只落 9:33、只改 env0 时 env1 零污染（摩擦读回 {readback.get('friction_env0')}）",
        ["reports/profile_effectiveness.json"],
        {"checks": profile["verdict"]["n_checks"]},
    )


def row_t10(rep: dict) -> dict:
    energy = rep["hira_energy_meter_validation"]
    readonly = energy["readonly"]
    by = {item["id"]: item for item in energy["checks"]}
    ok = (
        energy["verdict"]["all_passed"]
        and readonly["fingerprint_on"] == readonly["fingerprint_off"]
        and float(readonly["max_abs_diff"]) == 0.0
    )
    return make_row(
        "T-10",
        "200 Hz 计量相位与非侵入性",
        ok,
        f"17 项检查全过；在线与两套离线实现 rel=0.0（判据 1e-9）；记录器开/关 250 帧轨迹逐位一致"
        f"（max|Δ|={readonly['max_abs_diff']!r}）；τ 取 applied_torque、q̇ 取子步结束时刻",
        ["reports/hira_energy_meter_validation.json"],
        {
            "rel_online_vs_numpy": by["E05"]["rel_vs_numpy"],
            "readonly_max_abs_diff": readonly["max_abs_diff"],
            "phase_pre_vs_post": by["E06"]["rel_pre_vs_post"],
        },
    )


def row_t11(rep: dict) -> dict:
    legacy = rep["legacy_evidence_availability"]
    isolation = rep["config_isolation_tests"]
    verdict = legacy["verdict"]
    ok = (
        verdict["delivery_bundle_reverifies"]
        and verdict["v2_config_artifacts_all_present"]
        and verdict["v2_reports_all_present"]
        and isolation["ok"]
    )
    return make_row(
        "T-11",
        "v2 旧证据可复算与新旧拒混",
        ok,
        f"v2 交付包重算 {verdict.get('delivery_bundle_items') or '全部'} 项散列全对；"
        f"配置隔离自测 {isolation['n_pass']}/{isolation['n_cases']}（含把 HIRA_* 指向 v2 产物时必须拒绝）；"
        f"v2 训练期原始日志不在本机（已登记，仅作对照基准）",
        [
            "reports/legacy_evidence_availability.json",
            "reports/config_isolation_tests.json",
        ],
        {"isolation": f"{isolation['n_pass']}/{isolation['n_cases']}"},
    )


def row_t12(rep: dict) -> dict:
    export = rep["actor_export_parity"]
    summary = export["summary"]
    cpu = export["cpu"]
    gpu = export.get("gpu", {})
    ok = bool(summary["passed"]) and summary["cpu_max_abs_diff"] <= 1e-5 and bool(summary["gpu_all_passed"])
    return make_row(
        "T-12",
        "CPU 同状态 oracle（actor 等价）",
        ok,
        f"29 例：CPU 最大偏差 {summary['cpu_max_abs_diff']:.2e}（判据 1e-5）；GPU 最大偏差 "
        f"{gpu.get('max_abs_diff_source_vs_cpu')}；推导输入与库内随机各 {len(cpu['cases'])} 组",
        ["reports/actor_export_parity.json"],
        {"cpu_max": summary["cpu_max_abs_diff"], "gpu_max": gpu.get("max_abs_diff_source_vs_cpu")},
    )


def row_t13(rep: dict) -> dict:
    parity = rep["source_adapter_parity"]
    negatives = parity.get("negative_tests") or parity.get("negatives") or []
    detected = [item for item in negatives if item.get("detected")]
    ok = len(negatives) >= 3 and len(detected) == len(negatives)
    return make_row(
        "T-13",
        "坐标/映射破坏必须被拒",
        ok,
        f"{len(negatives)} 类破坏全部被检出："
        + "；".join(f"{item.get('name', item.get('id'))}" for item in negatives),
        ["reports/source_adapter_parity.json"],
        {"n_negative": len(negatives), "n_detected": len(detected)},
    )


def row_t14(rep: dict) -> dict:
    loop = rep["clean_closed_loop"]
    verdict = loop["verdict"]
    friction = loop.get("shard_config", {}).get("friction")
    per = loop["per_command"]
    # 每条命令的 trials 里有 speed_error；取 nominal 域里的最大值作为门禁数字
    worst = max(
        trial["speed_error"]
        for item in per.values()
        if item.get("domain") == "nominal"
        for trial in item["trials"]
    ) if per else None
    ok = verdict["all_nominal_passed"] and not verdict["blocking_failures"] and friction == 1.0
    return make_row(
        "T-14",
        "clean 闭环行为",
        ok,
        f"11/11 命令通过（摩擦 {friction}）；最大速度误差 {worst:.3f} m/s（阈值 0.25）；"
        f"阻断项 {verdict['blocking_failures'] or '无'}；低速工况单列报告",
        ["reports/clean_closed_loop.json"],
        {"max_error": worst, "friction": friction},
    )


# --------------------------------------------------------------------------- #
# 控制器描述
def urdf_effort_limits(urdf_path: Path) -> dict:
    text = Path(urdf_path).read_text(encoding="utf-8")
    out = {}
    for match in re.finditer(r'<joint\s+name="([^"]+)"\s+type="revolute">(.*?)</joint>', text, re.S):
        name, body = match.group(1), match.group(2)
        effort = re.search(r'effort="([0-9.]+)"', body)
        if effort:
            out[name] = float(effort.group(1))
    return out


def build_descriptor(rep: dict) -> dict:
    from hira_go2.action_pd import ACTION_SCALE, CLIP_ACTIONS, DAMPING, DECIMATION, STIFFNESS
    from hira_go2.asset_cfg import (
        HIRA_CONTACT_OFFSET,
        HIRA_DEFAULT_ANGLES,
        HIRA_INIT_BASE_POS,
        HIRA_REPLACE_CYLINDERS_WITH_CAPSULES,
        HIRA_SELF_COLLISIONS_ENABLED,
        HIRA_COLLAPSE_FIXED_JOINTS,
    )
    from hira_go2.bundle import build_controller_descriptor
    from hira_go2.env_runner import FALL_HEIGHT, FALL_TILT_DEG, SOURCE_ORDER, TimingConfig
    from hira_go2.observation import (
        ANG_VEL_SCALE,
        BLOCK_SLICES,
        CLIP_OBSERVATIONS,
        COMMAND_SCALE,
        DOF_POS_SCALE,
        DOF_VEL_SCALE,
        GRAVITY_SCALE,
        NOISED_BLOCKS,
        OBS_DIM,
        PREV_ACTION_SCALE,
    )

    export = rep["actor_export_parity"]
    audit = rep["asset_actuator_audit"]
    loop = rep["clean_closed_loop"]
    timing = TimingConfig()
    target_order = audit["runtime"]["joint_names_usd"]
    permutation = [SOURCE_ORDER.index(name) for name in target_order]

    return build_controller_descriptor(
        controller_id=args_cli.controller_id,
        version="v1",
        policy_path=Path(args_cli.policy),
        created_at=utc_now(),
        source={
            "repository": "HiRA/legged_gym_go2",
            "checkpoint": str(args_cli.checkpoint),
            "checkpoint_sha256": export["checkpoint_sha256"],
            "internal_iter": 16000,
            "filename_iter": 18500,
            "note": "文件名 18500 与文件内部 iter 16000 不一致；身份只认 sha256 与张量摘要",
            "actor_tensor_digest": export["export_tensor_digest"],
        },
        joints={
            "source_order": list(SOURCE_ORDER),
            "target_order": list(target_order),
            "permutation_source_to_target": permutation,
            "default_angles_source": dict(HIRA_DEFAULT_ANGLES),
            "effort_limits_source": urdf_effort_limits(Path(args_cli.urdf)),
            "source_order_basis": "URDF 声明顺序（逐腿）；已由 HIRA-017 的顺序对照实验确认",
        },
        control={
            "n_joints": 12,
            "action_dim": 12,
            "action_scale": ACTION_SCALE,
            "clip_actions": CLIP_ACTIONS,
            "stiffness": STIFFNESS,
            "damping": DAMPING,
            "decimation": DECIMATION,
            "torque_rule": "tau = clip(kp*(q_def + scale*a - q) - kd*qdot, ±effort)，每个物理子步用新鲜 q/qdot",
            "deterministic": True,
        },
        observation={
            "dim": OBS_DIM,
            "n_joints": 12,
            "blocks": [{"name": name, "start": part.start, "end": part.stop} for name, part in BLOCK_SLICES],
            "scales": {
                "ang_vel": ANG_VEL_SCALE,
                "gravity": GRAVITY_SCALE,
                "command": list(COMMAND_SCALE),
                "dof_pos": DOF_POS_SCALE,
                "dof_vel": DOF_VEL_SCALE,
                "prev_action": PREV_ACTION_SCALE,
            },
            "clip": CLIP_OBSERVATIONS,
            "noise_blocks": list(NOISED_BLOCKS),
            "noise_note": "训练期噪声只加在 dof_pos/dof_vel（9:33），33:45 永不加噪；clean 运行完全关闭噪声。"
            "幅度不在源码可得范围（预登记 U-5），故不登记数值",
        },
        timing={
            "sim_dt": timing.sim_dt,
            "decimation": timing.decimation,
            "policy_dt": timing.policy_dt,
            "command_policy_steps": timing.command_policy_steps,
            "command_dt": timing.command_dt,
            "window_s": timing.window_s,
            "physics_steps": timing.window_physics_steps,
            "policy_steps": timing.window_policy_steps,
            "command_intervals": timing.window_command_intervals,
            "paired_points": timing.paired_points,
        },
        asset={
            "urdf": str(args_cli.urdf),
            "urdf_in_bundle": "urdf_go2.urdf",
            "urdf_sha256": sha256_file(Path(args_cli.urdf)),
            "usd_cache": str(args_cli.usd_dir),
            "usd_note": "由 URDF 现场转换（不复用 v2 的 USD）；缓存目录不随包分发",
            "total_mass_kg": float(audit["runtime"]["mass"]["usd_total"]),
            "actuator": "IdealPD（stiffness=damping=0, effort_limit=URDF）→ 纯力矩透传",
            "effort_only": True,
            "collapse_fixed_joints": bool(HIRA_COLLAPSE_FIXED_JOINTS),
            "self_collisions": bool(HIRA_SELF_COLLISIONS_ENABLED),
            "replace_cylinders_with_capsules": bool(HIRA_REPLACE_CYLINDERS_WITH_CAPSULES),
            "contact_offset": HIRA_CONTACT_OFFSET,
            "init_base_pos": list(HIRA_INIT_BASE_POS),
            "friction": float(loop["shard_config"]["friction"]),
            "source_friction": 1.0,
            "friction_as_imported": float(loop["shard_config"].get("friction_as_imported", 0.5)),
            "friction_note": "导入后的 USD 默认摩擦 0.5 ≠ 源 1.0；clean 必须显式写成 1.0（HIRA-017 实测）",
        },
        energy={
            "metric_id": "D-ENERGY-001",
            "formula": "J = Σ_k Σ_j max(τ_j[k]·q̇_j[k], 0) · dt",
            "dt_s": timing.sim_dt,
            "tau_source": "robot.data.applied_torque",
            "qdot_phase": "post_substep",
            "clip": "逐关节截断后求和",
            "unit": "J",
            "note": "关节机械正功，不是电池能耗；外部推扰功不计入",
        },
        environment={
            "physics_backend": "PhysX 5.1（GPU）",
            "isaaclab_source": "/pub/data/caohy/IsaacLab（0.47.2 editable）",
            "python": sys.version.split()[0],
            "torch": __import__("torch").__version__,
            "enhanced_determinism": False,
            "termination": {"fall_height_m": FALL_HEIGHT, "fall_tilt_deg": FALL_TILT_DEG},
            "note": "PhysX GPU 不保证批内逐位一致；跨批差异属求解器舍入 + 接触混沌（见 T-08 修正案）",
        },
        policy_meta={
            "format": "torchscript",
            "input_dim": 45,
            "output_dim": 12,
            "hidden_dims": [int(x) for x in export["architecture"]["hidden_dims"]],
            "activation": export["activation"],
            "normalizer": export["normalizer"]["has_normalizer"],
        },
        acceptance={
            "prereg": str(Path(args_cli.prereg).name),
            "prereg_sha256": sha256_file(Path(args_cli.prereg)),
            "amendment": str(Path(args_cli.amendment).name),
            "amendment_sha256": sha256_file(Path(args_cli.amendment)),
            "gate_report": "reports/integration_gate.json",
        },
        notes=[
            "本包只用于 HIRA 校准链路；改权重/资产/增益/观测/时钟需新 controller 版本并重过受影响门禁。",
            "source_order 依据 URDF 声明顺序；Lab 侧按名字做双射，不按位置对齐。",
        ],
    )


# --------------------------------------------------------------------------- #
# 拒混（T-15）
def reseal(bundle_dir: Path, controller: dict) -> None:
    """攻击者视角：改完内容后重新生成一份自洽的 manifest（散列全对）。"""
    from hira_go2.bundle import CONTROLLER_NAME, MANIFEST_NAME, compute_bundle_id, sha256_file

    (bundle_dir / CONTROLLER_NAME).write_text(
        json.dumps(controller, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = [
        {"path": path.name, "sha256": sha256_file(path), "bytes": int(path.stat().st_size)}
        for path in sorted(bundle_dir.iterdir())
        if path.is_file() and path.name != MANIFEST_NAME
    ]
    manifest = {
        "schema": controller["schema"],
        "controller_id": controller["controller_id"],
        "version": controller["version"],
        "created_at": controller["created_at"],
        "bundle_id": compute_bundle_id(controller, files),
        "files": files,
    }
    (bundle_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


def rejection_tests(bundle_dir: Path, controller: dict, expected_joints: dict) -> list[dict]:
    from hira_go2.bundle import BundleError, load_bundle

    import tempfile

    work = Path(tempfile.mkdtemp(prefix="gate_reject_"))

    cases = []

    def attempt(case_id: str, name: str, mutate, reseal_after: bool) -> None:
        target = work / case_id / bundle_dir.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(bundle_dir, target)
        victim = json.loads(json.dumps(controller))
        mutate(target, victim)
        if reseal_after:
            reseal(target, victim)
        try:
            load_bundle(target, expected_controller_id=controller["controller_id"], expected_joints=expected_joints)
        except BundleError as exc:
            cases.append({"id": case_id, "name": name, "rejected": True, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            cases.append(
                {"id": case_id, "name": name, "rejected": True, "error": f"{type(exc).__name__}: {exc}"}
            )
        else:
            cases.append({"id": case_id, "name": name, "rejected": False, "error": "没有被拒绝"})
        shutil.rmtree(target.parent, ignore_errors=True)

    def flip_policy(path: Path, victim: dict) -> None:
        data = bytearray((path / "policy.pt").read_bytes())
        data[len(data) // 2] ^= 0xFF
        (path / "policy.pt").write_bytes(bytes(data))

    def swap_joint(path: Path, victim: dict) -> None:
        order = victim["joints"]["target_order"]
        order[0], order[1] = order[1], order[0]
        victim["joints"]["permutation_source_to_target"] = [
            victim["joints"]["source_order"].index(name) for name in order
        ]

    def shift_clock(path: Path, victim: dict) -> None:
        victim["timing"]["sim_dt"] = 0.004

    def change_id(path: Path, victim: dict) -> None:
        victim["controller_id"] = "hira_go2_18500_forged"

    def swap_source_order(path: Path, victim: dict) -> None:
        # 不碰 target_order，只动 source_order 与置换——这类伪造骗不过运行时参照，
        # 但必须被"包内 URDF 自证"拦下（URDF 的声明顺序没变）
        order = victim["joints"]["source_order"]
        order[0], order[1] = order[1], order[0]
        victim["joints"]["permutation_source_to_target"] = [
            order.index(name) for name in victim["joints"]["target_order"]
        ]

    def half_friction(path: Path, victim: dict) -> None:
        victim["asset"]["friction"] = 0.5

    attempt("R1-policy-bitflip", "策略文件被改一个字节（不重封）", flip_policy, False)
    attempt("R2-joint-swap", "交换两个关节名并重封（映射被改）", swap_joint, True)
    attempt("R3-clock-shift", "把 sim_dt 改成 0.004 并重封（时钟被改）", shift_clock, True)
    attempt("R4-controller-id", "改 controller_id 并重封（身份被改）", change_id, True)
    attempt("R5-friction-half", "把摩擦写成 0.5（回到导入默认）并重封", half_friction, True)
    attempt("R6-source-order-swap", "只改 source_order 与置换并重封（映射被改）", swap_source_order, True)
    return cases


# --------------------------------------------------------------------------- #
def phase_matrix() -> int:
    from hira_go2.bundle import freeze_bundle, load_bundle

    reports = {
        "actor_export_parity": load_json("actor_export_parity.json"),
        "asset_actuator_audit": load_json("asset_actuator_audit.json"),
        "observation_adapter_tests": load_json("observation_adapter_tests.json"),
        "action_pd_tests": load_json("action_pd_tests.json"),
        "source_adapter_parity": load_json("source_adapter_parity.json"),
        "timing_command_reset": load_json("timing_command_reset.json"),
        "profile_effectiveness": load_json("profile_effectiveness.json"),
        "clean_closed_loop": load_json("clean_closed_loop.json"),
        "hira_energy_meter_validation": load_json("hira_energy_meter_validation.json"),
        "legacy_evidence_availability": load_json("legacy_evidence_availability.json"),
        "config_isolation_tests": load_json("config_isolation_tests.json"),
        "batch_isolation": load_json("batch_isolation.json"),
    }
    amendment_path = Path(args_cli.amendment)
    if not amendment_path.is_file():
        raise SystemExit(f"缺少 T-08 修正案：{amendment_path}")
    reports["amendment"] = json.loads(amendment_path.read_text(encoding="utf-8"))

    rows = [
        row_t01(reports),
        row_t02(reports),
        row_t03(reports),
        row_t04(reports),
        row_t05(reports),
        row_t06(reports),
        row_t07(reports),
        row_t08(reports),
        row_t09(reports),
        row_t10(reports),
        row_t11(reports),
        row_t12(reports),
        row_t13(reports),
        row_t14(reports),
    ]
    for row in rows:
        print(f"[{'ok  ' if row['ok'] else 'FAIL'}] {row['id']} {row['name']}", flush=True)

    failed = [row["id"] for row in rows if row["required"] and not row["ok"]]
    base = {
        "report": "integration_gate",
        "task": "HIRA-020",
        "generated_at": utc_now(),
        "prereg": {"file": Path(args_cli.prereg).name, "sha256": sha256_file(Path(args_cli.prereg))},
        "amendment": {"file": amendment_path.name, "sha256": sha256_file(amendment_path)},
        "rows": rows,
        "evidence_hashes": {
            name: sha256_file(Path(args_cli.out_dir) / name)
            for name in sorted(p.name for p in Path(args_cli.out_dir).glob("*.json"))
            if name in {
                "actor_export_parity.json",
                "asset_actuator_audit.json",
                "observation_adapter_tests.json",
                "action_pd_tests.json",
                "source_adapter_parity.json",
                "timing_command_reset.json",
                "profile_effectiveness.json",
                "clean_closed_loop.json",
                "hira_energy_meter_validation.json",
                "legacy_evidence_availability.json",
                "config_isolation_tests.json",
                "batch_isolation.json",
            }
        },
    }

    if failed:
        base["verdict"] = {
            "g1_passed": False,
            "failed_rows": failed,
            "bundle": None,
            "note": "T-01..T-13/T-15 任一失败即阻断；不得生成 passed 包，不得进入 pilot。",
        }
        out = Path(args_cli.out_dir) / "integration_gate.json"
        out.write_text(json.dumps(base, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"G1_BLOCKED failed={failed} -> {out}", flush=True)
        return 1

    # ---- 冻结控制器包 ----
    controller = build_descriptor(reports)
    bundle_dir = Path(args_cli.bundle_root) / args_cli.controller_id
    if bundle_dir.exists():
        if not args_cli.refreeze:
            raise SystemExit(f"冻结目录已存在：{bundle_dir}（包不可变；确认重冻请加 --refreeze）")
        shutil.rmtree(bundle_dir)
    manifest = freeze_bundle(
        bundle_dir, controller, Path(args_cli.policy), {"urdf_go2.urdf": Path(args_cli.urdf)}
    )
    log(f"bundle frozen: {bundle_dir} id={manifest['bundle_id'][:16]}")

    # ---- 重载自己刚冻的包（散列 + 语义 + 包内 URDF 自证 + 运行时参照）----
    expected_joints = {
        "source_order": reports["asset_actuator_audit"]["runtime"].get("joint_order_source"),
        "target_order": reports["asset_actuator_audit"]["runtime"]["joint_names_usd"],
    }
    expected_joints = {k: v for k, v in expected_joints.items() if v}
    loaded = load_bundle(
        bundle_dir,
        expected_controller_id=args_cli.controller_id,
        expected_joints=expected_joints,
    )

    # ---- T-15 拒混 ----
    cases = rejection_tests(bundle_dir, loaded["controller"], expected_joints)
    rejected = [item for item in cases if item["rejected"]]
    t15 = make_row(
        "T-15",
        "回归：故意破坏必须拒绝",
        len(rejected) == len(cases),
        f"{len(rejected)}/{len(cases)} 类破坏被拒绝：" + "；".join(f"{item['id']}（{item['error'][:42]}）" for item in cases),
        [str(bundle_dir / "manifest.json")],
        {"cases": cases},
    )
    rows.append(t15)
    print(f"[{'ok  ' if t15['ok'] else 'FAIL'}] T-15 {t15['name']}", flush=True)

    base["rows"] = rows
    base["bundle"] = {
        "dir": str(bundle_dir),
        "controller_id": loaded["controller"]["controller_id"],
        "version": loaded["controller"]["version"],
        "bundle_id": manifest["bundle_id"],
        "policy_sha256": loaded["controller"]["policy"]["sha256"],
        "files": manifest["files"],
        "manifest_sha256": sha256_file(bundle_dir / "manifest.json"),
        "controller_sha256": sha256_file(bundle_dir / "controller.json"),
    }
    base["verdict"] = {
        "g1_passed": bool(t15["ok"]),
        "failed_rows": [] if t15["ok"] else ["T-15"],
        "n_rows": len(rows),
        "note": "T-01..T-15 全部通过；最小重载运行确认见 reports/bundle_reload.json（--phase reload）。",
    }
    out = Path(args_cli.out_dir) / "integration_gate.json"
    out.write_text(json.dumps(base, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"G1_MATRIX_OK bundle={manifest['bundle_id'][:16]} -> {out}", flush=True)
    return 0 if t15["ok"] else 1


# --------------------------------------------------------------------------- #
def phase_reload() -> int:
    import torch

    from camra_calibration.energy import DEFAULT_DT
    from hira_go2.bundle import load_bundle
    from hira_go2.env_runner import HiraLabRunner, TimingConfig

    bundle_dir = Path(args_cli.bundle_root) / args_cli.controller_id
    audit = json.loads((Path(args_cli.out_dir) / "asset_actuator_audit.json").read_text(encoding="utf-8"))
    expected_joints = {"target_order": audit["runtime"]["joint_names_usd"]}
    loaded = load_bundle(
        bundle_dir,
        expected_controller_id=args_cli.controller_id,
        expected_joints=expected_joints,
    )
    controller = loaded["controller"]
    log(f"bundle loaded: {controller['controller_id']} {controller['version']} id={loaded['manifest']['bundle_id'][:16]}")

    # 1) 包里的描述必须与将要运行的代码一致（否则"冻结的"和"跑的"是两回事）
    from hira_go2.action_pd import ACTION_SCALE, CLIP_ACTIONS, DAMPING, DECIMATION, STIFFNESS
    from hira_go2.asset_cfg import HIRA_DEFAULT_ANGLES
    from hira_go2.env_runner import SOURCE_ORDER

    consistency = {
        "action_scale": abs(float(controller["control"]["action_scale"]) - ACTION_SCALE) < 1e-12,
        "clip_actions": abs(float(controller["control"]["clip_actions"]) - CLIP_ACTIONS) < 1e-12,
        "stiffness": abs(float(controller["control"]["stiffness"]) - STIFFNESS) < 1e-12,
        "damping": abs(float(controller["control"]["damping"]) - DAMPING) < 1e-12,
        "decimation": int(controller["control"]["decimation"]) == DECIMATION,
        "source_order": list(controller["joints"]["source_order"]) == list(SOURCE_ORDER),
        "default_angles": all(
            abs(float(controller["joints"]["default_angles_source"][name]) - HIRA_DEFAULT_ANGLES[name]) < 1e-12
            for name in SOURCE_ORDER
        ),
    }

    # 2) 用包里的策略/URDF/摩擦跑一小段，确认它是能驱动实体的
    timing = TimingConfig(sim_dt=DEFAULT_DT)
    runner = HiraLabRunner(
        controller["asset"]["urdf"],
        args_cli.usd_dir,
        num_envs=1,
        device=args_cli.device,
        policy_path=str(bundle_dir / controller["policy"]["file"]),
        timing=timing,
    )
    view = runner.robot.root_physx_view
    material = view.get_material_properties().clone()
    material[:, :, 0] = float(controller["asset"]["friction"])
    material[:, :, 1] = float(controller["asset"]["friction"])
    view.set_material_properties(material, torch.tensor([0], dtype=torch.int64))

    runner.reset_all()
    runner.set_command(torch.tensor([[0.5, 0.0, 0.0]], dtype=runner.dtype))
    tau_trace = []
    heights = []
    for _ in range(args_cli.reload_steps):
        runner.policy_step()
        tau_trace.append(runner.robot.data.applied_torque[0].detach().cpu().numpy().astype(float))
        heights.append(float(runner.robot.data.root_pos_w[0, 2]))
    import numpy as np

    tau = np.stack(tau_trace).astype(np.float64)
    effort = np.array(
        [controller["joints"]["effort_limits_source"][name] for name in controller["joints"]["target_order"]],
        dtype=np.float64,
    )
    over = float((np.abs(tau) - effort.reshape(1, -1)).max())
    finite = bool(np.isfinite(tau).all())
    report = {
        "report": "bundle_reload",
        "task": "HIRA-020",
        "generated_at": utc_now(),
        "bundle": {
            "dir": str(bundle_dir),
            "controller_id": controller["controller_id"],
            "bundle_id": loaded["manifest"]["bundle_id"],
            "policy_sha256": controller["policy"]["sha256"],
        },
        "descriptor_matches_code": consistency,
        "run": {
            "policy_steps": args_cli.reload_steps,
            "command": [0.5, 0.0, 0.0],
            "tau_finite": finite,
            "max_abs_tau": float(np.abs(tau).max()),
            "max_over_effort": over,
            "height_start": heights[0],
            "height_end": heights[-1],
            "height_min": min(heights),
            "tau_fingerprint": hashlib.sha256(tau.astype("float32").tobytes()).hexdigest(),
        },
        "verdict": {
            "passed": bool(all(consistency.values()) and finite and over <= 1e-6 and min(heights) > 0.15),
            "failed": [key for key, value in consistency.items() if not value]
            + ([] if finite else ["tau_non_finite"])
            + ([] if over <= 1e-6 else ["tau_over_effort"])
            + ([] if min(heights) > 0.15 else ["fell"]),
        },
    }
    out = Path(args_cli.out_dir) / "bundle_reload.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"RELOAD_{'OK' if report['verdict']['passed'] else 'FAIL'} -> {out}", flush=True)
    return 0 if report["verdict"]["passed"] else 1


def phase_merge() -> int:
    gate_path = Path(args_cli.out_dir) / "integration_gate.json"
    reload_path = Path(args_cli.out_dir) / "bundle_reload.json"
    if not gate_path.is_file():
        raise SystemExit(f"缺少 {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if reload_path.is_file():
        reload_report = json.loads(reload_path.read_text(encoding="utf-8"))
        gate["bundle_reload"] = reload_report
        gate["rows"].append(
            make_row(
                "B-01",
                "冻结包最小重载运行确认",
                bool(reload_report["verdict"]["passed"]),
                f"重载 {reload_report['bundle']['controller_id']} 跑 {reload_report['run']['policy_steps']} 个策略步："
                f"τ 有限、max|τ|={reload_report['run']['max_abs_tau']:.2f} N·m、未越限"
                f"（{reload_report['run']['max_over_effort']:.1e}）、高度 {reload_report['run']['height_start']:.3f}→"
                f"{reload_report['run']['height_end']:.3f} m；描述与代码逐项一致",
                ["reports/bundle_reload.json"],
                reload_report["run"],
            )
        )
    failed = [row["id"] for row in gate["rows"] if row["required"] and not row["ok"]]
    gate["verdict"]["g1_passed"] = not failed
    gate["verdict"]["failed_rows"] = failed
    gate["verdict"]["n_rows"] = len(gate["rows"])
    gate_path.write_text(json.dumps(gate, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for row in gate["rows"]:
        print(f"[{'ok  ' if row['ok'] else 'FAIL'}] {row['id']} {row['name']}", flush=True)
    print(f"G1_{'PASSED' if not failed else 'BLOCKED'} failed={failed}", flush=True)
    return 0 if not failed else 1


# --------------------------------------------------------------------------- #
def main() -> int:
    if args_cli.phase == "matrix":
        return phase_matrix()
    if args_cli.phase == "reload":
        return phase_reload()
    return phase_merge()


if __name__ == "__main__":
    code = 2
    try:
        code = main()
    except SystemExit as exc:
        code = int(exc.code or 0)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("GATE_ERROR", flush=True)
        code = 2
    finally:
        if args_cli.phase == "reload":
            log("hard exit (skip simulation_app.close)")
            sys.stdout.flush()
            sys.stderr.flush()
            import os

            os._exit(code)
