"""五个物理执行条件的真实生效验证（HIRA-018）。

复用 v2 的 profile 定义（协议 v1.0.2 的 `profiles`），但施加方式按新 runner 重写：
新环境没有 manager 事件，所以每个旋钮都是**逐 env 直接写 PhysX 并读回**，
"配置里有值"和"实际生效"必须分开证明。

    python tests/check_profile_effectiveness.py --headless --device cuda:0 ...

七个检查：
  P1 摩擦写入 → 读回一致，且只改 friction 相关条件
  P2 质量写入（base +3 kg）→ 读回一致、总质量同步增加
  P3 推扰是**水平速度冲击**：写入后一步内 root 速度增量等于设定值（不换算成牛顿外力）
  P4 观测噪声只落在 9:33，幅度在源 noise_scale 内；clean 完全无噪声
  P5 clean 两次运行的轨迹逐位一致（确定性）
  P6 各扰动条件的轨迹与 clean 不同（证明真的施加了，而不是只写了配置）
  P7 恢复 clean 后摩擦/质量读回回到基准（reset 不被旧的 profile 残留影响）
  P8 2 env 场景下只改 env0，env1 的质量与摩擦不变（无跨 env 污染）
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
ADAPTER = REPO / "CaMRA" / "locomotion" / "adapters"
sys.path.insert(0, str(ADAPTER))

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default=str(REPO / "HiRA/legged_gym_go2/resources/robots/go2/urdf/go2.urdf"))
parser.add_argument("--usd-dir", default="/pub/data/caohy/logs/camra_hira/assets/usd_cache")
parser.add_argument("--policy", default=str(EXP / "exported/policy.pt"))
parser.add_argument("--protocol", default=str(REPO / "CaMRA/calibration/config/calibration_protocol_v1.0.2.json"))
parser.add_argument("--out", default=str(EXP / "reports/profile_effectiveness.json"))
parser.add_argument("--seconds", type=float, default=4.0)
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

T0 = time.time()

#: 五个条件都用**固定值**（不采样），保证同一条命令重跑结果一致。
PROFILE_PLAN = {
    "clean": {"friction": 1.0, "added_mass_kg": 0.0, "push": None, "obs_noise": False},
    "friction_only": {"friction": 0.3, "added_mass_kg": 0.0, "push": None, "obs_noise": False},
    "mass_only": {"friction": 1.0, "added_mass_kg": 3.0, "push": None, "obs_noise": False},
    "push_only": {"friction": 1.0, "added_mass_kg": 0.0, "push": {"time_s": 1.0, "dv_xy": [0.5, 0.0]}, "obs_noise": False},
    "combined": {"friction": 0.3, "added_mass_kg": 3.0, "push": {"time_s": 1.0, "dv_xy": [0.5, -0.25]}, "obs_noise": True},
}
#: 源 noise_scales（doff_pos 0.01 / dof_vel 1.5），噪声只落 9:33。
NOISE_SCALES = (0.01, 1.5)


def log(text: str) -> None:
    print(f"[profile] {time.time() - T0:7.1f}s {text}", flush=True)


def check(checks: list, check_id: str, name: str, ok: bool, detail: str) -> None:
    checks.append({"id": check_id, "name": name, "ok": bool(ok), "detail": detail})


def apply_friction(runner, value: float, env_ids: list[int] | None = None) -> dict:
    """逐 env × 逐 shape 写摩擦，返回读回值。"""
    view = runner.robot.root_physx_view
    current = view.get_material_properties()
    n_envs, n_shapes, _ = current.shape
    target = current.clone()
    ids = list(range(n_envs)) if env_ids is None else env_ids
    for env_id in ids:
        target[env_id, :, 0] = value  # static
        target[env_id, :, 1] = value  # dynamic
        target[env_id, :, 2] = 0.0    # restitution
    view.set_material_properties(target[:, :, 0], target[:, :, 1], target[:, :, 2])
    readback = view.get_material_properties()
    return {
        "assigned": value,
        "n_shapes": int(n_shapes),
        "readback_static": [[round(float(v), 6) for v in row] for row in readback[:, :, 0].tolist()],
        "readback_dynamic": [[round(float(v), 6) for v in row] for row in readback[:, :, 1].tolist()],
        "max_abs_diff": float((readback[:, :, 0] - target[:, :, 0]).abs().max()),
    }


def capture_baseline(runner) -> dict:
    """记录基准质量/惯量/材质，供后续精确恢复（不能靠 delta=0 还原）。"""
    view = runner.robot.root_physx_view
    return {
        "masses": view.get_masses().clone(),
        "inertias": view.get_inertias().clone(),
        "material": view.get_material_properties().clone(),
    }


def restore_baseline(runner, baseline: dict) -> dict:
    view = runner.robot.root_physx_view
    view.set_masses(baseline["masses"])
    view.set_inertias(baseline["inertias"])
    material = baseline["material"]
    view.set_material_properties(material[:, :, 0], material[:, :, 1], material[:, :, 2])
    readback = view.get_masses()
    return {
        "mass_max_abs_diff": float((readback - baseline["masses"]).abs().max()),
        "material_max_abs_diff": float(
            (view.get_material_properties() - baseline["material"]).abs().max()
        ),
    }


def apply_mass(runner, delta_kg: float, env_ids: list[int] | None = None, link_index: int = 0) -> dict:
    """给指定 body 加质量，并按比例缩放惯量（与 IsaacLab 官方 randomize_rigid_body_mass 同法）。"""
    view = runner.robot.root_physx_view
    masses = view.get_masses()
    inertias = view.get_inertias()
    before_total = masses.sum(dim=1).clone()
    base_before = masses[:, link_index].clone()

    new_masses = masses.clone()
    new_inertias = inertias.clone()
    ids = list(range(masses.shape[0])) if env_ids is None else env_ids
    for env_id in ids:
        old = float(masses[env_id, link_index])
        new = max(old + delta_kg, 1e-3)
        ratio = new / old
        new_masses[env_id, link_index] = new
        new_inertias[env_id, link_index, :, :] *= ratio
    view.set_masses(new_masses)
    view.set_inertias(new_inertias)

    readback_masses = view.get_masses()
    return {
        "delta_kg": delta_kg,
        "link_index": link_index,
        "link_name": runner.robot.body_names[link_index],
        "base_before": [round(float(v), 6) for v in base_before.tolist()],
        "base_after": [round(float(v), 6) for v in readback_masses[:, link_index].tolist()],
        "total_before": [round(float(v), 6) for v in before_total.tolist()],
        "total_after": [round(float(v), 6) for v in readback_masses.sum(dim=1).tolist()],
        "max_abs_diff": float((readback_masses - new_masses).abs().max()),
    }


def main() -> int:
    from hira_go2.env_runner import HiraLabRunner, TimingConfig

    protocol = json.loads(Path(args_cli.protocol).read_text(encoding="utf-8"))
    defined = {item["id"]: item for item in protocol["profiles"]}
    missing = [key for key in PROFILE_PLAN if key not in defined]
    checks: list[dict] = []
    check(checks, "P0-profiles", "五个条件与协议定义同名", not missing, f"协议里缺：{missing}" if missing else "clean/friction_only/mass_only/push_only/combined 全部存在")

    timing = TimingConfig(window_s=args_cli.seconds)
    runner = HiraLabRunner(
        args_cli.urdf, args_cli.usd_dir, num_envs=1, device=args_cli.device,
        policy_path=args_cli.policy, timing=timing,
    )
    log("runner ready")
    runner.set_command(torch.zeros(1, 3))
    baseline_state = capture_baseline(runner)
    baseline_mass = float(baseline_state["masses"].sum())
    baseline_friction = float(baseline_state["material"][0, 0, 0])
    log(f"baseline mass={baseline_mass:.4f} friction={baseline_friction:.3f}")

    def run_seconds(seconds: float, push: dict | None = None) -> dict:
        """跑一段，可选在指定时刻做一次水平速度冲击；返回轨迹指纹与推送证据。"""
        n_steps = int(round(seconds / runner.timing.policy_dt))
        heights, speeds = [], []
        push_evidence = None
        push_done = False
        for _ in range(n_steps):
            if push and not push_done and runner.physics_step * runner.timing.sim_dt >= push["time_s"]:
                before = runner.robot.data.root_lin_vel_w.clone()
                commanded = torch.tensor(
                    [push["dv_xy"][0], push["dv_xy"][1], 0.0, 0.0, 0.0, 0.0],
                    device=args_cli.device, dtype=runner.dtype,
                )
                runner.robot.write_root_velocity_to_sim(before + commanded.reshape(1, 6))
                runner.robot.update(runner.timing.sim_dt)
                after = runner.robot.data.root_lin_vel_w.clone()
                push_evidence = {
                    "time_s": push["time_s"],
                    "dv_xy": push["dv_xy"],
                    "measured_dv_xy": [float(after[0, 0] - before[0, 0]), float(after[0, 1] - before[0, 1])],
                    "measured_dv_yaw": float(after[0, 5] - before[0, 5]),
                    "mechanism": "write_root_velocity_to_sim（水平速度冲击，未换算为牛顿外力）",
                }
                push_done = True
            runner.policy_step({})
            heights.append(float(runner.robot.data.root_pos_w[0, 2]))
            speeds.append(float(torch.linalg.norm(runner.robot.data.root_lin_vel_w[0, :2])))
        return {
            "n_policy_steps": n_steps,
            "mean_height": sum(heights) / len(heights),
            "mean_speed": sum(speeds) / len(speeds),
            "final_speed": speeds[-1],
            "push": push_evidence,
            "fingerprint": [round(value, 6) for value in heights[-5:] + speeds[-5:]],
        }

    # ---------------- P1/P2 写入与读回 ----------------
    runner.reset_all()
    friction_report = apply_friction(runner, PROFILE_PLAN["friction_only"]["friction"])
    check(
        checks, "P1-friction",
        "摩擦写入 0.3 后逐 shape 读回一致（仅 friction 条件）",
        friction_report["max_abs_diff"] < 1e-6
        and all(abs(value - 0.3) < 1e-6 for row in friction_report["readback_static"] for value in row),
        f"{friction_report['n_shapes']} 个 shape，读回 static 全 0.3，最大偏差 {friction_report['max_abs_diff']:.2e}",
    )
    mass_report = apply_mass(runner, PROFILE_PLAN["mass_only"]["added_mass_kg"])
    total_shift = mass_report["total_after"][0] - mass_report["total_before"][0]
    check(
        checks, "P2-mass",
        "base 加 3 kg 后读回一致，总质量同步 +3 kg",
        mass_report["max_abs_diff"] < 1e-6 and abs(total_shift - 3.0) < 1e-4,
        f"{mass_report['link_name']}: {mass_report['base_before'][0]:.3f}→{mass_report['base_after'][0]:.3f} kg，"
        f"总质量 {mass_report['total_before'][0]:.3f}→{mass_report['total_after'][0]:.3f}（Δ={total_shift:.4f}）",
    )

    # ---------------- P7 reset 回到基准 ----------------
    restored = restore_baseline(runner, baseline_state)
    runner.reset_all()
    restored_mass = float(runner.robot.root_physx_view.get_masses().sum())
    restored_friction = float(runner.robot.root_physx_view.get_material_properties()[0, 0, 0])
    check(
        checks, "P7-restore",
        "恢复 clean 后质量、惯量与材质读回基准值（reset 不吃旧 profile 残留）",
        abs(restored_mass - baseline_mass) < 1e-4
        and abs(restored_friction - baseline_friction) < 1e-6
        and restored["mass_max_abs_diff"] < 1e-6
        and restored["material_max_abs_diff"] < 1e-6,
        f"质量 {restored_mass:.4f}（基准 {baseline_mass:.4f}），摩擦 {restored_friction:.3f}（基准 {baseline_friction:.3f}），"
        f"恢复残差 mass={restored['mass_max_abs_diff']:.2e} material={restored['material_max_abs_diff']:.2e}",
    )

    # ---------------- P5 clean 确定性 ----------------
    runner.reset_all()
    clean_a = run_seconds(args_cli.seconds)
    runner.reset_all()
    clean_b = run_seconds(args_cli.seconds)
    check(
        checks, "P5-clean-determinism",
        "clean 两次运行轨迹逐位一致",
        clean_a["fingerprint"] == clean_b["fingerprint"],
        f"末 5 步高度/速度指纹一致={clean_a['fingerprint'] == clean_b['fingerprint']}",
    )

    # ---------------- P6 各条件轨迹与 clean 不同 ----------------
    trajectories = {"clean": clean_a}
    for profile_id in ("friction_only", "mass_only", "push_only"):
        plan = PROFILE_PLAN[profile_id]
        apply_friction(runner, plan["friction"])
        apply_mass(runner, plan["added_mass_kg"])
        runner.reset_all()
        trajectories[profile_id] = run_seconds(args_cli.seconds, push=plan["push"])
        log(f"{profile_id}: mean_speed={trajectories[profile_id]['mean_speed']:.3f} "
            f"height={trajectories[profile_id]['mean_height']:.3f}")
        # 复位旋钮（按基准快照精确恢复），避免串到下一个条件
        restore_baseline(runner, baseline_state)

    diffs = {
        profile_id: abs(trajectories[profile_id]["mean_height"] - clean_a["mean_height"])
        + abs(trajectories[profile_id]["mean_speed"] - clean_a["mean_speed"])
        for profile_id in ("friction_only", "mass_only", "push_only")
    }
    check(
        checks, "P6-perturbation-applied",
        "摩擦/质量/推扰三个条件的轨迹都与 clean 不同（证明真的施加了）",
        all(value > 1e-4 for value in diffs.values()),
        "；".join(f"{key}: Δ={value:.3e}" for key, value in diffs.items()),
    )

    # ---------------- P3 推扰是速度冲击 ----------------
    push_trial = trajectories["push_only"]["push"]
    push_ok = (
        push_trial is not None
        and abs(push_trial["measured_dv_xy"][0] - 0.5) < 1e-3
        and abs(push_trial["measured_dv_xy"][1] - 0.0) < 1e-3
        and abs(push_trial["measured_dv_yaw"]) < 1e-6
    )
    check(
        checks, "P3-push-velocity",
        "推扰写入后一步内水平速度增量等于设定 Δv，yaw 不受影响",
        push_ok,
        f"Δv 实测={push_trial['measured_dv_xy']}（设定 [0.5, 0.0]），Δyaw={push_trial['measured_dv_yaw']:.2e}"
        if push_trial else "推扰未触发",
    )

    # ---------------- P4 观测噪声只落 9:33 ----------------
    from hira_go2.observation import build_noise_vector

    runner.reset_all()
    noise_vector = build_noise_vector(*NOISE_SCALES)
    runner.obs_adapter.noise_scale = noise_vector.to(args_cli.device).reshape(1, 45)
    obs_clean = runner.obs_adapter.build(
        *runner_observation_inputs(runner)
    )["obs"]
    obs_noisy = runner.obs_adapter.build(*runner_observation_inputs(runner), include_noise=True)["obs"]
    delta = (obs_noisy - obs_clean).abs()
    block_delta = {
        "0:9": float(delta[:, :9].max()),
        "9:21": float(delta[:, 9:21].max()),
        "21:33": float(delta[:, 21:33].max()),
        "33:45": float(delta[:, 33:].max()),
    }
    noise_ok = (
        block_delta["0:9"] == 0.0
        and block_delta["33:45"] == 0.0
        and 0.0 < block_delta["9:21"] <= NOISE_SCALES[0]
        and 0.0 < block_delta["21:33"] <= NOISE_SCALES[1]
    )
    runner.obs_adapter.noise_scale = None
    check(
        checks, "P4-obs-noise",
        "噪声只落 9:21/21:33，幅度不超过源 noise_scale；其余块为 0",
        noise_ok,
        f"逐块最大变化 {block_delta}（上限 dof_pos {NOISE_SCALES[0]}、dof_vel {NOISE_SCALES[1]}）",
    )

    # ---------------- P8 跨 env 不污染 ----------------
    multi = HiraLabRunner(
        args_cli.urdf, args_cli.usd_dir, num_envs=2, device=args_cli.device,
        policy_path=args_cli.policy, timing=timing,
    )
    multi_before = multi.robot.root_physx_view.get_masses().clone()
    multi_friction_before = multi.robot.root_physx_view.get_material_properties().clone()
    apply_mass(multi, 3.0, env_ids=[0])
    apply_friction(multi, 0.3, env_ids=[0])
    multi_after = multi.robot.root_physx_view.get_masses()
    multi_friction_after = multi.robot.root_physx_view.get_material_properties()
    env0_changed = abs(float(multi_after[0].sum() - multi_before[0].sum()) - 3.0) < 1e-4
    env1_same = bool(torch.allclose(multi_after[1], multi_before[1], atol=1e-6)) and bool(
        torch.allclose(multi_friction_after[1], multi_friction_before[1], atol=1e-6)
    )
    check(
        checks, "P8-cross-env",
        "只改 env0 的质量与摩擦，env1 完全不变",
        env0_changed and env1_same,
        f"env0 质量 Δ={float(multi_after[0].sum() - multi_before[0].sum()):.4f}，env1 不变={env1_same}",
    )

    failed = [item["id"] for item in checks if not item["ok"]]
    report = {
        "report": "profile_effectiveness",
        "task": "HIRA-018",
        "device": args_cli.device,
        "seconds_per_condition": args_cli.seconds,
        "profile_plan": PROFILE_PLAN,
        "protocol_profiles": {key: defined[key] for key in PROFILE_PLAN},
        "baseline": {"total_mass": baseline_mass, "friction": baseline_friction},
        "readback": {"friction": friction_report, "mass": mass_report},
        "trajectories": trajectories,
        "obs_noise_blocks": block_delta,
        "checks": checks,
        "verdict": {"all_passed": not failed, "failed": failed, "n_checks": len(checks)},
    }
    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for item in checks:
        print(f"[{'ok ' if item['ok'] else 'FAIL'}] {item['id']:22s} {item['name']} :: {item['detail']}")
    print("PROFILE_OK" if not failed else f"PROFILE_FAIL {failed}")
    return 0 if not failed else 1


def runner_observation_inputs(runner):
    """取当前状态的观测输入（同一状态喂两次，才能把差异全部归给噪声）。"""
    from hira_go2.observation import body_angular_velocity_wxyz, projected_gravity_wxyz

    quat = runner.robot.data.root_quat_w
    return (
        body_angular_velocity_wxyz(quat, runner.robot.data.root_ang_vel_w),
        projected_gravity_wxyz(quat),
        runner.command,
        runner.robot.data.joint_pos,
        runner.robot.data.joint_vel,
    )


if __name__ == "__main__":
    exit_code = 2
    try:
        exit_code = main()
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("PROFILE_ERROR", flush=True)
        exit_code = 2
    finally:
        log("hard exit (skip simulation_app.close)")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
