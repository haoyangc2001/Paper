"""T-08 批量隔离：num_envs=1 与 32 必须给出同一条轨迹，且 env 之间无耦合（HIRA-020）。

预登记的判据（`integration_acceptance_prereg_v1.json#T-08`）：

    同命令同 seed，num_envs=1 与 32 各一次 → 批量结果与单 env 一致（τ 轨迹 atol 1e-4）；
    env 之间无耦合。

为什么它可能不成立：Isaac Lab 的 env 共享同一个场景与同一块地面，若某个量被误写成
"全局"（例如按 env0 的命令驱动所有 env、或把速度限幅写成场景级），32 个 env 就会
互相拉扯，而单 env 检查（HIRA-016/017）永远发现不了。

一个进程只能有一个 InteractiveScene，所以单 env 与 32 env 必须分进程跑，再合并比较。
比较时位置要减去各自的 `env_origins`——32 env 是按网格摆的，不减去会得到虚假的巨大差异。

    python tests/check_batch_isolation.py --headless --device cuda:0 --phase single
    python tests/check_batch_isolation.py --headless --device cuda:0 --phase batch
    python tests/check_batch_isolation.py --phase merge
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
ADAPTER = REPO / "CaMRA" / "locomotion" / "adapters"
CAL_SRC = REPO / "CaMRA" / "calibration" / "src"
for path in (str(ADAPTER), str(CAL_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

parser = argparse.ArgumentParser()
parser.add_argument("--phase", choices=("single", "batch", "merge"), default="single")
parser.add_argument("--envs", type=int, default=32)
parser.add_argument("--policy-steps", type=int, default=250)
parser.add_argument("--command", default="0.5,0,0")
parser.add_argument("--command-others", default="", help="给 env1..N-1 另一条命令，用于解耦判定")
parser.add_argument("--urdf", default=str(REPO / "HiRA/legged_gym_go2/resources/robots/go2/urdf/go2.urdf"))
parser.add_argument("--usd-dir", default="/pub/data/caohy/logs/camra_hira/assets/usd_cache")
parser.add_argument("--policy", default=str(EXP / "exported/policy.pt"))
parser.add_argument("--friction", type=float, default=1.0)
parser.add_argument("--atol-tau", type=float, default=1e-4)
parser.add_argument("--enhanced-determinism", action="store_true")
parser.add_argument("--origin-offset", default="0,0", help="只有单 env 对照用：把出生点平移一段距离")
parser.add_argument("--tag", default="")
parser.add_argument("--out-dir", default=str(EXP / "reports"))

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

T0 = time.time()


def log(text: str) -> None:
    print(f"[batch] {time.time() - T0:7.1f}s {text}", flush=True)


def run(num_envs: int, tag: str) -> dict:
    from camra_calibration.energy import DEFAULT_DT
    from hira_go2.env_runner import HiraLabRunner, TimingConfig

    timing = TimingConfig(sim_dt=DEFAULT_DT)
    runner = HiraLabRunner(
        args_cli.urdf,
        args_cli.usd_dir,
        num_envs=num_envs,
        device=args_cli.device,
        policy_path=args_cli.policy,
        timing=timing,
        enhanced_determinism=args_cli.enhanced_determinism,
    )
    view = runner.robot.root_physx_view
    material = view.get_material_properties().clone()
    material[:, :, 0] = args_cli.friction
    material[:, :, 1] = args_cli.friction
    view.set_material_properties(material, torch.arange(num_envs, dtype=torch.int64))
    log(f"{tag}: friction → {float(view.get_material_properties()[0, 0, 0]):.2f}, envs={num_envs}")

    command = [float(value) for value in args_cli.command.split(",")]
    offset = [float(value) for value in args_cli.origin_offset.split(",")]
    if any(abs(value) > 0 for value in offset):
        # 控制实验：把整个场景的出生点平移，物理上完全等价；若结果因此变化，
        # 说明"批内差异"其实是世界坐标舍入 + 混沌放大，而不是 env 之间的耦合。
        runner.env_origins = runner.env_origins + torch.tensor(
            [[offset[0], offset[1], 0.0]], device=runner.device, dtype=runner.dtype
        )
    runner.reset_all()
    cmd = torch.tensor([command], dtype=runner.dtype).repeat(num_envs, 1)
    others = [float(value) for value in args_cli.command_others.split(",")] if args_cli.command_others else None
    if others is not None and num_envs > 1:
        # 解耦判定：让其余 env 走完全相反的方向。若 env0 的轨迹仍然逐位不变，
        # 说明 env 之间在动力学上相互独立（这比"逐位相等"更能证明没有耦合）。
        cmd[1:] = torch.tensor([others], dtype=runner.dtype)
    runner.set_command(cmd)

    tau = np.zeros((args_cli.policy_steps, num_envs, 12), dtype=np.float32)
    pos = np.zeros((args_cli.policy_steps, num_envs, 3), dtype=np.float32)
    jpos = np.zeros((args_cli.policy_steps, num_envs, 12), dtype=np.float32)
    for index in range(args_cli.policy_steps):
        runner.policy_step()
        tau[index] = runner.order.target_to_source(runner.robot.data.applied_torque).detach().cpu().numpy()
        pos[index] = runner.robot.data.root_pos_w.detach().cpu().numpy()
        jpos[index] = runner.order.target_to_source(runner.robot.data.joint_pos).detach().cpu().numpy()

    origins = runner.env_origins.detach().cpu().numpy()
    pos_rel = pos.copy()
    pos_rel[:, :, 0] -= origins[None, :, 0]
    pos_rel[:, :, 1] -= origins[None, :, 1]

    out = Path(args_cli.out_dir) / f"batch_isolation_{tag}.npz"
    np.savez_compressed(out, tau=tau, pos_rel=pos_rel, jpos=jpos, origins=origins)
    payload = {
        "task": "HIRA-020/T-08",
        "tag": tag,
        "num_envs": num_envs,
        "policy_steps": args_cli.policy_steps,
        "command": command,
        "command_others": others,
        "friction": args_cli.friction,
        "array_file": str(out),
        "tau_shape": list(tau.shape),
        "final_root_height": pos[-1, :, 2].tolist(),
        "min_root_height": pos[:, :, 2].min(axis=0).tolist(),
        "max_tau_abs": float(np.abs(tau).max()),
        "enhanced_determinism": args_cli.enhanced_determinism,
        "origin_offset": offset,
    }
    (Path(args_cli.out_dir) / f"batch_isolation_{tag}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    log(f"{tag}: wrote {out} tau{tau.shape} max|τ|={payload['max_tau_abs']:.3f}")
    return payload


RUN_NOTES = {
    "single": "1 env，默认求解器（与 HIRA-016/017 相同设置）",
    "det_single": "1 env，enhanced determinism 打开",
    "single_at_env0": "1 env，先把出生点平移到 (6,-4)：与 32 env 批里 env0 的世界坐标相同",
    "single_shift": "1 env，出生点平移到 (2,2)：物理等价、只是换了世界坐标",
    "batch": "32 env，默认求解器",
    "det_batch": "32 env，enhanced determinism 打开",
    "decoupled": "32 env，env0 前进 [0.5,0,0]、其余 31 个 env 反向 [-0.5,0,0]",
}


def _load(out_dir: Path, tag: str):
    npz = out_dir / f"batch_isolation_{tag}.npz"
    meta = out_dir / f"batch_isolation_{tag}.json"
    if not npz.is_file() or not meta.is_file():
        return None, None
    return np.load(npz), json.loads(meta.read_text(encoding="utf-8"))


def _tau(z, env=0):
    return z["tau"][:, env, :].astype(np.float64)


def _curve(a, b) -> dict:
    d = np.abs(a - b).max(axis=1)
    first = int(np.argmax(d > 1e-4)) if (d > 1e-4).any() else -1
    return {
        "step0": float(d[0]),
        "step1": float(d[1]),
        "step3": float(d[min(3, d.size - 1)]),
        "step10": float(d[min(10, d.size - 1)]),
        "max": float(d.max()),
        "first_step_over_1e-4": first if first >= 0 else None,
        "series_head": [float(value) for value in d[:12]],
    }


def _behavior(z) -> dict:
    """末段 40 个策略步的速度 / 高度 / 平均力矩：用来做「行为等价」而不是「逐位等价」。"""
    pos = z["pos_rel"].astype(np.float64)
    tau = z["tau"].astype(np.float64)
    dt = 0.02
    vx = ((pos[-40:, :, 0] - pos[-41:-1, :, 0]) / dt).mean(axis=0)
    height = pos[:, :, 2].mean(axis=0)
    tau_mean = np.abs(tau).mean(axis=(0, 2))
    return {"vx": vx, "height": height, "tau_mean": tau_mean}


def merge() -> int:
    out_dir = Path(args_cli.out_dir)
    runs, missing = {}, []
    for tag in RUN_NOTES:
        z, meta = _load(out_dir, tag)
        if z is None:
            missing.append(tag)
            continue
        runs[tag] = {"npz": z, "meta": meta}
    if missing:
        raise SystemExit(f"缺少这些 run：{missing}")

    single = runs["single"]["npz"]
    batch = runs["det_batch"]["npz"]
    checks: list[dict] = []

    # ---- 1) 解耦判定（同一批内，只改其他 env 的命令）----
    dec_curve = _curve(_tau(batch), _tau(runs["decoupled"]["npz"]))
    other_curve = _curve(_tau(batch, 1), _tau(runs["decoupled"]["npz"], 1))
    checks.append(
        {
            "id": "T08-decoupling",
            "name": "env0 的轨迹不受其他 env 命令影响（解耦判定）",
            "ok": dec_curve["max"] == 0.0 and other_curve["max"] > 1e-3,
            "detail": f"32 env 里把其余 31 个 env 的命令由 +0.5 改成 −0.5：env0 的 τ 逐元素 max|Δ|={dec_curve['max']!r}"
            f"（要求逐位不变），同一时刻 env1 的 τ 变化 max|Δ|={other_curve['max']:.3f}"
            "（证明这次改动确实生效了，不是空操作）",
            "max_abs_diff_env0": dec_curve["max"],
            "max_abs_diff_env1": other_curve["max"],
        }
    )

    # ---- 2) 预登记原判据：1 env 与 32 env 的 τ 轨迹 ----
    traj = _curve(_tau(single), _tau(batch))
    checks.append(
        {
            "id": "T08-trajectory-prereg",
            "name": "预登记原判据：1 env 与 32 env 的 τ 轨迹一致（atol 1e-4）",
            "ok": traj["max"] <= args_cli.atol_tau,
            "detail": f"max|Δ|={traj['max']:.3f} N·m（判据 {args_cli.atol_tau:g}）；"
            f"首步已差 {traj['step0']:.2e}，第 {traj['first_step_over_1e-4']} 个策略步越过 1e-4，之后指数放大",
            "curve": traj,
        }
    )
    matched = _curve(_tau(runs["single_at_env0"]["npz"]), _tau(batch))
    checks.append(
        {
            "id": "T08-trajectory-matched-origin",
            "name": "把单 env 放到与 env0 相同的世界坐标后仍不一致",
            "ok": matched["max"] <= args_cli.atol_tau,
            "detail": f"1 env@(6,-4) vs 32 env 的 env0：max|Δ|={matched['max']:.3f}；"
            f"首步 {matched['step0']:.2e}。说明差异不是「位置没对齐」造成的，而是求解器对 1 个 vs 32 个刚体并行求解的舍入差异",
            "curve": matched,
        }
    )

    # ---- 3) 对照：单个 env 只挪动出生点，同样发散 ----
    shift = _curve(_tau(runs["det_single"]["npz"]), _tau(runs["single_shift"]["npz"]))
    checks.append(
        {
            "id": "T08-control-translation",
            "name": "对照实验：只有一个 env、只把出生点平移 2 m，轨迹同样发散",
            "ok": True,  # 这是一条诊断，不是判据
            "is_criterion": False,
            "detail": f"物理完全等价，仅世界坐标 (0,0)→(2,2)：max|Δ|={shift['max']:.3f} N·m，"
            f"首步 {shift['step0']:.2e}、越过 1e-4 于第 {shift['first_step_over_1e-4']} 个策略步——"
            "与第 2 项同一量级，说明「批内/跨批差异」来自坐标相关舍入 + 接触混沌放大，不是 env 间耦合",
            "curve": shift,
        }
    )

    # ---- 4) 行为等价（distribution 层面）----
    b_beh = _behavior(batch)
    s_beh = _behavior(single)
    zscores = {}
    for key in ("vx", "height", "tau_mean"):
        sigma = float(np.std(b_beh[key]))
        zscores[key] = float((s_beh[key][0] - float(np.mean(b_beh[key]))) / max(sigma, 1e-12))
    zmax = max(abs(value) for value in zscores.values())
    checks.append(
        {
            "id": "T08-behavioral-equivalence",
            "name": "单 env 与批量的行为统计一致（落在批内散布之内）",
            "ok": zmax <= 3.0,
            "detail": f"末段 40 步：单 env vx={s_beh['vx'][0]:.4f}，32 env {b_beh['vx'].mean():.4f}±{b_beh['vx'].std():.4f}；"
            f"高度 {s_beh['height'][0]:.4f} vs {b_beh['height'].mean():.4f}±{b_beh['height'].std():.4f}；"
            f"平均|τ| {s_beh['tau_mean'][0]:.3f} vs {b_beh['tau_mean'].mean():.3f}±{b_beh['tau_mean'].std():.3f}；"
            f"最大偏离 {zmax:.2f}σ（判据 3σ）",
            "z_scores": zscores,
        }
    )

    # ---- 5) enhanced determinism 是否改变结果 ----
    mode = _curve(_tau(single), _tau(runs["det_single"]["npz"]))
    checks.append(
        {
            "id": "T08-solver-mode",
            "name": "诊断：enhanced determinism 对同一批大小是否改变结果",
            "ok": True,
            "is_criterion": False,
            "detail": f"1 env 默认 vs 1 env 增强确定性：max|Δ|={mode['max']!r}（逐位相同）→ 该开关在本场景是空操作，"
            "不足以把 1 env 与 32 env 对齐",
            "curve": mode,
        }
    )

    criteria = [item for item in checks if item.get("is_criterion", True)]
    failed = [item["id"] for item in criteria if not item["ok"]]
    report = {
        "report": "batch_isolation",
        "task": "HIRA-020",
        "prereg_row": "T-08",
        "runs": {
            tag: {
                "note": RUN_NOTES[tag],
                "num_envs": runs[tag]["meta"]["num_envs"],
                "command": runs[tag]["meta"]["command"],
                "command_others": runs[tag]["meta"].get("command_others"),
                "origin_offset": runs[tag]["meta"].get("origin_offset"),
                "enhanced_determinism": runs[tag]["meta"].get("enhanced_determinism"),
                "max_tau_abs": runs[tag]["meta"]["max_tau_abs"],
                "array_file": runs[tag]["meta"]["array_file"],
            }
            for tag in RUN_NOTES
        },
        "checks": checks,
        "verdict": {
            "criteria_passed": not failed,
            "failed": failed,
            "decoupling_exact": dec_curve["max"] == 0.0,
            "trajectory_bitwise_attainable": traj["max"] <= args_cli.atol_tau,
            "diagnosis": "1 env 与 32 env 的差异是求解器舍入 + 接触混沌放大，不是 env 间耦合；"
            "单 env 只平移出生点即可复现同量级发散（T08-control-translation）。"
            "预登记 T-08 的「τ 轨迹 atol 1e-4」在 GPU PhysX 上不可达，需协议裁定。",
        },
    }
    out = out_dir / "batch_isolation.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for item in checks:
        mark = "ok  " if item["ok"] else "FAIL"
        print(f"[{mark}] {item['id']} {item['name']}\n        {item['detail']}", flush=True)
    print(f"BATCH_VERDICT criteria_passed={not failed} failed={failed}", flush=True)
    return 0 if not failed else 1


def main() -> int:
    if args_cli.phase == "merge":
        return merge()
    # env 数量只由 --phase 决定；--tag 只是文件名后缀。
    # 之前写成 "tag=='single' → 1 env"，一旦自定义 tag 就会把单 env 跑成 32 env（已修）。
    tag = args_cli.tag or args_cli.phase
    envs = 1 if args_cli.phase == "single" else args_cli.envs
    run(envs, tag)
    print(f"BATCH_OK tag={tag}", flush=True)
    return 0


if __name__ == "__main__":
    code = 2
    try:
        code = main()
    except SystemExit as exc:
        code = int(exc.code or 0)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("BATCH_ERROR", flush=True)
        code = 2
    finally:
        log("hard exit (skip simulation_app.close)")
        sys.stdout.flush()
        sys.stderr.flush()
        import os

        os._exit(code)
