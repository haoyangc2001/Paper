"""闭环异常归因诊断（HIRA-017 附属）。

背景：clean 闭环里机身高度掉到 ~0.25 m（初始 0.42 m），速度跟踪也偏弱。
需要先把"PD 跟不住"与"策略主动蹲下"分开——前者是接入问题，后者是策略能力。

判据：
  * 若 |q - q_target| 很小、τ 未饱和 → PD 正常履行，姿态是策略给的 → 能力问题
  * 若 |q - q_target| 很大或 τ 长期饱和 → 增益/限幅/映射问题 → 接入问题

    python tests/diag_posture.py --headless --device cuda:0 ...
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

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default=str(REPO / "HiRA/legged_gym_go2/resources/robots/go2/urdf/go2.urdf"))
parser.add_argument("--usd-dir", default="/pub/data/caohy/logs/camra_hira/assets/usd_cache")
parser.add_argument("--policy", default=str(EXP / "exported/policy.pt"))
parser.add_argument("--out", default=str(EXP / "reports/posture_diagnosis.json"))
parser.add_argument("--seconds", type=float, default=5.0)
parser.add_argument("--order-mode", choices=("urdf", "lab"), default="urdf")
parser.add_argument("--warmup-steps", type=int, default=200)
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

T0 = time.time()


def trace_tau_abs(runner) -> list[float]:
    trace = runner.torque_adapter.last_trace
    return [abs(float(v)) for v in trace["tau_clipped_source"].reshape(-1).tolist()]


def log(text: str) -> None:
    print(f"[posture] {time.time() - T0:7.1f}s {text}", flush=True)


def main() -> int:
    from hira_go2.env_runner import HiraLabRunner, TimingConfig

    timing = TimingConfig(window_s=args_cli.seconds)
    runner = HiraLabRunner(
        args_cli.urdf, args_cli.usd_dir, num_envs=1, device=args_cli.device,
        policy_path=args_cli.policy, timing=timing, order_mode=args_cli.order_mode,
    )
    runner.set_command(torch.zeros(1, 3))
    # warmup 用零动作保持默认姿态；如果 warmup 太长，机器人在策略接管前就已经塌下去，
    # 之后评估的其实是"从塌陷状态恢复"，而不是源的"从初始姿态起步"。
    runner.warmup(args_cli.warmup_steps)
    log(f"warmup done ({args_cli.warmup_steps} steps)")

    default_source = runner.default_source.reshape(-1)
    log(f"order_mode={args_cli.order_mode}")
    # 关节限位（源顺序），用于区分"PD 跟不上"与"目标本身越限"
    limits = runner.robot.data.joint_pos_limits[0]
    upper_source = runner.order.target_to_source(limits[:, 1])
    lower_source = runner.order.target_to_source(limits[:, 0])
    effort_source = runner.torque_adapter.effort_limit_source.reshape(-1)
    n_steps = int(round(args_cli.seconds / timing.policy_dt))
    track_errors: list[float] = []
    inside_errors: list[float] = []
    outside_counts: list[int] = []
    tau_per_joint: list[list[float]] = []
    tau_abs: list[float] = []
    heights: list[float] = []
    joint_rows = []

    for _ in range(n_steps):
        record: dict = {}
        runner.policy_step(record)
        q_target = torch.tensor(record["q_target_row0"], device=args_cli.device, dtype=runner.dtype)
        # q 在 Lab（USD）顺序，q_target 在源顺序：必须按名字换到同一顺序再比，
        # 否则比出来的是"错位差"（第一版就因此得到 3.5 rad 的假误差）
        q = runner.order.target_to_source(runner.robot.data.joint_pos)[0]
        err = (q - q_target).abs()
        inside = (q_target >= lower_source) & (q_target <= upper_source)
        track_errors.append(float(err.max()))
        if inside.any():
            inside_errors.append(float(err[inside].max()))
        outside_counts.append(int((~inside).sum()))
        tau_per_joint.append(trace_tau_abs(runner))
        tau_abs.append(float(record["tau_max_abs"]))
        heights.append(float(record["height_row0"]))
        joint_rows.append(
            {
                "q_err_max": float(err.max()),
                "q_err_mean": float(err.mean()),
                "q_offset_from_default": [round(float(v), 4) for v in (q - default_source).tolist()],
            }
        )

    last = joint_rows[-1]
    per_joint = []
    for index, name in enumerate(summary_order := [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]):
        per_joint.append(
            {
                "joint": name,
                "mean_abs_tau": float(sum(row[index] for row in tau_per_joint) / len(tau_per_joint)),
                "effort_limit": float(effort_source[index]),
                "limit_lower": float(lower_source[index]),
                "limit_upper": float(upper_source[index]),
                "default_angle": float(default_source[index]),
            }
        )

    summary = {
        "report": "posture_diagnosis",
        "task": "HIRA-017-diagnostic",
        "device": args_cli.device,
        "order_mode": args_cli.order_mode,
        "warmup_steps": args_cli.warmup_steps,
        "seconds": args_cli.seconds,
        "n_policy_steps": n_steps,
        "height_start": float(heights[0]),
        "height_end": float(heights[-1]),
        "height_min": float(min(heights)),
        "max_track_error": float(max(track_errors)),
        "mean_track_error": float(sum(track_errors) / len(track_errors)),
        "max_track_error_inside_limits": float(max(inside_errors)) if inside_errors else None,
        "mean_targets_outside_limits": float(sum(outside_counts) / len(outside_counts)),
        "per_joint": per_joint,
        "max_tau": float(max(tau_abs)),
        "final_joint_offset_from_default": last["q_offset_from_default"],
        "default_angles_source_order": [round(float(v), 4) for v in default_source.tolist()],
        "source_order": [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ],
    }
    inside_err = summary["max_track_error_inside_limits"]
    summary["verdict"] = {
        "pd_tracks_well_inside_limits": bool(inside_err is not None and inside_err < 0.15),
        "tau_saturated": summary["max_tau"] >= 35.5,
        "mean_targets_outside_limits": summary["mean_targets_outside_limits"],
        "interpretation": (
            "限位内的目标 PD 能跟上；大误差来自目标越限 → 策略主动给出越限姿态（能力/后端差异）"
            if (inside_err is not None and inside_err < 0.15)
            else "限位内也存在大误差 → 优先怀疑增益/限幅/映射"
        ),
    }

    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"height {summary['height_start']:.3f}→{summary['height_end']:.3f} "
        f"track_err max={summary['max_track_error']:.4f} mean={summary['mean_track_error']:.4f} "
        f"tau max={summary['max_tau']:.2f}")
    log(f"joint offset from default: {summary['final_joint_offset_from_default']}")
    print("POSTURE_DIAG_DONE", flush=True)
    return 0


if __name__ == "__main__":
    exit_code = 2
    try:
        exit_code = main()
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("POSTURE_DIAG_ERROR", flush=True)
        exit_code = 2
    finally:
        log("hard exit (skip simulation_app.close)")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
