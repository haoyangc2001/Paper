"""C-08 归因扫描：前向速度是否随 yaw 指令耦合升高（HIRA-017 附属）。

C-08 = [0.4, 0, 0.3] 实测 vx=0.660（命令 0.4，误差 0.288 > 0.25）。
若 vx 随 wz 单调上升，说明是"转向时步态变快"的策略耦合，属能力问题；
若 [0.4,0,0] 本身就偏离，则要怀疑接入侧的速度语义。

    python tests/diag_wz_coupling.py --headless --device cuda:0 ...
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
parser.add_argument("--out", default=str(EXP / "reports/wz_coupling.json"))
parser.add_argument("--seconds", type=float, default=20.0)
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

T0 = time.time()
COMMANDS = [
    [0.4, 0.0, 0.0],
    [0.4, 0.0, 0.15],
    [0.4, 0.0, 0.3],
    [0.4, 0.0, 0.45],
    [0.2, 0.0, 0.3],
]


def log(text: str) -> None:
    print(f"[wz] {time.time() - T0:7.1f}s {text}", flush=True)


def main() -> int:
    from hira_go2.env_runner import HiraLabRunner, TimingConfig
    from hira_go2.observation import body_angular_velocity_wxyz, quat_rotate_inverse_wxyz

    timing = TimingConfig(window_s=args_cli.seconds)
    runner = HiraLabRunner(
        args_cli.urdf, args_cli.usd_dir, num_envs=1, device=args_cli.device,
        policy_path=args_cli.policy, timing=timing,
    )
    log("runner ready")

    def body_velocity():
        quat = runner.robot.data.root_quat_w
        lin = quat_rotate_inverse_wxyz(quat, runner.robot.data.root_lin_vel_w)
        ang = body_angular_velocity_wxyz(quat, runner.robot.data.root_ang_vel_w)
        return float(lin[0, 0]), float(lin[0, 1]), float(ang[0, 2])

    results = []
    for command in COMMANDS:
        runner.reset_all()
        runner.set_command(torch.tensor([command], dtype=runner.dtype))
        n_steps = int(round(args_cli.seconds / timing.policy_dt))
        vx, vy, wz, heights = [], [], [], []
        for _ in range(n_steps):
            runner.policy_step({})
            x, y, w = body_velocity()
            vx.append(x)
            vy.append(y)
            wz.append(w)
            heights.append(float(runner.robot.data.root_pos_w[0, 2]))
        row = {
            "command": command,
            "mean_vx": sum(vx) / len(vx),
            "mean_vy": sum(vy) / len(vy),
            "mean_wz": sum(wz) / len(wz),
            "rmse_vx": (sum((value - command[0]) ** 2 for value in vx) / len(vx)) ** 0.5,
            "min_height": min(heights),
        }
        results.append(row)
        log(f"cmd={command} v=({row['mean_vx']:+.3f},{row['mean_vy']:+.3f},{row['mean_wz']:+.3f}) "
            f"rmse_vx={row['rmse_vx']:.3f} h_min={row['min_height']:.3f}")

    vz = [row for row in results if abs(row["command"][0] - 0.4) < 1e-9]
    monotone = all(
        vz[index + 1]["mean_vx"] >= vz[index]["mean_vx"] - 1e-3 for index in range(len(vz) - 1)
    )
    payload = {
        "report": "wz_coupling",
        "task": "HIRA-017-diagnostic",
        "device": args_cli.device,
        "seconds": args_cli.seconds,
        "rows": results,
        "vx_increases_with_wz": monotone,
        "interpretation": (
            "vx 随 wz 单调升高 → 转向时步态加速，属策略行为（能力），不是映射错误"
            if monotone
            else "vx 与 wz 无明显单调关系 → 需要继续排查接入侧"
        ),
    }
    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"monotone={monotone}")
    print("WZ_SCAN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    exit_code = 2
    try:
        exit_code = main()
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("WZ_SCAN_ERROR", flush=True)
        exit_code = 2
    finally:
        log("hard exit (skip simulation_app.close)")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
