"""补一个"源摩擦下的 C-08"对照（HIRA-017/018 交叉检查）。

HIRA-017 的闭环是在**导入默认摩擦 0.5** 下跑的；源（isaacgym/URDF）默认是 1.0。
HIRA-018 才发现这个差异，所以这里在 friction=1.0 下重测 C-02（对照）与 C-08（争议项），
看 C-08 的超速是否与摩擦有关。

    python tests/diag_clean_friction.py --headless --device cuda:0 ...
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
parser.add_argument("--out", default=str(EXP / "reports/clean_friction_check.json"))
parser.add_argument("--seconds", type=float, default=30.0)
parser.add_argument("--friction", type=float, default=1.0)
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

T0 = time.time()
COMMANDS = [
    {"id": "C-02", "cmd": [0.5, 0.0, 0.0]},
    {"id": "C-08", "cmd": [0.4, 0.0, 0.3]},
]


def log(text: str) -> None:
    print(f"[fric] {time.time() - T0:7.1f}s {text}", flush=True)


def main() -> int:
    from hira_go2.env_runner import HiraLabRunner, TimingConfig
    from hira_go2.observation import body_angular_velocity_wxyz, quat_rotate_inverse_wxyz

    timing = TimingConfig(window_s=args_cli.seconds)
    runner = HiraLabRunner(
        args_cli.urdf, args_cli.usd_dir, num_envs=1, device=args_cli.device,
        policy_path=args_cli.policy, timing=timing,
    )
    view = runner.robot.root_physx_view
    material = view.get_material_properties().clone()
    as_imported = float(material[0, 0, 0])
    material[:, :, 0] = args_cli.friction
    material[:, :, 1] = args_cli.friction
    view.set_material_properties(material, torch.tensor([0], dtype=torch.int64))
    log(f"friction {as_imported:.2f} → {float(view.get_material_properties()[0, 0, 0]):.2f}")

    def body_velocity():
        quat = runner.robot.data.root_quat_w
        lin = quat_rotate_inverse_wxyz(quat, runner.robot.data.root_lin_vel_w)
        ang = body_angular_velocity_wxyz(quat, runner.robot.data.root_ang_vel_w)
        return float(lin[0, 0]), float(lin[0, 1]), float(ang[0, 2])

    rows = []
    for item in COMMANDS:
        runner.reset_all()
        runner.set_command(torch.tensor([item["cmd"]], dtype=runner.dtype))
        n_steps = int(round(args_cli.seconds / timing.policy_dt))
        values = []
        for _ in range(n_steps):
            runner.policy_step({})
            values.append(body_velocity())
        mean_vx = sum(v[0] for v in values) / len(values)
        mean_vy = sum(v[1] for v in values) / len(values)
        mean_wz = sum(v[2] for v in values) / len(values)
        error = ((mean_vx - item["cmd"][0]) ** 2 + (mean_vy - item["cmd"][1]) ** 2) ** 0.5
        rows.append(
            {
                "id": item["id"],
                "cmd": item["cmd"],
                "mean_vx": mean_vx,
                "mean_vy": mean_vy,
                "mean_wz": mean_wz,
                "speed_error": error,
                "passed_tracking": error <= 0.25,
            }
        )
        log(f"{item['id']} cmd={item['cmd']} v=({mean_vx:+.3f},{mean_vy:+.3f},{mean_wz:+.3f}) "
            f"err={error:.3f} pass={error <= 0.25}")

    payload = {
        "report": "clean_friction_check",
        "task": "HIRA-017/018 cross-check",
        "device": args_cli.device,
        "friction_as_imported": as_imported,
        "friction_used": args_cli.friction,
        "seconds": args_cli.seconds,
        "rows": rows,
        "note": "HIRA-017 的 33 条是在导入默认摩擦 0.5 下跑的；这里在源值 1.0 下重测两条命令作对照",
    }
    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FRICTION_CHECK_DONE", flush=True)
    return 0


if __name__ == "__main__":
    exit_code = 2
    try:
        exit_code = main()
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("FRICTION_CHECK_ERROR", flush=True)
        exit_code = 2
    finally:
        log("hard exit (skip simulation_app.close)")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
