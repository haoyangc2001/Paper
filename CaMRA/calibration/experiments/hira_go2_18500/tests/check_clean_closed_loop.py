"""clean 工况闭环评估（HIRA-017）。

按预登记的 11 条命令 × 3 个固定种子 = 33 条、每条 30 s。判据全部来自
`config/integration_acceptance_prereg_v1.json` 的 `closed_loop_sequence.behavioral_thresholds`：

- fall：root 高度 < 0.15 m 或姿态 > 60°
- direction：C-02..C-07 的平均机身速度符号必须与命令一致
- tracking_mid_speed：中速工况平均速度误差 ≤ 0.25 m/s
- saturation_rate：力矩饱和步占比 ≤ 30%
- stability：30 s 内无摔倒、无非有限值

分片运行（一个进程只能有一个场景，且 33 条串行太慢）：

    python tests/check_clean_closed_loop.py --phase run --shard 0 --num-shards 2 --device cuda:0 ...
    python tests/check_clean_closed_loop.py --phase merge

**注意**：clean 工况是确定性仿真（无噪声、无随机化），所以三个种子本应给出完全相同的轨迹。
报告会把这一点直接验证并写出来——相同轨迹既是"无隐藏随机化"的证据，
也意味着重复不是"独立样本"，统计上不能当 33 个独立点用。
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
parser.add_argument("--phase", choices=("run", "merge"), default="run")
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--num-shards", type=int, default=1)
parser.add_argument("--urdf", default=str(REPO / "HiRA/legged_gym_go2/resources/robots/go2/urdf/go2.urdf"))
parser.add_argument("--usd-dir", default="/pub/data/caohy/logs/camra_hira/assets/usd_cache")
parser.add_argument("--policy", default=str(EXP / "exported/policy.pt"))
parser.add_argument("--prereg", default=str(EXP / "config/integration_acceptance_prereg_v1.json"))
parser.add_argument("--out-dir", default=str(EXP / "reports"))
parser.add_argument("--window", type=float, default=30.0)
parser.add_argument("--warmup-steps", type=int, default=0)

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
#: 每工况固定 3 个 seed，登记在报告里；不因结果重跑。
SEEDS = (11, 22, 33)


def log(text: str) -> None:
    print(f"[loop] {time.time() - T0:7.1f}s {text}", flush=True)


def euler_roll_pitch(quat: "torch.Tensor") -> tuple[float, float]:
    """wxyz → (roll, pitch) 角度（度）。"""
    w, x, y, z = [float(v) for v in quat]
    roll = math.degrees(math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))
    return roll, pitch


def run_trial(runner, command: list[float], seed: int, window_s: float, body_vel_fn, warmup_steps: int = 0) -> dict:
    """一条 30 s 恒定命令的 trial。"""
    runner.reset_all()
    # 源的行为是 reset 后立刻由策略接管；warmup 只用于时钟测试。
    # 这里默认 0：否则机器人会在策略接管前先"零动作下垂"，评估就变成从塌陷姿态恢复。
    if warmup_steps:
        runner.warmup(warmup_steps)
    runner.set_command(torch.tensor([command], dtype=runner.dtype))

    samples = []
    n_intervals = int(round(window_s / runner.timing.command_dt))
    for index in range(n_intervals):
        runner.command_index = index
        for _ in range(runner.timing.command_policy_steps):
            record: dict = {}
            runner.policy_step(record)
            runner.handle_terminations()
            body_vx, body_vy, body_wz = body_vel_fn()
            roll, pitch = euler_roll_pitch(runner.robot.data.root_quat_w[0])
            samples.append(
                {
                    "t": record["sim_time"],
                    "vx": body_vx,
                    "vy": body_vy,
                    "wz": body_wz,
                    "roll": roll,
                    "pitch": pitch,
                    "height": record["height_row0"],
                    "tilt": record["tilt_deg_row0"],
                    "tau_max": record["tau_max_abs"],
                    # 饱和按"任一关节在任一子步被裁"计，而不是拿最大力矩跟全局上限比：
                    # hip 的 23.7 远低于 calf 的 35.55，用全局上限会漏掉 hip 饱和
                    "saturated": record["n_saturated_joints"] > 0,
                }
            )
    return {
        "command": command,
        "seed": seed,
        "samples": samples,
        "terminations": len(runner.trace.terminations),
    }


def summarize(trial: dict, thresholds: dict) -> dict:
    samples = trial["samples"]
    command = trial["command"]
    n = len(samples)
    finite = all(
        math.isfinite(item[key]) for item in samples for key in ("vx", "vy", "wz", "roll", "pitch", "height", "tau_max")
    )
    mean_vx = sum(item["vx"] for item in samples) / n
    mean_vy = sum(item["vy"] for item in samples) / n
    mean_wz = sum(item["wz"] for item in samples) / n
    rmse_vx = math.sqrt(sum((item["vx"] - command[0]) ** 2 for item in samples) / n)
    rmse_vy = math.sqrt(sum((item["vy"] - command[1]) ** 2 for item in samples) / n)
    rmse_wz = math.sqrt(sum((item["wz"] - command[2]) ** 2 for item in samples) / n)
    min_height = min(item["height"] for item in samples)
    max_tilt = max(item["tilt"] for item in samples)
    max_abs_roll = max(abs(item["roll"]) for item in samples)
    max_abs_pitch = max(abs(item["pitch"]) for item in samples)
    saturation_fraction = sum(1 for item in samples if item["saturated"]) / n
    speed_error = math.hypot(mean_vx - command[0], mean_vy - command[1])

    # 方向：命令主轴上平均速度符号必须一致（源符号约定：都为正相关）
    direction_ok = None
    axis = None
    if abs(command[0]) > 1e-9 and abs(command[0]) >= abs(command[1]):
        axis, measured, wanted = "vx", mean_vx, command[0]
        direction_ok = (measured * wanted) > 0
    elif abs(command[1]) > 1e-9:
        axis, measured, wanted = "vy", mean_vy, command[1]
        direction_ok = (measured * wanted) > 0
    elif abs(command[2]) > 1e-9:
        axis, measured, wanted = "wz", mean_wz, command[2]
        direction_ok = (measured * wanted) > 0

    fall = trial["terminations"] > 0 or min_height < thresholds["fall_height"] or max_tilt > thresholds["fall_tilt_deg"]
    return {
        "command": command,
        "seed": trial["seed"],
        "n_samples": n,
        "mean_vx": mean_vx,
        "mean_vy": mean_vy,
        "mean_wz": mean_wz,
        "rmse_vx": rmse_vx,
        "rmse_vy": rmse_vy,
        "rmse_wz": rmse_wz,
        "speed_error": speed_error,
        "min_height": min_height,
        "max_tilt_deg": max_tilt,
        "max_abs_roll_deg": max_abs_roll,
        "max_abs_pitch_deg": max_abs_pitch,
        "max_tau": max(item["tau_max"] for item in samples),
        "saturation_fraction": saturation_fraction,
        "terminations": trial["terminations"],
        "finite": finite,
        "fall": fall,
        "direction_axis": axis,
        "direction_ok": direction_ok,
    }


def main() -> int:
    outp = Path(args_cli.out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    if args_cli.phase == "merge":
        return merge(outp, args_cli)

    from hira_go2.env_runner import HiraLabRunner, TimingConfig
    from hira_go2.observation import body_angular_velocity_wxyz, quat_rotate_inverse_wxyz

    prereg = json.loads(Path(args_cli.prereg).read_text(encoding="utf-8"))
    sequence = prereg["closed_loop_sequence"]
    commands = sequence["commands"]
    thresholds = {
        "fall_height": 0.15,
        "fall_tilt_deg": 60.0,
        "tracking_mid_speed": 0.25,
        "saturation_rate": 0.30,
    }

    trials = [(cmd, seed) for cmd in commands for seed in SEEDS]
    mine = [trial for index, trial in enumerate(trials) if index % args_cli.num_shards == args_cli.shard]
    log(f"shard {args_cli.shard}/{args_cli.num_shards}: {len(mine)} trials on {args_cli.device}")

    timing = TimingConfig(window_s=args_cli.window)
    runner = HiraLabRunner(
        args_cli.urdf, args_cli.usd_dir, num_envs=1, device=args_cli.device,
        policy_path=args_cli.policy, timing=timing,
    )
    log("runner ready")

    def body_velocity() -> tuple[float, float, float]:
        quat = runner.robot.data.root_quat_w
        lin = runner.robot.data.root_lin_vel_w
        ang = runner.robot.data.root_ang_vel_w
        v_body = quat_rotate_inverse_wxyz(quat, lin)
        w_body = body_angular_velocity_wxyz(quat, ang)
        return float(v_body[0, 0]), float(v_body[0, 1]), float(w_body[0, 2])

    results = []
    for cmd, seed in mine:
        torch.manual_seed(seed)
        trial = run_trial(runner, cmd["cmd"], seed, args_cli.window, body_velocity, args_cli.warmup_steps)
        summary = summarize(trial, thresholds)
        summary["id"] = cmd["id"]
        summary["label"] = cmd["label"]
        results.append(summary)
        log(
            f"{cmd['id']} seed={seed} v=({summary['mean_vx']:+.3f},{summary['mean_vy']:+.3f},"
            f"{summary['mean_wz']:+.3f}) err={summary['speed_error']:.3f} "
            f"h_min={summary['min_height']:.3f} sat={summary['saturation_fraction']:.2%} "
            f"term={summary['terminations']} fall={summary['fall']}"
        )

    shard_payload = {
        "phase": "clean_closed_loop_shard",
        "task": "HIRA-017",
        "shard": args_cli.shard,
        "num_shards": args_cli.num_shards,
        "device": args_cli.device,
        "window_s": args_cli.window,
        "warmup_steps": args_cli.warmup_steps,
        "seeds": list(SEEDS),
        "thresholds": thresholds,
        "trials": results,
    }
    path = outp / f"clean_closed_loop.shard{args_cli.shard}.json"
    path.write_text(json.dumps(shard_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {path}")
    print("SHARD_OK", flush=True)
    return 0


def merge(outp: Path, args) -> int:
    """合并分片、按预登记判据判定，并检查重复 trial 是否逐位一致。"""
    prereg = json.loads(Path(args.prereg).read_text(encoding="utf-8"))
    commands = {item["id"]: item for item in prereg["closed_loop_sequence"]["commands"]}
    shards = sorted(outp.glob("clean_closed_loop.shard*.json"))
    if not shards:
        raise SystemExit("没有分片结果")
    trials = []
    for path in shards:
        trials.extend(json.loads(path.read_text(encoding="utf-8"))["trials"])
    trials.sort(key=lambda item: (item["id"], item["seed"]))

    # 重复一致性：同一条命令的三个种子应当给出完全相同的轨迹（clean 无随机化）
    determinism = {}
    for cmd_id in commands:
        group = [trial for trial in trials if trial["id"] == cmd_id]
        if len(group) < 2:
            continue
        base = group[0]
        same = all(
            abs(trial["mean_vx"] - base["mean_vx"]) < 1e-12
            and abs(trial["mean_vy"] - base["mean_vy"]) < 1e-12
            and abs(trial["mean_wz"] - base["mean_wz"]) < 1e-12
            and abs(trial["min_height"] - base["min_height"]) < 1e-12
            for trial in group[1:]
        )
        determinism[cmd_id] = {"n": len(group), "identical": same}

    mid_speed_ids = {"C-02", "C-03", "C-04", "C-05", "C-06", "C-07", "C-08"}
    direction_ids = {"C-02", "C-03", "C-04", "C-05", "C-06", "C-07"}
    low_speed_ids = {"C-09", "C-10", "C-11"}

    per_command = {}
    for cmd_id, cmd in commands.items():
        group = [trial for trial in trials if trial["id"] == cmd_id]
        if not group:
            continue
        checks = {
            "stability": all(trial["finite"] and not trial["fall"] for trial in group),
            "saturation": all(trial["saturation_fraction"] <= 0.30 for trial in group),
            "direction": all(trial["direction_ok"] for trial in group) if cmd_id in direction_ids else None,
            "tracking": all(trial["speed_error"] <= 0.25 for trial in group) if cmd_id in mid_speed_ids else None,
        }
        enforced = {key: value for key, value in checks.items() if value is not None}
        per_command[cmd_id] = {
            "label": cmd["label"],
            "cmd": cmd["cmd"],
            "domain": "low_speed" if cmd_id in low_speed_ids else "nominal",
            "checks": checks,
            "passed": all(enforced.values()),
            "trials": group,
        }

    # 低速域：单列，不参与"通过/不通过"，但记录事实
    blocking = {
        cmd_id: info
        for cmd_id, info in per_command.items()
        if info["domain"] == "nominal" and not info["passed"]
    }
    report = {
        "report": "clean_closed_loop",
        "task": "HIRA-017",
        "window_s": prereg["closed_loop_sequence"]["window_s"],
        "seeds": list(SEEDS),
        "n_trials": len(trials),
        "thresholds": {
            "fall": "root 高度 < 0.15 m 或倾角 > 60°",
            "direction": "C-02..C-07 平均机身速度符号与命令一致",
            "tracking_mid_speed": "中速工况平均速度误差 ≤ 0.25 m/s",
            "saturation_rate": "力矩饱和步占比 ≤ 30%",
            "stability": "30 s 内无摔倒、无非有限值",
        },
        "determinism": determinism,
        "per_command": per_command,
        "low_speed_note": "C-09..C-11 按预登记单列为能力边界：跟踪差不作为接入错误，也不得删除该命令域",
        "blocking_failures": sorted(blocking.keys()),
        "verdict": {
            "all_nominal_passed": not blocking,
            "blocking_failures": sorted(blocking.keys()),
            "low_speed_failures": sorted(
                cmd_id for cmd_id, info in per_command.items()
                if info["domain"] == "low_speed" and not info["passed"]
            ),
            "deterministic_repeats": all(item["identical"] for item in determinism.values()) if determinism else None,
        },
    }
    out = outp / "clean_closed_loop.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for cmd_id, info in per_command.items():
        first = info["trials"][0]
        print(
            f"[{'ok ' if info['passed'] else 'FAIL'}] {cmd_id} {info['label']:6s} "
            f"cmd={info['cmd']} v=({first['mean_vx']:+.3f},{first['mean_vy']:+.3f},{first['mean_wz']:+.3f}) "
            f"err={first['speed_error']:.3f} h_min={first['min_height']:.3f} "
            f"sat={first['saturation_fraction']:.1%} fall={first['fall']}"
        )
    print(f"[merge] trials={len(trials)} blocking={report['verdict']['blocking_failures']}")
    print("CLEAN_LOOP_OK" if report["verdict"]["all_nominal_passed"] else "CLEAN_LOOP_FAIL")
    return 0 if report["verdict"]["all_nominal_passed"] else 1


if __name__ == "__main__":
    exit_code = 2
    try:
        exit_code = main()
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("LOOP_ERROR", flush=True)
        exit_code = 2
    finally:
        if args_cli.phase != "merge":
            log("hard exit (skip simulation_app.close)")
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
        sys.exit(exit_code)
