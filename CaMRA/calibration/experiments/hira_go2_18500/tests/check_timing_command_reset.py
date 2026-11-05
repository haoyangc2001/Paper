"""三时钟、外部命令与逐 env 终止重置验证（HIRA-016）。

一个 Isaac Lab 进程里只能有一份场景（`InteractiveScene` 的 prim 路径固定为
`/World/envs/env_*`，第二个场景会撞 `A prim already exists`），
所以本脚本分两次运行、再把两份结果合并：

    # 1) 单 env：warmup、30 s 窗口、命令保真、q_target、命令覆盖负测试
    python tests/check_timing_command_reset.py --phase main ...

    # 2) 4 env：逐 env 终止、跨 env 隔离
    python tests/check_timing_command_reset.py --phase termination ...

    # 3) 合并（纯 python，不需要 Isaac）
    python tests/check_timing_command_reset.py --phase merge ...

验证的是**实际计数与时间戳**，不是配置里写了多少。
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
parser.add_argument("--phase", choices=("main", "termination", "merge"), default="main")
parser.add_argument("--urdf", default=str(REPO / "HiRA/legged_gym_go2/resources/robots/go2/urdf/go2.urdf"))
parser.add_argument("--usd-dir", default="/pub/data/caohy/logs/camra_hira/assets/usd_cache")
parser.add_argument("--policy", default=str(EXP / "exported/policy.pt"))
parser.add_argument("--out", default=str(EXP / "reports/timing_command_reset.json"))
parser.add_argument("--trace", default=str(EXP / "reports/timing_command_trace.jsonl"))
parser.add_argument("--window", type=float, default=30.0)

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
    # merge 阶段不需要 torch / Isaac，也就在没有 torch 的环境里能跑
    import torch  # noqa: E402

T0 = time.time()


def log(text: str) -> None:
    print(f"[timing] {time.time() - T0:7.1f}s {text}", flush=True)


def deltas_of(records: list[dict]) -> set[int]:
    return {b["physics_step"] - a["physics_step"] for a, b in zip(records, records[1:])}


# --------------------------------------------------------------------------- #
def phase_main(args, checks: list[dict]) -> dict:
    from hira_go2.env_runner import HiraLabRunner, TimingConfig

    timing = TimingConfig(window_s=args.window)
    log(f"config window={timing.window_s}s sim_dt={timing.sim_dt} policy_dt={timing.policy_dt} "
        f"command_dt={timing.command_dt}")

    runner = HiraLabRunner(
        args.urdf, args.usd_dir, num_envs=1, device=args.device,
        policy_path=args.policy, timing=timing,
    )
    log("runner ready")

    runner.warmup()
    warmup_physics = runner.physics_step
    warmup_policy = runner.policy_step_count

    commands = [
        [0.0, 0.0, 0.0],
        [0.3, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.3, 0.0],
        [0.0, 0.0, 0.5],
    ]
    runner.run_window(commands)
    log("window done")
    trace = runner.trace
    n_policy = len(trace.policy_steps)
    expected_policy = timing.window_policy_steps
    expected_physics = timing.window_physics_steps
    expected_intervals = timing.window_command_intervals

    def add(check_id, name, ok, detail):
        checks.append({"id": check_id, "name": name, "ok": bool(ok), "detail": detail})

    add(
        "C1-warmup",
        "warmup 只推进 200 个物理步，不产生策略步",
        warmup_physics == timing.warmup_physics_steps and warmup_policy == 0,
        f"physics_step={warmup_physics}（期望 {timing.warmup_physics_steps}），policy_step={warmup_policy}",
    )
    add(
        "C2-counters",
        "6000 物理步 / 1500 策略步 / 300 命令区间 / 301 配对点",
        (runner.physics_step - warmup_physics) == expected_physics
        and runner.policy_step_count == expected_policy
        and n_policy == expected_policy
        and runner.command_index == expected_intervals
        and timing.paired_points == expected_intervals + 1,
        f"physics={runner.physics_step - warmup_physics}/{expected_physics}，"
        f"policy={runner.policy_step_count}/{expected_policy}，intervals={runner.command_index}/{expected_intervals}，"
        f"paired_points={timing.paired_points}",
    )
    deltas = deltas_of(trace.policy_steps)
    add(
        "C3-decimation",
        "相邻策略步之间恰好 4 个物理子步",
        deltas == {timing.decimation},
        f"实测步距集合={sorted(deltas)}（期望 {{{timing.decimation}}}）",
    )

    per_interval: dict[int, int] = {}
    for record in trace.policy_steps:
        per_interval[record["command_index"]] = per_interval.get(record["command_index"], 0) + 1
    interval_counts = sorted(set(per_interval.values()))
    add(
        "C4-command-interval",
        "每个命令区间恰好 5 次重新观测与推理",
        interval_counts == [timing.command_policy_steps] and len(per_interval) == expected_intervals,
        f"区间数={len(per_interval)}，每区间策略步数集合={interval_counts}",
    )

    fidelity_max = 0.0
    deadband_violations = []
    for record in trace.policy_steps:
        vx, vy, wz = record["command_row0"]
        expected = [2.0 * vx, 2.0 * vy, 0.25 * wz]
        actual = record["obs_row0"][6:9]
        fidelity_max = max(fidelity_max, max(abs(a - e) for a, e in zip(actual, expected)))
        if abs(vx - 0.1) < 1e-6 and abs(actual[0] - 0.2) > 1e-6:
            deadband_violations.append(record["policy_step"])
    sample = next(
        (record["obs_row0"][6] for record in trace.policy_steps if abs(record["command_row0"][0] - 0.1) < 1e-6),
        float("nan"),
    )
    add(
        "C5-command-fidelity",
        "命令块 = [2vx, 2vy, 0.25wz]，0.1 m/s 未被清零、vy 未被吞掉",
        fidelity_max <= 1e-6 and not deadband_violations,
        f"最大偏差={fidelity_max:.3e}，低速候选违规={len(deadband_violations)}，"
        f"cmd=[0.1,0,0] → obs[6]={sample:.3f}",
    )

    default_source = runner.default_source.reshape(-1).tolist()
    qt_max = 0.0
    for record in trace.policy_steps:
        for index in range(12):
            expected = default_source[index] + 0.25 * record["clipped_action_row0"][index]
            qt_max = max(qt_max, abs(record["q_target_row0"][index] - expected))
    add(
        "C6-q-target",
        "窗口内 q_target 恒等于 default + 0.25·clipped_action",
        qt_max <= 1e-6,
        f"最大偏差={qt_max:.3e}",
    )

    # N2 旧命令覆盖：写入新命令后必须立刻生效（复用同一场景）
    runner.set_command(torch.tensor([[0.5, 0.0, 0.0]]))
    first_obs = float(runner.observe()[0, 6])
    runner.set_command(torch.tensor([[0.2, 0.0, 0.0]]))
    second_obs = float(runner.observe()[0, 6])
    add(
        "N2-command-override",
        "新命令在一个策略步内生效（0.5→0.2 对应 obs 1.0→0.4），无内部锁定或重采样",
        abs(first_obs - 1.0) < 1e-6 and abs(second_obs - 0.4) < 1e-6,
        f"obs[6]: {first_obs:.3f} → {second_obs:.3f}（期望 1.0 → 0.4）",
    )

    # N1 错的 decimation：合成一条步距为 3 的轨迹，检查器必须报错
    synthetic = [{"physics_step": 3 * index, "policy_step": index} for index in range(10)]
    add(
        "N1-wrong-decimation",
        "decimation 写成 3 时步距检查会报错（检查器有效性）",
        deltas_of(synthetic) == {3} and deltas_of(synthetic) != {4},
        "合成轨迹步距集合={3}，与期望 {4} 不符 → 检查器判失败。"
        "说明：真实重建 decimation=3 的场景需要第二个 InteractiveScene，"
        "而 prim 路径（/World/envs/env_*）不允许同进程存在两份场景",
    )

    trace_path = Path(args.trace)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as handle:
        for record in trace.policy_steps:
            slim = {
                key: record[key]
                for key in (
                    "policy_step", "physics_step", "sim_time", "command_index",
                    "command_row0", "tau_max_abs", "height_row0", "tilt_deg_row0",
                )
            }
            slim["obs_command_block"] = record["obs_row0"][6:9]
            handle.write(json.dumps(slim, ensure_ascii=False) + "\n")

    return {
        "timing": {
            "sim_dt": timing.sim_dt,
            "policy_dt": timing.policy_dt,
            "command_dt": timing.command_dt,
            "decimation": timing.decimation,
            "command_policy_steps": timing.command_policy_steps,
            "window_s": timing.window_s,
            "warmup_physics_steps": timing.warmup_physics_steps,
            "paired_points": timing.paired_points,
        },
        "counters": {
            "warmup_physics_steps": warmup_physics,
            "window_physics_steps": runner.physics_step - warmup_physics,
            "policy_steps": runner.policy_step_count,
            "command_intervals": runner.command_index,
            "paired_points": timing.paired_points,
            "terminations_in_window": len(trace.terminations),
        },
        "command_sequence": commands,
        "device": args.device,
    }


# --------------------------------------------------------------------------- #
def phase_termination(args, checks: list[dict]) -> dict:
    from hira_go2.env_runner import HiraLabRunner, TimingConfig

    def add(check_id, name, ok, detail):
        checks.append({"id": check_id, "name": name, "ok": bool(ok), "detail": detail})

    timing = TimingConfig(window_s=args.window)
    runner = HiraLabRunner(
        args.urdf, args.usd_dir, num_envs=4, device=args.device,
        policy_path=args.policy, timing=timing,
    )
    log("multi-env runner ready")
    runner.set_command(torch.tensor([[0.2, 0.0, 0.0]]).repeat(4, 1))
    for _ in range(6):
        runner.policy_step({})

    def force_low(env_id: int, height: float) -> None:
        pose = torch.zeros((1, 7), device=args.device, dtype=runner.dtype)
        pose[:, 0:2] = runner.robot.data.root_pos_w[env_id : env_id + 1, 0:2]
        pose[:, 2] = height
        pose[:, 3] = 1.0
        runner.robot.write_root_pose_to_sim(pose, env_ids=torch.tensor([env_id], device=args.device))

    # 顺序很重要：先跑完这一策略步（所有 env 都会更新上一动作），
    # 再抓快照，最后才重置。否则会把"新动作写入"误当成"跨 env 污染"。
    force_low(0, 0.08)
    runner.policy_step({})
    before_prev = runner.obs_adapter.prev_action.clone()
    before_cmd = runner.command.clone()
    done_ids = runner.handle_terminations()
    runner.robot.update(timing.sim_dt)
    after_prev = runner.obs_adapter.prev_action.clone()
    after_cmd = runner.command.clone()
    heights = runner.robot.data.root_pos_w[:, 2]

    captured = [record for record in runner.trace.terminations if record.env_id == 0]
    add(
        "C7-termination-capture",
        "env0 摔倒时先抓到终止状态再重置（高度 < 0.15 m）",
        bool(captured) and captured[0].root_height < 0.15 and int(done_ids.numel()) >= 1,
        f"抓到 {len(captured)} 条；root_height={captured[0].root_height:.3f}，tilt={captured[0].tilt_deg:.1f}°"
        if captured else "未抓到终止记录",
    )
    others_stable = all(
        abs(float(after_prev[i].abs().max()) - float(before_prev[i].abs().max())) < 1e-9 for i in (1, 2, 3)
    )
    add(
        "C8-per-env-reset",
        "只重置 env0：其上一动作清零、root 回 0.42，其他 env 的上一动作与命令不变",
        float(after_prev[0].abs().max()) == 0.0
        and others_stable
        and bool(torch.equal(after_cmd, before_cmd))
        and abs(float(heights[0]) - 0.42) < 0.05,
        f"env0 prev_abs {float(before_prev[0].abs().max()):.3f}→{float(after_prev[0].abs().max()):.3f}，"
        f"env1..3 不变={others_stable}，命令不变={bool(torch.equal(after_cmd, before_cmd))}，"
        f"env0 高度={float(heights[0]):.3f}",
    )

    # N3 跨 env 污染：不同命令 + 真的重置一次 env0
    runner.set_command(torch.tensor([[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2], [-0.2, 0.0, 0.0]]))
    force_low(0, 0.08)
    runner.policy_step({})
    prev_before = runner.obs_adapter.prev_action.clone()
    obs_before = runner.observe().clone()
    reset_ids = runner.handle_terminations()
    obs_after = runner.observe().clone()
    prev_after = runner.obs_adapter.prev_action.clone()
    stable = bool(torch.allclose(obs_after[1:, 6:9], obs_before[1:, 6:9], atol=1e-9)) and bool(
        torch.equal(prev_after[1:], prev_before[1:])
    )
    add(
        "N3-cross-env",
        "重置 env0（确实发生）不改变其他 env 的命令块与上一动作",
        int(reset_ids.numel()) >= 1 and stable,
        f"本次重置 env 数={int(reset_ids.numel())}，env1..3 命令块最大变化="
        f"{float((obs_after[1:, 6:9] - obs_before[1:, 6:9]).abs().max()):.3e}，"
        f"上一动作不变={bool(torch.equal(prev_after[1:], prev_before[1:]))}",
    )

    return {
        "terminations": [
            {
                "env_id": record.env_id,
                "physics_step": record.physics_step,
                "policy_step": record.policy_step,
                "command_index": record.command_index,
                "sim_time": record.sim_time,
                "root_height": record.root_height,
                "tilt_deg": record.tilt_deg,
                "command": record.command,
            }
            for record in runner.trace.terminations
        ],
        "device": args.device,
    }


# --------------------------------------------------------------------------- #
def phase_merge(args) -> int:
    out = Path(args.out)
    main_path = out.with_suffix(".main.json")
    term_path = out.with_suffix(".termination.json")
    payload = json.loads(main_path.read_text(encoding="utf-8"))
    termination = json.loads(term_path.read_text(encoding="utf-8"))
    checks = payload["checks"] + termination["checks"]
    failed = [item["id"] for item in checks if not item["ok"]]
    report = {
        "report": "timing_command_reset",
        "task": "HIRA-016",
        "device": payload.get("device"),
        "timing": payload["timing"],
        "counters": payload["counters"],
        "command_sequence": payload["command_sequence"],
        "terminations": termination["terminations"],
        "checks": checks,
        "trace_file": str(Path(args.trace)),
        "verdict": {"all_passed": not failed, "failed": failed, "n_checks": len(checks)},
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for item in checks:
        print(f"[{'ok ' if item['ok'] else 'FAIL'}] {item['id']:24s} {item['name']} :: {item['detail']}")
    print(f"[merge] checks={len(checks)} failed={failed}")
    print("TIMING_OK" if not failed else f"TIMING_FAIL {failed}")
    return 0 if not failed else 1


# --------------------------------------------------------------------------- #
def main() -> int:
    checks: list[dict] = []
    if args_cli.phase == "merge":
        return phase_merge(args_cli)

    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args_cli.phase == "main":
        payload = phase_main(args_cli, checks)
        target = out.with_suffix(".main.json")
    else:
        payload = phase_termination(args_cli, checks)
        target = out.with_suffix(".termination.json")

    failed = [item["id"] for item in checks if not item["ok"]]
    payload.update(
        {
            "report": f"timing_command_reset_{args_cli.phase}",
            "task": "HIRA-016",
            "phase": args_cli.phase,
            "checks": checks,
            "verdict": {"all_passed": not failed, "failed": failed, "n_checks": len(checks)},
        }
    )
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for item in checks:
        print(f"[{'ok ' if item['ok'] else 'FAIL'}] {item['id']:24s} {item['name']} :: {item['detail']}")
    print(f"[{args_cli.phase}] checks={len(checks)} failed={failed}")
    print("PHASE_OK" if not failed else f"PHASE_FAIL {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    exit_code = 2
    try:
        exit_code = main()
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("TIMING_ERROR", flush=True)
        exit_code = 2
    finally:
        if args_cli.phase != "merge":
            log("hard exit (skip simulation_app.close)")
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
        sys.exit(exit_code)
