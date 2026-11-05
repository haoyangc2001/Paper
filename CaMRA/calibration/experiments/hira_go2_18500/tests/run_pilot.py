"""HIRA-022：20 条五条件 pilot 的采集入口（一个进程跑完全部 5 批）。

清单 = 4 个命令族（straight_shift / arc_combined / alternating / start_stop_turn，
与 v2 同一套 `camra_calibration.commands.build_program` 生成器）× 5 个 profile，
共 20 条；seed 用 6110..6129（pilot 专属区间，与后续 split 互斥）。

分批规则（HIRA-021 的约束）：同批 obs_noise / window_s / warmup_s 必须一致 →
按 profile 分 5 批，每批 4 条（一个 env 一条）。

支持断点续跑：先 plan_attempts，DONE 完整的 trial 直接 skip。

    python tests/run_pilot.py --headless --device cuda:0 --run-id pilot_<ts>
    python tests/check_pilot_results.py --phase merge   # 之后独立验证
"""

from __future__ import annotations

import argparse
import json
import resource
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
parser.add_argument("--run-id", default=None, help="缺省 = pilot_<UTC时间戳>")
parser.add_argument("--runs-root", default="/pub/data/caohy/logs/camra_hira/runs")
parser.add_argument("--bundle", default=str(EXP / "frozen/hira_go2_18500"))
parser.add_argument("--prereg", default=str(EXP / "config/integration_acceptance_prereg_v1.json"))
parser.add_argument("--usd-dir", default="/pub/data/caohy/logs/camra_hira/assets/usd_cache")
# --device 由 AppLauncher 提供
parser.add_argument("--seed-base", type=int, default=6110, help="pilot 专属 seed 区间起点")
parser.add_argument("--window", type=float, default=30.0)

_PRE = parser.parse_known_args()[0]
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from camra_calibration.commands import build_program  # noqa: E402
from hira_go2.bundle import load_bundle  # noqa: E402
from hira_go2.collector import HiraCollector, TrialSpec, plan_attempts, sample_spec  # noqa: E402

T0 = time.time()

FAMILIES = ("straight_shift", "arc_combined", "alternating", "start_stop_turn")
PROFILES = ("clean", "friction_only", "mass_only", "push_only", "combined")


def log(text: str) -> None:
    print(f"[pilot] {time.time() - T0:7.1f}s {text}", flush=True)


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main() -> int:
    run_id = args_cli.run_id or f"pilot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    run_dir = Path(args_cli.runs_root) / run_id
    if run_dir.exists():
        print(f"run 目录已存在，按断点续跑处理：{run_dir}", flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(Path(args_cli.bundle), expected_controller_id=Path(args_cli.bundle).name)
    config_path = run_dir / "collect_config.json"
    if not config_path.is_file():
        config_path.write_text(
            json.dumps(
                {
                    "window_s": args_cli.window,
                    "usd_dir": args_cli.usd_dir,
                    "warmup_s": 1.0,
                    "seed_range_pilot": [args_cli.seed_base, args_cli.seed_base + 19],
                    "seed_rule": "pilot 专属区间 [6110,6129]；后续 split 必须避开（HIRA-024 登记互斥表）",
                    "families": list(FAMILIES),
                    "profiles": list(PROFILES),
                    "program_generator": "camra_calibration.commands.build_program（与 v2 同一代码路径）",
                },
                ensure_ascii=False,
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )

    # ---- 生成 20 条 spec（命令表显式存进 spec）----
    specs: list[TrialSpec] = []
    index = 0
    for family in FAMILIES:
        for profile in PROFILES:
            seed = args_cli.seed_base + index
            program = build_program(family, seed, duration_s=args_cli.window)
            spec = sample_spec(
                trial_id=f"hira-pilot-{family}-{profile}-0000",
                split="pilot", program=family, profile=profile, seed=seed,
                commands=[list(c) for c in program.commands],
                window_s=args_cli.window,
            )
            specs.append(spec)
            index += 1
    log(f"{len(specs)} 条 pilot spec 就绪（seed {args_cli.seed_base}..{args_cli.seed_base + 19}）")

    collector = HiraCollector(
        args_cli.bundle, run_dir, args_cli.prereg, config_path,
        num_envs=len(FAMILIES), device=args_cli.device,
        expected_joints={"target_order": bundle["controller"]["joints"]["target_order"]},
    )
    log(f"collector ready: controller={collector.controller_id} bundle={collector.bundle_id[:16]}")

    # ---- 按 profile 分批（同批 obs_noise 必须一致）----
    batch_stats = []
    for profile in PROFILES:
        batch = [s for s in specs if s.profile == profile]
        plan = {item["trial_id"]: item for item in plan_attempts(run_dir, batch)}
        attempts = []
        runnable = []
        for spec in batch:
            item = plan[spec.trial_id]
            if item["action"] == "skip":
                attempts.append(None)
                log(f"skip {spec.trial_id}（DONE 完整）")
            elif item["action"] == "blocked_business_failure":
                attempts.append(None)
                log(f"blocked {spec.trial_id}（业务失败，不自动重试）")
            else:
                attempts.append(item["attempt"])
                runnable.append((spec, item["attempt"]))
        # execute_batch 需要整批 env；把已完成的槽位用"占位 spec"跑空不划算——
        # 直接只跑未完成的：按 runnable 数量重建小批（num_envs 固定为 4，
        # 已完成的槽位复用同批第一條 runnable spec 的干净副本是错的。
        # 简化：本批若有 skip，就用占位 TrialSpec 填满 env 槽位，但不写 DONE
        # （占位跑出来的数据进一个 scratch 目录，事后删除）。
        if not runnable:
            log(f"profile {profile}: 全部已完成/阻断，跳过")
            continue
        # 填满槽位：占位 = runnable[0] 的拷贝，trial_id 改成 scratch
        fill = list(runnable)
        scratch_used = []
        while len(fill) < collector.num_envs:
            spec0, attempt0 = runnable[0]
            scratch = TrialSpec(
                trial_id=f"hira-scratch-{profile}-{len(fill)}", split="scratch",
                program=spec0.program, profile=spec0.profile, seed=spec0.seed + 7000,
                commands=spec0.commands, window_s=spec0.window_s, warmup_s=spec0.warmup_s,
                friction=spec0.friction, added_mass_kg=spec0.added_mass_kg,
                obs_noise=spec0.obs_noise, initial_pose=spec0.initial_pose,
                push_events=spec0.push_events,
            )
            fill.append((scratch, 1))
            scratch_used.append(scratch.trial_id)
        run_specs = [item[0] for item in fill]
        run_attempts = [item[1] for item in fill]
        t0 = time.time()
        collector.execute_batch(run_specs, run_attempts)
        wall = time.time() - t0
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        sizes = {
            s.trial_id: dir_size(run_dir / s.trial_id)
            for s in run_specs if s.trial_id not in scratch_used
        }
        batch_stats.append(
            {"profile": profile, "n_trials": len(runnable), "scratch_slots": scratch_used,
             "wall_s": round(wall, 1), "wall_per_trial_s": round(wall / len(runnable), 1),
             "rss_max_mb": round(rss_mb, 1),
             "dir_bytes": sizes}
        )
        log(f"profile {profile}: {len(runnable)} 条，wall {wall:.1f}s，RSS {rss_mb:.0f} MB")
        if scratch_used:
            for trial_id in scratch_used:
                shutil.rmtree(run_dir / trial_id, ignore_errors=True)
            log(f"清理占位 scratch 目录：{scratch_used}")

    stats = {
        "report": "pilot_run_stats",
        "task": "HIRA-022",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "bundle_id": collector.bundle_id,
        "controller_id": collector.controller_id,
        "experiment_id": collector.experiment_id,
        "seed_range": [args_cli.seed_base, args_cli.seed_base + 19],
        "window_s": args_cli.window,
        "batches": batch_stats,
        "total_dir_bytes": dir_size(run_dir),
        "rss_max_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        "wall_total_s": round(time.time() - T0, 1),
    }
    (run_dir / "run_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    log(f"完成：{stats['wall_total_s']}s，目录 {stats['total_dir_bytes'] / 1e6:.1f} MB")
    print("PILOT_RUN_OK", flush=True)
    return 0


if __name__ == "__main__":
    code = 2
    try:
        code = main()
    except SystemExit as exc:
        code = int(exc.code or 0)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("PILOT_RUN_ERROR", flush=True)
        code = 2
    finally:
        log("hard exit (skip simulation_app.close)")
        sys.stdout.flush()
        sys.stderr.flush()
        import os

        os._exit(code)
