"""HIRA-025 起：按已锁定的 manifest 采集某个 split（通用入口）。

与 run_pilot.py 的区别：trial 清单不在这里生成，而是**逐行读 manifests/<split>.jsonl**
（HIRA-024 已冻结），命令表用同一个 `build_program` 重建后按 `program_sha256` 对账。
批内要求 obs_noise / window_s / warmup_s 一致（HIRA-021 约束）→ 按 profile 分批。

    python tests/run_split.py --headless --device cuda:0 --split baseline_fit
    python tests/run_split.py --headless --device cuda:0 --split baseline_dev --num-envs 8

断点续跑：每批先 plan_attempts，DONE 完整的 trial 直接 skip；不足一批时用 scratch
占位填满 env 槽位，跑完即删（scratch 目录不参与任何统计）。
"""

from __future__ import annotations

import argparse
import hashlib
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
parser.add_argument("--run-id", default=None, help="缺省 = <split>_<UTC时间戳>")
parser.add_argument("--runs-root", default="/pub/data/caohy/logs/camra_hira/runs")
parser.add_argument("--split", required=True)
parser.add_argument("--manifests", default=str(EXP / "manifests"))
parser.add_argument("--bundle", default=str(EXP / "frozen/hira_go2_18500"))
parser.add_argument("--prereg", default=str(EXP / "config/integration_acceptance_prereg_v1.json"))
parser.add_argument("--usd-dir", default="/pub/data/caohy/logs/camra_hira/assets/usd_cache")
parser.add_argument("--num-envs", type=int, default=8, help="批内 env 数；同一 run 内固定")
parser.add_argument("--window", type=float, default=30.0)
parser.add_argument("--max-batches", type=int, default=0, help=">0 时只跑前 N 批（冒烟用）")
# HIRA-029：数据锁是采集的唯一入口，任何散列不符都在启动仿真之前拒绝
parser.add_argument("--lock", default=str(EXP / "config/data_lock_hira_calibration_v1.json"))
parser.add_argument("--margins", default=str(EXP / "config/margins_hira_v1.json"))
parser.add_argument("--maps", default=str(EXP / "config/near_obstacle_maps_v1.json"))
# --device 由 AppLauncher 提供

_PRE = parser.parse_known_args()[0]
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from camra_calibration.commands import build_program  # noqa: E402
from hira_go2.bundle import load_bundle  # noqa: E402
from camra_calibration.lock import DataLock, LockError, enforce_collection  # noqa: E402
from hira_go2.collector import HiraCollector, TrialSpec, plan_attempts, sample_spec  # noqa: E402

T0 = time.time()


def log(text: str) -> None:
    print(f"[split] {time.time() - T0:7.1f}s {text}", flush=True)


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(split: str) -> list[dict]:
    path = Path(args_cli.manifests) / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"manifest 为空：{path}")
    return rows


def build_specs(rows: list[dict]) -> tuple[list[TrialSpec], list[dict]]:
    """按 manifest 重建 spec，并逐条对账命令表散列。"""
    specs, audit = [], []
    for row in rows:
        program = build_program(
            row["family"], row["seed"], duration_s=row["duration_s"], command_dt=row["command_dt"]
        )
        commands = [list(map(float, c)) for c in program.commands]
        commands_sha = hashlib.sha256(
            json.dumps(commands, separators=(",", ":")).encode()
        ).hexdigest()
        spec = sample_spec(
            trial_id=row["trial_id"], split=row["split"], program=row["family"],
            profile=row["profile"], seed=row["seed"], commands=commands,
            window_s=args_cli.window,
        )
        audit.append(
            {
                "trial_id": row["trial_id"],
                "manifest_program_sha256": row["program_sha256"],
                "rebuilt_program_sha256": commands_sha,
                "template_sha256": row.get("template_sha256"),
                "rebuilt_template_sha256": program.sha256(),
                "matches": commands_sha == row["program_sha256"]
                and program.sha256() == row.get("template_sha256"),
            }
        )
        specs.append(spec)
    return specs, audit


def main() -> int:
    split = args_cli.split
    run_id = args_cli.run_id or f"{split}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    run_dir = Path(args_cli.runs_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- G2 前置：数据锁门禁（仿真启动前）----
    lock_path = Path(args_cli.lock)
    if not lock_path.is_file():
        raise SystemExit(f"缺少数据锁 {lock_path}：正式采集必须由锁放行（HIRA-029）")
    lock = DataLock.load(lock_path, verify_hash=True)
    try:
        enforce_collection(
            lock, split, REPO,
            manifest_dir=Path(args_cli.manifests),
            maps_path=args_cli.maps if split == "near_obstacle" else None,
            margins_path=args_cli.margins,
        )
    except LockError as exc:
        raise SystemExit(f"数据锁拒绝 {split!r}：{exc}") from exc
    log(f"lock ok: {lock_path.name} sha={lock.sha256[:16]} split={split}")

    bundle = load_bundle(Path(args_cli.bundle), expected_controller_id=Path(args_cli.bundle).name)

    rows = load_manifest(split)
    manifest_path = Path(args_cli.manifests) / f"{split}.jsonl"
    log(f"{len(rows)} 条 {split} manifest 载入（sha {sha256_file(manifest_path)[:16]}）")

    specs, audit = build_specs(rows)
    bad = [item["trial_id"] for item in audit if not item["matches"]]
    if bad:
        raise SystemExit(f"命令表重建对账失败 {len(bad)} 条：{bad[:5]}")

    config_path = run_dir / "collect_config.json"
    if not config_path.is_file():
        config_path.write_text(
            json.dumps(
                {
                    "task": "HIRA-025",
                    "split": split,
                    "manifest": manifest_path.name,
                    "manifest_sha256": sha256_file(manifest_path),
                    "window_s": args_cli.window,
                    "warmup_s": 1.0,
                    "num_envs": args_cli.num_envs,
                    "usd_dir": args_cli.usd_dir,
                    "bundle_id": bundle["manifest"]["bundle_id"],
                    "data_lock": {"file": lock_path.name, "sha256": lock.sha256,
                                  "stage": lock.payload.get("stage")},
                    "controller_id": bundle["controller"]["controller_id"],
                    "num_trials": len(specs),
                    "program_rebuild": "camra_calibration.commands.build_program（逐条按 program_sha256 对账）",
                    "batch_rule": "同批 obs_noise / window_s / warmup_s 必须一致 → 按 profile 分批",
                },
                ensure_ascii=False,
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )

    collector = HiraCollector(
        args_cli.bundle, run_dir, args_cli.prereg, config_path,
        num_envs=args_cli.num_envs, device=args_cli.device,
        expected_joints={"target_order": bundle["controller"]["joints"]["target_order"]},
    )
    log(f"collector ready: controller={collector.controller_id} bundle={collector.bundle_id[:16]}")

    # ---- 按 profile 分批（obs_noise 是 obs 级开关，不能逐 env）----
    by_profile: dict[str, list[TrialSpec]] = {}
    for spec in specs:
        by_profile.setdefault(spec.profile, []).append(spec)

    batch_stats = []
    batches_done = 0
    for profile in sorted(by_profile):
        group = by_profile[profile]
        # 先整体 plan 一次，把已完成/阻断的剔掉
        plan = {item["trial_id"]: item for item in plan_attempts(run_dir, group)}
        runnable = []
        for spec in group:
            item = plan[spec.trial_id]
            if item["action"] == "skip":
                continue
            if item["action"] == "blocked_business_failure":
                log(f"blocked {spec.trial_id}（业务失败，不自动重试）")
                continue
            runnable.append((spec, item["attempt"]))
        log(f"profile {profile}: {len(group)} 条，待跑 {len(runnable)} 条")

        cursor = 0
        while cursor < len(runnable):
            chunk = runnable[cursor:cursor + args_cli.num_envs]
            cursor += len(chunk)
            fill = list(chunk)
            scratch_used = []
            while len(fill) < collector.num_envs:
                spec0, _a = chunk[0]
                scratch = TrialSpec(
                    trial_id=f"hira-scratch-{profile}-{len(fill)}", split="scratch",
                    program=spec0.program, profile=spec0.profile, seed=spec0.seed + 9000,
                    commands=spec0.commands, window_s=spec0.window_s, warmup_s=spec0.warmup_s,
                    friction=spec0.friction, added_mass_kg=spec0.added_mass_kg,
                    obs_noise=spec0.obs_noise, initial_pose=spec0.initial_pose,
                    push_events=spec0.push_events,
                )
                fill.append((scratch, 1))
                scratch_used.append(scratch.trial_id)

            t0 = time.time()
            collector.execute_batch([item[0] for item in fill], [item[1] for item in fill])
            wall = time.time() - t0
            batches_done += 1
            batch_stats.append(
                {
                    "profile": profile, "n_trials": len(chunk), "scratch_slots": scratch_used,
                    "wall_s": round(wall, 1),
                    "wall_per_trial_s": round(wall / max(len(chunk), 1), 2),
                    "rss_max_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
                    "trial_ids": [item[0].trial_id for item in chunk],
                }
            )
            log(
                f"  批 {batches_done}: {len(chunk)} 条（profile {profile}），wall {wall:.1f}s，"
                f"单条 {wall / max(len(chunk), 1):.2f}s"
            )
            for trial_id in scratch_used:
                shutil.rmtree(run_dir / trial_id, ignore_errors=True)
            if args_cli.max_batches and batches_done >= args_cli.max_batches:
                log(f"--max-batches={args_cli.max_batches} 达到，提前停止")
                cursor = len(runnable)
                break

    stats = {
        "report": "split_run_stats",
        "task": "HIRA-025",
        "split": split,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "bundle_id": collector.bundle_id,
        "controller_id": collector.controller_id,
        "num_envs": args_cli.num_envs,
        "window_s": args_cli.window,
        "n_manifest_rows": len(rows),
        "n_batches": batches_done,
        "batches": batch_stats,
        "total_dir_bytes": dir_size(run_dir),
        "wall_total_s": round(time.time() - T0, 1),
        "rss_max_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
    }
    (run_dir / "run_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    log(f"完成：{stats['wall_total_s']}s，{batches_done} 批，目录 {stats['total_dir_bytes'] / 1e6:.1f} MB")
    print(f"SPLIT_{split.upper()}_RUN_OK", flush=True)
    return 0


if __name__ == "__main__":
    code = 2
    try:
        code = main()
    except SystemExit as exc:
        code = int(exc.code or 0)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("SPLIT_RUN_ERROR", flush=True)
        code = 2
    finally:
        log("hard exit (skip simulation_app.close)")
        sys.stdout.flush()
        sys.stderr.flush()
        import os

        os._exit(code)
