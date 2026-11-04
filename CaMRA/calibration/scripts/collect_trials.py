#!/usr/bin/env python3
"""CAL-015 轨迹采集器：按清单跑指定 split / trial 子集。

一个进程只建一次 env（同进程二次 gym.make 会挂死，见 §15.6），按 bucket
（combined vs 其余）分进程执行；批内每个 env 对应一条独立 trial：

  * 命令：``TableProgram`` 每 env 一张 10 Hz 命令表（warmup 1 s 强制零命令）；
  * 执行：冻结 v2 策略 + 既有 ``ExternalCommandWrapper``，观测/动作/增益不变；
  * 物理条件：逐 env 直写摩擦/质量/初态（CAL-013），profile 采样可追溯；
  * 记录：200 Hz 力矩/关节速度/根状态 + 50 Hz 状态 + 1 s warmup 边界；
  * 名义：同一命令程序 + 同一起点积分出 301 个配对点。

任何 env 触发 termination 立即停止整批，避免 IsaacLab 自动复位把新轨迹拼进旧 trial；
未终止的兄弟 env 记为 interrupted，三者（completed/terminated/interrupted）都落盘计数。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAL = HERE.parent
for p in (str(CAL / "src"), str(CAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from camra_calibration.commands import build_program  # noqa: E402
from camra_calibration.energy import steps_for  # noqa: E402
from camra_calibration.nominal import PlanarState, integrate  # noqa: E402
from camra_calibration.profiles import sample_applied, sample_initial_pose  # noqa: E402
from camra_calibration.protocol import Protocol  # noqa: E402
from camra_calibration.store import TrialStore, plan_batches, resume_filter  # noqa: E402
from camra_calibration.timeline import Timeline  # noqa: E402

BUCKET_OF = {"clean": "off", "friction_only": "off", "mass_only": "off",
             "push_only": "off", "combined": "on"}


def load_trials(manifest_dir: Path, split: str) -> list[dict]:
    path = manifest_dir / f"{split}.jsonl"
    if not path.is_file():
        raise SystemExit(f"清单不存在: {path}")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def bucket(profile: str) -> str:
    try:
        return BUCKET_OF[profile]
    except KeyError:
        raise SystemExit(f"未知 profile {profile!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--bucket", choices=("off", "on"), required=True,
                    help="一个进程只处理一个 bucket（corruption 开关是 cfg 级）")
    ap.add_argument("--num_envs", type=int, default=8)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = 全部剩余")
    ap.add_argument("--max-batches", type=int, default=0, help="0 = 不限制")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--env-seed", type=int, default=100)
    ap.add_argument("--allow-overwrite", action="store_true",
                    help="默认跳过已完成 trial；加此参数才会重采并新建 attempt 目录")
    ap.add_argument("--lock", default=None,
                    help="正式 split（calibration/heldout/near_obstacle）必须给出数据锁 JSON")
    ap.add_argument("--maps", default=None, help="near_obstacle 必须给出冻结地图 JSON")
    ap.add_argument("--margins", default=None,
                    help="margin 冻结文件（默认 config/margins_v1.json）；检验/近障前置条件")
    args = ap.parse_args()

    # ---- 正式数据门禁：任何一项不满足就在启动仿真前退出 ----
    from camra_calibration.lock import DataLock, LockError, enforce_collection
    if args.split in ("calibration", "heldout", "near_obstacle"):
        if not args.lock:
            raise SystemExit(f"REFUSED: {args.split} 是正式数据，必须给出 --lock 数据锁")
        try:
            dlock = DataLock.load(args.lock)
            enforce_collection(dlock, args.split, Path(__file__).resolve().parent.parent,
                               manifest_dir=Path(args.manifest_dir),
                               maps_path=Path(args.maps) if args.maps else None,
                               margins_path=Path(args.margins) if args.margins else None,
                               allow_overwrite=args.allow_overwrite)
        except LockError as exc:
            raise SystemExit(f"REFUSED: {exc}")
        print(f"LOCK OK: {dlock.path} sha {dlock.sha256[:16]} split {args.split}")
    elif args.lock:
        try:
            dlock = DataLock.load(args.lock)
            if dlock.list_entry(args.split) is not None:
                bad = dlock.verify_artifacts(Path(__file__).resolve().parent.parent)
                if bad:
                    raise SystemExit("REFUSED: 锁与现场不一致：\n  " + "\n  ".join(bad[:5]))
        except LockError as exc:
            raise SystemExit(f"REFUSED: {exc}")

    proto = Protocol.load()
    tl = Timeline.from_protocol(proto)
    warmup_s = tl.warmup_s
    window_s = tl.measured_window_s
    warmup_ctrl = round(warmup_s / tl.policy_dt)
    window_ctrl = round(window_s / tl.policy_dt)
    window_steps = steps_for(window_s, tl.sim_dt)
    pair_dt = tl.pairing_dt

    store = TrialStore(Path(args.out_root), args.split)
    all_trials = load_trials(Path(args.manifest_dir), args.split)
    mine = [t for t in all_trials if bucket(t["profile"]) == args.bucket]
    mine = mine[args.offset:]
    if args.limit > 0:
        mine = mine[:args.limit]
    todo, skipped = resume_filter(mine, store, allow_overwrite=args.allow_overwrite)
    if not todo:
        print(json.dumps({"split": args.split, "bucket": args.bucket,
                          "selected": len(mine), "skipped_done": skipped,
                          "remaining": 0}, ensure_ascii=False))
        return 0

    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True, device=args.device)
    sim_app = launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401,F403
    import unitree_rl_lab.tasks  # noqa: F401,F403
    from sim.batch_program import TableProgram
    from sim.profiles_sim import (apply_friction, apply_initial_pose, apply_mass,
                                  apply_push, configure_env_for_profile)
    from sim.recorder import TrialRecorder
    from unitree_rl_lab.utils.external_command import DEFAULT_LIMITS, ExternalCommandWrapper
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

    n_env = min(args.num_envs, len(todo))
    env_cfg = parse_env_cfg("CaMRA-Go2-Velocity", device=args.device, num_envs=n_env,
                            use_fabric=True, entry_point_key="env_cfg_entry_point")
    env_cfg.seed = args.env_seed
    env_cfg.episode_length_s = warmup_s + window_s + 0.2
    rep = configure_env_for_profile(env_cfg, sample_applied(todo[0]["profile"], todo[0]["seed"]))
    raw = gym.make("CaMRA-Go2-Velocity", cfg=env_cfg)
    env = ExternalCommandWrapper(raw, limits=DEFAULT_LIMITS, enforce_limits=True, on_reset="zero")
    u = raw.unwrapped
    robot = u.scene["robot"]
    sim = u.sim
    sim_dt = float(sim.get_physics_dt())
    step_dt = float(u.step_dt)
    dec = int(round(step_dt / sim_dt))
    if int(round(u.max_episode_length)) <= warmup_ctrl + window_ctrl:
        raise RuntimeError("trial 时限不足（应为 warmup+window+余量）")

    bundle = Path("/pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2")
    policy = torch.jit.load(str(bundle / "exported" / "policy.pt"), map_location=args.device).eval()
    bundle_tree = hashlib.sha256(
        json.dumps(sorted(p.name for p in bundle.iterdir())).encode()).hexdigest()

    orig_step = sim.step
    stats = {"completed": 0, "terminated": 0, "truncated": 0, "interrupted": 0}
    batches_done = 0

    for batch_index, batch in enumerate(plan_batches(todo, n_env)):
        if args.max_batches and batches_done >= args.max_batches:
            break
        bi = batch_index * n_env
        k = len(batch)
        # ---- 每 env 一条 trial：命令表 / profile / 初态 ----
        tables, applied, poses, progs = [], [], [], []
        for tr in batch:
            pr = build_program(tr["family"], tr["seed"], duration_s=window_s,
                               command_dt=pair_dt)
            progs.append(pr)
            tables.append([list(c) for c in pr.commands[:window_ctrl // 5]])
            applied.append(sample_applied(tr["profile"], tr["seed"], duration_s=window_s))
            poses.append(sample_initial_pose(tr["seed"]))
        padded = [[[0.0, 0.0, 0.0] for _ in range(window_ctrl // 5)] for _ in range(n_env - k)]
        table_t = torch.tensor(tables + padded, dtype=torch.float32)
        program = TableProgram(table_t, dt=pair_dt, warmup_s=warmup_s, device=args.device)

        obs, _ = env.reset()
        obs = obs["policy"]
        env.set_program(program)
        for i in range(k):
            apply_friction(robot, [i], applied[i].friction, restitution=applied[i].restitution)
            apply_mass(robot, [i], applied[i].added_mass_kg)
            apply_initial_pose(robot, [i], [poses[i]["px"]], [poses[i]["py"]], [poses[i]["yaw"]])
        initial_pose_rows = []
        for i in range(n_env):
            tr_pos = robot.root_physx_view.get_root_transforms()[i, :3].tolist()
            initial_pose_rows.append([float(v) for v in tr_pos])

        rec = TrialRecorder(robot=robot, num_envs=n_env, dt=sim_dt,
                            expected_steps=window_steps, n_joints=robot.num_joints,
                            record_physics=True, expected_control_steps=window_ctrl,
                            label=f"{args.split}#{bi // n_env}")

        def patched_step(*a, **kw):
            out = orig_step(*a, **kw)
            rec.after_physics_step()
            return out

        sim.step = patched_step
        rec.begin_warmup()
        started = time.time()
        outcome = "completed"
        stop_ctrl = window_ctrl
        term_mask = torch.zeros(n_env, dtype=torch.bool)
        with torch.no_grad():
            for _ in range(warmup_ctrl):
                act = policy(obs)
                obs, _r, _te, _tr, _in = env.step(act)
                obs = obs["policy"]
            if rec.warmup_steps != warmup_ctrl * dec:
                raise RuntimeError(f"warmup 步数 {rec.warmup_steps} ≠ {warmup_ctrl * dec}")
            rec.begin_window()
            for c in range(window_ctrl):
                act = policy(obs)
                obs, _r, te, tr_, _in = env.step(act)
                obs = obs["policy"]
                cmd = u.command_manager.get_command("base_velocity").detach().clone()
                rec.after_policy_step(act, cmd)
                if bool(te.any()):
                    term_mask = te.detach().cpu()
                    outcome = "terminated"
                    stop_ctrl = c + 1
                    break
                if bool(tr_.any()):
                    term_mask = tr_.detach().cpu()
                    outcome = "truncated"
                    stop_ctrl = c + 1
                    break
        sim.step = orig_step
        elapsed = time.time() - started
        rec.finish(outcome)
        arrays = rec.stack()
        steps_executed = rec.window_steps
        batches_done += 1

        # ---- 逐 trial 落盘 ----
        for i, tr in enumerate(batch):
            trial_outcome = outcome
            if outcome != "completed" and not bool(term_mask[i]):
                trial_outcome = "interrupted"
            stats[trial_outcome] += 1
            d, attempt = store.prepare(tr["trial_id"])
            phys = {"tau": arrays["tau"][:, i, :], "qdot": arrays["qdot"][:, i, :],
                    "root_pos": arrays["root_pos_200hz"][:, i, :],
                    "root_quat": arrays["root_quat_200hz"][:, i, :],
                    "root_lin_vel": arrays["root_lin_vel_200hz"][:, i, :],
                    "root_ang_vel": arrays["root_ang_vel_200hz"][:, i, :],
                    "sim_dt": torch.tensor([sim_dt])}
            store.write_npz(d, "physics_200hz.npz", phys)
            state = {k.replace("_50hz", ""): v[:, i, :] for k, v in arrays.items() if k.endswith("_50hz")}
            store.write_npz(d, "state_50hz.npz", state)

            # 配对：t=0 初态 + 每 0.1 s（50 Hz 第 5k-1 条） → 301 点
            pos50 = arrays["root_pos_50hz"][:, i, :2]
            quat50 = arrays["root_quat_50hz"][:, i, :]
            idx = torch.arange(1, window_ctrl // 5 + 1) * 5 - 1
            idx = idx[idx < pos50.shape[0]]
            pair_pos = torch.cat([torch.tensor(initial_pose_rows[i][:2]).float().unsqueeze(0),
                                  pos50[idx]], dim=0)
            n_pairs = pair_pos.shape[0]
            # 时间轴必须 float64：float32 无法精确表示 0.1，300 段后偏差达 1.5e-6，
            # 会触碰协议 time_sync 容差（见 §18.2）
            t = torch.arange(n_pairs, dtype=torch.float64) * pair_dt
            nom = integrate(PlanarState(poses[i]["px"], poses[i]["py"], poses[i]["yaw"]),
                            progs[i].commands[:window_ctrl // 5 + 1], pair_dt)
            nom_pos = torch.tensor([[s.px, s.py] for s in nom], dtype=torch.float32)
            nom_yaw = torch.tensor([s.yaw for s in nom], dtype=torch.float32)
            dev = torch.linalg.norm(pair_pos - nom_pos[:n_pairs], dim=-1)
            store.write_npz(d, "pairing_10hz.npz", {
                "t": t, "exec_pos": pair_pos, "nominal_pos": nom_pos[:n_pairs],
                "nominal_yaw": nom_yaw[:n_pairs], "deviation": dev,
                "cmd": torch.tensor([[list(c) for c in progs[i].commands[:n_pairs]]],
                                    dtype=torch.float32).squeeze(0),
            })

            energy = {"unit": "J", "metric_id": "D-ENERGY-001",
                      "j_exec": float(rec.total_j[i].item()),
                      "j_nom": None, "j_nom_status": "not_available_pending_power_model",
                      "steps_executed": steps_executed,
                      "per_joint_j": [float(x) for x in rec.per_joint_j[i].tolist()]}
            store.write_json(d, "energy.json", energy)
            meta = {
                "trial_id": tr["trial_id"], "split": args.split, "family": tr["family"],
                "profile": tr["profile"], "seed": tr["seed"], "attempt": attempt,
                "controller": "camra_go2_velocity_it2999_v2",
                "protocol_version": proto.version, "protocol_sha256": proto.sha256,
                "bundle_tree_sha256": tr.get("program_sha256", bundle_tree),
                "program_sha256": progs[i].sha256(),
                "outcome": trial_outcome, "steps_executed": steps_executed,
                "steps_expected": window_steps, "control_steps": stop_ctrl,
                "warmup_steps": rec.warmup_steps, "warmup_ctrl_steps": warmup_ctrl,
                "pairing_points": n_pairs, "pairing_dt": pair_dt,
                "profile_applied": applied[i].to_manifest_record(),
                "initial_state": {"px": poses[i]["px"], "py": poses[i]["py"],
                                  "yaw": poses[i]["yaw"],
                                  "spawn_pos_xyz": initial_pose_rows[i]},
                "sim_dt": sim_dt, "policy_dt": step_dt, "decimation": dec,
                "episode_length_s": env_cfg.episode_length_s,
                "env_index": i, "batch_index": bi // n_env,
                "batch_size": k, "wall_seconds": round(elapsed, 3),
                "cfg_report": rep.to_dict(),
                "energy": {"unit": "J", "j_exec": energy["j_exec"]},
                "notes": ("同一命令程序生成名义轨迹；energy.j_nom 待功率模型（CAL-019/020）"
                          if trial_outcome == "completed" else
                          f"outcome={trial_outcome}，未外推补足 30 s"),
            }
            store.write_json(d, "meta.json", meta)
            store.append_record({
                "trial_id": tr["trial_id"], "split": args.split, "family": tr["family"],
                "profile": tr["profile"], "seed": tr["seed"], "attempt": attempt,
                "outcome": trial_outcome, "steps_executed": steps_executed,
                "dir": str(d.relative_to(Path(args.out_root))),
                "j_exec": energy["j_exec"], "pairing_points": n_pairs,
            })
            store.mark_done(d)
        print(json.dumps({"batch": bi // n_env, "trials": [t["trial_id"] for t in batch],
                          "outcome": outcome, "steps": steps_executed,
                          "wall_s": round(elapsed, 1)}, ensure_ascii=False), flush=True)

    print(json.dumps({"split": args.split, "bucket": args.bucket,
                      "selected": len(mine), "skipped_done": skipped,
                      "processed": sum(stats.values()), "stats": stats,
                      "batches": batches_done}, ensure_ascii=False, indent=1))
    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
