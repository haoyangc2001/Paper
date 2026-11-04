#!/usr/bin/env python3
"""CAL-013 无头验证：五类 profile 的配置 diff、PhysX 读回与追溯记录。

构建策略：非 combined 的四个 profile 在 cfg 级完全一致（观测 corruption 都关），
共用一次 env 构建，逐 profile 重新施加 per-env 值；combined（corruption 开）
单独构建一次。共 2 次 env 构建。

对每个 profile：
  1. 采样两条 trial 的 applied 配置（seed 6000/6001）与初态位姿；
  2. cfg 级报告（configure_env_for_profile：移除隐藏随机化、设置观测噪声）；
  3. 逐 env 写摩擦/质量（创建后）与初态位姿（reset 后），全部读回校验；
  4. push 类 profile 对每个 env 做一次真实写入读回；
  5. 落盘逐 profile JSON 与汇总 knob-diff / cfg-level diff。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAL = HERE.parent
for p in (str(CAL / "src"), str(CAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from camra_calibration.profiles import (  # noqa: E402
    PROFILE_IDS,
    knob_diff,
    sample_applied,
    sample_initial_pose,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="CaMRA-Go2-Velocity")
    ap.add_argument("--num_envs", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs=2, default=[6000, 6001])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--profiles", nargs="+", default=list(PROFILE_IDS),
                    help="本次进程要跑的 profile 子集（每次进程只建一个 env）")
    args = ap.parse_args()

    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True, device=args.device)
    sim_app = launcher.app

    import gymnasium as gym
    import torch  # noqa: F401
    import isaaclab_tasks  # noqa: F401,F403
    import unitree_rl_lab.tasks  # noqa: F401,F403
    from sim.profiles_sim import (
        apply_friction, apply_initial_pose, apply_mass, apply_push,
        configure_env_for_profile,
    )
    from unitree_rl_lab.utils.external_command import DEFAULT_LIMITS, ExternalCommandWrapper
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(args.seeds)
    applied_all = {pid: [sample_applied(pid, s) for s in seeds] for pid in PROFILE_IDS}
    poses_all = {pid: [sample_initial_pose(s) for s in seeds] for pid in PROFILE_IDS}
    env_ids = list(range(args.num_envs))

    def make_env():
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs,
                                use_fabric=True, entry_point_key="env_cfg_entry_point")
        env_cfg.seed = 100
        return env_cfg

    def build_with(applied) -> tuple[object, object]:
        env_cfg = make_env()
        report = configure_env_for_profile(env_cfg, applied)
        raw = gym.make(args.task, cfg=env_cfg)
        env = ExternalCommandWrapper(raw, limits=DEFAULT_LIMITS, enforce_limits=True, on_reset="zero")
        obs, _ = env.reset()
        return env, report

    def probe(env, applied, poses) -> dict:
        robot = env.unwrapped.scene["robot"]
        fr = [apply_friction(robot, [i], applied[i].friction,
                             restitution=applied[i].restitution) for i in env_ids]
        ms = [apply_mass(robot, [i], applied[i].added_mass_kg) for i in env_ids]
        po = [apply_initial_pose(robot, [i], [poses[i]["px"]], [poses[i]["py"]],
                                 [poses[i]["yaw"]]) for i in env_ids]
        push_rec = []
        for i in env_ids:
            if applied[i].push_events:
                t, vx, vy = applied[i].push_events[0]
                rec = apply_push(robot, i, vx, vy)
                rec["scheduled_t"] = t
                push_rec.append(rec)
        items = fr + ms + po + push_rec
        return {"friction": fr, "mass": ms, "initial_pose": po, "push": push_rec,
                "all_ok": all(x["ok"] for x in items)}

    subset = list(args.profiles)
    for pid in subset:
        if pid not in PROFILE_IDS:
            raise ValueError(f"未知 profile {pid!r}")
    results: dict[str, dict] = {}
    overall_ok = True

    # 同一进程只建一次 env（实测同一进程 close 后再 gym.make 会挂死）；
    # 子集内 corruption 开关一致（clean 组全关，combined 开），首个 profile 决定 cfg。
    env, report = build_with(applied_all[subset[0]][0])
    for pid in subset:
        rb = probe(env, applied_all[pid], poses_all[pid])
        overall_ok &= rb["all_ok"]
        results[pid] = {
            "profile_id": pid,
            "cfg_report": report.to_dict(),
            "remaining_event_terms": _remaining(env),
            "applied": [a.to_dense_dict() for a in applied_all[pid]],
            "initial_poses": poses_all[pid],
            "readback": rb,
        }
        print(f"[{pid}] readback ok={rb['all_ok']} corruption={report.obs_corruption_enabled}")

    # 注意：sim_app.close() 会强制退出进程，所有证据必须在此之前落盘
    for pid in subset:
        (out_dir / f"{pid}.json").write_text(
            json.dumps(results[pid], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    (out_dir / (f"{'_'.join(subset)}.status.json")).write_text(
        json.dumps({"profiles": subset, "overall_ok": overall_ok}, sort_keys=True) + "\n",
        encoding="utf-8")
    print("PROFILE_RUN_DONE", "ok" if overall_ok else "FAILED", flush=True)
    env.close()
    sim_app.close()
    return 0 if overall_ok else 2


def _remaining(env) -> dict:
    """剩余事件项的实际参数（证明 reset 随机化已退化）。"""
    events = env.unwrapped.cfg.events
    remain = {}
    for name in ("reset_base", "reset_robot_joints"):
        term = getattr(events, name, None)
        if term is None:
            continue
        remain[name] = {k: (list(v) if isinstance(v, (list, tuple)) else str(v))
                        for k, v in getattr(term, "params", {}).items()
                        if k in ("pose_range", "velocity_range", "position_range")}
    return remain


if __name__ == "__main__":
    sys.exit(main())
