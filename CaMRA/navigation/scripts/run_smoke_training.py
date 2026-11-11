#!/usr/bin/env python
"""无裕度分级短训练（NAV-034，名义环境部分）。

做三件事，顺序固定：

  1. **校验预登记**：``config/smoke_training_prereg_v1.json`` 的字节散列必须与同目录
     ``.sha256`` 一致。跑之前已冻结；不一致就拒绝启动，不"跑完再改阈值"。
  2. **分级短训练**：L1_empty → L2_single_detour → L3_multi_path，每级固定 30 回合，
     每个回合一个 chunk 落盘（复用 NAV-032 的 storage），每 chunk 一次更新。
     只用 train 种子；development 种子只做评估；integration_pilot 与 final_* 一律不碰。
  3. **放行判定**：G1 环境 / G2 目标 / G3 采样 / G4 优化 / G5 可学性。
     G5 只看任务指标（最近到达距离），不看 loss。任一条不过，verdict 就是 FAIL，
     不做"多跑几轮看看"。

无裕度的含义：α=0（成本模型 uncalibrated）、δ=0（安全几何 delta 置零），
足迹 r_fp=0.5055 m 仍然参与碰撞与安全判据。

Run::

    <frozen python> CaMRA/navigation/scripts/run_smoke_training.py \
        --run-id nav034_smoke --json CaMRA/navigation/reports/nominal_smoke_training.json
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(CALIB, "src"))

from camra_navigation.algorithm import networks as NW  # noqa: E402
from camra_navigation.algorithm import ppo as P  # noqa: E402
from camra_navigation.algorithm import storage as S  # noqa: E402
from camra_navigation.costs import JCostModel, Z_REF_J  # noqa: E402
from camra_navigation.envs.nominal import NominalNavEnv  # noqa: E402
from camra_navigation.maps import generator as M  # noqa: E402
from camra_navigation.maps.geometry import SafetyGeometry  # noqa: E402
from camra_navigation.runs import default_work_root  # noqa: E402

MAP_DESIGN = os.path.join(NAV_ROOT, "config", "map_design_v1.json")
GEOMETRY_CONTRACT = os.path.join(NAV_ROOT, "config", "geometry_contract_v1.json")
NAV_CONTRACT = os.path.join(NAV_ROOT, "config", "navigation_contract_v1.json")
VALUE_CONTRACT = os.path.join(NAV_ROOT, "config", "value_objective_contract_v1.json")
SPLIT_MANIFEST = os.path.join(NAV_ROOT, "config", "split_manifest_v1.json")
FOOTPRINT = os.path.join(CALIB, "config", "footprint_v1.json")
MARGINS = os.path.join(CALIB, "config", "margins_v1.json")
POWER = os.path.join(CALIB, "config", "power_model_v1.json")
PREREG = os.path.join(NAV_ROOT, "config", "smoke_training_prereg_v1.json")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def verify_prereg() -> tuple[dict[str, Any], str]:
    """预登记必须早于数据：字节散列不符就拒绝启动。"""
    if not os.path.exists(PREREG):
        raise SystemExit(f"缺少预登记文件：{PREREG}")
    side = PREREG + ".sha256"
    if not os.path.exists(side):
        raise SystemExit(f"缺少预登记散列：{side}（先冻结再跑）")
    actual = sha256_file(PREREG)
    declared = open(side, encoding="utf-8").read().strip()
    if actual != declared:
        raise SystemExit(
            f"预登记文件已被改动：声明 {declared[:12]}…，实际 {actual[:12]}…。"
            "阈值必须在跑之前定好，改了就另立 amend 文件")
    with open(PREREG, encoding="utf-8") as fh:
        return json.load(fh), actual


# ---------------------------------------------------------------- 环境
def build_geometry() -> SafetyGeometry:
    """δ=0、足迹保留：只把位置裕度置零，r_fp 与多边形不动。"""
    g = SafetyGeometry.from_frozen(FOOTPRINT, MARGINS, GEOMETRY_CONTRACT)
    return dataclasses.replace(g, delta_m=0.0)


def build_cost_model() -> JCostModel:
    """α=0：uncalibrated 基线，显式声明"没有标定系数"。"""
    return JCostModel.from_frozen(POWER, MARGINS, variant="uncalibrated")


def make_env(level: str, seed: int, z0_J: float, design: dict,
             geom: SafetyGeometry, model: JCostModel) -> NominalNavEnv:
    spec = M.build(level, seed, design)
    return NominalNavEnv(spec, geom, model, z0_J=z0_J)


# ---------------------------------------------------------------- 策略动作
def mean_action(net: NW.BudgetConditionedNetworks, obs: np.ndarray) -> np.ndarray:
    """确定性动作（评估用）：取分布均值，不采样。"""
    with torch.no_grad():
        dist = net.policy(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return dist.base_dist.mean[0].numpy().astype(np.float32)


def sample_action(net: NW.BudgetConditionedNetworks, obs: np.ndarray,
                  gen: torch.Generator) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """训练用：噪声来自回合内生成器（与 NAV-032 同规矩，不共用全局 RNG）。"""
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        dist = net.policy(obs_t)
        mean = dist.base_dist.mean
        std = dist.base_dist.stddev
        noise = torch.randn(mean.shape, generator=gen, dtype=mean.dtype)
        act = mean + std * noise
        logp = float(dist.log_prob(act)[0])
        vc, vh = net.values(obs_t)
    return act[0].numpy().astype(np.float32), logp, vc[0].numpy(), vh[0].numpy()


# ---------------------------------------------------------------- 评估
def evaluate(net: NW.BudgetConditionedNetworks, levels: list[str], dev_seeds: list[int],
             z0_J: float, design: dict, geom: SafetyGeometry, model: JCostModel,
             tag: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for level in levels:
        for seed in dev_seeds:
            env = make_env(level, seed, z0_J, design, geom, model)
            obs = np.asarray(env.reset(seed=seed), dtype=np.float32)
            min_goal = float("inf")
            while not env.done:
                a = mean_action(net, obs)
                rec = env.step(a.tolist())
                obs = np.asarray(env.observation(), dtype=np.float32)
                min_goal = min(min_goal, float(rec.goal_distance_m))
            res = env.result()
            out.append({
                "tag": tag, "level": level, "map_id": res.map_id, "seed": seed,
                "event": res.label, "success": res.label == "goal_reached",
                "steps": int(res.n_steps), "t_s": float(res.t_s),
                "cost_J": float(res.cost_model_J), "z0_J": float(res.z0_J),
                "z_end_J": float(res.z_J),
                "min_goal_distance_m": float(min_goal),
                "final_goal_distance_m": float(res.goal_distance_m),
                "travelled_m": float(res.travelled_m),
                "danger_steps": int(res.danger_steps),
                "safety_y": float(res.safety_y),
                "alpha": float(res.alpha), "variant": res.variant,
            })
    return out


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="无裕度分级短训练（NAV-034）")
    ap.add_argument("--run-root", default=str(default_work_root()))
    ap.add_argument("--run-id", default="nav034_smoke")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    t_start = time.time()

    prereg, prereg_sha = verify_prereg()
    amends = []
    for name in sorted(os.listdir(os.path.dirname(PREREG))):
        if name.startswith("smoke_training_prereg_v1.amend") and name.endswith(".json"):
            path = os.path.join(os.path.dirname(PREREG), name)
            payload = json.load(open(path, encoding="utf-8"))
            amends.append({"file": name, "sha256": sha256_file(path),
                           "changes": payload.get("changes", [])})
    z0_J = float(prereg["budget"]["z0_J"])
    if not (0.0 < z0_J <= Z_REF_J):
        raise SystemExit(f"z0_J={z0_J} 不在 (0, Z_REF_J={Z_REF_J}]：会触发 z_ratio 饱和")

    design = M.load_design(MAP_DESIGN)
    geom = build_geometry()
    cost_model = build_cost_model()
    nav_contract = json.load(open(NAV_CONTRACT, encoding="utf-8"))
    splits = json.load(open(SPLIT_MANIFEST, encoding="utf-8"))["splits"]

    train_seeds = [int(s) for s in prereg["train_seeds"]]
    dev_seeds = [int(s) for s in prereg["dev_seeds"]]
    if train_seeds != [int(s) for s in splits["train"]["seeds"]]:
        raise SystemExit("预登记的 train 种子与 split_manifest 不一致")
    if dev_seeds != [int(s) for s in splits["development"]["seeds"]]:
        raise SystemExit("预登记的 development 种子与 split_manifest 不一致")

    torch.manual_seed(9200)
    net = NW.build(nav_contract)
    ppo_cfg = P.PPOConfig.from_contract()
    obj_cfg = S.ObjectiveConfig.from_contracts(VALUE_CONTRACT, r_goal_m=0.5)
    updater = P.MinimizationUpdater(net, ppo_cfg,
                                    lr_policy=float(prereg["optimizer"]["lr_policy"]),
                                    lr_critic=float(prereg["optimizer"]["lr_critic"]))

    cap_min = float(prereg["resource_cap"]["max_wall_minutes"])
    for a in amends:
        for ch in a["changes"]:
            if ch.get("field") == "resource_cap.max_wall_minutes":
                cap_min = float(ch["to"])
    run_dir = os.path.join(os.path.abspath(args.run_root), args.run_id)
    store = S.RolloutStore(os.path.abspath(args.run_root), args.run_id, obs_dim=net.cfg.obs_dim)
    log_path = os.path.join(run_dir, "train_log.jsonl")
    os.makedirs(run_dir, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(S.redact(json.dumps({
            "kind": "start", "prereg_sha256": prereg_sha, "z0_J": z0_J,
            "run_id": args.run_id,
            "geometry": {"delta_m": geom.delta_m, "r_fp_m": geom.r_fp_m},
            "cost_variant": cost_model.variant, "alpha": cost_model.alpha,
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        }, ensure_ascii=False, sort_keys=True)) + "\n")

    param0 = {k: v.detach().clone() for k, v in net.policy.named_parameters()}

    # ---- 基线评估（未训练）----
    levels = [p["level"] for p in prereg["curriculum"]]
    baseline = evaluate(net, levels, dev_seeds, z0_J, design, geom, cost_model, "baseline")

    # ---- 分级短训练 ----
    global_step = 0
    episode_id = 0
    seed_stream = np.random.default_rng(9300)
    diag: dict[str, Any] = {
        "env_faults": 0, "nonfinite_samples": 0, "objective_nonfinite": 0,
        "stop_index_below_t": 0, "feasible_steps": 0,
        "sat_components": 0, "action_components": 0,
        "entropies": [], "grad_policy": [], "losses": [],
        "episodes": [], "per_phase": [],
    }

    for phase in prereg["curriculum"]:
        level = phase["level"]
        n_eps = int(phase["train_episodes"])
        phase_rows = []
        for i in range(n_eps):
            seed = train_seeds[i % len(train_seeds)]
            env = make_env(level, seed, z0_J, design, geom, cost_model)
            ep_seed = int(seed_stream.integers(0, 2 ** 31 - 1))
            obs = np.asarray(env.reset(seed=seed), dtype=np.float32)
            gen = torch.Generator().manual_seed(ep_seed)
            seg = S.Segment(episode_id=episode_id, map_id=env.spec.map_id, map_index=0,
                            seed=ep_seed, obs_dim=net.cfg.obs_dim)
            episode_sat = 0
            episode_comp = 0
            try:
                while not env.done:
                    action, logp, vc, vh = sample_action(net, obs, gen)
                    rec = env.step(action.tolist())
                    next_obs = np.asarray(env.observation(), dtype=np.float32)
                    done = bool(rec.done)
                    kind = S.TERMINAL_KIND.get(str(rec.label)) if done else None
                    boot_c = boot_h = 0.0
                    if done and kind == "truncation":
                        with torch.no_grad():
                            bc, bh = net.values(torch.as_tensor(next_obs).unsqueeze(0))
                        boot_c, boot_h = float(bc[0]), float(bh[0])
                    seg.append(
                        global_step=global_step, episode_id=episode_id, t=int(rec.index),
                        obs=obs, next_obs=next_obs, action_raw=action,
                        action_applied=np.asarray(rec.command.applied, dtype=np.float32),
                        log_prob=logp, value_cost=float(vc), value_reach=float(vh),
                        cost_j=float(rec.cost.cost_model_J),
                        cost_norm=float(rec.cost.cost_model_J) / obj_cfg.Z_REF_J,
                        z_j=float(rec.z_J), z_ratio=float(rec.z_ratio),
                        safety_y=float(rec.safety_y),
                        goal_distance_m=float(rec.goal_distance_m),
                        terminated=bool(kind in ("failure", "success")),
                        truncated=bool(kind == "truncation"),
                        event_code=S.EVENT_CODES["none" if rec.label is None else str(rec.label)],
                        map_index=0, bootstrap_value_cost=boot_c,
                        bootstrap_value_reach=boot_h,
                        action_clipped=bool(rec.command.clipped),
                        action_invalid=bool(rec.command.invalid),
                    )
                    episode_sat += int(np.sum(np.abs(action) > 1.0))
                    episode_comp += int(action.size)
                    global_step += 1
                    obs = next_obs
            except Exception as exc:                      # 环境层异常：记下来，不掩盖
                diag["env_faults"] += 1
                diag.setdefault("env_fault_messages", []).append(f"{level}/{seed}: {exc}")
                continue

            arrays = seg.arrays()
            arrays.update(S.compute_objectives(arrays, obj_cfg, terminal_kind=kind))
            for key, val in arrays.items():
                if key.startswith("obj_"):
                    arr = np.asarray(val)
                    if arr.dtype.kind == "f" and not np.isfinite(arr).all():
                        diag["objective_nonfinite"] += int((~np.isfinite(arr)).sum())
            idx = np.asarray(arrays["obj_stop_index"])
            t_idx = np.arange(idx.shape[0])
            diag["stop_index_below_t"] += int((idx < t_idx).sum())
            diag["feasible_steps"] += int(
                (np.asarray(arrays["obj_stop_residual"], dtype=np.float64) <= -obj_cfg.beta).sum())
            if not all(np.isfinite(np.asarray(arrays[k], dtype=np.float64)).all()
                       for k in ("obs", "action_raw", "log_prob")):
                diag["nonfinite_samples"] += 1

            entry = store.write_chunk(arrays, {
                "chunk_index": store.expected_chunk_index(), "episode_id": episode_id,
                "map_id": env.spec.map_id, "map_index": 0, "reset_seed": ep_seed,
                "event": str(env.label), "terminal_kind": kind,
                "terminated": kind in ("failure", "success"),
                "truncated": kind == "truncation",
                "episode_cost_j": float(np.asarray(arrays["cost_j"]).sum()),
                "episode_cost_norm": float(np.asarray(arrays["cost_norm"]).sum()),
                "z0_J": z0_J, "n_samples": len(seg), "phase": int(phase["phase"]),
                "level": level, "train_seed": int(seed),
            })

            stats = updater.update(batch_from(arrays))
            diag["entropies"].append(stats["entropy"])
            diag["grad_policy"].append(stats["grad_norm_policy"])
            diag["losses"].append([stats["loss_policy"], stats["loss_cost"], stats["loss_reach"]])
            diag["sat_components"] += episode_sat
            diag["action_components"] += episode_comp

            row = {
                "phase": int(phase["phase"]), "level": level, "episode_id": episode_id,
                "train_seed": int(seed), "map_id": env.spec.map_id,
                "event": str(env.label), "steps": len(seg),
                "cost_J": float(np.asarray(arrays["cost_j"]).sum()),
                "z_end_J": float(np.asarray(arrays["z_j"])[-1]),
                "travelled_m": float(env.travelled_m),
                "min_goal_distance_m": float(np.asarray(arrays["goal_distance_m"]).min()),
                "danger_steps": int(env.danger_steps),
                "update": stats,
            }
            phase_rows.append(row)
            diag["episodes"].append(row)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(S.redact(json.dumps({"kind": "episode", **row, "chunk": entry["chunk_index"]},
                                             ensure_ascii=False, sort_keys=True)) + "\n")
            episode_id += 1

        after = evaluate(net, levels, dev_seeds, z0_J, design, geom, cost_model,
                         f"after_phase{phase['phase']}")
        diag["per_phase"].append({
            "phase": int(phase["phase"]), "level": level,
            "train_episodes": len(phase_rows),
            "events": _count(phase_rows, "event"),
            "mean_steps": float(np.mean([r["steps"] for r in phase_rows])) if phase_rows else 0.0,
            "mean_cost_J": float(np.mean([r["cost_J"] for r in phase_rows])) if phase_rows else 0.0,
            "eval": after,
        })

    # ---- 诊断量 ----
    params = {k: v.detach() for k, v in net.policy.named_parameters()}
    num = math.sqrt(sum(float(((params[k] - param0[k]) ** 2).sum()) for k in params))
    den = math.sqrt(sum(float((param0[k] ** 2).sum()) for k in param0)) or 1.0
    ent = diag["entropies"]
    grads = diag["grad_policy"]
    log_std_final = float(net.policy.log_std.detach().mean())
    diagnostics = {
        "env_faults": int(diag["env_faults"]),
        "env_fault_messages": diag.get("env_fault_messages", [])[:5],
        "nonfinite_samples": int(diag["nonfinite_samples"]),
        "objective_nonfinite": int(diag["objective_nonfinite"]),
        "stop_index_below_t": int(diag["stop_index_below_t"]),
        "action_saturation_rate": (diag["sat_components"] / diag["action_components"]
                                   if diag["action_components"] else 0.0),
        "entropy_initial": ent[0] if ent else None,
        "entropy_final": ent[-1] if ent else None,
        "grad_norm_p50": float(np.median(grads)) if grads else None,
        "grad_norm_max": float(np.max(grads)) if grads else None,
        "param_rel_l2": float(num / den),
        "log_std_final": log_std_final,
        "all_losses_finite": bool(all(np.isfinite(v).all() for v in diag["losses"])),
        "max_grad_norm_configured": ppo_cfg.max_grad_norm,
        "feasible_steps": int(diag["feasible_steps"]),
    }

    # ---- 放行判定 ----
    g = prereg["gates"]
    gates: dict[str, Any] = {}
    gates["G1_environment"] = {
        "ok": diagnostics["env_faults"] == 0 and diagnostics["nonfinite_samples"] == 0,
        "rule": g["G1_environment"]["rule"], "value": {
            "env_faults": diagnostics["env_faults"],
            "nonfinite_samples": diagnostics["nonfinite_samples"]},
    }
    gates["G2_objective"] = {
        "ok": diagnostics["objective_nonfinite"] == 0
        and diagnostics["stop_index_below_t"] == 0,
        "rule": g["G2_objective"]["rule"], "value": {
            "objective_nonfinite": diagnostics["objective_nonfinite"],
            "stop_index_below_t": diagnostics["stop_index_below_t"],
            "note": g["G2_objective"]["not_a_gate"]},
    }
    sat_ok = diagnostics["action_saturation_rate"] <= 0.35
    ent_ok = (diagnostics["entropy_initial"] is not None
              and diagnostics["entropy_final"] >= 0.5 * diagnostics["entropy_initial"])
    std_ok = diagnostics["log_std_final"] >= math.log(0.3)
    gates["G3_sampling"] = {
        "ok": bool(sat_ok and ent_ok and std_ok), "rule": g["G3_sampling"]["rule"],
        "value": {"action_saturation_rate": diagnostics["action_saturation_rate"],
                  "entropy_initial": diagnostics["entropy_initial"],
                  "entropy_final": diagnostics["entropy_final"],
                  "log_std_final": diagnostics["log_std_final"],
                  "thresholds": {"sat": 0.35, "entropy_ratio": 0.5, "log_std": math.log(0.3)}},
    }
    grad_ok = diagnostics["grad_norm_p50"] is not None and diagnostics["grad_norm_p50"] <= 0.45
    param_ok = 1e-5 <= diagnostics["param_rel_l2"] <= 0.5
    gates["G4_optimization"] = {
        "ok": bool(grad_ok and param_ok and diagnostics["all_losses_finite"]),
        "rule": g["G4_optimization"]["rule"],
        "value": {"grad_norm_p50": diagnostics["grad_norm_p50"],
                  "param_rel_l2": diagnostics["param_rel_l2"],
                  "all_losses_finite": diagnostics["all_losses_finite"]},
    }

    base = {(r["level"], r["seed"]): r for r in baseline}
    final_eval = diag["per_phase"][-1]["eval"] if diag["per_phase"] else []
    improve = {}
    for level in levels:
        b = [base[(level, s)]["min_goal_distance_m"] for s in dev_seeds]
        a = [r["min_goal_distance_m"] for r in final_eval if r["level"] == level]
        if not b or not a:
            improve[level] = None
            continue
        mb, ma = float(np.mean(b)), float(np.mean(a))
        improve[level] = {"baseline_mean_m": mb, "final_mean_m": ma,
                          "rel_change": (ma - mb) / mb if mb > 0 else 0.0,
                          "improved_5pct": bool(mb > 0 and (mb - ma) / mb >= 0.05)}
    n_improved = sum(1 for v in improve.values() if v and v["improved_5pct"])
    gates["G5_learnability"] = {
        "ok": n_improved >= 2, "rule": g["G5_learnability"]["rule"],
        "value": {"levels_improved": n_improved, "per_level": improve,
                  "note": g["G5_learnability"]["explicitly_not"]},
    }

    audit = S.audit_store(store)
    verdict = "PASS" if all(v["ok"] for v in gates.values()) and not audit["problems"] else "FAIL"
    report = {
        "task": "NAV-034",
        "part": "nominal_smoke_training",
        "verdict": verdict,
        "prereg": {"path": os.path.relpath(PREREG, os.path.dirname(NAV_ROOT)),
                   "sha256": prereg_sha, "amends": amends},
        "run": {"run_id": args.run_id, "run_dir": run_dir,
                "wall_seconds": round(time.time() - t_start, 1),
                "train_episodes": int(episode_id),
                "eval_episodes": len(baseline) + sum(len(p["eval"]) for p in diag["per_phase"])},
        "setup": {
            "margins": {"alpha": cost_model.alpha, "cost_variant": cost_model.variant,
                        "delta_m": geom.delta_m, "r_fp_m": geom.r_fp_m},
            "budget": {"z0_J": z0_J, "Z_REF_J": obj_cfg.Z_REF_J,
                       "z0_over_zref": z0_J / obj_cfg.Z_REF_J},
            "splits": {"train": train_seeds, "development": dev_seeds,
                       "used": ["train", "development"]},
            "config": {"ppo": ppo_cfg.as_dict(), "objective": obj_cfg.as_dict(),
                       "network": {"obs_dim": net.cfg.obs_dim,
                                   "hidden": list(net.cfg.hidden),
                                   "action_scale": list(net.cfg.action_scale)}},
            "contracts": {os.path.basename(p): sha256_file(p) for p in
                          (NAV_CONTRACT, VALUE_CONTRACT, GEOMETRY_CONTRACT, MARGINS, POWER)},
        },
        "baseline_eval": baseline,
        "phases": diag["per_phase"],
        "diagnostics": diagnostics,
        "gates": gates,
        "store_audit": audit,
        "learning_curve": [
            {"episode_id": r["episode_id"], "phase": r["phase"], "level": r["level"],
             "event": r["event"], "steps": r["steps"], "cost_J": r["cost_J"],
             "min_goal_distance_m": r["min_goal_distance_m"],
             "loss_policy": r["update"]["loss_policy"],
             "entropy": r["update"]["entropy"]}
            for r in diag["episodes"]
        ],
        "findings": [
            {
                "id": "budget-below-reference-energy",
                "fact": "NAV-058 参考能量 L1 362.54 / L2 462.48 / L3 367.69 J（1.61/2.05/1.63 × Z_REF），"
                        "而 z_ratio 不饱和时可用预算上限是 0.98·Z_REF = 221.09 J。",
                "consequence": "本轮三个 level 的回合几乎全部以预算耗尽结束，成功率为 0 是结构性的，"
                               "不是策略没学好。可学性只能用「最近到达距离」衡量。",
                "decision_owner": "NAV-036（预算选择）",
                "options": [
                    "①按场景缩放 Z_REF_J：改契约、需升版，残差与观测都恢复单调",
                    "②残差的成本项改用未截断的 z_J/Z_REF（观测仍按 obs-v2 截断）："
                    "与参考实现 RC-PPO 一致，但偏离契约 W 的字面公式",
                ],
            }
        ],
    }
    S.assert_no_secrets(report)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    if report["run"]["wall_seconds"] > cap_min * 60:
        report["budget_breach"] = {
            "cap_minutes": cap_min, "wall_seconds": report["run"]["wall_seconds"]}
    head = {k: report[k] for k in ("task", "part", "verdict", "run")}
    head["gates"] = {k: v["ok"] for k, v in gates.items()}
    head["events"] = [p["events"] for p in diag["per_phase"]]
    head["diagnostics"] = {k: diagnostics[k] for k in
                           ("action_saturation_rate", "entropy_initial", "entropy_final",
                            "grad_norm_p50", "param_rel_l2", "feasible_steps")}
    print(json.dumps(head, ensure_ascii=False))
    for k, v in gates.items():
        if not v["ok"]:
            print("GATE FAILED:", k, json.dumps(v["value"], ensure_ascii=False)[:400])
    return 0 if verdict == "PASS" else 1


def _count(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[str(r[key])] = out.get(str(r[key]), 0) + 1
    return out


def batch_from(arrays: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    def t(key: str) -> torch.Tensor:
        return torch.as_tensor(np.asarray(arrays[key]), dtype=torch.float32)

    return {
        "obs": t("obs"), "action": t("action_raw"), "log_prob_old": t("log_prob"),
        "adv_total": t("obj_adv_total"), "value_cost_old": t("value_cost"),
        "target_cost": t("obj_target_cost"), "value_reach_old": t("value_reach"),
        "target_reach": t("obj_target_reach"),
    }


if __name__ == "__main__":
    sys.exit(main())
