"""轨迹缓冲与可恢复短训练循环的验证（NAV-032）。

要证的两组事情：

**一、离线可重建**（acceptance 1）
    把 chunk 从盘上读回来，只拿**原始字段**（观测/动作/log_prob/双 value/y/z/能量/
    终止标记）重算全部目标，与落盘值逐位相同；轨迹索引、样本数、终止掩码都能
    独立对账。重算走两条独立路径：仓库里的 ``compute_objectives``，以及本文件里
    手写的 O(T²) 暴力实现（不 import ``targets`` 的矩阵版本），后者用来防"两边
    共用同一个错"。

**二、中断-恢复等价于连续跑**（acceptance 2）
    同一轮短训练跑三遍：连续跑、中途停-续、写完 chunk 就崩-再恢复。三者的
    模型/优化器/RNG/回合种子流/样本内容必须**逐位相同**，日志里每个回合
    只出现一次（无重复写入）。

    这里刻意用了两种中断位置：
      * 停在回合中途（半截回合必须靠环境快照接上）
      * 停在"chunk 已提交、更新与存点都没做"（只能靠确定性重放接上）

Run::

    <frozen python> CaMRA/navigation/tests/test_rollout_resume.py \
        --json CaMRA/navigation/reports/rollout_resume_check.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(CALIB, "src"))

from camra_navigation.algorithm import storage as S  # noqa: E402

TRAIN_SCRIPT = os.path.join(NAV_ROOT, "scripts", "train_nominal.py")
VALUE_CONTRACT = os.path.join(NAV_ROOT, "config", "value_objective_contract_v1.json")

RESULTS: list[dict[str, Any]] = []

#: 短训练规模。回合数不多但足够覆盖"停在中途"和"崩在 chunk 之后"两种位置。
EPISODES = 3
STOP_AT = 5


def check(name: str, ok: bool, detail: Any) -> bool:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    return bool(ok)


def load_trainer_module():
    spec = importlib.util.spec_from_file_location("nav032_train", TRAIN_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------- 独立参考实现
def brute_force_objectives(arrays: dict[str, np.ndarray], cfg: S.ObjectiveConfig,
                           terminal_kind: str) -> dict[str, np.ndarray]:
    """手写 O(T²) 版本：不调用 targets 的任何矩阵函数。"""
    c = np.asarray(arrays["cost_norm"], dtype=np.float64)
    z = np.asarray(arrays["z_ratio"], dtype=np.float64)
    vc = np.asarray(arrays["value_cost"], dtype=np.float64)
    vh = np.asarray(arrays["value_reach"], dtype=np.float64)
    y = np.asarray(arrays["safety_y"], dtype=np.float64)
    d = np.asarray(arrays["goal_distance_m"], dtype=np.float64)
    Tn = c.shape[0]

    g = np.empty(Tn)
    for t in range(Tn):
        g[t] = cfg.G_INSIDE if d[t] <= cfg.r_goal_m else (d[t] - cfg.r_goal_m) / cfg.L_REF_m
    h = np.empty(Tn)
    for t in range(Tn):
        h[t] = max(g[t], cfg.C * y[t])
    if terminal_kind == "failure":
        h[Tn - 1] = cfg.fail_penalty
    elif terminal_kind == "success":
        h[Tn - 1] = g[Tn - 1]
    h_ext = np.concatenate([h, [h[Tn - 1]]])

    # 成本目标：显式双层求和
    term = float(np.asarray(arrays["bootstrap_value_cost"])[-1]) if terminal_kind == "truncation" else 0.0
    target_cost = np.empty(Tn)
    for t in range(Tn):
        target_cost[t] = sum(c[s] for s in range(t, Tn)) + term
    if terminal_kind == "failure":
        target_cost[Tn - 1] = cfg.fail_penalty

    # reach 目标：min 型 Bellman 的朴素反向递推
    vdp = np.empty(Tn + 1)
    vdp[Tn] = h_ext[Tn]
    for t in range(Tn - 1, -1, -1):
        vdp[t] = min(h_ext[t], (1.0 - cfg.gamma_h) * h_ext[t] + cfg.gamma_h * vdp[t + 1])

    # 停止分段：完整 W 矩阵，逐行取首个最小
    stop_index = np.empty(Tn, dtype=np.int64)
    stop_residual = np.empty(Tn)
    for t in range(Tn):
        best_val, best_tau = np.inf, -1
        for tau in range(t, Tn + 1):
            spent = sum(c[s] for s in range(t, tau))
            if tau == Tn:
                spent += term
            w = max(h_ext[tau], spent - z[t])
            if w < best_val:
                best_val, best_tau = w, tau
        stop_index[t] = best_tau
        stop_residual[t] = best_val

    return {
        "obj_g_norm": g,
        "obj_h": h,
        "obj_target_cost": target_cost,
        "obj_target_reach": vdp[:Tn],
        "obj_stop_index": stop_index,
        "obj_stop_residual": stop_residual,
        "obj_stop_now": (stop_index == np.arange(Tn)),
        "obj_total_value": np.maximum(vh, vc - z),
        "obj_adv_cost": target_cost - vc,
        "obj_adv_reach": vdp[:Tn] - vh,
        "obj_adv_total": stop_residual - np.maximum(vh, vc - z),
        "obj_feasible": stop_residual <= -cfg.beta,
    }


def reset_alias_problems(chunks: list[dict[str, np.ndarray]]) -> list[str]:
    """检出"把复位后的新状态当成旧回合 bootstrap"这一错法。

    合法数据里，回合末步的 ``next_obs`` 是**终止状态**；下一回合的首步 ``obs``
    是复位后的状态。两者在有初始扰动时不可能逐位相同。
    """
    out: list[str] = []
    for i in range(len(chunks) - 1):
        last_next = np.asarray(chunks[i]["next_obs"][-1], dtype=np.float32)
        next_first = np.asarray(chunks[i + 1]["obs"][0], dtype=np.float32)
        if np.array_equal(last_next, next_first):
            out.append(f"chunk {i} 的末步 next_obs 与 chunk {i + 1} 的首步 obs 逐位相同"
                       "（疑似用了复位后的状态做 bootstrap）")
    return out


# --------------------------------------------------------------- 状态比较
def diff_states(a: Any, b: Any, path: str = "") -> list[str]:
    """递归比较两份状态；只报差异的路径，不打印大张量。"""
    out: list[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}/{k}: 只在恢复后出现")
            elif k not in b:
                out.append(f"{path}/{k}: 只在连续跑里出现")
            else:
                out.extend(diff_states(a[k], b[k], f"{path}/{k}"))
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: 长度 {len(a)} ≠ {len(b)}"]
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(diff_states(x, y, f"{path}[{i}]"))
        return out
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            return [f"{path}: 一边是张量另一边不是"]
        if a.shape != b.shape:
            return [f"{path}: 形状 {tuple(a.shape)} ≠ {tuple(b.shape)}"]
        if not torch.equal(a.cpu(), b.cpu()):
            return [f"{path}: 张量不相等（max|Δ|="
                    f"{float((a.cpu().float() - b.cpu().float()).abs().max()):.3e}）"]
        return out
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not np.array_equal(np.asarray(a), np.asarray(b)):
            return [f"{path}: 数组不等"]
        return out
    if a != b:
        out.append(f"{path}: {a!r} ≠ {b!r}")
    return out


def checkpoint_state(run_dir: str) -> dict[str, Any]:
    ck = S.atomic_load(os.path.join(run_dir, "state.pt"))
    # manifest 的整文件散列不值一比：它含 run_id 与 npz 的 zip 时间戳，
    # 换个 run 目录就必然不同。chunk 的 content_sha256 才是内容，另行逐位比。
    buf = {k: v for k, v in ck["buffer"].items() if k != "manifest_sha256"}
    return {
        "model": ck["model"],
        "optimizers": ck["optimizers"],
        "rng": ck["rng"],
        "counters": ck["counters"],
        "normalization": ck["normalization"],
        "config": ck["config"],
        "buffer": buf,
    }


def chunk_arrays(run_root: str, run_id: str) -> tuple[list[dict[str, np.ndarray]], list[str]]:
    store = S.RolloutStore(run_root, run_id, obs_dim=41)
    arrays = [store.load_chunk(i) for i in range(store.n_chunks)]
    return arrays, [e["content_sha256"] for e in store.entries]


def log_records(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "train_log.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# --------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--keep", action="store_true", help="保留运行目录便于检查")
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    TN = load_trainer_module()

    root = tempfile.mkdtemp(prefix="nav032_")
    cfg = S.ObjectiveConfig.from_contracts(VALUE_CONTRACT, r_goal_m=0.5)

    def make(run_id: str, extra: list[str]):
        argv = ["--run-root", root, "--run-id", run_id,
                "--episodes", str(EPISODES), "--jitter-m", "0.05",
                "--z0-j", "180.0", "--seed-base", "9100"] + extra
        return TN.Trainer(TN.build_argparser().parse_args(argv))

    # ---- 连续跑 ----
    a = make("cont", [])
    outcome_a = a.run()
    run_a = a.run_dir

    # ---- 中途停-续（回合中途）----
    b1 = make("stopgo", ["--stop-after-samples", str(STOP_AT)])
    outcome_b1 = b1.run()
    b2 = make("stopgo", ["--resume"])
    outcome_b2 = b2.run()
    run_b = b2.run_dir

    # ---- 崩在 chunk 之后（更新与存点都没做）----
    c1 = make("crash", ["--crash-after-chunk", "0"])
    crashed = False
    try:
        c1.run()
    except SystemExit as exc:
        crashed = int(exc.code or 0) == 90
    c2 = make("crash", ["--resume"])
    outcome_c = c2.run()
    run_c = c2.run_dir

    check("three_runs_completed",
          outcome_a == "finished" and outcome_b2 == "finished" and outcome_c == "finished",
          {"continuous": outcome_a, "stop_then_resume": [outcome_b1, outcome_b2],
           "crash_then_resume": outcome_c})
    check("stop_landed_mid_episode",
          outcome_b1 == "stopped_after_samples" and b1.episodes_done == 0
          and b1.seg is not None and len(b1.seg) == STOP_AT,
          {"outcome": outcome_b1, "episodes_done": b1.episodes_done,
           "partial_samples": 0 if b1.seg is None else len(b1.seg)})
    check("crash_injection_fired", crashed, {"exit_code_expected": 90})

    # ================= 一、离线可重建 =================
    store_a = S.RolloutStore(root, "cont", obs_dim=41)
    audit = S.audit_store(store_a)
    check("store_audit_clean", not audit["problems"],
          {"problems": audit["problems"], "n_chunks": audit["n_chunks"],
           "n_samples": audit["n_samples"], "events": audit["event_counts"],
           "terminal_kinds": audit["terminal_kind_counts"]})

    chunks, hashes = chunk_arrays(root, "cont")
    total = sum(len(c["global_step"]) for c in chunks)
    steps = np.concatenate([c["global_step"] for c in chunks])
    check("trajectory_index_contiguous",
          total == audit["n_samples"] and np.array_equal(steps, np.arange(total)),
          {"n_samples": total, "first": int(steps[0]), "last": int(steps[-1]),
           "sum_of_chunks": total})

    per_ep = []
    ok_counts = True
    for i, c in enumerate(chunks):
        n = len(c["global_step"])
        per_ep.append({"chunk": i, "episode_id": int(c["episode_id"][0]), "n_samples": n,
                       "t_first": int(c["t"][0]), "t_last": int(c["t"][-1])})
        if int(c["t"][0]) != 0 or not np.array_equal(c["t"], np.arange(n)):
            ok_counts = False
        if len(set(np.asarray(c["episode_id"]).tolist())) != 1:
            ok_counts = False
    check("per_episode_counts_and_index", ok_counts, {"episodes": per_ep})

    # 后继一致性 + 复位不串味
    succ_ok = True
    for c in chunks:
        if not np.array_equal(c["obs"][1:], c["next_obs"][:-1]):
            succ_ok = False
    check("successor_observation_consistent", succ_ok,
          {"note": "同回合内 obs[t+1] 必须等于 next_obs[t]（逐位）"})

    real_alias = reset_alias_problems(chunks)
    fake = [dict(chunks[0]), dict(chunks[1])]
    fake[0]["next_obs"] = np.array(fake[0]["next_obs"], copy=True)
    fake[0]["next_obs"][-1] = np.asarray(fake[1]["obs"][0], dtype=np.float32)
    fake_alias = reset_alias_problems(fake)
    check("reset_state_not_used_as_bootstrap", not real_alias and bool(fake_alias),
          {"real_problems": real_alias, "negative_control_hits": len(fake_alias),
           "note": "真实数据无命中；把复位后状态硬塞进末步 next_obs 时检测器必须报出来"})

    # 终止掩码
    mask_ok = True
    mask_rows = []
    for i, c in enumerate(chunks):
        term = np.asarray(c["terminated"], dtype=bool)
        trunc = np.asarray(c["truncated"], dtype=bool)
        kind = str(store_a.entries[i]["terminal_kind"])
        expect_term = kind in ("failure", "success")
        if int(term.sum()) != (1 if expect_term else 0):
            mask_ok = False
        if int(trunc.sum()) != (0 if expect_term else 1):
            mask_ok = False
        if not bool(term[-1] or trunc[-1]) or bool(term[-1] and trunc[-1]):
            mask_ok = False
        mask_rows.append({"chunk": i, "event": store_a.entries[i]["event"],
                          "terminal_kind": kind,
                          "terminated_steps": int(term.sum()),
                          "truncated_steps": int(trunc.sum())})
    check("termination_mask_single_and_split", mask_ok, {"rows": mask_rows})

    # 只用原始字段重算，与落盘值逐位比较（两条独立路径）
    raw_only = []
    recompute_bad: list[str] = []
    brute_bad: list[str] = []
    for i, c in enumerate(chunks):
        kind = str(store_a.entries[i]["terminal_kind"])
        raw = {k: v for k, v in c.items() if not k.startswith("obj_")}
        back = S.compute_objectives(raw, cfg, terminal_kind=kind)
        for key, val in back.items():
            if not np.allclose(np.asarray(c[key]), np.asarray(val), rtol=0.0, atol=0.0):
                recompute_bad.append(f"chunk {i}: {key}")
        bf = brute_force_objectives(raw, cfg, kind)
        for key, val in bf.items():
            if not np.allclose(np.asarray(c[key]), np.asarray(val), rtol=0.0, atol=1e-12):
                brute_bad.append(f"chunk {i}: {key} "
                                 f"(max|Δ|={np.max(np.abs(np.asarray(c[key]) - np.asarray(val))):.3e})")
        raw_only.append({"chunk": i, "terminal_kind": kind, "n_samples": len(c["global_step"])})
    check("objectives_recomputed_from_raw_fields", not recompute_bad,
          {"problems": recompute_bad, "chunks": raw_only})
    check("objectives_match_independent_bruteforce", not brute_bad,
          {"problems": brute_bad,
           "note": "手写 O(T²) 版本不调用 targets 的矩阵实现，tolerance 1e-12"})

    # 目标语义抽查：终止项与停止分段
    sem = []
    sem_ok = True
    for i, c in enumerate(chunks):
        kind = store_a.entries[i]["terminal_kind"]
        c_arr = np.asarray(c["cost_norm"], dtype=np.float64)
        tc = np.asarray(c["obj_target_cost"], dtype=np.float64)
        if kind == "truncation":
            want_last = float(np.asarray(c["bootstrap_value_cost"])[-1])
            ok = abs(tc[-1] - (c_arr[-1] + want_last)) < 1e-12
        elif kind == "failure":
            ok = abs(tc[-1] - cfg.fail_penalty) < 1e-12
        else:
            ok = abs(tc[-1] - c_arr[-1]) < 1e-12
        stop_now = int(np.asarray(c["obj_stop_now"]).sum())
        idx = np.asarray(c["obj_stop_index"], dtype=np.int64)
        ok &= bool((idx >= np.arange(len(idx))).all())
        sem_ok &= bool(ok)
        sem.append({"chunk": i, "terminal_kind": kind,
                    "target_cost_last": float(tc[-1]),
                    "cost_last": float(c_arr[-1]),
                    "bootstrap_last": float(np.asarray(c["bootstrap_value_cost"])[-1]),
                    "stop_now_steps": stop_now, "ok": bool(ok)})
    check("terminal_bootstrap_semantics", sem_ok, {"rows": sem,
          "note": "failure 末步目标=FAIL_PENALTY；truncation 末步=成本+V_c(x_T)；success=成本"})

    # 成功分支真实回合里没出现（随机策略三回合都没到目标），单独造一段：
    # 同一组数，只改终止性质，用来验"success 不给 bootstrap"和"成功步的 V_h 取 g_norm"。
    syn = {
        "cost_norm": np.full(4, 0.01),
        "z_ratio": np.linspace(0.5, 0.4, 4),
        "value_cost": np.zeros(4),
        "value_reach": np.zeros(4),
        "safety_y": np.array([0.0, 0.0, 0.0, 1.0]),
        "goal_distance_m": np.array([1.5, 0.9, 0.45, 0.2]),
        "terminated": np.array([False, False, False, True]),
        "truncated": np.array([False, False, False, False]),
        "bootstrap_value_cost": np.array([0.0, 0.0, 0.0, 0.7]),
    }
    ok_succ = S.compute_objectives(syn, cfg, terminal_kind="success")
    trunc_syn = dict(syn)
    trunc_syn["terminated"] = np.array([False, False, False, False])
    trunc_syn["truncated"] = np.array([False, False, False, True])
    ok_trunc = S.compute_objectives(trunc_syn, cfg, terminal_kind="truncation")
    check("success_branch_ignores_bootstrap",
          abs(ok_succ["obj_target_cost"][-1] - 0.01) < 1e-12
          and abs(ok_succ["obj_target_reach"][-1] - cfg.G_INSIDE) < 1e-12
          and abs(ok_succ["obj_h"][-1] - cfg.G_INSIDE) < 1e-12
          and abs(ok_trunc["obj_target_cost"][-1] - 0.71) < 1e-12,
          {"success_target_cost_last": float(ok_succ["obj_target_cost"][-1]),
           "success_target_reach_last": float(ok_succ["obj_target_reach"][-1]),
           "success_h_last": float(ok_succ["obj_h"][-1]),
           "y_last": 1.0,
           "truncation_target_cost_last": float(ok_trunc["obj_target_cost"][-1]),
           "note": "同样 0.01 成本 + 0.7 bootstrap：success 用 0.01，truncation 用 0.71；"
                   "成功步即使 y=1 也取 g_norm 而非 C·y"})

    saturated = dict(syn)
    saturated["z_ratio"] = np.ones(4)
    guard_raised = False
    try:
        S.compute_objectives(saturated, cfg, terminal_kind="success")
    except S.StorageError as exc:
        guard_raised = "饱和" in str(exc)
    check("saturated_z_ratio_refused", guard_raised,
          {"note": "z_ratio 触到 ±1 时 W 定义在截断量上，直接拒算而不是给个看着像数的结果"})

    # ================= 二、中断-恢复等价于连续跑 =================
    state_a = checkpoint_state(run_a)
    state_b = checkpoint_state(run_b)
    state_c = checkpoint_state(run_c)
    diff_b = diff_states(state_a, state_b)
    diff_c = diff_states(state_a, state_c)
    check("resume_matches_continuous_state", not diff_b,
          {"differences": diff_b[:20], "n_differences": len(diff_b),
           "note": "模型/优化器/RNG/计数器/归一化/配置全部逐位比较"})
    check("crash_replay_matches_continuous_state", not diff_c,
          {"differences": diff_c[:20], "n_differences": len(diff_c)})

    chunks_b, hashes_b = chunk_arrays(root, "stopgo")
    chunks_c, hashes_c = chunk_arrays(root, "crash")
    same_b = hashes == hashes_b and len(chunks) == len(chunks_b)
    same_c = hashes == hashes_c and len(chunks) == len(chunks_c)
    sample_diff_b = []
    sample_diff_c = []
    for i in range(min(len(chunks), len(chunks_b))):
        for key in ("obs", "action_raw", "log_prob", "obj_adv_total"):
            if not np.array_equal(chunks[i][key], chunks_b[i][key]):
                sample_diff_b.append(f"chunk {i}/{key}")
    for i in range(min(len(chunks), len(chunks_c))):
        for key in ("obs", "action_raw", "log_prob", "obj_adv_total"):
            if not np.array_equal(chunks[i][key], chunks_c[i][key]):
                sample_diff_c.append(f"chunk {i}/{key}")
    check("resume_writes_identical_chunks",
          same_b and not sample_diff_b,
          {"chunk_hashes_equal": same_b, "differences": sample_diff_b, "n_chunks": len(chunks_b)})
    check("crash_replay_writes_identical_chunks",
          same_c and not sample_diff_c,
          {"chunk_hashes_equal": same_c, "differences": sample_diff_c, "n_chunks": len(chunks_c)})

    # 无重复写入：日志里每个 episode 恰好一条"正常"记录，chunk 数 = 回合数
    logs_a = log_records(run_a)
    logs_b = log_records(run_b)
    logs_c = log_records(run_c)
    ep_lines = lambda recs: [r for r in recs if r.get("kind") == "episode"]
    per_ep_a = [r["episode_id"] for r in ep_lines(logs_a)]
    per_ep_b = [r["episode_id"] for r in ep_lines(logs_b)]
    per_ep_c = [r["episode_id"] for r in ep_lines(logs_c)]
    check("no_duplicate_episode_writes",
          per_ep_a == list(range(EPISODES)) and sorted(per_ep_b) == list(range(EPISODES))
          and sorted(per_ep_c) == list(range(EPISODES))
          and len(hashes) == len(hashes_b) == len(hashes_c) == EPISODES,
          {"continuous": per_ep_a, "stop_then_resume": per_ep_b, "crash_then_resume": per_ep_c,
           "chunks": [len(hashes), len(hashes_b), len(hashes_c)]})

    replayed = c2.resume_info.get("replayed_chunks", [])
    resumed_seg = b2.resume_info.get("chunks_on_disk")
    check("resume_paths_exercised",
          bool(replayed) and resumed_seg == 0,
          {"crash_replayed_chunks": replayed,
           "stop_run_chunks_on_disk_at_resume": resumed_seg,
           "note": "崩-恢复必须走重放；停-恢复必须从半截回合接着跑（当时盘上还没有 chunk）"})

    # ---- 守卫：配置变了不许续跑；状态文件被改要拒绝 ----
    guard_args = TN.build_argparser().parse_args(
        ["--run-root", root, "--run-id", "cont", "--episodes", str(EPISODES + 1),
         "--resume"])
    t_guard = TN.Trainer(guard_args)
    refused = False
    try:
        t_guard.resume()
    except SystemExit as exc:
        refused = "拒绝恢复" in str(exc)
    check("resume_refused_on_config_change", refused,
          {"note": "回合数从 3 改成 4 后必须拒绝，而不是悄悄接着跑"})

    ck_path = os.path.join(run_c, "state.pt")
    with open(ck_path, "rb") as fh:
        payload = bytearray(fh.read())
    payload[-1] ^= 0xFF
    with open(ck_path, "wb") as fh:
        fh.write(payload)
    tamper_refused = False
    try:
        S.atomic_load(ck_path)
    except S.StorageError as exc:
        tamper_refused = "校验和" in str(exc)
    check("tampered_checkpoint_refused", tamper_refused,
          {"note": "改一个字节后 sha256 旁路校验必须拦住"})

    # ---- 凭证 ----
    secret_raised = False
    try:
        S.assert_no_secrets({"run": {"api_key": "abc"}})
    except S.StorageError:
        secret_raised = True
    redacted = S.redact("warning token=abc123 remaining 3")
    bad_keys = []
    for recs in (logs_a, logs_b, logs_c):
        for r in recs:
            stack = [("", r)]
            while stack:
                path, node = stack.pop()
                if isinstance(node, dict):
                    for k, v in node.items():
                        if any(b in str(k).lower() for b in S.FORBIDDEN_KEYS):
                            bad_keys.append(f"{path}/{k}")
                        stack.append((f"{path}/{k}", v))
    check("no_credentials_in_logs",
          secret_raised and redacted == "warning token=*** remaining 3" and not bad_keys,
          {"assert_raised": secret_raised, "redacted": redacted, "bad_keys": bad_keys})

    # ---- 检查点内容完整性（rng/优化器/配置/归一化 都在）----
    ck = S.atomic_load(os.path.join(run_a, "state.pt"))
    need = {"model", "optimizers", "rng", "config", "normalization", "buffer", "counters"}
    norm = ck["normalization"]
    check("checkpoint_carries_rng_optimizers_config_normalization",
          need.issubset(ck.keys())
          and set(ck["optimizers"]) == {"policy", "critic_cost", "critic_reach"}
          and len(ck["rng"]["torch"]["torch"]) > 0
          and "episode_seed_stream" in ck["rng"]
          and norm["has_stateful_normalizer"] is False
          and "per-minibatch" in norm["advantage_normalization"],
          {"keys": sorted(ck.keys()), "normalization": norm,
           "optimizer_roles": sorted(ck["optimizers"])})

    failed = [r for r in RESULTS if not r["ok"]]
    summary = {
        "task": "NAV-032",
        "n_checks": len(RESULTS),
        "n_failed": len(failed),
        "verdict": "PASS" if not failed else "FAIL",
        "config": {"episodes": EPISODES, "stop_at_samples": STOP_AT,
                   "objective": cfg.as_dict()},
        "runs": {"run_root": root, "continuous": run_a, "stop_then_resume": run_b,
                 "crash_then_resume": run_c},
        "open_items": [
            {
                "id": "advantage-sign-vs-contract-wording",
                "issue": "value_objective_contract_v1.json 的 sign_conventions[5] 写"
                         "「优势 A > 0 表示该动作优于平均水平（此处为成本更低）」，"
                         "但同一契约的 training_targets.policy_loss 是"
                         "「最小化 mean(max(ratio·A, clip(ratio)·A))」。两者代数上不能同时成立："
                         "dL/dlogp ∝ A，最小化 L 会压低 A>0 的动作概率，因此 A>0 必须是「更贵」。",
                "resolution": "本轮按代数落地：A = 目标成本 − V（A>0 更贵），与 NAV-031 的有限差分结论一致。"
                              "契约措辞需下次升版更正；本轮不改冻结文件。",
                "affects": ["NAV-031", "NAV-032", "NAV-033"],
            },
            {
                "id": "policy-advantage-not-lambda-smoothed",
                "issue": "参考实现 RC-PPO 的 policy 优势是对「停止即走的残差」做 λ 平滑 GAE；"
                         "本轮用 W_sel − V_total（残差的一阶差）。",
                "resolution": "W_sel 已含整条尾段信息，再做 λ 平滑会把 τ* 之后的步混进来，"
                              "与契约「按 τ* 截断、其后不计入优势」冲突。是否改回 λ 平滑留 NAV-034 做对照。",
                "affects": ["NAV-032", "NAV-034"],
            },
            {
                "id": "reach-bootstrap-uses-measured-h-not-critic",
                "issue": "截断时 V_c 的目标用 critic 的 bootstrap（契约明文：该段成本 + V_c(x_T)），"
                         "而 V_h 的目标用实测 h_T = max(g_norm(x_T), C·y)，不调 critic。",
                "resolution": "与参考实现一致（reach_append 用的就是环境给的终端 reach），"
                              "且避免用被训练的同一个网络给自己造目标。两种口径不同，NAV-033 做对照时要分开处理。",
                "affects": ["NAV-032", "NAV-033"],
            },
        ],
        "checks": RESULTS,
    }
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    head = {k: v for k, v in summary.items() if k != "checks"}
    print(json.dumps(head, ensure_ascii=False))
    for item in failed:
        print("FAILED:", item["check"], json.dumps(item["detail"], ensure_ascii=False)[:600])
    # 写报告时保留运行目录：报告里的路径必须还能查到（否则"可复核"只是句空话）
    if not (args.keep or args.json):
        shutil.rmtree(root, ignore_errors=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
