"""双评价目标与停止分段的独立验证（NAV-030）。

验证策略是"两条独立路径互证"，不依赖参考实现的自我一致性：
  A. 手算短轨迹：5 类场景（安全到达 / 先撞后到达 / 无目标 / 到达但超支 / 终端-截断），
     每个数字先在本文件里写死，再与 targets.py 比较。
  B. 独立枚举 oracle：完全另写一份 O(T²) 逐点枚举（不同代码路径），随机数据上逐位比对。
  C. 恒等式：γ_c=1 时同一总成本不同发生时刻给出相同残差；λ-return 与显式公式一致；
     min 型 Bellman 收缩。
  D. 参考转写（``_ref_*``）只作交叉核对：无 done 时 GAE 必须与定义一致；
     ``calculate_indexs3`` 的转写结果差异如实记录，不静默修正。

Run:

    <frozen python> CaMRA/navigation/tests/test_targets_oracle.py \
        --json CaMRA/navigation/reports/targets_oracle_check.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))

from camra_navigation.algorithm import targets as T  # noqa: E402

RESULTS: list[dict[str, Any]] = []


def check(name: str, ok: bool, detail: Any) -> bool:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    return bool(ok)


# ---------------------------------------------------------------- 独立枚举 oracle
def bruteforce_stop(cost, z, reach, vc_terminal=0.0, gamma=1.0):
    """完全独立的 O(T²) 枚举：逐 t 逐 τ 直接累加，不复用 targets 的任何函数。"""
    c = list(map(float, cost))
    Tn = len(c)
    idx, wsel = [], []
    for t in range(Tn):
        best_tau, best_w = None, math.inf
        for tau in range(t, Tn + 1):
            acc = 0.0
            for s in range(t, tau):
                acc += c[s] * (gamma ** (s - t))
            if tau == Tn:
                acc += vc_terminal
            w = max(float(reach[tau]), acc - float(z[t]))
            if w < best_w:                     # 严格小于 ⇒ 首个最小值
                best_tau, best_w = tau, w
        idx.append(best_tau)
        wsel.append(best_w)
    return np.array(idx, dtype=int), np.array(wsel, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    obj = T.load_objective()

    # ---- A. 手算场景 -----------------------------------------------------
    # A1 安全到达：预算充裕、目标可达，应一直走到目标集内
    cost = np.array([0.2, 0.2])
    z = np.array([1.0, 1.0])
    reach = np.array([0.5, 0.3, -0.001])
    idx, w, done = T.select_stop(cost, z, reach, vc_terminal=0.0)
    check("hand_safe_arrival",
          idx.tolist() == [2, 2] and abs(w[0] + 0.001) < 1e-12
          and done.tolist() == [0, 0] and bool(T.is_feasible(w).all()),
          {"cost": cost.tolist(), "z": z.tolist(), "reach": reach.tolist(),
           "index": idx.tolist(), "W": w.tolist(), "feasible": T.is_feasible(w).tolist(),
           "hand": "t=0: τ=0→max(0.5,-1.0)=0.5; τ=1→max(0.3,-0.8)=0.3; "
                   "τ=2→max(-0.001,-0.6)=-0.001 最小"})

    # A2 先撞后到达：安全历史被破坏后 h=1.0 恒正，任何终点都不可行，应立刻停
    h_hit = T.safety_step(np.array([-0.001, -0.001, -0.001]), np.array([1, 1, 1]))
    idx2, w2, done2 = T.select_stop(np.array([0.1, 0.1]), np.array([1.0, 1.0]), h_hit)
    check("hand_hit_then_arrive",
          h_hit.tolist() == [1.0, 1.0, 1.0] and idx2.tolist() == [0, 1]
          and done2.tolist() == [1, 1] and not bool(T.is_feasible(w2).any()),
          {"h": h_hit.tolist(), "index": idx2.tolist(), "W": w2.tolist(),
           "done": done2.tolist(),
           "hand": "y=1 ⇒ h=max(g,C·y)=1.0；所有 τ 的 W 都等于 1.0（并列）"
                   "⇒ argmin 取首个，即 τ=t，故 index=[0,1]、done=[1,1]"})

    # A3 无目标（到不了）：reach 恒正，永远不可行
    reach3 = np.array([0.5, 0.4, 0.3, 0.2])
    idx3, w3, _ = T.select_stop(np.array([0.01, 0.01, 0.01]), np.array([1.0, 1.0, 1.0]), reach3)
    check("hand_no_goal",
          idx3.tolist() == [3, 3, 3] and np.allclose(w3, [0.2, 0.2, 0.2])
          and not bool(T.is_feasible(w3).any()),
          {"index": idx3.tolist(), "W": w3.tolist(),
           "hand": "reach 单调降到 0.2 但始终 >0 ⇒ 选最远 τ=3，W=0.2>0 不可行"})

    # A4 到达但超支：已在目标集内，但预算为负
    cost4 = np.array([0.1, 0.1])
    z4 = np.array([-0.2, -0.2])
    reach4 = np.array([-0.001, -0.001, -0.001])
    idx4, w4, done4 = T.select_stop(cost4, z4, reach4)
    check("hand_arrived_but_overspent",
          idx4.tolist() == [0, 1] and abs(w4[0] - 0.2) < 1e-12
          and done4.tolist() == [1, 1] and not bool(T.is_feasible(w4).any()),
          {"index": idx4.tolist(), "W": w4.tolist(),
           "hand": "spent(0,0)=0，z=-0.2 ⇒ W=max(-0.001, 0.2)=0.2 > 0 ⇒ 已到目标但超支，"
                   "不可行；t=1 同理，并列时取 τ=t"})

    # A5 终端/截断：bootstrap 只影响 τ=T 这一列
    cost5 = np.array([0.1, 0.1])
    z5 = np.array([0.5, 0.5])
    reach5 = np.array([0.2, 0.15, 0.1])
    idx5a, w5a, _ = T.select_stop(cost5, z5, reach5, vc_terminal=0.3)
    idx5b, w5b, _ = T.select_stop(cost5, z5, reach5, vc_terminal=1.0)
    check("hand_terminal_bootstrap",
          idx5a.tolist() == [2, 2] and abs(w5a[0] - 0.1) < 1e-12
          and idx5b.tolist() == [1, 1] and abs(w5b[0] - 0.15) < 1e-12,
          {"bootstrap_0.3": {"index": idx5a.tolist(), "W": w5a.tolist()},
           "bootstrap_1.0": {"index": idx5b.tolist(), "W": w5b.tolist()},
           "hand": "b=0.3: τ=2→max(0.1,0.2+0.3-0.5=0.0)=0.1；"
                   "b=1.0: τ=2→max(0.1,0.7)=0.7 反而更差，改选 τ=1→0.15"})

    # ---- B. 独立枚举 oracle ---------------------------------------------
    rng = np.random.default_rng(args.seed)
    max_idx_diff, max_w_diff, n_cases = 0, 0.0, 120
    worst = None
    for _ in range(n_cases):
        Tn = int(rng.integers(1, 9))
        c = rng.uniform(0.0, 0.5, Tn)
        z = rng.uniform(-0.5, 1.0, Tn)
        r = rng.uniform(-0.2, 0.8, Tn + 1)
        b = float(rng.choice([0.0, 0.2, 0.6]))
        g = float(rng.choice([1.0, 0.95]))
        i1, w1, _ = T.select_stop(c, z, r, vc_terminal=b, gamma=g)
        i2, w2 = bruteforce_stop(c, z, r, vc_terminal=b, gamma=g)
        d_idx = int(np.abs(i1 - i2).max())
        d_w = float(np.abs(w1 - w2).max())
        if d_idx > max_idx_diff or d_w > max_w_diff:
            worst = {"T": Tn, "gamma": g, "bootstrap": b}
        max_idx_diff = max(max_idx_diff, d_idx)
        max_w_diff = max(max_w_diff, d_w)
    check("bruteforce_oracle_matches", max_idx_diff == 0 and max_w_diff < 1e-12,
          {"n_cases": n_cases, "max_index_diff": max_idx_diff,
           "max_W_abs_diff": max_w_diff, "worst": worst,
           "note": "三条互相独立的实现：向量化 spent 矩阵 / 逐点枚举 / 手算"})

    # ---- C. 共享停止 vs 分开取时刻 --------------------------------------
    # 构造：τ=0 便宜但没到目标；τ=1/2 到目标但超支。
    # 正确做法（同一 τ 取 max 再 argmin）→ W=0.3 不可行；
    # 错误做法（成本项与安全项各取自己的最优 τ）→ −0.05 误判可行。
    c6 = np.array([0.5, 0.0])
    z6 = np.array([0.2, 0.2])
    reach6 = np.array([0.9, -0.05, -0.05])
    idx6, w6, _ = T.select_stop(c6, z6, reach6)
    wmat = T.stop_residual(c6, z6, reach6)
    cost_term = T.spent_between(c6) - z6[:, None]
    sep_wrong = float(max(cost_term[0].min(), reach6.min()))
    check("shared_stop_not_separable",
          idx6.tolist() == [1, 1] and abs(w6[0] - 0.3) < 1e-12
          and sep_wrong < 0.0 < w6[0]
          and not bool(T.is_feasible(w6[0])) and bool(sep_wrong <= -obj["beta"]),
          {"index": idx6.tolist(), "W_shared": w6.tolist(),
           "W_separate_WRONG": sep_wrong, "W_matrix": wmat.round(6).tolist(),
           "hand": "W(0,0)=max(0.9,-0.2)=0.9; W(0,1)=max(-0.05,0.3)=0.3; "
                   "W(0,2)=max(-0.05,0.3)=0.3 ⇒ 首个最小 τ=1",
           "note": "分开取时刻得 -0.05（判可行），共享停止得 +0.3（判不可行）；"
                   "契约明文要求同一停止时刻，两个 critic 不得各取有利终点"})

    # ---- D. 恒等式 -------------------------------------------------------
    # D1 γ_c=1：同一总成本、不同发生时刻 ⇒ 残差相同
    a = T.stop_residual(np.array([0.3, 0.0]), np.array([0.5, 0.5]),
                        np.array([0.4, 0.4, 0.4]))[0, 2]
    b2 = T.stop_residual(np.array([0.0, 0.3]), np.array([0.5, 0.5]),
                         np.array([0.4, 0.4, 0.4]))[0, 2]
    check("undiscounted_timing_invariance", abs(a - b2) < 1e-12,
          {"W_cost_early": a, "W_cost_late": b2,
           "note": "契约要求未折扣：同样 0.3 的成本发生在第 0 步或第 1 步，W(τ=2) 必须相同"})

    # D2 min 型 Bellman：常量 h ⇒ V_h = h；γ<1 收缩
    h_const = np.full(5, 0.7)
    v_const = T.reach_values(h_const, 0.999)
    h_dec = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    v_dec = T.reach_values(h_dec, 0.999)
    manual = min(1.0, 0.001 * 1.0 + 0.999 * min(0.8, 0.001 * 0.8 + 0.999 * min(
        0.6, 0.001 * 0.6 + 0.999 * min(0.4, 0.001 * 0.4 + 0.999 * 0.2))))
    check("reach_bellman_contraction",
          np.allclose(v_const, 0.7) and abs(v_dec[0] - manual) < 1e-12
          and bool((v_dec <= h_dec + 1e-12).all()),
          {"const_h_V": v_const.tolist(), "decreasing_h_V": v_dec.tolist(),
           "hand_V0": manual, "note": "V_h <= h 恒成立（min 型）；常量 h 时不动点即 h"})

    # D3 成本 GAE：γ=1 且无 done ⇒ 目标 = 剩余成本之和 + bootstrap
    c7 = np.array([0.05, 0.10, 0.15])
    v7 = np.array([0.4, 0.3, 0.2])
    adv7, tgt7 = T.cost_gae(c7, v7, np.zeros(3), last_value=0.18, lam=1.0)
    check("cost_gae_monte_carlo",
          abs(tgt7[0] - (0.30 + 0.18)) < 1e-12
          and abs(tgt7[1] - (0.25 + 0.18)) < 1e-12
          and np.allclose(adv7, tgt7 - v7),
          {"targets": tgt7.tolist(), "advantages": adv7.tolist(),
           "hand": "λ=1、γ=1 ⇒ 目标退化为蒙特卡洛：target[0]=0.05+0.10+0.15+0.18=0.48"})

    # λ=0.95：与显式 λ-return 逐项对照（不是蒙特卡洛，别把两者混为一谈）
    adv7b, tgt7b = T.cost_gae(c7, v7, np.zeros(3), last_value=0.18, lam=0.95)
    # n 步 λ-return（带 bootstrap）的显式展开，不是蒙特卡洛：
    #   Gλ = Σ_{k<n} λ^k c_k + Σ_{k=1}^{n-1} (1−λ)λ^{k−1} V_k + λ^{n−1} V_n
    lam = 0.95
    n = 3
    lam_hand = (sum(lam ** k * c7[k] for k in range(n))
                + sum((1 - lam) * lam ** (k - 1) * v7[k] for k in range(1, n))
                + lam ** (n - 1) * 0.18)
    check("cost_gae_lambda_return",
          abs(tgt7b[0] - lam_hand) < 1e-12 and tgt7b[0] < tgt7[0],
          {"target_lam095": tgt7b[0], "hand_lambda_return": lam_hand,
           "target_lam1": tgt7[0],
           "note": "λ<1 时目标是 λ 加权混合，必然小于等于蒙特卡洛值；两者不可混用"})

    # D4 失败终止：不 bootstrap，改用终止代价
    adv8, tgt8 = T.cost_gae(np.array([0.05, 0.10]), np.array([0.4, 0.3]),
                            np.array([0.0, 1.0]), last_value=0.18)
    check("cost_gae_failure_no_bootstrap",
          abs(tgt8[1] - 0.10) < 1e-12,
          {"targets": tgt8.tolist(),
           "note": "done=1 处 target=c+V，不再加 bootstrap；截断才用 last_value"})

    # D5 到达评价 λ-return 与显式公式一致
    h9 = np.array([0.9, 0.7, 0.5, 0.3])
    vh9 = np.array([0.85, 0.65, 0.45, 0.25])
    adv9, q9 = T.reach_gae(h9, vh9, np.zeros(3), gamma_h=0.99, lam=0.95)
    vdp = T.reach_values(h9, 0.99)
    hand = (1 - 0.95) * vdp[0] + 0.95 * ((1 - 0.95) * vdp[1] + 0.95 * (
        (1 - 0.95) * vdp[2] + 0.95 * ((1 - 0.95) * vdp[3] + 0.95 * vdp[3])))
    check("reach_lambda_return",
          abs(q9[0] - hand) < 1e-12 and abs(adv9[0] - (q9[0] - vdp[0])) < 1e-12,
          {"Q0": q9[0], "hand": hand, "adv0": adv9[0],
           "note": "λ-return 作用于 min 型 DP 值；λ=0.95、γ_h=0.99"})

    # ---- E. 参考转写交叉核对 --------------------------------------------
    c10 = np.array([0.05, 0.10, 0.15])
    v10 = np.array([0.4, 0.3, 0.2])
    a1, t1 = T.cost_gae(c10, v10, np.zeros(3), 0.18)
    a2, t2 = T._ref_gae2(1.0, 0.95, c10, v10, np.zeros(3), np.array([0.18]))
    check("reference_gae2_agrees_without_done",
          float(np.abs(a1 - a2).max()) < 1e-12 and float(np.abs(t1 - t2).max()) < 1e-12,
          {"adv_diff": float(np.abs(a1 - a2).max()),
           "target_diff": float(np.abs(t1 - t2).max()),
           "note": "无 done 时参考的 roll 掩码退化，两条路径必须逐位一致"})

    # 参考 indexs3 的转写：只记录差异，不改写参考副本
    c11 = np.array([0.2, 0.2])
    z11 = np.array([1.0, 1.0])
    r11 = np.array([0.5, 0.3, -0.001])
    e11 = z11.copy()
    ref_idx, ref_done = T._ref_indexs3(1.0, c11, e11, r11[:, None],
                                       np.array([0.0]), np.array([0.0]))
    i_def, w_def, _ = T.select_stop(c11, z11, r11)
    T_len = int(c11.shape[0])
    def_ok = bool((i_def >= np.arange(i_def.shape[0])).all() and (i_def <= T_len).all())
    ref_ok = bool((ref_idx >= np.arange(ref_idx.shape[0])[:, None]).all()
                  and (ref_idx <= T_len).all())
    ref_same = bool(np.array_equal(i_def, ref_idx[:, 0]))
    check("reference_indexs3_crosscheck",
          def_ok,                    # 门禁只认定义侧；参考差异只记录
          {"definition_index": i_def.tolist(),
           "definition_in_range": def_ok,
           "definition_W": w_def.tolist(),
           "reference_index": ref_idx[:, 0].tolist(),
           "reference_done": ref_done[:, 0].tolist(),
           "reference_in_range": ref_ok,
           "same": ref_same,
           "status": "reference transcribed as-is (roll/[::-1]/inf kept)",
           "finding": ("参考转写给出越界索引（t=1 处返回 0 < t），说明 calculate_indexs3 的 "
                       "argmin 位置依赖其内部 roll/反转构造，不能直接当作 τ 使用；"
                       "本任务以契约定义的枚举为准，差异如实记录，不修改参考副本")})

    failed = [r for r in RESULTS if not r["ok"]]
    summary = {
        "task": "NAV-030",
        "n_checks": len(RESULTS),
        "n_failed": len(failed),
        "verdict": "PASS" if not failed else "FAIL",
        "objective": obj,
        "checks": RESULTS,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    head = {k: v for k, v in summary.items() if k != "checks"}
    print(json.dumps(head, ensure_ascii=False))
    for item in failed:
        print("FAILED:", item["check"], json.dumps(item["detail"], ensure_ascii=False)[:400])
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
