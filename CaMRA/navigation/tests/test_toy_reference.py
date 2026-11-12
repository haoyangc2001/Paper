"""可枚举小问题的独立算法对照（NAV-033）。

做法：把一个**受限**的导航问题缩到能穷举的规模，用"把所有轨迹全列一遍"当精确参考，
核对判据的代数性质。受限问题的定义（下称 toy）：

  * 5×5 格、格距 5 m、动作集只有 4 个方向，每移动一格成本常数 0.1（归一化成本，
    0.1 × Z_REF_J ≈ 22.6 J）。**这不是功率模型的输出**——本步校核判据的代数一致性，
    不校核能量模型。
  * 走出格子或走进墙 = 碰撞终止；走进危险格 = 安全历史 y 置 1 且不可逆；
    走到目标 = 成功终止；步数上限 HORIZON，超了算截断。
  * 可达性、最小成本、最小安全残差全部由穷举得到，不用任何学习量。

四个本质不同的案例（覆盖验收要求的四类）：

  A 可达          开阔地图，目标 4 步可达，最小成本 0.4，预算 0.8
  B 有路但预算不足  同一张图，预算 0.3 < 0.4
  C 先危险后到达    唯一通路穿过危险格：到达时 y=1，h = max(g, C·1) = C
  D 不可达        目标被墙围死，任何预算都不行

被核对的是仓库实现：``targets.select_stop`` / ``stop_residual`` / ``is_feasible`` /
``reach_values`` 与 ``storage.compute_objectives``。参考实现一律在本文件里重写
（O(T²) 双层循环 + 状态空间反向归纳），**不 import** 它们的矩阵版本。

边界声明：本文件只在这个受限问题上把穷举结果叫"精确参考"。4 个离散动作与连续
动作集不同，两者最优值之间没有可比性保证，因此这个结论不能外推成连续导航的
全局最优，也不能用来评价任何连续动作策略。这一点写进报告的 scope_note。

Run::

    <frozen python> CaMRA/navigation/tests/test_toy_reference.py \
        --json CaMRA/navigation/reports/toy_algorithm_validation.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(CALIB, "src"))

from camra_navigation.algorithm import storage as S  # noqa: E402
from camra_navigation.algorithm import targets as T  # noqa: E402

VALUE_CONTRACT = os.path.join(NAV_ROOT, "config", "value_objective_contract_v1.json")

CELL_M = 5.0
MOVE_COST = 0.1
HORIZON = 5
ACTIONS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))
ACTION_NAMES = {(1, 0): "E", (0, 1): "N", (-1, 0): "W", (0, -1): "S"}

RESULTS: list[dict[str, Any]] = []


def check(name: str, ok: bool, detail: Any) -> bool:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    return bool(ok)


def load_constants() -> dict[str, float]:
    """只读契约里的数值约定；参考实现不从这个文件以外取任何常数。"""
    with open(VALUE_CONTRACT, encoding="utf-8") as fh:
        c = json.load(fh)
    q, est = c["quantities"], c["estimator"]
    return {
        "beta": float(q["beta"]["value"]),
        "C": float(q["C"]["value"]),
        "Z_REF_J": float(q["Z_REF_J"]["value"]),
        "L_REF_m": float(q["g_norm"]["L_REF_m"]),
        "G_INSIDE": float(q["g_norm"]["G_INSIDE"]),
        "gamma_h": float(est["gamma_h_init"]),
        "gamma_cost": float(est["gamma_cost"]),
        "fail_penalty": float(c["bootstrap"]["FAIL_PENALTY"]["value"]),
        "r_goal_m": 0.5,
    }


# ---------------------------------------------------------------- toy
@dataclass(frozen=True)
class Toy:
    name: str
    open_cells: frozenset[tuple[int, int]]
    danger: frozenset[tuple[int, int]]
    start: tuple[int, int]
    goal: tuple[int, int]

    def g_norm(self, cell: tuple[int, int], k: dict[str, float]) -> float:
        d = CELL_M * math.hypot(cell[0] - self.goal[0], cell[1] - self.goal[1])
        return k["G_INSIDE"] if d <= k["r_goal_m"] else (d - k["r_goal_m"]) / k["L_REF_m"]

    def h(self, cell: tuple[int, int], y: int, k: dict[str, float]) -> float:
        return max(self.g_norm(cell, k), k["C"] * float(y))

    def successors(self, cell: tuple[int, int], y: int):
        """[(next_cell_or_None, 新 y)]；None 表示撞墙（终止，h 取 FAIL_PENALTY）。"""
        out = []
        for a in ACTIONS:
            nxt = (cell[0] + a[0], cell[1] + a[1])
            if nxt not in self.open_cells:
                out.append((None, y))
            else:
                out.append((nxt, 1 if (nxt in self.danger or y) else 0))
        return out


def build_toys() -> dict[str, Toy]:
    open_all = frozenset((i, j) for i in range(5) for j in range(5))
    return {
        "reachable": Toy("reachable", open_all, frozenset(), (0, 0), (4, 0)),
        "danger_first": Toy("danger_first", frozenset((i, 0) for i in range(5)),
                            frozenset({(3, 0)}), (0, 0), (4, 0)),
        "unreachable": Toy("unreachable",
                           frozenset(c for c in open_all if c not in {(3, 4), (4, 3)}),
                           frozenset(), (0, 0), (4, 4)),
    }


# ---------------------------------------------------------------- 穷举
def enumerate_paths(toy: Toy, k: dict[str, float]) -> list[dict[str, Any]]:
    """列出从起点出发的所有轨迹**及其每个前缀**，不剪枝、不去重。

    撞墙那一步记为"停在原地 + 终止"，该步的 h 直接取 FAIL_PENALTY——
    契约里失败步的 V_c/V_h 目标就是 FAIL_PENALTY，不该拿墙外的距离算。
    """
    out: list[dict[str, Any]] = []

    def record(cells: list[tuple[int, int]], ys: list[int], terminal: str | None,
               moves: str, h_last: float | None = None) -> None:
        h = np.array([toy.h(c, y, k) for c, y in zip(cells, ys)], dtype=np.float64)
        if h_last is not None:
            h[-1] = h_last
        out.append({
            "cells": tuple(cells),
            "ys": tuple(ys),
            "terminal": terminal,
            "moves": moves,
            "costs": np.full(len(cells) - 1, MOVE_COST),
            "h": h,
        })

    def walk(cells: list[tuple[int, int]], ys: list[int], moves: str) -> None:
        record(cells, ys, "goal" if cells[-1] == toy.goal else None, moves)
        if len(cells) - 1 >= HORIZON or cells[-1] == toy.goal:
            return
        for a in ACTIONS:
            name = ACTION_NAMES[a]
            nxt = (cells[-1][0] + a[0], cells[-1][1] + a[1])
            if nxt not in toy.open_cells:
                record(cells + [cells[-1]], ys + [ys[-1]], "collision", moves + name,
                       h_last=k["fail_penalty"])
            else:
                ny = 1 if (nxt in toy.danger or ys[-1]) else 0
                walk(cells + [nxt], ys + [ny], moves + name)

    walk([toy.start], [0], "")
    return out


# ---------------------------------------------------------------- 状态空间 DP
def dp(toy: Toy, k: dict[str, float]):
    """(cell, y, 剩余步数) 上的两个不动点。与"沿一条序列递归"是两套实现。"""

    @lru_cache(maxsize=None)
    def vh(cell: tuple[int, int], y: int, left: int) -> float:
        here = toy.h(cell, y, k)
        if cell == toy.goal or left == 0:
            return here
        best = min(k["fail_penalty"] if nx is None else vh(nx, ny, left - 1)
                   for nx, ny in toy.successors(cell, y))
        return min(here, (1.0 - k["gamma_h"]) * here + k["gamma_h"] * best)

    @lru_cache(maxsize=None)
    def vc(cell: tuple[int, int], left: int) -> float:
        """最小累计成本；不看危险（危险是 V_h 的事），撞墙/超时记 FAIL_PENALTY。"""
        if cell == toy.goal:
            return 0.0
        if left == 0:
            return k["fail_penalty"]
        best = min(k["fail_penalty"] if nx is None else MOVE_COST + vc(nx, left - 1)
                   for nx, _ in toy.successors(cell, 0))
        return min(best, k["fail_penalty"])

    return vh, vc


# ---------------------------------------------------------------- 参考：残差
def brute_matrix(costs: np.ndarray, z: np.ndarray, h_full: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """O(T²) 的 W 矩阵、逐行最优停止、done 与选择值；独立于 targets。"""
    Tn = costs.shape[0]
    W = np.full((Tn, Tn + 1), np.inf)
    for t in range(Tn):
        spent = 0.0
        for tau in range(t, Tn + 1):
            if tau > t:
                spent += costs[tau - 1]
            W[t, tau] = max(float(h_full[tau]), spent - float(z[t]))
    index = np.argmin(W, axis=1)
    residual = W[np.arange(Tn), index]
    return W, index, residual, (index == np.arange(Tn))


def w_from_start(path: dict[str, Any], z0: float) -> tuple[float, int]:
    """从起点看的最优停止残差与 τ*（O(T)，只算 W(0,τ) 一行）。"""
    costs, h = path["costs"], path["h"]
    best, best_tau, spent = float(h[0]), 0, 0.0
    for tau in range(1, h.shape[0]):
        spent += float(costs[tau - 1])
        w = max(float(h[tau]), spent - z0)
        if w < best:
            best, best_tau = w, tau
    return best, best_tau


def z_decreasing(path: dict[str, Any], z0: float) -> np.ndarray:
    costs = path["costs"]
    return z0 - np.concatenate([[0.0], np.cumsum(costs)[:-1]]) if costs.size else np.zeros(0)


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    k = load_constants()
    toys = build_toys()
    cfg = S.ObjectiveConfig.from_contracts(VALUE_CONTRACT, r_goal_m=k["r_goal_m"])
    paths = {name: enumerate_paths(toy, k) for name, toy in toys.items()}

    counts = {}
    for name, ps in paths.items():
        term = {"goal": 0, "collision": 0, None: 0}
        for p in ps:
            term[p["terminal"]] += 1
        counts[name] = {"paths": len(ps), "goal": term["goal"],
                        "collision": term["collision"], "open_ended": term[None],
                        "max_steps": max(len(p["cells"]) - 1 for p in ps)}
    check("toy_cases_enumerated",
          counts["reachable"]["goal"] > 0 and counts["unreachable"]["goal"] == 0
          and counts["danger_first"]["goal"] > 0
          and counts["danger_first"]["paths"] < counts["reachable"]["paths"],
          counts)

    vh_ok, vc_ok = dp(toys["reachable"], k)
    min_safe_cost = vc_ok(toys["reachable"].start, HORIZON)
    vh_ok_danger, _ = dp(toys["danger_first"], k)

    def best_over(name: str, z0: float) -> tuple[float, dict[str, Any]]:
        best, info = math.inf, {}
        for p in paths[name]:
            w, tau = w_from_start(p, z0)
            if w < best:
                best = w
                spent = float(np.sum(p["costs"][:tau]))
                info = {"moves": p["moves"], "terminal": p["terminal"], "stop_tau": tau,
                        "spent_at_stop": spent, "h_at_stop": float(p["h"][tau]),
                        "cells": [list(c) for c in p["cells"][: tau + 1]]}
        return best, info

    # ---- A 可达 ----
    # 注意 h = max(g_norm, C·y)：y=0 时最后的 max(·, 0) 会把 G_INSIDE=−0.001 抹成 0，
    # 所以"到达且没碰过危险"的残差恰好是 0，判据 W <= −beta（beta=0）判为可行。
    w_a, info_a = best_over("reachable", 0.8)
    check("case_A_reachable_feasible",
          w_a <= -k["beta"] and info_a["terminal"] == "goal"
          and abs(info_a["spent_at_stop"] - min_safe_cost) < 1e-12
          and info_a["stop_tau"] == 4
          and abs(info_a["h_at_stop"]) < 1e-12,
          {"w_star": w_a, "info": info_a, "budget": 0.8,
           "dp_min_cost": min_safe_cost,
           "note": "预算 0.8 > 成本 0.4：最优停止就在目标那一步，残差 = h(goal) = 0（可行，判据是 <=）"})

    # ---- B 有路但预算不足 ----
    w_b, info_b = best_over("reachable", 0.3)
    n_goal_b = sum(1 for p in paths["reachable"] if p["terminal"] == "goal")
    check("case_B_route_exists_but_budget_short",
          n_goal_b > 0 and w_b > -k["beta"]
          and abs(w_b - (min_safe_cost - 0.3)) < 1e-12
          and info_b["h_at_stop"] < info_b["spent_at_stop"] - 0.3,
          {"w_star": w_b, "goal_paths": n_goal_b, "min_cost": min_safe_cost, "budget": 0.3,
           "cost_term": info_b["spent_at_stop"] - 0.3, "h_term": info_b["h_at_stop"],
           "note": "路是通的（有 goal 轨迹），但 W = 最小成本−预算 > 0，成本项压过到达项"})

    # ---- C 先危险后到达 ----
    w_c, info_c = best_over("danger_first", 10.0)
    goal_danger = [p for p in paths["danger_first"] if p["terminal"] == "goal"]
    residue_at_goal = max(float(p["h"][-1]) for p in goal_danger) if goal_danger else None
    check("case_C_danger_before_arrival",
          w_c > -k["beta"] and goal_danger and all(max(p["ys"]) == 1 for p in goal_danger)
          and abs(residue_at_goal - k["C"]) < 1e-12,
          {"w_star": w_c, "budget": 10.0, "C": k["C"],
           "residual_if_arriving": residue_at_goal, "info": info_c,
           "goal_paths_all_hit_danger": all(max(p["ys"]) == 1 for p in goal_danger),
           "note": "预算给到 10 也不行：唯一通路穿过危险格，到达那一步 h = max(G_INSIDE, C·1) = C；"
                   "最优其实是「进危险格之前就停」（残差 = 那格之前的最小 h），仍然 > 0"})

    # ---- D 不可达 ----
    w_d, info_d = best_over("unreachable", 10.0)
    h_floor = min(float(p["h"].min()) for p in paths["unreachable"])
    check("case_D_unreachable",
          w_d > -k["beta"] and counts["unreachable"]["goal"] == 0
          and abs(w_d - h_floor) < 1e-12,
          {"w_star": w_d, "h_floor_over_all_prefixes": h_floor, "budget": 10.0, "info": info_d,
           "note": "预算是 10（远大于任何花费），spent−z 恒为负，残差退化为「到过的最安全那格的 h」"})

    # ---- 预算符号翻转点：应恰等于 DP 的最小成本 ----
    rows = []
    mismatch = 0
    for i in range(21):
        z0 = 0.30 + 0.01 * i
        w, _ = best_over("reachable", z0)
        impl = bool(T.is_feasible(np.array([w]), k["beta"])[0])
        ref = bool(z0 + 1e-12 >= min_safe_cost)
        mismatch += int(impl != ref)
        rows.append({"z0": round(z0, 4), "w": round(w, 6), "impl": impl, "ref": ref})
    w_flip, _ = best_over("reachable", min_safe_cost)
    check("budget_sign_flip_at_exact_min_cost",
          mismatch == 0 and abs(w_flip) < 1e-12,
          {"mismatches": mismatch, "dp_min_cost": min_safe_cost, "w_at_min_cost": w_flip,
           "sweep": rows,
           "note": "预算恰好等于最小成本时 W = 0 且判可行（W <= −beta，beta=0）；少一分就不行"})

    # ---- 逐前缀：select_stop 与 O(T²) 暴力全等 ----
    bad: list[str] = []
    n_cmp = 0
    for name in ("reachable", "danger_first", "unreachable"):
        for p in paths[name]:
            if p["costs"].size == 0:
                continue
            costs = p["costs"]
            z = z_decreasing(p, 0.45)
            h_full = p["h"]
            i_i, r_i, d_i = T.select_stop(costs, z, h_full, vc_terminal=0.0, gamma=1.0)
            _, i_r, r_r, d_r = brute_matrix(costs, z, h_full)
            n_cmp += 1
            if not (np.array_equal(i_i, i_r) and np.array_equal(d_i, d_r)
                    and np.allclose(r_i, r_r, rtol=0.0, atol=1e-12)):
                bad.append(f"{name}:{p['moves'] or 'stop'}")
    check("stop_rule_matches_bruteforce_on_every_prefix", not bad,
          {"paths_compared": n_cmp, "mismatches": bad[:10],
           "note": "三条 toy 的全部轨迹（含碰撞终止、含不同长度）逐前缀比 τ*、W、done"})

    # ---- 双 critic 联合判据：max 不是 sum ----
    n_sum_pass, examples = 0, []
    for p in goal_danger:
        reach_term = float(p["h"][-1])
        cost_term = float(np.sum(p["costs"])) - 10.0
        if reach_term + cost_term <= 0 < max(reach_term, cost_term):
            n_sum_pass += 1
            if len(examples) < 3:
                examples.append({"moves": p["moves"], "reach": reach_term, "cost": cost_term,
                                 "max": max(reach_term, cost_term),
                                 "sum": reach_term + cost_term})
    check("joint_criterion_max_not_sum", n_sum_pass > 0,
          {"cases_where_sum_would_pass": n_sum_pass, "examples": examples,
           "note": "「装得进预算但到不了」：求和会抵消成负数（误判可行），取 max 不会——"
                   "这正是契约 residual.why_not_sum 的反例"})

    # ---- 危险历史单调 + y=1 之后 V_h >= C ----
    mono = all(all(b >= a for a, b in zip(p["ys"], p["ys"][1:]))
               for ps in paths.values() for p in ps)
    viol = []
    toy_c = toys["danger_first"]
    for cell in toy_c.open_cells:
        for left in range(HORIZON + 1):
            v = vh_ok_danger(cell, 1, left)
            if v < k["C"] - 1e-12:
                viol.append({"cell": list(cell), "left": left, "V_h": v})
    check("danger_history_monotone_and_floor", mono and not viol,
          {"y_monotone_all_paths": mono, "vh_below_C": viol[:5],
           "note": "y 只置 1 不复位；y=1 时 V_h >= C 对所有 (cell, 剩余步数) 成立——"
                   "min 型递推 + h>=C 的直接推论，状态空间穷举验过"})

    # ---- 状态空间反向归纳 vs 序列递归 ----
    toy_a = toys["reachable"]
    cells, ys, left = [toy_a.start], [0], HORIZON
    while cells[-1] != toy_a.goal and left > 0:
        choice = min(toy_a.successors(cells[-1], ys[-1]),
                     key=lambda s: k["fail_penalty"] if s[0] is None
                     else vh_ok(s[0], s[1], left - 1))
        nxt, ny = choice
        if nxt is None:
            break
        cells.append(nxt)
        ys.append(ny)
        left -= 1
    h_seq = np.array([toy_a.h(c, y, k) for c, y in zip(cells, ys)], dtype=np.float64)
    seq = T.reach_values(h_seq, k["gamma_h"])
    dp_seq = np.array([vh_ok(c, y, HORIZON - t) for t, (c, y) in enumerate(zip(cells, ys))])
    dev = float(np.max(np.abs(seq - dp_seq))) if seq.size else 0.0
    check("reach_dp_matches_sequence_recursion",
          dev < 1e-12 and cells[-1] == toy_a.goal,
          {"max_abs_dev": dev, "path_len": len(cells) - 1, "values": seq.tolist(),
           "note": "状态空间反向归纳（本文件）与 targets.reach_values（仓库）在贪心轨迹上逐点相等"})

    # ---- 落到管线：compute_objectives 的终端处理 ----
    pipe_ok = True
    rows_pipe = []
    goal_path = next(p for p in paths["reachable"] if p["terminal"] == "goal")
    collision_path = next(p for p in paths["reachable"] if p["terminal"] == "collision")
    for p, kind, expect in ((goal_path, "success", MOVE_COST),
                            (collision_path, "failure", k["fail_penalty"])):
        Tn = len(p["cells"]) - 1
        z_seq = 0.45 - MOVE_COST * np.arange(Tn)
        dist = [CELL_M * math.hypot(c[0] - 4, c[1] - 0) for c in p["cells"]]
        arr = {
            "cost_norm": np.full(Tn, MOVE_COST),
            # schema v2 (NAV-062): no suffix = x_t, *_next = x_{t+1}
            "z_norm": z_seq,
            "z_ratio": z_seq.copy(),
            "value_cost": np.zeros(Tn),
            "value_reach": np.zeros(Tn),
            "safety_y": np.asarray(p["ys"][:-1], dtype=np.float64),
            "goal_distance_m": np.array(dist[:-1]),
            "z_norm_next": z_seq - MOVE_COST,
            "z_ratio_next": z_seq - MOVE_COST,
            "safety_y_next": np.asarray(p["ys"][1:], dtype=np.float64),
            "goal_distance_m_next": np.array(dist[1:]),
            "terminated": np.array([False] * (Tn - 1) + [True]),
            "truncated": np.array([False] * Tn),
            "bootstrap_value_cost": np.zeros(Tn),
        }
        out = S.compute_objectives(arr, cfg, terminal_kind=kind)
        last = float(out["obj_target_cost"][-1])
        rows_pipe.append({"kind": kind, "moves": p["moves"], "target_cost_last": last,
                          "expected": expect,
                          "target_cost": np.asarray(out["obj_target_cost"]).tolist()})
        pipe_ok &= abs(last - expect) < 1e-12

    Tn = 3
    arr_tr = {
        "cost_norm": np.full(Tn, MOVE_COST),
        "z_norm": np.array([0.45, 0.35, 0.25]),
        "z_ratio": np.array([0.45, 0.35, 0.25]),
        "value_cost": np.zeros(Tn),
        "value_reach": np.zeros(Tn),
        "safety_y": np.zeros(Tn),
        "goal_distance_m": np.array([12.5, 10.0, 7.5]),
        "z_norm_next": np.array([0.35, 0.25, 0.15]),
        "z_ratio_next": np.array([0.35, 0.25, 0.15]),
        "safety_y_next": np.zeros(Tn),
        "goal_distance_m_next": np.array([10.0, 7.5, 5.0]),
        "terminated": np.array([False, False, False]),
        "truncated": np.array([False, False, True]),
        "bootstrap_value_cost": np.array([0.0, 0.0, 0.7]),
    }
    tr_last = float(S.compute_objectives(arr_tr, cfg, terminal_kind="truncation")["obj_target_cost"][-1])
    pipe_ok &= abs(tr_last - (MOVE_COST + 0.7)) < 1e-12
    rows_pipe.append({"kind": "truncation", "target_cost_last": tr_last,
                      "expected": MOVE_COST + 0.7})
    check("toy_targets_go_through_storage_pipeline", pipe_ok, {"rows": rows_pipe,
          "note": "success → 实际成本；failure → FAIL_PENALTY；truncation → 成本 + V_c(x_T)"})

    # ---- gamma_c = 1 时 τ*(t) 与 t 无关 ----
    # 用预算紧到让成本项压过到达项的预算（0.1），否则两边都被 h 压住、看不出折扣的影响
    costs, h_full = goal_path["costs"], goal_path["h"]
    z = z_decreasing(goal_path, 0.1)
    i1, r1, _ = T.select_stop(costs, z, h_full, vc_terminal=0.0, gamma=1.0)
    id_, rd, _ = T.select_stop(costs, z, h_full, vc_terminal=0.0, gamma=0.9)
    check("undiscounted_residual_stop_time_independent",
          len(set(i1.tolist())) == 1 and not np.allclose(r1, rd, rtol=0, atol=1e-12),
          {"gamma_1_index": i1.tolist(), "gamma_1_residual": r1.tolist(),
           "gamma_09_index": id_.tolist(), "gamma_09_residual": rd.tolist(),
           "note": "γ_c=1 时 spent 对 t 可加，W(t,τ) 与 t 无关 → 每步重取得到同一个 τ*；"
                   "γ≠1 时 τ* 会随 t 变，这就是契约把 γ_c 钉死在 1 的原因"})

    # ---- 残差对 t 的可加性（能抓住 γ 指数写反这类错误）----
    z80 = z_decreasing(goal_path, 0.8)
    Wm = T.stop_residual(costs, z80, h_full, vc_terminal=0.0, gamma=1.0)
    cum = np.concatenate([[0.0], np.cumsum(costs)])
    W_ref = np.full_like(Wm, np.inf)
    for t in range(costs.shape[0]):
        for tau in range(t, costs.shape[0] + 1):
            W_ref[t, tau] = max(float(h_full[tau]), float(cum[tau]) - 0.8)
    mask = np.isfinite(W_ref)
    ok_add = bool(np.allclose(Wm[mask], W_ref[mask], rtol=0.0, atol=1e-12))
    check("residual_uses_decreasing_budget_correctly", ok_add,
          {"max_abs_dev": float(np.max(np.abs(Wm[mask] - W_ref[mask]))),
           "note": "z(t)=z0−spent(0,t) 代入后 W(t,τ) 必须只依赖 τ（= spent(0,τ)−z0）；"
                   "γ 指数写反或 z 方向写反都会让这条断言失败"})

    failed = [r for r in RESULTS if not r["ok"]]
    summary = {
        "task": "NAV-033",
        "n_checks": len(RESULTS),
        "n_failed": len(failed),
        "verdict": "PASS" if not failed else "FAIL",
        "toy": {
            "grid": "5x5, cell = %.1f m" % CELL_M,
            "move_cost_norm": MOVE_COST,
            "horizon": HORIZON,
            "actions": sorted(set(ACTION_NAMES.values())),
            "constants": k,
            "cases": counts,
            "dp_min_safe_cost": min_safe_cost,
        },
        "scope_note": (
            "本报告的「精确参考」只对这个受限问题成立：4 个离散动作、确定性转移、常数步成本、"
            "5 步上限。它与连续动作导航的最优值之间没有可比性保证，不能据此声称任何连续策略的最优性，"
            "也不能用来评价学习到的策略有多好——只用来核对判据（停止分段、预算符号、联合判据、"
            "危险历史）与穷举结果是否一致。"
        ),
        "checks": RESULTS,
    }
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    print(json.dumps({x: y for x, y in summary.items() if x != "checks"}, ensure_ascii=False))
    for item in failed:
        print("FAILED:", item["check"], json.dumps(item["detail"], ensure_ascii=False)[:500])
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
