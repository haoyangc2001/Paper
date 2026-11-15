"""NAV-064 修正 ①/②：成本信号进入策略梯度、min 型递推的反例。

验收标准先冻结在 config/objective_signal_fix_prereg_v1.json，本文件是它的
可执行形式。所有判据都用确定性数据，不依赖训练随机性。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_NAV_ROOT = os.path.join(_REPO, "CaMRA", "navigation")
sys.path.insert(0, os.path.join(_NAV_ROOT, "src"))
sys.path.insert(0, _NAV_ROOT)

import numpy as np  # noqa: E402

from camra_navigation.algorithm import storage as S  # noqa: E402
from camra_navigation.algorithm import targets as T  # noqa: E402

VALUE_CONTRACT = os.path.join(_NAV_ROOT, "config", "value_objective_contract_v1.json")
PREREG = os.path.join(_NAV_ROOT, "config", "objective_signal_fix_prereg_v1.json")
RESULTS: list[dict] = []


def check(name: str, ok: bool, detail: dict) -> None:
    RESULTS.append({"name": name, "ok": bool(ok), "detail": detail})
    print(("  OK   " if ok else "  FAIL ") + name, flush=True)
    if not ok:
        print("       " + json.dumps(detail, ensure_ascii=False, default=str)[:1400], flush=True)


def cfg(**over) -> S.ObjectiveConfig:
    import dataclasses
    base = S.ObjectiveConfig.from_contracts(VALUE_CONTRACT, r_goal_m=0.5)
    return dataclasses.replace(base, **over) if over else base


def segment(costs, z, dist, y, *, d_next=None, y_next=None, z_next=None,
            term="truncation", vc_terminal=0.0, vc=0.0, vh=0.0):
    n = len(costs)
    d_next = list(d_next if d_next is not None else list(dist[1:]) + [dist[-1]])
    y_next = list(y_next if y_next is not None else list(y[1:]) + [y[-1]])
    z_next = list(z_next if z_next is not None else list(z[1:]) + [z[-1] - costs[-1]])
    ta = np.zeros(n, dtype=bool)
    tr = np.zeros(n, dtype=bool)
    (ta if term != "truncation" else tr)[-1] = True
    b_c = np.zeros(n)
    b_c[-1] = vc_terminal
    return {
        "cost_norm": np.asarray(costs, np.float64),
        "z_norm": np.asarray(z, np.float64),
        "z_ratio": np.asarray(z, np.float64),
        "value_cost": np.full(n, vc, np.float64),
        "value_reach": np.full(n, vh, np.float64),
        "safety_y": np.asarray(y, np.float64),
        "goal_distance_m": np.asarray(dist, np.float64),
        "z_norm_next": np.asarray(z_next, np.float64),
        "z_ratio_next": np.asarray(z_next, np.float64),
        "safety_y_next": np.asarray(y_next, np.float64),
        "goal_distance_m_next": np.asarray(d_next, np.float64),
        "terminated": ta,
        "truncated": tr,
        "bootstrap_value_cost": b_c,
        "bootstrap_value_reach": np.zeros(n),
    }


def syn(n=40, seed=7):
    """一条确定性轨迹：成本有波动、距离递减、预算由正转负。"""
    rng = np.random.default_rng(seed)
    c = rng.uniform(0.002, 0.020, n)
    z = 0.8 - np.cumsum(c)
    dist = np.linspace(18.0, 0.3, n) + rng.normal(0.0, 0.15, n)
    d_next = np.concatenate([dist[1:], [0.2]])
    return segment(c, z, dist, np.zeros(n), d_next=d_next,
                   z_next=z - c, term="truncation", vc_terminal=0.3)


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 and b.std() > 1e-12 else 0.0


BASE = "/pub/data/caohy/work/camra_navigation"
RUNS = {"nav064_r5_full_s0": 0.40,
        "nav064_r5_uncalibrated_s0": 0.12,
        "nav064_r1c_uncalibrated_s0": 0.25}
_CACHE: dict = {}


def real_chunks(run: str, limit: int = 60):
    """返回可直接喂给 compute_objectives 的原始字段；无远端目录时为空列表。"""
    import glob
    if run in _CACHE:
        return _CACHE[run]
    if not os.path.isdir(BASE):
        _CACHE[run] = []
        return []
    out = []
    for f in sorted(glob.glob(f"{BASE}/{run}/rollout_env*/rollout/chunk_*.npz"))[:limit]:
        d = np.load(f)
        raw = {k: v for k, v in d.items() if not k.startswith("obj_")}
        try:
            S.compute_objectives(raw, cfg(), terminal_kind="truncation")
        except S.StorageError:
            continue
        out.append(raw)
    _CACHE[run] = out
    return out


def check_f1_zero_weight_regression():
    """F1: w=0 时逐位等价于改动前（obj_g_train == obj_g，其余不受影响）。"""
    a = syn()
    out = S.compute_objectives(a, cfg(), terminal_kind="truncation")
    ok = np.array_equal(out["obj_g"], out["obj_g_train"])
    # 与"旧公式"直接对照：target_total 必须仍等于 g_ext 上的 λ-return 减 baseline
    g = np.maximum(out["obj_h"], -a["z_norm"])
    g_ext = np.concatenate([g, [float(out["obj_g_terminal"][0])]])
    done = np.concatenate([np.asarray(out["obj_stop_now"], int), [1]])
    want = T.lambda_return(g_ext, done, cfg().gae_lambda)[:len(g)] - out["obj_total_value"]
    ok &= np.allclose(out["obj_target_total"], want, rtol=0.0, atol=0.0)
    check("f1_zero_weight_is_bit_identical", ok,
          {"g_equals_g_train": bool(np.array_equal(out["obj_g"], out["obj_g_train"])),
           "max_abs_delta_target": float(np.max(np.abs(out["obj_target_total"] - want)))})


def check_f2_f3_signal_and_direction():
    """F2/F3: 成本信号进入优势，到达信号方向不翻转（真实训练 chunk）。"""
    rows = []
    ok = True
    have = False
    for run, thr in RUNS.items():
        cw, c0, hw, h0 = [], [], [], []
        for raw in real_chunks(run):
            o0 = S.compute_objectives(raw, cfg(), terminal_kind="truncation")
            ow = S.compute_objectives(raw, cfg(joint_cost_weight=50.0),
                                      terminal_kind="truncation")
            c0.append(corr(raw["cost_norm"], o0["obj_adv_total"]))
            cw.append(corr(raw["cost_norm"], ow["obj_adv_total"]))
            h0.append(corr(o0["obj_h"], o0["obj_adv_total"]))
            hw.append(corr(o0["obj_h"], ow["obj_adv_total"]))
        if not cw:
            rows.append({"run": run, "skipped": True})
            continue
        have = True
        m, m0 = float(np.median(cw)), float(np.median(c0))
        mh, mh0 = float(np.median(hw)), float(np.median(h0))
        passed = m >= thr and m > m0 and (run.startswith("nav064_r5_full") or mh > 0.0)
        ok &= passed
        rows.append({"run": run, "chunks": len(cw), "threshold": thr,
                     "med_w0": round(m0, 4), "med_w50": round(m, 4),
                     "corr_h_w0": round(mh0, 4), "corr_h_w50": round(mh, 4),
                     "pass": passed})
    ok &= have
    check("f2_f3_cost_signal_up_without_flipping_arrival", ok, {"rows": rows})

    # 代数关系：g_train 与 g 只差前 T 项的 w·c（浮点容差）
    a = syn()
    ow = S.compute_objectives(a, cfg(joint_cost_weight=50.0), terminal_kind="truncation")
    delta = ow["obj_g_train"] - ow["obj_g"]
    check("f2_g_train_differs_only_by_weighted_step_cost",
          bool(np.allclose(delta, 50.0 * a["cost_norm"], rtol=0.0, atol=1e-12)),
          {"max_abs_delta": float(np.max(np.abs(delta - 50.0 * a["cost_norm"])))})


def check_f4_criteria_unchanged():
    """F4: W / τ* / 可行性判据 / 成本与 reach 目标逐位不受成本项影响。"""
    a = syn()
    o0 = S.compute_objectives(a, cfg(), terminal_kind="truncation")
    ow = S.compute_objectives(a, cfg(joint_cost_weight=50.0), terminal_kind="truncation")
    keys = ("obj_stop_index", "obj_stop_residual", "obj_feasible",
            "obj_target_cost", "obj_target_reach", "obj_adv_cost",
            "obj_adv_reach", "obj_h", "obj_h_terminal", "obj_g", "obj_spent")
    bad = [k for k in keys if not np.array_equal(o0[k], ow[k])]
    check("f4_feasibility_and_other_targets_unchanged", not bad, {"changed": bad})


def check_f5_min_recursion_counterexample():
    """F5: 启用 min 型递推会翻转到达梯度——默认必须关闭。

    用**按 run 聚合的中位数**判定，不用单 chunk：单回合的 corr 波动大，
    只看"符号翻转的 chunk 数"会得到不稳定的弱判据。
    """
    base = cfg()
    rows, inversions = [], 0
    for run in RUNS:
        h_off_all, h_on_all = [], []
        for raw in real_chunks(run, limit=40):
            off = S.compute_objectives(raw, cfg(joint_cost_weight=50.0),
                                       terminal_kind="truncation")
            on = S.compute_objectives(raw, cfg(joint_cost_weight=50.0,
                                               joint_min_recursion=True),
                                      terminal_kind="truncation")
            h_off_all.append(corr(off["obj_h"], off["obj_adv_total"]))
            h_on_all.append(corr(off["obj_h"], on["obj_adv_total"]))
        if not h_off_all:
            continue
        m_off, m_on = float(np.median(h_off_all)), float(np.median(h_on_all))
        inverted = m_off > 0.0 and m_on < 0.0
        inversions += int(inverted)
        rows.append({"run": run, "chunks": len(h_off_all),
                     "med_h_off": round(m_off, 4), "med_h_on": round(m_on, 4),
                     "inverted": inverted})
    ok = (base.joint_min_recursion is False
          and len(rows) >= 2 and inversions >= 2)
    check("f5_min_recursion_would_invert_arrival_gradient", ok,
          {"default_off": not base.joint_min_recursion,
           "n_runs": len(rows), "n_inversions": inversions, "rows": rows,
           "note": "参考 reach 到达时为 -300；本仓库 h 被 max(·,0) 抹平，"
                   "min 递推把到达方向翻转，见 amend3"})
    g = np.linspace(1.0, 0.0, 12)
    baseline = np.zeros(11)
    done = np.zeros(12, dtype=int)
    v = T.joint_min_return(g, baseline, done, gamma=0.999, lam=0.95,
                           terminal_baseline=0.0)
    ok2 = (v.shape == (11,) and bool(np.all(np.isfinite(v)))
           and bool(np.all(v >= 0.0)) and bool(np.all(v <= float(g.max()) + 1e-9)))
    check("f5_joint_min_return_is_well_defined", ok2,
          {"shape": list(v.shape),
           "v_head": [round(float(x), 5) for x in v[:4]],
           "value_range": [round(float(v.min()), 6), round(float(v.max()), 6)],
           "note": "参考 min 型目标恒在 [min(g), max(g)] 内且有限；"
                   "归一化 GAE 系数是倒序滚动的，不要求单调"})


def check_f6_real_chunk_evidence():
    """F6: 真实 chunk 上 w=50 的绝对阈值（与 F2 的方向判据互补）。"""
    rows, ok, have = [], True, False
    for run, thr in RUNS.items():
        chunks = real_chunks(run, limit=90)
        if not chunks:
            rows.append({"run": run, "skipped": True})
            continue
        vals = []
        for raw in chunks:
            ow = S.compute_objectives(raw, cfg(joint_cost_weight=50.0),
                                      terminal_kind="truncation")
            vals.append(corr(raw["cost_norm"], ow["obj_adv_total"]))
        have = True
        m = float(np.median(vals))
        ok &= m >= thr
        rows.append({"run": run, "chunks": len(vals), "threshold": thr,
                     "median": round(m, 4), "pass": m >= thr})
    ok &= have
    check("f6_real_chunk_thresholds", bool(ok), {"rows": rows})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    prereg = json.load(open(PREREG, encoding="utf-8"))
    assert prereg["status"] == "frozen_before_code_change", "预登记已被改动"
    assert prereg["change_1_cost_signal"]["weight"] == 50.0, "权重与测试不一致"

    check_f1_zero_weight_regression()
    check_f2_f3_signal_and_direction()
    check_f4_criteria_unchanged()
    check_f5_min_recursion_counterexample()
    check_f6_real_chunk_evidence()

    failed = [r for r in RESULTS if not r["ok"]]
    report = {"task": "NAV-064-objective-signal-fix", "n_checks": len(RESULTS),
              "n_failed": len(failed), "verdict": "PASS" if not failed else "FAIL",
              "prereg_sha256": __import__("hashlib").sha256(
                  open(PREREG, "rb").read()).hexdigest()[:16],
              "checks": RESULTS}
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2, default=str)
    print(json.dumps({k: report[k] for k in ("task", "n_checks", "n_failed", "verdict")},
                     ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
