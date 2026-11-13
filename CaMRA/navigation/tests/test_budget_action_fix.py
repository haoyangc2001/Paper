"""NAV-064 修正 A+B 的验收测试（预登记 budget_action_fix_prereg_v1.json）。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_NAV_ROOT = os.path.join(_REPO, "CaMRA", "navigation")
sys.path.insert(0, os.path.join(_NAV_ROOT, "src"))
sys.path.insert(0, _NAV_ROOT)
sys.path.insert(0, os.path.join(_REPO, "CaMRA", "calibration", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from camra_navigation.algorithm import networks as NW  # noqa: E402
from camra_navigation.budget import accounting as ACC  # noqa: E402

NAV = _NAV_ROOT
PREREG = os.path.join(NAV, "config", "budget_action_fix_prereg_v1.json")
OLD_REPORT = os.path.join(NAV, "reports", "train_map_reference_energy.json")
NEW_REPORT = os.path.join(NAV, "reports", "reference_energy_consistent_v1.json")
NAV_CONTRACT = os.path.join(NAV, "config", "navigation_contract_v1.json")
WORK = "/pub/data/caohy/work/camra_navigation"
RESULTS: list[dict] = []


def check(name: str, ok: bool, detail: dict) -> None:
    RESULTS.append({"name": name, "ok": bool(ok), "detail": detail})
    print(("  OK   " if ok else "  FAIL ") + name, flush=True)
    if not ok:
        print("       " + json.dumps(detail, ensure_ascii=False, default=str)[:1200], flush=True)


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 and b.std() > 1e-12 else 0.0


def ab1_accounting_closes() -> None:
    """AB1: full 的预算与扣费闭合，uncalibrated 逐位不变。"""
    from camra_calibration.power import PowerModel
    C = os.path.join(_REPO, "CaMRA", "calibration", "config")
    pm = PowerModel.from_dict(json.load(open(os.path.join(C, "power_model_v1.json"))))
    v = 0.465
    row = [v, 0.0, 0.0]
    p = float(pm.step_power([row, row])[0])
    t = 14.0 / v
    nominal = p * t
    a = ACC.alpha_for_variant("full")
    charged = p * (1 + a) * t
    consistent = ACC.consistent_e_ref(nominal, "full")
    ok = (abs(consistent - charged) < 1e-9
          and ACC.consistent_e_ref(nominal, "uncalibrated") == nominal
          and abs(ACC.budget_J(1.0, nominal, "uncalibrated", "nominal") - nominal) < 1e-12
          and abs(ACC.budget_J(1.0, nominal, "full", "nominal") - nominal) < 1e-12)
    check("ab1_consistent_e_ref_closes_charge", ok,
          {"nominal_J": round(nominal, 3), "alpha_full": a,
           "full_charge_J": round(charged, 3), "consistent_J": round(consistent, 3),
           "uncal_unchanged": True,
           "note": "默认 nominal 口径下两个变体的预算与改动前逐位相同"})


def ab2_derived_report() -> None:
    """AB2: 24 图派生报告完整、来源 hash 正确、原冻结报告未被改写。"""
    d = json.load(open(NEW_REPORT, encoding="utf-8"))
    old_hash = hashlib.sha256(open(OLD_REPORT, "rb").read()).hexdigest()
    rows = d["maps"]
    ok = (d["n_maps"] == 24 == len(rows)
          and d["source_sha256"] == old_hash
          and all({"e_ref_nominal_J", "e_ref_consistent_full_J",
                   "e_ref_consistent_uncalibrated_J", "alpha_full"} <= set(r) for r in rows)
          and all(r["e_ref_consistent_uncalibrated_J"] == r["e_ref_nominal_J"] for r in rows)
          and all(abs(r["e_ref_consistent_full_J"] - r["e_ref_nominal_J"] * (1 + d["alpha_full"])) < 1e-9
                  for r in rows))
    check("ab2_derived_report_and_frozen_source_untouched", ok,
          {"n_maps": len(rows), "source_sha16": old_hash[:16],
           "alpha_full": d["alpha_full"],
           "max_delta_J": round(max(r["delta_full_J"] for r in rows), 2)})


def ab3_bounded_offline() -> None:
    """AB3a: 新初始化权重上 B 消除顶格；AB3b: 已发散权重上如实记录 B 无效。

    预登记 v1.1.0 修订：B 是训练期结构性修复，不是旧权重的热补丁。
    """
    nav = json.load(open(NAV_CONTRACT, encoding="utf-8"))
    torch.manual_seed(1)
    o = torch.randn(3000, 41)
    fresh = {}
    for tag, bounded in (("off", False), ("on", True)):
        torch.manual_seed(7)
        net = NW.build(nav, action_bounded=bounded)
        net.eval()
        torch.manual_seed(3)
        with torch.no_grad():
            a = net.policy(o).sample().numpy()
        # "顶格"按物理意义定义：动作离 ±1 小于 1e-6，等价于下发命令与
        # COMMAND_MAX 的差可忽略。tanh 严格 <1 但浮点可到 0.9995，
        # 用 0.999 会把边界效应误判成饱和（假阴性）。
        SAT = 1.0 - 1e-6
        fresh[tag] = {"sat": float(np.mean(np.abs(a) >= SAT)),
                      "max_abs": float(np.abs(a).max())}
    ok_a = fresh["on"]["sat"] == 0.0 and fresh["on"]["max_abs"] < 1.0 \
        and fresh["on"]["sat"] < fresh["off"]["sat"]
    check("ab3a_bounded_removes_saturation_on_fresh_weights", ok_a,
          {"off": fresh["off"], "on": fresh["on"], "sat_threshold": SAT,
           "note": "新初始化：顶格 31.7%→0%，动作严格在 (-1,1)"})

    rows, ok_b = [], True
    if os.path.isdir(WORK):
        for run in ("nav064_r6_full_s0", "nav064_r6_uncalibrated_s0"):
            p = f"{WORK}/{run}/state.pt"
            if not os.path.exists(p):
                continue
            state = torch.load(p, map_location="cpu", weights_only=False)
            vals = {}
            for tag, bounded in (("off", False), ("on", True)):
                net = NW.build(nav, action_bounded=bounded)
                net.load_state_dict(state["model"])
                net.eval()
                torch.manual_seed(2)
                with torch.no_grad():
                    a = net.policy(o).sample().numpy()
                vals[tag] = float(np.mean(np.abs(a) >= 1.0 - 1e-6))
            rows.append({"run": run, "sat_off": round(vals["off"], 3),
                         "sat_on": round(vals["on"], 3),
                         "B_ineffective": vals["on"] > 0.9})
            ok_b &= vals["on"] > 0.9
        ok_b &= len(rows) == 2
    else:
        rows.append({"skipped": "无远端工作目录"})
    check("ab3b_bounded_cannot_rescue_diverged_weights", bool(ok_b),
          {"rows": rows,
           "note": "均值头 |mean| 中位 120~154 时 tanh 直接饱和；"
                   "第 7 轮必须重新初始化策略头，不得热启动 r6 权重"})


def ab4_logprob_normalised() -> None:
    """AB4: 有界模式 log_prob 归一化、熵有限、动作严格在 (-1,1)。"""
    from camra_navigation.algorithm.networks import SquashedNormal
    d = SquashedNormal(torch.zeros(1, 1), torch.ones(1, 1) * 0.8)
    xs = torch.linspace(-0.999999, 0.999999, 100001).reshape(-1, 1)
    with torch.no_grad():
        vals = d.log_prob(xs).exp()
        integ = float((vals[:-1] + vals[1:]).sum() / 2 * (xs[1] - xs[0]))
    a = d.sample(torch.Size([4096]))
    ok = (0.98 < integ < 1.02
          and bool(torch.isfinite(d.entropy()).all())
          and bool((a.abs() < 1.0).all())
          and bool(torch.isfinite(d.log_prob(a)).all()))
    check("ab4_squashed_logprob_normalised", ok,
          {"integral": round(integ, 5), "max_abs_action": round(float(a.abs().max()), 6),
           "entropy": round(float(d.entropy().mean()), 4)})


def ab5_defaults_regression() -> None:
    """AB5: 两个开关都关闭时，网络输出与口径与旧语义逐位相同。"""
    nav = json.load(open(NAV_CONTRACT, encoding="utf-8"))
    torch.manual_seed(11)
    o = torch.randn(8, 41)
    a = NW.build(nav)
    b = NW.build(nav, action_bounded=False)
    b.load_state_dict(a.state_dict())
    torch.manual_seed(23)
    with torch.no_grad():
        sa = a.policy(o).sample()
    torch.manual_seed(23)
    with torch.no_grad():
        sb = b.policy(o).sample()
    same = bool(torch.equal(sa, sb))
    e = 363.0
    f = ACC.budget_J(1.3, e, "full", "nominal")
    check("ab5_defaults_bit_identical", same and abs(f - 1.3 * e) < 1e-12,
          {"network_same": same, "budget_nominal_J": round(f, 3),
           "note": "默认 nominal / action_bounded=False 时全部既有测试应仍全绿"})


def ab6_joint_feasibility() -> None:
    """AB6: A+B 同时启用时，参考路线在 k>=1 处 W<=0（定义上可达）。"""
    from camra_calibration.power import PowerModel
    C = os.path.join(_REPO, "CaMRA", "calibration", "config")
    pm = PowerModel.from_dict(json.load(open(os.path.join(C, "power_model_v1.json"))))
    v = 0.465
    row = [v, 0.0, 0.0]
    p = float(pm.step_power([row, row])[0])
    t = 14.0 / v
    rows = []
    ok = True
    for variant in ("uncalibrated", "full"):
        a = ACC.alpha_for_variant(variant)
        charged = p * (1 + a) * t
        e_nom = p * t
        for k in (0.7, 1.0, 1.15, 1.3):
            z_old = k * e_nom
            z_new = ACC.budget_J(k, e_nom, variant, "consistent")
            w_old = charged - z_old
            w_new = charged - z_new
            rows.append({"variant": variant, "k": k,
                         "W_old": round(w_old, 2), "W_new": round(w_new, 2),
                         "feasible_new": w_new <= 0.0})
            if k >= 1.0:
                ok &= abs(w_new) < 1e-9 or w_new <= 0.0
            if variant == "uncalibrated":
                ok &= abs(w_new - w_old) < 1e-9
            if variant == "full" and k == 1.0:
                ok &= w_old > 0.0 and abs(w_new) < 1e-9
    check("ab6_joint_makes_budget_feasible", bool(ok),
          {"rows": rows,
           "note": "只证明定义上可达（参考路线 + 参考速度）；不宣称现状网络能走出"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    prereg = json.load(open(PREREG, encoding="utf-8"))
    assert prereg["status"] == "frozen_before_code_change", "预登记状态被改动"
    assert prereg["version"] == "1.1.0", "须使用修订后的判据 v1.1.0"

    ab1_accounting_closes()
    ab2_derived_report()
    ab3_bounded_offline()
    ab4_logprob_normalised()
    ab5_defaults_regression()
    ab6_joint_feasibility()

    failed = [r for r in RESULTS if not r["ok"]]
    report = {"task": "NAV-064-budget-action-fix", "n_checks": len(RESULTS),
              "n_failed": len(failed), "verdict": "PASS" if not failed else "FAIL",
              "prereg_sha256": hashlib.sha256(open(PREREG, "rb").read()).hexdigest()[:16],
              "checks": RESULTS}
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2, default=str)
    print(json.dumps({k: report[k] for k in ("task", "n_checks", "n_failed", "verdict")},
                     ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
