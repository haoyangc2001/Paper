"""预算账本口径（NAV-064 修正 A）。

问题（2026-11-13 只读诊断）：``maps.generator.route_energy`` 用**名义功率**
``sum P_nom(u)*dt`` 生成 ``E_ref``；而 ``costs.JCostModel.step`` 扣费是
``(1 + alpha) * P_nom(u) * dt``。``alpha`` 属于 full 变体（冻结 margins 里
0.190483），uncalibrated 是 0。于是 full 变体天生差 19%：

    14 m 参考路线，v=0.465 m/s：名义 363 J，full 实扣 432 J
    k=1.0 预算 363 J  =>  即使完美按参考路线走也必然超支

本模块只提供**口径一致的派生量**，不覆盖、不重生成 NAV-036 冻结的
``reports/train_map_reference_energy.json``。所有旧调用点默认走
``budget_accounting="nominal"``（逐位回归）；新口径必须由预登记显式启用。

不改的东西：功率模型、扣费公式、``Z_REF_J``、``z_ratio`` 裁剪、判据
``W <= -beta``、以及任何历史报告文件。
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

_THIS = os.path.abspath(__file__)
# budget/ -> camra_navigation/ -> src/ -> navigation/
NAV_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS))))
CALIB = os.path.join(os.path.dirname(NAV_ROOT), "calibration")
MARGINS_PATH = os.path.join(CALIB, "config", "margins_v1.json")
E_REF_REPORT = os.path.join(NAV_ROOT, "reports", "train_map_reference_energy.json")
OUT_REPORT = os.path.join(NAV_ROOT, "reports", "reference_energy_consistent_v1.json")


class BudgetAccountingError(RuntimeError):
    """账本口径输入不合法或冻结来源散列不符。"""


def sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def alpha_for_variant(variant: str) -> float:
    """从冻结 margins 读出 alpha；uncalibrated 恒为 0（不读文件，避免误用）。"""
    if variant not in ("full", "uncalibrated"):
        raise BudgetAccountingError(f"未知 variant={variant!r}")
    if variant == "uncalibrated":
        return 0.0
    payload = json.load(open(MARGINS_PATH, encoding="utf-8"))
    a = float(payload["alpha"])
    if not (a > 0.0):
        raise BudgetAccountingError(f"margins.alpha 应为正数，得到 {a}")
    return a


def consistent_e_ref(e_ref_nominal_J: float, variant: str) -> float:
    """把名义口径的 E_ref 换算成与扣费一致的预算基准。

    ``consistent = nominal * (1 + alpha(variant))``。uncalibrated 的 alpha 为 0，
    因此返回值与输入逐位相同（这是有意的：它本来就没有口径差）。
    """
    e = float(e_ref_nominal_J)
    if not (e > 0.0):
        raise BudgetAccountingError(f"E_ref 应为正数，得到 {e}")
    return e * (1.0 + alpha_for_variant(variant))


def budget_J(k: float, e_ref_nominal_J: float, variant: str,
             accounting: str = "nominal") -> float:
    """``z0 = k * E_ref``；``accounting`` 决定用哪一口径的 E_ref。

    ``nominal``     —— 旧语义，逐位回归（默认）
    ``consistent``  —— A 修正，预算与扣费同口径
    """
    k = float(k)
    if not (k > 0.0):
        raise BudgetAccountingError(f"k 应为正数，得到 {k}")
    if accounting == "nominal":
        e = float(e_ref_nominal_J)
    elif accounting == "consistent":
        e = consistent_e_ref(e_ref_nominal_J, variant)
    else:
        raise BudgetAccountingError(f"未知 accounting={accounting!r}")
    return k * e


def derive_report() -> dict[str, Any]:
    """生成 24 张图的派生报告；**不写**原冻结报告。"""
    src = json.load(open(E_REF_REPORT, encoding="utf-8"))
    a_full = alpha_for_variant("full")
    rows = []
    for m in src["maps"]:
        e = float(m["e_ref_J"])
        rows.append({
            "map_id": m["map_id"],
            "split": m["split"],
            "e_ref_nominal_J": e,
            "alpha_full": a_full,
            "e_ref_consistent_full_J": consistent_e_ref(e, "full"),
            "e_ref_consistent_uncalibrated_J": consistent_e_ref(e, "uncalibrated"),
            "delta_full_J": consistent_e_ref(e, "full") - e,
        })
    return {
        "schema": "camra-reference-energy-consistent-v1",
        "task": "NAV-064 (fix A)",
        "source": "reports/train_map_reference_energy.json",
        "source_sha256": sha256(E_REF_REPORT),
        "margins": MARGINS_PATH,
        "margins_sha256": sha256(MARGINS_PATH),
        "alpha_full": a_full,
        "alpha_uncalibrated": 0.0,
        "n_maps": len(rows),
        "rule": "z0 = k * E_ref * (1 + alpha(variant))；nominal 口径保持旧语义不变",
        "note": "派生量，不覆盖 NAV-036 冻结报告；旧调用点默认 nominal",
        "maps": rows,
    }


if __name__ == "__main__":
    d = derive_report()
    with open(OUT_REPORT, "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=2)
    print(json.dumps({"written": OUT_REPORT, "n_maps": d["n_maps"],
                      "source_sha256_16": d["source_sha256"][:16],
                      "alpha_full": d["alpha_full"]}, ensure_ascii=False))
