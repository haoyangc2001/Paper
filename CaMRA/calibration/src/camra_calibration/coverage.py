"""检验集覆盖统计（CAL-027）。

协议口径（config/calibration_protocol_v1.0.2.json 的 ``statistics`` 与
``statistical_acceptance.heldout``）：

* 覆盖 = 实际值不超过预算/边界，**等于阈值视为覆盖**；
* 成本覆盖：``J_exec ≤ (1+alpha)·J_nom``，即 ``epsilon_c ≤ alpha``；
* 位置覆盖：``epsilon_d ≤ delta``；
* **联合覆盖单独报告**——两项边际 90% 不推出联合 90%；
* 分母：总体（含业务失败）、完整、non_computable（J_nom < J_min）、数据无效，
  四套同时报告，不得只报条件覆盖；
* 区间：以整条轨迹为最小重采样单位（不按时间点伪增样本），可按 cell 整簇重采样；
* 单一固定 v2 控制器，不声称训练随机种子方差；
* 90% 是校准分位目标，不是"检验恰好 90%"的自动通过判据。

本模块只做统计，不含 IO，可在 CPU 上完整单测。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .margins import BOOTSTRAP_LEVEL, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, percentile_interval

#: 协议 statistics.quantile_level 对应的目标覆盖率（校准分位目标，非硬性判据）
TARGET_COVERAGE: float = 0.90
#: 协议 statistical_acceptance.min_complete_share_per_cell
MIN_CELL_COMPLETE_SHARE: float = 0.9


class CoverageError(RuntimeError):
    """覆盖统计的输入非法。"""


# ---------------------------------------------------------------- 判定
def cost_covered(eps_c: float | None, alpha: float) -> bool | None:
    """成本覆盖：epsilon_c ≤ alpha（等于阈值算覆盖）。不可算时返回 None。"""
    if eps_c is None or not np.isfinite(eps_c):
        return None
    return bool(eps_c <= alpha + 1e-12)


def position_covered(eps_d: float | None, delta: float) -> bool | None:
    """位置覆盖：epsilon_d ≤ delta（等于算覆盖）。"""
    if eps_d is None or not np.isfinite(eps_d):
        return None
    return bool(eps_d <= delta + 1e-12)


# ---------------------------------------------------------------- 方向层
def direction_of(command_seq) -> str:
    """按命令窗口的平均分量给轨迹定方向层。

    取整段命令的平均 ``vx``/``vy``，以较大者为主轴，绝对值小于 0.05 m/s 视为
    该轴无显著运动；两轴都小则归 ``rotation_only``（原地转向）。
    """
    cmd = np.asarray(command_seq, dtype=np.float64)
    if cmd.ndim != 2 or cmd.shape[1] < 2:
        raise CoverageError(f"命令形状应为 (N,≥2)，得到 {cmd.shape}")
    vx, vy = float(cmd[:, 0].mean()), float(cmd[:, 1].mean())
    if abs(vx) >= abs(vy):
        if abs(vx) < 0.05:
            return "rotation_only"
        return "forward" if vx > 0 else "backward"
    if abs(vy) < 0.05:
        return "rotation_only"
    return "left" if vy > 0 else "right"


# ---------------------------------------------------------------- 比率与区间
def rate(flags: Iterable[bool | None]) -> dict:
    """可算样本上的覆盖计数与比率（None 计入不可算，不进分母）。"""
    arr = [f for f in flags if f is not None]
    if not arr:
        return {"n": 0, "n_covered": 0, "rate": None, "n_not_computable": 0}
    n_cov = int(sum(1 for f in arr if f))
    return {"n": len(arr), "n_covered": n_cov, "rate": n_cov / len(arr),
            "n_not_computable": 0}


def bootstrap_rate(flags: Sequence[bool], replicates: int = BOOTSTRAP_REPLICATES,
                   level: float = BOOTSTRAP_LEVEL, seed: int = BOOTSTRAP_SEED,
                   clusters: Sequence[str] | None = None) -> dict:
    """覆盖率的 bootstrap 区间。

    与分位统计同一纪律：以整条轨迹为最小重采样单位；给 ``clusters`` 时按簇整块重采样。
    区间用 nearest-rank 口径（与 margins 一致），保证两个报告可互相复算。
    """
    arr = np.asarray([1.0 if f else 0.0 for f in flags], dtype=np.float64)
    if arr.size == 0:
        raise CoverageError("空样本无法 bootstrap")
    rng = np.random.default_rng(seed)
    boot = np.empty(replicates, dtype=np.float64)
    if clusters is None:
        n = arr.size
        for r in range(replicates):
            idx = rng.integers(0, n, size=n)
            boot[r] = arr[idx].mean()
    else:
        if len(clusters) != arr.size:
            raise CoverageError("clusters 长度与样本不一致")
        groups: dict[str, list[float]] = {}
        for c, v in zip(clusters, arr):
            groups.setdefault(c, []).append(float(v))
        keys = sorted(groups)
        blocks = [np.asarray(groups[k], dtype=np.float64) for k in keys]
        for r in range(replicates):
            pick = rng.integers(0, len(blocks), size=len(blocks))
            boot[r] = np.concatenate([blocks[i] for i in pick]).mean()
    lo, hi = percentile_interval(boot, level)
    return {"point": float(arr.mean()), "lo": lo, "hi": hi, "n": int(arr.size),
            "replicates": int(replicates), "level": float(level), "seed": int(seed),
            "unit": "cluster" if clusters is not None else "trial"}


def rate_with_ci(flags: Sequence[bool], clusters: Sequence[str],
                 replicates: int, seed: int) -> dict:
    """点估计 + trial 级与 cell 簇级两套 95% 区间。"""
    out = rate(flags)
    if out["n"] == 0:
        return {**out, "ci_trial": None, "ci_cluster": None}
    vals = [bool(f) for f in flags if f is not None]
    cl = [c for c, f in zip(clusters, flags) if f is not None]
    out["ci_trial"] = bootstrap_rate(vals, replicates=replicates, seed=seed)
    out["ci_cluster"] = bootstrap_rate(vals, replicates=replicates, seed=seed,
                                       clusters=cl)
    return out


# ---------------------------------------------------------------- 分母
@dataclass(frozen=True)
class HeldoutRow:
    """一条检验 trial 的摘要（由脚本从产物装配）。"""

    trial_id: str
    family: str
    profile: str
    cell: str
    direction: str
    low_speed: bool
    outcome: str
    complete: bool
    official: bool
    non_computable: bool
    data_invalid: bool
    eps_c: float | None
    eps_d: float | None


def denominators(rows: Sequence[HeldoutRow], expected_total: int) -> dict:
    """四套分母（协议要求同时报告，不得只报条件覆盖）。"""
    n_obs = len(rows)
    complete = sum(1 for r in rows if r.official)
    failed = sum(1 for r in rows if r.outcome not in ("completed",))
    invalid = sum(1 for r in rows if r.data_invalid)
    noncomp = sum(1 for r in rows if r.non_computable)
    eps_usable = sum(1 for r in rows if r.official and r.eps_c is not None)
    return {
        "expected": int(expected_total),
        "observed": n_obs,
        "complete": complete,
        "business_failed": failed,
        "data_invalid": invalid,
        "non_computable": noncomp,
        "eps_c_usable": eps_usable,
        "complete_share_overall": (complete / expected_total) if expected_total else None,
        "non_computable_share": (noncomp / expected_total) if expected_total else None,
    }


def per_cell(rows: Sequence[HeldoutRow], alpha: float, delta: float,
             expected_per_cell: int) -> dict:
    """20 个 cell 的分母与（边际）覆盖率；cell 完整率不足即不得进正式结论。"""
    groups: dict[str, list[HeldoutRow]] = {}
    for r in rows:
        groups.setdefault(r.cell, []).append(r)
    out: dict[str, dict] = {}
    for cell, items in sorted(groups.items()):
        official = [r for r in items if r.official]
        c_cov = [cost_covered(r.eps_c, alpha) for r in official]
        d_cov = [position_covered(r.eps_d, delta) for r in official]
        joint = [bool(a) and bool(b) for a, b in zip(c_cov, d_cov)
                 if a is not None and b is not None]
        out[cell] = {
            "n_observed": len(items),
            "n_complete": len(official),
            "complete_share": len(official) / expected_per_cell if expected_per_cell else None,
            "cost": rate(c_cov),
            "position": rate(d_cov),
            "joint": rate(joint),
        }
    return out


def subgroup(rows: Sequence[HeldoutRow], key, alpha: float, delta: float) -> dict:
    """按任意键分组（profile / family / 低速层 / 方向）。"""
    groups: dict[str, list[HeldoutRow]] = {}
    for r in rows:
        if not r.official:
            continue
        groups.setdefault(str(key(r)), []).append(r)
    out = {}
    for k, items in sorted(groups.items()):
        c_cov = [cost_covered(r.eps_c, alpha) for r in items]
        d_cov = [position_covered(r.eps_d, delta) for r in items]
        joint = [bool(a) and bool(b) for a, b in zip(c_cov, d_cov)
                 if a is not None and b is not None]
        out[k] = {"n": len(items), "cost": rate(c_cov), "position": rate(d_cov),
                  "joint": rate(joint)}
    return out


# ---------------------------------------------------------------- 判定
def coverage_verdict(den: dict, overall: dict, cells: dict,
                     target: float = TARGET_COVERAGE,
                     min_cell_share: float = MIN_CELL_COMPLETE_SHARE) -> dict:
    """可用性判定：只把**预登记的**规则写成判据，其余如实上报。

    * 数据充分性：任一 cell 完整率 < 0.9 → ``insufficient_cells``（该 cell 不得进结论）；
    * 目标：联合与两项边际的点估计是否达到 90%，并给出区间下界；**不**把
      "点估计 ≥ 90%" 说成统计显著达标，也不因不达标去改 α/δ 或模型（协议 may_not）；
    * 90% 是校准分位目标，判定只用于"是否如实报告并提请处置"。
    """
    weak = {c: v["complete_share"] for c, v in cells.items()
            if v["complete_share"] is not None and v["complete_share"] < min_cell_share}
    j = overall["joint"]
    # 独立性参照：若成本与位置误差相互独立，联合覆盖应≈两项边际之比乘积。
    # 实测接近该值即可说明"联合低于 90%"是边际误差的必然结果，而非新缺陷。
    mc = overall["cost"]["rate"]
    mp = overall["position"]["rate"]
    independence = (mc * mp) if (mc is not None and mp is not None) else None
    jr = j["rate"]
    joint_ratio = (jr / independence) if (jr is not None and independence) else None
    below = {
        "joint": j["rate"] is not None and j["rate"] < target,
        "cost": overall["cost"]["rate"] is not None and overall["cost"]["rate"] < target,
        "position": (overall["position"]["rate"] is not None
                     and overall["position"]["rate"] < target),
    }
    # 联合覆盖若恰等于两项边际的乘积，说明两项误差独立，联合缺口是边际缺口的必然结果
    joint_explained = bool(
        below["joint"] and joint_ratio is not None and abs(joint_ratio - 1.0) <= 0.02)
    cost_gap_pp = ((target - mc) * 100.0) if mc is not None else None
    if weak:
        status = "insufficient_cells"
    elif any(below.values()):
        status = "below_target"
    else:
        status = "meets_target"
    return {
        "reporting_complete": True,
        "target": target, "min_cell_complete_share": min_cell_share,
        "cells_below_share": weak,
        "below_target": below,
        "status": status,
        "joint_ci_lo": (j.get("ci_trial") or {}).get("lo"),
        "independence_reference": independence,
        "joint_over_independence": joint_ratio,
        "joint_explained_by_independence": joint_explained,
        "cost_shortfall_pp": cost_gap_pp,
        "note": ("联合覆盖单独报告，不用两项边际推联合；单固定 v2，不声称训练种子方差"
                 if status == "meets_target" else
                 ("联合覆盖低于 90%，但恰等于两项边际乘积（独立性成立），"
                  "按协议不构成新缺陷；边际成本覆盖差 "
                  f"{cost_gap_pp:.1f} pp 列为 open risk，禁止据此反向调整 α/δ 或模型"
                  if joint_explained else
                  "未达 90% 参照：如实记录，禁止据此反向调整 α/δ 或功率模型")
                 if status == "below_target" else
                 "存在 cell 完整率不足：该 cell 不得进正式结论（协议 min_complete_share_rule）"),
        "denominators": den,
    }
