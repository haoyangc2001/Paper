"""配对、误差复算与失败分类（CAL-016）。

口径（协议 statistics / invalid_rules）：

* 配对网格是 **10 Hz 公共时间网格**（301 点，含 t=0 与 t=30 s），位置按时间**线性配对**；
  角度（yaw）只用于运动学与诊断，位置不做任何相位对齐（不做时间平移、不做 DTW）。
* **只有完整窗口**（outcome=completed、6000 步、数据自洽）才产生正式 ε_c / ε_d；
  部分窗口（terminated / interrupted / truncated）只保留**前缀诊断**，
  在总体覆盖里计入失败侧，且明确标注 ``official=false``。
* 失败分三类，互不混用：
  ``business_failure``（停滞、摔倒、提前结束）、``data_invalid``（NaN、乱序时钟、缺分片、
  形状不符、重复 trial、状态来源错误）、``infra_retry``（GPU OOM、进程崩溃等，见采集日志）。
* 近零分母：``J_nom < J_min`` 记 ``non_computable``，单列统计、不进 α 的分位。
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

PAIR_DT = 0.1
PAIR_POINTS = 301
J_MIN = 1.0
LOW_SPEED_VX = 0.3
LOW_SPEED_WZ = 0.3

BUSINESS_REASONS = ("stalled", "low_speed", "terminated", "interrupted", "truncated")
DATA_INVALID_REASONS = ("nan", "time_jump", "non_monotonic", "zero_window", "shape_mismatch",
                        "missing_shard", "duplicate_trial", "wrong_source")


@dataclass(frozen=True)
class Issue:
    reason: str
    detail: str = ""
    source: str = ""

    def to_dict(self) -> dict:
        return {"reason": self.reason, "detail": self.detail, "source": self.source}


@dataclass
class TrialErrors:
    """一条 trial 的误差复算结果（含分类与诊断）。"""

    trial_id: str
    outcome: str
    complete: bool
    official: bool
    epsilon_c: float | None = None
    epsilon_d: float | None = None
    epsilon_d_50hz: float | None = None
    j_exec: float | None = None
    j_nom: float | None = None
    non_computable: bool = False
    low_speed_share: float | None = None
    prefix_only: bool = False
    issues: list[Issue] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def error_class(self) -> str:
        reasons = {i.reason for i in self.issues}
        if reasons & set(DATA_INVALID_REASONS):
            return "data_invalid"
        if reasons & set(BUSINESS_REASONS) or not self.complete:
            return "business_failure"
        return "ok"

    def to_dict(self) -> dict:
        return {
            "trial_id": self.trial_id, "outcome": self.outcome,
            "complete": self.complete, "official": self.official,
            "error_class": self.error_class,
            "epsilon_c": self.epsilon_c, "epsilon_d": self.epsilon_d,
            "epsilon_d_50hz": self.epsilon_d_50hz,
            "j_exec": self.j_exec, "j_nom": self.j_nom,
            "non_computable": self.non_computable,
            "low_speed_share": self.low_speed_share,
            "prefix_only": self.prefix_only,
            "issues": [i.to_dict() for i in self.issues],
            "notes": self.notes,
        }


# ---------------------------------------------------------------- 基础校验
def check_time_grid(t: Sequence[float], dt: float = PAIR_DT, tol: float | None = None,
                    source: str = "") -> list[Issue]:
    """10 Hz 公共网格自检：起点 0、等间隔、单调、有限、点数与端点。

    容差默认按**存储精度**取：float32 网格取 1e-5（0.1 在 float32 下不可精确表示，
    300 段累积可达 1.5e-6），float64 网格取 1e-6（协议 ``time_sync_abs_s``）。
    采集器现已写 float64；此分支用于兼容早期 float32 产物，并在报告里可见。
    """
    issues: list[Issue] = []
    raw = np.asarray(t)
    if tol is None:
        tol = 1e-5 if raw.dtype == np.float32 else 1e-6
    arr = raw.astype(np.float64)
    if arr.ndim != 1 or arr.size == 0:
        return [Issue("shape_mismatch", f"t 形状 {arr.shape}", source)]
    if not np.isfinite(arr).all():
        issues.append(Issue("nan", "t 含非有限值", source))
        return issues
    if arr.size < 2:
        issues.append(Issue("zero_window", f"只有 {arr.size} 个点", source))
        return issues
    if abs(arr[0]) > tol:
        issues.append(Issue("time_jump", f"起点 {arr[0]} ≠ 0", source))
    diffs = np.diff(arr)
    if (diffs <= 0).any():
        issues.append(Issue("non_monotonic", "时间非严格递增", source))
    elif abs(diffs - dt).max() > tol:
        issues.append(Issue("time_jump", f"间隔偏离 {dt}：max|Δ-{dt}|={abs(diffs - dt).max():.3e}", source))
    return issues


def check_positions(exec_pos, nom_pos, source: str = "") -> list[Issue]:
    issues: list[Issue] = []
    a = np.asarray(exec_pos, dtype=np.float64)
    b = np.asarray(nom_pos, dtype=np.float64)
    if a.shape != b.shape:
        issues.append(Issue("shape_mismatch", f"exec {a.shape} vs nominal {b.shape}", source))
        return issues
    if a.ndim != 2 or a.shape[1] != 2:
        issues.append(Issue("shape_mismatch", f"期望 (N,2)，得到 {a.shape}", source))
        return issues
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        issues.append(Issue("nan", "位置含非有限值", source))
    return issues


def deviation(exec_pos, nom_pos) -> np.ndarray:
    """逐点平面欧氏距离；要求同形（不做插值、不做对齐）。"""
    a = np.asarray(exec_pos, dtype=np.float64)
    b = np.asarray(nom_pos, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"形状不一致: {a.shape} vs {b.shape}")
    return np.linalg.norm(a - b, axis=-1)


def deviation_on_grid(t_query, exec_pos, t_grid, nom_pos) -> np.ndarray:
    """把名义轨迹线性插值到查询时刻后求偏差（只做位置插值，不做相位对齐）。"""
    tq = np.asarray(t_query, dtype=np.float64)
    tg = np.asarray(t_grid, dtype=np.float64)
    if tq.min() < tg.min() - 1e-9 or tq.max() > tg.max() + 1e-9:
        raise ValueError("查询时刻超出名义网格范围（禁止外推）")
    nom = np.asarray(nom_pos, dtype=np.float64)
    interp = np.stack([np.interp(tq, tg, nom[:, 0]), np.interp(tq, tg, nom[:, 1])], axis=-1)
    return deviation(np.asarray(exec_pos, dtype=np.float64), interp)


# ---------------------------------------------------------------- ε 复算
def epsilon_c(j_exec: float, j_nom: float | None, j_min: float = J_MIN) -> tuple[float | None, str]:
    """成本超支 ε_c；返回 (值 或 None, 原因)。"""
    if j_nom is None:
        return None, "j_nom_missing"
    if not (math.isfinite(j_exec) and math.isfinite(j_nom)):
        return None, "nan"
    if j_nom < j_min:
        return None, "non_computable"
    if j_nom <= 0:
        return None, "non_computable"
    return (j_exec - j_nom) / j_nom, "ok"


def low_speed_share(cmd, vx_max: float = LOW_SPEED_VX, wz_max: float = LOW_SPEED_WZ) -> float:
    """命令序列中低速占比（|vx|≤0.3 且 |wz|≤0.3，与采集侧 strict 口径一致）。"""
    arr = np.asarray(cmd, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if arr.ndim != 2 or arr.shape[-1] < 3:
        raise ValueError(f"cmd 形状应为 (N,3)，得到 {arr.shape}")
    hit = (np.abs(arr[:, 0]) <= vx_max) & (np.abs(arr[:, 2]) <= wz_max)
    return float(hit.mean())


def compute_trial_errors(*, trial_id: str, outcome: str, t_pair, exec_pos, nom_pos,
                         j_exec: float | None, j_nom: float | None,
                         j_min: float = J_MIN, cmd=None,
                         t50=None, exec50_pos=None, steps_executed: int | None = None,
                         expected_steps: int = 6000, source: str = "") -> TrialErrors:
    """单条 trial 的误差复算（含分类、前缀诊断、50 Hz 敏感性）。"""
    issues: list[Issue] = []
    notes: list[str] = []
    complete = (outcome == "completed")

    issues += check_time_grid(t_pair, source=f"{source}:pairing_10hz")
    issues += check_positions(exec_pos, nom_pos, source=f"{source}:pairing_10hz")

    dev = None
    if not issues:
        dev = deviation(exec_pos, nom_pos)
        if steps_executed is not None and complete and steps_executed != expected_steps:
            issues.append(Issue("shape_mismatch",
                                f"completed 但 steps={steps_executed} ≠ {expected_steps}", source))

    e_c = e_c_reason = None
    if dev is not None:
        if complete:
            e_c, e_c_reason = epsilon_c(j_exec, j_nom, j_min)
            if e_c_reason == "j_nom_missing":
                notes.append("j_nom 未就绪（功率模型属 CAL-019/020），ε_c 暂不可算")
            elif e_c_reason == "non_computable":
                notes.append(f"J_nom < J_min={j_min}，记 non_computable")
        else:
            notes.append("部分窗口：仅保留前缀诊断，ε 非正式")

    e_d = float(dev.max()) if dev is not None else None
    e_d50 = None
    if dev is not None and complete and t50 is not None and exec50_pos is not None:
        try:
            d50 = deviation_on_grid(t50, exec50_pos, np.asarray(t_pair, dtype=np.float64), nom_pos)
            e_d50 = float(d50.max())
        except ValueError as exc:
            issues.append(Issue("time_jump", f"50 Hz 敏感性无法计算: {exc}", source))

    share = None
    if cmd is not None:
        try:
            share = low_speed_share(cmd)
        except ValueError as exc:
            issues.append(Issue("shape_mismatch", str(exc), source))

    if not complete and outcome not in BUSINESS_REASONS:
        issues.append(Issue("wrong_source", f"未知 outcome={outcome!r}", source))

    return TrialErrors(
        trial_id=trial_id, outcome=outcome, complete=complete,
        official=bool(complete and dev is not None and not issues),
        epsilon_c=e_c, epsilon_d=e_d, epsilon_d_50hz=e_d50,
        j_exec=j_exec, j_nom=j_nom, non_computable=(e_c_reason == "non_computable"),
        low_speed_share=share, prefix_only=not complete,
        issues=issues, notes=notes,
    )


# ---------------------------------------------------------------- 重复检测
def detect_duplicates(trial_ids: Sequence[str]) -> dict[str, int]:
    """同一 trial_id 出现多次（例如多个 attempt 目录都在）→ 计为 data_invalid。

    返回 {trial_id: 次数}，只包含次数 > 1 的项。调用方负责据此给出
    ``duplicate_trial`` 分类，不得静默取其一。
    """
    counts = Counter(trial_ids)
    return {k: v for k, v in sorted(counts.items()) if v > 1}


# ---------------------------------------------------------------- 汇总
def aggregate(rows: Iterable[TrialErrors], expected_total: int | None = None) -> dict:
    """配对报告：全体/完整/失败/无效四套分母，任何排除都带原因与来源。"""
    rows = list(rows)
    by_class = Counter(r.error_class for r in rows)
    reasons = Counter()
    for r in rows:
        for i in r.issues:
            reasons[f"{r.error_class}:{i.reason}"] += 1

    complete = [r for r in rows if r.complete and r.error_class == "ok"]
    valid_c = [r for r in complete if r.epsilon_c is not None]
    nc = [r for r in complete if r.non_computable]

    def stats(values):
        vals = [v for v in values if v is not None]
        if not vals:
            return None
        arr = np.asarray(vals, dtype=np.float64)
        return {"n": int(arr.size), "min": float(arr.min()), "max": float(arr.max()),
                "mean": float(arr.mean())}

    return {
        "denominators": {
            "all": expected_total if expected_total is not None else len(rows),
            "observed": len(rows),
            "complete": len(complete),
            "business_failed": by_class.get("business_failure", 0),
            "data_invalid": by_class.get("data_invalid", 0),
            "ok": by_class.get("ok", 0),
        },
        "exclusions": dict(sorted(reasons.items())),
        "epsilon_c": {"n_available": len(valid_c),
                      "n_non_computable": len(nc),
                      "stats": stats([r.epsilon_c for r in valid_c]),
                      "official_only": True},
        "epsilon_d": {"stats": stats([r.epsilon_d for r in rows if r.official]),
                      "prefix_diagnostic": stats([r.epsilon_d for r in rows if not r.official])},
        "epsilon_d_50hz": {"stats": stats([r.epsilon_d_50hz for r in rows if r.epsilon_d_50hz is not None])},
        "low_speed_share": stats([r.low_speed_share for r in rows if r.low_speed_share is not None]),
        "unit": "J",
        "notes": "complete 才有正式 ε；prefix_diagnostic 仅诊断，不计入正式分位",
    }
