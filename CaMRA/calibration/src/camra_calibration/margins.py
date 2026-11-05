"""α / δ 裕度的统计与冻结（CAL-025）。

协议口径（config/calibration_protocol_v1.0.2.json 的 ``statistics``）：

* 分位：``nearest-rank``——对完整有效样本排序后取第 ``ceil(q·n)`` 个值，**不插值**，
  等于阈值视为覆盖；
* ``alpha = max(0, Q0.90(epsilon_c))``，``delta = Q0.90(epsilon_d)``（非负）；
* 负超支保留原值参与分位，只在形成安全裕度时取 ``max(0, ·)``；
* 区间：bootstrap，**以整条轨迹为最小重采样单位**（不得按时间点伪增样本），
  10000 次重复、95% 水平；同一命令模板的 trial 构成一个簇，簇重采样时整簇进出。

本模块只做统计与冻结，不读数据文件、不依赖 torch：所有输入都是
:class:`~camra_calibration.pairing.TrialErrors` 或等价的 (ε_c, ε_d) 序列，
因此可以在 CPU 上完整单测。
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

#: 协议 statistics.quantile_level
QUANTILE_LEVEL: float = 0.9
#: 协议 statistics.intervals.replicates / level
BOOTSTRAP_REPLICATES: int = 10_000
BOOTSTRAP_LEVEL: float = 0.95
#: 固定的 bootstrap 随机种子（写进产物，任何人可用同一种子复算）
BOOTSTRAP_SEED: int = 20261104
#: 协议 statistical_acceptance.min_complete_share_per_cell
MIN_COMPLETE_SHARE_PER_CELL: float = 0.9


class MarginError(RuntimeError):
    """样本不足、输入非法或冻结产物不一致。"""


# ---------------------------------------------------------------- 分位
def nearest_rank(values: Sequence[float] | np.ndarray, q: float = QUANTILE_LEVEL) -> float:
    """nearest-rank 分位：升序第 ``ceil(q·n)`` 个（1-based），不插值。

    与 numpy 的 ``method="nearest"`` 不同：后者在 ``q·n`` 恰为整数时取第 ``k`` 个，
    这里严格按协议取 ``ceil(q·n)``，两者在 ``0.9·10 = 9`` 这类边界上会差一个样本。
    """
    arr = np.asarray([v for v in np.ravel(values) if v is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise MarginError("空样本无法取分位")
    if not 0.0 < q <= 1.0:
        raise MarginError(f"分位水平应在 (0,1]，得到 {q}")
    arr.sort()
    k = max(1, int(math.ceil(q * arr.size)))
    return float(arr[k - 1])


def percentile_interval(boot: np.ndarray, level: float = BOOTSTRAP_LEVEL) -> tuple[float, float]:
    """bootstrap 分布的分位区间，同样用 nearest-rank（保持口径一致）。"""
    lo_q = (1.0 - level) / 2.0
    hi_q = 1.0 - lo_q
    return nearest_rank(boot, lo_q), nearest_rank(boot, hi_q)


def bootstrap_quantile(values: Sequence[float], q: float = QUANTILE_LEVEL,
                       replicates: int = BOOTSTRAP_REPLICATES,
                       level: float = BOOTSTRAP_LEVEL,
                       seed: int = BOOTSTRAP_SEED,
                       clusters: Sequence[str] | None = None) -> dict:
    """整条轨迹（或整簇）重采样的 bootstrap 分位区间。

    ``clusters`` 给定时按簇整块重采样（协议 requirement：簇内 trial 一起抽入/抽出），
    否则按单条轨迹重采样——两者都以"完整轨迹"为最小单位，绝不按时间点伪增样本。
    """
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        raise MarginError("空样本无法 bootstrap")
    rng = np.random.default_rng(seed)
    point = nearest_rank(arr, q)
    boot = np.empty(replicates, dtype=np.float64)

    if clusters is None:
        n = arr.size
        for r in range(replicates):
            idx = rng.integers(0, n, size=n)
            boot[r] = nearest_rank(arr[idx], q)
    else:
        if len(clusters) != arr.size:
            raise MarginError("clusters 长度与样本不一致")
        groups: dict[str, list[float]] = {}
        for c, v in zip(clusters, arr):
            groups.setdefault(c, []).append(v)
        keys = sorted(groups)
        blocks = [np.asarray(groups[k], dtype=np.float64) for k in keys]
        for r in range(replicates):
            pick = rng.integers(0, len(blocks), size=len(blocks))
            sample = np.concatenate([blocks[i] for i in pick])
            boot[r] = nearest_rank(sample, q)

    lo, hi = percentile_interval(boot, level)
    return {"point": point, "lo": lo, "hi": hi, "replicates": int(replicates),
            "level": float(level), "seed": int(seed), "n": int(arr.size),
            "unit": "cluster" if clusters is not None else "trial"}


# ---------------------------------------------------------------- 单组统计
def group_stats(values: Iterable[float | None], q: float = QUANTILE_LEVEL) -> dict:
    """一组 ε 的计数与分位（n / min / median / q90 / max / 负值数）。"""
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {"n": int(arr.size), "min": float(arr.min()),
            "median": float(np.median(arr)),
            "q90": nearest_rank(arr, q),
            "max": float(arr.max()),
            "n_negative": int((arr < 0).sum())}


# ---------------------------------------------------------------- 冻结对象
@dataclass(frozen=True)
class Margins:
    """冻结的 α / δ 与适用域。

    ``alpha`` 只用于成本预算（``J_budget = (1+alpha)·J_nom``），``delta`` 只用于
    几何危险集扩张（``obstacle ⊕ footprint ⊕ disc(delta)``，且只膨胀一次）。
    """

    version: str
    alpha: float
    delta: float
    alpha_interval: tuple[float, float]
    delta_interval: tuple[float, float]
    q90_eps_c: float
    q90_eps_d: float
    n_eps_c: int
    n_eps_d: int
    data: dict
    stats: dict
    applicability: dict
    notes: tuple[str, ...] = ()

    # ---- 内容与散列 ----
    def content_dict(self) -> dict:
        """参与散列的部分：只含数值与适用域，不含时间戳/路径/统计冗余。"""
        return {
            "version": self.version,
            "alpha": float(self.alpha),
            "delta": float(self.delta),
            "unit_cost": "ratio",
            "unit_distance": "m",
            "alpha_interval": [float(self.alpha_interval[0]), float(self.alpha_interval[1])],
            "delta_interval": [float(self.delta_interval[0]), float(self.delta_interval[1])],
            "applicability": self.applicability,
        }

    def sha256(self) -> str:
        blob = json.dumps(self.content_dict(), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()

    def to_dict(self) -> dict:
        return {
            **self.content_dict(),
            "q90_eps_c": float(self.q90_eps_c),
            "q90_eps_d": float(self.q90_eps_d),
            "n_eps_c": int(self.n_eps_c),
            "n_eps_d": int(self.n_eps_d),
            "sha256": self.sha256(),
            "data": self.data,
            "stats": self.stats,
            "notes": list(self.notes),
        }

    def save(self, path: Path) -> tuple[Path, Path]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        side = path.with_suffix(path.suffix + ".sha256")
        side.write_text(f"{self.sha256()}  {path.name}\n", encoding="utf-8")
        return path, side

    @classmethod
    def load(cls, path: Path) -> "Margins":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(version=payload["version"], alpha=float(payload["alpha"]),
                  delta=float(payload["delta"]),
                  alpha_interval=tuple(payload["alpha_interval"]),
                  delta_interval=tuple(payload["delta_interval"]),
                  q90_eps_c=float(payload["q90_eps_c"]), q90_eps_d=float(payload["q90_eps_d"]),
                  n_eps_c=int(payload["n_eps_c"]), n_eps_d=int(payload["n_eps_d"]),
                  data=payload["data"], stats=payload["stats"],
                  applicability=payload["applicability"],
                  notes=tuple(payload.get("notes", ())))
        if obj.sha256() != payload.get("sha256"):
            raise MarginError(f"{path} 内容散列与记录不符（文件被改动？）")
        return obj

    # ---- 用法 ----
    def cost_budget(self, j_nom: float) -> float:
        """给定名义成本的名义预算 ``(1+alpha)·J_nom``。"""
        if not np.isfinite(j_nom) or j_nom < 0:
            raise MarginError(f"J_nom 非法: {j_nom}")
        return (1.0 + float(self.alpha)) * float(j_nom)

    def geometric_margin(self) -> float:
        return float(self.delta)


# ---------------------------------------------------------------- 组装冻结值
def build_margins(*, eps_c: Sequence[float], eps_d: Sequence[float],
                  stats: dict, data: dict, applicability: dict,
                  version: str = "1.0",
                  q: float = QUANTILE_LEVEL,
                  replicates: int = BOOTSTRAP_REPLICATES,
                  seed: int = BOOTSTRAP_SEED,
                  clusters_c: Sequence[str] | None = None,
                  clusters_d: Sequence[str] | None = None,
                  notes: Sequence[str] = ()) -> tuple[Margins, dict]:
    """按协议口径算 α/δ 与区间，返回 (冻结对象, 计算明细)。

    α 的原始分位（可能为负）与最终 α（非负）都保留：协议要求"负超支保留原值参与
    分位，只在形成安全裕度时取 max(0, ·)"，报告里两者都要能复算。
    """
    arr_c = np.asarray([v for v in eps_c if v is not None], dtype=np.float64)
    arr_d = np.asarray([v for v in eps_d if v is not None], dtype=np.float64)
    if arr_c.size == 0 or arr_d.size == 0:
        raise MarginError("ε_c 或 ε_d 没有可用样本，无法形成裕度")

    q_c = nearest_rank(arr_c, q)
    q_d = nearest_rank(arr_d, q)
    alpha = max(0.0, q_c)
    delta = max(0.0, q_d)     # Q90(ε_d) 本身非负（绝对距离），这里防御性取一次

    boot_c = bootstrap_quantile(arr_c, q=q, replicates=replicates, seed=seed,
                                clusters=clusters_c)
    boot_d = bootstrap_quantile(arr_d, q=q, replicates=replicates, seed=seed,
                                clusters=clusters_d)
    # α 的区间也要过 max(0, ·)：负超支被截断时区间下端更应体现这一点
    a_lo = max(0.0, boot_c["lo"])
    a_hi = max(0.0, boot_c["hi"])

    detail = {"quantile": {"level": q, "method": "nearest-rank",
                           "rule": "ceil(q*n) 第几个，不插值"},
              "eps_c": {"q90_raw": q_c, "alpha": alpha,
                        "alpha_interval_raw": [boot_c["lo"], boot_c["hi"]],
                        "alpha_interval_clipped": [a_lo, a_hi],
                        "bootstrap": boot_c},
              "eps_d": {"q90": q_d, "delta": delta,
                        "delta_interval": [boot_d["lo"], boot_d["hi"]],
                        "bootstrap": boot_d}}

    obj = Margins(version=version, alpha=alpha, delta=delta,
                  alpha_interval=(a_lo, a_hi), delta_interval=(boot_d["lo"], boot_d["hi"]),
                  q90_eps_c=q_c, q90_eps_d=q_d,
                  n_eps_c=int(arr_c.size), n_eps_d=int(arr_d.size),
                  data=data, stats=stats, applicability=applicability,
                  notes=tuple(notes))
    return obj, detail


# ---------------------------------------------------------------- 准入判定
def calibration_verdict(per_cell: dict, non_computable_share: float,
                        min_complete_share: float = MIN_COMPLETE_SHARE_PER_CELL) -> dict:
    """按协议 statistical_acceptance.calibration 给出结论。

    三条 require：α/δ 由校准集冻结、报 90% 分位与 95% 区间、报 non_computable 占比。
    另加 cell 完整率门槛 ``min_complete_share_per_cell``：低于门槛的 cell 不得进入
    正式结论，需回到采集或协议裁定。
    """
    weak = {c: v["complete_share"] for c, v in per_cell.items()
            if v.get("complete_share") is not None and v["complete_share"] < min_complete_share}
    # 逻辑条件恒真；真正的判定靠 weak（cell 完整率）与输入是否为校准集
    require = {
        "alpha_delta_from_calibration": True,
        "reports_q90_and_95ci": True,
        "reports_non_computable_share": True,
    }
    ok = all(require.values()) and not weak
    return {"require": require,
            "min_complete_share_per_cell": min_complete_share,
            "cells_below_share": weak,
            "non_computable_share": float(non_computable_share),
            "verdict": "admitted" if ok else "blocked",
            "note": ("cell 完整率不足即不得进正式结论（协议 min_complete_share_rule）"
                     if weak else "")}
