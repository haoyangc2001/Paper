"""非负名义功率模型与 J_nom 独立复算（CAL-020）。

口径来自 D-ENERGY-001 / §10.3 / §10.5，任何实现不得偏离：

    P̂_nom(i) = Σ_m w_m · φ_m(u_i, u_{i-1}, ..., u_{i-K})     w_m ≥ 0
    J_nom     = Σ_{i=0..299} P̂_nom(i) · 0.1                  [ J ]

三条硬约束：

1. **因果**：第 i 个区间的预测只用到 ``u_i`` 及其过去 ``K`` 条命令；
   未来命令、执行侧力矩/能量、扰动标签一律不可见（构造性保证，见
   :func:`feature_matrix` 与 test_power 的前缀不变性测试）。
2. **非负**：所有特征由 ``|vx|,|vy|,|wz|`` 及其两两乘积构成（恒 ≥0），
   权重由 NNLS 求解（恒 ≥0），因此 ``P̂ ≥ 0`` 与 ``J_nom ≥ 0`` 是构造性的；
   截断比例仍逐次计算并上报，不得静默。
3. **可复算**：模型文件（权重 + 特征 schema + 输入域 + 版本）足以从命令序列
   独立算出 ``J_nom``，不依赖任何执行侧数据；散列由
   :meth:`PowerModel.sha256` 给出。

单位一律为焦耳（机械功）与瓦（机械功率），窗口 30 s / 300 个 0.1 s 区间 /
6000 个 0.005 s 物理步。
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

POWER_VERSION = "power-v1"
N_INTERVALS = 300          # 30 s / 0.1 s
STEP_COST_DT = 0.1         # 每个命令区间 = 20 个 0.005 s 物理步
SUBSTEPS_PER_INTERVAL = 20
J_MIN = 1.0                # §10.4 预登记近零阈值 [J]
ADMIT_Q90 = 0.10           # §10.5 准入：验证集逐轨迹相对误差 90% 分位上限
LOW_SPEED_VX = 0.3
LOW_SPEED_WZ = 0.3
TRANSITION_DELTA = 0.15    # 相邻命令 L1 变化超过此值记为"过渡"区间

# 特征块：全部由命令的绝对值/平方/乘积构成，因此恒非负。
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "linear": ("bias", "abs_vx", "abs_vy", "abs_wz"),
    "quadratic": ("bias", "abs_vx", "abs_vy", "abs_wz",
                  "vx2", "vy2", "wz2", "abs_vx_wz", "abs_vy_wz", "abs_vx_vy"),
}


class PowerModelError(RuntimeError):
    """模型不合法、输入形状错误或求解失败。"""


# ---------------------------------------------------------------- 输入与特征
def as_commands(commands) -> np.ndarray:
    """把命令点序列规范为 float64 的 (N,3) 数组；拒收 NaN/inf 与错形状。"""
    arr = np.asarray(commands, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise PowerModelError(f"命令形状应为 (N,3)，得到 {arr.shape}")
    if arr.shape[0] < 2:
        raise PowerModelError(f"至少需要 2 个命令点，得到 {arr.shape[0]}")
    if not np.isfinite(arr).all():
        raise PowerModelError("命令含 NaN/inf")
    return arr


def _base_block(cmd: np.ndarray, base: str) -> np.ndarray:
    names = FEATURE_SETS[base]
    ax, ay, az = np.abs(cmd[:, 0]), np.abs(cmd[:, 1]), np.abs(cmd[:, 2])
    cols = {
        "bias": np.ones_like(ax),
        "abs_vx": ax, "abs_vy": ay, "abs_wz": az,
        "vx2": ax * ax, "vy2": ay * ay, "wz2": az * az,
        "abs_vx_wz": ax * az, "abs_vy_wz": ay * az, "abs_vx_vy": ax * ay,
    }
    return np.stack([cols[n] for n in names], axis=1)


def feature_names(base: str, history_k: int) -> tuple[str, ...]:
    """展开带历史块的特征名（``lag1:abs_vx`` …），顺序即权重顺序。"""
    if base not in FEATURE_SETS:
        raise PowerModelError(f"未知特征块 {base!r}（可用：{sorted(FEATURE_SETS)}）")
    if history_k < 0:
        raise PowerModelError(f"history_k 必须 ≥ 0，得到 {history_k}")
    out: list[str] = list(FEATURE_SETS[base])
    for lag in range(1, history_k + 1):
        out.extend(f"lag{lag}:{n}" for n in FEATURE_SETS[base])
    return tuple(out)


def feature_matrix(commands, base: str, history_k: int) -> np.ndarray:
    """(n_intervals, n_features) 设计矩阵；第 i 行只依赖 ``u[0..i]``。

    区间 i 覆盖 ``[i·0.1, (i+1)·0.1)``，共 ``N-1`` 个区间（末点不再驱动区间）。
    历史块取 ``u[max(0, i-lag)]``——越界时钳到首点，因此 t=0 起的预测也合法，
    不会因为"没有历史"而改变起点行为。
    """
    cmd = as_commands(commands)
    n = cmd.shape[0] - 1
    blocks = [_base_block(cmd[:n], base)]
    for lag in range(1, history_k + 1):
        idx = np.maximum(np.arange(n) - lag, 0)
        blocks.append(_base_block(cmd[idx], base))
    out = np.concatenate(blocks, axis=1)
    if out.shape[1] != len(feature_names(base, history_k)):
        raise PowerModelError("特征名与矩阵列数不一致")
    return out


def trajectory_design(commands, base: str, history_k: int) -> np.ndarray:
    """整条轨迹的聚合特征 Φ = 0.1 · Σ_i φ_i，使 ``J_nom = w·Φ``。"""
    return STEP_COST_DT * feature_matrix(commands, base, history_k).sum(axis=0)


# ---------------------------------------------------------------- 统计工具
def q_nearest_rank(values: Iterable[float], q: float = 0.9) -> float:
    """nearest-rank 分位（与 §10.6 的 α/δ 口径一致，不插值）。"""
    arr = np.sort(np.asarray(list(values), dtype=np.float64))
    if arr.size == 0:
        raise PowerModelError("空样本无法取分位")
    if not 0.0 < q <= 1.0:
        raise PowerModelError(f"q 必须落在 (0,1]，得到 {q}")
    k = max(int(math.ceil(q * arr.size)), 1)
    return float(arr[k - 1])


def summarize(values: Iterable[float]) -> dict:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"n": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {"n": int(arr.size), "q90": q_nearest_rank(arr, 0.9),
            "median": float(np.median(arr)), "mean": float(arr.mean()),
            "min": float(arr.min()), "max": float(arr.max())}


def transition_share(commands, delta: float = TRANSITION_DELTA) -> float:
    """相邻命令 L1 变化超过 ``delta`` 的区间占比（"过渡段"密度）。"""
    cmd = as_commands(commands)
    d = np.abs(np.diff(cmd, axis=0)).sum(axis=1)
    return float((d > delta).mean())


def low_speed_share(commands) -> float:
    cmd = as_commands(commands)
    hit = (np.abs(cmd[:, 0]) <= LOW_SPEED_VX) & (np.abs(cmd[:, 2]) <= LOW_SPEED_WZ)
    return float(hit.mean())


# ---------------------------------------------------------------- 求解
def fit_nonneg(design: np.ndarray, targets: np.ndarray, mode: str = "relative") -> np.ndarray:
    """非负最小二乘（NNLS），列归一化以改善条件数。

    ``mode="relative"``：按 ``1/J_exec`` 加权，等价于最小化 Σ((ΔJ)/J)²，与
    §10.5 的准入指标同向；``mode="absolute"`` 仅作诊断对照，不参与选型。
    """
    A = np.asarray(design, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if A.ndim != 2 or y.ndim != 1 or A.shape[0] != y.size:
        raise PowerModelError(f"设计矩阵/目标形状不匹配：{A.shape} vs {y.shape}")
    if not (np.isfinite(A).all() and np.isfinite(y).all()):
        raise PowerModelError("设计矩阵或目标含 NaN/inf")
    if (y <= 0).any():
        raise PowerModelError("J_exec 必须为正（相对加权不可用于非正目标）")
    if mode == "relative":
        w = 1.0 / y
    elif mode == "absolute":
        w = np.ones_like(y)
    else:
        raise PowerModelError(f"未知 fit mode {mode!r}")
    Aw = A * w[:, None]
    yw = y * w
    scale = np.linalg.norm(Aw, axis=0)
    scale[scale <= 0] = 1.0
    try:
        from scipy.optimize import nnls  # 延迟导入：纯逻辑测试不强制 scipy
    except ModuleNotFoundError as exc:  # pragma: no cover - 环境问题
        raise PowerModelError("需要 scipy.optimize.nnls 才能拟合：请在该环境安装 scipy") from exc
    coef, _residual = nnls(Aw / scale, yw, maxiter=10000)
    weights = coef / scale
    if not np.isfinite(weights).all():
        raise PowerModelError("NNLS 返回非有限权重")
    if (weights < -1e-12).any():
        raise PowerModelError("NNLS 返回负权重（数值异常）")
    return np.clip(weights, 0.0, None)


# ---------------------------------------------------------------- 模型
@dataclass(frozen=True)
class PowerModel:
    """冻结后的非负名义功率模型。"""

    weights: tuple[float, ...]
    feature_names: tuple[str, ...]
    base: str
    history_k: int
    fit_mode: str
    train: dict
    domain: dict
    version: str = POWER_VERSION
    step_cost_dt: float = STEP_COST_DT
    n_intervals: int = N_INTERVALS
    j_min: float = J_MIN
    notes: tuple[str, ...] = ()

    # ---- 校验 ----
    def __post_init__(self) -> None:
        if len(self.weights) != len(self.feature_names):
            raise PowerModelError(
                f"权重数与特征数不一致：{len(self.weights)} vs {len(self.feature_names)}")
        if tuple(self.feature_names) != feature_names(self.base, self.history_k):
            raise PowerModelError("特征名与 base/history_k 展开结果不一致")
        if any(w < 0 for w in self.weights):
            raise PowerModelError("权重必须非负")
        if any(not math.isfinite(w) for w in self.weights):
            raise PowerModelError("权重必须有限")

    # ---- 预测 ----
    @property
    def _w(self) -> np.ndarray:
        return np.asarray(self.weights, dtype=np.float64)

    def step_power(self, commands) -> np.ndarray:
        """逐区间功率 P̂（W），长度 = 命令点数 − 1。"""
        phi = feature_matrix(commands, self.base, self.history_k)
        p = phi @ self._w
        return np.clip(p, 0.0, None)   # 构造性非负，clip 只为防御浮点 -0

    def j_nom(self, commands) -> float:
        """整窗名义能量 J_nom [J]（不做 J_min 判定）。"""
        return float(self.step_power(commands).sum() * self.step_cost_dt)

    def truncation_share(self, commands) -> float:
        """被截断为 0 的区间占比（应为 0，用于证明非负性是构造性的）。"""
        raw = feature_matrix(commands, self.base, self.history_k) @ self._w
        if raw.size == 0:
            return 0.0
        return float((raw < 0.0).mean())

    def is_full_window(self, commands) -> bool:
        return len(as_commands(commands)) - 1 == self.n_intervals

    def in_domain(self, commands, tol: float = 1e-6) -> tuple[bool, dict]:
        """输入域检查：各轴 ``|v|`` 不得超出拟合集观测包络（含容差）。"""
        cmd = as_commands(commands)
        checks = {"abs_vx": np.abs(cmd[:, 0]), "abs_vy": np.abs(cmd[:, 1]),
                  "abs_wz": np.abs(cmd[:, 2])}
        bad: dict[str, dict] = {}
        for axis, vals in checks.items():
            hi = float(self.domain.get(axis, {}).get("max", float("inf")))
            if float(vals.max()) > hi + tol:
                bad[axis] = {"observed_max": float(vals.max()), "domain_max": hi}
        return (not bad), bad

    def intercept_share(self, commands) -> float:
        """截距项对 J_nom 的贡献占比（用于证明没有靠大截距掩盖问题）。"""
        phi = trajectory_design(commands, self.base, self.history_k)
        total = float(phi @ self._w)
        if total <= 0:
            return float("nan")
        bias_idx = [i for i, n in enumerate(self.feature_names) if n == "bias"]
        if not bias_idx:
            return 0.0
        return float(phi[bias_idx[0]] * self._w[bias_idx[0]] / total)

    # ---- 序列化 ----
    def content_dict(self) -> dict:
        """参与散列的**模型内容**：决定预测的一切，不含任何时间戳或溯源元数据。

        这样同一批拟合数据重跑得到同一散列（可证明"重跑没有改动模型"），
        而 ``train`` 里的 frozen_at / 提交号等易变字段不会污染身份。
        """
        return {
            "version": self.version,
            "unit": "J",
            "power_unit": "W",
            "feature_names": list(self.feature_names),
            "feature_schema_sha256": self.schema_sha256(),
            "weights": [float(w) for w in self.weights],
            "base": self.base,
            "history_k": int(self.history_k),
            "fit_mode": self.fit_mode,
            "step_cost_dt": float(self.step_cost_dt),
            "n_intervals": int(self.n_intervals),
            "substeps_per_interval": SUBSTEPS_PER_INTERVAL,
            "j_min": float(self.j_min),
            "domain": self.domain,
        }

    def to_dict(self) -> dict:
        """落盘内容 = 模型内容 + 溯源（train/notes），溯源不参与散列。"""
        out = self.content_dict()
        out["train"] = self.train
        out["notes"] = list(self.notes)
        return out

    @classmethod
    def from_dict(cls, payload: dict) -> "PowerModel":
        return cls(
            weights=tuple(float(w) for w in payload["weights"]),
            feature_names=tuple(payload["feature_names"]),
            base=str(payload["base"]), history_k=int(payload["history_k"]),
            fit_mode=str(payload.get("fit_mode", "relative")),
            train=dict(payload.get("train", {})), domain=dict(payload.get("domain", {})),
            version=str(payload.get("version", POWER_VERSION)),
            step_cost_dt=float(payload.get("step_cost_dt", STEP_COST_DT)),
            n_intervals=int(payload.get("n_intervals", N_INTERVALS)),
            j_min=float(payload.get("j_min", J_MIN)),
            notes=tuple(payload.get("notes", ())),
        )

    def schema_sha256(self) -> str:
        blob = json.dumps({"base": self.base, "history_k": self.history_k,
                           "feature_names": list(self.feature_names),
                           "step_cost_dt": self.step_cost_dt,
                           "n_intervals": self.n_intervals},
                          sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def sha256(self) -> str:
        """模型身份散列：只覆盖 :meth:`content_dict`（预测相关内容，无时间戳）。"""
        blob = json.dumps(self.content_dict(), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()

    def save(self, path: Path | str) -> str:
        """写模型 JSON 与 ``.sha256`` 侧车（侧车格式与协议一致）。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.to_dict(), sha256=self.sha256())
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")
        side = p.with_suffix(p.suffix + ".sha256")
        side.write_text(f"{self.sha256()}  {p.name}\n", encoding="utf-8")
        return self.sha256()

    @classmethod
    def load(cls, path: Path | str, verify_hash: bool = True) -> "PowerModel":
        p = Path(path)
        if not p.is_file():
            raise PowerModelError(f"模型文件不存在: {p}")
        payload = json.loads(p.read_text(encoding="utf-8"))
        model = cls.from_dict(payload)
        if verify_hash:
            recorded = payload.get("sha256")
            actual = model.sha256()
            if recorded is not None and recorded != actual:
                raise PowerModelError(f"模型散列不匹配：recorded {recorded} / actual {actual}")
            side = p.with_suffix(p.suffix + ".sha256")
            if side.is_file():
                want = side.read_text(encoding="utf-8").split()[0]
                if want != actual:
                    raise PowerModelError(f"模型侧车散列不匹配：{want} / {actual}")
        return model


# ---------------------------------------------------------------- 选型与评估
def select_candidate(candidates: dict[str, dict], tie_tol: float = 0.005) -> str:
    """预登记选型规则：先比 dev q90，落在 ``tie_tol`` 内的按参数少→K 小→名字升序。

    ``candidates[name]`` 需含 ``dev_q90``、``n_params``、``history_k``。
    """
    if not candidates:
        raise PowerModelError("没有候选模型")
    for name, c in candidates.items():
        if not math.isfinite(float(c["dev_q90"])):
            raise PowerModelError(f"候选 {name} 的 dev_q90 非有限")
    best = min(float(c["dev_q90"]) for c in candidates.values())
    close = [n for n, c in candidates.items() if float(c["dev_q90"]) <= best + tie_tol]
    close.sort(key=lambda n: (int(candidates[n]["n_params"]),
                              int(candidates[n]["history_k"]), n))
    return close[0]


@dataclass(frozen=True)
class EvalSample:
    trial_id: str
    family: str
    j_exec: float
    commands: np.ndarray

    def __post_init__(self) -> None:
        if not math.isfinite(self.j_exec) or self.j_exec <= 0:
            raise PowerModelError(f"{self.trial_id}: J_exec 必须为正有限值")


def evaluate(model: PowerModel, samples: Sequence[EvalSample], j_min: float | None = None) -> dict:
    """逐轨迹误差 + 分组报告（overall / family / 低速 / 过渡 / 平稳）。

    误差定义按 §10.5：``|J_nom − J_exec| / J_exec``；同时给出带符号相对偏差
    ``(J_nom − J_exec) / J_exec`` 以看系统性高/低估。
    """
    thr = model.j_min if j_min is None else float(j_min)
    rows: list[dict] = []
    for s in samples:
        p = model.step_power(s.commands)
        jn = float(p.sum() * model.step_cost_dt)
        ok_dom, viol = model.in_domain(s.commands)
        rows.append({
            "trial_id": s.trial_id, "family": s.family, "j_exec": float(s.j_exec),
            "j_nom": jn,
            "rel_err": abs(jn - s.j_exec) / s.j_exec,
            "rel_bias": (jn - s.j_exec) / s.j_exec,
            "non_computable": bool(jn < thr),
            "in_domain": ok_dom, "domain_violations": viol,
            "low_speed_share": low_speed_share(s.commands),
            "transition_share": transition_share(s.commands),
        })

    usable = [r for r in rows if not r["non_computable"]]

    def block(sel: list[dict]) -> dict:
        out = summarize(r["rel_err"] for r in sel)
        out["bias_q90"] = (q_nearest_rank([r["rel_bias"] for r in sel], 0.9)
                           if sel else None)
        out["bias_median"] = float(np.median([r["rel_bias"] for r in sel])) if sel else None
        return out

    families = sorted({r["family"] for r in rows})
    groups = {
        "family": {f: block([r for r in usable if r["family"] == f]) for f in families},
        "low_speed": {
            "yes": block([r for r in usable if r["low_speed_share"] >= LOW_SPEED_VX]),
            "no": block([r for r in usable if r["low_speed_share"] < LOW_SPEED_VX]),
        },
        "transition": {
            "heavy": block([r for r in usable if r["transition_share"] >= 0.2]),
            "light": block([r for r in usable if r["transition_share"] < 0.2]),
        },
    }
    return {
        "n": len(rows),
        "n_non_computable": len(rows) - len(usable),
        "n_out_of_domain": sum(1 for r in rows if not r["in_domain"]),
        "overall": block(usable),
        "groups": groups,
        "rows": rows,
    }


def admission(dev: dict, validate: dict, threshold: float = ADMIT_Q90,
              truncation_share: float = 0.0) -> tuple[bool, dict]:
    """§10.5 准入判定：dev/validate 的 q90 ≤ 阈值 + 非负性 + 截断为 0。"""
    checks = {
        "dev_q90_le_threshold": (dev["overall"].get("q90") is not None
                                 and dev["overall"]["q90"] <= threshold),
        "validate_q90_le_threshold": (validate["overall"].get("q90") is not None
                                      and validate["overall"]["q90"] <= threshold),
        "nonneg_constructive": truncation_share <= 0.0,
        "non_computable_share_ok": (validate["n"] == 0
                                    or validate["n_non_computable"] / validate["n"] <= 0.2),
    }
    return all(checks.values()), checks
