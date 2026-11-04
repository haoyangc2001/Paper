"""步成本接口（CAL-022）：把冻结名义功率模型接到"预算扣减"上。

本模块**只交付接口与口径**，不实现也不训练导航器。上层（CAL-030 之后的导航）按以下
契约消费，任何单位或语义变化都必须出新版本并重新散列。

单位与语义
----------
* 机械功/机械功率：``c_nom = P̂_nom(u_t) · 0.1 s``，单位 **J**；
* 预算扣减：``c_t = (1 + α) · c_nom``，α 来自 CAL-025，本模块不预设；
* **旧论文的 a.u. 预算与本预算不可混用**：本模块只接受 ``unit="J"``，
  任何 ``a.u.``/相对量都会在构造时被拒绝（见 :class:`BudgetState`）。

因果与状态
----------
冻结模型 ``power-v1``（C2_quad_k0）的 ``history_k = 0``，即只用当前命令，命令本身就是
完整的 Markov 输入。接口仍按 ``history_k = K`` 一般化：

* ``K = 0``：无增广状态，``step(u)`` 即可；
* ``K > 0``：上层状态必须增广为 ``(x, u_{t-1..t-K})``，:class:`StepCost` 负责维护这个
  滑窗，并在 ``reset`` 时按 ``reset_rule`` 决定清零还是沿用（跨 episode 不得沿用）。

不能假设"命令就是全部 Markov 状态"：一旦换成 ``K > 0`` 的模型，导航观测必须同步增广，
否则预算会用到不存在的历史。本模块把这条写成显式字段 ``augmented_state_spec``。
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .power import J_MIN, STEP_COST_DT, PowerModel, PowerModelError

COST_UNIT = "J"
BUDGET_KINDS = ("mechanical_J",)
RESET_RULES = ("zero_history", "require_explicit")


class StepCostError(RuntimeError):
    """接口使用不符契约（单位、形状、历史、预算口径）。"""


# ---------------------------------------------------------------- 预算状态
@dataclass
class BudgetState:
    """预算容器。单位只允许机械焦耳；与旧 a.u. 预算**不共享**同一实例。"""

    initial_J: float
    spent_J: float = 0.0
    unit: str = COST_UNIT
    kind: str = "mechanical_J"

    def __post_init__(self) -> None:
        if self.unit != COST_UNIT:
            raise StepCostError(f"只接受 unit={COST_UNIT!r}，得到 {self.unit!r}（旧 a.u. 预算请另建实例）")
        if self.kind not in BUDGET_KINDS:
            raise StepCostError(f"未知预算口径 {self.kind!r}（可用：{BUDGET_KINDS}）")
        if not math.isfinite(self.initial_J) or self.initial_J <= 0:
            raise StepCostError(f"initial_J 必须为正有限值，得到 {self.initial_J}")

    @property
    def remaining_J(self) -> float:
        return self.initial_J - self.spent_J

    @property
    def exhausted(self) -> bool:
        return self.spent_J >= self.initial_J

    @property
    def z_ratio(self) -> float:
        """RC-PPO 里的 ``z / z_ref``：剩余预算比例，落在 [0, 1]。"""
        return max(0.0, self.remaining_J / self.initial_J)

    def charge(self, cost_J: float) -> float:
        if not math.isfinite(cost_J) or cost_J < 0:
            raise StepCostError(f"单步成本必须是非负有限值，得到 {cost_J}")
        self.spent_J += cost_J
        return self.remaining_J

    def to_dict(self) -> dict:
        return {"unit": self.unit, "kind": self.kind, "initial_J": self.initial_J,
                "spent_J": self.spent_J, "remaining_J": self.remaining_J,
                "exhausted": self.exhausted, "z_ratio": self.z_ratio}


# ---------------------------------------------------------------- 单步成本
@dataclass(frozen=True)
class StepCostRecord:
    """一步（0.1 s）的成本记录；上层可直接序列化进回放/日志。"""

    t_s: float
    command: tuple[float, float, float]
    power_W: float
    cost_nom_J: float
    cost_J: float                 # 含 (1+α) 的实际扣减
    alpha: float
    unit: str = COST_UNIT

    def to_dict(self) -> dict:
        return {"t_s": self.t_s, "command": list(self.command), "power_W": self.power_W,
                "cost_nom_J": self.cost_nom_J, "cost_J": self.cost_J, "alpha": self.alpha,
                "unit": self.unit}


class StepCost:
    """把 :class:`PowerModel` 包装成逐步成本发生器。

    ``dt_s`` 默认 0.1 s，必须与模型训练时的命令间隔一致（``PowerModel.step_cost_dt``），
    否则直接用会得到错位的能量——构造时强制校验。
    """

    def __init__(self, model: PowerModel, alpha: float = 0.0, dt_s: float | None = None,
                 reset_rule: str = "zero_history", budget: BudgetState | None = None) -> None:
        self.model = model
        self.alpha = float(alpha)
        if self.alpha < 0:
            raise StepCostError(f"alpha 必须 ≥ 0，得到 {self.alpha}")
        self.dt_s = float(model.step_cost_dt if dt_s is None else dt_s)
        if abs(self.dt_s - model.step_cost_dt) > 1e-12:
            raise StepCostError(
                f"dt_s={self.dt_s} 与模型的 step_cost_dt={model.step_cost_dt} 不一致")
        if reset_rule not in RESET_RULES:
            raise StepCostError(f"未知 reset_rule {reset_rule!r}（可用：{RESET_RULES}）")
        self.reset_rule = reset_rule
        self.budget = budget
        self._history: list[tuple[float, float, float]] = []
        self._t = 0.0
        self._steps = 0

    # ---- 元信息 ----
    @property
    def history_k(self) -> int:
        return int(self.model.history_k)

    @property
    def needs_history(self) -> bool:
        return self.history_k > 0

    def augmented_state_spec(self) -> dict:
        """给上层看的观测增广说明；``K = 0`` 时不需要额外状态。"""
        return {
            "history_k": self.history_k,
            "augmented_state": ("none: command is the full Markov input" if not self.needs_history
                                else f"append last {self.history_k} commands"),
            "reset_rule": self.reset_rule,
            "dt_s": self.dt_s,
            "unit": COST_UNIT,
            "model_sha256": self.model.sha256(),
        }

    def schema(self) -> dict:
        return {
            "schema": "camra-step-cost-v1",
            "dt_s": self.dt_s,
            "unit": COST_UNIT,
            "history_k": self.history_k,
            "augmented_state_spec": self.augmented_state_spec(),
            "step": ["t_s", "command", "power_W", "cost_nom_J", "cost_J", "alpha", "unit"],
            "budget": ["unit", "kind", "initial_J", "spent_J", "remaining_J",
                       "exhausted", "z_ratio"],
            "notes": [
                "c_nom = P̂_nom(u_t)·0.1 s；c_t = (1+α)·c_nom。",
                "单位只允许 J；旧论文的 a.u. 预算不得与本预算混用。",
                "K>0 时必须把最近 K 条命令并入状态，reset 时按 reset_rule 处理。",
            ],
        }

    # ---- 生命周期 ----
    def reset(self, history: Sequence[Sequence[float]] | None = None) -> None:
        """新 episode 开始：时间归零，按 reset_rule 处理历史滑窗。"""
        self._t = 0.0
        self._steps = 0
        if self.reset_rule == "require_explicit":
            if history is None or len(history) != self.history_k:
                raise StepCostError(
                    f"reset_rule=require_explicit 需要显式给出 {self.history_k} 条历史命令")
            self._history = [tuple(float(v) for v in row) for row in history]
        else:
            self._history = [(0.0, 0.0, 0.0) for _ in range(self.history_k)]

    # ---- 单步 ----
    def _features_power(self, command: Sequence[float]) -> float:
        """算"当前命令驱动的那一个区间"的功率：当前命令 + 最近 K 条历史。

        构造 K+2 个命令点 ``[h_{t-K}, …, h_{t-1}, u_t, u_t]``，模型的 ``step_power``
        返回 K+1 个区间功率，索引 ``K`` 恰好以 ``u_t`` 为当前命令、以历史为滞后项。
        这样与 :func:`camra_calibration.power.feature_matrix` 的滞后约定完全一致。
        """
        cmd = tuple(float(v) for v in command)
        arr = np.asarray(list(self._history) + [cmd, cmd], dtype=np.float64)
        power = self.model.step_power(arr)
        return float(power[self.history_k])

    def step(self, command: Sequence[float]) -> StepCostRecord:
        cmd = tuple(float(v) for v in command)
        if len(cmd) != 3:
            raise StepCostError(f"命令应为 [vx,vy,wz]，得到 {cmd}")
        if not all(math.isfinite(v) for v in cmd):
            raise StepCostError(f"命令含 NaN/inf: {cmd}")
        # 与 power.feature_matrix 的"越界钳到首点"一致：episode 第一条命令就是它自己的历史。
        if self._steps == 0 and self.reset_rule == "zero_history" and self.history_k > 0:
            self._history = [cmd] * self.history_k
        p = self._features_power(cmd)
        cost_nom = p * self.dt_s
        cost = (1.0 + self.alpha) * cost_nom
        if self.budget is not None:
            self.budget.charge(cost)
        rec = StepCostRecord(t_s=self._t, command=cmd, power_W=p, cost_nom_J=cost_nom,
                             cost_J=cost, alpha=self.alpha)
        if self.needs_history:
            self._history = (self._history + [cmd])[-self.history_k:]
        self._t += self.dt_s
        self._steps += 1
        return rec


# ---------------------------------------------------------------- 便利函数
def trajectory_cost(model: PowerModel, commands, alpha: float = 0.0,
                    reset: bool = True) -> dict:
    """整条命令序列的累计成本与 J_nom 对照。

    与固定 30 s 窗口无关：任何长度（≥2 个命令点）都能算，用于验证"逐步相加 = J_nom"。
    """
    arr = np.asarray(commands, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 2:
        raise StepCostError(f"命令序列形状应为 (N≥2,3)，得到 {arr.shape}")
    sc = StepCost(model, alpha=alpha)
    if reset:
        sc.reset()
    records = [sc.step(row) for row in arr[:-1]]
    total_nom = float(sum(r.cost_nom_J for r in records))
    total = float(sum(r.cost_J for r in records))
    j_nom = float(model.j_nom(arr))
    return {
        "n_steps": len(records),
        "horizon_s": len(records) * sc.dt_s,
        "cost_nom_sum_J": total_nom,
        "cost_sum_J": total,
        "j_nom_J": j_nom,
        "abs_diff_J": abs(total_nom - j_nom),
        "rel_diff": abs(total_nom - j_nom) / max(j_nom, 1e-12),
        "equal": abs(total_nom - j_nom) <= 1e-9 * max(1.0, abs(j_nom)),
        "alpha": alpha,
        "records": records,
    }


def energy_ratio(j_exec: float, j_nom: float, j_min: float = J_MIN) -> tuple[float | None, str]:
    """执行/名义能量比 ε_c = (J_exec − J_nom)/J_nom。

    与 :func:`camra_calibration.pairing.epsilon_c` 同口径：``J_nom < j_min`` 记
    ``non_computable``；**相等时返回 0.0 而不是被夹到极小正数**——人工同能量案例必须
    原样保留 0。
    """
    if not (math.isfinite(j_exec) and math.isfinite(j_nom)):
        return None, "nan"
    if j_nom < j_min:
        return None, "non_computable"
    return (j_exec - j_nom) / j_nom, "ok"


def schema_sha256(schema: dict) -> str:
    blob = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def write_schema(path: Path | str, model: PowerModel, alpha: float | None = None) -> str:
    """落盘 step-cost schema（含模型散列），返回 schema 散列。"""
    sc = StepCost(model, alpha=0.0 if alpha is None else alpha)
    payload = dict(sc.schema(), model_sha256=model.sha256(),
                   model_version=model.version, alpha_default=0.0 if alpha is None else alpha)
    payload["sha256"] = schema_sha256({k: v for k, v in payload.items() if k != "sha256"})
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                 encoding="utf-8")
    p.with_suffix(p.suffix + ".sha256").write_text(f"{payload['sha256']}  {p.name}\n",
                                                   encoding="utf-8")
    return payload["sha256"]
