"""轨迹缓冲、离线目标重建与可恢复状态（NAV-032）。

这一层只管三件事，不含任何训练逻辑：

  1. **一条样本记录什么**（``RAW_FIELDS``）——观测、原动作、实际动作、log_prob、
     两个 value、y/z、能量、终止/截断、事件、地图与回合 id，一律逐字段写全，
     派生量单独放在 ``OBJ_FIELDS`` 里，便于"离线从原始字段重算并对账"。
  2. **怎么落盘而不重复写**：一个 chunk = 一个完整回合；文件先写临时名再
     ``os.replace``，manifest 是提交记录；chunk 序号必须等于已有条目数，
     ``global_step`` 必须与上一 chunk 首尾相接、chunk 内严格 +1。
  3. **中断/恢复要存什么**：模型、三个优化器、torch/numpy 两个 RNG、回合种子流、
     环境快照、半截回合的样本、配置与归一化口径。缺一项，"恢复 vs 连续"就不再
     逐位相同——``tests/test_rollout_resume.py`` 正是用这个比较来证明清单完整。

三处与参考实现（``RC-PPO/rl/EC-EFPPO.py``）**有意不同**，都写在下面各自的位置上：

  * 参考在一个定长 rollout 里自动复位，环境 done 与停止分段混用同一个 ``done``，
    跨 reset 的样本会被当成旧回合的后续；这里每个 chunk 只装一个回合，环境终止
    由 ``terminated`` / ``truncated`` 两个独立字段表达（NAV-030 的要求）。
  * 参考的联合策略目标是 ``g=max(reach,−energy)`` 上的 GAE(γ_reach_init, λ)；
    这里同样在 ``g=max(h,−z)`` 上做 λ-return，但 γ 取 **1**（``gamma_joint=γ_c``），
    因为契约的目标是未折扣 J，且 W 定义在未折扣量上。NAV-063 之前这里用
    ``W_sel − V_total`` 一次性差（无 λ 平滑），是 M6 的主要缺口。
  * 参考的 reach 目标用 critic 的 ``last_val_h`` 当尾部；这里用实测的
    ``h_T = max(g_norm(x_T), C·y)``，理由与参考一致：x_T 之后无观测，
    用被训练的同一个网络给自己造目标会把误差固化成目标。

单位与尺度全部沿用契约，不在这里另设：``c = c_model_J / Z_REF_J``（无量纲）、
``z_ratio`` 为同一归一化的有符号剩余预算、``V_h`` 与 ``g_norm`` 同为无量纲。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

import numpy as np

from . import targets as T
from ..observation import z_ratio_clip as obs_z_ratio_clip

# ---------------------------------------------------------------- schema
#: ``v2`` (NAV-062) 显式存 ``t`` 与 ``t+1`` 两侧的量。``v1`` 的 ``z_ratio``/
#: ``safety_y``/``goal_distance_m`` 语义是"步后"，被当作 ``h_t`` 用，属 t/t+1 混配；
#: 旧 chunk 按 v1 读，新 chunk 按 v2 写，两者都保留。
SCHEMA_BUFFER = "camra-navigation-rollout-v2"
SCHEMA_CHECKPOINT = "camra-navigation-checkpoint-v1"

#: 事件码。0 表示"该步未结束回合"，其余与名义环境 ``StepRecord.label`` 同名。
EVENT_CODES: dict[str, int] = {
    "none": 0,
    "collision": 1,
    "out_of_bounds": 2,
    "goal_reached": 3,
    "budget_exhausted": 4,
    "stagnation": 5,
    "timeout": 6,
}
EVENT_NAMES: dict[int, str] = {v: k for k, v in EVENT_CODES.items()}

#: 回合结束的三种性质，对应契约 ``bootstrap.types``：
#: ``failure`` 无 bootstrap 且给 FAIL_PENALTY；``success`` 无 bootstrap、目标取实际值；
#: ``truncation`` 用 ``x_T`` 的价值 bootstrap。
#:
#: 归类理由（NAV-062 按 NAV-061 事件真值表定死）：
#:   collision / out_of_bounds  —— 任务失败，吸收态，终止事件
#:   goal_reached              —— 任务成功，终止事件
#:   stagnation / timeout      —— 检测器/时限判停，属截断，必须 bootstrap
#:
#: ``budget_exhausted`` **不再是终止事件**（参考实现的 ``is_terminal`` 只看时间）。
#: 该键只为 ``v1`` 旧 chunk 保留解析能力，新代码不会产生它。
TERMINAL_KIND: dict[str, str] = {
    "collision": "failure",
    "out_of_bounds": "failure",
    "goal_reached": "success",
    "stagnation": "truncation",
    "timeout": "truncation",
    "budget_exhausted": "failure",  # legacy v1 chunks only
}


class StorageError(RuntimeError):
    """缓冲/状态文件的契约被违反（重复写入、错位、缺字段、哈希不符）。"""


TERMINAL_TAIL_ESTIMATOR = "rcppo-terminal-tail-v5"
FEEDBACK_ESTIMATORS = ("rcppo-feedback-v4", TERMINAL_TAIL_ESTIMATOR)


def terminal_kind_for_event(label: str, cfg: "ObjectiveConfig") -> str:
    """Versioned task ending; legacy experiment semantics remain readable."""
    if cfg.estimator_version in FEEDBACK_ESTIMATORS and label in ("timeout", "stagnation"):
        return "task_end"
    try:
        return TERMINAL_KIND[label]
    except KeyError as exc:
        raise StorageError(f"unknown terminal event {label!r}") from exc


# ---------------------------------------------------------------- 字段表
@dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: str
    shape: tuple[int, ...] = ()


def raw_fields(obs_dim: int, action_dim: int = 3) -> tuple[FieldSpec, ...]:
    """逐样本原始字段。顺序即 npz 内的键序（``content_hash`` 依赖它，不要重排）。

    ``schema v2``（NAV-062）把两侧写全，命名规则固定：
    无后缀 = 决策时刻 ``t``；``*_next`` = 下一时刻 ``t+1``。
    因此 ``value_*(x_t)`` 只能与 ``z_ratio``/``safety_y``/``goal_distance_m`` 配，
    ``z_ratio_next`` 只允许出现在 ``x_{t+1}`` 的输入里。
    """
    return (
        FieldSpec("global_step", "<i8"),
        FieldSpec("episode_id", "<i8"),
        FieldSpec("t", "<i4"),
        FieldSpec("obs", "<f4", (obs_dim,)),
        FieldSpec("next_obs", "<f4", (obs_dim,)),
        FieldSpec("action_raw", "<f4", (action_dim,)),
        FieldSpec("action_applied", "<f4", (action_dim,)),
        FieldSpec("log_prob", "<f4"),
        FieldSpec("value_cost", "<f4"),
        FieldSpec("value_reach", "<f4"),
        FieldSpec("cost_j", "<f8"),
        FieldSpec("cost_norm", "<f8"),
        FieldSpec("z_j", "<f8"),
        #: 未裁剪的有符号比值 ``z_J / Z_REF_J``：目标用它（可加），观测不用它。
        FieldSpec("z_norm", "<f8"),
        FieldSpec("z_ratio", "<f4"),
        FieldSpec("safety_y", "<f4"),
        FieldSpec("goal_distance_m", "<f4"),
        FieldSpec("z_norm_next", "<f8"),
        FieldSpec("z_ratio_next", "<f4"),
        FieldSpec("safety_y_next", "<f4"),
        FieldSpec("goal_distance_m_next", "<f4"),
        FieldSpec("terminated", "?"),
        FieldSpec("truncated", "?"),
        FieldSpec("event_code", "i1"),
        FieldSpec("map_index", "<i2"),
        FieldSpec("bootstrap_value_cost", "<f4"),
        FieldSpec("bootstrap_value_reach", "<f4"),
        FieldSpec("action_clipped", "?"),
        FieldSpec("action_invalid", "?"),
    )


def obj_fields() -> tuple[FieldSpec, ...]:
    """由原始字段离线重算的派生量（存一份，用于逐位对账，不参与采样）。"""
    return (
        FieldSpec("obj_g_norm", "<f8"),
        FieldSpec("obj_h", "<f8"),
        FieldSpec("obj_h_terminal", "<f8"),
        FieldSpec("obj_z_norm", "<f8"),
        FieldSpec("obj_g", "<f8"),
        FieldSpec("obj_g_train", "<f8"),
        FieldSpec("obj_g_terminal", "<f8"),
        FieldSpec("obj_spent", "<f8"),
        FieldSpec("obj_target_cost", "<f8"),
        FieldSpec("obj_adv_cost", "<f8"),
        FieldSpec("obj_target_reach", "<f8"),
        FieldSpec("obj_adv_reach", "<f8"),
        FieldSpec("obj_total_value", "<f8"),
        FieldSpec("obj_target_total", "<f8"),
        FieldSpec("obj_adv_total", "<f8"),
        FieldSpec("obj_stop_index", "<i4"),
        FieldSpec("obj_stop_residual", "<f8"),
        FieldSpec("obj_stop_now", "?"),
        FieldSpec("obj_feasible", "?"),
        FieldSpec("obj_z_saturated", "?"),
        FieldSpec("obj_cost_return", "<f8"),
    )


# ---------------------------------------------------------------- 目标口径
@dataclass(frozen=True)
class ObjectiveConfig:
    """目标计算所需的全部常数，一律从只读契约读出。"""

    gamma_cost: float
    gamma_h: float
    gae_lambda: float
    beta: float
    C: float
    Z_REF_J: float
    L_REF_m: float
    G_INSIDE: float
    fail_penalty: float
    r_goal_m: float
    n_actions: int = 3
    #: γ of the joint λ-return (the policy target).  Contract: the objective is
    #: undiscounted J and W is defined on undiscounted quantities, so this is
    #: γ_c = 1, not the reference's ``GAMMA_REACH_INIT`` (0.999).  Kept explicit
    #: so the deviation is visible and testable instead of implicit.
    gamma_joint: float | None = None
    #: NAV-064 修正 ①（预登记 config/objective_signal_fix_prereg_v1.json）：
    #: 训练用即时代价里逐步成本项 ``w·c_t`` 的权重。0 = 关闭（与改动前逐位等价）。
    #: 只影响策略用的 obj_target_total / obj_adv_total，不影响 W / τ* / 可行性判据。
    joint_cost_weight: float = 0.0
    #: NAV-064 成本项遮蔽（预登记 objective_cost_mask_prereg_v1）：True 时
    #: ``w·c_t`` 只在已超支（z<0）的步计入即时代价。默认 False 与改动前逐位等价。
    #: 只影响策略用的 g_train；W / τ* / 可行性判据一律不动。
    joint_cost_on_overspend_only: bool = False
    #: NAV-064 r12→r13：预算**内**的逐步成本权重（``base_weight``）。
    #: 与 ``joint_cost_weight`` 配合：超支步仍用大权重 w，未超支步用这个小权重。
    #: 实测依据（reports/contract_margin_eval.json）：把预算抬高 15% 后，策略把多出来的
    #: 额度直接花掉（能耗/预算从 92% 涨到 114-123%），因为未超支时成本项被遮蔽，
    #: 「省着用」没有任何梯度。默认 0.0，表示不额外惩罚未超支步（与改动前逐位等价）。
    joint_cost_base_weight: float = 0.0
    #: NAV-064 修正 ②：是否启用参考的 min 型联合递推。默认 False——参考环境
    #: 的 reach 到达时为大负数，本仓库 h 被 max(·,0) 抹平，启用会翻转到达梯度
    #: （见 amend3 的三条数值反例）。保留可调用实现供对照与反例测试。
    joint_min_recursion: bool = False
    #: NAV-064 r14 h 语义版本（用户 2026-11-15T17:32 授权）：
    #: - ``"v1"``（默认，r1-r13）：安全态 ``h = max(g_norm, 0)``，到达无奖励。
    #: - ``"v2_deep_well"``：安全态 ``h = g_norm``（目标集内为深井 G_INSIDE=-3），
    #:   违规态仍 ``max(g_norm, C·y)``。必须与值契约 v1.3.0（G_INSIDE=-3）与
    #:   min 型递推一起由预登记显式声明，训练器校验三者同现。
    h_semantics: str = "v1"
    estimator_version: str = "legacy-v2"

    @classmethod
    def from_contracts(cls, value_contract_path: str, *, r_goal_m: float) -> "ObjectiveConfig":
        with open(value_contract_path, encoding="utf-8") as fh:
            c = json.load(fh)
        est, q = c["estimator"], c["quantities"]
        fail = c["bootstrap"]["FAIL_PENALTY"]
        if (est.get("version") == TERMINAL_TAIL_ESTIMATOR
                and est.get("min_gae_terminal_weights") != "geometric_terminal_tail"):
            raise StorageError("terminal-tail-v5 requires explicit terminal weight semantics")
        return cls(
            gamma_cost=float(est["gamma_cost"]),
            gamma_h=float(est["gamma_h_init"]),
            gae_lambda=float(est["gae_lambda"]),
            beta=float(q["beta"]["value"]),
            C=float(q["C"]["value"]),
            Z_REF_J=float(q["Z_REF_J"]["value"]),
            L_REF_m=float(q["g_norm"]["L_REF_m"]),
            G_INSIDE=float(q["g_norm"]["G_INSIDE"]),
            fail_penalty=float(fail["value"]),
            r_goal_m=float(r_goal_m),
            gamma_joint=float(est.get("gamma_joint", est["gamma_cost"])),
            estimator_version=str(est.get("version", "legacy-v2")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def g_norm_from_distance(distance_m: np.ndarray, cfg: ObjectiveConfig) -> np.ndarray:
    """``(dist − r_goal) / L_REF``，进入目标集时取 ``G_INSIDE``（契约定义）。"""
    d = np.asarray(distance_m, dtype=np.float64)
    return np.where(d <= cfg.r_goal_m, cfg.G_INSIDE, (d - cfg.r_goal_m) / cfg.L_REF_m)


# ---------------------------------------------------------------- 离线重算
def compute_objectives(
    arrays: Mapping[str, np.ndarray],
    cfg: ObjectiveConfig,
    *,
    terminal_kind: str | None = None,
) -> dict[str, np.ndarray]:
    """从一个回合的**原始字段**重算全部目标；返回长度 T（``h`` 为 T+1）。

    记号沿用契约：``c_t`` 为归一化成本，``z_t`` 为**未裁剪**的有符号比值，
    ``h_t = max(g_norm(x_t), C·y_t)``。

    **两侧严格分开（NAV-062 修正 t/t+1 混配）**：无后缀字段是决策时刻 ``t`` 的量
    （``obs``/``value_*`` 就在这个状态上算的），``*_next`` 是 ``t+1``。旧实现用
    ``goal_distance_m``（步后）与 ``safety_y``（步后）组装 ``h_t``，等于把
    ``h_{t+1}`` 配给了 ``V(x_t)``，再在末尾重复补一次 ``h[-1]``。现在：

      ``h_t``      t=0..T-1，由 ``goal_distance_m``/``safety_y`` 直接算；
      ``h_T``      由 ``goal_distance_m_next[-1]``/``safety_y_next[-1]`` 算，**只补一次**；
      ``h_ext``    长度 T+1，供 W(t,T) 与 min 型递推收尾。

    终端处理（契约 ``terminal_and_truncation`` 表，逐条落地）：

      failure     ``V_c`` 目标在失败步**严格等于** FAIL_PENALTY（不叠加该步成本），
                  ``h_T = FAIL_PENALTY``；无 bootstrap。
      success     ``V_c`` 目标 = 尾段实际成本（终端值 0），``h_T = g_norm(x_T)``；
                  无 bootstrap。
      truncation  ``V_c`` 目标 = 尾段成本 + ``V_c(x_T)``（采集时存下的 critic 值），
                  ``h_T`` 用实测 ``max(g_norm(x_T), C·y_T)``。

    **三套估计器共用同一个停止掩码（NAV-063）**：
    :func:`targets.select_stop` 逐时刻在尾段取 τ*，``done[t]=1[τ*(t)=t]`` 同时喂给

      ① 成本：``cost_to_stop`` —— 到 τ* 的实测成本，仅 τ*=T 加 ``V_c(x_T)``；
      ② reach：``lambda_return`` 作用在 min 型 DP 值 ``V_h`` 上；
      ③ 联合策略：``lambda_return`` 作用在训练用即时代价
         ``g_train=max(h,−z)+w·c`` 上，减去 ``V_total=max(V_h, V_c−z)``。
         NAV-064 起 w 由 ``ObjectiveConfig.joint_cost_weight`` 控制（默认 0，
         训练预登记冻结为 50）；只改策略梯度，不改 W / τ* / 可行性判据。

    旧实现里成本取整回合 suffix、reach 取未掩码的最小值、policy 用
    ``W_sel − V_total`` 一次性差，三者各自隐含一个不同的终点（M6）。
    ``W`` 的 ``z`` 用**未裁剪**比值：观测里的截断值只是编码，不能进目标。
    """
    c = np.asarray(arrays["cost_norm"], dtype=np.float64).reshape(-1)
    missing = [k for k in ("z_norm", "z_ratio_next", "safety_y_next",
                           "goal_distance_m_next") if k not in arrays]
    if missing:
        raise StorageError(
            f"样本缺少 t/t+1 配对字段 {missing}：这是 schema v1 的旧 chunk，"
            "不能用于训练；请用 NAV-062 之后写入的 chunk"
        )
    z = np.asarray(arrays["z_norm"], dtype=np.float64).reshape(-1)
    z_next = np.asarray(arrays["z_norm_next"], dtype=np.float64).reshape(-1)
    vc = np.asarray(arrays["value_cost"], dtype=np.float64).reshape(-1)
    vh = np.asarray(arrays["value_reach"], dtype=np.float64).reshape(-1)
    y = np.asarray(arrays["safety_y"], dtype=np.float64).reshape(-1)
    goal_d = np.asarray(arrays["goal_distance_m"], dtype=np.float64).reshape(-1)
    y_next = np.asarray(arrays["safety_y_next"], dtype=np.float64).reshape(-1)
    goal_d_next = np.asarray(arrays["goal_distance_m_next"], dtype=np.float64).reshape(-1)
    term = np.asarray(arrays["terminated"], dtype=bool).reshape(-1)
    trunc = np.asarray(arrays["truncated"], dtype=bool).reshape(-1)
    b_c = np.asarray(arrays["bootstrap_value_cost"], dtype=np.float64).reshape(-1)
    feedback = cfg.estimator_version in FEEDBACK_ESTIMATORS
    repaired = cfg.estimator_version == "rcppo-r14-v3" or feedback
    if repaired and "bootstrap_value_reach" not in arrays:
        raise StorageError("rcppo-r14-v3 requires the actual final reach critic")
    b_h = np.asarray(arrays.get("bootstrap_value_reach", np.zeros_like(b_c)),
                     dtype=np.float64).reshape(-1)

    Tn = c.shape[0]
    if Tn == 0:
        raise StorageError("空回合无法计算目标")
    for name, arr in (("z_norm", z), ("value_cost", vc), ("value_reach", vh),
                      ("safety_y", y), ("goal_distance_m", goal_d),
                      ("z_norm_next", z_next),
                      ("safety_y_next", y_next), ("goal_distance_m_next", goal_d_next),
                      ("terminated", term), ("truncated", trunc),
                      ("bootstrap_value_cost", b_c), ("bootstrap_value_reach", b_h)):
        if arr.shape[0] != Tn:
            raise StorageError(f"{name} 长度 {arr.shape[0]} ≠ T={Tn}")
    if terminal_kind not in (("failure", "success", "truncation", "task_end")
                             if feedback else ("failure", "success", "truncation")):
        raise StorageError(f"未知 terminal_kind={terminal_kind!r}")
    if feedback:
        if (cfg.gamma_cost != 1.0 or cfg.gamma_joint != 1.0
                or cfg.joint_cost_weight != 0.0 or cfg.joint_cost_base_weight != 0.0):
            raise StorageError("feedback-v4 requires undiscounted costs/joint reach and no additive costs")
        if "event_code" not in arrays:
            raise StorageError("feedback-v4 requires explicit ending event identity")
        event = EVENT_NAMES.get(int(np.asarray(arrays["event_code"])[-1]))
        expected_kind = "truncation" if event == "none" else terminal_kind_for_event(event, cfg)
        if terminal_kind != expected_kind:
            raise StorageError(f"feedback-v4 ending {event!r} requires {expected_kind!r}")
        if bool(trunc[-1]) != (terminal_kind == "truncation"):
            raise StorageError("feedback-v4 task ends and external slices have different flags")
        if (np.any(y_next < y) or np.any(y < 0.0) or np.any(y_next > 1.0)
                or not np.allclose(y_next[:-1], y[1:], atol=1e-12, rtol=0.0)):
            raise StorageError("feedback-v4 requires consistent irreversible safety history")
    if bool(term[0: Tn - 1].any()) or bool(trunc[0: Tn - 1].any()):
        raise StorageError("回合中间出现终止/截断标记：一个 chunk 只允许装一个完整回合")
    if bool(term[-1]) and bool(trunc[-1]):
        raise StorageError("最后一步同时标记 terminated 与 truncated")
    if not (bool(term[-1]) or bool(trunc[-1])):
        raise StorageError("最后一步既未终止也未截断：chunk 不是完整回合")
    # 预算在段内越负是正常业务事实（预算耗尽不再终止回合），而**段首**也不一定
    # 是回合起点：窗口切片会给仍在跑的回合开新段，此时 z[0] 已经是负的。
    # 因此"z 必须落在观测编码范围内"不能作为段级硬错误——初始预算的检查放在
    # 知道 z0 的地方（train_formal.BudgetSampler 对 z0 > clip*Z_REF 直接拒绝）。
    # 段内饱和只作诊断记录：观测在 ±clip 处饱和，W 用的是未裁剪 z_norm，可加性
    # 不受影响（NAV-064 实测：满预算跑到 30 s 时 z_norm 可达 −1.6）。
    z_clip = float(obs_z_ratio_clip())
    z_saturated = np.abs(z) > z_clip

    # --- t 侧：h_t = max(g_norm(x_t), C·y_t) ----------------------------
    # r14 起语义可由预登记选 v2（深井）：安全态 h = g_norm 不被 0 裁剪，
    # 违规态（含分级严重度）不变，见 targets.safety_step_v2。
    g = g_norm_from_distance(goal_d, cfg)
    if getattr(cfg, "h_semantics", "v1") == "v2_deep_well":
        h_t = np.asarray(T.safety_step_v2(g, y, cfg.C), dtype=np.float64)
    else:
        h_t = np.asarray(T.safety_step(g, y, cfg.C), dtype=np.float64)
    # --- T 侧：终态只补一次 -------------------------------------------------
    g_T = float(g_norm_from_distance(goal_d_next[-1:], cfg)[0])
    y_T = float(y_next[-1])
    if getattr(cfg, "h_semantics", "v1") == "v2_deep_well":
        h_T_measured = float(max(g_T, cfg.C * y_T)) if y_T > 0.0 else g_T
    else:
        h_T_measured = float(max(g_T, cfg.C * y_T))
    if terminal_kind == "failure":
        h_T = float(cfg.fail_penalty)
    else:
        # success 与 truncation 都用实测值。契约 terminal 表把 success 写成
        # "h_T = g_T"，但 safety_history.arrival_rule 与事件真值表都要求
        # "危险后到达"h=1.0→不可行；若在到达处丢掉 C·y，等于让到达清零 y，
        # 正是契约要求用反例拦住的那件事。故此处以不可逆安全历史为准：
        # h_T = max(g_T, C·y_T)，success 与 truncation 的差别只在成本终端项
        # （success 不 bootstrap，truncation 加 V_c(x_T)）。
        h_T = h_T_measured
    h_ext = np.concatenate([h_t, [h_T]])

    # --- 终端成本项：只有截断才 bootstrap --------------------------------
    if terminal_kind == "truncation":
        term_cost = float(b_c[-1])           # x_T 的 V_c，采集时存下
    else:
        term_cost = 0.0                      # 成功与失败都不 bootstrap

    # A slice has both a measured stopping signal at x_T and its actual critic.
    # The min Bellman operator permits stopping there, so its boundary is capped
    # by that measured signal. True terminal states use no learned continuation.
    reach_boundary = (min(h_T, float(b_h[-1])) if terminal_kind == "truncation"
                      else h_T)
    if feedback and terminal_kind == "truncation":
        reach_boundary = float(T.bounded_reach_continuation(
            reach_boundary, h_T, y_T, cfg.C))

    # --- 共同停止：三套估计器共用同一个 τ* 与 done ----------------------
    gamma_joint = cfg.gamma_cost if cfg.gamma_joint is None else float(cfg.gamma_joint)
    stop_index, stop_residual, stop_now = T.select_stop(
        c, z, h_ext, vc_terminal=term_cost, gamma=cfg.gamma_cost)
    if terminal_kind == "failure":
        # 失败是吸收态：回合在 T 处以 FAIL_PENALTY 结束，agent 不能"提前声明停止"
        # 来回避它。若不强制，预算宽松时 argmin 会选一个更早的 τ*，联合目标就
        # 只剩 0.1 量级的安全价差，失败惩罚根本没进前序目标（契约明确要求
        # 失败惩罚传播到全部相关目标）。因此失败回合的停止点固定为 T。
        stop_index = np.full(Tn, Tn, dtype=np.int64)
        stop_now = np.zeros(Tn, dtype=bool)
        W = T.stop_residual(c, z, h_ext, vc_terminal=term_cost, gamma=cfg.gamma_cost)
        stop_residual = W[np.arange(Tn), stop_index]
    done_ext = np.concatenate([np.asarray(stop_now, dtype=np.int64), [1]])
    g_ext = T.joint_sequence(h_t, z, h_T, z_T=float(z_next[-1]),
                             vc_terminal=term_cost)

    # --- ① 成本估计器：**到停止**的实测成本 + 仅 τ*=T 的 bootstrap -------
    # 旧实现取整回合 suffix，停止之后的成本会污染 τ*<T 的目标（且与"到停止"
    # 的语义不符）。这里与 W 共用 spent 矩阵的同一列，末步成本不漏不重。
    target_cost = T.cost_to_stop(c, stop_index, vc_terminal=term_cost,
                                 gamma=cfg.gamma_cost)
    cumulative_cost_target = target_cost.copy()
    if terminal_kind == "failure":
        # 契约：失败步的 V_c 目标严格等于 FAIL_PENALTY（不叠加该步成本）
        target_cost = target_cost.copy()
        target_cost[-1] = cfg.fail_penalty
    adv_cost = target_cost - vc

    # --- ② reach 估计器：min 型 DP 值序列上的 λ-return（同一 done） ------
    # 序列长度 T+1（含终端），done_ext 的末项固定为 1（终端即停止）
    vdp_ext = np.asarray(T.reach_values(h_ext, cfg.gamma_h), dtype=np.float64)
    target_reach = T.lambda_return(vdp_ext, done_ext, cfg.gae_lambda)[:Tn]
    adv_reach = target_reach - vh

    # --- ③ 联合策略估计器：g = max(h, −z) 上的 λ-return（同一 done） -----
    # 参考实现在 g_append 上做 GAE(γ_reach_init, λ=0.95)；本实现的 γ 取
    # cfg.gamma_joint = γ_c = 1（契约的目标是未折扣 J，W 也定义在未折扣量上，
    # 用 γ<1 会偷偷给后发生的成本打折，见 value 契约 gamma_cost_rationale）。
    # 这是与参考的**有意偏离**，差异量在 reports/rcppo_alignment.json 里单列。
    # NAV-064 修正 ①：把逐步成本以权重 w 加进即时代价，让「这一步花了多少」
    # 真正进入策略梯度。g_train 与 g_ext 只差前 T 项的 w·c_t；终端无动作，不变。
    # w=0 时 g_train 是 g_ext 的副本，下面逐位等价于改动前（回归保护）。
    g_train = T.joint_training_sequence(
        g_ext, c, cfg.joint_cost_weight,
        overspend_mask=(z < 0.0) if cfg.joint_cost_on_overspend_only else None,
        base_weight=cfg.joint_cost_base_weight)
    total_value = np.maximum(vh, vc - z)
    if cfg.joint_min_recursion and not repaired:
        # 严格参考形态：_ref_gae_reach4 已减过 baseline，Q 直接就是目标
        target_total = T.joint_min_return(g_train, total_value, done_ext,
                                          gamma_joint, cfg.gae_lambda,
                                          terminal_baseline=max(
                                              reach_boundary,
                                              term_cost - float(z_next[-1])))
    else:
        target_total = T.lambda_return(g_train, done_ext, cfg.gae_lambda)[:Tn]
    adv_total = target_total - total_value
    if repaired:
        if not cfg.joint_min_recursion or cfg.h_semantics != "v2_deep_well":
            raise StorageError("rcppo-r14-v3 requires the registered min/deep-well semantics")
        effective_cost = c.copy()
        if terminal_kind == "failure":
            # Existing failure convention: replace the final step by the
            # absorbing penalty, rather than adding the final action twice.
            effective_cost[-1] = cfg.fail_penalty
        vc_ext = np.concatenate([vc, [term_cost]])
        vh_continuation = (T.bounded_reach_continuation(
            vh, h_t, y, cfg.C, allow_stop=terminal_kind != "failure")
                           if feedback else vh)
        vh_ext = np.concatenate([vh_continuation, [reach_boundary]])
        joint_boundary = max(reach_boundary, term_cost - float(z_next[-1]))
        continuation_total = (np.maximum(vh_continuation, vc - z)
                              if feedback else total_value)
        total_ext = np.concatenate([continuation_total, [joint_boundary]])
        shared_done = np.asarray(stop_now, dtype=bool)
        terminal_tail = (cfg.estimator_version == TERMINAL_TAIL_ESTIMATOR
                         and terminal_kind != "truncation")
        target_cost = T.cost_gae_return(effective_cost, vc_ext, shared_done,
                                       cfg.gamma_cost, cfg.gae_lambda)
        target_reach = T.min_gae_return(h_ext, vh_ext, shared_done,
                                       cfg.gamma_h, cfg.gae_lambda,
                                       force_continue=terminal_kind == "failure",
                                       terminal_tail=terminal_tail)
        target_total = T.min_gae_return(g_train, total_ext, shared_done,
                                       gamma_joint, cfg.gae_lambda,
                                       force_continue=terminal_kind == "failure",
                                       terminal_tail=terminal_tail)
        adv_cost = target_cost - vc
        adv_reach = target_reach - vh
        adv_total = target_total - total_value
    # 诊断用"到停止的成本"由同一目标减掉终端项得到，不重复计算
    spent_to_stop = (cumulative_cost_target
                     - np.where(stop_index == Tn, term_cost, 0.0))

    return {
        "obj_g_norm": g,
        "obj_h": h_t,
        "obj_h_terminal": np.full(Tn, h_T, dtype=np.float64),
        "obj_z_norm": z,
        "obj_spent": spent_to_stop,
        "obj_g": g_ext[:Tn],
        "obj_g_train": g_train[:Tn],
        "obj_g_terminal": np.full(Tn, float(g_ext[Tn]), dtype=np.float64),
        "obj_target_cost": target_cost,
        "obj_adv_cost": adv_cost,
        "obj_target_reach": target_reach,
        "obj_adv_reach": adv_reach,
        "obj_total_value": total_value,
        "obj_target_total": target_total,
        "obj_adv_total": adv_total,
        "obj_stop_index": stop_index.astype(np.int32),
        "obj_stop_residual": stop_residual,
        "obj_stop_now": np.asarray(stop_now, dtype=bool),
        "obj_feasible": np.asarray(T.is_feasible(stop_residual, cfg.beta), dtype=bool),
        "obj_z_saturated": np.asarray(z_saturated, dtype=bool),
        "obj_cost_return": target_cost,
    }


# ---------------------------------------------------------------- 回合缓冲
class Segment:
    """一个回合的样本累积器（内存内）；只接受与字段表一致的值。"""

    def __init__(self, *, episode_id: int, map_id: str, map_index: int, seed: int,
                 obs_dim: int, action_dim: int = 3) -> None:
        self.episode_id = int(episode_id)
        self.map_id = str(map_id)
        self.map_index = int(map_index)
        self.seed = int(seed)
        self.obs_dim = int(obs_dim)
        self.fields = raw_fields(obs_dim, action_dim)
        self._data: dict[str, list[Any]] = {f.name: [] for f in self.fields}

    def __len__(self) -> int:
        return len(self._data["global_step"])

    def append(self, **values: Any) -> None:
        missing = [f.name for f in self.fields if f.name not in values]
        if missing:
            raise StorageError(f"样本缺少字段：{missing}")
        extra = [k for k in values if k not in self._data]
        if extra:
            raise StorageError(f"未知字段：{extra}")
        for spec in self.fields:
            v = values[spec.name]
            if spec.shape:
                arr = np.asarray(v, dtype=np.float32).reshape(spec.shape)
            elif spec.dtype == "?":
                arr = bool(v)
            elif spec.name in ("global_step", "episode_id"):
                arr = int(v)
            elif spec.dtype == "<f8":
                arr = float(v)
            elif spec.dtype == "<f4":
                arr = float(v)
            else:
                arr = int(v)
            self._data[spec.name].append(arr)

    def arrays(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for spec in self.fields:
            out[spec.name] = np.asarray(self._data[spec.name], dtype=spec.dtype)
        return out

    def to_state(self) -> dict[str, Any]:
        """可放进 checkpoint 的纯 numpy 表示（中断时保存半截回合用）。"""
        return {
            "episode_id": self.episode_id,
            "map_id": self.map_id,
            "map_index": self.map_index,
            "seed": self.seed,
            "data": self.arrays(),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any], *, obs_dim: int,
                   action_dim: int = 3) -> "Segment":
        seg = cls(episode_id=state["episode_id"], map_id=state["map_id"],
                  map_index=state["map_index"], seed=state["seed"],
                  obs_dim=obs_dim, action_dim=action_dim)
        for spec in seg.fields:
            arr = np.asarray(state["data"][spec.name], dtype=spec.dtype)
            seg._data[spec.name] = list(arr)
        return seg


def content_hash(arrays: Mapping[str, np.ndarray]) -> str:
    """字段名 + dtype + shape + 原始字节的 sha256，与 zip 时间戳无关。"""
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode("utf-8"))
        h.update(str(a.dtype).encode("ascii"))
        h.update(str(a.shape).encode("ascii"))
        h.update(a.tobytes())
    return h.hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


class RolloutStore:
    """按 chunk 追加写盘；序号、步号、样本数三重对账，重复写入直接拒绝。"""

    def __init__(self, root: str, run_id: str, *, obs_dim: int, action_dim: int = 3) -> None:
        self.root = os.path.join(root, run_id, "rollout")
        self.run_id = run_id
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.manifest_path = os.path.join(self.root, "manifest.jsonl")
        os.makedirs(self.root, exist_ok=True)
        self.entries: list[dict[str, Any]] = self._read_manifest()

    # ---- 读 ----
    def _read_manifest(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.manifest_path):
            return []
        out = []
        with open(self.manifest_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    @property
    def n_chunks(self) -> int:
        return len(self.entries)

    @property
    def n_samples(self) -> int:
        return sum(int(e["n_samples"]) for e in self.entries)

    @property
    def last_global_step(self) -> int:
        return int(self.entries[-1]["global_step_end"]) if self.entries else -1

    def chunk_path(self, index: int) -> str:
        return os.path.join(self.root, f"chunk_{index:05d}.npz")

    # ---- 写 ----
    def expected_chunk_index(self) -> int:
        return len(self.entries)

    def write_chunk(self, arrays: Mapping[str, np.ndarray], meta: MutableMapping[str, Any]) -> dict[str, Any]:
        """提交一个 chunk。序号错位、步号不接续、样本数不符、重复覆盖都会抛错。"""
        idx = int(meta.get("chunk_index", -1))
        if idx != self.expected_chunk_index():
            raise StorageError(
                f"chunk 序号错位：期望 {self.expected_chunk_index()}，收到 {idx}")
        n = int(len(arrays["global_step"]))
        if n == 0:
            raise StorageError("空 chunk 不写盘")
        steps = np.asarray(arrays["global_step"], dtype=np.int64)
        if not np.array_equal(steps, np.arange(steps[0], steps[0] + n)):
            raise StorageError("chunk 内 global_step 不连续")
        if int(steps[0]) != self.last_global_step + 1:
            raise StorageError(
                f"global_step 未接续：上一 chunk 止于 {self.last_global_step}，本 chunk 起于 {int(steps[0])}")
        if int(meta.get("n_samples", n)) != n:
            raise StorageError(f"meta.n_samples {meta.get('n_samples')} ≠ 实际 {n}")

        meta = dict(meta)
        meta.update({
            "schema": SCHEMA_BUFFER,
            "chunk_index": idx,
            "run_id": self.run_id,
            "n_samples": n,
            "global_step_start": int(steps[0]),
            "global_step_end": int(steps[-1]),
            "content_sha256": content_hash(arrays),
            "file": os.path.basename(self.chunk_path(idx)),
        })

        path = self.chunk_path(idx)
        if os.path.exists(path):
            # 崩溃后重放：内容一致视为同一 chunk（幂等提交），否则拒绝覆盖
            with np.load(path) as z:
                old = {k: z[k] for k in z.files}
            if content_hash(old) != meta["content_sha256"]:
                raise StorageError(f"chunk {idx} 已存在且内容不同，拒绝覆盖：{path}")
        else:
            tmp = path + ".tmp.npz"
            np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in arrays.items()})
            os.replace(tmp, path)

        meta["file_sha256"] = file_sha256(path)
        with open(self.manifest_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(meta, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.entries.append(meta)
        return meta

    # ---- 读回 ----
    def load_chunk(self, index: int) -> dict[str, np.ndarray]:
        if not 0 <= index < len(self.entries):
            raise StorageError(f"chunk {index} 不在 manifest 中")
        path = self.chunk_path(index)
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}
        expect = self.entries[index]["content_sha256"]
        if content_hash(arrays) != expect:
            raise StorageError(f"chunk {index} 内容哈希不符：文件被改动过")
        return arrays


def audit_store(store: RolloutStore) -> dict[str, Any]:
    """只看 manifest + 落盘内容做一次完整对账（不需要环境，也不需要重训）。"""
    problems: list[str] = []
    events: dict[str, int] = {}
    kinds: dict[str, int] = {}
    expected_next = 0
    total = 0
    for pos, entry in enumerate(store.entries):
        if int(entry["chunk_index"]) != pos:
            problems.append(f"第 {pos} 条 manifest 的 chunk_index={entry['chunk_index']}，序号错位")
        if int(entry["global_step_start"]) != expected_next:
            problems.append(
                f"chunk {pos} 起始步 {entry['global_step_start']} ≠ 期望 {expected_next}（有缺口或重复）")
        try:
            arrays = store.load_chunk(pos)
        except StorageError as exc:
            problems.append(str(exc))
            continue
        steps = np.asarray(arrays["global_step"], dtype=np.int64)
        n = int(entry["n_samples"])
        if steps.shape[0] != n:
            problems.append(f"chunk {pos} 样本数 {steps.shape[0]} ≠ manifest {n}")
        if not np.array_equal(steps, np.arange(steps[0], steps[0] + steps.shape[0])):
            problems.append(f"chunk {pos} 内 global_step 不连续")
        if int(steps[-1]) != int(entry["global_step_end"]):
            problems.append(f"chunk {pos} 结束步与 manifest 不符")
        term = np.asarray(arrays["terminated"], dtype=bool)
        trunc = np.asarray(arrays["truncated"], dtype=bool)
        if bool((term | trunc)[:-1].any()):
            problems.append(f"chunk {pos} 回合中间出现终止/截断标记")
        if not bool((term | trunc)[-1]):
            problems.append(f"chunk {pos} 末步既未终止也未截断（chunk 不是完整回合）")
        if bool(term[-1]) and bool(trunc[-1]):
            problems.append(f"chunk {pos} 末步同时 terminated 与 truncated")
        name = EVENT_NAMES.get(int(np.asarray(arrays["event_code"])[-1]), "?")
        events[name] = events.get(name, 0) + 1
        kind = TERMINAL_KIND.get(name, "?")
        kinds[kind] = kinds.get(kind, 0) + 1
        expected_next = int(entry["global_step_end"]) + 1
        total += n
    if total != store.n_samples:
        problems.append(f"样本数对账失败：逐 chunk 求和 {total} ≠ manifest {store.n_samples}")
    if total != expected_next:
        problems.append(f"global_step 覆盖 [0,{total}) 失败，末步结束于 {expected_next - 1}")
    return {
        "problems": problems,
        "n_chunks": store.n_chunks,
        "n_samples": store.n_samples,
        "next_global_step": expected_next,
        "event_counts": events,
        "terminal_kind_counts": kinds,
        "chunk_content_sha256": [e["content_sha256"] for e in store.entries],
    }



# ---------------------------------------------------------------- 环境快照
#: 名义环境里承重的可变状态。**这份清单就是 resume 正确性的全部依赖**：
#: 漏一个字段，"中断-恢复"与"连续跑"就不再逐位相同，测试会立刻失败。
#: ``_records`` 保留整段（用于 stagnation 扫描与回合汇总），``_history`` 同理。
ENV_SCALAR_FIELDS: tuple[str, ...] = (
    "t_s", "index", "done", "label", "safety_y",
    "travelled_m", "danger_steps", "internal_violation_steps",
)
ENV_TUPLE_FIELDS: tuple[str, ...] = ("_pose", "_applied_last")


def snapshot_env(env: Any) -> dict[str, Any]:
    """把名义环境的可变状态取成纯 python 对象（能进 torch.save）。"""
    snap: dict[str, Any] = {"scalars": {}, "tuples": {}, "history": [], "records": [],
                            "budget": {}, "cost_model": {}}
    for name in ENV_SCALAR_FIELDS:
        snap["scalars"][name] = getattr(env, name)
    for name in ENV_TUPLE_FIELDS:
        snap["tuples"][name] = list(getattr(env, name))
    snap["history"] = [tuple(h) for h in env._history]
    snap["records"] = list(env._records)
    snap["budget"] = {"spent_J": float(env.budget.spent_J), "charges": int(env.budget.charges)}
    snap["cost_model"] = {"t_s": float(env.cost_model._t_s), "steps": int(env.cost_model._steps)}
    return snap


def restore_env(env: Any, snap: Mapping[str, Any]) -> None:
    """``snapshot_env`` 的逆；不重新 reset，直接回到快照时刻。"""
    for name in ENV_SCALAR_FIELDS:
        setattr(env, name, snap["scalars"][name])
    for name in ENV_TUPLE_FIELDS:
        setattr(env, name, tuple(snap["tuples"][name]))
    env._history = [tuple(h) for h in snap["history"]]
    env._records = list(snap["records"])
    env.budget.spent_J = float(snap["budget"]["spent_J"])
    env.budget.charges = int(snap["budget"]["charges"])
    env.cost_model._t_s = float(snap["cost_model"]["t_s"])
    env.cost_model._steps = int(snap["cost_model"]["steps"])


# ---------------------------------------------------------------- 机密
#: 只要出现这些键，就说明有凭证被写进了日志/状态文件。
FORBIDDEN_KEYS: tuple[str, ...] = (
    "password", "passwd", "secret", "token", "api_key", "apikey",
    "private_key", "access_key", "credential", "authorization", "cookie",
)
_SECRET_VALUE_PREFIXES = ("ghp_", "gho_", "sk-", "xoxb-", "AKIA", "-----BEGIN")


def assert_no_secrets(obj: Any, path: str = "") -> None:
    """递归检查键名与"看起来像密钥"的值；命中即抛错，不静默放行。"""
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            low = str(k).lower()
            if any(bad in low for bad in FORBIDDEN_KEYS):
                raise StorageError(f"日志/状态里出现疑似凭证字段：{path}/{k}")
            assert_no_secrets(v, f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_no_secrets(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        if obj.startswith(_SECRET_VALUE_PREFIXES):
            raise StorageError(f"日志/状态里出现疑似密钥值：{path}")


def redact(text: str) -> str:
    """把 ``k=v`` 形式里落在禁用键上的值替换成 ``***``（日志行用）。"""
    out = text
    for bad in FORBIDDEN_KEYS:
        for sep in ("=", ": "):
            needle = f"{bad}{sep}"
            start = 0
            while True:
                i = out.lower().find(needle, start)
                if i < 0:
                    break
                j = i + len(needle)
                k = j
                while k < len(out) and out[k] not in " ,;}\n\t\"'":
                    k += 1
                out = out[:j] + "***" + out[k:]
                start = j + 3
    return out


# ---------------------------------------------------------------- 状态文件
def atomic_save(obj: Any, path: str) -> str:
    """临时文件 + fsync + ``os.replace``；返回写入文件的 sha256。"""
    import torch

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    digest = file_sha256(path)
    with open(path + ".sha256", "w", encoding="utf-8") as fh:
        fh.write(digest + "\n")
    return digest


def atomic_load(path: str) -> Any:
    """读状态文件并校验旁路 sha256；不一致说明写入被打断或文件被改。"""
    import torch

    side = path + ".sha256"
    if not os.path.exists(path):
        raise StorageError(f"状态文件不存在：{path}")
    if not os.path.exists(side):
        raise StorageError(f"状态文件缺少校验和：{side}")
    expect = open(side, encoding="utf-8").read().strip()
    actual = file_sha256(path)
    if expect != actual:
        raise StorageError(f"状态文件校验和不符（写入被打断或文件被改）：{path}")
    with open(path, "rb") as fh:
        return torch.load(fh, map_location="cpu", weights_only=False)


def normalization_block(*, adv_mean: float | None = None,
                        adv_std: float | None = None) -> dict[str, Any]:
    """归一化口径声明。

    本实现**没有**带状态的归一化：obs-v2 的每个字段已按契约归一化（有界、无量纲），
    优势按 minibatch 即时 ``(A − mean)/(std + 1e-8)`` 标准化，不累积滑动统计。
    因此状态文件里不需要"可恢复的统计量"，但必须把口径和最近一批的统计量写清楚——
    一旦将来引入带状态的归一化（RunningMeanStd 之类），这里必须扩展成真正的状态。
    """
    return {
        "obs_normalization": "none：obs-v2 各字段已由契约归一化（无量纲、有界）",
        "advantage_normalization": "per-minibatch stateless：(A − mean)/(std + 1e-8)",
        "has_stateful_normalizer": False,
        "last_adv_mean": None if adv_mean is None else float(adv_mean),
        "last_adv_std": None if adv_std is None else float(adv_std),
    }


def rng_state() -> dict[str, Any]:
    import torch

    return {
        "torch": torch.get_rng_state().numpy().tobytes(),
        "torch_cuda": None,
    }


def restore_rng(state: Mapping[str, Any]) -> None:
    import torch

    buf = np.frombuffer(state["torch"], dtype=np.uint8).copy()
    torch.set_rng_state(torch.from_numpy(buf))


def generator_state(gen: np.random.Generator) -> dict[str, Any]:
    return dict(gen.bit_generator.state)


def restore_generator(gen: np.random.Generator, state: Mapping[str, Any]) -> None:
    gen.bit_generator.state = dict(state)
