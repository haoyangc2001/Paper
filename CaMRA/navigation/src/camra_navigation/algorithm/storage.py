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
  * 参考的 policy 优势 = 对"停止即走的残差" g 做 λ 平滑的 GAE；这里用
    ``W_sel − V_total``（残差的一阶差），因为 ``W_sel`` 已含整条尾段信息，
    再做 λ 平滑会把 τ* 之后的步混进来，与"按 τ* 截断、其后不计入优势"冲突。
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

    @classmethod
    def from_contracts(cls, value_contract_path: str, *, r_goal_m: float) -> "ObjectiveConfig":
        with open(value_contract_path, encoding="utf-8") as fh:
            c = json.load(fh)
        est, q = c["estimator"], c["quantities"]
        fail = c["bootstrap"]["FAIL_PENALTY"]
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

    ``W`` 用 :func:`camra_navigation.algorithm.targets.select_stop` 逐时刻在尾段重取
    停止（共享同一 τ*），``V_total = max(V_h, V_c − z_t)``，policy 目标
    ``W_sel − V_total``（见模块文档第 2 条差异）。``W`` 的 ``z`` 用**未裁剪**比值：
    观测里的截断值只是编码，不能进目标，否则 ``W`` 不再可加。
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
    vc = np.asarray(arrays["value_cost"], dtype=np.float64).reshape(-1)
    vh = np.asarray(arrays["value_reach"], dtype=np.float64).reshape(-1)
    y = np.asarray(arrays["safety_y"], dtype=np.float64).reshape(-1)
    goal_d = np.asarray(arrays["goal_distance_m"], dtype=np.float64).reshape(-1)
    y_next = np.asarray(arrays["safety_y_next"], dtype=np.float64).reshape(-1)
    goal_d_next = np.asarray(arrays["goal_distance_m_next"], dtype=np.float64).reshape(-1)
    term = np.asarray(arrays["terminated"], dtype=bool).reshape(-1)
    trunc = np.asarray(arrays["truncated"], dtype=bool).reshape(-1)
    b_c = np.asarray(arrays["bootstrap_value_cost"], dtype=np.float64).reshape(-1)

    Tn = c.shape[0]
    if Tn == 0:
        raise StorageError("空回合无法计算目标")
    for name, arr in (("z_norm", z), ("value_cost", vc), ("value_reach", vh),
                      ("safety_y", y), ("goal_distance_m", goal_d),
                      ("safety_y_next", y_next), ("goal_distance_m_next", goal_d_next),
                      ("terminated", term), ("truncated", trunc),
                      ("bootstrap_value_cost", b_c)):
        if arr.shape[0] != Tn:
            raise StorageError(f"{name} 长度 {arr.shape[0]} ≠ T={Tn}")
    if terminal_kind not in ("failure", "success", "truncation"):
        raise StorageError(f"未知 terminal_kind={terminal_kind!r}")
    if bool(term[0: Tn - 1].any()) or bool(trunc[0: Tn - 1].any()):
        raise StorageError("回合中间出现终止/截断标记：一个 chunk 只允许装一个完整回合")
    if bool(term[-1]) and bool(trunc[-1]):
        raise StorageError("最后一步同时标记 terminated 与 truncated")
    if not (bool(term[-1]) or bool(trunc[-1])):
        raise StorageError("最后一步既未终止也未截断：chunk 不是完整回合")
    z_clip = float(obs_z_ratio_clip())
    if float(np.abs(z).max()) > z_clip:
        raise StorageError(
            f"未裁剪的 z_norm={float(np.abs(z).max()):.4f} 超出观测编码范围 ±{z_clip}："
            "W 需要未裁减值，而观测已饱和；初始预算须满足 z0_J <= "
            f"{z_clip}*Z_REF_J={z_clip * cfg.Z_REF_J:.1f} J"
        )

    # --- t 侧：h_t = max(g_norm(x_t), C·y_t) ----------------------------
    g = g_norm_from_distance(goal_d, cfg)
    h_t = np.asarray(T.safety_step(g, y, cfg.C), dtype=np.float64)
    # --- T 侧：终态只补一次 -------------------------------------------------
    g_T = float(g_norm_from_distance(goal_d_next[-1:], cfg)[0])
    h_T_measured = float(max(g_T, cfg.C * float(y_next[-1])))
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

    # --- 成本目标：尾段累计 + 终端项 (gamma_c = 1，未折扣，与目标定义一致) ---
    suffix = np.cumsum(c[::-1])[::-1]
    if terminal_kind == "failure":
        term_cost = 0.0                     # 失败步单独按 FAIL_PENALTY 覆盖
    elif terminal_kind == "success":
        term_cost = 0.0                     # 无 bootstrap，终端价值为 0
    else:
        term_cost = float(b_c[-1])           # 截断：x_T 的 V_c，采集时存下
    target_cost = suffix + term_cost
    if terminal_kind == "failure":
        target_cost = target_cost.copy()
        target_cost[-1] = cfg.fail_penalty
    adv_cost = target_cost - vc

    # --- reach 目标：实测 h 序列上的 min 型 Bellman 不动点 ---
    target_reach = T.reach_values(h_ext, cfg.gamma_h)[:Tn]
    adv_reach = target_reach - vh

    # --- 停止分段（共享 τ*），policy 用残差一阶差 ---
    stop_index, stop_residual, stop_now = T.select_stop(
        c, z, h_ext, vc_terminal=term_cost, gamma=cfg.gamma_cost)
    total_value = np.maximum(vh, vc - z)
    adv_total = stop_residual - total_value

    return {
        "obj_g_norm": g,
        "obj_h": h_t,
        "obj_h_terminal": np.full(Tn, h_T, dtype=np.float64),
        "obj_z_norm": z,
        "obj_spent": suffix,
        "obj_target_cost": target_cost,
        "obj_adv_cost": adv_cost,
        "obj_target_reach": target_reach,
        "obj_adv_reach": adv_reach,
        "obj_total_value": total_value,
        "obj_target_total": stop_residual,
        "obj_adv_total": adv_total,
        "obj_stop_index": stop_index.astype(np.int32),
        "obj_stop_residual": stop_residual,
        "obj_stop_now": np.asarray(stop_now, dtype=bool),
        "obj_feasible": np.asarray(T.is_feasible(stop_residual, cfg.beta), dtype=bool),
        "obj_cost_return": suffix + term_cost,
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
