"""200 Hz 真实机械功计量与导航轨迹记录（NAV-027）。

能量契约与冻结口径逐字一致（D-ENERGY-001 / :mod:`camra_calibration.energy`，
参照实现 :mod:`locomotion.adapters.hira_go2.energy_recorder`）：

    J = Σ_k Σ_j max(τ_j[k] · q̇_j[k], 0) · dt        k = 1..N,  dt = 0.005 s

本模块在 HiRA 版语义之上新增导航所需的三件事：

**一、活跃 env 掩码与导航终止**
    批量导航中各 env 的终止时刻不同（fall / collision / goal_reached /
    timeout / 批内异步失败）。:meth:`record` 接受 ``active_mask``：被掩掉的
    env 该子步**留档但不累加**（``valid=False``）；:meth:`mark_event` 冻结
    被终止 env 的后续所有子步。事件步号 ``event_step_k`` 一并留档。

**二、完整性标记**
    每个 env 在 :meth:`finish` 时得到唯一 status：

    - ``complete``            无事件且活跃步数 == expected_steps
    - ``timeout``             正常时限结束（显式 mark_event("timeout")）
    - ``early_termination``   提前终止（显式 mark_event）
    - ``incomplete_async``    无事件但步数不足（批内异步失败）
    - ``recorder_fault``      自检发现非有限值 / 求和断裂 / 跨 reset 写入

**三、不跨 reset 累加**
    结构性保证三层：warmup 期只计数不进缓冲；``begin_window`` 拒绝复用
    已含数据的窗口；:meth:`observe_reset` 在窗口活动期捕获对已累加 env
    的复位写入并立即标记 fault、冻结累加。上层导航循环不调用 gym 自动
    reset，复位必须走 ``full_reset`` + ``observe_reset``。

计量口径不变的三条硬约束（与参照实现一致）：

1. τ 用实际施加值（``applied_torque`` == PhysX 收到的 ``applied_effort``）；
   PD 请求力矩 / computed torque 只作对照留档，绝不参与累加。
2. 相位：``sim.step()`` 之前写入的力矩属于刚刚执行的子步；``q̇`` 取该子步
   结束时刻的 ``joint_vel``，``q̇_pre``（上子步末速度）留档供离线双口径复算。
3. 逐关节 ``max(τ·q̇, 0)`` 截断**之后**再求和，乘法在 float64 中做；
   原始 τ/q̇ 以 float32 留档，离线复算升 float64，输入与在线完全相同。

``record`` 只读张量、只写自己的缓冲区，不做任何 ``set_*`` / ``write_*`` /
``sim.step()``；开/关记录器不改变仿真轨迹。
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch

try:  # D-ENERGY-001 的唯一实现；不在这里复制一份常量
    from camra_calibration.energy import DEFAULT_DT, DEFAULT_JOINTS
except ModuleNotFoundError as exc:  # pragma: no cover - 环境缺失时给出可读报错
    raise ModuleNotFoundError(
        "需要把 CaMRA/calibration/src 放进 PYTHONPATH（能量契约的唯一实现）"
    ) from exc

N_JOINTS = DEFAULT_JOINTS

STATUS_COMPLETE = "complete"
STATUS_TIMEOUT = "timeout"
STATUS_EARLY = "early_termination"
STATUS_ASYNC = "incomplete_async"
STATUS_FAULT = "recorder_fault"


class RecorderError(RuntimeError):
    """窗口未开始、容量溢出、终止后写入或输入不合法。"""


class NavEnergyRecorder:
    """逐物理子步记录 τ / q̇ / 位姿，按活跃掩码累计正机械功。

    对外接口与 ``TrialRecorder`` 同形（begin_warmup / begin_window /
    record / finish），另加导航语义（active_mask / mark_event /
    observe_reset）。
    """

    def __init__(
        self,
        capacity_steps: int,
        num_envs: int,
        dt: float = DEFAULT_DT,
        n_joints: int = N_JOINTS,
        device: str | torch.device = "cuda:0",
        label: str = "",
        expected_steps: int | None = None,
    ):
        if capacity_steps <= 0:
            raise RecorderError(f"capacity_steps 必须 > 0，得到 {capacity_steps}")
        if num_envs <= 0:
            raise RecorderError(f"num_envs 必须 > 0，得到 {num_envs}")
        if dt <= 0:
            raise RecorderError(f"dt 必须 > 0，得到 {dt}")
        self.dt = float(dt)
        self.num_envs = int(num_envs)
        self.n_joints = int(n_joints)
        self.device = torch.device(device)
        self.label = label
        self.capacity = int(capacity_steps)
        self.expected_steps = expected_steps

        T, N, J = self.capacity, self.num_envs, self.n_joints
        f32, f64, b8 = torch.float32, torch.float64, torch.bool
        # 原始留档（float32，与 v2 记录器一致）
        self.tau_applied = torch.zeros((T, N, J), dtype=f32, device=self.device)
        self.tau_request = torch.zeros((T, N, J), dtype=f32, device=self.device)
        self.qdot_pre = torch.zeros((T, N, J), dtype=f32, device=self.device)
        self.qdot_post = torch.zeros((T, N, J), dtype=f32, device=self.device)
        self.root_pos = torch.zeros((T, N, 3), dtype=f32, device=self.device)
        self.valid = torch.ones((T, N), dtype=b8, device=self.device)
        # 累加（float64）
        self.energy_step = torch.zeros((T, N), dtype=f64, device=self.device)
        self.total_j = torch.zeros(N, dtype=f64, device=self.device)
        self.per_joint_j = torch.zeros((N, J), dtype=f64, device=self.device)

        self.k = 0                    # 全局子步号（含无效子步，保证相位可对账）
        self.warmup_steps = 0
        self.n_record_calls = 0       # record() 调用总数（防漏子步的对账口径）
        self.active = torch.ones(N, dtype=b8, device=self.device)
        self.events: list[list[dict]] = [[] for _ in range(N)]
        self._env_active_steps = torch.zeros(N, dtype=torch.int64, device=self.device)
        self._in_window = False
        self._finished = False
        self._fault: list[str] = []

    # ------------------------------------------------------------------ #
    # 窗口生命周期
    def begin_warmup(self) -> None:
        """进入 warmup：只计数，不进缓冲区（warmup 能量绝不混进窗口）。"""
        self._in_window = False
        self.warmup_steps = 0

    def begin_window(self) -> None:
        if self.k:
            # 只拒绝"窗口缓冲已有数据"；warmup 期的 record 调用只计数不进缓冲
            raise RecorderError("窗口已有数据，不能复用；跨 reset 必须换新记录器实例")
        self._in_window = True

    def _check_window(self) -> None:
        if not self._in_window:
            raise RecorderError("窗口未开始（warmup 或复位步不得计入能量）")
        if self._finished:
            raise RecorderError("记录器已 finish，不能再累加")
        if self.k >= self.capacity:
            raise RecorderError(
                f"物理步数超过缓冲 {self.capacity}（窗口配置与容量不一致）"
            )

    # ------------------------------------------------------------------ #
    # 导航终止与复位防护
    def mark_event(self, env_ids: Iterable[int], kind: str, detail: str = "") -> None:
        """登记导航事件（提前终止 / 正常时限），冻结这些 env 的后续子步。

        ``kind`` 约定 ``early_termination:<event>`` 或 ``timeout``；
        其他取值按 unknown 事件留档并在 status 推导时视为提前终止。
        """
        for eid in env_ids:
            eid = int(eid)
            if not (0 <= eid < self.num_envs):
                raise RecorderError(f"env id 越界: {eid}")
            if self._finished:
                raise RecorderError("记录器已 finish，不能再登记事件")
            self.events[eid].append(
                {"kind": kind, "step_k": self.k, "detail": detail,
                 "active_steps_at_event": int(self._env_active_steps[eid])}
            )
            self.active[eid] = False

    def observe_reset(self, env_ids: Iterable[int]) -> None:
        """上层 ``full_reset`` 必须调用的防护钩子。

        窗口活动期对已累加过能量的 env 复位 = 跨 reset 写入风险：
        立即记 fault 并冻结该 env；warmup 期或空窗口是正常操作，放行。
        """
        for eid in env_ids:
            eid = int(eid)
            if self._in_window and not self._finished and int(self._env_active_steps[eid]) > 0:
                self._fault.append(
                    f"cross_reset_write: env{eid} 在窗口内被复位"
                    f"（已累加 {int(self._env_active_steps[eid])} 个活跃子步）"
                )
                self.active[eid] = False

    # ------------------------------------------------------------------ #
    # 每子步
    def record(
        self,
        *,
        tau_applied_lab: torch.Tensor,
        qdot_post_lab: torch.Tensor,
        qdot_pre_lab: torch.Tensor | None = None,
        tau_request_lab: torch.Tensor | None = None,
        root_pos_lab: torch.Tensor | None = None,
        active_mask: Sequence[bool] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """记录一个物理子步；返回该子步各 env 的能量（J，无效 env 为 0）。

        ``_lab`` 后缀强制：输入为 Isaac Lab 关节顺序（经 NAV-024/025 验证与
        v2 契约顺序逐位一致）。``active_mask`` 为 None 时全体活跃。
        """
        self.n_record_calls += 1
        if not self._in_window:
            self.warmup_steps += 1
            return torch.zeros(self.num_envs, dtype=torch.float64, device=self.device)
        self._check_window()

        dtype, device = torch.float32, self.device
        applied = tau_applied_lab.to(device, dtype).reshape(self.num_envs, self.n_joints)
        post = qdot_post_lab.to(device, dtype).reshape(self.num_envs, self.n_joints)
        if qdot_pre_lab is not None:
            self.qdot_pre[self.k].copy_(
                qdot_pre_lab.to(device, dtype).reshape(self.num_envs, self.n_joints)
            )
        if tau_request_lab is not None:
            # 未限幅的 PD 请求力矩：只作对照留档，绝不参与累加
            self.tau_request[self.k].copy_(
                tau_request_lab.to(device, dtype).reshape(self.num_envs, self.n_joints)
            )
        if root_pos_lab is not None:
            self.root_pos[self.k].copy_(
                root_pos_lab.to(device, dtype).reshape(self.num_envs, 3)
            )

        if active_mask is None:
            step_active = self.active.clone()
        else:
            step_active = torch.as_tensor(active_mask, dtype=torch.bool, device=device) & self.active

        # 契约：逐关节截断再求和，乘法在 float64（float32 乘积会引入 1e-8 级误差）
        power = torch.clamp(applied.to(torch.float64) * post.to(torch.float64), min=0.0)
        step_j = (power.sum(dim=-1) * self.dt) * step_active.to(torch.float64)

        self.tau_applied[self.k].copy_(applied)
        self.qdot_post[self.k].copy_(post)
        self.valid[self.k].copy_(step_active)
        self.energy_step[self.k] = step_j
        self.total_j += step_j
        self.per_joint_j += power * (self.dt * step_active.unsqueeze(-1).to(torch.float64))
        self._env_active_steps += step_active.to(torch.int64)
        self.k += 1
        return step_j

    # ------------------------------------------------------------------ #
    # 结果
    def env_status(self, eid: int) -> str:
        """单 env 完整性标记推导。"""
        evs = self.events[eid]
        steps = int(self._env_active_steps[eid])
        expected = self.expected_steps
        kinds = {e["kind"] for e in evs}
        early = any(k.startswith("early_termination") for k in kinds)
        if early:
            return STATUS_EARLY
        if "timeout" in kinds:
            return STATUS_TIMEOUT
        if expected is not None and steps == expected:
            return STATUS_COMPLETE
        if expected is not None and steps < expected:
            return STATUS_ASYNC
        return STATUS_COMPLETE if expected is None else STATUS_ASYNC

    def finish(self) -> dict:
        self._finished = True
        steps = self.k
        status = [self.env_status(i) for i in range(self.num_envs)]
        total = self.total_j.detach().cpu().to(torch.float64).numpy()
        per_step = self.energy_step[:steps].detach().cpu().to(torch.float64).numpy()
        per_joint = self.per_joint_j.detach().cpu().to(torch.float64).numpy()
        valid = self.valid[:steps].detach().cpu().numpy()
        result = {
            "label": self.label,
            "dt_s": self.dt,
            "n_joints": self.n_joints,
            "num_envs": self.num_envs,
            "steps": steps,
            "warmup_steps": self.warmup_steps,
            "n_record_calls": self.n_record_calls,
            "elapsed_s": steps * self.dt,
            "expected_steps": self.expected_steps,
            "env_active_steps": [int(v) for v in self._env_active_steps.detach().cpu().numpy()],
            "env_status": status,
            "events": self.events,
            "faults": list(self._fault),
            "total_j": total,
            "energy_step_j": per_step,
            "per_joint_j": per_joint,
            "valid_mask": valid,
            "tau_applied": self.tau_applied[:steps].detach().cpu().numpy(),
            "tau_request": self.tau_request[:steps].detach().cpu().numpy(),
            "qdot_pre": self.qdot_pre[:steps].detach().cpu().numpy(),
            "qdot_post": self.qdot_post[:steps].detach().cpu().numpy(),
            "root_pos_w": self.root_pos[:steps].detach().cpu().numpy(),
            "unit": "J",
            "metric_id": "D-ENERGY-001",
        }
        # 数据性故障（NaN/求和断裂等）在 finish 时自检：非空则整体标 fault
        self_problems = self._compute_problems(result)
        result["problems"] = self_problems
        if self_problems:
            result["env_status"] = [STATUS_FAULT for _ in status]
        return result

    def _compute_problems(self, result: dict) -> list[str]:
        """自检：步数、非负、逐步/逐关节求和与总量一致、fault 登记。"""
        out: list[str] = list(self._fault)
        total = result["total_j"]
        per_step = result["energy_step_j"]
        per_joint = result["per_joint_j"]
        valid = result["valid_mask"]
        if result["steps"] != len(per_step):
            out.append(f"逐步记录长度 {len(per_step)} ≠ 步数 {result['steps']}")
        for name, arr in (
            ("total_j", total),
            ("energy_step_j", per_step),
            ("per_joint_j", per_joint),
            ("tau_applied", result["tau_applied"]),
            ("qdot_post", result["qdot_post"]),
        ):
            flat = [float(x) for x in arr.reshape(-1)]
            if not all(math.isfinite(v) for v in flat):
                out.append(f"{name} 含非有限值")
        if per_step.size and float(per_step[valid].min() if valid.any() else 0.0) < 0:
            out.append("出现负的单步能量（逐关节截断应保证非负）")
        # 逐步求和与总量对账：无效子步能量恒 0，全部子步求和 == 总量
        scale = max(1.0, float(abs(total).max()) if total.size else 1.0)
        for env in range(self.num_envs):
            if abs(float(per_step[:, env].sum()) - float(total[env])) > 1e-12 * scale:
                out.append(f"env{env}: 逐步求和不等于总量")
            if abs(float(per_joint[env].sum()) - float(total[env])) > 1e-12 * scale:
                out.append(f"env{env}: 逐关节求和不等于总量")
        return out

    def problems(self, result: dict) -> list[str]:
        """公开自检接口（幂等；finish 时已自动运行一次并写入 result["problems"]）。"""
        return self._compute_problems(result)


def offline_recompute(result: dict, env_id: int) -> dict:
    """从留档 τ / q̇ 独立复算单 env 能量（camra_calibration.energy 的独立实现）。

    只用 ``valid`` 子步；返回总焦耳、活跃子步数与在线值的绝对/相对差。
    """
    from camra_calibration.energy import integrate_series

    tau = result["tau_applied"][:, env_id, :][result["valid_mask"][:, env_id]]
    qd = result["qdot_post"][:, env_id, :][result["valid_mask"][:, env_id]]
    recomputed = integrate_series(tau, qd, dt=result["dt_s"])
    online = float(result["total_j"][env_id])
    abs_diff = abs(recomputed - online)
    rel = abs_diff / max(abs(online), 1e-12)
    n_valid = int(result["valid_mask"][:, env_id].sum())
    return {
        "env": env_id,
        "online_total_j": online,
        "offline_recomputed_j": recomputed,
        "abs_diff_j": abs_diff,
        "rel_diff": rel,
        "n_valid_substeps": n_valid,
        "active_steps_reported": result["env_active_steps"][env_id],
        "steps_match": n_valid == result["env_active_steps"][env_id],
        "dt_s": result["dt_s"],
        "elapsed_s": result["steps"] * result["dt_s"],
    }
