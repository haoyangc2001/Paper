"""200 Hz 关节机械功计量与子步相位记录（HIRA-019）。

契约口径与 v2 完全一致（D-ENERGY-001 / :mod:`camra_calibration.energy`）：

    J = Σ_k Σ_j max(τ_j[k] · q̇_j[k], 0) · dt        k = 1..N,  dt = 0.005 s

三件事必须用证据说清，本模块把它们做成结构性保证：

**一、τ 用"实际施加"的那一个**
    ``robot.data.applied_torque`` 在 Isaac Lab 中等于 ``actuator.applied_effort``，
    而 ``write_data_to_sim()`` 正是把同一个 ``applied_effort`` 交给
    ``root_physx_view.set_dof_actuation_forces(...)``。PD 请求力矩（未限幅）和
    ``computed_torque`` 都**不是**实际施加值；本模块只接受调用方传入的
    ``tau_applied_lab``，并把另外两者作为对照留档，绝不参与累加。

**二、相位**
    力矩在 ``sim.step()`` 之前写入，所以步后读到的 ``applied_torque`` 属于
    **刚刚执行完的那个子步**；速度取 ``robot.update(dt)`` 之后的 ``joint_vel``，
    即该子步**结束时刻**的速度。这与 v2 记录器 ``sim/recorder.py`` 的顺序
    （先取 ``applied_torque`` → ``robot.update(dt)`` → 取 ``joint_vel``）逐字对应。
    为把这个选择变成可检验的事实，本模块同时留档 ``qdot_pre``，
    离线可以分别按 pre/post 两种口徑复算并比较差异。

**三、只读**
    ``record()`` 不做任何 ``set_*`` / ``write_*`` / ``sim.step()``；只读张量、
    只写自己的缓冲区。开/关记录器的轨迹必须逐位一致（由测试跨进程对照）。

累加用 float64 在设备端完成；原始 τ/q̇ 以 float32 留档（与 v2 记录器同精度），
离线复算直接用这些 float32 值升到 float64，因此两边输入完全相同，
差异只可能来自求和实现本身。
"""

from __future__ import annotations

import math

import torch

try:  # D-ENERGY-001 的唯一实现；不在这里复制一份常量
    from camra_calibration.energy import DEFAULT_DT, DEFAULT_JOINTS
except ModuleNotFoundError as exc:  # pragma: no cover - 环境缺失时给出可读报错
    raise ModuleNotFoundError(
        "需要把 CaMRA/calibration/src 放进 PYTHONPATH（能量契约的唯一实现）"
    ) from exc

N_JOINTS = DEFAULT_JOINTS


class EnergyRecorderError(RuntimeError):
    """窗口未开始、容量溢出或输入不合法。"""


class HiRAEnergyRecorder:
    """逐物理子步记录 τ / q̇ / 关节加速度，并用 float64 累计正功。

    只观察。对外接口刻意与 v2 的 ``TrialRecorder`` 同形（begin_warmup /
    begin_window / finish），后续采集器可以直接替换。
    """

    def __init__(
        self,
        order,
        capacity_steps: int,
        dt: float = DEFAULT_DT,
        num_envs: int = 1,
        n_joints: int = N_JOINTS,
        device: str | torch.device = "cuda:0",
        label: str = "",
        expected_steps: int | None = None,
    ):
        if capacity_steps <= 0:
            raise EnergyRecorderError(f"capacity_steps 必须 > 0，得到 {capacity_steps}")
        if dt <= 0:
            raise EnergyRecorderError(f"dt 必须 > 0，得到 {dt}")
        self.order = order
        self.dt = float(dt)
        self.num_envs = int(num_envs)
        self.n_joints = int(n_joints)
        self.device = torch.device(device)
        self.label = label
        self.capacity = int(capacity_steps)
        self.expected_steps = expected_steps

        T, N, J = self.capacity, self.num_envs, self.n_joints
        f32, f64 = torch.float32, torch.float64
        # 原始留档（float32，与 v2 记录器一致）
        self.tau_applied = torch.zeros((T, N, J), dtype=f32, device=self.device)
        self.tau_written = torch.zeros((T, N, J), dtype=f32, device=self.device)
        self.tau_request = torch.zeros((T, N, J), dtype=f32, device=self.device)
        self.tau_computed = torch.zeros((T, N, J), dtype=f32, device=self.device)
        self.qdot_pre = torch.zeros((T, N, J), dtype=f32, device=self.device)
        self.qdot_post = torch.zeros((T, N, J), dtype=f32, device=self.device)
        self.joint_acc = torch.zeros((T, N, J), dtype=f32, device=self.device)
        self.root_pos = torch.zeros((T, N, 3), dtype=f32, device=self.device)
        # 累加（float64）
        self.energy_step = torch.zeros((T, N), dtype=f64, device=self.device)
        self.total_j = torch.zeros(N, dtype=f64, device=self.device)
        self.per_joint_j = torch.zeros((N, J), dtype=f64, device=self.device)

        self.k = 0
        self.warmup_steps = 0
        self._in_window = False
        self._finished = False
        self._saturated_substeps = 0

    # ------------------------------------------------------------------ #
    # 窗口生命周期
    def begin_warmup(self) -> None:
        """进入 warmup：只计数，不进缓冲区（warmup 的能量绝不允许混进窗口）。"""
        self._in_window = False
        self.warmup_steps = 0

    def begin_window(self) -> None:
        if self.k:
            raise EnergyRecorderError("窗口已有数据，不能重复 begin_window")
        self._in_window = True

    def _check_window(self) -> None:
        if not self._in_window:
            raise EnergyRecorderError("窗口未开始（warmup 或复位步不得计入能量）")
        if self._finished:
            raise EnergyRecorderError("记录器已 finish，不能再累加")
        if self.k >= self.capacity:
            raise EnergyRecorderError(
                f"物理步数超过缓冲 {self.capacity}（窗口配置与容量不一致）"
            )

    # ------------------------------------------------------------------ #
    # 每子步
    def record(
        self,
        *,
        tau_applied_lab: torch.Tensor,
        tau_written_source: torch.Tensor,
        qdot_pre_source: torch.Tensor,
        qdot_post_lab: torch.Tensor,
        tau_request_source: torch.Tensor | None = None,
        tau_computed_lab: torch.Tensor | None = None,
        joint_acc_lab: torch.Tensor | None = None,
        root_pos_lab: torch.Tensor | None = None,
        saturated_joints: int = 0,
    ) -> torch.Tensor:
        """记录一个物理子步；返回该子步各 env 的能量（J）。

        参数里的 ``_lab`` / ``_source`` 后缀是**强制**的：关节顺序混错是这类
        验证里最容易发生、又最容易被"往返一致"掩盖的错误（HIRA-013 踩过）。
        """
        if not self._in_window:
            self.warmup_steps += 1
            return torch.zeros(self.num_envs, dtype=torch.float64, device=self.device)
        self._check_window()

        dtype, device = torch.float32, self.device
        applied_lab = tau_applied_lab.to(device, dtype).reshape(self.num_envs, self.n_joints)
        written_src = tau_written_source.to(device, dtype).reshape(self.num_envs, self.n_joints)
        pre_src = qdot_pre_source.to(device, dtype).reshape(self.num_envs, self.n_joints)
        post_lab = qdot_post_lab.to(device, dtype).reshape(self.num_envs, self.n_joints)

        applied_src = self.order.target_to_source(applied_lab)
        post_src = self.order.target_to_source(post_lab)

        # 契约：逐关节截断再求和，且乘法在 float64 里做（float32 乘积会引入 1e-8 级误差）
        power = torch.clamp(applied_src.to(torch.float64) * post_src.to(torch.float64), min=0.0)
        step_j = power.sum(dim=-1) * self.dt

        self.tau_applied[self.k].copy_(applied_src)
        self.tau_written[self.k].copy_(written_src)
        self.qdot_pre[self.k].copy_(pre_src)
        self.qdot_post[self.k].copy_(post_src)
        if tau_request_source is not None:
            # 未限幅的 PD 请求力矩：只作对照留档，绝不参与累加
            self.tau_request[self.k].copy_(
                tau_request_source.to(device, dtype).reshape(self.num_envs, self.n_joints)
            )
        if tau_computed_lab is not None:
            self.tau_computed[self.k].copy_(
                self.order.target_to_source(
                    tau_computed_lab.to(device, dtype).reshape(self.num_envs, self.n_joints)
                )
            )
        if joint_acc_lab is not None:
            self.joint_acc[self.k].copy_(
                self.order.target_to_source(
                    joint_acc_lab.to(device, dtype).reshape(self.num_envs, self.n_joints)
                )
            )
        if root_pos_lab is not None:
            self.root_pos[self.k].copy_(root_pos_lab.to(device, dtype).reshape(self.num_envs, 3))

        self.energy_step[self.k] = step_j
        self.total_j += step_j
        self.per_joint_j += power * self.dt
        self._saturated_substeps += int(saturated_joints)
        self.k += 1
        return step_j

    # ------------------------------------------------------------------ #
    # 结果
    def finish(self) -> dict:
        self._finished = True
        steps = self.k
        total = self.total_j.detach().cpu().to(torch.float64).numpy()
        per_step = self.energy_step[:steps].detach().cpu().to(torch.float64).numpy()
        per_joint = self.per_joint_j.detach().cpu().to(torch.float64).numpy()
        return {
            "label": self.label,
            "dt_s": self.dt,
            "n_joints": self.n_joints,
            "num_envs": self.num_envs,
            "steps": steps,
            "warmup_steps": self.warmup_steps,
            "elapsed_s": steps * self.dt,
            "expected_steps": self.expected_steps,
            "partial": bool(self.expected_steps is not None and steps != self.expected_steps),
            "total_j": total,
            "energy_step_j": per_step,
            "per_joint_j": per_joint,
            "saturated_substeps": self._saturated_substeps,
            "tau_applied_source": self.tau_applied[:steps].detach().cpu().numpy(),
            "tau_written_source": self.tau_written[:steps].detach().cpu().numpy(),
            "tau_request_source": self.tau_request[:steps].detach().cpu().numpy(),
            "tau_computed_source": self.tau_computed[:steps].detach().cpu().numpy(),
            "qdot_pre_source": self.qdot_pre[:steps].detach().cpu().numpy(),
            "qdot_post_source": self.qdot_post[:steps].detach().cpu().numpy(),
            "joint_acc_source": self.joint_acc[:steps].detach().cpu().numpy(),
            "root_pos_w": self.root_pos[:steps].detach().cpu().numpy(),
            "unit": "J",
            "metric_id": "D-ENERGY-001",
        }

    def problems(self, result: dict) -> list[str]:
        """自检：步数、非负、逐步/逐关节求和与总量一致。"""
        out: list[str] = []
        total = result["total_j"]
        per_step = result["energy_step_j"]
        per_joint = result["per_joint_j"]
        if result["steps"] != len(per_step):
            out.append(f"逐步记录长度 {len(per_step)} ≠ 步数 {result['steps']}")
        for name, arr in (("total_j", total), ("energy_step_j", per_step), ("per_joint_j", per_joint)):
            if not all(math.isfinite(float(v)) for v in [float(x) for x in arr.reshape(-1)]):
                out.append(f"{name} 含非有限值")
        if per_step.size and float(per_step.min()) < 0:
            out.append("出现负的单步能量（逐关节截断应保证非负）")
        scale = max(1.0, float(abs(total).max()) if total.size else 1.0)
        for env in range(self.num_envs):
            if abs(float(per_step[:, env].sum()) - float(total[env])) > 1e-12 * scale:
                out.append(f"env{env}: 逐步求和不等于总量")
            if abs(float(per_joint[env].sum()) - float(total[env])) > 1e-12 * scale:
                out.append(f"env{env}: 逐关节求和不等于总量")
        return out


def project_to_envs(result: dict, env_id: int) -> dict:
    """把多 env 的留档压成单 env 视图，供离线复算使用。"""
    squeezed = dict(result)
    for key in (
        "tau_applied_source",
        "tau_written_source",
        "tau_request_source",
        "tau_computed_source",
        "qdot_pre_source",
        "qdot_post_source",
        "joint_acc_source",
    ):
        squeezed[key] = result[key][:, env_id, :]
    squeezed["root_pos_w"] = result["root_pos_w"][:, env_id, :]
    squeezed["total_j"] = float(result["total_j"][env_id])
    squeezed["energy_step_j"] = result["energy_step_j"][:, env_id]
    squeezed["per_joint_j"] = result["per_joint_j"][env_id]
    return squeezed
