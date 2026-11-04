"""按 env 下发不同命令程序（CAL-015）。

仓库的外部命令桥接里 ``CommandProgram.at(t)`` 收到的是**逐 env 的时间向量**，
并要求返回 ``(num_envs, 3)``——因此一个 program 对象就可以给不同 env 下发不同
命令，只要它持有每 env 的命令表。

``TableProgram`` 用 (N, T, 3) 的 10 Hz 命令表 + 表外扩，按 ``idx = floor(t/dt)``
gather，避免每步做 python 循环。``warmup_s`` 之前一律零命令。
"""
from __future__ import annotations

from typing import Sequence

import torch


class TableProgram:
    """每 env 一张 10 Hz 命令表的零阶保持程序。"""

    name = "calibration_table"

    def __init__(self, tables: Sequence[Sequence[Sequence[float]]], dt: float = 0.1,
                 warmup_s: float = 0.0, device: str | torch.device = "cpu"):
        t = torch.as_tensor(tables, dtype=torch.float32, device=device)
        if t.ndim != 3 or t.shape[-1] != 3:
            raise ValueError(f"tables 形状应为 (N, T, 3)，得到 {tuple(t.shape)}")
        if dt <= 0:
            raise ValueError("dt 必须 > 0")
        self.tables = t
        self.num_envs = int(t.shape[0])
        self.n_points = int(t.shape[1])
        self.dt = float(dt)
        self.warmup_s = float(warmup_s)
        self.device = t.device

    def at(self, t: torch.Tensor) -> torch.Tensor:
        """t: (num_envs,) 秒 → (num_envs, 3)。"""
        tt = torch.as_tensor(t, dtype=torch.float32, device=self.device).flatten()
        if tt.numel() != self.num_envs:
            raise ValueError(f"期望 {self.num_envs} 个时间，得到 {tt.numel()}")
        elapsed = tt - self.warmup_s
        idx = torch.floor(elapsed / self.dt).long()
        idx = idx.clamp(min=0, max=self.n_points - 1)
        out = self.tables[torch.arange(self.num_envs, device=self.device), idx]
        # warmup 期间（elapsed < 0）强制零命令
        zero_mask = (elapsed < 0).unsqueeze(-1)
        return torch.where(zero_mask, torch.zeros_like(out), out)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return self.at(t)

    def describe(self) -> str:
        return f"{self.name}(N={self.num_envs}, T={self.n_points}, dt={self.dt}, warmup={self.warmup_s})"
