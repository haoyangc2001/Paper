"""双评价目标、停止分段与 GAE（NAV-030）。

迁移自只读参考 ``RC-PPO/rl/gae.py``：``calculate_indexs3`` / ``calculate_gae2`` /
``calculate_gae_reach4``。三者在这里按**三层**组织，互不混淆：

  (1) **数学定义**（训练与判定用这一层）：来自 NAV-056 契约
      ``config/value_objective_contract_v1.json``
        · 成本 c_t = (1+α)·P_nom(u_t)·dt / Z_REF，未折扣，γ_c = 1
        · 残差 W(t,τ) = max( reach(τ), spent(t,τ) + [τ=T]·V_c(T) − z_ratio(t) )
        · τ*(t) = argmin_{τ∈[t,T]} W(t,τ)，**取首个最小值**（不是全局只选一次）
        · 可行判据 W <= −beta（beta = 0.0）
        · V_h 的 min 型 Bellman：V_h(t) = min(h_t, (1−γ_h)·h_t + γ_h·V_h(t+1))
  (2) **参考实现行为**（``*_reference`` 后缀，仅用于交叉核对）：逐行转写 JAX 扫描，
      保留其 roll / mask / inf 原样，不做静默修正。与定义若不一致，测试里带数值记录。
  (3) **本轮边界处理**（本文件显式规定，并写进测试）：
        · terminated / truncated 用独立字段；截断用 critic bootstrap，失败给终止代价，
          两者不混用（NAV-030 要求）
        · 停止分段与安全历史各自独立字段：``select_stop`` 只读 reach 序列，
          不读碰撞/越界事件；安全历史由调用方传入的 h 体现
        · τ=T 时 spent 只累计到 T−1 再加一次 V_c(T)，不重复计尾段

符号约定（与契约 sign_conventions 一致）：成本为正；V_c >= 0；V_h <= 0 表示可安全到达；
z_ratio = clip(z_J / Z_REF_J, −1, 1)；beta = 0.0。
"""

from __future__ import annotations

import json
import os
from typing import Any, Sequence

import numpy as np

_THIS = os.path.abspath(__file__)
NAV_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS))))
VALUE_CONTRACT_PATH = os.path.join(NAV_ROOT, "config", "value_objective_contract_v1.json")


def load_objective() -> dict[str, Any]:
    """读取价值目标契约里的数值约定（beta / C / Z_REF_J / gamma）。"""
    with open(VALUE_CONTRACT_PATH, encoding="utf-8") as fh:
        c = json.load(fh)
    return {
        "beta": float(c["quantities"]["beta"]["value"]),
        "C": float(c["quantities"]["C"]["value"]),
        "Z_REF_J": float(c["quantities"]["Z_REF_J"]["value"]),
        "L_REF_m": float(c["quantities"]["g_norm"]["L_REF_m"]),
        "G_INSIDE": float(c["quantities"]["g_norm"]["G_INSIDE"]),
        "gamma_cost": float(c["estimator"]["gamma_cost"]),
        "gae_lambda": float(c["estimator"]["gae_lambda"]),
        "gamma_h_init": float(c["estimator"]["gamma_h_init"]),
        "gamma_h_final": float(c["estimator"]["gamma_h_final"]),
    }


# ======================================================================
# (1) 数学定义
# ======================================================================
def spent_between(cost_norm: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """``spent[t, τ]`` = Σ_{s=t}^{τ−1} γ^{s−t} c_s，返回 (T, T+1)；τ<t 处为 0。"""
    c = np.asarray(cost_norm, dtype=np.float64)
    T = c.shape[0]
    if gamma <= 0:
        raise ValueError("gamma 必须为正")
    disc = c * (gamma ** np.arange(T))           # Σ γ^s c_s 便于前缀和
    cum = np.concatenate([[0.0], np.cumsum(disc)])
    scale = gamma ** (-np.arange(T))             # 再乘 γ^{-t} 得到 Σ γ^{s-t} c_s
    spent = scale[:, None] * (cum[None, :] - cum[:T][:, None])   # (T, T+1)
    tau = np.arange(T + 1)[None, :]
    own = np.arange(T)[:, None]
    return np.where(tau >= own, spent, 0.0)


def stop_residual(cost_norm: np.ndarray, z_ratio: np.ndarray, reach: np.ndarray,
                  vc_terminal: float = 0.0, gamma: float = 1.0) -> np.ndarray:
    """``W(t, τ)`` 完整矩阵 (T, T+1)，τ<t 处为 +inf（不可选）。

    ``reach`` 可以是逐安全代价 h（参考行为）或到达评价 V_h（契约写法），长度 T+1。
    τ=T 时加上 ``vc_terminal``（截断 bootstrap；真失败应另行给终止代价，不在此处混用）。
    """
    c = np.asarray(cost_norm, dtype=np.float64)
    z = np.asarray(z_ratio, dtype=np.float64).reshape(-1)
    r = np.asarray(reach, dtype=np.float64).reshape(-1)
    T = c.shape[0]
    if z.shape[0] != T:
        raise ValueError(f"z_ratio 长度 {z.shape[0]} ≠ T={T}")
    if r.shape[0] != T + 1:
        raise ValueError(f"reach 长度 {r.shape[0]} ≠ T+1={T + 1}")
    spent = spent_between(c, gamma)
    spent[:, T] += float(vc_terminal)            # 只加一次，且只在 τ=T
    W = np.maximum(r[None, :], spent - z[:, None])
    tau = np.arange(T + 1)[None, :]
    own = np.arange(T)[:, None]
    return np.where(tau >= own, W, np.inf)


def select_stop(cost_norm: np.ndarray, z_ratio: np.ndarray, reach: np.ndarray,
                vc_terminal: float = 0.0, gamma: float = 1.0
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐时刻在**尾段**重新取最优停止：返回 (index, W_selected, done)。

    ``index[t] = argmin_τ W(t,τ)``（首个最小值），``done[t] = 1[index[t] == t]``。
    契约明确要求"每步对尾段重新取最优"，不是一次性全局 argmin。
    """
    W = stop_residual(cost_norm, z_ratio, reach, vc_terminal, gamma)
    index = np.argmin(W, axis=1)
    W_sel = W[np.arange(W.shape[0]), index]
    done = (index == np.arange(W.shape[0])).astype(np.int64)
    return index, W_sel, done


def is_feasible(residual: np.ndarray, beta: float | None = None) -> np.ndarray:
    """``W <= −beta`` 判可行（beta=0 时即 W <= 0）。"""
    b = load_objective()["beta"] if beta is None else float(beta)
    return np.asarray(residual, dtype=np.float64) <= -b


def reach_values(h: np.ndarray, gamma_h: float) -> np.ndarray:
    """契约的 min 型 Bellman 递推：V_h(t) = min(h_t, (1−γ)·h_t + γ·V_h(t+1))。

    ``h`` 长度 T+1（h_T 为终端状态的安全代价），返回同长 V_h；γ<1 保证收缩。
    """
    if not (0.0 < gamma_h < 1.0):
        raise ValueError("gamma_h 必须在 (0,1) 内（min 型算子的收缩性要求）")
    hh = np.asarray(h, dtype=np.float64).reshape(-1)
    v = np.empty_like(hh)
    v[-1] = hh[-1]
    for t in range(hh.shape[0] - 2, -1, -1):
        v[t] = min(hh[t], (1.0 - gamma_h) * hh[t] + gamma_h * v[t + 1])
    return v


def cost_gae(cost_norm: np.ndarray, values: np.ndarray, done: np.ndarray,
             last_value: float, gamma: float = 1.0, lam: float = 0.95
             ) -> tuple[np.ndarray, np.ndarray]:
    """成本 critic 的 GAE（对应参考 calculate_gae2）。

    成本为正、γ_c=1 时目标是未折扣累计成本：``target = Σ c + V(x_T)``。
    ``done[t]=1`` 表示第 t 步之后回合结束（不再 bootstrap）。
    """
    c = np.asarray(cost_norm, dtype=np.float64).reshape(-1)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    d = np.asarray(done, dtype=np.float64).reshape(-1)
    T = c.shape[0]
    adv = np.zeros(T, dtype=np.float64)
    gae = 0.0
    for t in range(T - 1, -1, -1):
        nxt = last_value if t == T - 1 else v[t + 1]
        nonterminal = 1.0 - d[t]
        delta = c[t] + gamma * nxt * nonterminal - v[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    return adv, adv + v


def reach_gae(h: np.ndarray, vh: np.ndarray, done: np.ndarray, gamma_h: float,
              lam: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """到达评价的 λ-return（对应参考 calculate_gae_reach4 的数学定义）。

    ``Q(t) = (1−λ)·Σ_{k<n} λ^k V_h(t+k) + λ^n V_h(t+n)``，n 截断到首个 done 之前；
    ``done[t]=1`` 时直接取 Q(t)=V_h(t)（优势 0）。返回 (advantage, Q)。
    """
    hh = np.asarray(h, dtype=np.float64)
    hv = np.asarray(vh, dtype=np.float64)
    dd = np.asarray(done, dtype=int)
    vdp = reach_values(hh, gamma_h)
    T = vdp.shape[0] - 1
    q = vdp.copy()
    for t in range(T):
        if dd[t]:
            q[t] = vdp[t]
            continue
        acc, w = 0.0, 1.0
        k = 0
        while t + k < T and not dd[t + k] and k < T:
            acc += (1.0 - lam) * w * vdp[t + k]
            w *= lam
            k += 1
        acc += w * vdp[min(t + k, T)]
        q[t] = acc
    return (q - vdp)[:T], q[:T]


# ======================================================================
# (2) 参考实现转写（只用于交叉核对，不参与训练）
# ======================================================================
def _ref_gae2(gamma: float, lam: float, reward: np.ndarray, value: np.ndarray,
              done: np.ndarray, last_value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``calculate_gae2`` 的逐行转写（reward 为**成本**，与参考的符号约定一致）。"""
    r = np.asarray(reward, dtype=np.float64).reshape(-1)
    v = np.asarray(value, dtype=np.float64).reshape(-1)
    d = np.asarray(done, dtype=np.float64).reshape(-1)
    lv = float(np.asarray(last_value).reshape(-1)[0])
    T = r.shape[0]
    next_done = np.roll(d, -1)
    next_done[-1] = next_done[-2] if T > 1 else 0.0
    adv = np.zeros(T)
    gae = 0.0
    for t in range(T - 1, -1, -1):
        nxt = lv if t == T - 1 else v[t + 1]
        delta = (r[t] + gamma * nxt * (1.0 - next_done[t])) * (1.0 - d[t]) - v[t]
        gae = delta + gamma * lam * (1.0 - next_done[t]) * (1.0 - d[t]) * gae
        adv[t] = gae
    return adv, adv + v


def _ref_gae_reach4(gamma: float, lam: float, T_hs: np.ndarray, T_Vhs: np.ndarray,
                    done: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``calculate_gae_reach4`` 的逐行转写（保留 roll/mask/归一化原样）。"""
    hs_all = np.asarray(T_hs, dtype=np.float64)
    vh_all = np.asarray(T_Vhs, dtype=np.float64)
    d = np.asarray(done, dtype=int)
    Tp1, nh = hs_all.shape
    T = Tp1 - 1

    next_vh = np.zeros((T + 1, nh))
    next_vh[0] = vh_all[T]
    coeffs = np.zeros((T + 1, nh))
    pre_done = np.zeros(nh, dtype=float)
    q = np.zeros((T, nh))
    ar = np.arange(T + 1)

    for ii in range(T - 1, -1, -1):
        hs = hs_all[ii]
        vh_t1 = vh_all[ii + 1]
        done_row = float(d[ii])
        roll = np.roll(coeffs, 1, axis=0)
        coeffs = (roll * lam * (1.0 - pre_done)
                  + roll * (lam / (1.0 - lam)) * pre_done) * (1.0 - done_row)
        coeffs[0] = 1.0
        mask = (ar < ii + 1).astype(float)[:, None]
        done_processed = 0.0 if done_row == 0.0 else np.inf
        disc = (1.0 - gamma) * hs + gamma * (next_vh + done_processed)
        row = np.minimum(hs, disc) * mask
        normed = coeffs / coeffs.sum(axis=0)
        q[ii] = (row * normed).sum(axis=0)
        row = np.roll(row, 1, axis=0)
        row[0] = vh_t1
        next_vh = row
        pre_done = np.full(nh, done_row)
    return q - vh_all[:-1], q


def _ref_indexs3(gamma: float, reward: np.ndarray, energy: np.ndarray,
                 T_hs: np.ndarray, last_value: np.ndarray,
                 last_value_reach: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``calculate_indexs3`` 的逐行转写（保留 roll / [::-1] / inf 原样）。"""
    r = np.asarray(reward, dtype=np.float64)
    e = np.asarray(energy, dtype=np.float64)
    hs_all = np.asarray(T_hs, dtype=np.float64)
    lv = np.asarray(last_value, dtype=np.float64).reshape(1, -1)
    lvh = np.asarray(last_value_reach, dtype=np.float64).reshape(1, -1)
    Tp1, nh = hs_all.shape
    T = Tp1 - 1

    next_vs = np.full((T, nh), np.inf)
    next_vh = np.full((T, nh), np.inf)
    mask1 = np.full((T, 1), np.inf)
    done = np.zeros((T + 1, nh))
    idx = np.zeros((T, nh), dtype=int)

    for ii in range(T - 1, -1, -1):
        vs_row = mask1 * (r[ii] + gamma * next_vs)
        vs_row[ii] = 0.0
        vh_row = next_vh.copy()
        vh_row[ii] = hs_all[ii]
        v_total = np.maximum(vs_row - e[-ii - 1], vh_row)[::-1]
        v_next = np.maximum(gamma ** ii * lv + v_total[-1] - e[-ii - 1], lvh)
        v_all = np.concatenate([v_total, v_next], axis=0)
        index_1 = np.argmin(v_all, axis=0)
        done[index_1, np.arange(nh)] = 1.0
        mask1 = np.roll(mask1, 1, axis=0)
        mask1[0] = 1.0
        idx[ii] = index_1
        next_vs = vs_row
        next_vh = vh_row
    return idx, done


# ======================================================================
# (3) 便捷封装
# ======================================================================
def step_cost_j(power_w: Sequence[float], dt: float = 0.1, alpha: float = 0.0,
                z_ref_j: float | None = None) -> np.ndarray:
    """``c = (1+α)·P·dt / Z_REF``（契约的 c_model 归一化形式）。"""
    zref = load_objective()["Z_REF_J"] if z_ref_j is None else float(z_ref_j)
    return (1.0 + float(alpha)) * np.asarray(power_w, dtype=np.float64) * float(dt) / zref


def z_ratio_from_budget(z_j: np.ndarray, z_ref_j: float | None = None) -> np.ndarray:
    """``clip(z_J / Z_REF_J, −1, 1)``，与 obs-v2 的 z_ratio 同式同值。"""
    zref = load_objective()["Z_REF_J"] if z_ref_j is None else float(z_ref_j)
    return np.clip(np.asarray(z_j, dtype=np.float64) / zref, -1.0, 1.0)


def safety_step(g_norm: np.ndarray, y: np.ndarray, C: float | None = None) -> np.ndarray:
    """逐安全代价 ``h = max(g_norm, C·y)``；y=1（安全历史被破坏）时 C·y 必为正。"""
    cc = load_objective()["C"] if C is None else float(C)
    return np.maximum(np.asarray(g_norm, dtype=np.float64),
                      cc * np.asarray(y, dtype=np.float64))
