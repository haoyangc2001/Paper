#!/usr/bin/env python3
"""把训练目标里各项的**数值尺度**算清楚（只用冻结常数与实测数字，不训练）。

问题的形式化：策略的目标是最小化  g = max(h, -z) + w·c（λ-return 后做优势）。
我们把「原地不动」「在预算内到达」「到达但超支 5%」「到达但超支 15%」四种轨迹
在同一个 l1_empty-s2001 场景下的目标值算出来，看哪个更小。

输入常数全部来自冻结契约：
  value_objective_contract_v1.json : L_REF_m=15, r_goal=0.5, G_INSIDE=-0.001,
                                     C=1.0, FAIL_PENALTY=1.0, Z_REF_J=550
  power_model_v1.json (NAV-059 解析): P(v)=3.0284+4.7226v+31.5420v^2 [W]
  train_map_reference_energy.json : l1_empty-s2001 路线 13.95 m, E_ref=362.6 J
"""
import json
import os

L_REF = 15.0
R_GOAL = 0.5
G_INSIDE = -0.001
Z_REF = 550.0
FAIL_PENALTY = 1.0
C_SAFETY = 1.0
PA, PB, PC = 3.0284321184264877, 4.722608162350005, 31.5420191257139
ROUTE_M = 13.95
E_REF_J = 362.6
T_DEADLINE = 30.0


def power(v):
    return PA + PB * v + PC * v * v


def h_at(dist_m):
    return G_INSIDE if dist_m <= R_GOAL else (dist_m - R_GOAL) / L_REF


def total(dist_end_m, energy_J, w, mode):
    """mode: 'always' = 全程计成本；'overspend' = 只有超出 E_ref 的部分计成本。"""
    over = max(0.0, energy_J - E_REF_J)
    bill = energy_J if mode == "always" else over
    return h_at(dist_end_m) + w * bill / Z_REF


def main():
    v_dead = ROUTE_M / T_DEADLINE
    print("场景 l1_empty-s2001：路线 %.2f m，时限 %.0f s，截止速度 %.3f m/s" % (ROUTE_M, T_DEADLINE, v_dead))
    print("冻结常数：L_REF=%.0f m，G_INSIDE=%.3f，Z_REF=%.0f J，C=%.1f，FAIL_PENALTY=%.1f"
          % (L_REF, G_INSIDE, Z_REF, C_SAFETY, FAIL_PENALTY))
    print("功率模型 P(v)=%.4f+%.4f v+%.4f v^2，P(%.3f)=%.2f W" % (PA, PB, PC, v_dead, power(v_dead)))
    print()

    print("每米的两种信号（以 0.465 m/s 直行为例）")
    c_per_step = power(v_dead) * 0.1 / Z_REF
    d_per_step = v_dead * 0.1 / L_REF
    print("  每 0.1 s：距离项变化 %.5f，归一化成本 %.5f" % (d_per_step, c_per_step))
    print("  每 1 m：距离项 %.4f；w=50 时成本项 %.4f → 成本是距离的 %.1f 倍"
          % (1 / L_REF, 50 * c_per_step / (v_dead * 0.1), 50 * c_per_step / (v_dead * 0.1) * L_REF))
    print("  到达的额外收益：G_INSIDE=%.3f（相对离终点 0.5 m 处 h=0），失败罚 %.1f" % (G_INSIDE, FAIL_PENALTY))
    print()

    print("四种轨迹的目标值（越小越好）")
    cases = [
        ("原地不动（0 m，0 J）", ROUTE_M, 0.0),
        ("在预算内到达（13.95 m，362.6 J）", G_INSIDE, E_REF_J),
        ("到达但超支 5%（381 J）", G_INSIDE, E_REF_J * 1.05),
        ("到达但超支 15%（417 J）", G_INSIDE, E_REF_J * 1.15),
        ("还差 1 m 停下（预算内，322 J）", 1.0, 322.0),
    ]
    for w in (50.0, 5.0, 1.0):
        for mode in ("overspend", "always"):
            print("  w=%.0f，%s 口径：" % (w, mode))
            for name, d, e in cases:
                print("     %-32s %8.3f" % (name, total(d, e, w, mode)))
    print()

    print("到达要优于原地不动，w 的上限：")
    for over_pct in (0.0, 0.05, 0.10, 0.15):
        over_J = E_REF_J * over_pct
        w_max = h_at(ROUTE_M) * Z_REF / over_J if over_J > 0 else float("inf")
        print("  超支 %2.0f%%（%.1f J）：w < %.1f" % (over_pct * 100, over_J, w_max))
    print("  全程计成本时：w < %.2f" % (h_at(ROUTE_M) * Z_REF / E_REF_J))


if __name__ == "__main__":
    main()
