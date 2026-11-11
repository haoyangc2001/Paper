#!/usr/bin/env python
"""算法 G3 门禁的汇总表（NAV-035）。

G3 不是"再跑一次测试"，而是把已经做过的十几道检查按五个维度摆到一张表上，逐条给出
证据文件、文件里的实际结论、以及**这条证据能证明什么、不能证明什么**。所以这个脚本
不重算任何数值——它读证据文件的散列与实际 verdict，只做三件事：

  1. 表里的每一项都必须能在盘上找到证据文件（找不到就是硬错，不是"略过"）；
  2. 证据文件的实际 verdict 与表里声明的期望不一致时，标 `verdict_mismatch` 并要求解释；
  3. 输出 blocked / green 的判定，以及**阻塞项需要谁来裁定**。

刻意不做的事：不把"网络能跑"当放行、不把 loss 下降当学会、不用"再跑几轮看看"代替判定。

Run::

    <frozen python> CaMRA/navigation/scripts/build_gate_g3.py \
        --json CaMRA/navigation/reports/gate_G3.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPORTS = os.path.join(NAV_ROOT, "reports")

DIMENSIONS = ("数值目标", "梯度方向", "轨迹边界", "可枚举对照", "短训练证据", "双critic与minimization")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


#: (id, 维度, 检查项, 证据文件, 期望 verdict, 我给的灯, 这条能证明什么 / 不能证明什么)
TABLE: tuple[tuple[str, str, str, str, str, str, str], ...] = (
    # ---------------- 数值目标
    ("G3-01", "数值目标", "成本模型与预算扣减口径（(1+α)·P·dt、按 applied 扣减）",
     "cost_budget_check.json", "PASS", "green",
     "证明：成本在 20 项检查里与冻结功率模型逐位一致，α 只在 full 变体里生效。"
     "不证明：真实执行功与模型功的差（那是 energy_meter 的事）"),
    ("G3-02", "数值目标", "价值目标、V_c/V_h 尺度、共同停止规则对齐契约",
     "value_objective_alignment.json", "PASS", "green",
     "证明：W=max(V_h, V_c−z)、共享 τ*、W<=−β 判据与契约逐条对齐，含分开取时刻的反例。"
     "不证明：这些量在真实数据上的分布是否合理"),
    ("G3-03", "数值目标", "参考能量与预算量级（54 张图，与策略无关）",
     "budget_fairness_check.json", "PASS", "green",
     "证明：参考能量 L1 362.5 / L2 462.5 / L3 367.7 J 可复算（与策略无关）。"
     "NAV-035 用它重定了归一化尺度 Z_REF_J：225.6069 → 550 J，见 G3-26 与 contract_bump"),
    ("G3-04", "数值目标", "指令域（applied 命令的实测极值裁定）",
     "command_domain_resolution.json", "NO_VERDICT", "green",
     "证明：五个独立来源一致，vx∈[-1,1]、vy∈[-0.4,0.4]、wz∈[-1,1]；margins_v1 里 ±2.0 的文本是过时描述。"
     "不证明：域内每一点的执行质量（那是死区与低速的事）"),

    # ---------------- 梯度方向
    ("G3-05", "梯度方向", "成本最小化更新的符号（max 而非 min、熵项负号、直接下降）",
     "minimization_update_check.json", "PASS", "green",
     "证明：12 项检查，梯度方向由中心差分独立验证（autograd 与差分一致 1e-9），"
     "裁剪域内/域外、零优势、熵方向、价值损失手算全过。"
     "不证明：这个方向在长训里是否稳定（round G4 已给出裁剪常态化的证据）"),

    # ---------------- 轨迹边界
    ("G3-06", "轨迹边界", "名义环境数学（解析解 / RK4 / 手算功率）",
     "nominal_math_validation.json", "PASS", "green",
     "证明：名义积分与解析几何、RK4、手写功率式一致到 1e-12/1e-9。不证明：物理仿真的跟踪精度"),
    ("G3-07", "轨迹边界", "名义环境行为（复位、终止优先级、记录）",
     "nominal_env_check.json", "PASS", "green",
     "证明：终止优先级、复位指纹、逐步记录口径。不证明：跨环境一致性"),
    ("G3-08", "轨迹边界", "安全几何（足迹、膨胀一次、碰撞判据）",
     "safety_geometry_check.json", "PASS", "green",
     "证明：分离轴判据、物理碰撞与名义违约分开、膨胀只做一次。不证明：δ 的取值是否适合所有执行带宽"),
    ("G3-09", "轨迹边界", "位姿与足迹对齐",
     "pose_footprint_alignment.json", "PASS", "green",
     "证明：名义位姿与足迹中心的对应关系。不证明：真实基座漂移"),
    ("G3-10", "轨迹边界", "Isaac 场景与名义地图的映射一致性",
     "scene_mapping_check.json", "PASS", "green",
     "证明：地图原点/朝向/边界在两侧一致。不证明：每个 env 的初始位姿分布"),
    ("G3-11", "轨迹边界", "控制时序（10/50/200 Hz、命令到首个子步）",
     "control_timing_check.json", "PASS", "green",
     "证明：每个上层周期 5 次低层推理 + 20 个物理子步，命令写入相位正确。不证明：端到端延迟对闭环的影响"),
    ("G3-12", "轨迹边界", "复位与预热对齐（warmup 能量不混进窗口）",
     "reset_warmup_alignment.json", "PASS", "green",
     "证明：5/5，warmup 只计数不进缓冲，跨 reset 写入会被判 fault。不证明：异步批内失败的处理"),
    ("G3-13", "轨迹边界", "200 Hz 机械功计量（在线 vs 离线复算）",
     "energy_meter_validation.json", "PASS", "green",
     "证明：9/9，逐关节截断后求和，在线与离线复算一致。不证明：τ 与 q̇ 的采样相位误差上界"),
    ("G3-14", "轨迹边界", "执行带宽：低速（0.2–0.31 m/s）与侧移",
     "low_speed_energy_executability.json", "FAIL", "amber",
     "**已知局限**：五档低速 usable 全 0%（判据 ≥90%），实测/名义能耗 1.9–7.0 倍；纯侧移无响应。"
     "已由用户裁定按方案 A 记录局限、不重训、不禁用低速。影响：低速段与侧移实际上是不可用域"),
    ("G3-15", "轨迹边界", "执行带宽：角速率死区",
     "yawrate_deadband.json", "NO_VERDICT", "amber",
     "**实测**：|wz|<0.2 几乎不转、0.3 为 21%、0.5 为 71%。影响：闭环指令的可用角速率下界约 0.5 rad/s，"
     "而契约域是 ±1.0 —— 域内有一半实际执行不了。这条与 G3-14 同属执行带宽，未单独裁定"),

    # ---------------- 可枚举对照
    ("G3-16", "可枚举对照", "受限问题的穷举对照（停止分段/预算符号/联合判据/危险历史）",
     "toy_algorithm_validation.json", "PASS", "green",
     "证明：13/13，740 条轨迹上 select_stop 与 O(T²) 暴力零不一致，预算翻转点恰等于最小成本，"
     "sum-vs-max 反例找到实数。不证明：连续动作域的最优性（报告 scope_note 已声明）"),
    ("G3-17", "可枚举对照", "双目标与停止分段的逐位 oracle",
     "targets_oracle_check.json", "PASS", "green",
     "证明：15/15，120 例随机短轨迹与独立 O(T²) 枚举逐位一致（索引差 0、W 差 3.3e-16）。"
     "不证明：GAE 在长回合上的偏差-方差"),
    ("G3-18", "可枚举对照", "缓冲与可恢复性（样本/模型/优化器/RNG 连续）",
     "rollout_resume_check.json", "PASS", "green",
     "证明：24/24，连续/停-续/崩-续三种跑法逐位相同、无重复写入。不证明：物理环境上的可恢复性"),

    # ---------------- 短训练证据
    ("G3-19", "短训练证据", "无裕度分级短训练（三级课程，预登记）",
     "nominal_smoke_training.json", "FAIL", "amber",
     "**原始结论 FAIL**（G4 判据量错对象），原样保留。可学性信号为真："
     "development 最近到达距离 13.98→6.60 / 13.97→8.41 / 13.97→6.92 m"),
    ("G3-20", "短训练证据", "G4 判据复判（撤回量错对象的判据后重判）",
     "nominal_smoke_training_rejudge.json", "PASS_REJUDGE", "green",
     "证明：按 amend2 的三条与裁剪无关的症状复判，五门全绿；裁剪 97.8% 触发作为整改前置项。"
     "不证明：长训下的优化稳定性"),
    ("G3-21", "短训练证据", "学习候选的指令分布（名义侧，与规则控制器同尺子）",
     "candidate_command_stats.json", "NO_VERDICT", "deferred",
     "**新发现**：候选的指令变化率是规则控制器的 22.4 倍、|wz| 均值 16.6 倍、保持时长 0.1 s（每步都变）、"
     "且出现 0.254 m/s 的侧移（规则控制器恒为 0），15.1% 的步落在 |wz|<0.2 的完全死区。"
     "见 OI-1"),

    # ---------------- 双 critic 与 minimization 的直接证据
    ("G3-22", "双critic与minimization", "三网参数完全分离、动作映射契约一致",
     "network_contract_check.json", "PASS", "green",
     "证明：20/20，policy/critic_cost/critic_reach 参数互不共享（77318/76801/76801），"
     "仿射映射而非 tanh，观测切片具名。不证明：网络容量是否够用"),
    ("G3-23", "双critic与minimization", "两个 critic 的目标与 GAE 定义（γ_c=1、min 型 V_h）",
     "targets_oracle_check.json", "PASS", "green",
     "证明：成本目标为未折扣累计、V_h 为 min 型 Bellman 收缩、截断/终止分开。"
     "注意：实现里 policy 优势用 W_sel−V_total 而非参考的 λ 平滑 GAE（差异清单 D3）"),
    ("G3-24", "双critic与minimization", "最小化更新的端到端（参数只在上层变、冻结不动）",
     "minimization_update_check.json", "PASS", "green",
     "证明：15 步更新把动作第 0 维从 8.7e-5 推到 −0.316，冻结 v2 散列不变。不证明：这是最优更新方式"),

    # ---------------- 上游门禁
    ("G3-25", "轨迹边界", "上游门禁（环境/物理接口/标定适用性）",
     "gate_G2A.json", "PASS", "green",
     "证明：G0/G1/G2/G2A 全部通过；G2A 的 ε_c 成本项全程通过、ε_d 按 δ(T) 表登记（fit_not_validation）。"
     "注意：δ(T) 只覆盖闭环指令约 10–15 s，20 s 起线性超限"),
    ("G3-26", "数值目标", "归一化尺度与场景参考能量匹配（NAV-035 重定 Z_REF_J）",
     "observation_contract_check.json", "PASS", "green",
     "证明：z_ref_provenance 校验的是**性质**而不是某个算式——尺度 ≥ 1.05 × 场景最大参考能量、"
     "≤ 1.20 × 它、且三级参考成本归一化后都 < FAIL_PENALTY=1.0。旧值 225.6 J 只有最大参考能量的 0.46 倍，"
     "三条全不满足（可行性被判死、观测饱和、惩罚弱于成功）。不证明：550 就是唯一正确取值——它只是满足"
     "这三条的最小整数级选择"),
)


def load_evidence(name: str) -> dict[str, Any]:
    path = os.path.join(REPORTS, name)
    if not os.path.exists(path):
        return {"exists": False, "path": name}
    d = json.load(open(path, encoding="utf-8"))
    verdict = None
    for key in ("verdict", "rejudged_verdict", "status"):
        if isinstance(d, dict) and d.get(key) is not None:
            verdict = d[key]
            break
    return {"exists": True, "path": name, "sha256": sha256_file(path), "verdict": verdict,
            "keys": sorted(d.keys())[:14] if isinstance(d, dict) else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    items = []
    problems = []
    for (gid, dim, item, evidence, expected, light, note) in TABLE:
        ev = load_evidence(evidence)
        if not ev["exists"]:
            problems.append(f"{gid}: 证据文件不存在 {evidence}")
        elif expected not in ("NO_VERDICT", "PASS_REJUDGE") and ev["verdict"] != expected:
            problems.append(f"{gid}: 证据实际 verdict={ev['verdict']!r}，表里写的是 {expected!r}")
        items.append({"id": gid, "dimension": dim, "item": item,
                      "evidence": ev, "expected_verdict": expected,
                      "light": light, "note": note})

    by_light = {"green": 0, "amber": 0, "red": 0, "deferred": 0}
    for it in items:
        by_light[it["light"]] += 1

    open_items = [
        {
            "id": "OI-1",
            "title": "学习候选的执行适用性：判定前提未满足，转长训出口条件",
            "status": "deferred_by_user",
            "evidence": "reports/candidate_command_stats.json",
            "facts": {
                "change_rate_mean": {"candidate": 1.177, "rule_controller": 0.0526, "ratio": 22.4},
                "mean_abs_wz": {"candidate": 0.639, "rule_controller": 0.038, "ratio": 16.6},
                "hold_mean_s": {"candidate": 0.100, "rule_controller": 0.402},
                "mean_abs_vy": {"candidate": 0.254, "rule_controller": 0.0},
                "deadband_share_below_0p2_wz": 0.1506,
                "log_std_final": 0.0016,
                "entropy_first_last": [4.2568, 4.2616],
            },
            "diagnosis": "抖动不是策略风格，是探索噪声没退火：log_std 从 0 只走到 0.0016（std 仍≈1），"
                         "而按 std=1 推算的每步指令变化量约 1.2，实测 1.18 —— 两者对得上。"
                         "90 次更新、lr=3e-4 不足以退火。",
            "resolution": "用户裁定：等一个真收敛的候选再量。物理侧复查（NAV-060 口径）不进本轮分母，"
                          "作为 NAV-036 的**出口条件**：长训产出的候选必须用同一把尺子重测，"
                          "变化率/保持时长/死区占比/侧移四项都要给出数值。",
            "residual_risk": "若长训后变化率仍高一个数量级，说明不是退火问题而是目标本身不惩罚抖动；"
                             "届时需要动作平滑项或对指令变化率加约束，属于训练目标层面的改动。",
        },
        {
            "id": "OI-2",
            "title": "预算尺度：已重定 Z_REF_J（225.6069087767049 → 550.0 J）",
            "status": "resolved",
            "evidence": "config/value_objective_contract_v1.json 1.1.0 + config/snapshots/ + "
                        "reports/observation_contract_check.json（z_ref_provenance）",
            "facts": {
                "z_ref_J": {"before": 225.6069087767049, "after": 550.0},
                "reference_energy_J_upper": {"L1": 363.89, "L2": 489.40, "L3": 371.33},
                "ratio_to_max_after": 1.124,
                "normalized_reference_cost_after": {"L1": 0.662, "L2": 0.890, "L3": 0.675},
                "fail_penalty": 1.0,
            },
            "why_it_was_blocking": "z_ratio 必须不饱和残差才说得通，而 z0 <= Z_REF_J；旧尺度低于全部场景的"
                                   "参考能量，于是任何策略在整个回合里都不会被判可行（6214 个样本里 "
                                   "feasible_steps=0）。同时旧尺度下参考成本归一化后是 1.61–2.05 > "
                                   "FAIL_PENALTY=1.0，等于「成功」比「失败惩罚」更贵。",
            "resolution": "把尺度按场景参考能量重定：550 J = 1.05 × 489.40 向上取整。三份契约升版"
                          "（value 1.1.0 / navigation 1.4.0 / budget 1.1.0）并留 byte 级快照；"
                          "observation.py 的常数与出处说明重写；受影响的 11 项检查全部重跑通过。",
            "verification": "z_ref_provenance 改为性质校验：尺度 ∈ [1.05, 1.20] × 场景最大参考能量，"
                            "且三级参考成本归一化后 < FAIL_PENALTY。旧值三条全不满足，新值三条全满足。",
        },
        {
            "id": "OI-3",
            "title": "策略优势用 W_sel − V_total，与参考实现的 λ 平滑 GAE 不同",
            "status": "accepted_with_comparison_planned",
            "evidence": "docs/algorithm_implementation.md 的 D3",
            "facts": {"ours": "adv = W_sel − V_total", "reference": "λ 平滑 GAE of stop-now residual"},
            "resolution": "用户裁定：保持现状，在 NAV-036 的训练里补一组对照。理由：契约明文要求"
                          "「按 τ* 截断、其后不计入优势」，λ 平滑会把 τ* 之后的步混进来；"
                          "参考实现的做法在本项目里没有被证明更优。",
            "residual_risk": "对照组若显示 λ 平滑明显更好，需要改实现并重跑 G3 的梯度方向与 oracle 两项。",
        },
        {
            "id": "OI-4",
            "title": "max_grad_norm 过紧，97.8% 的更新被裁剪",
            "status": "carry_over_into_training_config",
            "evidence": "reports/nominal_smoke_training_rejudge.json（carry_over_items）",
            "facts": {"grad_norm_p50": 1.035, "grad_norm_max": 2.445, "cap": 0.5,
                      "clip_active_share": 0.978},
            "resolution": "写进长训配置：max_grad_norm 0.5 → 2.5（实测最大 2.445）。"
                          "契约 estimator 明确该量属可调优化超参，不需要升版。"
                          "长训后的 G4 复判必须看「裁剪是否仍常态触发」，不再用梯度绝对值当病征。",
        },
    ]

    release_conditions = [
        "长训配置必须显式写 max_grad_norm = 2.5（OI-4），并在训练日志里记录裁剪触发占比",
        "长训产出的候选必须用 check_candidate_commands.py 同一把尺子复测变化率/保持时长/死区占比/侧移，"
        "作为 NAV-036 的出口条件（OI-1）；四项都要给数，不许只说「看起来更稳了」",
        "长训必须补一组 λ 平滑优势的对照（OI-3），否则 D3 这条语义偏离一直是未验证状态",
        "任何一项配置变更（域、观测、事件、预算规则）都要新版本 + 重跑 G3 对应项，不许就地改",
    ]

    contract_bump = {
        "what": "Z_REF_J 225.6069087767049 → 550.0",
        "versions": {"value_objective_contract_v1.json": ["1.0.0", "1.1.0"],
                     "navigation_contract_v1.json": ["1.3.0", "1.4.0"],
                     "budget_comparison_contract_v1.json": ["1.0.0", "1.1.0"]},
        "snapshots": ["config/snapshots/value_objective_contract_v1.zref225.6069.json",
                      "config/snapshots/navigation_contract_v1.zref225.6069.json",
                      "config/snapshots/budget_comparison_contract_v1.zref225.6069.json"],
        "why": "一个常数修三条缺陷：可行性被判死（6214 样本 feasible_steps=0）、观测 z_ratio 饱和导致策略"
               "看不见预算、FAIL_PENALTY=1.0 低于参考成本 1.61–2.05（成功比失败惩罚更贵）",
        "code_changes": ["src/camra_navigation/observation.py（常数 + 出处说明；costs 由它 import）",
                         "src/camra_navigation/algorithm/ppo.py（默认值，实际由契约覆盖）",
                         "src/camra_navigation/algorithm/networks.py（单位说明文字）"],
        "test_changes": ["tests/test_observation_contract.py：z_ratio 的饱和取值区间改成在新尺度下真会饱和的值；"
                         "z_ref_provenance 由「复算某个算式」改成「校验三条性质」",
                         "tests/test_minimization_update.py：断言值 225.6069 → 550.0"],
        "rerun": ["value_objective_alignment", "observation_contract_check", "cost_budget_check",
                  "network_contract_check", "targets_oracle_check", "minimization_update_check",
                  "toy_algorithm_validation", "nominal_math_validation", "nominal_env_check",
                  "budget_fairness_check", "rollout_resume_check"],
        "rerun_result": "11 项全部 exit=0 / PASS",
        "not_rerun_note": "几何/地图/时序/划分这几类不引用 Z_REF_J 或 z_ratio，其报告未重生成；"
                          "它们在新尺度下数值不变（只涉及 J 与几何量）",
        "left_under_old_contract": "reports/nominal_smoke_training*.json 与 candidate_command_stats.json 是"
                                   "在 1.0.0 尺度下产生的（z0=221.09 J 的绝对预算不变），"
                                   "结论仍然成立（预算在目标前耗尽），但其中的 z_ratio 绝对值按旧尺度解释",
    }

    differences = [
        {"id": "D1", "class": "工程近似（不改评分语义，已在任务内证明）",
         "original": "JAX/Flax（optax、jax.lax.scan）", "ours": "PyTorch + 三个独立 Adam",
         "why": "NAV-008 冻结上层框架；数值等价性由 minimization_update_check 的 numpy 转写交叉核对"},
        {"id": "D2", "class": "工程近似（不改评分语义，已证明）",
         "original": "一个 rollout 里自动复位，env done 与停止分段混用同一个 done",
         "ours": "terminated / truncated 分开，一回合一个 chunk",
         "why": "跨 reset 的样本会被当成旧回合的后续，这是数据污染；NAV-032 用逐位恢复测试证明"},
        {"id": "D3", "class": "语义偏离（待对照，OI-3）",
         "original": "policy 优势 = 对 stop-now 残差 g 做 λ 平滑的 GAE",
         "ours": "adv = W_sel − V_total",
         "why": "W_sel 已含整条尾段；λ 平滑会把 τ* 之后的步混进来，与「按 τ* 截断」冲突。谁对没有证据"},
        {"id": "D4", "class": "语义偏离（有理由，已记录）",
         "original": "reach 目标用 critic 的 last_val_h 作为尾部",
         "ours": "用实测 h_T = max(g_norm(x_T), C·y)",
         "why": "避免用被训练的同一个网络给自己造目标；与参考的 reach_append 用法一致"},
        {"id": "D5", "class": "工程近似（不改评分语义，已证明）",
         "original": "动作映射含 tanh 压缩", "ours": "仿射 clip(a,−1,1)×(1,0.4,1)",
         "why": "契约要求低速段精确可达；tanh(1)=0.7616 会系统性压低指令"},
        {"id": "D6", "class": "实现缺陷修正（记录在案）",
         "original": "calculate_indexs3 的逐行转写会返回越界索引",
         "ours": "按契约的 argmin 语义重写，并用 O(T²) 枚举逐位核对",
         "why": "参考实现的 argmin 依赖内部 roll/反转构造，不能直接当 τ 用；NAV-030 记录未改参考副本"},
        {"id": "D7", "class": "结构性差异（发现，未处理）",
         "original": "每个 rollout 多 epoch、多 minibatch，PPO 比例裁剪生效",
         "ours": "每回合一次全批更新 → ratio 恒为 1，clip_eps 实际未被使用",
         "why": "等于带基线的策略梯度；NAV-034 的日志显示比例裁剪触发率恒为 0。"
                "是否改多 epoch 属于训练配置，不是本轮门禁项"},
        {"id": "D8", "class": "契约内部不一致（记录，未改契约）",
         "original": "sign_conventions[5]：A>0 表示动作更好（成本更低）",
         "ours": "A = 目标成本 − V，A>0 更贵",
         "why": "与同一契约的 max 形式最小化损失代数冲突（dL/dlogp ∝ A）。按代数落地，契约措辞待升版"},
        {"id": "D9", "class": "语义问题（待裁定，OI-2）",
         "original": "energy 用未截断的累计值构造残差",
         "ours": "残差用 z_ratio（±1 截断）",
         "why": "z0 > Z_REF 时成本项退化为 min(z0, Z_REF)，而全部场景的参考能量都 > Z_REF"},
    ]

    blocking = [o for o in open_items if o["status"] == "open"]
    verdict = "BLOCKED" if blocking else "PASS_WITH_CONDITIONS"
    report = {
        "task": "NAV-035",
        "schema": "camra-navigation-gate-g3-v1",
        "verdict": verdict,
        "summary": {
            "n_items": len(items),
            "lights": by_light,
            "releasable": not blocking,
            "why_conditions": "3 项 amber 是用户已裁定的已知局限（执行带宽），1 项 deferred 是经批准的"
                              "长训出口条件；其余全绿。放行条件是 release_conditions 里那四条，"
                              "其中三条会在长训的预登记里变成可检查的条目",
            "blocking_open_items": [o["id"] for o in blocking],
        },
        "release_conditions": release_conditions,
        "contract_bump": contract_bump,
        "checklist": items,
        "dimensions": {d: [it["id"] for it in items if it["dimension"] == d] for d in DIMENSIONS},
        "differences_original_vs_engineering": differences,
        "open_items": open_items,
        "frozen_versions_for_formal_training": {
            "value_objective_contract_v1.json": "1.1.0",
            "navigation_contract_v1.json": "1.4.0",
            "budget_comparison_contract_v1.json": "1.1.0",
            "geometry_contract_v1.json": "见文件（本轮未改）",
            "note": "NAV-036 的预登记必须引用这些版本号与散列，不许引用旧快照",
        },
        "evidence_integrity": {"problems": problems},
        "method": (
            "本表不重算任何数值：每项读证据文件的散列与实际 verdict，表里声明的期望值与实际不符就报错。"
            "「能证明什么/不能证明什么」逐条写死，避免把「过了 A 检查」当成「A 所代表的性质全部成立」。"
        ),
    }
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    print(json.dumps({"verdict": report["verdict"], "n_items": len(items),
                      "lights": by_light, "problems": problems,
                      "open_items": [o["id"] for o in open_items]}, ensure_ascii=False))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
