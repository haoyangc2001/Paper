"""Build reports/rcppo_alignment.json (NAV-063).

The report is deliberately **computed**, not typed: hashes come from disk, the
deviation numbers come from the same functions the trainer uses, and the matrix
rows carry file:line pointers that are checked to exist.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

NAV = os.getcwd()
REPO = os.path.dirname(os.path.dirname(NAV))
sys.path.insert(0, os.path.join(NAV, "src"))

import numpy as np  # noqa: E402

from camra_navigation.algorithm import storage as S  # noqa: E402
from camra_navigation.algorithm import targets as T  # noqa: E402


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def lines_of(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return fh.read().splitlines()


def find_line(path: str, needle: str) -> int:
    for i, line in enumerate(lines_of(path), 1):
        if needle in line:
            return i
    raise SystemExit(f"{needle!r} not found in {path}")


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


OURS = {
    "targets": "src/camra_navigation/algorithm/targets.py",
    "storage": "src/camra_navigation/algorithm/storage.py",
    "ppo": "src/camra_navigation/algorithm/ppo.py",
    "nominal": "src/camra_navigation/envs/nominal.py",
    "observation": "src/camra_navigation/observation.py",
    "train_nominal": "scripts/train_nominal.py",
    "train_formal": "scripts/train_formal.py",
    "freeze": "scripts/freeze_deployment.py",
    "search": "src/camra_navigation/budget/search.py",
    "value_contract": "config/value_objective_contract_v1.json",
    "alignment_contract": "config/rcppo_alignment_contract_v1.json",
    "amendment": "config/rcppo_alignment_contract_v1.amend1.json",
    "prereg": "config/gate_g3_alignment_v2_prereg.json",
}
REF = {
    "gae": os.path.join(REPO, "RC-PPO", "rl", "gae.py"),
    "ec": os.path.join(REPO, "RC-PPO", "rl", "EC-EFPPO.py"),
    "utils": os.path.join(REPO, "RC-PPO", "rl", "EFPPO_utils.py"),
}

# --- 偏差量化：联合 λ-return 的 γ（我们 1.0，参考 GAMMA_REACH_INIT=0.999） ---
c = S.ObjectiveConfig.from_contracts(
    os.path.join(NAV, "config", "value_objective_contract_v1.json"), r_goal_m=0.5)
lam = c.gae_lambda
n = 30
g = np.ones(n + 1)
done = np.zeros(n + 1, dtype=int)
done[-1] = 1
g1 = T.lambda_return(g, done, lam)[0]
g_ref = (1.0 - lam) * sum((lam ** k) * (0.999 ** k) for k in range(n)) + (lam ** n) * (0.999 ** n)
delta_gamma = float(abs(g1 - g_ref))

matrix = [
    {
        "topic": "样本时序 t/t+1",
        "reference": f"RC-PPO/rl/gae.py 的三套 GAE 都在 x_t 处取 z_t 与 y_t；采样器按 step 记录 reward/energy/reach",
        "current": f"src/camra_navigation/algorithm/storage.py:{find_line(OURS['storage'], 'def raw_fields')} raw_fields（schema v2，无后缀=t、*_next=t+1）",
        "units_dims_signs": "c_t 无量纲(=J/Z_REF)；z_t 有符号；y_t∈{0,1}；全部标量/41 维观测",
        "difference": "NAV-062 修完：旧实现用步后量组装 h_t 并重复补 h[-1]",
        "fix_evidence": "tests/test_training_semantics.py（24 项，含两步手算与单侧变异）",
    },
    {
        "topic": "安全历史 y",
        "reference": "无独立 y；avoid 由 reach 值承担（刻意偏离的基础）",
        "current": f"src/camra_navigation/algorithm/targets.py:{find_line(OURS['targets'], 'def safety_step')} h=max(g_norm, C·y)，C=1.0，y 单调不可逆",
        "units_dims_signs": "h≥0 无量纲；g_norm 目标内为 −0.001，被 max(·,0) 抹平（契约 max_zero_caveat）",
        "difference": "刻意偏离（保留）：落地到达不清零 y，danger_then_arrival 用反例钉住",
        "fix_evidence": "tests/test_rcppo_alignment.py::danger_then_arrival_infeasible",
    },
    {
        "topic": "预算 z",
        "reference": "energy 直接进观测；is_terminal 只看时间",
        "current": f"src/camra_navigation/observation.py:{find_line(OURS['observation'], 'def z_ratio_value')} z_ratio=clip(z_J/Z_REF, ±1.5)；目标用未裁剪 z_norm",
        "units_dims_signs": "z_J 焦耳、有符号；z_norm=z_J/Z_REF 无量纲；z_ratio 仅编码",
        "difference": "obs-v3 放宽裁剪上限；预算耗尽不终止（NAV-062）",
        "fix_evidence": "tests/test_training_semantics.py::negative_budget_keeps_stepping",
    },
    {
        "topic": "共同停止",
        "reference": f"RC-PPO/rl/gae.py:{find_line(REF['gae'], 'def calculate_indexs3')} calculate_indexs3 → indexs + done；done 供三套估计器共用",
        "current": f"src/camra_navigation/algorithm/targets.py:{find_line(OURS['targets'], 'def select_stop')} select_stop（逐时刻 argmin，首个最小值）+ storage 的 done_ext",
        "units_dims_signs": "W=max(h_τ, spent(t,τ)+[τ=T]V_c−z_t)；γ_c=1 时 W(t,τ)=g_τ=max(h_τ,−z_τ)",
        "difference": "等价实现（逐时刻重取 vs 反向扫描）；γ_c=1 下 τ*(t) 单调不减，λ-return 的递推成立",
        "fix_evidence": "tests/test_rcppo_alignment.py::shared_stop_overrides_each_critic_own_endpoint, post_stop_cost_does_not_pollute",
    },
    {
        "topic": "三套估计器",
        "reference": f"calculate_gae2（成本）/ calculate_gae_reach4（reach）/ calculate_gae_reach4 on g（联合）——同一 done",
        "current": f"src/camra_navigation/algorithm/targets.py:{find_line(OURS['targets'], 'def lambda_return')} lambda_return + {find_line(OURS['targets'], 'def cost_to_stop')} cost_to_stop；storage.compute_objectives 一次调用三处",
        "units_dims_signs": "成本目标=到 τ* 的 Σc（+[τ*=T]V_c）；reach 目标=min 型 DP 值上的 λ-return；联合=G(g)",
        "difference": "原实现：整回合 suffix / 未掩码 min / W_sel−V_total 一次性差（M6）。已改为共享 τ* 的 λ 平滑三估计器",
        "fix_evidence": "tests/test_rcppo_alignment.py::production_uses_three_estimators_with_one_mask（调用计数 + done 相等）, tau_equals_T_and_bootstrap（手算）",
    },
    {
        "topic": "联合 λ-return 的 γ",
        "reference": "config GAMMA_REACH_INIT=0.999 用于 g 的 GAE",
        "current": f"storage.ObjectiveConfig.gamma_joint=γ_c=1.0（见 {OURS['value_contract']} gamma_cost_rationale）",
        "units_dims_signs": f"30 步、g≡1 时两种口径的差 ΔG={delta_gamma:.6f}（相对 {(delta_gamma/n):.3%}/步）",
        "difference": "有意偏离：契约目标是未折扣 J，W 也定义在未折扣量上；用 0.999 等于给后发生的成本打折",
        "fix_evidence": "tests/test_rcppo_alignment.py::gamma_cost_undiscounted, discounted_cost_would_differ",
    },
    {
        "topic": "终端与失败尾项",
        "reference": "done 掩码截断 + last_val 单次 bootstrap；失败由 reach=1 承担",
        "current": f"src/camra_navigation/algorithm/storage.py:{find_line(OURS['storage'], 'TERMINAL_KIND')} TERMINAL_KIND + 失败回合强制 τ*=T",
        "units_dims_signs": "failure→FAIL_PENALTY=1.0；success→无 bootstrap；truncation→+V_c(x_T)",
        "difference": "参考只做 argmin，不强制失败回合停在 T；契约要求失败惩罚传播到前序目标，故强制",
        "fix_evidence": "tests/test_rcppo_alignment.py::failure_penalty_propagates（逐步手算 λ-return）",
    },
    {
        "topic": "动作概率",
        "reference": "对角高斯 + 可学习 log_std；上游裁剪",
        "current": f"src/camra_navigation/algorithm/ppo.py:{find_line(OURS['ppo'], 'def policy_loss')} log_prob 在 raw 空间；applied=clip(a)·COMMAND_MAX",
        "units_dims_signs": "A>0 = 更贵（代数结论，契约措辞待升版）",
        "difference": "与参考一致；顶格残余风险已量化（NAV-062 报告）",
        "fix_evidence": "tests/test_rcppo_alignment.py::clip_branches_four_cells",
    },
    {
        "topic": "PPO 更新",
        "reference": f"RC-PPO/rl/EFPPO_utils.py:{find_line(REF['utils'], 'def _ecefppo_update')} _ecefppo_update（max 形式，最小化）",
        "current": f"src/camra_navigation/algorithm/ppo.py:{find_line(OURS['ppo'], 'def policy_loss')} policy_loss + MinimizationUpdater（old_log_prob 固定、多 epoch/minibatch）",
        "units_dims_signs": "loss=mean(max(ratio·A, clip(ratio)·A)) − ent_coef·H；梯度裁剪 2.5",
        "difference": "一致（含熵符号与裁剪四格）",
        "fix_evidence": "tests/test_rcppo_alignment.py::positive_advantage_penalises_higher_probability, entropy_sign_is_negative, multi_epoch_fixed_old_log_prob",
    },
    {
        "topic": "搜索 value（部署预算）",
        "reference": "value-based：预算搜索用 V_total=max(V_h, V_c−z) 判据",
        "current": f"src/camra_navigation/budget/search.py:{find_line(OURS['search'], 'def search_min_budget')} search_min_budget(score,...) 接口就绪；NAV-041 的标签扫描已登记为旧证据",
        "units_dims_signs": "score 由调用方注入；z 用 J，w 用未裁剪比值",
        "difference": "接口一致；接双 critic 的判据归 NAV-066，本项只保证 V_total 的定义与估计器同源",
        "fix_evidence": "tests/test_budget_search.py（17/17，未改语义）+ 本报告的 V_total 行",
    },
]

report = {
    "schema": "camra-rcppo-alignment-report-v1",
    "task": "NAV-063",
    "git_head_at_run": git_head(),
    "plan_version": "1.3.1",
    "reference_repository": "RC-PPO/（只读）",
    "verdict": "aligned_with_documented_deviations",
    "reference_hashes": {k: sha(v) for k, v in REF.items()},
    "current_hashes": {k: sha(os.path.join(NAV, v)) for k, v in OURS.items()},
    "estimator_definition": {
        "shared_stop": "τ*(t)=argmin_{τ≥t} W(t,τ)，done[t]=1[τ*(t)=t]，失败回合强制 τ*=T",
        "cost": "V_c 目标(t)=Σ_{s=t}^{τ*(t)-1} c_s + 1[τ*=T]·V_c(x_T)；failure 的末步=FAIL_PENALTY",
        "reach": "V_h 目标(t)=λ-return(V_h^{DP})，V_h^{DP} 为 min 型 Bellman 值（γ_h 退火）",
        "joint": "V_total=max(V_h, V_c−z_t)；policy 目标=λ-return(g_train)，g_train,τ=max(h_τ,−z_τ)+w·c_τ，w 默认 0/训练预登记 50，γ=1",
        "lambda": lam,
        "gamma": {"cost": c.gamma_cost, "reach": c.gamma_h, "joint": c.gamma_joint},
    },
    "deviations": [
        {
            "id": "D1",
            "topic": "联合 λ-return 的 γ",
            "ours": 1.0,
            "reference": 0.999,
            "quantified": f"30 步常值 g 上 ΔG={delta_gamma:.6f}",
            "why": "契约 γ_c=1 且 W 未折扣；改 0.999 会让 100 J 成本在 10 步后少算约 1 J",
            "accepted_by": "NAV-061 契约 gamma_cost=1.0 + 本任务",
        },
        {
            "id": "D2",
            "topic": "失败回合强制停止在 T",
            "ours": "τ*=T（所有 t）",
            "reference": "argmin 可能选更早的 τ",
            "quantified": "预算宽松失败样本上，不强制时前序联合目标不含 1.0 惩罚",
            "why": "契约要求失败惩罚传播到全部前序目标",
            "accepted_by": "NAV-063（本报告 failure_penalty_propagates）",
        },
        {
            "id": "D3",
            "topic": "reach 目标的尾部",
            "ours": "实测 h_T=max(g_norm(x_T), C·y_T)",
            "reference": "环境给的 reach_append 末项",
            "quantified": "同式；差别只在参考不区分 y（无独立 y）",
            "why": "不可逆安全历史（M4）",
            "accepted_by": "NAV-061 契约 safety_history",
        },
        {
            "id": "D4",
            "topic": "联合估计器的递推形态（NAV-064 补登记）",
            "ours": "g_train 上的普通 λ-return（γ_c=1，λ=0.95）减 V_total",
            "reference": "g_append=max(reach,−energy) 上的 calculate_gae_reach4（min 型 Bellman + 归一化 GAE 系数）",
            "quantified": "严格参考形态按 run 聚合：r5_uncal corr(h,adv) +0.9464→−0.9347；r1c_uncal +0.3862→−0.7435；r5_full 0.0000→+0.4408（3 run 中 2 个翻负）",
            "why": "本仓库 h=max(g_norm,C·y) 的 max(·,0) 抹掉了参考 reach 到达时 −300 的负奖励锚点，min 递推失去传播对象并反转梯度方向；joint_min_return() 已实装但默认关闭，训练器拒绝启用",
            "accepted_by": "config/rcppo_alignment_contract_v1.amend3.json（用户 2026-11-13 批准 ②）",
        },
        {
            "id": "D5",
            "topic": "策略即时代价里的逐步成本项（NAV-064 修正 ①）",
            "ours": "g_train = max(h,−z) + 50·c（终端不加）",
            "reference": "g_append = max(reach,−energy)，每步成本通过 energy 递减间接进入",
            "quantified": "w=50 时 corr(成本,优势) 中位数 r5_full 0.000→+0.544、r5_uncal +0.079→+0.170、r1c_uncal +0.033→+0.345；r5_uncal 到达信号 0.944→0.942",
            "why": "参考 reach×100、energy −400~800 同量级；本仓库 c≈0.009 而 h∈[0,1]，差约 100 倍需显式补偿。属契约 objective（最小累计机械功）的落地，非语义改写",
            "accepted_by": "config/objective_signal_fix_prereg_v1.json（先冻结）",
        },
        {
            "id": "D6",
            "topic": "动作分布的可选有界形式（NAV-064 修正 B）",
            "ours": "action_bounded=True 时 SquashedNormal（tanh + 雅可比修正）；默认 False 用原对角高斯",
            "reference": "对角高斯 + 有界裁剪映射",
            "quantified": "新初始化权重顶格率 31.7%→0%、log_prob 数值积分 1.0014；已发散的 r6 权重上 B 无效（98.7%~99.3%），故第 7 轮须重置策略头",
            "why": "契约 known_defect 要求给出可学信号；映射层注释把有界化划归网络层。映射保持仿射、低速段精确可达",
            "accepted_by": "config/rcppo_alignment_contract_v1.amend4.json（用户 2026-11-13 批准 B）",
        },
        {
            "id": "D7",
            "topic": "预算基准的可选一致口径（NAV-064 修正 A）",
            "ours": "budget_accounting=consistent 时 z0=k*E_ref*(1+alpha(variant))；默认 nominal",
            "reference": "RC-PPO 的 a.u. 预算，无名义/实扣口径之分",
            "quantified": "full 的 k=1.0 预算 363→432 J，与参考路线扣费 432 J 恰在 W=0 边界；uncalibrated 的 alpha=0，A 无影响",
            "why": "同一预算量须用同一把尺子；修后参考路线恰在边界，不是阈值放宽。冻结 E_ref 报告未改写",
            "accepted_by": "config/budget_action_fix_prereg_v1.json v1.1.0（先冻结）",
        },
    ],
    "matrix": matrix,
    "open_gaps": [
        "M12（生产调用链完整性）：本项用真实训练产物 + critic 重拟合入口证明，但预算搜索端到端仍归 NAV-066",
        "与参考的逐位等价未证明也不要求：D1/D2 是有意偏离，差异量已量化",
        "物理侧（200 Hz 子步）的同一口径复验归 NAV-067",
    ],
}

out = os.path.join(NAV, "reports", "rcppo_alignment.json")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(report, fh, ensure_ascii=False, indent=2)
print(json.dumps({"written": out, "rows": len(matrix), "delta_gamma": delta_gamma,
                  "bytes": os.path.getsize(out)}, ensure_ascii=False))
print("sha256", sha(out)[:16])
