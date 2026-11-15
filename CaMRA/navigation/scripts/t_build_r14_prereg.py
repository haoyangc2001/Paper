#!/usr/bin/env python3
"""构造并冻结 NAV-064 第 14 轮预登记（RC-PPO 对齐轮）。

以 r13 预登记为底稿，只改预授权的语义变更项；其余字段逐字继承。
任何意外差异都会被断言拦下。产出：
  config/training_loop_prereg_r14.json
  config/training_loop_prereg_r14.json.sha256
"""
import hashlib
import json
import os

NAV = "/home/caohy/repositories/Paper/CaMRA/navigation"


def sha_of(rel: str) -> str:
    with open(os.path.join(NAV, rel), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main() -> None:
    r13 = json.load(open(os.path.join(NAV, "config/training_loop_prereg_r13.json"),
                         encoding="utf-8"))
    r14 = json.loads(json.dumps(r13))          # deep copy

    # ---- 1) 版本与身份 ------------------------------------------------------
    r14["version"] = "r14.0.0"
    r14["round"] = 14
    r14["status"] = "frozen_before_round_14"
    r14["created_at"] = "2026-11-15T18:10:00+08:00"
    r14["task"] = (
        "RC-PPO 对齐轮（用户 2026-11-15T17:32 三点决策授权）：四项对齐改动 + 并入两项独立问题。"
        "①深井：G_INSIDE -0.001 → -3.0（值契约 v1.3.0），h 语义 v2_deep_well（安全态 h=g_norm "
        "不再被 max(·,0) 抹平；违规态 max(g,C·y) 不变）——修复第 38 条记录定案的『到达无奖励』；"
        "②启用参考的 min 型联合递推（amend3 反例的根因即符号体系，深井后重验门禁）；"
        "③撤加性罚：joint_cost_weight 50→0、base_weight 2→0（成本只经 −z 进 max 与 V_c−z，"
        "『省着用』交由预算抽样 k∈[0.6,1.3] 与部署期预算搜索 NAV-066）；"
        "④obs-v4：z_ratio 编码 ±1.5→±3.0（D9 修复，成本通道不再对 critic 饱和）。"
        "并入：⑤安全项连续化（分级严重度，轻擦边 floor=0.2 严格便宜于深撞=1.0，不可逆 max-历史）；"
        "⑥停滞规则 0.05→0.02 m + 触发现场日志。全程补诊断日志：每局 episode_diag"
        "（停滞现场/严重度/终点状态）、每 update 的 corr_g_adv / corr_c_adv / h 分布。"
        "M10 预算搜索按推荐归 NAV-066，本轮不做。"
    )

    # ---- 2) 授权 ------------------------------------------------------------
    r14["authorization"] = {
        "needed": True,
        "why": "值契约语义升版（G_INSIDE/h 符号/分级安全）+ 观测编码升版（obs-v4）+ "
               "此前冻结关闭的 min 型递推开关 + 停滞终止参数变更；r1-r13 结果不可与新版直接比较",
        "granted_by": "用户 2026-11-15T17:32『请按以下三点决策继续推进』（正式开工 + "
                      "并入安全项与停滞规则 + M10 按推荐留给 NAV-066）",
        "granted_at": "2026-11-15T17:32:00+08:00",
        "scope": "6 个 run（2 方法 × 3 种子，协议 per_round.runs=6，无臂拆分）；"
                 "契约余量版 T=45 s / k=1.15 / E_ref 沿用 r13（唯一变量是目标函数语义与安全/终止规则）；"
                 "门禁：T4 反例测试（3 个历史 run corr(h,adv)>0）不过不开训",
        "overrides": [
            "r13 stops.falsified『不追加第 14 轮』——用户 17:32 明确授权开工",
            "config/wallclock_amend_v1.json：墙钟上限维持取消",
        ],
        "cap_note": "墙钟不设上限；必须跑满 600 updates 与全部 6 run，不得因耗时缩短。",
    }

    # ---- 3) 绑定文件 --------------------------------------------------------
    bind = r14["binding"]
    bind["value_contract"] = {
        "path": "config/value_objective_contract_v1.json",
        "sha256": sha_of("config/value_objective_contract_v1.json"),
        "version": "1.3.0",
        "note": "r14 授权修订版（深井 + 分级安全），取代 r13 绑定的 v1.2.0 sha",
    }
    bind["navigation_contract"] = {
        "path": "config/navigation_contract_v1.json",
        "sha256": sha_of("config/navigation_contract_v1.json"),
        "version": "1.6.0",
        "note": "obs-v4：observation_spec.version 与 z_ratio range ±3.0",
    }
    bind["round_13_prereg"] = {
        "path": "config/training_loop_prereg_r13.json",
        "sha256": sha_of("config/training_loop_prereg_r13.json"),
    }
    bind["round_13_report"] = {
        "path": "reports/training_round_r13.json",
        "sha256": sha_of("reports/training_round_r13.json"),
    }
    bind["round_13_negative"] = {
        "path": "reports/negative_result_r13.json",
        "sha256": sha_of("reports/negative_result_r13.json"),
    }
    bind["h_semantics_test"] = {
        "path": "tests/test_h_semantics_v2.py",
        "sha256": sha_of("tests/test_h_semantics_v2.py"),
        "note": "HS1-HS9：深井/分级安全/停滞现场/obs-v4 的单元验收",
    }

    # ---- 4) 假设与证伪 ------------------------------------------------------
    r14["hypothesis"] = (
        "第 38/39 条记录定案：r7-r13 的失败根源是目标函数尺度（到达无奖励 G_INSIDE=-0.001、"
        "加性罚压过前进信号 w=50/base=2、观测饱和）。r12 旧策略在 T=45/k=1.15 下 SR 1.00"
        "（uncal_s0_nomask）证明运动能力本身存在。深井 + min 递推恢复『到达』的方向信号"
        "（终端联合值 = -min(3, 剩余预算)，到达严格优于差 1 m 停下，省电直接放大深井）；"
        "撤加性罚后前进不再净亏。预期：corr_g_adv > 0 稳定出现，dev sr_feasible 中位 ≥ 0.1。"
        "分级安全的预期：full 变体危险步显著下降（贴边不再=撞死同价）。"
        "替代解释（须被排除）：效果来自契约余量本身而非目标函数——r13 的 B 臂（同契约、base=0、"
        "但无深井无 min 递推）正是对照：它 12 run 全部 sr_feasible=0，若 r14 显著好于 r13-B，"
        "则语义变更是净贡献。"
    )
    r14["falsification"] = (
        "满足任一即判本轮假设不成立：(a) T4 门禁：3 个历史 run 上 corr(h, adv_total) 任一 ≤ 0"
        "（深井没修好方向，min 递推仍翻车）→ 不开训，回退方案 A（终点超额一次性计费）；"
        "(b) 6 run 全部跑满 600 updates 后 dev（6 图 × 3 种子 × k=1.15，T=45 s）sr_feasible "
        "中位 < 0.10；(c) smoke 后全量日志 corr_g_adv 的 update 级中位 ≤ 0 且训练期到达率"
        "< r13 同口径。触发即停，写正式负结果；不改判据、不放宽阈值、不追加轮次（除用户另行授权）。"
    )

    # ---- 5) 方法：6 run，无臂 ------------------------------------------------
    r14["method"] = {
        "variants": ["uncalibrated", "full"],
        "seeds": [0, 1, 2],
        "runs": 6,
        "arms": None,
        "baseline_comparison": "r13 B 臂（同契约、w=0、无深井/min 递推）为本轮对照",
        "dead_step_mask": "继承（enabled=true，阈值 0.5；分级严重度下 y≤0.5 的轻度违规步不再被屏蔽，正合 F2 的意图）",
        "steering_prior": "继承（gain=0.2）",
        "cost_curriculum": "关闭（w=0，课程无意义）",
    }

    # ---- 6) 目标函数开关 ----------------------------------------------------
    r14["objective_signal_fix"] = {
        "joint_cost_weight": 0.0,
        "joint_cost_base_weight": 0.0,
        "joint_min_recursion": True,
        "cost_on_overspend_only": False,
        "h_semantics": "v2_deep_well",
        "note": "成本不再以任何加性罚进梯度，只经 −z 存量进 max 与 V_c−z 进 W；"
                "min 递推须与 v2_deep_well 同现（训练器强校验）；"
                "深井生效需值契约 v1.3.0（G_INSIDE=-3.0），已绑定。",
    }

    # ---- 7) 课程关闭 --------------------------------------------------------
    r14["cost_curriculum"] = {
        "enabled": False,
        "form": "不适用：joint_cost_weight=0，没有可课程的成本项",
        "unchanged_from": "r13 的结构字段保留但关闭",
    }

    # ---- 8) 新增：停滞规则与安全连续化 --------------------------------------
    r14["stagnation"] = {
        "window_s": 3.0,
        "min_displacement_m": 0.02,
        "note": "r12/r13 用 0.05：最好的一局（uncal_s2_b0）在 41.4 s、距目标 2.70 m、剩 50 J 时被判"
                "停滞——是否误杀需证据。阈值放宽到 0.02（最后 1-2 m 低速精调的典型速度 "
                "0.007-0.017 m/step·s 之下不再误判），并在触发时记录完整现场"
                "（位置/距目标/余电/窗口位移/剩余时间）供事后判别。",
    }
    r14["safety_continuity"] = {
        "enabled": True,
        "severity_floor": 0.2,
        "severity_ref_depth_m": 0.25,
        "form": "violation 严重度 = clip(floor + (1-floor)·depth/ref_depth, floor, 1)；"
                "y = 不可逆 max-历史；撞死回合仍以 FAIL_PENALTY=1.0 终止",
        "evidence": "F1：二值盖章下贴边=深撞同价（h 恒 1），r13 full 危险步 941-1818、"
                    "碰撞 42-71%%；F2：盖章后无梯度。HS1-HS9 单测验收。",
    }

    # ---- 9) 其余字段 --------------------------------------------------------
    r14["reporting"] = (
        "reports/training_round_r14.json + 台账；须与 r13（同契约、旧语义）并列："
        "sr_feasible、sr、能耗/预算比、危险步、max_safety_severity 分布、停滞触发统计"
        "（误杀/真卡分类）、corr_g_adv/corr_c_adv 的 update 级轨迹；"
        "并注明 r14 为 v1.3.0 语义 + obs-v4，不可与 r1-r13 直接比较"
    )
    r14["resource"] = {
        "per_round_wall_clock_cap_h": None,
        "total_wall_clock_cap_h": None,
        "wallclock_policy": "quality_first_no_wallclock_cap",
        "max_rounds": 4,
        "runs_this_round": 6,
        "used_so_far": {"rounds": 13, "calendar_hours": 19.76},
        "runs_used_total": 82,
        "note": "6 run（2 方法 × 3 种子），协议 per_round.runs=6；墙钟不设上限，"
                "必须跑满 600 updates。",
    }
    r14["data_access"] = dict(r13["data_access"])
    r14["stops"] = {
        "met": "立刻停止，进入 NAV-065 候选冻结",
        "wall_clock": "不适用：墙钟上限维持取消",
        "infrastructure": "换 run id 重跑，与原 attempt 一并报告",
        "falsified": "写正式负结果并结案；不追加第 15 轮（除用户另行授权）",
        "semantic_doubt": "若 sr_feasible 上升但能耗比未改善，或反之，须分项如实报告，不得择一报喜",
    }
    r14["contract_variant"] = {
        "is_variant": True,
        "differs_from_r13": [
            "value_objective_contract v1.2.0 → v1.3.0（G_INSIDE -0.001 → -3.0，分级安全语义）",
            "objective_signal_fix.h_semantics：新增 v2_deep_well",
            "objective_signal_fix：w 50→0、base 2→0、min_recursion false→true、overspend_mask false",
            "observation obs-v3 → obs-v4（z_ratio clip 1.5 → 3.0）",
            "safety_continuity：新增（分级严重度）",
            "stagnation.min_displacement_m 0.05 → 0.02（+触发现场日志）",
            "cost_curriculum：关闭",
        ],
        "comparability": "不可与 r1-r13 的 SR/能耗/危险步直接比较（h 语义、观测编码、终止规则均变）；"
                         "唯一有效对照是 r13 B 臂（同契约同 k 同 T，仅差语义）",
        "why_now": "第 38/39 条记录：13 轮失败定案于目标函数尺度；RC-PPO 参考理论对齐是治本路径",
    }
    r14["amendments"] = [
        {
            "version": "r14.0.0",
            "date": "2026-11-15T18:10:00+08:00",
            "old_sha256": sha_of("config/training_loop_prereg_r13.json"),
            "trigger": "r13 判据 (c) 触发后第 38/39 条记录定案目标函数尺度问题；用户 17:32 授权",
            "change": "四项 RC-PPO 对齐 + 安全连续化 + 停滞放宽（详见 task 字段）",
            "unchanged": "契约余量（T=45/k=1.15/E_ref_t45）、dead_step_mask、转向先验、"
                         "动作初始化、600 updates、算法超参、dev 划分、fresh init、预算抽样带",
        }
    ]

    # ---- 守卫：与 r13 的顶层键差异必须是预期集合 ----------------------------
    changed = {k for k in set(r14) | set(r13)
               if json.dumps(r14.get(k), sort_keys=True) != json.dumps(r13.get(k), sort_keys=True)}
    expected_changed = {
        "version", "round", "status", "created_at", "task", "authorization",
        "binding", "hypothesis", "falsification", "method",
        "objective_signal_fix", "cost_curriculum", "stagnation",
        "safety_continuity", "reporting", "resource", "stops",
        "contract_variant", "amendments",
    }
    unexpected = changed - expected_changed
    assert not unexpected, f"意外变更的顶层键: {unexpected}"
    assert r14["environment"] == r13["environment"], "environment 不得变更"
    assert r14["budget"] == r13["budget"], "budget 不得变更"
    assert r14["algorithm"] == r13["algorithm"], "algorithm 不得变更"
    assert r14["window_semantics"] == r13["window_semantics"], "window_semantics 不得变更"
    assert r14["reference_energy"] == r13["reference_energy"], "reference_energy 不得变更"
    assert r14["action_init_fix"] == r13["action_init_fix"], "action_init_fix 不得变更"
    assert r14["steering_prior"] == r13["steering_prior"], "steering_prior 不得变更"
    assert r14["dead_step_mask"] == r13["dead_step_mask"], "dead_step_mask 不得变更"
    assert r14["action_bounded"] == r13["action_bounded"], "action_bounded 不得变更"

    out = os.path.join(NAV, "config", "training_loop_prereg_r14.json")
    payload = json.dumps(r14, ensure_ascii=False, indent=1) + "\n"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(payload)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    with open(out + ".sha256", "w", encoding="utf-8") as fh:
        fh.write(digest + "\n")
    print("r14 prereg frozen:", digest)
    print("value_contract sha:", r14["binding"]["value_contract"]["sha256"][:16], "...")
    print("nav_contract sha:", r14["binding"]["navigation_contract"]["sha256"][:16], "...")


if __name__ == "__main__":
    main()
