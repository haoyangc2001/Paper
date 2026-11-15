"""Build/freeze the round-13 prereg pair.

Round 13 tests one thing: does the in-budget step cost (``joint_cost_base_weight``)
make the policy stop spending the budget it is given?  To make that question
measurable at all, the task itself gets the margin that S1 measured as missing:

    deadline 30 -> 45 s          (v_deadline = L/45 = 0.311 m/s, the energy optimum)
    E_ref     recomputed at 45 s (reports/train_map_reference_energy_t45.json)
    budget    k = 1.15 on that E_ref

Both arms share the contract; the only difference is ``joint_cost_base_weight``
(2.0 in arm A, 0.0 in arm B).  Round 12's dead-step mask stays on in both arms, so
this round does not re-litigate it.
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "training_loop_prereg_r12.json")
assert os.path.exists(TEMPLATE), TEMPLATE


def sha_of(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def build(base_weight, path, version, status, label):
    d = json.load(open(TEMPLATE, encoding="utf-8"))
    d["version"] = version
    d["round"] = 13
    d["created_at"] = "2026-11-15T13:20:00+08:00"
    d["status"] = status

    d["task"] = (
        "契约余量 + 预算内成本项的实际训练验证轮（%s）：S1 实测发现预算恰好等于"
        "「参考路线按截止速度跑完」的能耗、时限恰好用满，两者同时零余量，"
        "且策略会把多给的预算直接花掉（能耗/预算 92%%→114-123%%）。"
        "本轮把时限放宽到 45 s 并按新时限重算 E_ref（让 0.311 m/s 的省能速度可行），"
        "同时检验「预算内逐步成本也进梯度」能否让策略真的省着用。" % label
    )

    d["authorization"] = {
        "needed": True,
        "why": "新正式训练轮次；改变任务时限与预算口径（评分语义）；启用一个此前默认关闭的成本项开关",
        "granted_by": "用户 2026-11-15T12:57 / 13:13『可以，那就按照你的分析继续推进任务』",
        "granted_at": "2026-11-15T13:13:00+08:00",
        "scope": "12 个 run（2 臂 × 2 方法 × 3 种子）；契约余量版 T=45 s / k=1.15 / E_ref 重算；"
                 "A 臂 joint_cost_base_weight=2.0，B 臂 0.0；其余超参与 r12 逐字段一致",
        "overrides": [
            "r12 stops.falsified『不追加第 13 轮』——用户 13:13 明确授权继续推进",
            "config/wallclock_amend_v1.json：取消单轮 3.0 h 与累计 18.0 h 两项上限",
        ],
        "cap_note": "本轮不设墙钟上限；仍须跑满 600 updates 与每臂 6 个 run，不得因耗时缩短。",
    }

    d["binding"] = {
        **(d.get("binding") or {}),
        "round_12_prereg": {"path": "config/training_loop_prereg_r12.json",
                            "sha256": sha_of(os.path.join(HERE, "training_loop_prereg_r12.json"))},
        "round_12_report": {"path": "reports/training_round_r12.json",
                            "sha256": sha_of(os.path.join(HERE, "reports", "training_round_r12.json"))},
        "round_12_negative": {"path": "reports/negative_result_r12.json",
                              "sha256": sha_of(os.path.join(HERE, "reports", "negative_result_r12.json"))},
        "contract_margin_eval": {"path": "reports/contract_margin_eval.json",
                                 "sha256": sha_of(os.path.join(HERE, "reports", "contract_margin_eval.json"))},
        "ref_energy_t45": {"path": "reports/train_map_reference_energy_t45.json",
                           "sha256": sha_of(os.path.join(HERE, "reports", "train_map_reference_energy_t45.json"))},
        "wallclock_amend": {"path": "config/wallclock_amend_v1.json",
                            "sha256": sha_of(os.path.join(HERE, "config", "wallclock_amend_v1.json"))},
    }

    d["environment"] = dict(d["environment"])
    d["environment"]["t_max_s"] = 45.0
    d["environment"]["t_max_s_note"] = (
        "r12 为 30.0。改到 45.0 的依据：v_deadline=L/45=0.311 m/s 恰等于能耗最省速度 "
        "v*=0.3099（NAV-059），使「省能」与「按时」不再互相排斥；30 s 时 v_deadline=0.465-0.571，"
        "比 v* 高 50%%，每米能耗被迫高出约 7%%，而预算正是按该速度算的 → 零余量。"
    )

    d["reference_energy"] = {
        "report": "reports/train_map_reference_energy_t45.json",
        "budget_k": 1.15,
        "form": "z0 = budget_k × E_ref(T=45)，E_ref 仍是『参考路线按 v=L/45 跑完』的能耗",
        "why_k": "E_ref 按定义就是最小可行能耗，任何路径/转向误差都会超支；实测最佳策略的"
                 "每米能耗只比解析模型高 3-4%%，但路径长 3-5%%，故需 10-15%% 余量才可测",
        "saturation_note": "full 变体叠加 consistent 口径 (1+α)=1.19 后，最大图 z0≈579 J > Z_REF=550，"
                           "比值 1.05 仍在观测截断 1.5 之内，不产生编码饱和；不调整 Z_REF",
        "unchanged_from": "r12（r12 用 reports/train_map_reference_energy.json，k=1.0）",
    }

    d["objective_signal_fix"] = dict(d["objective_signal_fix"])
    d["objective_signal_fix"]["joint_cost_base_weight"] = float(base_weight)
    d["objective_signal_fix"]["joint_cost_base_weight_note"] = (
        "预算内（未超支）逐步成本的权重。实测依据：预算抬高 15%% 后策略把额度直接花掉，"
        "因为未超支步被 cost_on_overspend_only 整体遮蔽；w=50 与 base=2 的比例按"
        "『超支后每步成本仍显著重于预算内』设定（2/50=4%%）。默认 0.0 = 与 r12 逐位等价。"
    )

    d["hypothesis"] = (
        "两个病分开测。①契约：把时限与预算都放到有余量（T=45 s、k=1.15、E_ref 重算）后，"
        "已经会开的策略应当能稳定到达——实测证据：同样契约为 r12 uncal_s0_nomask 拿到 6/6 到达"
        "（4/6 预算内、零危险步），说明 30 s 与零余量预算截断了一批本来能到的回合。"
        "②省能：在余量契约下预算被直接花掉（r11 uncal_s0 的 L2 从 92%% 涨到 114-116%%、"
        "L3 从 110%% 涨到 123%%），因为未超支步的成本项被遮蔽。故给 base_weight=2.0 后，"
        "预期能耗/预算比中位回落到 ≤1.00，且 sr_feasible 中位上升。"
        "替代解释（须被排除）：能耗或 SR 的变化只是『预算变大』的机械结果——由 B 臂（同契约、"
        "base=0）承担对照，A 与 B 的差异才是成本项的净贡献。"
    )

    d["falsification"] = (
        "满足任一即判本轮假设不成立："
        "(a) A 臂与 B 臂的 sr_feasible 中位差 < 0.05，且能耗/预算比中位差 < 0.05"
        "（成本项既不改善可行性也不改善省能，判『预算内计成本无效』）；"
        "(b) A 臂能耗/预算比中位 > 1.10（余量 15%% 仍被吃光）；"
        "(c) 两方法 sr_feasible 中位 < 0.10 且 A 臂到达数不优于 r12 的 2/6。"
        "触发即停，写正式负结果；不改判据、不放宽阈值、不追加轮次（除用户另行授权）。"
    )

    d["method"] = {
        "variants": ["uncalibrated", "full"],
        "seeds": [0, 1, 2],
        "runs": 12,
        "arms": {
            "A": {"suffix": "", "joint_cost_base_weight": float(base_weight), "runs": 6},
            "B": {"suffix": "_b0", "joint_cost_base_weight": 0.0, "runs": 6,
                  "purpose": "同契约下关掉成本项的对照臂，用来分离『契约变大』与『成本项』两个效应"},
        },
        "dead_step_mask": "继承 r12（enabled=true）；本轮不重新讨论",
        "steering_prior": "继承 r11/r12（gain=0.2）",
        "cost_curriculum": "继承 r11/r12（warmup 150、w=50、超支遮蔽仍开）",
    }

    d["round_acceptance"] = {
        "primary_metric": "sr_feasible = 到达 ∧ 全程 safety_y=0 ∧ z_end>=0",
        "closing_tier_source": "development 6 图 × 3 种子 × k=1.15（T=45 s），确定性均值动作",
        "sr_feasible_median": 0.10,
        "energy_ratio_median": 1.00,
        "danger_steps_median_full": 0,
        "note": "sr_feasible 是本轮主判据；历史 sr（label==goal_reached）与 sr_within_budget 并列报告，"
                "便于与 r1-r12 对照，但不得单独引用 sr 报喜",
        "budget_response_on_dev": {"grid_k": [1.0, 1.15, 1.3],
                                   "purpose": "诊断：余量继续增大时到达与能耗如何变化"},
        "ideal_tier": {"sr_feasible": 0.9, "collision_rate": 0.0, "energy_ratio": 1.0},
    }

    d["resource"] = {
        "per_round_wall_clock_cap_h": None,
        "total_wall_clock_cap_h": None,
        "wallclock_policy": "quality_first_no_wallclock_cap（config/wallclock_amend_v1.json）",
        "max_rounds": 4,
        "runs_this_round": 12,
        "used_so_far": {"rounds": 12, "calendar_hours": 18.21},
        "runs_used_total": 76,
        "note": "12 run（2 臂 × 2 方法 × 3 种子）；墙钟不设上限，但必须跑满 600 updates；"
                "实测 r12 单 run 2.6-2.7 h（8 run 并行），本轮 12 run 预计更久，不作停止条件。",
    }

    d["stops"] = {
        "met": "立刻停止，进入 NAV-065 候选冻结",
        "wall_clock": "不适用：墙钟上限已由 config/wallclock_amend_v1.json 取消",
        "infrastructure": "换 run id 重跑，与原 attempt 一并报告",
        "falsified": "写正式负结果并结案；不追加第 14 轮（除用户另行授权）",
        "semantic_doubt": "若 sr_feasible 上升但能耗比未改善，或反之，须分项如实报告，不得择一报喜",
    }

    d["reporting"] = (
        "reports/training_round_r13.json + 台账；须与 r12 与契约余量评估并列："
        "sr_feasible、sr、能耗/预算比、危险步、到达用时、A/B 臂差、无效样本率、"
        "并注明契约余量版不可与 r1-r12 直接比较"
    )

    d["contract_variant"] = {
        "is_variant": True,
        "differs_from_r12": ["environment.t_max_s 30.0 -> 45.0",
                             "reference_energy.report -> train_map_reference_energy_t45.json",
                             "budget_k 1.0 -> 1.15",
                             "objective_signal_fix.joint_cost_base_weight 0.0 -> %.1f" % base_weight],
        "comparability": "不可与 r1-r12 的 SR/能耗直接比较；历史报告保持原样，本轮另立文件",
        "why_now": "S1 实测：到达需额外 3.4-6.2 s、能耗超预算 5-12%%；零余量判据使"
                   "『到达且预算内』成为单点可行域，无法区分『学不会』与『测不出』",
    }

    d["amendments"] = [{
        "version": version,
        "date": "2026-11-15T13:20:00+08:00",
        "old_sha256": sha_of(TEMPLATE),
        "trigger": "r12 证伪后 S0/S1 实测确认零余量契约；用户 13:13 授权继续推进",
        "change": "契约余量（T=45 s / k=1.15 / E_ref 重算）+ 预算内成本项（arms A/B）",
        "unchanged": "dead_step_mask=true、转向先验 0.2、成本课程 150/50、超支遮蔽、fresh init、"
                     "600 updates、算法超参、dev 划分与地图",
    }]

    body = json.dumps(d, ensure_ascii=False, indent=2) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    with open(path + ".sha256", "w", encoding="utf-8", newline="\n") as f:
        f.write(sha + "\n")
    return sha, d


def main():
    a_sha, a = build(2.0, os.path.join(HERE, "training_loop_prereg_r13.json"),
                     "r13.0.0", "frozen_before_round_13", "A 臂 预算内计成本")
    b_sha, b = build(0.0, os.path.join(HERE, "training_loop_prereg_r13_b0.json"),
                     "r13.0.0b", "frozen_before_round_13", "B 臂 对照")
    # 守卫：两臂之间只有 base_weight 不同。显式列出『按设计允许不同』的键，
    # 其余键必须逐字段相同——否则 A/B 对照就不再是单变量对比。
    a2, b2 = dict(a), dict(b)
    a2["objective_signal_fix"] = dict(a2["objective_signal_fix"])
    b2["objective_signal_fix"] = dict(b2["objective_signal_fix"])
    a2["objective_signal_fix"].pop("joint_cost_base_weight")
    b2["objective_signal_fix"].pop("joint_cost_base_weight")
    allowed = ("version", "status", "task", "amendments", "authorization",
               "contract_variant", "method")
    for k in allowed:
        a2.pop(k, None); b2.pop(k, None)
    diff = [k for k in set(a2) | set(b2) if a2.get(k) != b2.get(k)]
    assert not diff, "A/B 两臂除 base_weight 外出现差异：%s" % diff
    assert a["environment"]["t_max_s"] == 45.0 and a["reference_energy"]["budget_k"] == 1.15
    assert a["objective_signal_fix"]["joint_cost_base_weight"] == 2.0
    assert b["objective_signal_fix"]["joint_cost_base_weight"] == 0.0
    assert a["resource"]["per_round_wall_clock_cap_h"] is None
    print("A arm sha=%s\nB arm sha=%s" % (a_sha, b_sha))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
