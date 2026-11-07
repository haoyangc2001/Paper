"""HIRA-037：HiRA 与旧 v2 在统一口径下的对比（报告级）。

**这一节的能力边界，先说清楚**：

v2 的逐 trial 原始数据（68 GB trials）**已不在本机**——`/pub/data/caohy/logs/camra` 只剩
3.1 GB 的 TASK 日志，NFS 家目录与 `.local` 备份也没有。所以本脚本**不能复算 v2**，
也就做不了成对差异与簇级区间。它做的是：

* 把 v2 的**冻结产物**（margins / power model / footprint，仓库里可直接读）与
  **交接笔记里的汇总数字**（带节号与行号）摆到同一张表上；
* 逐项标注**来源、分母、单位、可比性**；
* 明确声明「报告级、未复算、非配对」，不把汇总数字之差说成因果。

计划的 `constraints_and_notes` 允许这条路径，但要求"不伪装成已复算"——本报告每行都有出处，
且 `recomputed=false` 写在 JSON 顶层。

    python tests/compare_hira_v2.py
"""

from __future__ import annotations

import json
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
# v2 的冻结物只在 git 仓库里（/pub 工作副本只同步 adapters/src/实验目录），用绝对路径读
V2_CONFIG = Path("/home/caohy/repositories/Paper/CaMRA/calibration/config")
CAL = REPO / "CaMRA" / "calibration"

#: v2 的汇总数字只存在于交接笔记里，这里逐条带出处（文件 + 小节/行号），便于复核
V2_HANDOFF = "CaMRA/calibration/docs/implementation_handoff.md"


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    v2_margins = load(V2_CONFIG / "margins_v1.json")
    v2_power = load(V2_CONFIG / "power_model_v1.json")
    v2_fp = load(V2_CONFIG / "footprint_v1.json")["polygon"]

    h_margins = load(EXP / "config/margins_hira_v1.json")
    h_power = load(EXP / "config/power_model_hira_v1.json")
    h_fp = load(EXP / "config/footprint_hira_v1.json")["polygon"]

    h_heldout = load(EXP / "reports/heldout_coverage.json")
    h_near = load(EXP / "reports/near_obstacle_evaluation.json")
    h_cal = load(EXP / "reports/calibration_collection_integrity.json")

    def rate_of(report: dict, key: str) -> dict:
        b = report["full_sample"]["overall"][key]
        return {"rate": b["rate"], "n_covered": b.get("n_covered"), "n": b.get("n"),
                "ci_lo": (b.get("ci_cluster") or {}).get("lo"),
                "ci_hi": (b.get("ci_cluster") or {}).get("hi")}

    # ---------------- 配置可比性矩阵 ----------------
    config = [
        {"dimension": "命令域", "v2": "vx ±1.0 / vy ±0.4 / wz ±1.0 m/s,rad/s",
         "hira": "同（逐字段继承协议）", "comparable": "same",
         "note": "协议 v1.0.2 的命令域被 HiRA 协议逐字段继承"},
        {"dimension": "命令程序生成器", "v2": "camra_calibration.commands.build_program",
         "hira": "同一个函数（清单逐条按 program_sha256 重建对账）", "comparable": "same",
         "note": "同 seed + 同生成器 = 同命令表，这是跨控制器匹配的前提"},
        {"dimension": "五个 profile", "v2": "clean/friction_only/mass_only/push_only/combined",
         "hira": "同五档，旋钮范围继承协议", "comparable": "same",
         "note": "profile 采样由 seed 确定性生成"},
        {"dimension": "机器人资产", "v2": "unitree_ros go2_description → USD",
         "hira": "legged_gym go2.urdf → Isaac Lab 导入（29 links / 27 colliders）",
         "comparable": "different",
         "note": "外形同为 Go2，但不是同一份资产；质量/惯量/碰撞体不同（§12、§14）"},
        {"dimension": "执行器与 PD", "v2": "kp=25.0, kd=0.5, action_scale=0.25",
         "hira": "kp=20.0, kd=0.5, action_scale=0.25（源 Go2 配置值）", "comparable": "different",
         "note": "kp 差 25%（沿用各自来源），这是与网络权重并列的第二个差异源"},
        {"dimension": "力矩限幅", "v2": "见 v2 冻结包 deploy.yaml",
         "hira": "逐关节 URDF effort：hip/thigh 23.7、calf 35.55 N·m", "comparable": "different",
         "note": "限幅不同会改变饱和行为"},
        {"dimension": "时序", "v2": "物理 200 Hz / 策略 50 Hz / 命令 10 Hz",
         "hira": "同（且逐项实测：6000/1500/300/301）", "comparable": "same",
         "note": "HiRA 侧用实测计数而非读配置（§18）"},
        {"dimension": "能量口径", "v2": "applied_torque × 子步末速度，逐关节截断，float32 乘法",
         "hira": "同公式，float64 乘法", "comparable": "same_with_note",
         "note": "跨控制器比 J 时须注明 float32 vs float64（系统差 ~2.6e-10 相对，§25）"},
        {"dimension": "协议版本", "v2": "calibration_protocol v1.0.2",
         "hira": "calibration_protocol_hira_v1（继承 v1.0.2，差异 9 项全为身份/簿记）",
         "comparable": "same_with_note",
         "note": "统计口径零变更；差异见 reports/protocol_diff_from_v2.json"},
        {"dimension": "重试口径", "v2": "含 D-007（baseline_validate +31 个 __a2）与 D-013（near +2）两次整批重采",
         "hira": "1820 条全部单 attempt，零重采、零 resume", "comparable": "different",
         "note": "v2 的 heldout/near 分母是「按最大 attempt 去重后」的 397/197；HiRA 是原始 completed 377/188。**这是最重要的分母差异**"},
    ]

    # ---------------- 指标对比 ----------------
    metrics = []

    metrics.append({
        "metric": "α（成本裕度）", "unit": "ratio",
        "v2": {"value": v2_margins["alpha"],
               "ci": v2_margins.get("alpha_interval"),
               "denominator": "790 completed",
               "source": f"{V2_HANDOFF} §28 / config/margins_v1.json"},
        "hira": {"value": h_margins["alpha"], "ci": h_margins.get("alpha_interval"),
                 "denominator": "771 completed",
                 "source": "config/margins_hira_v1.json（HIRA-031）"},
        "comparable": "same_definition", "recomputed": False,
        "note": f"同为 Q0.90(ε_c) nearest-rank；HiRA 是 v2 的 {h_margins['alpha']/v2_margins['alpha']:.2f}×。"
                "不同控制器 + 不同资产 + 不同 kp，不能归因于单一变量",
    })
    metrics.append({
        "metric": "δ（几何裕度）", "unit": "m",
        "v2": {"value": v2_margins["delta"], "ci": v2_margins.get("delta_interval"),
               "denominator": "同上", "source": f"{V2_HANDOFF} §28 / config/margins_v1.json"},
        "hira": {"value": h_margins["delta"], "ci": h_margins.get("delta_interval"),
                 "denominator": "同上", "source": "config/margins_hira_v1.json（HIRA-031）"},
        "comparable": "same_definition", "recomputed": False,
        "note": f"HiRA 是 v2 的 {h_margins['delta']/v2_margins['delta']:.2f}×。膨胀主要来自扰动档"
                "（HiRA per-profile δ q90：clean 1.85 → combined 12.50）",
    })

    hh = {k: rate_of(h_heldout, k) for k in ("cost", "position", "joint")}
    v2_heldout = {
        "cost": {"rate": 0.8690, "n_covered": 345, "n": 397, "ci": [0.8363, 0.9018]},
        "position": {"rate": 0.9144, "n_covered": 363, "n": 397, "ci": [0.8866, 0.9421]},
        "joint": {"rate": 0.7935, "n_covered": 315, "n": 397, "ci": [0.7531, 0.8338]},
    }
    for key, label in (("cost", "heldout 成本覆盖"), ("position", "heldout 位置覆盖"),
                       ("joint", "heldout 联合覆盖")):
        metrics.append({
            "metric": label, "unit": "覆盖率",
            "v2": {"value": v2_heldout[key]["rate"],
                   "ci": v2_heldout[key]["ci"],
                   "denominator": f"{v2_heldout[key]['n_covered']}/{v2_heldout[key]['n']}，按最大 attempt 去重",
                   "source": f"{V2_HANDOFF} §27 表（行 1819–1821）"},
            "hira": {"value": hh[key]["rate"], "ci": [hh[key]["ci_lo"], hh[key]["ci_hi"]],
                     "denominator": f"{hh[key]['n_covered']}/{hh[key]['n']}，completed 且可算",
                     "source": "reports/heldout_coverage.json（HIRA-033）"},
            "comparable": "different_denominator", "recomputed": False,
            "note": "分母不同（397 vs 377）且 v2 含历史重采；只能作描述性比较，不能当同一批样本的配对差异",
        })

    hn = {k: rate_of(h_near, k) for k in ("cost", "position", "joint")}
    v2_near = {
        "cost": {"rate": 0.8883, "n_covered": 175, "n": 197},
        "position": {"rate": 0.9340, "n_covered": 184, "n": 197},
        "joint": {"rate": 0.8274, "n_covered": 163, "n": 197},
    }
    for key, label in (("cost", "near 成本覆盖"), ("position", "near 位置覆盖"),
                       ("joint", "near 联合覆盖")):
        metrics.append({
            "metric": label, "unit": "覆盖率",
            "v2": {"value": v2_near[key]["rate"], "ci": None,
                   "denominator": f"{v2_near[key]['n_covered']}/{v2_near[key]['n']}，补采后按最大 attempt 计",
                   "source": f"{V2_HANDOFF} §32.4（行 2063–2065）"},
            "hira": {"value": hn[key]["rate"], "ci": [hn[key]["ci_lo"], hn[key]["ci_hi"]],
                     "denominator": f"{hn[key]['n_covered']}/{hn[key]['n']}，completed 且可算",
                     "source": "reports/near_obstacle_evaluation.json（HIRA-035）"},
            "comparable": "different_denominator", "recomputed": False,
            "note": "同上：分母 197 vs 188，且 v2 含 2 条补采",
        })

    metrics.append({
        "metric": "near 几何重叠", "unit": "条 / 百分比",
        "v2": {"value": 138 / 197, "raw": "138/197",
               "denominator": "197，含 2 条补采",
               "source": f"{V2_HANDOFF} §32.4 与 §31.1"},
        "hira": {"value": 151 / 200, "raw": "151/200", "denominator": "200，completed+bf",
                 "source": "reports/near_obstacle_evaluation.json（HIRA-035）"},
        "comparable": "same_method_different_denominator", "recomputed": False,
        "note": "同一判据（足迹与障碍几何重叠）：v2 70.1%，HiRA 75.5%，同一量级",
    })
    metrics.append({
        "metric": "near 越界", "unit": "条",
        "v2": {"value": 13, "denominator": "197 可算", "source": f"{V2_HANDOFF} §32.4"},
        "hira": {"value": 15, "denominator": "200", "source": "reports/near_obstacle_evaluation.json"},
        "comparable": "same_method_different_denominator", "recomputed": False,
        "note": "场地边界墙两边都未建模，均为几何余量判定",
    })

    metrics.append({
        "metric": "功率模型 validate q90", "unit": "相对误差",
        "v2": {"value": 0.0554, "denominator": "n=79",
               "source": f"{V2_HANDOFF} 行 1259"},
        "hira": {"value": 0.0588, "denominator": "n=80",
                 "source": "config/power_model_hira_v1.json（HIRA-026，一次性 validate）"},
        "comparable": "same_definition", "recomputed": False,
        "note": "各自独立拟合、独立验证；两套模型不可混用，J_nom 不能跨控制器互换",
    })

    metrics.append({
        "metric": "足迹面积 / 基宽", "unit": "m² / m",
        "v2": {"value": v2_fp["area"], "min_width": v2_fp["min_width"],
               "n_vertices": len(v2_fp["vertices"]),
               "denominator": "clean 轨迹重建", "source": "CaMRA/calibration/config/footprint_v1.json"},
        "hira": {"value": h_fp["area"], "min_width": h_fp["min_width"],
                 "n_vertices": len(h_fp["vertices"]),
                 "denominator": "623,757 帧 / 420 条", "source": "config/footprint_hira_v1.json（HIRA-027）"},
        "comparable": "different_asset", "recomputed": False,
        "note": f"HiRA 面积为 v2 的 {h_fp['area']/v2_fp['area']:.2f}×、基宽 {h_fp['min_width']/v2_fp['min_width']:.2f}×；"
                "资产不同，轮廓不可互用",
    })

    metrics.append({
        "metric": "校准完整率（800 条）", "unit": "completed/quota",
        "v2": {"value": 790 / 800, "raw": "790/800", "denominator": "800",
               "source": f"{V2_HANDOFF} 行 1757"},
        "hira": {"value": (h_cal["counts"]["completed"] / 800), "raw": "771/800",
                 "denominator": "800", "source": "reports/calibration_collection_integrity.json"},
        "comparable": "same_definition", "recomputed": False,
        "note": "v2 有 10 条 terminated、HiRA 有 29 条业务失败（其中 160 条噪声事故已零删除隔离后重采）",
    })

    metrics.append({
        "metric": "格级完整率缺口", "unit": "格数",
        "v2": {"value": 0, "raw": "20/20 ≥ 0.9（补采后）",
               "source": f"{V2_HANDOFF} §32.4（cells_below_share = {{}}）"},
        "hira": {"value": 7, "raw": "calibration 2 + heldout 3 + near 2",
                 "source": "reports/{calibration_collection_integrity,heldout_collection_integrity,"
                           "near_obstacle_collection_integrity}.json"},
        "comparable": "different_retry_policy", "recomputed": False,
        "note": "v2 达到 20/20 靠的是 D-013 一次性补采（历史特例，§9.3）；HiRA 按新契约不补采，"
                "缺口如实留存。**这不是能力差异，是重试政策差异**，不能直接比",
    })

    summary = [
        "α 0.1905 → 0.3516（1.85×）、δ 2.6139 → 9.0322 m（3.46×）：HiRA 在 30 s 开环、受扰动时漂得更远。",
        "heldout：成本 86.90% → 90.98%（+4.08 pp），位置 91.44% → 89.66%（−1.78 pp），"
        "联合 79.35% → 81.70%（+2.35 pp）。成本变好、位置变差，方向不一致，**不能概括为「更好」或「更差」**。",
        "near：成本 88.83% → 93.62%、位置 93.40% → 98.40%、联合 82.74% → 92.02%，"
        "但这两组 δ 差 3.46 倍，覆盖率被更大的 δ 抬高了——**δ 越大越容易「覆盖」，这不是性能改善**。",
        "near 几何：重叠 70.1% → 75.5%、越界 13 → 15，同一量级。两边都说明 δ 是必要条件而非充分条件。",
        "格级缺口：v2 20/20（靠一次性补采）vs HiRA 17/20（不补采）。这是重试政策差异，不是能力差异。",
    ]

    caveats = [
        "**报告级，未复算**：v2 的逐 trial 原始数据已不在本机（/pub 只剩 3.1 GB TASK 日志），"
        "本报告不做任何 v2 复算，全部 v2 数字来自其冻结产物与交接笔记（逐行标注出处）。",
        "**非配对**：只能比汇总数字，做不了「同 seed 同命令」的成对差异，也无法给簇级区间。"
        "清单层面 baseline/heldout/near 与 v2 有 100% 模板匹配（§26），但缺 v2 的逐 trial 轨迹与扰动 trace，配对检验无从做起。",
        "**多重差异源**：控制器包同时更换了权重、默认姿态、执行器增益（kp 25→20）与机器人资产。"
        "任何指标差异都不能归因于网络权重一项。",
        "**分母口径**：v2 的 397 / 197 是「按最大 attempt 去重后」，含历史重采；HiRA 的 377 / 188 是原始 completed。"
        "引用时必须带分母。",
        "**δ 不可横比覆盖**：δ 从 2.61 涨到 9.03 会自动抬高 near 的「位置覆盖」，"
        "把 93.40% → 98.40% 当性能提升是错的（§37 已证明 98.40% 覆盖下仍有 75.5% 几何重叠）。",
        "**旧报告未改**：v2 的交接笔记、计划、锁、报告一字未动；本报告只读引用。",
    ]

    payload = {
        "report": "hira_vs_v2_comparison",
        "task": "HIRA-037",
        "recomputed": False,
        "evidence_level": "report_level_only",
        "why_not_recomputed": (
            "v2 逐 trial 原始数据不在本机：/pub/data/caohy/logs/camra 仅 3.1 GB（TASK 日志），"
            "无 trials/ 目录；/pub/data/caohy/camra_run 不存在；NFS 家目录 logs 仅 233 MB；"
            "Paper/.local 备份不存在。计划的 constraints_and_notes 要求此情形必须显式声明、不得伪装成已复算。"
        ),
        "v2_sources": {
            "frozen_artifacts": ["CaMRA/calibration/config/margins_v1.json",
                                 "CaMRA/calibration/config/power_model_v1.json",
                                 "CaMRA/calibration/config/footprint_v1.json"],
            "report_level": f"{V2_HANDOFF} §27（heldout）、§32.4（near 重算）、§28（α/δ）、行 1259（功率 validate）",
        },
        "hira_sources": ["config/margins_hira_v1.json", "config/power_model_hira_v1.json",
                         "config/footprint_hira_v1.json", "reports/heldout_coverage.json",
                         "reports/near_obstacle_evaluation.json",
                         "reports/calibration_collection_integrity.json",
                         "frozen/hira_go2_18500/controller.json"],
        "config_comparability": config,
        "metrics": metrics,
        "summary": summary,
        "caveats": caveats,
        "correction": (
            "更正 HIRA-033 的 remaining_work 引用错误：该处写「v2 的 CAL-027（cost 0.8883 / position 0.9340 / "
            "joint 0.8274，n=197）」有误。0.8883/0.9340/0.8274 是 v2 **近障**（§32.4，n=197 补采后）的数字；"
            "v2 的 CAL-027（heldout）实为 0.8690 / 0.9144 / 0.7935（n=397，§27）。本报告按后者引用。"
        ),
    }

    out = EXP / "reports/hira_vs_v2_comparison.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    # ---------------- Markdown ----------------
    L: list[str] = []
    A = L.append
    A("# HiRA 与 v2 统一口径对比（HIRA-037）")
    A("")
    A("> **证据等级：报告级（未复算、非配对）**。v2 的逐 trial 原始数据已不在本机，")
    A("> 本报告不做任何 v2 复算；v2 数字全部来自其冻结产物与交接笔记，逐行标注出处。")
    A("")
    A("## 1. 配置可比性")
    A("")
    A("| 维度 | v2 | HiRA | 可比性 | 说明 |")
    A("|---|---|---|---|---|")
    tag = {"same": "可比", "same_with_note": "可比（须注明）", "different": "**不可比**",
           "different_asset": "**不可比**"}
    COMP_LABEL = {
        "same_definition": "同口径",
        "different_denominator": "**分母不同**",
        "same_method_different_denominator": "同判据、**分母不同**",
        "different_asset": "**资产不同**",
        "different_retry_policy": "**重试政策不同**",
    }

    def fmt_val(v) -> str:
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, (int, float)):
            fv = float(v)
            return str(int(fv)) if fv.is_integer() else f"{fv:.4f}"
        return str(v)
    for c in config:
        A(f"| {c['dimension']} | {c['v2']} | {c['hira']} | {tag.get(c['comparable'], c['comparable'])} | {c['note']} |")
    A("")
    A("## 2. 指标对比")
    A("")
    A("| 指标 | v2（分母） | HiRA（分母） | 单位 | 可比性 |")
    A("|---|---|---|---|---|")
    for m in metrics:
        v2v = m["v2"].get("raw") or fmt_val(m["v2"]["value"])
        hiv = m["hira"].get("raw") or fmt_val(m["hira"]["value"])
        v2d = m["v2"].get("denominator", "—")
        hid = m["hira"].get("denominator", "—")
        A(f"| {m['metric']} | {v2v}（{v2d}） | {hiv}（{hid}） | "
          f"{m['unit']} | {COMP_LABEL.get(m['comparable'], m['comparable'])} |")
    A("")
    A("### 逐项注释与出处")
    A("")
    for m in metrics:
        A(f"- **{m['metric']}**：{m['note']}")
        A(f"  - v2 来源：`{m['v2']['source']}`")
        A(f"  - HiRA 来源：`{m['hira']['source']}`")
    A("")
    A("## 3. 结论（描述性，不作因果）")
    A("")
    for s in summary:
        A(f"- {s}")
    A("")
    A("## 4. 边界与不可比项")
    A("")
    for c in caveats:
        A(f"- {c}")
    A("")
    A("## 5. 一处更正")
    A("")
    A(payload["correction"])
    A("")
    (EXP / "reports/hira_vs_v2_comparison.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print(json.dumps({
        "n_config_dims": len(config), "n_metrics": len(metrics),
        "alpha_ratio": round(h_margins["alpha"] / v2_margins["alpha"], 3),
        "delta_ratio": round(h_margins["delta"] / v2_margins["delta"], 3),
        "heldout_cost": [0.8690, round(hh["cost"]["rate"], 4)],
        "heldout_position": [0.9144, round(hh["position"]["rate"], 4)],
        "heldout_joint": [0.7935, round(hh["joint"]["rate"], 4)],
        "near_overlap": ["138/197", "151/200"],
        "recomputed": False,
    }, ensure_ascii=False, indent=1))
    print("COMPARE_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
