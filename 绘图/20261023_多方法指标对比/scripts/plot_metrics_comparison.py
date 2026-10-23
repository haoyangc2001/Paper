"""Validate manually specified assumptions and render the comparison figure.

This script never generates point estimates, interval endpoints, or trial records.
The only data transformations are unit checks, xerr lengths, and scene averages.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import runpy
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_and_check(data_dir: Path, partial: bool = False):
    protocol = json.loads((data_dir / "protocol.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (data_dir / "assumed_intervals.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    methods = {item["id"] for item in protocol["methods"]}
    scenarios = {item["id"]: index for index, item in enumerate(protocol["scenarios"])}
    metrics = {item["id"]: item for item in protocol["metrics"]}
    source_path = REPO / protocol["source_script"]
    original = runpy.run_path(str(source_path), run_name="read_only_plot_audit")["MAIN_DATA"]
    empirical = protocol["data_status"] == "empirical"
    if not empirical and (protocol["data_status"] != "hypothetical" or protocol["confidence_level"] is not None):
        raise ValueError("Hypothetical ranges must not claim an empirical confidence level.")
    if empirical:
        if protocol.get("verify_legacy_centers", True):
            raise ValueError("Measured data must not be constrained to the old hypothetical centers.")
        if not protocol.get("empirical_sources") or not protocol.get("sample_count_verified"):
            raise ValueError("Measured data require source records and verified sampling information.")
        if protocol["interval_kind"] == "hypothetical_estimate_range":
            raise ValueError("Replace the assumed interval definition with the actual estimator.")
        if not protocol.get("empirical_caption_en") or not protocol.get("empirical_caption_zh"):
            raise ValueError("Supply measured-data captions describing sampling, interval estimator, budgets, and costs.")
        confidence = protocol.get("confidence_level")
        if confidence is not None and not 0 < confidence < 1:
            raise ValueError("Confidence level must be a fraction between 0 and 1, or null for non-CI intervals.")
    lookup = {}
    for number, row in enumerate(rows, 1):
        if row["id"] != f"P{number:03d}":
            raise ValueError(f"Unexpected point ID/order: {row['id']}")
        method, scenario, metric = row["method"], row["scenario"], row["metric"]
        if method not in methods or scenario not in scenarios or metric not in metrics:
            raise ValueError(f"Unknown category: {row['id']}")
        key = method, scenario, metric
        if key in lookup:
            raise ValueError(f"Duplicate point: {key}")
        values = [row[name] for name in ("lower", "estimate", "upper")]
        if any(not isinstance(value, (float, int)) or not math.isfinite(value) for value in values):
            raise ValueError(f"Invalid numeric value: {key}")
        lower, center, upper = values
        if not 0 <= lower <= center <= upper or lower == upper:
            raise ValueError(f"Invalid interval: {key}")
        if metric != "Cost" and upper > 100:
            raise ValueError(f"Rate outside probability range: {key}")
        axis_min, axis_max = metrics[metric]["limits"]
        if lower < axis_min or upper > axis_max:
            raise ValueError(f"Interval would be clipped by axis: {key}")
        definition = metrics[metric]
        source_value = original[definition["source_label"]][method][scenarios[scenario]] * definition["source_scale"]
        if protocol.get("verify_legacy_centers", True) and not math.isclose(center, source_value, abs_tol=1e-8):
            raise ValueError(f"Unrecorded center change: {key}, {source_value} -> {center}")
        if len(row.get("rationale", "")) < 25:
            raise ValueError(f"Point-specific rationale missing: {key}")
        lookup[key] = row

    groups = {(row["method"], row["scenario"]) for row in rows}
    remaining = {}
    for method, scenario in sorted(groups):
        if {metric for m, s, metric in lookup if (m, s) == (method, scenario)} != set(metrics):
            raise ValueError(f"Incomplete four-metric group: {method}, {scenario}")
        success = lookup[method, scenario, "SR"]["estimate"]
        collision = lookup[method, scenario, "CR"]["estimate"]
        if success + collision > 100 + 1e-8:
            raise ValueError(f"Mutually exclusive success/collision exceed 100%: {method}, {scenario}")
        remaining[f"{method}/{scenario}"] = round(100 - success - collision, 4)
    expected = len(methods) * len(scenarios) * len(metrics)
    if not partial and len(rows) != expected:
        raise ValueError(f"Expected {expected} points, received {len(rows)}")

    overlaps = []
    for scenario in scenarios:
        for metric in metrics:
            a = lookup.get(("HD-MCRA", scenario, metric))
            b = lookup.get(("RC-PPO", scenario, metric))
            if a and b:
                overlaps.append({"scenario": scenario, "metric": metric,
                                 "ranges_overlap": max(a["lower"], b["lower"]) <= min(a["upper"], b["upper"])})
    report = {
        "status": "passed", "data_status": protocol["data_status"],
        "interval_kind": protocol["interval_kind"], "confidence_level": protocol["confidence_level"],
        "point_count": len(rows), "expected_full_count": expected,
        "scene_point_counts": dict(Counter(row["scenario"] for row in rows)),
        "four_metric_groups_checked": len(groups), "source_centers_match": True if protocol.get("verify_legacy_centers", True) else "not_required_for_replacement",
        "intervals_inside_axes": True, "remaining_non_success_non_collision_percent": remaining,
        "main_comparison_overlap": overlaps,
        "overlap_interpretation": "Descriptive checks only; interval overlap is not a significance test.",
        "physical_or_statistical_validation": False,
        "hashes": {"protocol": sha256(data_dir / "protocol.json"),
                   "points": sha256(data_dir / "assumed_intervals.jsonl"),
                   "original_generator": sha256(source_path)},
    }
    return protocol, rows, lookup, report


def export_review(protocol, rows, lookup, report, output_dir: Path):
    empirical = protocol["data_status"] == "empirical"
    csv_name = "point_estimates.csv" if empirical else "assumed_point_estimates.csv"
    fields = ["id", "method", "scenario", "metric", "estimate", "lower", "upper", "data_status", "interval_kind", "rationale"]
    with (output_dir / csv_name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "data_status": protocol["data_status"], "interval_kind": protocol["interval_kind"]})

    details = ["# 84 个点的数值、区间与来源说明" if empirical else "# 84 个点的假设范围与逐点理由", "",
               protocol["interval_note"] if empirical else "所有值均为假设数据；区间为人工指定的估计范围，不是实测置信区间、标准差或标准误。",
               "原始权威记录为 data/assumed_intervals.jsonl。本文件由该记录导出，修改应回到原始文件。", ""]
    for scenario in protocol["scenarios"]:
        details += [f"## {scenario['label_zh']}", ""]
        for method in protocol["methods"]:
            details += [f"### {method['label']}", "",
                        "| 点编号 | 指标 | 中心值 | 左端 | 右端 | 逐点理由 |", "| --- | --- | ---: | ---: | ---: | --- |"]
            for metric in protocol["metrics"]:
                row = lookup[method["id"], scenario["id"], metric["id"]]
                details.append(f"| {row['id']} | {metric['label_zh']} ({metric['unit']}) | {row['estimate']:.1f} | {row['lower']:.1f} | {row['upper']:.1f} | {row['rationale']} |")
            details.append("")

    cost_unit = next(metric["unit"] for metric in protocol["metrics"] if metric["id"] == "Cost")
    details += ["## 等权场景均值" if empirical else "## 等权场景均值（均为假设）", "", f"| 方法 | SR (%) | CR (%) | BVR (%) | 成本（{cost_unit}） |", "| --- | ---: | ---: | ---: | ---: |"]
    for method in protocol["methods"]:
        means = [sum(lookup[method["id"], scene["id"], metric["id"]]["estimate"] for scene in protocol["scenarios"]) / 3 for metric in protocol["metrics"]]
        details.append(f"| {method['label']} | " + " | ".join(f"{value:.1f}" for value in means) + " |")
    details += ["", "三个场景在此汇总中等权；不平均区间端点来冒充总体置信区间。", ""]
    if not empirical:
        details += ["RA 的 BVR 假设场景值 12、20、28 的等权均值为 20.0%，替代旧稿的 22.0%。", ""]
    (output_dir / "推演明细.md").write_text("\n".join(details), encoding="utf-8")

    caption = """# 配套图注

## 当前假设数据预览用图注

Scenario-level navigation performance of seven methods. Each row is a method; circles, triangles, and squares denote moderate clutter, narrow passages, and dense obstacles. Panels report success rate (SR), collision rate (CR), budget violation rate (BVR), and execution cost. Blue denotes HD-MCRA, orange RC-PPO, and rose RA. The PPO variants use light and dark purple, and the CPPO variants use light and dark green, with the lighter shade indicating beta L and the darker shade beta H. All point estimates are assumed data. Horizontal bars show manually specified hypothetical estimate ranges for layout review; they are not measured confidence intervals, standard deviations, or standard errors. Cost retains the provisional units of the draft.

七种方法在三个导航场景下的整体表现。每行对应一种方法，圆形、三角形、方形分别对应普通障碍、窄通道和密集障碍。四列展示成功率、碰撞率、预算违规率和执行成本。HD-MCRA为蓝色、RC-PPO为橙色、RA为玫红色；PPO两种设置使用深浅紫色，CPPO两种设置使用深浅绿色，浅色对应beta L、深色对应beta H。所有点估计均为假设数据；横向短线为逐点指定的假设估计范围，用于版式审阅，不是实测置信区间、标准差或标准误。成本沿用草稿单位。

## 使用边界

- 不能根据本图声称显著优于基线、获得某个覆盖概率、满足真实预算或已验证实机安全。
- 区间覆盖的是每个汇总估计的示意范围，不是一次运行的取值范围。各指标区间不构成联合分布。
- 不同场景的三个点不是重复实验样本，不用它们计算误差线。
- 成本必须与到达情况一起解释；在成本定义与失败计费未明确前，不将较低均值单独等同更高效率。
- 正式替换时，全部中心与区间应来自同一明确统计协议下的真实记录，重新确定区间类型、统计单位和置信水平，并重写此图注。
"""
    if empirical:
        caption = "# 配套图注\n\n" + protocol["empirical_caption_en"] + "\n\n" + protocol["empirical_caption_zh"] + "\n"
    (output_dir / "caption.md").write_text(caption, encoding="utf-8")
    (output_dir / "data_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render(protocol, lookup, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    layout = protocol["layout"]
    plt.rcParams.update({"font.family": layout["font_family"], "font.size": layout["font_pt"],
                         "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False})
    fig = plt.figure(figsize=(layout["width_mm"] / 25.4, layout["height_mm"] / 25.4))
    bottom = layout.get("bottom_margin_mm", 19.5) / layout["height_mm"]
    panel_height = 1 - bottom - layout.get("top_margin_mm", 10.0) / layout["height_mm"]
    axes = []
    for index, metric in enumerate(protocol["metrics"]):
        ax = fig.add_axes([0.157 + index * 0.208, bottom, 0.178, panel_height])
        axes.append(ax)
        for method_index, method in enumerate(protocol["methods"]):
            if method_index % 2 == 0:
                ax.axhspan(method_index - 0.46, method_index + 0.46, color="#F4F6F8", linewidth=0, zorder=0)
            for scene in protocol["scenarios"]:
                row = lookup[method["id"], scene["id"], metric["id"]]
                ax.errorbar(row["estimate"], method_index + scene["offset"],
                            xerr=[[row["estimate"] - row["lower"]], [row["upper"] - row["estimate"]]],
                            fmt=scene["marker"], linestyle="none", color=method["color"],
                            ecolor=method["color"], elinewidth=0.8, capsize=1.65, capthick=0.7,
                            markersize=3.8, markeredgewidth=0.45, markeredgecolor="white",
                            zorder=4)
        ax.set_xlim(*metric["limits"])
        ax.set_xticks(metric["ticks"])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}"))
        ax.set_ylim(6.53, -0.53)
        ax.set_yticks(range(7))
        if index == 0:
            display_labels = [method["label"].replace(" (beta L)", r"$_{\beta_L}$").replace(" (beta H)", r"$_{\beta_H}$") for method in protocol["methods"]]
            ax.set_yticklabels(display_labels)
            for tick, method in zip(ax.get_yticklabels(), protocol["methods"]):
                tick.set_color(method["color"])
                if method["id"] in ("HD-MCRA", "RC-PPO"):
                    tick.set_fontweight("bold")
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0, pad=7)
        ax.tick_params(axis="x", length=2.6, width=0.6, color="#7C858D", pad=3)
        ax.grid(axis="x", color="#DCE1E5", linewidth=0.5, zorder=1)
        for side in ("top", "left", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#7C858D")
        ax.spines["bottom"].set_linewidth(0.6)
        direction = r" $\uparrow$" if metric["direction"] == "higher" else (r" $\downarrow$" if metric["direction"] == "lower" else "")
        ax.set_title(f"({chr(97 + index)}) {metric['label']}" + direction, fontsize=8, pad=11, color="#202B35")
        ax.set_xlabel("(%)" if metric["id"] != "Cost" else metric.get("axis_label", metric["unit"]), labelpad=4, color="#47515A")
    handles = [Line2D([], [], linestyle="none", marker=scene["marker"], color="#606870", markersize=4.4, label=scene["label"]) for scene in protocol["scenarios"]]
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.56, 0.025), ncol=3,
               frameon=False, fontsize=8, handletextpad=0.4, columnspacing=1.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    legend_box = legend.get_window_extent(renderer)
    if any(legend_box.overlaps(ax.xaxis.label.get_window_extent(renderer)) for ax in axes):
        plt.close(fig)
        raise ValueError("Legend overlaps an axis label")
    # Measure the closest vertical marker separation at the final print size.
    point_rows = sorted(index + scene["offset"] for index in range(len(protocol["methods"])) for scene in protocol["scenarios"])
    centers_px = [axes[0].transData.transform((0, row))[1] for row in point_rows]
    min_separation_pt = min(abs(a - b) for a, b in zip(centers_px, centers_px[1:])) * 72 / fig.dpi
    marker_gap_pt = min_separation_pt - (3.8 + 0.45)
    if marker_gap_pt < 1.0:
        plt.close(fig)
        raise ValueError(f"Markers are too close vertically: {marker_gap_pt:.2f} pt gap")
    clipped = []
    for text in fig.findobj(match=matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if box.x0 < canvas.x0 - 1 or box.y0 < canvas.y0 - 1 or box.x1 > canvas.x1 + 1 or box.y1 > canvas.y1 + 1:
                clipped.append(text.get_text())
    if clipped:
        plt.close(fig)
        raise ValueError(f"Text outside figure: {clipped}")
    metadata = {"Title": "Scenario-level navigation comparison - assumed data preview",
                "Subject": "Hypothetical point estimates and manually specified ranges; not experimental evidence or empirical confidence intervals.",
                "Author": "Paper figure workspace", "Keywords": "assumed data; hypothetical intervals; layout preview",
                "CreationDate": datetime(2026, 10, 23, tzinfo=timezone.utc),
                "ModDate": datetime(2026, 10, 23, tzinfo=timezone.utc)}
    empirical = protocol["data_status"] == "empirical"
    if empirical:
        metadata.update({"Title": "Scenario-level navigation comparison", "Subject": protocol["interval_note"],
                         "Keywords": protocol["interval_kind"], "CreationDate": None, "ModDate": None})
    basename = "metrics_comparison" if empirical else "metrics_comparison_assumed"
    fig.savefig(output_dir / f"{basename}.pdf", metadata=metadata)
    fig.savefig(output_dir / f"{basename}.png", dpi=layout["dpi"],
                metadata={"Description": metadata["Subject"]})
    plt.close(fig)
    return {"width_mm": layout["width_mm"], "height_mm": layout["height_mm"],
            "minimum_marker_clear_gap_pt": round(marker_gap_pt, 3),
            "outside_canvas_text": clipped, "marker_count": 84, "interval_count": 84,
            "rates_unit": "percent", "cost_unit": next(metric["unit"] for metric in protocol["metrics"] if metric["id"] == "Cost"),
            "errorbar_geometry": "xerr=(estimate-lower, upper-estimate); not an interval estimator"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    protocol, rows, lookup, report = load_and_check(args.data_dir, args.allow_partial)
    if args.check_only:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.allow_partial:
        parser.error("Partial data are only accepted with --check-only")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    qa = render(protocol, lookup, args.output_dir)
    export_review(protocol, rows, lookup, report, args.output_dir)
    generated_names = {"assumed_point_estimates.csv", "推演明细.md", "caption.md", "data_audit.json",
                       "metrics_comparison_assumed.pdf", "metrics_comparison_assumed.png"} if protocol["data_status"] == "hypothetical" else {"point_estimates.csv", "推演明细.md", "caption.md", "data_audit.json", "metrics_comparison.pdf", "metrics_comparison.png"}
    qa["files"] = {path.name: sha256(path) for path in sorted(args.output_dir.iterdir()) if path.is_file() and path.name in generated_names}
    qa["script_sha256"] = sha256(Path(__file__))
    qa["data_hashes"] = report["hashes"]
    (args.output_dir / "render_manifest.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "rendered", "points": len(rows), "output_dir": str(args.output_dir), "layout": qa}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
