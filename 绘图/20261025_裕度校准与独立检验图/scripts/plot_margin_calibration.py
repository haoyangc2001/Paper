"""Exact ECDFs from reviewed paired summaries. This script never generates data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from compute_statistics import ROOT, read_rows, validate_rows, quantile, summarize


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calculate(path):
    rows = read_rows(path)
    errors = validate_rows(rows)
    if errors:
        raise ValueError(errors)
    groups = {s: [r for r in rows if r["split"] == s] for s in ("calibration", "held_out")}
    if not all(groups.values()):
        raise ValueError("Both calibration and held-out groups are required")
    limits = {k: quantile([r[k] for r in groups["calibration"]]) for k in ("cost_overrun", "max_deviation_m")}
    return {s: summarize(r, limits) for s, r in groups.items()}, limits


def bounds_and_ticks(metric, stats, scale):
    values = [v["metrics"][metric][key]*scale for v in stats.values() for key in ("min", "max")]
    lo, hi = min(values), max(values)
    if metric == "cost_overrun":
        lower = min(-5, math.floor(lo/5)*5)
        upper = max(30, math.ceil(hi/5)*5)
        ticks = sorted(set([lower] + list(range(0, int(upper)+1, 10))))
    else:
        lower = 0
        upper = max(0.22, math.ceil(hi/0.02)*0.02)
        step = 0.1 if upper <= 0.3 else 10**math.floor(math.log10(upper))/2
        ticks = [round(i*step, 6) for i in range(math.floor(upper/step)+1)]
    return (lower, upper), ticks


def caption(stats, limits):
    c, t = stats["calibration"], stats["held_out"]
    a, d = limits["cost_overrun"], limits["max_deviation_m"]
    cm, gm = t["metrics"]["cost_overrun"], t["metrics"]["max_deviation_m"]
    return (
        "Illustrative margin calibration and held-out assessment (all data are hypothetical). "
        f"Exact empirical cumulative distributions use {c['n']} calibration pairs and {t['n']} held-out pairs. "
        "(a) Signed relative cost overrun, (J_exec - J_nom)/J_nom. "
        "(b) Maximum time-aligned planar position discrepancy under matched initial conditions and commands. "
        f"The calibration 90th percentiles fix alpha = {100*a:g}% and delta = {d:g} m; "
        "vertical dashed lines mark these thresholds and horizontal dashed lines mark the 90% calibration target. "
        "Blue circles and orange diamonds mark each ECDF evaluated at the fixed threshold. "
        f"Held-out coverage is {cm['covered_n']}/{t['n']} ({100*cm['coverage']:.2f}%) for cost and "
        f"{gm['covered_n']}/{t['n']} ({100*gm['coverage']:.2f}%) for position. "
        f"Calibration coverage is {100*c['metrics']['cost_overrun']['coverage']:.3f}% and "
        f"{100*c['metrics']['max_deviation_m']['coverage']:.2f}% because ties are included. "
        "Held-out records do not select thresholds. These constructed summaries illustrate the workflow, "
        "not measured independent validation or a joint safety guarantee.\n"
    )


def draw(data, config_path, output, panel="both", statistics_path=None):
    data, config_path, output = data.resolve(), config_path.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new output directory to preserve prior render versions")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stats, limits = calculate(data)
    if statistics_path:
        reference = json.loads(statistics_path.read_text(encoding="utf-8"))
        if reference["source_sha256"] != sha256(data) or reference["thresholds"] != limits or reference["groups"] != stats:
            raise ValueError("Statistics file is stale or disagrees with current input")
    if config["target_percent"] != 90:
        raise ValueError("The locked estimator uses Q90; change the statistical protocol before its label")
    colors = config["colors"]
    width = config["width_mm"] if panel == "both" else config["width_mm"]/2
    plt.rcParams.update({
        "font.family": config["font_family"], "font.size": config["font_pt"],
        "axes.labelsize": config["font_pt"], "axes.titlesize": config["title_pt"],
        "xtick.labelsize": config["font_pt"], "ytick.labelsize": config["font_pt"],
        "text.color": colors["text"], "axes.labelcolor": colors["text"],
        "xtick.color": colors["text"], "ytick.color": colors["text"],
        "axes.edgecolor": "#A8B1B9", "axes.linewidth": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "margin-calibration-v1", "path.simplify": False,
        "savefig.facecolor": "white", "mathtext.fontset": "dejavusans",
    })
    panels = config["panels"] if panel == "both" else [config["panels"][0]]
    fig, axes = plt.subplots(1, len(panels), figsize=(width/25.4, config["height_mm"]/25.4), sharey=True, squeeze=False)
    layout = dict(config["layout"])
    if panel != "both":
        layout["left"] = 0.26
        layout["right"] = 0.94
    fig.subplots_adjust(**layout)
    panel_records = []
    for index, spec in enumerate(panels):
        ax = axes[0, index]
        metric, scale = spec["metric"], spec["scale"]
        threshold = limits[metric]*scale
        xlim, ticks = bounds_and_ticks(metric, stats, scale)
        ax.set(xlim=xlim, ylim=(0, 100), xticks=ticks, yticks=[0, 25, 50, 75, 100])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
        ax.tick_params(length=2, width=0.5, pad=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=colors["grid"], linewidth=0.5)
        ax.axhline(config["target_percent"], color=colors["reference"], lw=0.65, ls=(0, (2.5, 2.0)), zorder=1)
        ax.axvline(threshold, color=colors["reference"], lw=0.65, ls=(0, (2.5, 2.0)), zorder=1)
        curves = {}
        for split in ("calibration", "held_out"):
            summary = stats[split]["metrics"][metric]
            points = summary["ecdf"]
            # where='post': F(x) is the count <= x; tied values produce one exact jump.
            xx = [xlim[0]] + [p["x"]*scale for p in points] + [xlim[1]]
            yy = [0] + [100*p["fraction"] for p in points] + [100]
            linestyle = "-" if split == "calibration" else (0, (3.2, 1.6))
            ax.step(xx, yy, where="post", color=colors[split], linewidth=config["line_width_pt"], linestyle=linestyle, zorder=3, clip_on=False)
            coverage = summary["coverage"]*100
            ax.plot(threshold, coverage, marker="o" if split == "calibration" else "D", markersize=2.7,
                    markeredgewidth=0.65, markeredgecolor=colors[split], markerfacecolor="white" if split == "calibration" else colors[split], zorder=5)
            curves[split] = {"x": xx, "percent": yy, "n": stats[split]["n"], "covered_n": summary["covered_n"], "threshold_point": [threshold, coverage]}
        # Literal Greek preserves PDF text extraction as well as visual glyphs.
        symbol = "α" if spec["symbol"] == "alpha" else "δ"
        limit_label = f"{symbol} = {threshold:g}%" if metric == "cost_overrun" else f"{symbol} = {threshold:g} m"
        box = {"facecolor": "white", "edgecolor": "none", "pad": 0.6}
        ax.text(0.035, config["threshold_label_y"], limit_label, transform=ax.transAxes, va="top", fontsize=config["font_pt"], color=colors["text"], bbox=box)
        coverage = stats["held_out"]["metrics"][metric]["coverage"]*100
        ax.text(0.97, 0.32, f"Held-out at {symbol}", transform=ax.transAxes, ha="right", va="center", color=colors["held_out"], fontsize=config["font_pt"], bbox=box)
        ax.text(0.97, 0.18, f"{coverage:.2f}%", transform=ax.transAxes, ha="right", va="center", color=colors["held_out"], fontsize=8, fontweight="semibold", bbox=box)
        ax.set_title(spec["title"], loc="left", pad=5, fontweight="semibold")
        ax.set_xlabel(spec["xlabel"], labelpad=4)
        if index == 0:
            ax.set_ylabel("Cumulative fraction (%)", labelpad=4)
        panel_records.append({"metric": metric, "scale": scale, "xlim": xlim, "threshold": threshold, "curves": curves})
    handles = [Line2D([], [], color=colors[s], lw=config["line_width_pt"], ls="-" if s == "calibration" else (0, (3.2, 1.6))) for s in ("calibration", "held_out")]
    labels = [f"Calibration (n={stats['calibration']['n']})", f"Held-out (n={stats['held_out']['n']})"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 1), ncol=2 if panel == "both" else 1, frameon=False, borderaxespad=0, handlelength=1.8, handletextpad=0.5, columnspacing=1.2, labelspacing=0.3)
    target = Line2D([], [], color=colors["reference"], lw=0.65, ls=(0, (2.5, 2.0)))
    fig.legend([target], ["90% calibration target"], loc="upper center", bbox_to_anchor=(0.52, 0.90 if panel == "both" else 0.88), frameon=False, borderaxespad=0, handlelength=1.8)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_bounds = []
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer).transformed(fig.transFigure.inverted())
            text_bounds.append({"text": artist.get_text(), "font_pt": artist.get_fontsize(), "bounds_fraction": list(box.bounds)})
    stem = "margin_calibration" if panel == "both" else "cost_ecdf"
    fixed_date = datetime(2026, 10, 25, tzinfo=timezone.utc)
    fig.savefig(output / f"{stem}.pdf", metadata={"Title": "Margin calibration and held-out assessment", "Subject": "All data are hypothetical", "Author": "", "CreationDate": fixed_date, "ModDate": fixed_date})
    fig.savefig(output / f"{stem}.svg", metadata={"Date": "2026-10-25", "Description": "All data are hypothetical"})
    fig.savefig(output / f"{stem}.png", dpi=config["dpi"], metadata={"Description": "All data are hypothetical; see accompanying caption."})
    # Fixed pixel scale of 96 dpi is a physical-size approximation, not display calibration.
    fig.savefig(output / "single_column_preview.png", dpi=96)
    plt.close(fig)
    (output / "caption.txt").write_text(caption(stats, limits), encoding="utf-8")
    manifest = {
        "data_status": "hypothetical", "data_path": str(data), "data_sha256": sha256(data),
        "config_path": str(config_path), "config_sha256": sha256(config_path), "script_sha256": sha256(Path(__file__)),
        "width_mm": width, "height_mm": config["height_mm"], "dpi": config["dpi"],
        "panel": panel, "thresholds": limits, "target_percent": 90,
        "panels": panel_records, "text_bounds": text_bounds,
        "output_hashes": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    (output / "render_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "thresholds": limits, "coverage_percent": {s: {k: v["coverage"]*100 for k, v in group["metrics"].items()} for s, group in stats.items()}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/paired_rollouts.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "config/style.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--statistics", type=Path)
    parser.add_argument("--panel", choices=("both", "cost"), default="both")
    args = parser.parse_args()
    draw(args.data, args.config, args.output, args.panel, args.statistics)
