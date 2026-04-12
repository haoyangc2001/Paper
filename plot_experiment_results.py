from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


IEEE_COLUMN_WIDTH_IN = 3.5
SINGLE_FIG_SIZE = (3.3, 2.45)
GRID_FIG_SIZE = (3.45, 1.7)

SCENARIOS = [
    "Moderately Cluttered",
    "Narrow-Passage",
    "Densely Obstructed",
]

DISPLAY_SCENARIOS = [
    "Moderately\nCluttered",
    "Narrow-\nPassage",
    "Densely\nObstructed",
]

METHODS = [
    "HD-MCRA",
    r"PPO$_{\beta L}$",
    r"PPO$_{\beta H}$",
    r"CPPO$_{\beta L}$",
    r"CPPO$_{\beta H}$",
]

COLORS = {
    "HD-MCRA": "#9B1D20",
    r"PPO$_{\beta L}$": "#5DA5DA",
    r"PPO$_{\beta H}$": "#9A9A9A",
    r"CPPO$_{\beta L}$": "#59A14F",
    r"CPPO$_{\beta H}$": "#F28E2B",
}

DATA = {
    "SR": {
        "HD-MCRA": [0.85, 0.73, 0.67],
        r"PPO$_{\beta L}$": [0.82, 0.57, 0.49],
        r"PPO$_{\beta H}$": [0.78, 0.52, 0.45],
        r"CPPO$_{\beta L}$": [0.78, 0.50, 0.44],
        r"CPPO$_{\beta H}$": [0.88, 0.69, 0.60],
    },
    "CR": {
        "HD-MCRA": [0.08, 0.15, 0.20],
        r"PPO$_{\beta L}$": [0.18, 0.30, 0.35],
        r"PPO$_{\beta H}$": [0.12, 0.24, 0.29],
        r"CPPO$_{\beta L}$": [0.07, 0.13, 0.18],
        r"CPPO$_{\beta H}$": [0.10, 0.17, 0.22],
    },
    "Average Cost": {
        "HD-MCRA": [62, 79, 115],
        r"PPO$_{\beta L}$": [84, 96, 143],
        r"PPO$_{\beta H}$": [76, 88, 134],
        r"CPPO$_{\beta L}$": [71, 90, 128],
        r"CPPO$_{\beta H}$": [78, 99, 145],
    },
}

METRIC_CONFIG = {
    "SR": {
        "ylabel": "SR (%)",
        "title": "Success Rate (SR)",
        "ylim": (0.0, 1.0),
        "formatter": "percent",
        "filename": "figure_sr_grouped",
        "legend_loc": "upper right",
        "legend_anchor": (0.985, 0.985),
    },
    "CR": {
        "ylabel": "CR (%)",
        "title": "Collision Rate (CR)",
        "ylim": (0.0, 1.0),
        "formatter": "percent",
        "filename": "figure_cr_grouped",
        "legend_loc": "upper right",
        "legend_anchor": (0.985, 0.985),
    },
    "Average Cost": {
        "ylabel": "Average Cost",
        "title": "Average Cost",
        "ylim": (0.0, 160.0),
        "formatter": "float",
        "filename": "figure_average_cost_grouped",
        "legend_loc": "upper left",
        "legend_anchor": (0.015, 0.985),
    },
}


def configure_fonts() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.default"] = "regular"


def format_label(value: float, formatter: str) -> str:
    if formatter == "percent":
        return f"{value * 100:.0f}%"
    return f"{value:.0f}"


def style_axes(ax: plt.Axes, metric_name: str, compact: bool = True) -> None:
    config = METRIC_CONFIG[metric_name]
    metric_size = 4.7 if compact else 8.0
    tick_size = 4.0 if compact else 6.0

    ax.set_title("")
    ax.set_ylabel("")
    ax.set_xlabel(config["title"], fontsize=metric_size, labelpad=2, fontweight="bold")
    ax.set_xticks(np.arange(len(DISPLAY_SCENARIOS)))
    ax.set_xticklabels(DISPLAY_SCENARIOS, fontsize=tick_size)
    ax.set_ylim(*config["ylim"])
    ax.tick_params(axis="y", labelsize=tick_size)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.02)

    if config["formatter"] == "percent":
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))


def draw_metric(
    ax: plt.Axes,
    metric_name: str,
    compact: bool = True,
    include_legend: bool = True,
    include_value_labels: bool = True,
):
    config = METRIC_CONFIG[metric_name]
    metric_data = DATA[metric_name]

    x = np.arange(len(SCENARIOS))
    width = 0.11 if compact else 0.15
    line_width = 0.45 if compact else 0.8
    label_font = 3.8 if compact else 8.0

    for idx, method in enumerate(METHODS):
        offset = (idx - (len(METHODS) - 1) / 2) * width
        values = metric_data[method]
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=method,
            color=COLORS[method],
            edgecolor="black",
            linewidth=line_width,
            zorder=3,
        )
        if include_value_labels:
            ax.bar_label(
                bars,
                labels=[format_label(v, config["formatter"]) for v in values],
                padding=1,
                fontsize=label_font,
            )

    style_axes(ax, metric_name, compact=compact)

    if include_legend:
        ax.legend(
            loc=config["legend_loc"],
            bbox_to_anchor=config["legend_anchor"],
            ncol=1,
            frameon=True,
            framealpha=0.9,
            facecolor="white",
            edgecolor="#B0B0B0",
            fontsize=3.4 if compact else 10,
            borderaxespad=0.08,
            labelspacing=0.08,
            handlelength=0.75,
            handletextpad=0.2,
            borderpad=0.12,
        )


def plot_metric(metric_name: str, output_dir: Path) -> None:
    config = METRIC_CONFIG[metric_name]
    fig, ax = plt.subplots(figsize=SINGLE_FIG_SIZE)
    draw_metric(ax, metric_name, compact=True, include_legend=True, include_value_labels=False)
    fig.tight_layout(pad=0.45)

    png_path = output_dir / f"{config['filename']}.png"
    pdf_path = output_dir / f"{config['filename']}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def plot_metrics_grid(output_dir: Path) -> None:
    metric_order = ["SR", "CR", "Average Cost"]
    fig, axes = plt.subplots(1, 3, figsize=GRID_FIG_SIZE)

    for ax, metric_name in zip(axes, metric_order):
        draw_metric(
            ax,
            metric_name,
            compact=True,
            include_legend=True,
            include_value_labels=False,
        )

    fig.subplots_adjust(
        top=0.96,
        bottom=0.20,
        left=0.07,
        right=0.995,
        wspace=0.28,
    )

    png_path = output_dir / "figure_metrics_grid_single_column.png"
    pdf_path = output_dir / "figure_metrics_grid_single_column.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def write_latex_snippet(output_dir: Path) -> None:
    snippet = r"""\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/figure_metrics_grid_single_column}
\caption{Performance comparison across three navigation scenarios. From left to right: success rate, collision rate, and average cost.}
\label{fig:metrics_grid_single_column}
\end{figure}
"""
    (output_dir / "figure_metrics_grid_single_column.tex").write_text(snippet, encoding="utf-8")


def main() -> None:
    configure_fonts()
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "figures"
    output_dir.mkdir(exist_ok=True)

    for metric_name in METRIC_CONFIG:
        plot_metric(metric_name, output_dir)

    plot_metrics_grid(output_dir)
    write_latex_snippet(output_dir)

    print(f"Saved compact figures to: {output_dir}")


if __name__ == "__main__":
    main()
