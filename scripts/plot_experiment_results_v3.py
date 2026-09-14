from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


NS_MAIN = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
NS_REL = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}

SCENARIOS = [
    "Moderately Cluttered",
    "Narrow-Passage",
    "Densely Obstructed",
]
DISPLAY_SCENARIOS = [
    "Mod. Clut.",
    "Narrow Pass.",
    "Dense Obst.",
]
MAIN_METHODS = [
    "HD-MCRA",
    "RA",
    "RC-PPO",
    "PPO_beta_L",
    "PPO_beta_H",
    "CPPO_beta_L",
    "CPPO_beta_H",
]
BUDGET_METHODS = ["RA", "RC-PPO", "HD-MCRA"]
BUDGET_LEVELS = ["0.80", "0.85", "0.90", "0.96", "1.00", "1.05", "1.12", "1.20"]

DISPLAY_METHODS = {
    "HD-MCRA": "HD-MCRA",
    "RA": "RA",
    "RC-PPO": "RC-PPO",
    "PPO_beta_L": r"PPO$_{\beta L}$",
    "PPO_beta_H": r"PPO$_{\beta H}$",
    "CPPO_beta_L": r"CPPO$_{\beta L}$",
    "CPPO_beta_H": r"CPPO$_{\beta H}$",
}

COLORS = {
    "HD-MCRA": "#9B1D20",
    "RA": "#2F4B7C",
    "RC-PPO": "#4E9F3D",
    "PPO_beta_L": "#5DA5DA",
    "PPO_beta_H": "#9A9A9A",
    "CPPO_beta_L": "#B6992D",
    "CPPO_beta_H": "#F28E2B",
}

MAIN_METRIC_CONFIG = OrderedDict(
    {
        "SR": {
            "title": "Success Rate (SR)",
            "formatter": "percent",
            "ylim": (0.0, 1.0),
            "legend_loc": "upper right",
        },
        "CR": {
            "title": "Collision Rate (CR)",
            "formatter": "percent",
            "ylim": (0.0, 1.0),
            "legend_loc": "upper right",
        },
        "Average Cost": {
            "title": "Average Cost",
            "formatter": "float",
            "ylim": (0.0, 300.0),
            "legend_loc": "upper right",
        },
        "Minimum Clearance": {
            "title": "Minimum Clearance (m)",
            "formatter": "float2",
            "ylim": (0.0, 0.8),
            "legend_loc": "upper right",
        },
    }
)

BUDGET_METRIC_CONFIG = OrderedDict(
    {
        "SR": {
            "title": "Success Rate (SR)",
            "formatter": "percent",
            "ylim": (0.0, 1.0),
            "legend_loc": "upper right",
        },
        "CR": {
            "title": "Collision Rate (CR)",
            "formatter": "percent",
            "ylim": (0.0, 1.0),
            "legend_loc": "upper right",
        },
        "BVR": {
            "title": "Budget Violation Rate (BVR)",
            "formatter": "percent",
            "ylim": (0.0, 1.0),
            "legend_loc": "upper right",
        },
        "Average Cost": {
            "title": "Average Cost",
            "formatter": "float",
            "ylim": (0.0, 200.0),
            "legend_loc": "upper right",
        },
    }
)


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


def parse_workbook(path: Path) -> dict[str, list[list[str]]]:
    with ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall("rel:Relationship", NS_REL)
        }
        sheets: dict[str, list[list[str]]] = {}
        for sheet in workbook.findall("main:sheets/main:sheet", NS_MAIN):
            name = sheet.attrib["name"]
            rel_id = sheet.attrib[
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            ]
            target = rel_map[rel_id].replace("\\", "/")
            sheet_xml = ET.fromstring(archive.read(f"xl/{target}"))
            rows: list[list[str]] = []
            for row in sheet_xml.findall("main:sheetData/main:row", NS_MAIN):
                values: list[str] = []
                current_col = 1
                for cell in row.findall("main:c", NS_MAIN):
                    ref = cell.attrib.get("r", "")
                    col_name = "".join(ch for ch in ref if ch.isalpha())
                    if col_name:
                        target_col = 0
                        for ch in col_name:
                            target_col = target_col * 26 + (ord(ch.upper()) - 64)
                        while current_col < target_col:
                            values.append("")
                            current_col += 1
                    cell_type = cell.attrib.get("t")
                    if cell_type == "inlineStr":
                        text = "".join(
                            node.text or ""
                            for node in cell.findall(".//main:t", NS_MAIN)
                        )
                    else:
                        value_node = cell.find("main:v", NS_MAIN)
                        text = value_node.text if value_node is not None else ""
                    values.append(text)
                    current_col += 1
                rows.append(values)
            sheets[name] = rows
        return sheets


def load_table(path: Path, sheet_name: str) -> list[dict[str, str]]:
    rows = parse_workbook(path)[sheet_name]
    header = rows[0]
    table: list[dict[str, str]] = []
    for row in rows[1:]:
        if not any(str(cell).strip() for cell in row):
            continue
        padded = row + [""] * (len(header) - len(row))
        table.append({header[idx]: padded[idx] for idx in range(len(header))})
    return table


def build_main_data(path: Path) -> dict[str, dict[str, list[float]]]:
    table = load_table(path, "main_metric_wide")
    data: dict[str, dict[str, list[float]]] = OrderedDict()
    for record in table:
        metric = record["Metric"]
        method = record["Method"]
        data.setdefault(metric, OrderedDict())
        data[metric][method] = [float(record[scenario]) for scenario in SCENARIOS]
    return data


def build_budget_data(path: Path) -> dict[str, dict[str, list[float]]]:
    table = load_table(path, "budget_metric_wide")
    data: dict[str, dict[str, list[float]]] = OrderedDict()
    for record in table:
        metric = record["Metric"]
        method = record["Method"]
        data.setdefault(metric, OrderedDict())
        data[metric][method] = [float(record[level]) for level in BUDGET_LEVELS]
    return data


def format_label(value: float, formatter: str) -> str:
    if formatter == "percent":
        return f"{value * 100:.0f}%"
    if formatter == "float2":
        return f"{value:.2f}"
    return f"{value:.0f}"


def style_axes(ax: plt.Axes, title: str, ylim: tuple[float, float], formatter: str) -> None:
    ax.set_title("")
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=4.9)
    ax.set_ylim(*ylim)
    if formatter == "percent":
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.5,
        -0.23,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.2,
        fontweight="bold",
    )


def plot_main_grid(main_data: dict[str, dict[str, list[float]]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.8))
    axes = axes.flatten()
    x = np.arange(len(SCENARIOS))
    width = 0.082

    for ax, (metric, config) in zip(axes, MAIN_METRIC_CONFIG.items()):
        for idx, method in enumerate(MAIN_METHODS):
            offset = (idx - (len(MAIN_METHODS) - 1) / 2) * width
            values = main_data[metric][method]
            ax.bar(
                x + offset,
                values,
                width=width,
                color=COLORS[method],
                edgecolor="black",
                linewidth=0.25,
                zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(DISPLAY_SCENARIOS, fontsize=2.9)
        style_axes(ax, config["title"], config["ylim"], config["formatter"])
        add_panel_label(ax, config["title"])
        ax.legend(
            handles=[
                Patch(
                    facecolor=COLORS[method],
                    edgecolor="black",
                    linewidth=0.25,
                    label=DISPLAY_METHODS[method],
                )
                for method in MAIN_METHODS
            ],
            loc=config["legend_loc"],
            fontsize=3.2,
            frameon=True,
            framealpha=0.9,
            borderpad=0.15,
            labelspacing=0.12,
            handlelength=0.8,
            handletextpad=0.25,
            borderaxespad=0.18,
        )

    fig.subplots_adjust(top=0.98, bottom=0.14, left=0.10, right=0.99, wspace=0.32, hspace=0.55)

    png_path = output_dir / "figure_metrics_grid_single_column_v3.png"
    pdf_path = output_dir / "figure_metrics_grid_single_column_v3.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    snippet = r"""\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{../images/figure_metrics_grid_single_column_v3}
\caption{Performance comparison across three navigation scenarios. From top-left to bottom-right: success rate, collision rate, average cost, and minimum clearance.}
\label{fig:metrics_grid_single_column}
\end{figure}
"""
    (output_dir / "figure_metrics_grid_single_column_v3.tex").write_text(snippet, encoding="utf-8")


def plot_budget_grid(budget_data: dict[str, dict[str, list[float]]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.75))
    axes = axes.flatten()
    x = np.array([float(level) for level in BUDGET_LEVELS])
    markers = {"RA": "s", "RC-PPO": "o", "HD-MCRA": "^"}

    for ax, (metric, config) in zip(axes, BUDGET_METRIC_CONFIG.items()):
        for method in BUDGET_METHODS:
            values = budget_data[metric][method]
            ax.plot(
                x,
                values,
                marker=markers[method],
                markersize=2.2,
                linewidth=0.75,
                color=COLORS[method],
                markeredgecolor="black",
                markeredgewidth=0.20,
                label=DISPLAY_METHODS[method],
                zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(BUDGET_LEVELS, rotation=0, fontsize=4.5)
        style_axes(ax, config["title"], config["ylim"], config["formatter"])
        add_panel_label(ax, config["title"])
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=COLORS[method],
                    marker=markers[method],
                    linewidth=0.75,
                    markersize=2.6,
                    markeredgecolor="black",
                    markeredgewidth=0.20,
                    label=DISPLAY_METHODS[method],
                )
                for method in BUDGET_METHODS
            ],
            loc=config["legend_loc"],
            fontsize=3.5,
            frameon=True,
            framealpha=0.9,
            borderpad=0.15,
            labelspacing=0.12,
            handlelength=1.0,
            handletextpad=0.25,
            borderaxespad=0.18,
        )

    fig.subplots_adjust(top=0.98, bottom=0.16, left=0.10, right=0.99, wspace=0.34, hspace=0.60)

    png_path = output_dir / "figure_budget_sweep_v3.png"
    pdf_path = output_dir / "figure_budget_sweep_v3.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    snippet = r"""\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{../images/figure_budget_sweep_v3}
\caption{Budget-response curves under normalized initial budgets. All four subplots share the horizontal axis $z_0 / z_{\mathrm{ref}}$. From top-left to bottom-right: success rate, collision rate, budget violation rate, and average cost.}
\label{fig:budget_sweep}
\end{figure}
"""
    (output_dir / "figure_budget_sweep_v3.tex").write_text(snippet, encoding="utf-8")


def main() -> None:
    configure_fonts()
    script_dir = Path(__file__).resolve().parent
    workbook_path = script_dir.parent / "data" / "experiment_results_rcppo_ra_v3.xlsx"
    output_dir = script_dir.parent / "images"
    output_dir.mkdir(exist_ok=True)

    main_data = build_main_data(workbook_path)
    budget_data = build_budget_data(workbook_path)
    plot_main_grid(main_data, output_dir)
    plot_budget_grid(budget_data, output_dir)
    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
