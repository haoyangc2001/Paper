"""Reconcile assumed-data exports and audit the current CCDC manuscript.

Does not invent trial records or interval endpoints. Commands:
prepare: export expected workbook tables for the narrow Artifact Tool update;
render: refresh the budget plot and install the approved comparison plot;
check: compare actual workbook, JSONL, TeX tables, and included figure files;
pages: render the compiled PDF using local Poppler for visual inspection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import runpy
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "CCDC/output/1020/consistency_review"
WORKBOOK = ROOT / "CORL/data/experiment_results_rcppo_ra_v3.xlsx"
SOURCE = ROOT / "CORL/scripts/generate_experiment_results_v3.py"
POINT_DIR = ROOT / "绘图/20261023_多方法指标对比"
POINT_DATA = POINT_DIR / "data/mc_extension_v2"
POINT_OUTPUT = POINT_DIR / "output/mc_extension_v2"
TEX = ROOT / "CCDC/latex/1020/main.tex"


def modules():
    return (runpy.run_path(str(SOURCE), run_name="read_only"),
            runpy.run_path(str(ROOT / "CORL/scripts/plot_experiment_results_v3.py"), run_name="read_only"))


def tables(source):
    return {name: source[builder]() for name, builder in {
        "main_long_format": "build_main_long_rows",
        "main_metric_wide": "build_main_metric_wide_rows",
        "main_scenario_averages": "build_main_average_rows",
        "budget_long_format": "build_budget_long_rows",
        "budget_metric_wide": "build_budget_metric_wide_rows",
        "minimum_feasible_budget": "build_min_feasible_budget_rows",
        "metadata": "build_metadata_rows",
    }.items()}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare():
    source, plot = modules()
    write_json(REVIEW / "workbook_before.json", plot["parse_workbook"](WORKBOOK))
    write_json(REVIEW / "workbook_payload.json", {
        "workbook": str(WORKBOOK), "review_dir": str(REVIEW), "sheets": tables(source)})
    print(REVIEW / "workbook_payload.json")


def render():
    _, plot = modules()
    protocol = json.loads((POINT_DATA / "protocol.json").read_text(encoding="utf-8"))
    plot["COLORS"].update({row["id"]: row["color"] for row in protocol["methods"]})
    plot["configure_fonts"]()
    image_dir = ROOT / "CCDC/images/1020"
    plot["plot_budget_grid"](plot["build_budget_data"](WORKBOOK), image_dir)
    shutil.copy2(POINT_OUTPUT / "metrics_comparison_assumed.pdf", image_dir)
    write_json(REVIEW / "figure_sources.json", {
        "budget_data": plot["build_budget_data"](WORKBOOK),
        "budget_workbook_sha256": digest(WORKBOOK),
        "budget_figure_sha256": digest(image_dir / "figure_budget_sweep_v3.pdf"),
        "comparison_sha256": digest(image_dir / "metrics_comparison_assumed.pdf"),
        "colors": plot["COLORS"],
    })
    print("Refreshed budget figure and installed the single-column five-metric comparison figure.")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check():
    source, plot = modules()
    workbook = plot["parse_workbook"](WORKBOOK)
    expected = tables(source)
    assert list(workbook) == list(expected), "Workbook sheet order changed"
    cell_count = 0
    for name, rows in expected.items():
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                actual = workbook[name][r][c] if r < len(workbook[name]) and c < len(workbook[name][r]) else ""
                if name == "metadata" and r == 1:  # export timestamp is not a scientific value
                    continue
                if isinstance(value, (int, float)):
                    assert math.isclose(float(actual), value, abs_tol=1e-9), (name, r+1, c+1, actual, value)
                else:
                    assert actual == ("" if value is None else str(value)), (name, r+1, c+1, actual, value)
                cell_count += 1
    # Main scene values must survive the workbook update unchanged.
    before_path = REVIEW / "workbook_before.json"
    if before_path.exists():
        before = json.loads(before_path.read_text(encoding="utf-8"))
        for name in ["main_long_format", "main_metric_wide", "main_scenario_averages"]:
            for r, row in enumerate(before[name]):
                for c, old in enumerate(row):
                    now = workbook[name][r][c]
                    try:
                        assert math.isclose(float(now), float(old), abs_tol=1e-12)
                    except ValueError:
                        assert now == old

    metric_check = runpy.run_path(str(POINT_DIR / "scripts/plot_metrics_comparison.py"), run_name="read_only")
    protocol, points, _, point_audit = metric_check["load_and_check"](POINT_DATA)
    original_points = [json.loads(line) for line in (POINT_DIR / "data/assumed_intervals.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert points[:84] == original_points and len(points) == 105
    assert [metric["id"] for metric in protocol["metrics"]] == ["SR", "Cost", "CR", "BVR", "MC"]
    text = TEX.read_text(encoding="utf-8")
    tex_tables = {}
    for block in re.findall(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", text, re.S):
        label = re.search(r"\\label\{([^}]+)\}", block)[1]
        rows = {}
        for line in block.splitlines():
            cells = line.split(" & ")
            if len(cells) != 6 or cells[0] in {"Method", "Variant"}:
                continue
            method = re.sub(r"\\textbf\{([^}]+)\}", r"\1", cells[0])
            values = [float(re.search(r"[0-9]+(?:\.[0-9]+)?", cell)[0]) for cell in cells[1:]]
            if label in {"tab:main_results", "tab:ablation_draft"}:
                # Display order SR/Cost/CR/BVR/MC; compare in source metric order.
                values = [values[i] for i in (0, 2, 1, 3, 4)]
            rows[method] = values
        tex_tables[label] = rows
    labels = {"HD-MCRA":"HD-MCRA", "RC-PPO":"RC-PPO", "RA":"RA",
              "PPO_beta_L":r"PPO$_{\beta_L}$", "PPO_beta_H":r"PPO$_{\beta_H}$",
              "CPPO_beta_L":r"CPPO$_{\beta_L}$", "CPPO_beta_H":r"CPPO$_{\beta_H}$"}
    for method, label in labels.items():
        expected_row = list(source["main_average_row"](label, method)[1:])
        assert tex_tables["tab:main_results"][label] == expected_row, method
    # Keep the historical hierarchy row in the source, outside the current paper.
    ablation_labels = [r"w/o $\alpha$", r"w/o $\delta$", "w/o both (RC-PPO)", "HD-MCRA"]
    assert set(tex_tables["tab:ablation_draft"]) == set(ablation_labels)
    for label, row in zip(ablation_labels, source["ABLATION_ROWS"][1:]):
        assert tex_tables["tab:ablation_draft"][label] == list(row[1:]), label
    assert tex_tables["tab:ablation_draft"]["HD-MCRA"] == tex_tables["tab:main_results"]["HD-MCRA"]
    assert tex_tables["tab:ablation_draft"]["w/o both (RC-PPO)"] == tex_tables["tab:main_results"]["RC-PPO"]

    for method in source["BUDGET_METHODS"]:
        for metric in source["BUDGET_METRICS"]:
            values = source["BUDGET_DATA"][metric][method]
            assert math.isclose(values[4], sum(source["MAIN_DATA"][metric][method])/3, abs_tol=1e-12)
            if method == "RA" and metric != "BVR":
                assert max(values)-min(values) < 1e-12
        for sr, cr in zip(source["BUDGET_DATA"]["SR"][method], source["BUDGET_DATA"]["CR"][method]):
            assert 0 <= sr <= 1 and 0 <= cr <= 1 and sr+cr <= 1
    for method in source["MAIN_METHODS"]:
        for sr, cr in zip(source["MAIN_DATA"]["SR"][method], source["MAIN_DATA"]["CR"][method]):
            assert sr+cr <= 1 + 1e-12
    for row in source["ABLATION_ROWS"]:
        assert row[1]+row[2] <= 100
    assert source["minimum_feasible_budget_rows"]() == source["MIN_FEASIBLE_BUDGET_ROWS"]

    trajectory_root = ROOT / "绘图/20261021_轨迹与同步曲线"
    scenario = json.loads((trajectory_root / "data/versions/v2/scenario.json").read_text(encoding="utf-8"))
    summary = json.loads((trajectory_root / "output/publication_v4_final/data/summary.json").read_text(encoding="utf-8"))
    overview_root = ROOT / "绘图/20261021_部署与校准总览图/optimization_v17"
    style = json.loads((overview_root / "config/style.json").read_text(encoding="utf-8"))
    assert style["alpha"] == scenario["methods"]["hd_mcra"]["alpha"] == .18
    assert style["delta_m"] == scenario["methods"]["hd_mcra"]["delta_m"] == .12
    assert scenario["dt_s"] == .2 and scenario["horizon_s"] == 30
    assert scenario["command_limits"] == [.6, .4, 1.2]
    assert not scenario["budget_violation_terminal"]
    for method, row in summary["methods"].items():
        alpha = scenario["methods"][method]["alpha"]
        assert math.isclose(row["J_charged"], (1+alpha)*row["J_nom"], abs_tol=1e-8)
        assert math.isclose(row["relative_cost_overrun"], max(0,row["J_exec"]-row["J_nom"])/row["J_nom"], abs_tol=1e-8)
        assert row["budget_violation"] == (row["J_exec"] > row["budget"])
    images = ROOT / "CCDC/images/1020"
    for match in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text):
        assert (TEX.parent / match).is_file(), match
    assert digest(images / "metrics_comparison_assumed.pdf") == digest(POINT_OUTPUT / "metrics_comparison_assumed.pdf")
    assert digest(images / "deployment_calibration_overview.pdf") == digest(overview_root / "output/final/deployment_calibration_overview.pdf")
    assert digest(images / "trajectory_sync.pdf") == digest(trajectory_root / "output/publication_v4_final/trajectory_sync.pdf")
    assert r"\begin{figure*}" not in text, "Figures must use the requested single-column placement"
    assert r"\begin{table*}" not in text, "Tables must use the requested single-column placement"
    graphic_options = re.findall(r"\\includegraphics\[([^]]*)\]", text)
    assert len(graphic_options) == 8 and all(r"width=\columnwidth" in value for value in graphic_options)
    for name in ("office_sequence_candidate_A_numbered.jpg", "office_shortest_path_sequence_final.png"):
        assert digest(images / name) == digest(ROOT / "CORL/images" / name)
    placement_owners = {
        "fig:system_overview":"System Overview",
        "tab:main_results":"Overall Navigation Performance",
        "fig:metrics_draft":"Overall Navigation Performance",
        "fig:trajectory_sync":"Overall Navigation Performance",
        "fig:margin_calibration":"Execution Mismatch and Component Analysis",
        "tab:ablation_draft":"Execution Mismatch and Component Analysis",
        "fig:ablation_margins":"Execution Mismatch and Component Analysis",
        "fig:budget_draft":"Budget Selection and Margin Sensitivity",
        "fig:hardware_sequences":"Physical-Robot Deployment and Failure Analysis",
    }
    placed_labels = set()
    assert r"\usepackage{flafter}" in text and r"\usepackage{placeins}" in text
    for match in re.finditer(r"\\begin\{(figure|table)\}\[([^]]+)\](.*?)\\end\{\1\}", text, re.S):
        assert match[2] == "!htbp", "Use bounded local floats to avoid stretched gaps"
        label = re.search(r"\\label\{([^}]+)\}", match[3])[1]
        prefix = text[:match.start()]
        owner = re.findall(r"\\subsection\{([^}]+)\}", prefix)[-1]
        assert owner == placement_owners[label], (label, owner)
        assert rf"\ref{{{label}}}" in prefix, f"{label} precedes its introduction"
        remaining = text[match.end():]
        next_heading = re.search(r"\\(?:subsection|section)\{", remaining)
        assert next_heading and r"\FloatBarrier" in remaining[:next_heading.start()], label
        placed_labels.add(label)
    assert placed_labels == set(placement_owners)
    assert digest(images / "margin_calibration.pdf") == digest(ROOT / "绘图/20261025_裕度校准与独立检验图/output/final/margin_calibration.pdf")
    ablation_render = ROOT / "绘图/20261025_模块消融图/output/refined_v2"
    assert r"../../images/1020/ablation_horizontal.pdf" in text
    assert digest(images / "ablation_horizontal.pdf") == digest(ablation_render / "ablation_horizontal.pdf")
    ablation_manifest = json.loads((ablation_render / "render_manifest.json").read_text(encoding="utf-8"))
    assert ablation_manifest["distinct_main_centers"] == 16
    assert len(ablation_manifest["comparisons"]) == 12
    variant_labels = {"NO_ALPHA":r"w/o $\alpha$", "NO_DELTA":r"w/o $\delta$", "ZERO":"w/o both (RC-PPO)"}
    metric_columns = {"SR":0, "CR":1, "Cost":2, "BVR":3}
    for comparison in ablation_manifest["comparisons"]:
        column = metric_columns[comparison["metric"]]
        assert round(comparison["full_value"], 1) == tex_tables["tab:ablation_draft"]["HD-MCRA"][column]
        variant = variant_labels[comparison["configuration"]]
        assert round(comparison["ablated_value"], 1) == tex_tables["tab:ablation_draft"][variant][column]
    budget_project = ROOT / "绘图/20261025_预算扫描图"
    budget_render = budget_project / "output/horizontal_v1"
    assert r"../../images/1020/budget_sweep_horizontal.pdf" in text
    assert digest(images / "budget_sweep_horizontal.pdf") == digest(budget_render / "budget_sweep.pdf")
    budget_manifest = json.loads((budget_render / "render_manifest.json").read_text(encoding="utf-8"))
    assert budget_manifest["points_rendered"] == 96
    for point in budget_manifest["plotted_points"]:
        metric = "Average Cost" if point["metric"] == "Cost" else point["metric"]
        index = source["BUDGET_LEVELS"].index(f"{point['x']:.2f}")
        expected_value = source["BUDGET_DATA"][metric][point["method"]][index]
        assert math.isclose(point["value"], expected_value, abs_tol=1e-12)
    # Retain and check the archived old figure and its original data snapshot too.
    figure_sources = json.loads((REVIEW / "figure_sources.json").read_text(encoding="utf-8"))
    assert figure_sources["budget_workbook_sha256"] == digest(WORKBOOK)
    assert figure_sources["budget_figure_sha256"] == digest(images / "figure_budget_sweep_v3.pdf")
    assert figure_sources["budget_data"] == plot["build_budget_data"](WORKBOOK)
    assert digest(TEX.parent / "IEEEtran.cls") == digest(ROOT / "CCDC/latex/template/Conference-LaTeX-template_10-17-19/IEEEtran.cls")
    from pypdf import PdfReader
    pdf = ROOT / "CCDC/output/1020/main.pdf"
    pages = PdfReader(pdf).pages
    page_limit = 8
    within_page_limit = len(pages) <= page_limit
    assert text.index(r"\bibliography{references}") < text.index(r"\clearpage") < text.index(r"\appendices")
    pdf_text = " ".join(page.extract_text() for page in pages)
    assert "w/o both (RC-PPO)" in pdf_text and "73.3" in pdf_text
    assert pdf.stat().st_mtime >= TEX.stat().st_mtime, "Compiled manuscript is stale"
    log = pdf.with_suffix(".log").read_text(encoding="utf-8", errors="replace")
    assert not re.search(r"Overfull|LaTeX Warning|undefined references|^!", log, re.M)
    assert r"Underfull \vbox" not in log, "Unresolved vertical column stretch"
    report = {
        "status":"passed" if within_page_limit else "content_checks_passed_page_limit_exceeded",
        "data_status":"hypothetical", "empirical_validation":False,
        "workbook_cells_checked":cell_count, "main_table_values":35,
        "ablation_table_values":sum(len(row) for row in tex_tables["tab:ablation_draft"].values()),
        "hierarchy_ablation":"archived source only; excluded from current manuscript",
        "point_intervals":len(points), "reference_budget_anchors":12,
        "main_figure_metric_order":[metric["id"] for metric in protocol["metrics"]],
        "mc_points":21, "legacy_84_point_records_preserved":True,
        "minimum_feasible_budgets":source["MIN_FEASIBLE_BUDGET_ROWS"],
        "trajectory_accounting":"passed; separate illustrative single case, not aggregate experiment",
        "calibration_cross_figure_values":{"alpha":.18,"delta_m":.12},
        "main_scene_data_preserved":True, "template_class_unchanged":True,
        "single_column_figures":7, "single_column_image_assets":len(graphic_options),
        "single_column_tables":len(tex_tables), "anchored_to_own_subsections":placement_owners,
        "float_placement":"after introduction, flexible within subsection, barrier before next heading",
        "underfull_vbox_warnings":0,
        "hardware_photo_sources":"byte-identical to the two original CoRL image assets",
        "budget_figure":"budget_sweep_horizontal.pdf", "budget_points_checked":96,
        "ablation_figure":"ablation_horizontal.pdf", "ablation_centers_checked":16,
        "ablation_comparisons_checked":12,
        "compiled_pages":len(pages), "submission_page_limit":page_limit,
        "within_submission_page_limit":within_page_limit,
        "appendix_placement":"after references, starting on a new page",
        "compiled_pdf_sha256":digest(pdf),
        "remaining_unknowns":protocol["unresolved_for_empirical_replacement"],
        "sha256":{"manuscript":digest(TEX),"workbook":digest(WORKBOOK),"source":digest(SOURCE)},
    }
    write_json(REVIEW / "audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def pages():
    from PIL import Image, ImageDraw
    output = ROOT / "CCDC/output/1020"
    preview = output / "review"
    preview.mkdir(exist_ok=True)
    staging = ROOT / "_tmp_build_consistency/pdf_pages"
    staging.mkdir(parents=True, exist_ok=True)
    poppler = Path("C:/Users/smile/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe")
    subprocess.run([str(poppler),"-r","110","-png",str(output/"main.pdf"),str(staging/"page")],check=True)
    from pypdf import PdfReader
    count = len(PdfReader(output/"main.pdf").pages)
    sheet = Image.new("RGB", (1296, 586*math.ceil(count/3)), "#e9edf1")
    draw = ImageDraw.Draw(sheet)
    for i in range(count):
        shutil.copy2(staging/f"page-{i+1}.png", preview/f"page-{i+1}.png")
        page = Image.open(preview/f"page-{i+1}.png").convert("RGB")
        page.thumbnail((412,533))
        x,y = 10+432*(i%3),22+586*(i//3)
        sheet.paste(page,(x,y))
        draw.text((x,y-17),f"Page {i+1}",fill="black")
    sheet.save(preview/"contact-sheet.png")
    print(f"Rendered {count} manuscript pages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare","render","check","pages"])
    globals()[parser.parse_args().command]()
