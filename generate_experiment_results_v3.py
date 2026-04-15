from __future__ import annotations

from collections import OrderedDict
from datetime import date
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape
import zipfile


SCENARIOS = [
    "Moderately Cluttered",
    "Narrow-Passage",
    "Densely Obstructed",
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

MAIN_METRICS = [
    "SR",
    "CR",
    "Average Cost",
    "BVR",
    "Minimum Clearance",
]

BUDGET_LEVELS = ["0.80", "0.85", "0.90", "0.95", "1.00", "1.05", "1.10", "1.20"]
BUDGET_METHODS = ["RA", "RC-PPO", "HD-MCRA"]
BUDGET_METRICS = ["SR", "CR", "BVR", "Average Cost"]

MAIN_DATA: OrderedDict[str, OrderedDict[str, list[float]]] = OrderedDict(
    {
        "SR": OrderedDict(
            {
                "HD-MCRA": [0.85, 0.73, 0.62],
                "RA": [0.76, 0.61, 0.43],
                "RC-PPO": [0.84, 0.72, 0.56],
                "PPO_beta_L": [0.87, 0.67, 0.49],
                "PPO_beta_H": [0.57, 0.45, 0.25],
                "CPPO_beta_L": [0.55, 0.39, 0.19],
                "CPPO_beta_H": [0.89, 0.68, 0.52],
            }
        ),
        "CR": OrderedDict(
            {
                "HD-MCRA": [0.08, 0.15, 0.22],
                "RA": [0.06, 0.13, 0.20],
                "RC-PPO": [0.07, 0.15, 0.25],
                "PPO_beta_L": [0.11, 0.24, 0.34],
                "PPO_beta_H": [0.16, 0.31, 0.40],
                "CPPO_beta_L": [0.05, 0.14, 0.18],
                "CPPO_beta_H": [0.09, 0.19, 0.27],
            }
        ),
        "Average Cost": OrderedDict(
            {
                "HD-MCRA": [64.0, 80.0, 121.0],
                "RA": [76.0, 95.0, 141.0],
                "RC-PPO": [67.0, 84.0, 124.0],
                "PPO_beta_L": [70.0, 91.0, 138.0],
                "PPO_beta_H": [92.0, 105.0, 154.0],
                "CPPO_beta_L": [98.0, 118.0, 158.0],
                "CPPO_beta_H": [75.0, 96.0, 146.0],
            }
        ),
        "BVR": OrderedDict(
            {
                "HD-MCRA": [0.05, 0.09, 0.13],
                "RA": [0.12, 0.20, 0.28],
                "RC-PPO": [0.08, 0.14, 0.20],
                "PPO_beta_L": [0.11, 0.21, 0.30],
                "PPO_beta_H": [0.16, 0.27, 0.38],
                "CPPO_beta_L": [0.15, 0.22, 0.30],
                "CPPO_beta_H": [0.10, 0.17, 0.24],
            }
        ),
        "Minimum Clearance": OrderedDict(
            {
                "HD-MCRA": [0.40, 0.31, 0.23],
                "RA": [0.43, 0.34, 0.25],
                "RC-PPO": [0.38, 0.30, 0.22],
                "PPO_beta_L": [0.29, 0.19, 0.12],
                "PPO_beta_H": [0.24, 0.15, 0.09],
                "CPPO_beta_L": [0.45, 0.36, 0.28],
                "CPPO_beta_H": [0.34, 0.25, 0.18],
            }
        ),
    }
)

BUDGET_DATA: OrderedDict[str, OrderedDict[str, list[float]]] = OrderedDict(
    {
        "SR": OrderedDict(
            {
                "RA": [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],
                "RC-PPO": [0.58, 0.62, 0.65, 0.68, 0.707, 0.72, 0.73, 0.74],
                "HD-MCRA": [0.61, 0.64, 0.67, 0.70, 0.733, 0.745, 0.755, 0.765],
            }
        ),
        "CR": OrderedDict(
            {
                "RA": [0.130, 0.130, 0.130, 0.130, 0.130, 0.130, 0.130, 0.130],
                "RC-PPO": [0.210, 0.190, 0.180, 0.170, 0.157, 0.150, 0.145, 0.140],
                "HD-MCRA": [0.190, 0.180, 0.173, 0.160, 0.150, 0.145, 0.138, 0.130],
            }
        ),
        "BVR": OrderedDict(
            {
                "RA": [0.35, 0.31, 0.27, 0.23, 0.20, 0.16, 0.13, 0.10],
                "RC-PPO": [0.30, 0.26, 0.21, 0.17, 0.14, 0.11, 0.09, 0.06],
                "HD-MCRA": [0.23, 0.19, 0.15, 0.12, 0.09, 0.07, 0.06, 0.04],
            }
        ),
        "Average Cost": OrderedDict(
            {
                "RA": [104.0, 104.0, 104.0, 104.0, 104.0, 104.0, 104.0, 104.0],
                "RC-PPO": [85.0, 87.0, 88.0, 90.0, 91.7, 93.5, 95.0, 99.0],
                "HD-MCRA": [82.0, 84.0, 84.7, 86.5, 88.3, 90.0, 91.2, 95.0],
            }
        ),
    }
)

MIN_FEASIBLE_BUDGET_ROWS = [
    [
        "Average over evaluation set",
        "SR >= 60%, CR <= 20%, BVR <= 15%",
        "HD-MCRA",
        0.90,
        67.0,
        17.3,
        15.0,
        84.7,
    ],
    [
        "Average over evaluation set",
        "SR >= 60%, CR <= 20%, BVR <= 15%",
        "RC-PPO",
        1.00,
        70.7,
        15.7,
        14.0,
        91.7,
    ],
    [
        "Average over evaluation set",
        "SR >= 60%, CR <= 20%, BVR <= 15%",
        "RA",
        1.10,
        60.0,
        13.0,
        13.0,
        104.0,
    ],
]

ABLATION_ROWS = [
    ("w/o Hierarchy", 60.1, 27.5, 95.8, 19.8, 0.18),
    (r"w/o $\alpha$", 66.4, 19.7, 82.9, 21.4, 0.29),
    (r"w/o $\delta$", 63.7, 25.8, 74.6, 12.6, 0.19),
    ("RC-PPO-style w/o margins", 61.9, 28.3, 89.2, 24.1, 0.17),
    ("HD-MCRA (full)", 70.8, 18.6, 71.3, 11.2, 0.32),
]


def column_name(index: int) -> str:
    name = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def average(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def sheet_xml(rows: list[list[object]]) -> str:
    xml_rows: list[str] = []
    for row_idx, row in enumerate(rows, start=1):
        cells: list[str] = []
        for col_idx, value in enumerate(row, start=1):
            ref = f"{column_name(col_idx)}{row_idx}"
            if value is None or value == "":
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                text = escape(str(value))
                cells.append(
                    f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'
                )
        xml_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheetData>{''.join(xml_rows)}</sheetData></worksheet>"
    )


def build_main_long_rows() -> list[list[object]]:
    rows: list[list[object]] = [["Metric", "Scenario", "Method", "Value"]]
    for metric in MAIN_METRICS:
        for method in MAIN_METHODS:
            for scenario, value in zip(SCENARIOS, MAIN_DATA[metric][method]):
                rows.append([metric, scenario, method, value])
    return rows


def build_main_metric_wide_rows() -> list[list[object]]:
    rows: list[list[object]] = [["Metric", "Method", *SCENARIOS]]
    for metric in MAIN_METRICS:
        for method in MAIN_METHODS:
            rows.append([metric, method, *MAIN_DATA[metric][method]])
    return rows


def build_main_average_rows() -> list[list[object]]:
    rows: list[list[object]] = [["Metric", "Method", "Average"]]
    for metric in MAIN_METRICS:
        for method in MAIN_METHODS:
            rows.append([metric, method, round(average(MAIN_DATA[metric][method]), 4)])
    return rows


def build_budget_long_rows() -> list[list[object]]:
    rows: list[list[object]] = [["BudgetLevel", "Metric", "Method", "Value"]]
    for metric in BUDGET_METRICS:
        for method in BUDGET_METHODS:
            for level, value in zip(BUDGET_LEVELS, BUDGET_DATA[metric][method]):
                rows.append([level, metric, method, value])
    return rows


def build_budget_metric_wide_rows() -> list[list[object]]:
    rows: list[list[object]] = [["Metric", "Method", *BUDGET_LEVELS]]
    for metric in BUDGET_METRICS:
        for method in BUDGET_METHODS:
            rows.append([metric, method, *BUDGET_DATA[metric][method]])
    return rows


def build_min_feasible_budget_rows() -> list[list[object]]:
    rows: list[list[object]] = [[
        "Evaluation Scope",
        "Feasibility Criterion",
        "Method",
        "z0_over_zref_star",
        "SR_at_star (%)",
        "CR_at_star (%)",
        "BVR_at_star (%)",
        "Cost_at_star",
    ]]
    rows.extend(MIN_FEASIBLE_BUDGET_ROWS)
    return rows


def build_metadata_rows() -> list[list[object]]:
    today = date.today().isoformat()
    rows: list[list[object]] = [
        ["Field", "Value"],
        ["Generated On", today],
        ["Workbook Purpose", "Estimated v3 experiment package for RC-PPO/RA retrofit"],
        ["Source Anchors", "conference_101719.tex + experiment_results_backup_2026-04-13.xlsx"],
        ["Method Terminology", "RA = reach-avoid baseline; RC-PPO = minimum-cost reach-avoid baseline; HD-MCRA = hierarchical RC-PPO with data-driven conservative margins"],
        ["Budget Protocol", "Dense normalized budget sweep z0/z_ref in {0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20}"],
        ["Main Metrics", "SR, CR, Average Cost, BVR, Minimum Clearance"],
        ["Budget Metrics", "SR, CR, BVR, Average Cost"],
        ["Minimum Feasible Budget Definition", "Smallest z0/z_ref such that SR >= 60%, CR <= 20%, and BVR <= 15% on the evaluation set."],
        ["Literature Basis", "Hsu et al. RSS 2021; So et al. NeurIPS 2024; Achiam et al. ICML 2017; Tessler et al. ICLR 2019; Hu et al. RA-L 2021; Weerakoon et al. ICRA 2024"],
        ["Notes", "Existing PPO/CPPO/HD-MCRA SR-CR-Cost values are preserved whenever possible; new values are estimated for internal consistency and paper drafting."],
        [],
        ["Ablation Variant", "SR (%)", "CR (%)", "Cost", "BVR (%)", "Minimum Clearance (m)"],
    ]
    rows.extend([list(row) for row in ABLATION_ROWS])
    return rows


def workbook_xml(sheet_names: list[str]) -> str:
    sheets = "".join(
        f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        for idx, name in enumerate(sheet_names, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{sheets}</sheets></workbook>"
    )


def workbook_rels_xml(sheet_names: list[str]) -> str:
    rels = "".join(
        f'<Relationship Id="rId{idx}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{idx}.xml"/>'
        for idx, _ in enumerate(sheet_names, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{rels}</Relationships>"
    )


def root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
        'Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" '
        'Target="docProps/app.xml"/>'
        "</Relationships>"
    )


def content_types_xml(sheet_count: int) -> str:
    overrides = [
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/docProps/core.xml" '
        'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    overrides.extend(
        f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for idx in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f"{''.join(overrides)}</Types>"
    )


def app_xml(sheet_names: list[str]) -> str:
    titles = "".join(f"<vt:lpstr>{escape(name)}</vt:lpstr>" for name in sheet_names)
    count = len(sheet_names)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        "<Application>Codex</Application>"
        "<DocSecurity>0</DocSecurity>"
        "<ScaleCrop>false</ScaleCrop>"
        "<HeadingPairs><vt:vector size=\"2\" baseType=\"variant\">"
        "<vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant>"
        f"<vt:variant><vt:i4>{count}</vt:i4></vt:variant>"
        "</vt:vector></HeadingPairs>"
        f"<TitlesOfParts><vt:vector size=\"{count}\" baseType=\"lpstr\">{titles}</vt:vector></TitlesOfParts>"
        "<Company>OpenAI Codex</Company><LinksUpToDate>false</LinksUpToDate>"
        "<SharedDoc>false</SharedDoc><HyperlinksChanged>false</HyperlinksChanged>"
        "<AppVersion>1.0</AppVersion></Properties>"
    )


def core_xml() -> str:
    today = date.today().isoformat() + "T00:00:00Z"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        "<dc:title>HD-MCRA experiment estimates v3</dc:title>"
        "<dc:creator>OpenAI Codex</dc:creator>"
        "<cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{today}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{today}</dcterms:modified>'
        "</cp:coreProperties>"
    )


def write_xlsx(path: Path, sheets: OrderedDict[str, list[list[object]]]) -> None:
    sheet_names = list(sheets.keys())
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml(len(sheet_names)))
        archive.writestr("_rels/.rels", root_rels_xml())
        archive.writestr("docProps/app.xml", app_xml(sheet_names))
        archive.writestr("docProps/core.xml", core_xml())
        archive.writestr("xl/workbook.xml", workbook_xml(sheet_names))
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml(sheet_names))
        for idx, rows in enumerate(sheets.values(), start=1):
            archive.writestr(f"xl/worksheets/sheet{idx}.xml", sheet_xml(rows))


def main() -> None:
    workbook_sheets: OrderedDict[str, list[list[object]]] = OrderedDict(
        {
            "main_long_format": build_main_long_rows(),
            "main_metric_wide": build_main_metric_wide_rows(),
            "main_scenario_averages": build_main_average_rows(),
            "budget_long_format": build_budget_long_rows(),
            "budget_metric_wide": build_budget_metric_wide_rows(),
            "minimum_feasible_budget": build_min_feasible_budget_rows(),
            "metadata": build_metadata_rows(),
        }
    )
    output_path = Path(__file__).resolve().parent / "experiment_results_rcppo_ra_v3.xlsx"
    write_xlsx(output_path, workbook_sheets)
    print(f"Saved {output_path.name}")


if __name__ == "__main__":
    main()
