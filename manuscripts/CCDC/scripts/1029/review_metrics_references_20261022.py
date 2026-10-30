"""Retrieve paper references and preserve source evidence for a metrics-figure review.

This script never changes manuscript figures or generates experimental data.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import runpy
import sys
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import fitz
import requests
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "CCDC/references/metrics_review_20261022"
OLD = "CCDC/references/figure_review_20261020/papers/"
SOURCES = [
    ("abs", "2401.17583", OLD + "abs.pdf"),
    ("rcppo", "2410.22600", OLD + "rcppo.pdf"),
    ("resilient", "2310.03581", OLD + "resilient.pdf"),
    ("confined", "2403.00187", OLD + "confined.pdf"),
    ("sea_nav", "2603.09460", OLD + "sea_nav.pdf"),
    ("parkour", "2306.14874", OLD + "parkour.pdf"),
    ("vapor", "2309.07832", "CORL/references/2024 - VAPOR Legged Robot Navigation in Outdoor Vegetation Using Offline Reinforcement Learning.pdf"),
    ("autorl", "1809.10124", "CORL/references/2019 - Learning Navigation Behaviors End-to-End with AutoRL.pdf"),
    ("cpo", "1705.10528", "CORL/references/2017 - Constrained Policy Optimization.pdf"),
    ("respo", None, "CORL/references/2023 - Iterative Reachability Estimation for Safe Reinforcement Learning.pdf"),
    ("perceptive", "2201.08117", "CORL/references/2022 - Learning Robust Perceptive Locomotion for Quadrupedal Robots in the Wild.pdf"),
    ("iplanner", "2302.11434", None),
    ("viplanner", "2310.00982", None),
    ("safety_gymnasium", "2310.12567", None),
    ("omnisafe", "2305.09304", None),
    ("rliable", "2108.13264", None),
    ("vint", "2306.14846", None),
    ("nomad", "2310.07896", None),
    ("risk_distributional", "2309.14246", None),
    ("risk_bandit", "2510.14338", None),
    ("actsafe", "2410.09486", None),
]
QUERIES = [
    "quadruped navigation safety",
    "quadruped navigation benchmark",
    "risk aware quadrupedal locomotion",
    "safe reinforcement learning benchmark",
    "Benchmarking Safe Exploration in Deep Reinforcement Learning",
    "robot navigation success collision rate",
]
PREVIEW_PAGES = {
    "abs": [10], "rcppo": [8, 9, 10], "rliable": [8],
    "safety_gymnasium": [9], "risk_bandit": [6],
    "risk_distributional": [5], "resilient": [5], "autorl": [5, 8],
    "confined": [5], "nomad": [5], "vint": [7], "vapor": [6],
    "sea_nav": [5], "iplanner": [5], "viplanner": [5],
    "actsafe": [9], "respo": [8], "perceptive": [9],
    "cpo": [7, 8], "omnisafe": [10],
}


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def search(query: str) -> dict:
    url = "https://arxiv.org/search/"
    params = dict(query=query, searchtype="all", abstracts="show", order="-announced_date_first", size=50)
    result = {"query": query}
    try:
        response = requests.get(url, params=params, timeout=35)
        result.update(url=response.url, status=response.status_code)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        result["results"] = [
            {
                "title": item.select_one("p.title").get_text(" ", strip=True),
                "url": item.select_one("p.list-title a")["href"],
                "abstract_excerpt": item.get_text(" ", strip=True)[:1400],
            }
            for item in soup.select("li.arxiv-result")[:20]
        ]
    except Exception as error:
        result["error"] = str(error)
    return result


def retrieve(source: tuple) -> dict:
    key, arxiv_id, local_file = source
    result = {"key": key, "arxiv_id": arxiv_id, "retrieved_utc": datetime.now(timezone.utc).isoformat()}
    if arxiv_id:
        result["landing_url"] = "https://arxiv.org/abs/" + arxiv_id
        try:
            response = requests.get(result["landing_url"], timeout=35)
            result["landing_status"] = response.status_code
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.select_one("h1.title")
            result["title"] = title.get_text(" ", strip=True).removeprefix("Title:").strip() if title else soup.title.get_text()
            result["online_metadata_excerpt"] = soup.select_one("#abs").get_text(" ", strip=True)[:8500]
            pdf_link = soup.select_one("a.download-pdf")
            result["pdf_url"] = urljoin(response.url, pdf_link["href"]) if pdf_link else "https://arxiv.org/pdf/" + arxiv_id
            hist = soup.select_one(".submission-history")
            if hist:
                result["online_history"] = hist.get_text(" ", strip=True)
        except Exception as error:
            result["landing_error"] = str(error)
    if local_file:
        file = ROOT / local_file
        result["acquisition"] = "existing_local_original_no_duplicate"
    else:
        file = DEST / "papers" / (key + ".pdf")
        result["acquisition"] = "downloaded_public_original"
        if not file.exists():
            try:
                response = requests.get(result.get("pdf_url", "https://arxiv.org/pdf/" + arxiv_id), timeout=55)
                response.raise_for_status()
                if not response.content.startswith(b"%PDF"):
                    raise ValueError("Response is not a PDF")
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(response.content)
                result["download_status"] = response.status_code
                result["download_url"] = response.url
            except Exception as error:
                result["download_error"] = str(error)
                return result
    if not file.is_file():
        result["error"] = "Missing local PDF"
        return result
    result["file"] = file.relative_to(ROOT).as_posix()
    result["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    result["bytes"] = file.stat().st_size
    doc = fitz.open(file)
    result["pages"] = len(doc)
    result["read_version_header"] = doc[0].get_text()[:1500]
    texts = [p.get_text(sort=True) for p in doc]
    text_file = DEST / "text" / (key + ".txt")
    text_file.parent.mkdir(parents=True, exist_ok=True)
    text_file.write_text("\n\n".join("===== PDF PAGE " + str(i + 1) + " =====\n" + t for i, t in enumerate(texts)), encoding="utf-8")
    figure_index = []
    for index, text in enumerate(texts):
        for match in re.finditer(r"(?:Fig\.?|Figure)\s*\d+\s*[:.\-]", text):
            figure_index.append({"page": index + 1, "excerpt": text[match.start():match.start() + 1800]})
    result["figure_caption_candidates"] = figure_index
    if "title" not in result:
        result["title"] = doc[0].get_text().split("\n")[0]
    return result


def acquire(keys: list[str] | None = None) -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    if not keys:
        with ThreadPoolExecutor(max_workers=5) as pool:
            searches = list(pool.map(search, QUERIES))
        dump(DEST / "search_log.json", searches)
        for item in searches:
            print(json.dumps({"search": item["query"], "status": item.get("status"), "results": len(item.get("results", [])), "error": item.get("error")}), flush=True)
    selection = [s for s in SOURCES if not keys or s[0] in keys]
    with ThreadPoolExecutor(max_workers=5) as pool:
        acquired = list(pool.map(retrieve, selection))
    old_path = DEST / "source_manifest.json"
    old = json.loads(old_path.read_text(encoding="utf-8")) if keys and old_path.exists() else []
    by_key = {s["key"]: s for s in old + acquired}
    sources = [by_key[s[0]] for s in SOURCES if s[0] in by_key]
    dump(DEST / "source_manifest.json", sources)
    for item in acquired:
        print(json.dumps({k: item.get(k) for k in ("key", "title", "pages", "bytes", "file", "download_error", "landing_error")}, ensure_ascii=True), flush=True)


def scan() -> None:
    sources = json.loads((DEST / "source_manifest.json").read_text(encoding="utf-8"))
    for source in sources:
        print("\n" + source["key"] + " -- " + source.get("title", ""))
        for item in source.get("figure_caption_candidates", []):
            if item["page"] <= 12:
                print("p" + str(item["page"]) + " " + re.sub(r"\s+", " ", item["excerpt"][:550]))


def preview() -> None:
    sources = json.loads((DEST / "source_manifest.json").read_text(encoding="utf-8"))
    by_key = {s["key"]: s for s in sources}
    folder = DEST / "previews"
    folder.mkdir(exist_ok=True)
    previews = []
    for key, pages in PREVIEW_PAGES.items():
        doc = fitz.open(ROOT / by_key[key]["file"])
        for number in pages:
            file = folder / f"{key}_p{number:02}.png"
            doc[number - 1].get_pixmap(dpi=96).save(file)
            previews.append((key, number, file))
    for offset in range(0, len(previews), 9):
        group = previews[offset:offset + 9]
        sheet = Image.new("RGB", (1500, 2070), "#e4e8ed")
        draw = ImageDraw.Draw(sheet)
        for index, (key, number, file) in enumerate(group):
            x, y = (index % 3) * 500, (index // 3) * 690
            draw.text((x + 10, y + 5), f"{key} / PDF p{number}", fill="black")
            image = ImageOps.contain(Image.open(file).convert("RGB"), (480, 655))
            sheet.paste(image, (x + 10, y + 25))
        sheet.save(folder / f"contact_{offset // 9 + 1:02}.png")
    print(f"Rendered {len(previews)} source pages")


def render_cases() -> None:
    sources = json.loads((DEST / "source_manifest.json").read_text(encoding="utf-8"))
    by_key = {s["key"]: s for s in sources}
    cases = json.loads((DEST / "figure_cases.json").read_text(encoding="utf-8"))
    folder = DEST / "figures"
    folder.mkdir(exist_ok=True)
    atlas = fitz.open()
    for case in cases:
        source = by_key[case["source"]]
        doc = fitz.open(ROOT / source["file"])
        page = doc[case["page"] - 1]
        rect = fitz.Rect(case["rect"])
        page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), clip=rect).save(folder / (case["id"] + ".png"))
        dest = atlas.new_page(width=842, height=595)
        dest.insert_text((30, 28), case["title"], fontsize=13)
        label = f"Source: {case['source']} | PDF page {case['page']} | {case['reference']}"
        dest.insert_text((30, 48), label, fontsize=9)
        dest.insert_text((30, 64), source.get("landing_url", source["file"]), fontsize=8)
        target = fitz.Rect(30, 84, 812, 510)
        dest.show_pdf_page(target, doc, case["page"] - 1, clip=rect, keep_proportion=True)
        dest.insert_textbox(fitz.Rect(30, 525, 812, 575), case["lesson_en"], fontsize=10)
        if source.get("landing_url"):
            dest.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(30, 52, 812, 70), "uri": source["landing_url"]})
    atlas.save(DEST / "metrics_reference_atlas.pdf", deflate=True)
    for offset in range(0, len(cases), 10):
        group = cases[offset:offset + 10]
        sheet = Image.new("RGB", (1500, 1800), "#e4e8ed")
        draw = ImageDraw.Draw(sheet)
        for index, case in enumerate(group):
            x, y = (index % 2) * 750, (index // 2) * 360
            draw.text((x + 12, y + 6), case["id"] + " | " + case["reference"], fill="black")
            image = ImageOps.contain(Image.open(folder / (case["id"] + ".png")).convert("RGB"), (724, 325))
            sheet.paste(image, (x + (750 - image.width) // 2, y + 28 + (325 - image.height) // 2))
        sheet.save(folder / f"case_contact_{offset // 10 + 1:02}.png")
    print(f"Rendered {len(cases)} original excerpts and one vector PDF atlas")


def audit_current() -> None:
    figure = ROOT / "CCDC/images/1020/figure_metrics_grid_single_column_v3.pdf"
    doc = fitz.open(figure)
    page = doc[0]
    # IEEEtran's 21-pica column (252 TeX pt), expressed in PDF points.
    scale = (252 * 72 / 72.27) / page.rect.width
    spans = [s for b in page.get_text("dict")["blocks"] if b["type"] == 0 for line in b["lines"] for s in line["spans"]]
    generator = ROOT / "CORL/scripts/generate_experiment_results_v3.py"
    namespace = runpy.run_path(str(generator), run_name="metrics_review_read_only")
    data = namespace["MAIN_DATA"]
    main = ROOT / "CCDC/latex/1020/main.tex"
    text = main.read_text(encoding="utf-8")
    table = text.split("\\label{tab:main_results}")[1].split("\\end{table*}")[0]
    metrics = ["SR", "CR", "Average Cost", "BVR", "Minimum Clearance"]
    names = {
        "HD-MCRA": "HD-MCRA", "RC-PPO": "RC-PPO", "RA": "RA",
        "PPO_beta_L": r"PPO$_{\beta_L}$", "PPO_beta_H": r"PPO$_{\beta_H}$",
        "CPPO_beta_L": r"CPPO$_{\beta_L}$", "CPPO_beta_H": r"CPPO$_{\beta_H}$",
    }
    comparisons = []
    for method, latex in names.items():
        line = next(line for line in table.splitlines() if line.startswith(latex + " &"))
        values = [float(v) for v in re.findall(r"\\shortstack\{([\d.]+)", line)]
        for metric, value in zip(metrics, values):
            mean = sum(data[metric][method]) / 3
            if metric in ["SR", "CR", "BVR"]:
                mean *= 100
            decimals = 2 if metric == "Minimum Clearance" else 1
            calculated = round(mean, decimals)
            comparisons.append(dict(method=method, metric=metric, manuscript=value, generator_scene_mean=calculated, agrees=abs(value - calculated) < 10 ** (-decimals) / 2))
    sources = json.loads((DEST / "source_manifest.json").read_text(encoding="utf-8"))
    searches = json.loads((DEST / "search_log.json").read_text(encoding="utf-8"))
    dump(DEST / "current_figure_audit.json", {
        "scope": "Existing assumed aggregate values and source definitions only; no new trials, invented variances, or workbook modifications.",
        "workbook_checked": False,
        "figure": figure.relative_to(ROOT).as_posix(),
        "figure_sha256": hashlib.sha256(figure.read_bytes()).hexdigest(),
        "main_tex_sha256": hashlib.sha256(main.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
        "method_count": 7, "scenario_count": 3, "plotted_metric_count": 4, "bar_count": 84,
        "legend_effective_font_pt_approx": [round(s["size"] * scale, 2) for s in spans if s["text"] == "HD-MCRA"],
        "scenario_effective_font_pt_approx": [round(s["size"] * scale, 2) for s in spans if s["text"] == "Mod. Clut."],
        "bvr_scene_placeholders": dict(data["BVR"]),
        "table_source_comparisons": comparisons,
        "inconsistencies": [c for c in comparisons if not c["agrees"]],
        "paper_count": len(sources),
        "new_download_count": sum(s.get("acquisition") == "downloaded_public_original" for s in sources),
        "search_query_count": len(searches),
        "search_results_collected": sum(len(s.get("results", [])) for s in searches),
        "unique_search_result_urls": len({r["url"] for s in searches for r in s.get("results", [])}),
    })
    print(json.dumps({"papers": len(sources), "comparisons": len(comparisons), "inconsistencies": [c for c in comparisons if not c["agrees"]]}, ensure_ascii=True))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["acquire", "scan", "preview", "render", "audit"])
    parser.add_argument("--keys", nargs="*")
    args = parser.parse_args()
    if args.command == "acquire":
        acquire(args.keys)
    elif args.command == "scan":
        scan()
    elif args.command == "preview":
        preview()
    elif args.command == "render":
        render_cases()
    else:
        audit_current()
