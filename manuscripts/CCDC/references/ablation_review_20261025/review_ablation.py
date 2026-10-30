"""Read literature and current assumed ablations; never generate experiment values."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import runpy

from bs4 import BeautifulSoup
import fitz
import requests

ROOT = Path(__file__).resolve().parents[3]
DEST = Path(__file__).resolve().parent
MANIFEST = ROOT / "CCDC/references/metrics_review_20261022/source_manifest.json"
PAGES = {"resilient": [5], "perceptive": [19, 20, 22], "parkour": [18],
         "rliable": [8], "vapor": [6], "abs": [11, 12]}
QUERIES = ["quadruped ablation safety", "robot navigation ablation modules",
           "quadruped locomotion ablation study"]


def render_crops():
    cases = [
        {"id": "perceptive_s2_ablation", "file": "CORL/references/2022 - Learning Robust Perceptive Locomotion for Quadrupedal Robots in the Wild.pdf",
         "page": 22, "rect": [40, 505, 574, 657], "reference": "Fig. S2B"},
        {"id": "parkour_table_s6", "file": "CCDC/references/figure_review_20261020/papers/parkour.pdf",
         "page": 18, "rect": [309, 45, 568, 146], "reference": "Table S6"},
        {"id": "abs_table_vi", "file": "CCDC/references/figure_review_20261020/papers/abs.pdf",
         "page": 12, "rect": [49, 127, 301, 201], "reference": "Table VI"},
    ]
    for case in cases:
        path = DEST/"figures"/(case["id"]+".png")
        path.parent.mkdir(exist_ok=True)
        with fitz.open(ROOT/case["file"]) as pdf:
            pdf[case["page"]-1].get_pixmap(matrix=fitz.Matrix(3, 3), clip=fitz.Rect(case["rect"])).save(path)
        case["image"] = str(path.relative_to(ROOT))
    dump(DEST/"figure_cases.json", cases)


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def online(job):
    kind, value = job
    url = "https://arxiv.org/abs/"+value if kind == "paper" else "https://arxiv.org/search/"
    params = None if kind == "paper" else {"query": value, "searchtype": "all", "abstracts": "show",
                                            "order": "-announced_date_first", "size": 25}
    result = {"kind": kind, "value": value, "retrieved_utc": datetime.now(timezone.utc).isoformat()}
    try:
        response = requests.get(url, params=params, timeout=25)
        result.update(url=response.url, status=response.status_code)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        if kind == "paper":
            heading = soup.select_one("h1.title")
            result["title"] = heading.get_text(" ", strip=True) if heading else soup.title.get_text()
            history = soup.select_one(".submission-history")
            result["online_version_history"] = history.get_text(" ", strip=True) if history else None
        else:
            result["candidates"] = [{"title": x.select_one("p.title").get_text(" ", strip=True),
                                     "url": x.select_one("p.list-title a")["href"]}
                                    for x in soup.select("li.arxiv-result")[:15]]
    except Exception as error:
        result["error"] = str(error)
    return result


def main():
    render_crops()
    sources = json.loads(MANIFEST.read_text(encoding="utf-8"))
    scan = []
    selected = []
    for source in sources:
        file = ROOT/source["file"]
        hits = []
        with fitz.open(file) as pdf:
            for index, page in enumerate(pdf):
                normalized = re.sub(r"-\s*\n", "", page.get_text())
                matches = list(re.finditer(r"ablation|ablative|w/o", normalized, re.I))
                if matches:
                    hits.append({"page": index+1, "excerpts": [re.sub(r"\s+", " ", normalized[max(0,m.start()-120):m.end()+650]) for m in matches[:3]]})
            if source["key"] in PAGES:
                evidence = {"key": source["key"], "file": source["file"],
                            "arxiv_id": source["arxiv_id"], "title": source["title"],
                            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                            "version_note": "Figure references refer to this archived local PDF, not an assumed latest version.",
                            "pages": [{"page": p, "text": pdf[p-1].get_text()} for p in PAGES[source["key"]]]}
                selected.append(evidence)
                if source["key"] in ["perceptive", "parkour"]:
                    page_number = PAGES[source["key"]][-1]
                    path = DEST/"previews"/f"{source['key']}_p{page_number}.png"
                    path.parent.mkdir(exist_ok=True)
                    if not path.exists():
                        pdf[page_number-1].get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(path)
        scan.append({"key": source["key"], "title": source["title"], "ablation_keyword_pages": hits})
    dump(DEST/"literature_scan.json", scan)
    dump(DEST/"selected_source_pages.json", selected)

    tex = (ROOT/"CCDC/latex/1020/main.tex").read_text(encoding="utf-8")
    block = next(b for b in re.findall(r"\\begin\{table\}(.*?)\\end\{table\}", tex, re.S) if "tab:ablation_draft" in b)
    rows = []
    configurations = [(None, None), (0, 1), (1, 0), (0, 0), (1, 1)]
    for line in block.splitlines():
        cells = line.split(" & ")
        if len(cells) == 6 and cells[0] != "Variant":
            numbers = [float(re.search(r"\d+(?:\.\d+)?", c).group()) for c in cells[1:]]
            rows.append({"variant": cells[0], **dict(zip(["SR", "CR", "Cost", "BVR", "MC"], numbers))})
    source = runpy.run_path(str(ROOT/"CORL/scripts/generate_experiment_results_v3.py"), run_name="read_only")
    assert len(rows) == 5
    for row, expected, (alpha, delta) in zip(rows, source["ABLATION_ROWS"], configurations):
        assert [row[k] for k in ["SR", "CR", "Cost", "BVR", "MC"]] == list(expected[1:])
        row.update(alpha_on=alpha, delta_on=delta)
    full = rows[-1]
    differences = [{"variant": row["variant"], "difference_from_full": {k: round(row[k]-full[k], 6) for k in ["SR", "CR", "Cost", "BVR", "MC"]}} for row in rows[:-1]]
    dump(DEST/"current_ablation_snapshot.json", {"data_status": "hypothetical", "source": "CCDC/latex/1020/main.tex:tab:ablation_draft", "rate_units": "percent; differences are percentage points", "rows": rows, "differences": differences, "intervals_available": False, "training_histories_available": False, "per_trial_distributions_available": False})

    ids = [s["arxiv_id"] for s in selected]
    jobs = [("paper", value) for value in ids] + [("search", q) for q in QUERIES]
    with ThreadPoolExecutor(max_workers=4) as pool:
        online_results = list(pool.map(online, jobs))
    dump(DEST/"online_search.json", online_results)
    print(json.dumps({"pdfs_screened": len(scan), "focused_sources": len(selected),
                      "complete_focused_pages": sum(len(s["pages"]) for s in selected),
                      "online_requests": len(online_results),
                      "online_successes": sum(r.get("status") == 200 for r in online_results),
                      "ablation_rows_checked": len(rows),
                      "search_candidates": [{"query": r["value"], "count": len(r.get("candidates", []))} for r in online_results if r["kind"] == "search"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    if args.figures_only:
        render_crops()
    else:
        main()
