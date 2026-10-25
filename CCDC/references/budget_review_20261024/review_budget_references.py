"""Archive and inspect references for a budget-figure design review.

Reads existing manuscript/data files; never changes experimental data or plots.
The downloaded PDF is an unchanged public paper, not an authored document.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from bs4 import BeautifulSoup
import fitz
import requests

ROOT = Path(__file__).resolve().parents[3]
DEST = Path(__file__).resolve().parent
QUERIES = [
    '"budgeted reinforcement learning"',
    '"safe reinforcement learning" "budget"',
    '"constrained policy" "sensitivity"',
]
REFERENCES = [
    ("risk_distributional", "2309.14246", "CCDC/references/metrics_review_20261022/papers/risk_distributional.pdf", 5, "Fig. 6"),
    ("actsafe", "2410.09486", "CCDC/references/metrics_review_20261022/papers/actsafe.pdf", 9, "Fig. 4"),
    ("cpo", "1705.10528", "CORL/references/2017 - Constrained Policy Optimization.pdf", 7, "Fig. 1"),
    ("rcppo", "2410.22600", "CCDC/references/figure_review_20261020/papers/rcppo.pdf", 10, "Fig. 6"),
    ("autorl", "1809.10124", "CORL/references/2019 - Learning Navigation Behaviors End-to-End with AutoRL.pdf", 8, "Fig. 8"),
    ("budgeted_rl", "1903.01004", "CCDC/references/budget_review_20261024/papers/budgeted_rl.pdf", 8, "Fig. 3"),
]


def search(query):
    response = requests.get("https://arxiv.org/search/", params={
        "query": query, "searchtype": "all", "abstracts": "show", "order": "", "size": 50,
    }, timeout=35)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    rows = []
    for item in soup.select("li.arxiv-result"):
        title = item.select_one("p.title")
        anchor = item.select_one("p.list-title a")
        rows.append({"title": title.get_text(" ", strip=True), "url": anchor["href"]})
    return {"query": query, "url": response.url, "status": response.status_code, "results": rows}


def inspect(reference):
    key, arxiv_id, relative_path, page_number, figure = reference
    landing_url = f"https://arxiv.org/abs/{arxiv_id}"
    response = requests.get(landing_url, timeout=35)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.select_one("h1.title")
    path = ROOT / relative_path
    acquisition = "existing_local_original"
    if not path.exists():
        assert key == "budgeted_rl", "Only the newly found reference should be downloaded"
        pdf = requests.get(f"https://arxiv.org/pdf/{arxiv_id}", timeout=40)
        pdf.raise_for_status()
        assert pdf.content.startswith(b"%PDF")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pdf.content)
        acquisition = "downloaded_public_original"
    with fitz.open(path) as document:
        page = document[page_number - 1]
        result = {
            "key": key, "title": title.get_text(" ", strip=True),
            "url": landing_url, "landing_status": response.status_code,
            "local_file": relative_path, "acquisition": acquisition,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "pages": len(document), "read_page": page_number, "figure": figure,
            "page_text": page.get_text(),
        }
        if key == "budgeted_rl":
            previews = DEST / "previews"
            previews.mkdir(exist_ok=True)
            page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6)).save(str(previews / "budgeted_rl_p08.png"))
            page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), clip=fitz.Rect(65, 235, 552, 410)).save(str(previews / "budgeted_rl_fig3.png"))
    return result


def main():
    with ThreadPoolExecutor(max_workers=4) as pool:
        searches = list(pool.map(search, QUERIES))
        references = list(pool.map(inspect, REFERENCES))
    record = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Design research only; manuscript, figures, workbook and assumptions unchanged.",
        "searches": searches, "inspected_references": references,
        "discovery_notes": [
            "General Google/DuckDuckGo access encountered anti-bot pages; Bing results were irrelevant.",
            "Relevant discovery used arXiv search; candidate results are not all read papers.",
            "Existing local PDFs are reused, so their versions may differ from the current online version.",
        ],
    }
    (DEST / "source_review.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"queries": len(searches), "read_papers": len(references), "record": str(DEST / "source_review.json")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
