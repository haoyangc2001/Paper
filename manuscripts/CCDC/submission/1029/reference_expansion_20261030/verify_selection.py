"""Save primary-source metadata and citation evidence for ten newly cited works."""
import datetime
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import fitz
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
SELECTION = [
    ("hsu2021", "2112.12288", "https://www.roboticsproceedings.org/rss17/p077.html", "reach-avoid objective", True),
    ("jain2021pixels", "2011.11722", "https://proceedings.mlr.press/v155/jain21a.html", "hierarchical navigation and reusable low-level skills", False),
    ("hoeller2024", "2306.14874", "https://api.crossref.org/works/10.1126/scirobotics.adi7566", "high-level selection of locomotion skills", True),
    ("rudin2022parallel", "2109.11978", "https://proceedings.mlr.press/v164/rudin22a.html", "parallel reinforcement-learning locomotion training", False),
    ("tan2018simreal", "1804.10332", "https://www.roboticsproceedings.org/rss14/p10.html", "quadruped sim-to-real dynamics and latency randomization", False),
    ("peng2018dynamics", "1710.06537", "https://api.crossref.org/works/10.1109/ICRA.2018.8460528", "dynamics randomization for sim-to-real control", False),
    ("xie2021randomization", "2011.02404", "https://api.crossref.org/works/10.1109/ICRA48506.2021.9560837", "randomization effectiveness depends on control design", False),
    ("herbert2017fastrack", "1703.07373", "https://api.crossref.org/works/10.1109/CDC.2017.8263867", "planning-tracking mismatch and reachability-based tracking bounds", False),
    ("yang2023iplanner", "2302.11434", "https://www.roboticsproceedings.org/rss19/p064.html", "differentiable-cost-map geometric local planning", False),
    ("jenelten2024dtc", "2309.15462", "https://api.crossref.org/works/10.1126/scirobotics.adh5401", "model-based planning combined with robust learned tracking", False),
]


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "ManuscriptLiteratureReview/1.0"})
    for attempt in range(3):
        try:
            return urllib.request.urlopen(request, timeout=35).read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(4 + 3 * attempt)


def main():
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    ids = ",".join(row[1] for row in SELECTION)
    feed = ET.fromstring(get("https://export.arxiv.org/api/query?id_list=" + ids + "&max_results=20"))
    arxiv = {}
    for e in feed.findall("a:entry", ns):
        identifier = re.sub(r"v\d+$", "", e.findtext("a:id", "", ns).rsplit("/", 1)[-1])
        arxiv[identifier] = {
            "title": " ".join(e.findtext("a:title", "", ns).split()),
            "authors": [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)],
            "abstract": " ".join(e.findtext("a:summary", "", ns).split()),
            "doi": e.findtext("x:doi", "", ns),
        }
    records = []
    for key, identifier, primary_url, purpose, existing in SELECTION:
        primary = get(primary_url)
        if "api.crossref.org/works/" in primary_url:
            metadata = json.loads(primary)["message"]
            primary_title = metadata["title"][0]
            doi = metadata["DOI"]
            bibtex_url = "https://api.crossref.org/works/" + doi + "/transform/application/x-bibtex"
            bibtex = get(bibtex_url).decode("utf-8")
            primary_excerpt = metadata.get("abstract", "")
        else:
            soup = BeautifulSoup(primary, "html.parser")
            primary_title = (soup.find("meta", attrs={"name": "citation_title"}) or {}).get("content", "")
            text = soup.get_text(" ", strip=True)
            match = re.search(r"@(InProceedings|INPROCEEDINGS)\{.*?\n\}", text, re.S)
            if not match:
                raise ValueError("Primary BibTeX not found: " + primary_url)
            bibtex = match.group()
            primary_excerpt = text[:10000]
            bibtex_url = primary_url
            metadata = {"title": primary_title, "source": primary_url}
        record = {"key": key, "arxiv_id": identifier, "arxiv_url": "https://arxiv.org/abs/" + identifier,
                  "primary_url": primary_url, "bibtex_url": bibtex_url,
                  "already_in_bibliography_but_not_cited": existing,
                  "citation_purpose": purpose, "arxiv": arxiv[identifier],
                  "primary_metadata": metadata, "primary_excerpt": primary_excerpt,
                  "retrieved_bibtex": bibtex}
        records.append(record)
        print(key, arxiv[identifier]["title"], "VERIFIED_TWO_SOURCES", flush=True)
        (ROOT / "selected_sources.json").write_text(json.dumps({"retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        time.sleep(1.2)
    snippets = {}
    for key, identifier, _, _, _ in SELECTION:
        if key not in {"peng2018dynamics", "xie2021randomization", "herbert2017fastrack", "jenelten2024dtc", "rudin2022parallel"}:
            continue
        document = fitz.open(stream=get("https://arxiv.org/pdf/" + identifier), filetype="pdf")
        snippets[key] = []
        for i, page in enumerate(document):
            if i >= 8:
                break
            text = " ".join(page.get_text().split())
            patterns = {"peng2018dynamics": "randomiz", "xie2021randomization": "without dynamics randomization", "herbert2017fastrack": "tracking error bound", "jenelten2024dtc": "track", "rudin2022parallel": "PPO"}
            for match in list(re.finditer(patterns[key], text, re.I))[:3]:
                snippets[key].append({"page": i + 1, "excerpt": text[max(0, match.start()-160):match.end()+360]})
        print(key, "FULLTEXT_EVIDENCE", len(snippets[key]), flush=True)
    (ROOT / "fulltext_evidence.json").write_text(json.dumps(snippets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
