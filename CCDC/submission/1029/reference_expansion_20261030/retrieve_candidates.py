"""Retrieve identifiable literature candidates; never modifies manuscript sources."""
import datetime
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
QUERIES = [
    ("quadruped_navigation_locomotion", "(ti:quadruped OR ti:legged) AND (ti:navigation OR ti:locomotion)", 30),
    ("hierarchy_perceptive_control", "(ti:hierarchical OR ti:perceptive) AND (ti:quadruped OR ti:legged OR ti:locomotion)", 25),
    ("reach_avoid_learning", "ti:reach AND ti:avoid AND (all:reinforcement OR all:robot)", 25),
    ("sim_to_real_randomization", "(ti:sim-to-real OR ti:randomization) AND (all:locomotion OR all:robot)", 25),
    ("safe_robot_learning", "ti:safe AND ti:reinforcement AND ti:learning AND (all:robot OR all:constraint)", 20),
    ("velocity_tracking", "ti:tracking AND ti:control AND (all:quadruped OR all:legged)", 15),
]
NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}


def content(entry, key):
    return " ".join((entry.findtext(key, "", NS)).split())


def main():
    papers = {}
    searches = []
    for label, query, limit in QUERIES:
        params = urllib.parse.urlencode({"search_query": query, "max_results": limit, "sortBy": "relevance"})
        url = "https://export.arxiv.org/api/query?" + params
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ManuscriptLiteratureReview/1.0"})
            feed = ET.fromstring(urllib.request.urlopen(request, timeout=40).read())
            entries = feed.findall("a:entry", NS)
            searches.append({"topic": label, "url": url, "returned": len(entries)})
            for entry in entries:
                identifier = re.sub(r"v\d+$", "", content(entry, "a:id").rsplit("/", 1)[-1])
                if identifier in papers:
                    papers[identifier]["topics"].append(label)
                    continue
                title = content(entry, "a:title")
                exclusion = ""
                if re.search(r"stochastic", title, re.I) and re.search(r"minim(?:um|al)[ -]cost|cost", title, re.I) and re.search(r"reach", title, re.I):
                    exclusion = "User excluded stochastic minimum-cost reach-avoid work."
                elif re.search(r"language|semantic|vision-language", title, re.I):
                    exclusion = "Future semantic/language capability rather than current geometric method."
                papers[identifier] = {
                    "arxiv_id": identifier, "title": title,
                    "authors": [content(author, "a:name") for author in entry.findall("a:author", NS)],
                    "published": content(entry, "a:published"), "updated": content(entry, "a:updated"),
                    "abstract": content(entry, "a:summary"), "doi": content(entry, "x:doi"),
                    "journal_reference": content(entry, "x:journal_ref"),
                    "url": "https://arxiv.org/abs/" + identifier,
                    "topics": [label], "exclusion": exclusion,
                }
            print(label, len(entries), "unique", len(papers), flush=True)
        except Exception as exc:
            searches.append({"topic": label, "url": url, "error": str(exc)})
            print(label, "ERROR", exc, flush=True)
        time.sleep(3.1)
    result = {"retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "scope": "Current geometric quadruped MCRA, hierarchy, execution mismatch, safe learning; no future semantic or stochastic MCRA additions.",
              "searches": searches, "unique_count": len(papers), "papers": list(papers.values())}
    (ROOT / "candidates.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for i, paper in enumerate(papers.values(), 1):
        print(f"{i:03d}\t{paper['arxiv_id']}\t{paper['title']}\t{paper['exclusion']}")


if __name__ == "__main__":
    main()
