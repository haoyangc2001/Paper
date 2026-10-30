"""Archive the ten reviewed open-access papers, including two now removed citations."""
import datetime
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
DEST = ROOT.parents[2] / "references" / "1029" / "literature_expansion_20261030"
REMOVED = {"peng2018dynamics", "rudin2022parallel"}


def download(record):
    key = record["key"]
    folder = DEST / ("removed_from_manuscript" if key in REMOVED else "papers")
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (key + ".pdf")
    url = "https://arxiv.org/pdf/" + record["arxiv_id"]
    if target.exists():
        data = target.read_bytes()
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "AcademicReferenceArchive/1.0"})
        data = urllib.request.urlopen(request, timeout=45).read()
    if not data.startswith(b"%PDF-"):
        raise ValueError("Not PDF: " + key)
    with fitz.open(stream=data, filetype="pdf") as document:
        pages = len(document)
        if not pages or not document[0].get_text().strip():
            raise ValueError("Unreadable paper: " + key)
        first_page = " ".join(document[0].get_text().split())
    if not target.exists():
        target.write_bytes(data)
    return {"key": key, "title": record["arxiv"]["title"],
            "retained_in_manuscript": key not in REMOVED,
            "file": target.relative_to(DEST).as_posix(), "source_url": url,
            "primary_url": record["primary_url"], "version": "arXiv open-access manuscript",
            "pages": pages, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "first_page_excerpt": first_page[:1000]}


def main():
    records = json.loads((ROOT / "selected_sources.json").read_text(encoding="utf-8"))["records"]
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(download, record): record["key"] for record in records}
        for future in as_completed(pending):
            key = pending[future]
            try:
                result = future.result()
                results.append(result)
                print(key, "OK", result["pages"], "pages", flush=True)
            except Exception as exc:
                errors.append({"key": key, "error": str(exc)})
                print(key, "FAILED", exc, flush=True)
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "manifest.json").write_text(json.dumps({
        "downloaded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "records": sorted(results, key=lambda r: r["key"]), "errors": errors,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
