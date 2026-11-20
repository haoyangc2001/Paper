"""Audit the authorized EECR reference reduction without changing older versions."""
from __future__ import annotations

import datetime
import hashlib
import json
import re
from pathlib import Path

import fitz

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
RESEARCH = ROOT / "doc/research/20261118_eecr_reference_reduction"
OLD = BASE / "latex/1114"
NEW = BASE / "latex/1118"
OUT = BASE / "output/1118"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cited(text: str) -> set[str]:
    return {
        key.strip()
        for group in re.findall(r"\\cite\w*\{([^}]+)\}", text)
        for key in group.split(",")
    }


def entries(text: str) -> dict[str, str]:
    return {
        match.group(1): match.group(0).strip()
        for match in re.finditer(r"@\w+\{([^,]+),.*?(?=\n@|\Z)", text, re.S)
    }


def main() -> None:
    original = (OLD / "main.tex").read_text(encoding="utf-8")
    revised = (NEW / "main.tex").read_text(encoding="utf-8")
    removed = set(json.loads((RESEARCH / "removed_citations.json").read_text(encoding="utf-8")))
    assert cited(original) - cited(revised) == removed
    assert not cited(revised) - cited(original)
    assert len(cited(original)) == 23 and len(cited(revised)) == 16
    # The scientific method, experiments, results, captions and conclusion are protected.
    method_start = r"\section{Problem Formulation}"
    assert original.split(method_start, 1)[1] == revised.split(method_start, 1)[1]
    assert original.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0] == revised.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0]
    old_entries = entries((OLD / "references.bib").read_text(encoding="utf-8"))
    new_entries = entries((NEW / "references.bib").read_text(encoding="utf-8"))
    assert set(new_entries) == cited(revised)
    assert all(old_entries[key] == value for key, value in new_entries.items())
    official_class = next((BASE / "latex/template/official_latex").rglob("IEEEtran.cls"))
    assert sha(official_class) == sha(NEW / "IEEEtran.cls") == sha(OLD / "IEEEtran.cls")
    snapshots = json.loads((RESEARCH / "before_hashes.json").read_text(encoding="utf-8"))
    allowed_modified = {"manuscripts/EECR/README.md"}
    changed = [name for name, expected in snapshots.items() if name not in allowed_modified and (not (ROOT / name).is_file() or sha(ROOT / name) != expected)]
    assert not changed, changed
    assets = json.loads((BASE / "submission/1114/asset_manifest.json").read_text(encoding="utf-8"))
    assert all(sha(ROOT / item["target"]) == item["sha256"] for item in assets)
    log = (OUT / "main.log").read_text(encoding="utf-8", errors="replace")
    undefined = re.findall(r"[^\n]*(?:undefined|There were undefined references)[^\n]*", log, re.I)
    overfull = re.findall(r"Overfull \\[hv]box[^\n]*", log)
    assert not undefined and not overfull, (undefined, overfull)
    bbl = (OUT / "main.bbl").read_text(encoding="utf-8")
    assert set(re.findall(r"\\bibitem\{([^}]+)\}", bbl)) == cited(revised)
    doc = fitz.open(OUT / "main.pdf")
    all_text = "\n".join(page.get_text() for page in doc)
    references = all_text.split("REFERENCES", 1)[1]
    numbers = [int(number) for number in re.findall(r"^\[(\d+)\]", references, re.M)]
    assert numbers == list(range(1, 17)), numbers
    outside, fonts = [], {}
    for index, page in enumerate(doc, 1):
        assert abs(page.rect.width - 612) < 1 and abs(page.rect.height - 792) < 1
        for block in page.get_text("blocks"):
            if block[4].strip() and not (page.rect + (-1, -1, 1, 1)).contains(fitz.Rect(block[:4])):
                outside.append({"page": index, "bbox": list(block[:4])})
        for font in page.get_fonts(full=True):
            if font[0] not in fonts:
                name, extension, kind, data = doc.extract_font(font[0])
                fonts[font[0]] = {"name": name, "embedded": bool(data)}
    assert not outside
    assert all(font["embedded"] for font in fonts.values())
    result = {
        "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Reference selection and corresponding background examples only; no research-result validation",
        "source": "latex/1114/main.tex", "revision": "latex/1118/main.tex",
        "references_before": 23, "references_after": 16, "removed_keys": sorted(removed),
        "retained_bibliographic_entries_verbatim": True,
        "abstract_unchanged": True, "method_results_figures_conclusion_source_unchanged": True,
        "protected_existing_files_checked": len(snapshots) - len(allowed_modified),
        "protected_files_changed": changed, "allowed_existing_file_changes": sorted(allowed_modified),
        "original_image_assets_unchanged": len(assets), "official_class_unchanged": True,
        "pdf_sha256": sha(OUT / "main.pdf"), "pdf_pages": len(doc),
        "pdf_references_in_order": numbers, "all_fonts_embedded": True,
        "text_outside_pages": outside, "undefined_references": undefined, "overfull_boxes": overfull,
        "compile_warnings": re.findall(r"(?:LaTeX|Package [^\n]+) Warning:[^\n]*", log),
        "six_page_fee_target_met": len(doc) <= 6,
        "known_limit": "Reference reduction is complete, but the paper remains seven pages.",
        "visual_review": "qa/visual_review.md",
    }
    (OUT / "qa").mkdir(exist_ok=True)
    (OUT / "qa/verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pages": len(doc), "references": 16, "protected_files_changed": changed,
                      "pdf_sha256": result["pdf_sha256"], "six_page_target_met": result["six_page_fee_target_met"]}, indent=2))


if __name__ == "__main__":
    main()
