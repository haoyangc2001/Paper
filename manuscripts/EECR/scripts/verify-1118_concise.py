"""Check a six-page prose-only revision against its immutable 1118 baseline."""
from __future__ import annotations

import collections
import datetime
import hashlib
import json
import re
from pathlib import Path

import fitz
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
RECORDS = ROOT / "doc/research/20261118_eecr_prose_concision"
SOURCE = BASE / "latex/1118"
TARGET = BASE / "latex/1118_concise"
OUT = BASE / "output/1118_concise"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matches(pattern: str, source: str) -> list[str]:
    return re.findall(pattern, source, re.S)


def main() -> None:
    original = (SOURCE / "main.tex").read_text(encoding="utf-8")
    revised = (TARGET / "main.tex").read_text(encoding="utf-8")
    changes = json.loads((RECORDS / "replacements.json").read_text(encoding="utf-8"))
    reconstructed = original
    for change in changes:
        assert reconstructed.count(change["original"]) == 1, change["id"]
        reconstructed = reconstructed.replace(change["original"], change["revised"], 1)
    assert reconstructed == revised, "Unrecorded source changes"
    assert original.split(r"\section{Introduction}", 1)[0] == revised.split(r"\section{Introduction}", 1)[0]
    protected = {}
    for environment in ["equation", "figure", "table", "abstract", "IEEEkeywords"]:
        pattern = rf"\\begin\{{{environment}\}}.*?\\end\{{{environment}\}}"
        before, after = matches(pattern, original), matches(pattern, revised)
        assert before == after, environment
        protected[environment] = len(before)
    for command in ["cite", "label", "ref", "section", "subsection", "includegraphics"]:
        pattern = rf"\\{command}(?:\[[^\]]*\])?\{{[^}}]*\}}"
        assert collections.Counter(matches(pattern, original)) == collections.Counter(matches(pattern, revised)), command
    assert matches(r"\\cite\{[^}]+\}", original) == matches(r"\\cite\{[^}]+\}", revised)
    def numbers(source: str) -> collections.Counter:
        without_comments = re.sub(r"(?m)(?<!\\)%[^\n]*", "", source)
        return collections.Counter(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", without_comments))
    assert numbers(original) == numbers(revised), "Changed numeric tokens"
    assert sha(SOURCE / "references.bib") == sha(TARGET / "references.bib")
    official = next((BASE / "latex/template/official_latex").rglob("IEEEtran.cls"))
    assert sha(SOURCE / "IEEEtran.cls") == sha(TARGET / "IEEEtran.cls") == sha(official)
    snapshots = json.loads((RECORDS / "before_hashes.json").read_text(encoding="utf-8"))
    allowed = {"manuscripts/EECR/README.md"}
    changed = [name for name, expected in snapshots.items() if name not in allowed and (not (ROOT / name).is_file() or sha(ROOT / name) != expected)]
    assert not changed, changed
    log = (OUT / "main.log").read_text(encoding="utf-8", errors="replace")
    undefined = re.findall(r"[^\n]*(?:undefined|There were undefined references)[^\n]*", log, re.I)
    overfull = re.findall(r"Overfull \\[hv]box[^\n]*", log)
    assert not undefined and not overfull, (undefined, overfull)
    bbl = (OUT / "main.bbl").read_text(encoding="utf-8")
    bibliography_keys = matches(r"\\bibitem\{([^}]+)\}", bbl)
    assert len(bibliography_keys) == 16
    doc = fitz.open(OUT / "main.pdf")
    assert len(doc) == 6, f"Expected exactly six pages, got {len(doc)}"
    outside, fonts, pages = [], {}, []
    qa = OUT / "qa"
    qa.mkdir(exist_ok=True)
    sheet = Image.new("RGB", (1400, 2940), "#dddddd")
    draw = ImageDraw.Draw(sheet)
    for index, page in enumerate(doc):
        assert abs(page.rect.width - 612) < 1 and abs(page.rect.height - 792) < 1
        for block in page.get_text("blocks"):
            if block[4].strip() and not (page.rect + (-1, -1, 1, 1)).contains(fitz.Rect(block[:4])):
                outside.append({"page": index + 1, "bbox": list(block[:4])})
        for font in page.get_fonts(full=True):
            if font[0] not in fonts:
                name, extension, kind, data = doc.extract_font(font[0])
                fonts[font[0]] = {"name": name, "kind": kind, "embedded": bool(data)}
        pixmap = page.get_pixmap(matrix=fitz.Matrix(120/72, 120/72), alpha=False)
        pixmap.save(qa / f"page-{index + 1:02d}.png")
        thumbnail = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        thumbnail.thumbnail((680, 940))
        x, y = index % 2 * 700, index // 2 * 980
        sheet.paste(thumbnail, (x + 10, y + 25))
        draw.text((x + 12, y + 6), f"EECR concise page {index + 1}", fill="black")
        pages.append({"page": index + 1, "words_pdf_extraction": len(page.get_text().split())})
    sheet.save(qa / "contact-sheet.png")
    assert not outside
    assert all(font["embedded"] for font in fonts.values())
    all_text = "\n".join(page.get_text() for page in doc)
    reference_text = all_text.split("REFERENCES", 1)[1]
    reference_numbers = [int(number) for number in re.findall(r"^\[(\d+)\]", reference_text, re.M)]
    assert reference_numbers == list(range(1, 17)), reference_numbers
    result = {
        "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "baseline": "latex/1118/main.tex", "revision": "latex/1118_concise/main.tex",
        "recorded_prose_replacements": len(changes), "source_exactly_matches_replacement_record": True,
        "protected_environments_verbatim": protected,
        "preamble_title_authors_abstract_keywords_unchanged": True,
        "all_rendered_numeric_token_spellings_and_counts_unchanged": True,
        "citations_in_same_order": True, "labels_refs_headings_graphics_preserved": True,
        "bibliography_verbatim": True, "official_class_verbatim": True,
        "protected_existing_files_checked": len(snapshots) - len(allowed), "protected_existing_files_changed": changed,
        "allowed_existing_file_changes": sorted(allowed),
        "pdf_sha256": sha(OUT / "main.pdf"), "pages": len(doc), "reference_count": 16,
        "page_format": "IEEEtran conference, 10pt, two-column, Letter; original preamble unchanged",
        "all_fonts_embedded": True, "text_outside_page": outside,
        "undefined_references": undefined, "overfull_boxes": overfull,
        "compile_warnings": re.findall(r"(?:LaTeX|Package [^\n]+) Warning:[^\n]*", log),
        "rendered_pages": pages, "six_page_target_met": True,
        "manual_semantic_review": "See semantic_review.md; numerical/source checks alone do not prove semantic equivalence",
        "manual_visual_review": "See visual_review.md; tied to the PDF SHA256",
        "scope_limit": "Concision and source/layout consistency; no new empirical validation of research claims",
    }
    (qa / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pages": 6, "references": 16, "recorded_prose_replacements": len(changes),
                      "unchanged_equations": protected["equation"], "unchanged_figures": protected["figure"],
                      "unchanged_tables": protected["table"], "protected_changed_files": changed,
                      "overfull_boxes": overfull, "undefined_references": undefined,
                      "pdf_sha256": result["pdf_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
