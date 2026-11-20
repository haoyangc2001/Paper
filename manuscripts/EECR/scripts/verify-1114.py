"""Check migration identity, references, assets, protected files, and PDF layout."""
from __future__ import annotations

import datetime
import hashlib
import json
import re
from pathlib import Path

import fitz
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
SOURCE = BASE.parent / "CCDC"
OUT = BASE / "output/1114"
SUB = BASE / "submission/1114"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    original = (SOURCE / "latex/1029/main.tex").read_bytes()
    migrated = (BASE / "latex/1114/main.tex").read_bytes()
    restored = migrated.replace(
        b"% EECR official IEEEtran conference format. [ADAPTED: venue annotation only]",
        b"% CCDC official conference format. Do not override its layout or typography.",
    ).replace(b"../../images/1114/", b"../../images/1029/")
    assert original == restored, "Unexpected manuscript edit"
    assert sha(SOURCE / "latex/1029/references.bib") == sha(BASE / "latex/1114/references.bib")
    official = next((BASE / "latex/template/official_latex").rglob("IEEEtran.cls"))
    assert sha(official) == sha(BASE / "latex/1114/IEEEtran.cls")
    assets = json.loads((SUB / "asset_manifest.json").read_text(encoding="utf-8"))
    assert all(sha(REPO / a["source"]) == sha(REPO / a["target"]) == a["sha256"] for a in assets)
    before = json.loads((SUB / "protected_before_hashes.json").read_text(encoding="utf-8"))
    changed = [name for name, expected in before.items() if not (REPO / name).is_file() or sha(REPO / name) != expected]
    assert not changed, changed
    source_records = json.loads((SUB / "official_sources/index.json").read_text(encoding="utf-8"))
    downloads = json.loads((BASE / "latex/template/download_manifest.json").read_text(encoding="utf-8"))
    assert all(sha(BASE / r["file"]) == r["sha256"] for r in source_records + downloads)

    old = fitz.open(SOURCE / "output/1029/main.pdf")
    new = fitz.open(OUT / "main.pdf")
    assert len(old) == len(new)
    comparison, fonts, outside = [], {}, []
    pages_dir = OUT / "qa/pages/main"
    pages_dir.mkdir(parents=True, exist_ok=True)
    cell_w, cell_h = 540, 730
    sheet = Image.new("RGB", (cell_w * 2, cell_h * ((len(new) + 1) // 2)), "#dddddd")
    draw = ImageDraw.Draw(sheet)
    for i, page in enumerate(new):
        original_page = old[i]
        old_pix = original_page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        new_pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        comparison.append({"page": i + 1, "text_equal": original_page.get_text() == page.get_text(), "render_equal_108dpi": old_pix.samples == new_pix.samples})
        for font in page.get_fonts(full=True):
            xref = font[0]
            if xref not in fonts:
                name, ext, kind, content = new.extract_font(xref)
                fonts[xref] = {"name": name, "type": kind, "embedded": bool(content)}
        for block in page.get_text("blocks"):
            if block[4].strip() and not (page.rect + (-1, -1, 1, 1)).contains(fitz.Rect(block[:4])):
                outside.append({"page": i + 1, "bbox": list(block[:4])})
        pix = page.get_pixmap(matrix=fitz.Matrix(120 / 72, 120 / 72), alpha=False)
        pix.save(pages_dir / f"page-{i + 1:02d}.png")
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        img.thumbnail((cell_w - 20, cell_h - 40))
        x, y = (i % 2) * cell_w, (i // 2) * cell_h
        draw.text((x + 12, y + 8), f"EECR: page {i + 1}", fill="black")
        sheet.paste(img, (x + 10, y + 30))
    sheet.save(OUT / "qa/main-contact.png")
    assert all(c["text_equal"] and c["render_equal_108dpi"] for c in comparison), comparison
    assert not outside, outside
    assert all(f["embedded"] for f in fonts.values()), fonts
    log = (OUT / "main.log").read_text(encoding="utf-8", errors="replace")
    undefined = re.findall(r"[^\n]*(?:undefined|There were undefined references)[^\n]*", log, re.I)
    overfull = re.findall(r"Overfull \\[hv]box[^\n]*", log)
    assert not undefined, undefined
    assert not overfull, overfull
    text = "\n".join(page.get_text() for page in new)
    results = {
        "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Migration and formatting only; not validation of research claims or permission to submit",
        "source_content_preserved": True, "bibliography_identical": True,
        "official_IEEEtran_class_identical": True, "identical_figure_assets": len(assets),
        "protected_files_checked": len(before), "protected_changed_files": changed,
        "official_source_hashes_verified": len(source_records) + len(downloads),
        "pdf": {
            "sha256": sha(OUT / "main.pdf"), "pages": len(new),
            "page_sizes_pt": [list(p.rect) for p in new], "fonts": list(fonts.values()),
            "all_fonts_embedded": True, "encrypted": bool(new.is_encrypted),
            "text_blocks_outside_page": outside,
            "reference_count": len(re.findall(r"^\[\d+\]", re.split(r"\bREFERENCES\b", text)[-1], re.M)),
            "contact_sheet": "output/1114/qa/main-contact.png",
        },
        "named_pdf_comparison": comparison,
        "compile_log": {"undefined_references_or_citations": undefined, "overfull_boxes": overfull,
                        "warnings": re.findall(r"(?:LaTeX|Package [^\n]+) Warning:[^\n]*", log)},
        "venue_page_range": [4, 10], "page_requirement_met": 4 <= len(new) <= 10,
        "registration_included_pages": 6, "extra_pages": max(0, len(new) - 6),
        "anonymous_review_requirement": "not specified on checked public pages",
        "parallel_submission_permission": "not established", "submission_performed": False,
        "manual_visual_review": "See visual_review.md for evidence tied to PDF hash",
    }
    (OUT / "qa/verification.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pages": len(new), "references": results["pdf"]["reference_count"], "source_content_preserved": True,
                      "identical_images": len(assets), "protected_files_checked": len(before), "pdf_text_and_pixels_equal": True,
                      "page_requirement_met": results["page_requirement_met"], "extra_pages": results["extra_pages"],
                      "pdf_sha256": results["pdf"]["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
