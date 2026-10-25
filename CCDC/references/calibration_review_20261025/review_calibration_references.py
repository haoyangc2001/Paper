"""Read-only paper research: acquire sources and render reference pages."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import sys

import fitz
import requests
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
SOURCES = [
    {"id": "gentle_conformal", "url": "https://arxiv.org/pdf/2107.07511v6"},
    {"id": "safe_planning", "url": "https://arxiv.org/pdf/2210.10254v2"},
    {"id": "knowno", "url": "https://arxiv.org/pdf/2307.01928v2"},
    {
        "id": "rliable",
        "url": "https://arxiv.org/abs/2108.13264",
        "local": "CCDC/references/metrics_review_20261022/papers/rliable.pdf",
    },
]


def acquire(source: dict) -> dict:
    path = WORKSPACE / source["local"] if "local" in source else ROOT / "papers" / (source["id"] + ".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        response = requests.get(source["url"], timeout=90)
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            raise ValueError(f"Not PDF: {source['url']}")
        path.write_bytes(response.content)
    doc = fitz.open(path)
    blocks = []
    for page_number, page in enumerate(doc, start=1):
        blocks.append(f"\n--- PDF PAGE {page_number} ---\n{page.get_text()}")
    text = "\n".join(blocks)
    text_path = ROOT / "text" / (source["id"] + ".txt")
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text, encoding="utf-8")
    figures = []
    for match in re.finditer(r"(?:Figure|Fig\.)\s*\d+\s*[:.]", text):
        figures.append(text[max(0, match.start() - 100):match.start() + 1200])
    result = {
        **source,
        "path": str(path),
        "pages": len(doc),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "first_page": doc[0].get_text()[:700],
        "figure_contexts": figures,
    }
    doc.close()
    return result


def render_cases() -> None:
    cases = [
        {
            "id": "rliable_fig7",
            "path": "CCDC/references/metrics_review_20261022/papers/rliable.pdf",
            "page": 7,
            "figure": "7",
            "clip_pt": [125, 74, 488, 224],
            "title": "A. rliable: distribution profiles",
            "subtitle": "Shared legend, reference line, restrained styling",
        },
        {
            "id": "safe_planning_fig2",
            "path": "CCDC/references/calibration_review_20261025/papers/safe_planning.pdf",
            "page": 10,
            "figure": "2",
            "clip_pt": [162, 69, 434, 202],
            "title": "B. Safe Planning: errors + thresholds",
            "subtitle": "Calibration distribution; separate held-out checks",
        },
        {
            "id": "knowno_fig3",
            "path": "CCDC/references/calibration_review_20261025/papers/knowno.pdf",
            "page": 6,
            "figure": "3",
            "clip_pt": [340, 554, 510, 690],
            "title": "C. KnowNo: target vs. observed",
            "subtitle": "Report the gap from a specified target",
        },
    ]
    figure_dir = ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)
    tiles = []
    font_path = "C:/Windows/Fonts/arial.ttf"
    title_font = ImageFont.truetype(font_path, 30)
    subtitle_font = ImageFont.truetype(font_path, 23)
    for case in cases:
        doc = fitz.open(WORKSPACE / case["path"])
        pix = doc[case["page"] - 1].get_pixmap(matrix=fitz.Matrix(4, 4), clip=fitz.Rect(case["clip_pt"]), alpha=False)
        out_path = figure_dir / (case["id"] + ".png")
        pix.save(out_path)
        doc.close()
        crop = Image.open(out_path).convert("RGB")
        tile = Image.new("RGB", (1200, 670), "white")
        draw = ImageDraw.Draw(tile)
        draw.text((34, 22), case["title"], fill="#263544", font=title_font)
        draw.text((34, 68), case["subtitle"], fill="#687581", font=subtitle_font)
        crop.thumbnail((1120, 520), Image.Resampling.LANCZOS)
        tile.paste(crop, ((1200 - crop.width) // 2, 130 + (520 - crop.height) // 2))
        tiles.append(tile)
    atlas = Image.new("RGB", (1200, len(tiles) * 670), "white")
    for index, tile in enumerate(tiles):
        atlas.paste(tile, (0, index * 670))
    atlas.save(figure_dir / "reference_cases.png")
    (ROOT / "figure_cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(str(figure_dir / "reference_cases.png"))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1] == "render":
        render_cases()
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(acquire, SOURCES))
    (ROOT / "source_manifest.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for result in results:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
