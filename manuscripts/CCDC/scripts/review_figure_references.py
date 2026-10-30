"""Build a local reading set and render original paper pages for figure review."""
import hashlib
import json
import re
import shutil
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "CCDC/references/figure_review_20261020"
SELECTION = {
    "rcppo": "2024 - Solving Minimum-Cost Reach Avoid using Reinforcement Learning.pdf",
    "abs": "2024 - Agile But Safe Learning Collision-Free High-Speed Legged Locomotion.pdf",
    "parkour": "2024 - ANYmal Parkour Learning Agile Navigation for Quadrupedal Robots.pdf",
    "confined": "2024 - Learning to Walk in Confined Spaces Using 3D Representation.pdf",
    "resilient": "Resilient Legged Local Navigation Learning to Traverse with Compromised Perception End-to-End.pdf",
    "sea_nav": "SEA-Nav Efficient Policy Learning for Safe and Agile Quadruped Navigation in Cluttered Environments.pdf",
}

# Manually checked page-relative rectangles. These are reading excerpts of
# attributed original figures, never manuscript assets or new experimental data.
EXCERPTS = {
    "sea_nav": (6, (0.075, 0.045, 0.94, 0.235), "trajectory_velocity"),
    "parkour": (6, (0.09, 0.085, 0.92, 0.805), "hardware_synchronized_traces"),
    "resilient": (5, (0.085, 0.06, 0.495, 0.265), "event_and_command_tracking"),
    "confined": (5, (0.09, 0.055, 0.92, 0.267), "difficulty_heatmap"),
    "abs": (10, (0.08, 0.235, 0.49, 0.421), "safety_efficiency_tradeoff"),
    "rcppo": (9, (0.10, 0.30, 0.92, 0.495), "trajectory_and_cost"),
}


def main():
    (DEST / "papers").mkdir(parents=True, exist_ok=True)
    (DEST / "previews").mkdir(exist_ok=True)
    (DEST / "text").mkdir(exist_ok=True)
    (DEST / "figures").mkdir(exist_ok=True)
    manifest = []
    for key, name in SELECTION.items():
        source = ROOT / "CORL/references" / name
        target = DEST / "papers" / (key + ".pdf")
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise RuntimeError(f"Existing reference differs: {target}")
        if not target.exists():
            shutil.copy2(source, target)
        doc = fitz.open(target)
        pages = []
        for index, page in enumerate(doc):
            text = page.get_text()
            pages.append(f"\n\n--- PDF PAGE {index + 1} ---\n{text}")
            # Main-paper pages only: avoid rendering long appendix/reference runs.
            if index < min(len(doc), 9):
                image_path = DEST / "previews" / f"{key}_p{index + 1:02}.png"
                page.get_pixmap(matrix=fitz.Matrix(1.7, 1.7), alpha=False).save(image_path)
        (DEST / "text" / (key + ".txt")).write_text("".join(pages), encoding="utf-8")
        count = min(len(doc), 9)
        sheet = Image.new("RGB", (3 * 410, ((count + 2) // 3) * 570), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for index in range(count):
            with Image.open(DEST / "previews" / f"{key}_p{index + 1:02}.png") as im:
                thumb = ImageOps.contain(im, (400, 540))
                x, y = (index % 3) * 410, (index // 3) * 570
                sheet.paste(thumb, (x, y + 24))
                draw.text((x + 8, y + 6), f"{key} / PDF p.{index + 1}", fill="black")
        sheet.save(DEST / "previews" / (key + "_contact.png"))
        links = sorted({str(link.get("uri")) for page in doc for link in page.get_links() if link.get("uri")})
        page_number, coordinates, label = EXCERPTS[key]
        page = doc[page_number - 1]
        x0, y0, x1, y1 = coordinates
        clip = fitz.Rect(x0 * page.rect.width, y0 * page.rect.height,
                         x1 * page.rect.width, y1 * page.rect.height)
        page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), clip=clip, alpha=False).save(
            DEST / "figures" / f"{key}_{label}.png")
        item = {
            "id": key, "source": str(source.relative_to(ROOT)),
            "file": str(target.relative_to(DEST)), "pages": len(doc),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "provenance": "existing local reference; not newly downloaded",
            "links_in_pdf": links,
        }
        manifest.append(item)
        print(key, len(doc), "pages", "links:", json.dumps(links[:12], ensure_ascii=False))
    (DEST / "local_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
