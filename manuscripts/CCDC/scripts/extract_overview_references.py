"""Render attributed overview excerpts from the existing local reference PDFs.

These excerpts are for visual discussion, not assets for our manuscript.
Run with the base Miniconda Python, which provides PyMuPDF.
"""

from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[2]
REFERENCES = ROOT / "CCDC/references/figure_review_20261020"
EXCERPTS = (
    ("parkour", 4, (0.075, 0.057, 0.925, 0.610)),
    ("confined", 3, (0.160, 0.057, 0.840, 0.390)),
)


def main():
    destination = REFERENCES / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    for key, page_number, coordinates in EXCERPTS:
        with fitz.open(REFERENCES / "papers" / f"{key}.pdf") as document:
            page = document[page_number - 1]
            x0, y0, x1, y1 = coordinates
            clip = fitz.Rect(
                x0 * page.rect.width,
                y0 * page.rect.height,
                x1 * page.rect.width,
                y1 * page.rect.height,
            )
            output = destination / f"{key}_overview_fig2.png"
            page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), clip=clip, alpha=False).save(output)
            print(f"{key}: PDF page {page_number}, Fig. 2 -> {output}")


if __name__ == "__main__":
    main()
