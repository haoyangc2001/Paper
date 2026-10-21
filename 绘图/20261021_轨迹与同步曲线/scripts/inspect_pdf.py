"""Read-only PDF inspection. Run with existing base Python/PyMuPDF."""
import argparse,json
from pathlib import Path
import fitz
p=argparse.ArgumentParser();p.add_argument("pdf",type=Path);p.add_argument("--report",type=Path,required=True)
a=p.parse_args();doc=fitz.open(a.pdf);page=doc[0]
spans=[s for b in page.get_text("dict")["blocks"] if "lines" in b for l in b["lines"] for s in l["spans"]]
outside=[s["text"] for s in spans if not page.rect.contains(fitz.Rect(s["bbox"]))]
text=page.get_text()
result=dict(pages=len(doc),width_mm=page.rect.width*25.4/72,height_mm=page.rect.height*25.4/72,
            embedded_raster_images=len(page.get_images()),vector_drawings=len(page.get_drawings()),
            text_spans=len(spans),font_sizes=sorted(set(round(s["size"],2) for s in spans)),
            text_outside_page=outside,assumed_label="ASSUMED DATA" in text,
            panel_labels=all(x in text for x in ["(a)","(b)","(c)","(d)"]))
result["passed"]=(len(doc)==1 and abs(result["width_mm"]-184.2)<.01 and
                  not outside and result["embedded_raster_images"]==0 and
                  result["vector_drawings"]>0 and result["assumed_label"] and result["panel_labels"])
a.report.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2));raise SystemExit(0 if result["passed"] else 1)

