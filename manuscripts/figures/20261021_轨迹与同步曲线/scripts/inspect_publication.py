"""Inspect the paper-layout PDFs using the existing base PyMuPDF environment."""
import argparse
import json
from pathlib import Path
import fitz


def inspect(folder):
    result={}
    prohibited=["ASSUMED DATA","Assumed data","Gray: obstacle","Shared budget / interface",
                "Illustrative cost units; neither trial exceeds budget"]
    for name,label in [("trajectory_sync","(a)"),("panels/trajectory","(a)"),
                       ("panels/clearance","(b)"),("panels/cost","(c)"),("panels/tracking","(d)")]:
        with fitz.open(folder/(name+".pdf")) as doc:
            page=doc[0];text=page.get_text()
            spans=[s for b in page.get_text("dict")["blocks"] if "lines" in b for l in b["lines"] for s in l["spans"]]
            outside=[s["text"] for s in spans if not page.rect.contains(fitz.Rect(s["bbox"]))]
            r=dict(pages=len(doc),width_mm=page.rect.width*25.4/72,height_mm=page.rect.height*25.4/72,
                   raster_images=len(page.get_images()),vector_objects=len(page.get_drawings()),
                   outside_text=outside,prohibited_text=[x for x in prohibited if x in text],
                   expected_panel=label in text,data_status_metadata=doc.metadata.get("subject",""))
            r["passed"]=(len(doc)==1 and not outside and not r["prohibited_text"] and r["expected_panel"]
                         and r["raster_images"]==0 and r["vector_objects"]>0
                         and "Data status:" in r["data_status_metadata"])
            if name=="trajectory_sync":
                r["passed"] &= abs(r["width_mm"]-184.2)<.01 and all(x in text for x in ["(b)","(c)","(d)"])
            result[name]=r
    report={"passed":all(v["passed"] for v in result.values()),"files":result}
    (folder/"pdf_qa.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    if not report["passed"]:raise SystemExit(1)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("folder",type=Path)
    inspect(parser.parse_args().folder)
