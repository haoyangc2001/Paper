"""Verify exported dimensions/fonts and independently reproduced pixels/statistics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import fitz
from PIL import Image, ImageChops

from compute_statistics import ROOT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(folder):
    with fitz.open(folder / "margin_calibration.pdf") as pdf:
        page = pdf[0]
        spans = [span for block in page.get_text("dict")["blocks"] if "lines" in block for line in block["lines"] for span in line["spans"] if span["text"].strip()]
        dimensions = [page.rect.width*25.4/72, page.rect.height*25.4/72]
        text = page.get_text()
        fonts = page.get_fonts(full=True)
        pdf_info = {"pages":len(pdf),"mm":dimensions,"min_font_pt":min(s["size"] for s in spans),"text":text,"font_types":[f[2] for f in fonts]}
    with Image.open(folder / "margin_calibration.png") as im:
        png_info = {"size":list(im.size),"dpi":list(im.info.get("dpi",[]))}
    valid = pdf_info["pages"] == 1 and abs(dimensions[0]-88.9)<.01 and abs(dimensions[1]-52)<.01 and pdf_info["min_font_pt"]>=6.49
    valid = valid and "88.25%" in text and "86.50%" in text and all(abs(x-600)<1 for x in png_info["dpi"])
    return {"valid":valid,"pdf":pdf_info,"png":png_info}


def compare(reference, reproduction):
    with Image.open(reference / "margin_calibration.png") as a, Image.open(reproduction / "margin_calibration.png") as b:
        # Compare RGB explicitly: RGBA getbbox may hide RGB differences under an unchanged alpha channel.
        pixels_equal = a.size == b.size and ImageChops.difference(a.convert("RGB"), b.convert("RGB")).getbbox() is None
    ref = json.loads((reference / "render_manifest.json").read_text(encoding="utf-8"))
    new = json.loads((reproduction / "render_manifest.json").read_text(encoding="utf-8"))
    expected_stats = json.loads((ROOT / "data/statistics.json").read_text(encoding="utf-8"))
    new_stats = json.loads((reproduction / "statistics.json").read_text(encoding="utf-8"))
    same = {name: digest(reference/name)==digest(reproduction/name) for name in ["margin_calibration.png","margin_calibration.pdf","margin_calibration.svg","caption.txt","single_column_preview.png"]}
    report = {"pixels_equal":pixels_equal,"file_hash_equality":same,"statistics_equal":expected_stats==new_stats,"render_numeric_equal":ref["panels"]==new["panels"],"data_hash_equal":ref["data_sha256"]==new["data_sha256"]==digest(ROOT/"data/paired_rollouts.csv"),"config_hash_equal":ref["config_sha256"]==new["config_sha256"],"script_hash_equal":ref["script_sha256"]==new["script_sha256"]}
    report["valid"] = all(report[k] for k in ["pixels_equal","statistics_equal","render_numeric_equal","data_hash_equal","config_hash_equal","script_hash_equal"]) and all(same.values())
    return report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("inspect","compare"))
    p.add_argument("--folder",type=Path,default=ROOT/"output/final")
    p.add_argument("--reproduction",type=Path,default=ROOT/"output/reproduction_v1")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result=inspect(args.folder) if args.phase=="inspect" else compare(args.folder,args.reproduction)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))
    if not result["valid"]:
        raise SystemExit(1)
