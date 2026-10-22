"""Check actual text objects and paper-size renders, including mathematical labels."""
from pathlib import Path
import argparse
import json
import fitz
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--version", required=True)
args = parser.parse_args()
source = ROOT/"output/versions"/args.version
out = ROOT/"output/reviews"/args.version
if out.exists():
    raise SystemExit("Choose a new version; review evidence must be preserved.")
out.mkdir(parents=True)
metadata = json.loads((source/"build_metadata.json").read_text(encoding="utf-8"))
width, height = metadata["width_in"]*72, metadata["height_in"]*72
outside, overlaps = [], []
labels = metadata["labels"]
for i, first in enumerate(labels):
    a = first["bbox_figure"]
    if min(a) < -1e-5 or max(a) > 1+1e-5:
        outside.append(first["text"])
    for second in labels[i+1:]:
        b = second["bbox_figure"]
        dx = min(a[2], b[2])-max(a[0], b[0])
        dy = min(a[3], b[3])-max(a[1], b[1])
        if dx*width > .6 and dy*height > .6:
            overlaps.append([first["text"], second["text"], dx*dy*width*height])
with fitz.open(source/"deployment_calibration_overview.pdf") as pdf:
    page = pdf[0]
    spans = [s for block in page.get_text("dict")["blocks"] if "lines" in block
             for line in block["lines"] for s in line["spans"]]
    min_glyph = min(span["size"] for span in spans)
    page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72), alpha=False).save(out/"pdf_render.png")
im = Image.open(source/"deployment_calibration_overview.png").convert("RGB")
paper = im.resize((round(metadata["width_in"]*110), round(metadata["height_in"]*110)), Image.Resampling.LANCZOS)
paper.save(out/"paper_size.png")
ImageOps.grayscale(paper).save(out/"paper_size_gray.png")
reference = ROOT.parents[2]/"CCDC/references/figure_review_20261020/figures/confined_overview_fig2.png"
tiles = []
for title, path in [("Reference 2", reference), (args.version, source/"deployment_calibration_overview.png")]:
    if not path.is_file():
        continue
    image = Image.open(path).convert("RGB")
    image = image.resize((1000, round(1000*image.height/image.width)))
    tile = Image.new("RGB", (1020, image.height+36), "white")
    tile.paste(image, (10, 30)); ImageDraw.Draw(tile).text((12, 10), title, fill="black")
    tiles.append(tile)
comparison = Image.new("RGB", (1020*len(tiles), max(tile.height for tile in tiles)), "#EEE")
for i, tile in enumerate(tiles):
    comparison.paste(tile, (1020*i, 0))
comparison.save(out/"reference_comparison.png")
layout=metadata['execution_layout']
panels=[{'name':k,'rect':layout[k]} for k in ['actor','tracker','environment']]+layout['commands']
panel_overlaps=[]
for i,first in enumerate(panels):
    x,y,w,h=first['rect']
    for second in panels[i+1:]:
        u,v,s,t=second['rect']
        if min(x+w,u+s)-max(x,u)>1e-6 and min(y+h,v+t)-max(y,v)>1e-6:
            panel_overlaps.append([first['name'],second['name']])
alignments=[]
for item in layout['commands']:
    x,y,w,h=item['rect'];(x0,y0),(x1,y1)=item['link']
    alignments.append({'name':item['name'],'center_error':abs(x+w/2-(x0+x1)/2),
                       'within_link_x':x>=x0 and x+w<=x1,'vertical_gap':y-y0})
layout_pass=not panel_overlaps and all(q['center_error']<1e-12 and q['within_link_x'] and .01<=q['vertical_gap']<=.02 for q in alignments)
navigation=json.loads((ROOT/'config/execution_layout.json').read_text())['navigation_card']
nav_right=navigation[0]+navigation[2]
command_clear_of_navigation=all(q['rect'][0]>nav_right for q in layout['commands'])
layout_pass=layout_pass and command_clear_of_navigation
command_label_checks=[]
for item in layout['commands']:
    candidates=[q for q in labels if q['text']==item['label']]
    assert len(candidates)==1
    bounds=candidates[0]['bbox_figure']
    conflicts=[]
    for panel in panels:
        x,y,w,h=panel['rect']
        if min(bounds[2],x+w)-max(bounds[0],x)>.6/width and min(bounds[3],y+h)-max(bounds[1],y)>.6/height:
            conflicts.append(panel['name'])
    center=(item['link'][0][0]+item['link'][1][0])/2
    command_label_checks.append({'name':item['name'],'text':item['label'],
        'center_error':abs((bounds[0]+bounds[2])/2-center),
        'clear_of_panels':not conflicts,'panel_conflicts':conflicts,
        'clear_of_navigation':bounds[0]>nav_right,
        'above_own_image':bounds[1]>item['rect'][1]+item['rect'][3]})
layout_pass=layout_pass and all(q['center_error']<1e-9 and q['clear_of_panels'] and q['clear_of_navigation'] and q['above_own_image'] for q in command_label_checks)
interaction_title=next(q for q in labels if 'interaction' in q['text'])
title_box=interaction_title['bbox_figure'];environment=layout['environment']
interaction_title_check={'inside_stage_card':title_box[0]>=.015 and title_box[2]<=.985,
                         'below_scene':title_box[3]<environment[1]}
layout_pass=layout_pass and all(interaction_title_check.values())
report = {"version": args.version, "text_objects": len(labels), "overlaps": overlaps, "outside": outside,
          "panel_overlaps":panel_overlaps,"command_alignments":alignments,"command_clear_of_navigation":command_clear_of_navigation,"layout_pass":layout_pass,
          "command_label_checks":command_label_checks,"interaction_title_check":interaction_title_check,
          "minimum_base_font_pt": min(label["font_pt"] for label in labels), "minimum_pdf_glyph_font_pt": min_glyph,
          "pdf_size_pt": [width, height], "cross_stage_arrows": metadata["cross_stage_connections"],
          "schematic_networks": metadata["network_panels"],
          "note": "Text-overlap check groups each mathematical expression as one text object; no glyph-level false positives."}
report["pass"] = not outside and not overlaps and layout_pass and report["minimum_base_font_pt"] >= 7 and min_glyph >= 6.99
(out/"audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=True))
assert report["pass"]
