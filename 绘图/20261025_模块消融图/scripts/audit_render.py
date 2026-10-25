"""Inspect PDF dimensions/text and data-to-render mapping; no data generation."""
from pathlib import Path
import argparse
import json
import math
import fitz
from check_ablation_data import read_json, read_lines, validate

PROJECT=Path(__file__).resolve().parents[1]

def audit(out):
    manifest=read_json(out/'render_manifest.json')
    points=read_lines(PROJECT/'data/reviewed_points.jsonl')
    protocol=read_json(PROJECT/'data/plot_protocol.json')
    reviews=read_lines(PROJECT/'reviews/point_review.jsonl')
    errors=validate(points,protocol,reviews,manifest)
    pdf=out/next(p for p in manifest['files'] if p.endswith('.pdf'))
    with fitz.open(pdf) as document:
        assert len(document)==1
        page=document[0]
        width_mm=page.rect.width/72*25.4
        height_mm=page.rect.height/72*25.4
        spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
        min_size=min(s['size'] for s in spans if s['text'].strip())
        if not math.isclose(width_mm,manifest['width_mm'],abs_tol=.02):errors.append('PDF width mismatch')
        if not math.isclose(height_mm,manifest['height_mm'],abs_tol=.02):errors.append('PDF height mismatch')
        if min_size<6.49:errors.append('PDF font below 6.5pt')
    boxes=manifest['text_boxes']
    outside=[t['text'] for t in boxes if min(t['bbox_figure'][:2])<0 or max(t['bbox_figure'][2:])>1]
    overlap=[]
    for i,a in enumerate(boxes):
        a0,a1,a2,a3=a['bbox_figure']
        for b in boxes[i+1:]:
            b0,b1,b2,b3=b['bbox_figure']
            if min(a2,b2)-max(a0,b0)>.001 and min(a3,b3)-max(a1,b1)>.001:
                overlap.append([a['text'],b['text']])
    if outside:errors.append('text clipped by canvas')
    if overlap:errors.append('text boxes overlap')
    result={'passed':not errors,'errors':errors,'width_mm':width_mm,'height_mm':height_mm,'min_pdf_font_pt':min_size,
            'text_outside':outside,'text_overlap':overlap,'comparisons':len(manifest['comparisons']),
            'marker_instances':manifest['marker_instances'],'data_mapping_passed':not validate(points,protocol,reviews,manifest)}
    (out/'render_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
    if errors:raise SystemExit(1)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    audit(parser.parse_args().output)
