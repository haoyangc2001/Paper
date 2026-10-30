"""Compare independently rendered figure artifacts and their provenance."""
from pathlib import Path
import hashlib
import json
import fitz
from PIL import Image

PROJECT=Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    folders=[PROJECT/'output/final',PROJECT/'output/reproduction_v1']
    manifest=[json.loads((p/'render_manifest.json').read_text(encoding='utf-8')) for p in folders]
    checks={
        'same_input_hashes':manifest[0]['inputs']==manifest[1]['inputs'],
        'same_script_hash':manifest[0]['script_sha256']==manifest[1]['script_sha256'],
        'same_mapping':manifest[0]['comparisons']==manifest[1]['comparisons'],
        'same_text_positions':manifest[0]['text_boxes']==manifest[1]['text_boxes'],
        'same_caption':(folders[0]/'caption.md').read_bytes()==(folders[1]/'caption.md').read_bytes(),
    }
    with Image.open(folders[0]/'ablation_horizontal.png') as first, Image.open(folders[1]/'ablation_horizontal.png') as second:
        checks['same_png_pixels']=first.size==second.size and first.tobytes()==second.tobytes()
    pixels=[]
    for folder in folders:
        with fitz.open(folder/'ablation_horizontal.pdf') as pdf:
            pixels.append(pdf[0].get_pixmap(matrix=fitz.Matrix(2,2)).samples)
    checks['same_pdf_rendered_pixels']=pixels[0]==pixels[1]
    report={'passed':all(checks.values()),'checks':checks,'input_hashes':manifest[0]['inputs'],
            'final_sha256':{f.name:digest(f) for f in folders[0].iterdir() if f.is_file()},
            'note':'PDF/SVG metadata timestamps can differ; data mapping and rendered pixels are compared.'}
    (PROJECT/'reviews/reproduction_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':report['passed'],'checks':checks},ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':
    main()
