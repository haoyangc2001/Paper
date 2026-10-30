"""Independent output-directory rebuild; keep inputs and prior outputs intact."""
from pathlib import Path
import argparse,subprocess,hashlib,json,sys
import fitz
ROOT=Path(__file__).resolve().parents[1]
PY39="D:/codesoftware/Miniconda/envs/py39/python.exe"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(args):subprocess.run(args,check=True,cwd=ROOT)
ap=argparse.ArgumentParser();ap.add_argument("--version",default="final");ap.add_argument("--out",required=True);a=ap.parse_args()
out=Path(a.out).resolve()
if out.exists():raise SystemExit("Choose a new reproduction directory.")
out.mkdir(parents=True)
run([PY39,str(ROOT/"scripts/verify_inputs.py"),"--out",str(out/"input_checks.json")])
run([PY39,str(ROOT/"scripts/review_interaction.py"),"--out",str(out/"interaction_checks")])
run([PY39,str(ROOT/"scripts/draw_overview.py"),"--out",str(out/"graphics")])
base=ROOT/"output/final" if a.version=="final" else ROOT/"output/versions"/a.version
results=[]
for p in sorted(base.rglob("*")):
    if p.suffix not in [".png",".pdf",".svg"] or p.name.startswith('ccdc_figure_preview'):continue
    rel=p.relative_to(base);same=sha(p)==sha(out/"graphics"/rel)
    results.append({"file":str(rel),"sha256_equal":same})
assert len(results)==39 and all(v["sha256_equal"] for v in results)
run([sys.executable,str(ROOT/"scripts/build_preview.py"),"--out",str(out/"preview"),
     "--figure",str(out/"graphics/deployment_calibration_overview.pdf")])
preview=ROOT/"output/final" if a.version=="final" else ROOT/"preview/build"/a.version
page_checks=[]
with fitz.open(preview/"ccdc_figure_preview.pdf") as original,fitz.open(out/"preview/ccdc_figure_preview.pdf") as rebuilt:
    assert len(original)==len(rebuilt)==2
    for i in range(len(original)):
        texts=original[i].get_text()==rebuilt[i].get_text()
        pixels=original[i].get_pixmap(matrix=fitz.Matrix(2,2)).samples==rebuilt[i].get_pixmap(matrix=fitz.Matrix(2,2)).samples
        page_checks.append({"page":i+1,"text_equal":texts,"pixels_equal":pixels})
assert all(v["text_equal"] and v["pixels_equal"] for v in page_checks)
png_equal=sha(preview/"ccdc_figure_preview_page2.png")==sha(out/"preview/ccdc_figure_preview_page2.png")
assert png_equal
report={"pass":True,"graphics":results,"preview_pages":page_checks,"poppler_page2_png_equal":png_equal,
"environment":{"plot_python":PY39,"verification_python":sys.executable},
"note":"Same frozen inputs and scripts, new process/output directory; not a new physical experiment."}
(out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps({"pass":True,"graphics_exact":39,"preview_pages":2,"poppler_equal":png_equal}))
