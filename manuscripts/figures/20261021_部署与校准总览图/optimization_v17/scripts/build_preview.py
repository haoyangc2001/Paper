"""Compile an isolated preview using the unchanged CCDC class."""
from pathlib import Path
import argparse,json,subprocess,shutil
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser()
ap.add_argument("--version",default="final")
ap.add_argument("--out")
ap.add_argument("--figure")
a=ap.parse_args()
out=Path(a.out).resolve() if a.out else ROOT/"preview/build"/a.version
if out.exists():raise SystemExit("Choose a new --out directory; existing previews are preserved.")
out.mkdir(parents=True)
figure=Path(a.figure).resolve() if a.figure else (ROOT/"output/final" if a.version=="final" else ROOT/"output/versions"/a.version)/"deployment_calibration_overview.pdf"
for src,name in [(ROOT/"preview/IEEEtran.cls","IEEEtran.cls"),
                 (ROOT/"preview/ccdc_figure_preview.tex","ccdc_figure_preview.tex"),
                 (ROOT/"caption_en.tex","caption_en.tex"),(figure,"figure.pdf")]:
    shutil.copy2(src,out/name)
exe="D:/app/latex/texlive/2026/bin/windows/pdflatex.exe"
for _ in range(2):
    r=subprocess.run([exe,"-interaction=nonstopmode","-halt-on-error","ccdc_figure_preview.tex"],
                     cwd=out,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if r.returncode:print(r.stdout.decode(errors="replace"));raise SystemExit(r.returncode)
log=(out/"ccdc_figure_preview.log").read_text(errors="replace")
assert "OVERVIEW-TEXTWIDTH=516.0pt" in log
assert "Overfull" not in log and "Fatal error" not in log
subprocess.run(["pdftoppm","-f","2","-singlefile","-r","120","-png",
                str(out/"ccdc_figure_preview.pdf"),str(out/"ccdc_figure_preview_page2")],check=True)
report={"pass":True,"textwidth_tex_pt":516,"no_overfull":True,"out":str(out)}
(out/"preview_check.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,ensure_ascii=True))
