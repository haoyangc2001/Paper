"""Create visual comparison and inspect PDF structure using existing tools."""
import argparse,json
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
def contact():
    ref=Image.open(ROOT/"reference/sea_nav_figure4.png").convert("RGB")
    out=Image.open(ROOT/"output/trajectory_sync_assumed_v3.png").convert("RGB")
    width=1400
    ref=ref.resize((width,round(ref.height*width/ref.width)))
    out=out.resize((width,round(out.height*width/out.width)))
    canvas=Image.new("RGB",(2*width+60,max(ref.height,out.height)+80),"white")
    draw=ImageDraw.Draw(canvas)
    draw.text((20,20),"SEA-Nav Fig. 4 (reference)",fill="black")
    draw.text((width+40,20),"Assumed-data preview (same display width)",fill="black")
    canvas.paste(ref,(20,60));canvas.paste(out,(width+40,60))
    canvas.save(ROOT/"output/reference_comparison.png")
if __name__=="__main__":contact()

