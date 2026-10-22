"""Check unchanged paired geometry, equal scale, inset legibility and edit scope."""
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image,ImageChops
from scene_assets import ROOT,W,H,plt,safety_scene,sha,pd

ap=argparse.ArgumentParser()
ap.add_argument('--version',required=True)
args=ap.parse_args()
out=ROOT/'output/reviews'/args.version
out.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(ROOT/'data/reference/derived_hd_mcra.csv')
fig=plt.figure(figsize=(W,H))
rect=[.535,.610,.145,.145*W/(1600/1050)/H]
ax=fig.add_axes(rect)
selection=safety_scene(ax)
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
inset=ax.child_axes[0]
width,height=np.diff(inset.get_xlim())[0],np.diff(inset.get_ylim())[0]
screen=inset.transData.transform([[0,0],[.01,0],[0,.01]])
equal=bool(np.isclose(np.linalg.norm(screen[1]-screen[0]),np.linalg.norm(screen[2]-screen[0]),atol=1e-10))
row=df.iloc[34]
distance=float(np.hypot(row.x_exec_m-row.x_nom_m,row.y_exec_m-row.y_nom_m))
text_checks=[]
for text in ax.texts:
    bounds=text.get_window_extent(renderer)
    panel=ax.get_window_extent(renderer)
    text_checks.append({'text':text.get_text(),'font_pt':text.get_fontsize(),
        'inside_panel':bool(bounds.x0>=panel.x0 and bounds.x1<=panel.x1 and bounds.y0>=panel.y0 and bounds.y1<=panel.y1)})
plt.close(fig)
baseline=ROOT/'output/final'
old=Image.open(baseline/'deployment_calibration_overview.png').convert('RGB')
new=Image.open(ROOT/'output/versions'/args.version/'deployment_calibration_overview.png').convert('RGB')
diff=ImageChops.difference(old,new)
changed_bbox=diff.getbbox()
pixels=np.asarray(diff).copy()
x0,y0,x1,y1=[.530,.600,.690,.765]
pixels[round((1-y1)*new.height):round((1-y0)*new.height),round(x0*new.width):round(x1*new.width)]=0
unchanged_elsewhere=not np.any(pixels)
source=next(v for v in json.loads((ROOT/'source_manifest.json').read_text(encoding='utf8'))['files'] if v['path']=='data/reference/derived_hd_mcra.csv')
same_data=sha(ROOT/source['path'])==source['sha256']
for name in ['robot_execution','cost_scene']:
    assert sha(ROOT/'output/versions'/args.version/'panels/scenes'/(name+'.png'))==sha(baseline/'panels/scenes'/(name+'.png'))
panel=Image.open(ROOT/'output/versions'/args.version/'panels/scenes/safety_scene.png')
for width_px in [310,130]:
    panel.resize((width_px,round(width_px*panel.height/panel.width)),Image.Resampling.LANCZOS).save(out/('safety_scene_'+str(width_px)+'px.png'))
report={'pass':same_data and equal and abs(width-.2)<1e-12 and abs(height-.2)<1e-12
    and abs(distance-selection['deviation_m'])<1e-12 and unchanged_elsewhere and all(t['inside_panel'] and t['font_pt']>=7 for t in text_checks),
    'selection':selection,'distance_recomputed_m':distance,'window_m':[width,height],'equal_world_scale':equal,
    'text_checks':text_checks,'original_data_unchanged':same_data,'pixels_outside_safety_unchanged':unchanged_elsewhere,
    'changed_bbox_pixels':changed_bbox,'other_scene_pngs_unchanged':True}
(out/'safety_inset_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(report,ensure_ascii=True))
assert report['pass']
