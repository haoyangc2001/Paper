"""Verify the fixed-pose illustration and make readable review crops."""
from pathlib import Path
import argparse,hashlib,json,csv
import numpy as np
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
out=Path(a.out);assert not out.exists();out.mkdir(parents=True)
layout=read(ROOT/'config/interaction_layout.json');folder=ROOT/layout['asset_dir']
meta=read(folder/'interaction_metadata.json');cfg=read(ROOT/layout['scene_config'])
assert meta['config_sha256']==sha(ROOT/layout['scene_config'])
assert meta['poses_sha256']==sha(ROOT/'data/reference/derived_hd_mcra.csv')
assert meta['render_script_sha256']==sha(ROOT/'scripts/render_interaction_remote.py')
assert meta['model_sha256']==cfg['model_sha256']
assert meta['physics_steps']==0 and not meta['policy_executed'] and meta['source_is_assumed']
assert meta['pose_interpretation']=='one_robot_at_three_times' and meta['path_point_count']==75
assert meta['obstacles']==read(ROOT/'config/input_scenes.json')['obstacles']
assert meta['goal']==read(ROOT/'data/reference/scenario.json')['goal_center']
for name,digest in meta['files'].items():assert sha(folder/name)==digest
assert sha(folder/'interaction.png')==meta['image_sha256']
with open(ROOT/'data/reference/derived_hd_mcra.csv',newline='') as stream:rows=list(csv.DictReader(stream))
background=np.array(Image.open(folder/'background.png')).astype(float)
rebuilt=background.copy();pose_checks=[]
x0,y0,x1,y1=layout['crop'];height,width=meta['shape'][:2]
for record in meta['records']:
    row=rows[record['frame']]
    assert np.allclose(record['source_pose'],[float(row[k]) for k in ['x_exec_m','y_exec_m','yaw_exec_rad']],atol=1e-12)
    assert np.allclose(record['qpos'][:2],record['source_pose'][:2],atol=1e-12)
    assert np.allclose(record['foot_bottoms'],.001,atol=1e-8)
    rect=record['robot_bbox_px']
    assert rect[0]>x0*width and rect[2]<x1*width and rect[1]>y0*height and rect[3]<y1*height
    image=np.array(Image.open(folder/f'pose_{record["frame"]:03d}.png')).astype(float)
    changed=np.any(np.abs(image-background)>1,axis=2)
    opacity=record['opacity'];rebuilt[changed]=(1-opacity)*rebuilt[changed]+opacity*image[changed]
    pose_checks.append({'frame':record['frame'],'opacity':opacity,'bbox_inside_crop':True,'pose_matches_source':True})
expected=np.array(Image.open(folder/'interaction.png'))
assert np.array_equal(np.clip(np.rint(rebuilt),0,255).astype('uint8'),expected)
green=(background[:,:,1]>background[:,:,0]+25)&(background[:,:,1]>background[:,:,2]+10)
gy,gx=np.where(green);assert len(gx)>100
assert gx.min()>x0*width and gx.max()<x1*width and gy.min()>y0*height and gy.max()<y1*height
image=Image.open(folder/'interaction.png').convert('RGB')
crop=image.crop((round(x0*width),round(y0*height),round(x1*width),round(y1*height)))
crop.save(out/'interaction_crop.png')
for target in [100,300]:
    thumb=crop.resize((target,round(target*crop.height/crop.width)),Image.Resampling.LANCZOS)
    thumb.save(out/f'width_{target}.png')
    ImageOps.grayscale(thumb).save(out/f'width_{target}_gray.png')
report={'pass':True,'source_frames':pose_checks,'file_hashes_checked':len(meta['files']),'composite_pixel_identical':True,'goal_flag_inside_crop':True,'obstacle_count':len(meta['obstacles']),'one_original_executed_path':True,'physics_steps':0,'policy_executed':False,'new_data_generated':False,'crop':layout['crop'],'crop_aspect':crop.width/crop.height}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report))
