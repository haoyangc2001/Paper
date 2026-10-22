"""Render one static scene per existing pose and combine temporal ghosts."""
import argparse,csv,hashlib,json,os
from pathlib import Path
import xml.etree.ElementTree as ET

ap=argparse.ArgumentParser()
ap.add_argument('--model',required=True)
ap.add_argument('--config',required=True)
ap.add_argument('--poses',required=True)
ap.add_argument('--out',required=True)
ap.add_argument('--device',default='1')
a=ap.parse_args()
os.environ['MUJOCO_GL']='egl';os.environ['PYOPENGL_PLATFORM']='egl';os.environ['MUJOCO_EGL_DEVICE_ID']=a.device
import mujoco
import numpy as np
from PIL import Image

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def nums(values):return ' '.join(map(str,values))
def dump(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
src=Path(a.model).resolve();cfg=json.loads(Path(a.config).read_text())
assert sha(src)==cfg['model_sha256']
with open(a.poses,newline='') as stream:rows=list(csv.DictReader(stream))
assert len(rows)==75 and cfg['frames']==sorted(cfg['frames'])
assert len(cfg['frames'])==len(cfg['opacities'])==3 and cfg['opacities'][-1]==1
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
root=ET.parse(src).getroot();compiler=root.find('compiler')
compiler.set('meshdir',str((src.parent/compiler.get('meshdir','.')).resolve()))
for old in root.findall('visual'):root.remove(old)
visual=ET.SubElement(root,'visual')
ET.SubElement(visual,'global',offwidth=str(cfg['width']),offheight=str(cfg['height']))
ET.SubElement(visual,'quality',shadowsize='4096',offsamples='4')
ET.SubElement(visual,'headlight',diffuse='.65 .65 .65',ambient='.35 .35 .35',specular='.10 .10 .10')
asset=root.find('asset')
ET.SubElement(asset,'texture',name='interaction_floor_tex',type='2d',builtin='checker',width='512',height='512',rgb1=nums(cfg['floor']['rgb1']),rgb2=nums(cfg['floor']['rgb2']))
ET.SubElement(asset,'material',name='interaction_floor_mat',texture='interaction_floor_tex',texrepeat='2 2',texuniform='true',reflectance='0')
world=root.find('worldbody')
ET.SubElement(world,'geom',name='interaction_floor',type='plane',size='8 8 .05',material='interaction_floor_mat',group='0')
ET.SubElement(world,'light',pos='-2 -3 6',dir='.3 .3 -1',directional='true',diffuse='.5 .5 .5',castshadow='true')
for ob in cfg['obstacles']:
    x,y=ob['center'];h=ob['height_m']
    ET.SubElement(world,'geom',name=ob['id'],type='cylinder',pos=f'{x} {y} {h/2}',size=f'{ob["radius_m"]} {h/2}',rgba='.39 .48 .52 1',group='0')
# The complete path is taken verbatim from the original paired illustrative data.
for j in range(cfg['frames'][-1]):
    p,q=rows[j],rows[j+1]
    endpoints=[float(p['x_exec_m']),float(p['y_exec_m']),.012,float(q['x_exec_m']),float(q['y_exec_m']),.012]
    ET.SubElement(world,'geom',name=f'path_{j}',type='capsule',fromto=nums(endpoints),size=str(cfg['route_radius_m']),rgba='.13 .44 .63 1',group='0',contype='0',conaffinity='0')
gx,gy=cfg['goal'];fx=gx+cfg['goal_flag_offset_m'][0];fy=gy+cfg['goal_flag_offset_m'][1]
ET.SubElement(world,'geom',name='goal_area',type='cylinder',pos=f'{gx} {gy} .008',size='.17 .008',rgba='.22 .68 .48 1',group='0',contype='0',conaffinity='0')
ET.SubElement(world,'geom',name='goal_flag_pole',type='cylinder',pos=f'{fx} {fy} .38',size='.016 .38',rgba='.18 .55 .40 1',group='0',contype='0',conaffinity='0')
ET.SubElement(world,'geom',name='goal_flag',type='box',pos=f'{fx+.17} {fy} .66',size='.17 .014 .09',rgba='.22 .68 .48 1',group='0',contype='0',conaffinity='0')
xml=ET.tostring(root,encoding='unicode');(out/'interaction_scene.xml').write_text(xml)
model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
robot_ids=np.flatnonzero((model.geom_bodyid>0)&(model.geom_group!=3))
model.geom_group[robot_ids]=1
key=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,'home');assert key>=0
cc=cfg['camera'];cam=mujoco.MjvCamera();cam.lookat[:]=cc['lookat'];cam.distance=cc['distance'];cam.azimuth=cc['azimuth'];cam.elevation=cc['elevation']
opt=mujoco.MjvOption();opt.geomgroup[3]=0

def pose_at(frame):
    mujoco.mj_resetDataKeyframe(model,data,key)
    row=rows[frame];yaw=float(row['yaw_exec_rad'])
    data.qpos[:2]=[float(row['x_exec_m']),float(row['y_exec_m'])]
    data.qpos[3:7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
    mujoco.mj_forward(model,data)
    feet=[j for j in range(model.ngeom) if model.geom_type[j]==mujoco.mjtGeom.mjGEOM_SPHERE and abs(float(model.geom_size[j,0])-.022)<1e-6]
    assert feet
    data.qpos[2]+=-min(float(data.geom_xpos[j,2]-model.geom_size[j,0]) for j in feet)+.001
    mujoco.mj_forward(model,data)
    return [float(data.geom_xpos[j,2]-model.geom_size[j,0]) for j in feet]

records=[];files={}
with mujoco.Renderer(model,height=cfg['height'],width=cfg['width']) as renderer:
    pose_at(cfg['frames'][0]);opt.geomgroup[1]=0
    renderer.update_scene(data,camera=cam,scene_option=opt)
    background=renderer.render().copy();Image.fromarray(background).save(out/'background.png')
    c0,c1=renderer.scene.camera
    projection={'position':((c0.pos+c1.pos)/2).tolist(),'forward':np.asarray(c0.forward).tolist(),'up':np.asarray(c0.up).tolist(),'near':float(c0.frustum_near),'top':float(c0.frustum_top),'bottom':float(c0.frustum_bottom)}
    composite=background.astype(float)
    for frame,opacity in zip(cfg['frames'],cfg['opacities']):
        foot_bottoms=pose_at(frame);opt.geomgroup[1]=1
        renderer.update_scene(data,camera=cam,scene_option=opt)
        rgb=renderer.render().copy()
        renderer.enable_segmentation_rendering();renderer.update_scene(data,camera=cam,scene_option=opt)
        segmentation=renderer.render().copy();renderer.disable_segmentation_rendering()
        robot_mask=np.isin(segmentation[:,:,0],robot_ids)&(segmentation[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
        yy,xx=np.where(robot_mask);assert len(xx)>1000
        # Preserve occlusion and shade: RGB differences include robot shadows.
        changed=np.any(np.abs(rgb.astype(int)-background.astype(int))>1,axis=2)
        composite[changed]=(1-opacity)*composite[changed]+opacity*rgb[changed]
        name=f'pose_{frame:03d}'
        Image.fromarray(rgb).save(out/(name+'.png'))
        Image.fromarray((robot_mask*255).astype('uint8')).save(out/(name+'_mask.png'))
        row=rows[frame]
        records.append({'frame':frame,'time_s':float(row['t_s']),'opacity':opacity,'source_pose':[float(row['x_exec_m']),float(row['y_exec_m']),float(row['yaw_exec_rad'])],'qpos':data.qpos.tolist(),'foot_bottoms':foot_bottoms,'robot_bbox_px':[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],'visible_robot_pixels':int(len(xx))})
Image.fromarray(np.clip(np.rint(composite),0,255).astype('uint8')).save(out/'interaction.png')
for p in sorted(out.iterdir()):
    if p.is_file():files[p.name]=sha(p)
metadata={'backend':'MuJoCo EGL','mujoco_version':mujoco.__version__,'physics_steps':0,'policy_executed':False,'new_measurements':False,'source_is_assumed':True,'model_sha256':sha(src),'config_sha256':sha(a.config),'poses_sha256':sha(a.poses),'render_script_sha256':sha(__file__),'image_sha256':sha(out/'interaction.png'),'shape':[cfg['height'],cfg['width'],3],'camera':cc,'projection':projection,'records':records,'files':files,'obstacles':cfg['obstacles'],'goal':cfg['goal'],'path_point_count':cfg['frames'][-1]+1,'pose_interpretation':'one_robot_at_three_times','goal_flag_is_graphical_annotation':True,'ghost_method':'opacity composite of static renders, including shadows, with unchanged camera and obstacle occlusion','provenance':cfg['provenance']}
dump(out/'interaction_metadata.json',metadata)
assert sha(src)==cfg['model_sha256']
print(json.dumps({'pass':True,'frames':cfg['frames'],'image_sha256':metadata['image_sha256'],'robot_bboxes':[r['robot_bbox_px'] for r in records]}),flush=True)
