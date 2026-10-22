"""Offscreen Go2 command illustrations with geometric anchors for vector overlays."""
import argparse, hashlib, json, os
from pathlib import Path
import xml.etree.ElementTree as ET
ap=argparse.ArgumentParser()
for name in ['model','config','out']:ap.add_argument('--'+name,required=True)
ap.add_argument('--device',default='1');a=ap.parse_args()
os.environ['MUJOCO_GL']='egl';os.environ['PYOPENGL_PLATFORM']='egl';os.environ['MUJOCO_EGL_DEVICE_ID']=a.device
import mujoco
import numpy as np
from PIL import Image
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def nums(v):return ' '.join(map(str,v))
src=Path(a.model).resolve();cfg=json.loads(Path(a.config).read_text())
assert sha(src)==cfg['model_sha256']
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
for name,cc in cfg['views'].items():
    root=ET.parse(src).getroot();comp=root.find('compiler')
    comp.set('meshdir',str((src.parent/comp.get('meshdir','.')).resolve()))
    for v in root.findall('visual'):root.remove(v)
    visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'global',offwidth=str(cfg['width']),offheight=str(cfg['height']))
    ET.SubElement(visual,'quality',shadowsize='4096',offsamples='4')
    ET.SubElement(visual,'headlight',diffuse='.65 .65 .65',ambient='.35 .35 .35',specular='.10 .10 .10')
    asset=root.find('asset');world=root.find('worldbody')
    ET.SubElement(asset,'texture',name='floor_tex',type='2d',builtin='checker',width='512',height='512',rgb1='.48 .51 .52',rgb2='.53 .56 .57')
    ET.SubElement(asset,'material',name='floor_mat',texture='floor_tex',texrepeat='2 2',texuniform='true',reflectance='0')
    ET.SubElement(world,'geom',name='floor',type='plane',size='8 8 .05',material='floor_mat')
    ET.SubElement(world,'light',pos='-2 -3 6',dir='.3 .3 -1',directional='true',diffuse='.5 .5 .5',castshadow='true')
    xml=ET.tostring(root,encoding='unicode');(out/(name+'_scene.xml')).write_text(xml)
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
    key=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,'home');assert key>=0
    mujoco.mj_resetDataKeyframe(model,data,key);mujoco.mj_forward(model,data)
    feet=[j for j in range(model.ngeom) if model.geom_type[j]==mujoco.mjtGeom.mjGEOM_SPHERE and abs(float(model.geom_size[j,0])-.022)<1e-6]
    data.qpos[2]+=-min(float(data.geom_xpos[j,2]-model.geom_size[j,0]) for j in feet)+.001
    mujoco.mj_forward(model,data)
    cam=mujoco.MjvCamera();cam.lookat[:]=cc['lookat'];cam.distance=cc['distance'];cam.azimuth=cc['azimuth'];cam.elevation=cc['elevation']
    opt=mujoco.MjvOption();opt.geomgroup[3]=0
    with mujoco.Renderer(model,height=cfg['height'],width=cfg['width']) as ren:
        ren.update_scene(data,camera=cam,scene_option=opt)
        pixels=ren.render().copy();c0,c1=ren.scene.camera
        position=(c0.pos+c1.pos)/2;forward=np.array(c0.forward);up=np.array(c0.up)
        projection={'position':position.tolist(),'forward':forward.tolist(),'up':up.tolist(),'near':float(c0.frustum_near),'top':float(c0.frustum_top),'bottom':float(c0.frustum_bottom)}
        def project(points):
            rel=np.asarray(points)-position;z=rel@forward;right=np.cross(forward,up)
            return np.c_[.5+(rel@right)*c0.frustum_near/z/((c0.frustum_top-c0.frustum_bottom)*cfg['width']/cfg['height']),((rel@up)*c0.frustum_near/z-c0.frustum_bottom)/(c0.frustum_top-c0.frustum_bottom)]
        geometry={}
        if name=='velocity':
            z=float(data.qpos[2])+.13
            geometry['forward']=[[.03,0,z],[.44,0,z]]
            geometry['lateral']=[[.03,0,z],[.03,.29,z]]
            theta=np.linspace(-.15*np.pi,1.15*np.pi,65)
            geometry['yaw']=np.c_[-.18+.145*np.cos(theta),.145*np.sin(theta),np.full(len(theta),z+.035)].tolist()
        else:
            for joint,radius in [('FR_thigh_joint',.075),('FR_calf_joint',.063)]:
                jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,joint);assert jid>=0
                center=data.xanchor[jid].copy();axis=data.xaxis[jid].copy()
                # Put the arc just outside the visible right-side joint surface.
                center-=.025*axis
                u=np.array([1.,0,0]);u-=u@axis*axis;u/=np.linalg.norm(u);v=np.cross(axis,u)
                theta=np.linspace(-.30*np.pi,1.12*np.pi,65)
                geometry[joint]=(center+radius*(np.cos(theta)[:,None]*u+np.sin(theta)[:,None]*v)).tolist()
                geometry[joint+'_center']=[center.tolist()]
        projected={k:project(v).tolist() for k,v in geometry.items()}
    path=out/(name+'.png');Image.fromarray(pixels).save(path)
    meta={'name':name,'backend':'MuJoCo EGL','mujoco_version':mujoco.__version__,'physics_steps':0,'policy_executed':False,'home_pose':True,'qpos':data.qpos.tolist(),'camera':cc,'shape':list(pixels.shape),'model_sha256':sha(src),'config_sha256':sha(a.config),'script_sha256':sha(__file__),'image_sha256':sha(path),'projection':projection,'overlay_world':geometry,'overlay_uv':projected,'arrow_semantics':'schematic_command_categories_not_executed_actions','provenance':cfg['provenance']}
    (out/(name+'_metadata.json')).write_text(json.dumps(meta,indent=2))
    print(json.dumps({'view':name,'success':True,'image_sha256':sha(path)}),flush=True)
assert sha(src)==cfg['model_sha256']
