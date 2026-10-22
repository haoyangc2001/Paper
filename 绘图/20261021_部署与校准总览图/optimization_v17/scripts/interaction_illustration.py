"""Crop the verified temporal-ghost scene without stretching its geometry."""
from scene_assets import *

def interaction_aspect():
    layout=read(ROOT/'config/interaction_layout.json')
    meta=read(ROOT/layout['asset_dir']/'interaction_metadata.json')
    x0,y0,x1,y1=layout['crop'];h,w=meta['shape'][:2]
    return w*(x1-x0)/(h*(y1-y0))

def interaction_scene(ax):
    layout=read(ROOT/'config/interaction_layout.json')
    folder=ROOT/layout['asset_dir'];meta=read(folder/'interaction_metadata.json')
    image=folder/'interaction.png'
    assert sha(image)==meta['image_sha256'] and meta['physics_steps']==0
    assert not meta['policy_executed'] and meta['pose_interpretation']=='one_robot_at_three_times'
    clean(ax);ax.imshow(Image.open(image).convert('RGB'),extent=(0,1,0,1),aspect='auto')
    x0,y0,x1,y1=layout['crop']
    ax.set_xlim(x0,x1);ax.set_ylim(1-y1,1-y0)
