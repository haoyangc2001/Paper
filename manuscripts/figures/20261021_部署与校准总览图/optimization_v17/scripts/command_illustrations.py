"""Simulation scenes with projected body-motion and joint-rotation annotations."""
from scene_assets import *

def command_scene(ax,name):
    path=ROOT/'assets/command_renders'/(name+'.png')
    meta=read(path.with_name(name+'_metadata.json'))
    assert sha(path)==meta['image_sha256'] and meta['physics_steps']==0
    clean(ax);ax.imshow(Image.open(path).convert('RGB'),extent=(0,1,0,1),aspect='auto')
    colors={'forward':BLUE,'lateral':GREEN,'yaw':ORANGE,'FR_thigh_joint':ORANGE,'FR_calf_joint':ORANGE}
    for key,coords in meta['overlay_uv'].items():
        if key.endswith('_center'):continue
        p=np.asarray(coords);color=colors[key]
        effects=[pe.Stroke(linewidth=2.8,foreground='white'),pe.Normal()]
        if len(p)>2:
            ax.plot(p[:,0],p[:,1],color=color,lw=1.5,zorder=5,path_effects=effects)
            start=p[-6]
        else:start=p[0]
        ax.add_patch(FancyArrowPatch(start,p[-1],arrowstyle='-|>',mutation_scale=8.5,lw=1.5,color=color,
                                   zorder=6,shrinkA=0,shrinkB=0,path_effects=effects))
    # Equal fractional crop widths/heights retain the original camera aspect.
    x0,y0,x1,y1=(.12,.10,.89,.87) if name=='velocity' else (.20,.12,.90,.82)
    ax.set_xlim(x0,x1);ax.set_ylim(y0,y1)
    ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec='#A9B8C2',lw=.65,zorder=8))

def add_command_illustrations(fig, links, label_ax, label):
    layout=read(ROOT/'config/execution_layout.json')
    width=layout['command_width'];height=width*W/H/(1280/900)
    placements=[]
    for name,points in links.items():
        # The illustration is a label of this exact connection, centered over it.
        (x0,y),(x1,y1)=points
        assert y==y1 and x1-x0>width
        center=(x0+x1)/2
        rect=[center-width/2,y+layout['command_bottom_gap'],width,height]
        axis=fig.add_axes(rect);command_scene(axis,name)
        text=layout['command_labels'][name]
        label_y=rect[1]+rect[3]+layout['command_label_gap']
        label(label_ax,center,label_y,text,layout['command_label_font_pt'],
              BLUE if name=='velocity' else INK,'center',False,linespacing=1.2)
        placements.append({'name':name,'rect':rect,'link':points,'label':text,
                           'label_center':[center,label_y],
                           'horizontal_center_error':rect[0]+rect[2]/2-center})
    return placements

def export_command_components(out):
    for name in ['velocity','joint']:
        fig=plt.figure(figsize=(1.8,1.8*900/1280))
        axis=fig.add_axes([0,0,1,1]);command_scene(axis,name)
        save(fig,out/'panels/commands'/name);plt.close(fig)
