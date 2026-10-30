import argparse
from pathlib import Path
from matplotlib.lines import Line2D
from plot_style import ROOT,plt,COLORS,load_data,export,method_handles
from plot_trajectory import draw_trajectory,draw_zoom
from plot_metrics import draw_metric

def compose(data,out):
    fig=plt.figure(figsize=(184.2/25.4,100/25.4))
    ax=fig.add_axes([.065,.48,.40,.38]);draw_trajectory(ax,data)
    ax.set_title("(a) Same-scene trajectories",loc="left",pad=7)
    zoom=fig.add_axes([.085,.14,.125,.25]);draw_zoom(zoom,data)
    s,frames,events,summary=data
    fig.text(.24,.375,"Same budget and frozen interface",fontsize=7.5,fontweight="bold")
    fig.text(.24,.325,"Minimum clearance",fontsize=7,color="#5a626a")
    fig.text(.24,.280,"0.137 m",color=COLORS["hd_mcra"],fontsize=9)
    fig.text(.355,.280,"0.014 m",color=COLORS["rcppo"],fontsize=9)
    fig.text(.24,.232,"Arrival time",fontsize=7,color="#5a626a")
    fig.text(.24,.189,"15.0 s",color=COLORS["hd_mcra"],fontsize=9)
    fig.text(.355,.189,"13.4 s",color=COLORS["rcppo"],fontsize=9)
    fig.text(.24,.130,r"$\alpha=0.18,\ \delta=0.12$ m",fontsize=7.5)
    for kind,y in zip(["clearance","cost","tracking"],[.735,.455,.175]):
        metric=fig.add_axes([.565,y,.405,.185])
        draw_metric(metric,data,kind)
        if kind!="tracking":metric.set_xlabel("");metric.tick_params(labelbottom=False)
    fig.legend(handles=method_handles()+[
        Line2D([],[],color="#656c73",ls=(0,(3,2)),lw=.9,label="Nominal"),
        Line2D([],[],color="#656c73",ls=(0,(5,2,1,2)),lw=.9,label="Calibrated nominal cost")],
        loc="upper center",bbox_to_anchor=(.52,1.005),ncol=4,frameon=False,handlelength=2.1,columnspacing=1.4,fontsize=7.2)
    fig.text(.065,.070,"Gray: obstacle  |  pale: robot footprint  |  dotted: + delta",fontsize=6.7,color="#555e66")
    fig.text(.065,.027,"ASSUMED DATA  •  A/B use each method's passage time  •  Illustrative cost units; neither trial exceeds budget",fontsize=6.5,color="#555e66")
    export(fig,out)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data-dir",type=Path);p.add_argument("--output",type=Path,default=ROOT/"output/trajectory_sync_assumed_v1")
    a=p.parse_args();compose(load_data(a.data_dir),a.output)

