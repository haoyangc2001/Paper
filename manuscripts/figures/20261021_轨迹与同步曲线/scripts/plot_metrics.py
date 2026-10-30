import argparse
from pathlib import Path
import numpy as np
from matplotlib.ticker import MaxNLocator
from plot_style import ROOT,plt,COLORS,LABELS,load_data,axis_style,export,method_handles,line_handles

KINDS=["clearance","cost","tracking"]
def draw_metric(ax,data,kind,events_enabled=True,publication=False):
    s,frames,events,summary=data
    for m,df in frames.items():
        c=COLORS[m]
        if kind=="clearance":
            ax.plot(df.t_s,df.clearance_exec_m,color=c)
        elif kind=="cost":
            ax.plot(df.t_s,df.J_nom,color=c,ls=(0,(2,2)),lw=.8,alpha=.55)
            ax.plot(df.t_s,df.J_exec,color=c)
            if m=="hd_mcra":
                ax.plot(df.t_s,df.J_charged,color=c,ls=(0,(5,2,1,2)),lw=.9,alpha=.85)
                ax.fill_between(df.t_s.to_numpy(),df.J_nom.to_numpy(),df.J_charged.to_numpy(),color=c,alpha=.06)
        else:
            ax.plot(df.t_s.iloc[:-1],df.tracking_error_mps.iloc[:-1],color=c,lw=1.15)
    if kind=="clearance":
        ax.axhline(0,color="#72777c",ls=(0,(3,2)),lw=.7)
        ax.set_ylim(-.04,1.45);ax.set_yticks([0,.5,1])
        if publication:
            low=min(0,min(df.clearance_exec_m.min() for df in frames.values()))
            high=max(df.clearance_exec_m.max() for df in frames.values())
            span=max(high-low,1e-6)
            ax.set_ylim(low-.03*span,high+.10*span)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4,steps=[1,2,2.5,5,10]))
        ax.set_ylabel("Clearance (m)")
        ax.set_title("(b) Executed body clearance",loc="left",pad=5)
        if events_enabled:
            for _,e in events.iterrows():
                ax.axvline(e.t_s,color=COLORS[e.method],lw=.55,alpha=.4,ls=":")
                if publication:
                    ax.text(e.t_s,.88,e.event_id,transform=ax.get_xaxis_transform(),
                            ha="center",fontsize=6.8,color=COLORS[e.method])
                else:
                    ax.text(e.t_s,1.27,e.event_id,ha="center",fontsize=6.8,color=COLORS[e.method])
    elif kind=="cost":
        ax.axhline(s["initial_budget"],color="#5e656b",ls=(0,(4,2)),lw=.8)
        ax.set_ylim(0,98);ax.set_yticks([0,40,80])
        ax.set_ylabel("Cost (a.u.)")
        ax.set_title("(c) Cumulative cost and budget",loc="left",pad=5)
        if publication:
            high=max(s["initial_budget"],max(df[["J_nom","J_exec","J_charged"]].max().max() for df in frames.values()))
            ax.set_ylim(0,max(high*1.15,1e-6))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3,steps=[1,2,2.5,4,5,10]))
            unit=s["cost"].get("axis_unit",s["cost"]["unit"])
            if unit=="illustrative cost units":unit="a.u."
            ax.set_ylabel(f"Cost ({unit})")
            ax.annotate(f"Common budget {s['initial_budget']:g}",
                        xy=(.02,s["initial_budget"]),xycoords=("axes fraction","data"),
                        xytext=(0,2),textcoords="offset points",fontsize=6.7,color="#4f565c")
            end=frames["hd_mcra"].iloc[-1]
            if abs(s["initial_budget"]-end.J_charged)<.20*high:
                end=frames["hd_mcra"].iloc[int(.82*(len(frames["hd_mcra"])-1))]
            ax.annotate(r"$(1+\alpha)J_{\rm nom}$",xy=(end.t_s,end.J_charged),
                        xytext=(-2,3),textcoords="offset points",color=COLORS["hd_mcra"],ha="right",fontsize=7)
        else:
            ax.text(.3,86.6,"Common budget 85",fontsize=6.7,color="#4f565c")
            ax.text(14.8,75.5,r"$(1+\alpha)J_{\rm nom}$",color=COLORS["hd_mcra"],ha="right",fontsize=7)
    else:
        ax.set_ylim(0,.065);ax.set_yticks([0,.03,.06])
        if publication:
            high=max(df.tracking_error_mps.max() for df in frames.values())
            ax.set_ylim(0,max(high*1.20,1e-6))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3,steps=[1,2,3,5,10]))
        ax.set_ylabel("Error (m/s)")
        ax.set_title("(d) Velocity tracking error",loc="left",pad=5)
    if events_enabled and kind!="clearance":
        for _,e in events.iterrows():
            ax.axvline(e.t_s,color=COLORS[e.method],lw=.45,alpha=.22,ls=":")
    ax.set_xlim(0,15.3);ax.set_xticks([0,3,6,9,12,15])
    if publication:
        end=max(df.t_s.max() for df in frames.values())
        ticks=MaxNLocator(nbins=5,steps=[1,1.5,2,2.5,3,4,5,6,8,10]).tick_values(0,max(end,1e-6))
        ax.set_xlim(0,ticks[-1]*1.02);ax.set_xticks(ticks)
    ax.set_xlabel("Elapsed time (s)")
    axis_style(ax)

def standalone(data,kind,out,publication=False):
    if publication:
        fig=plt.figure(figsize=(4.5,2.3))
        ax=fig.add_axes([.16,.31,.80,.53]);draw_metric(ax,data,kind,publication=True)
        handles=method_handles()+(line_handles(calibrated=True) if kind=="cost" else [])
        fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.56,.005),
                   ncol=2,frameon=False,fontsize=6.6)
        export(fig,out,publication=True,data_status=data[0]["data_status"])
        return
    fig=plt.figure(figsize=(4.5,2.4))
    ax=fig.add_axes([.16,.31,.80,.53])
    draw_metric(ax,data,kind)
    fig.legend(handles=method_handles(),loc="lower center",bbox_to_anchor=(.56,.07),ncol=2,frameon=False)
    fig.text(.5,.025,"Assumed data | "+("dashed: nominal; dash-dot: calibrated nominal" if kind=="cost" else "A/B: each method's own passage times"),ha="center",fontsize=6.6)
    export(fig,out)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data-dir",type=Path);p.add_argument("--output-dir",type=Path,default=ROOT/"output/panels")
    a=p.parse_args()
    for k in KINDS:standalone(load_data(a.data_dir),k,a.output_dir/(k+"_v1"))

