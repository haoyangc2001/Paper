"""Publication-sized budget curves from reviewed values, never synthetic trials.

Run with --output-dir output/new_version. Existing output directories are refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.text import Text
import fitz

from check_budget_data import ROOT, check, read_json, read_points, review_check

POPLER = Path("C:/Users/smile/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(data_dir, style_path):
    protocol = read_json(data_dir/"plot_protocol.json")
    if not protocol.get("review_complete") or protocol.get("review_count") != 96:
        raise ValueError("Complete the individual 96-point review before plotting")
    points = read_points(data_dir/"reviewed_points.jsonl")
    if any(p.get("review_decision") not in ["retain","deterministic_correction","user_confirmed_revision"] for p in points):
        raise ValueError("Pending or missing review decision")
    if set(protocol.get("thresholds",{})) != {"SR_min","CR_max","BVR_max"}:
        raise ValueError("All three acceptance thresholds are required")
    audit = check(points,protocol)
    records = read_points(ROOT/protocol["review_log"])
    # Compare review origins to the frozen baseline, then compare the adopted values.
    review_check(read_points(data_dir/"baseline_points.json"),records)
    latest = {r["point_id"]:r for r in records}
    assert all(p["value"] == latest[p["point_id"]]["adopted_value"] for p in points)
    if protocol.get('completion_version'):
        for p in points:
            if p['metric']!='Cost':
                assert p['n']==900 and abs(p['value']-p['count']/900)<1e-10
    return points,protocol,read_json(style_path),audit


def caption(protocol, audit):
    selected = ", ".join(f"{m}: {v['budget_ratio']:.2f}" for m,v in audit["minimum_feasible"].items())
    if protocol.get('completion_version'):
        return '# 内部模拟稿预算图\n\n合成数据；不是实测记录。900例分母的计数一致性修订，旧输入保留。\n\nBudget-response curves with equal-scene averaging; SR/Cost/BVR/CR; reference budgets120/140/190 a.u.;900 conditional trials per setting. The reference point reproduces the main table. Smallest passing tested ratios: '+selected+'. No continuous-optimum or significance claim.\n'
    return f"""# 预算扫描图配套图注

## English caption

Budget-response curves (assumed data), averaged equally across three scenarios.
All panels use the normalized initial budget, z0/zref. The reference at 1.00 matches
the main comparison table. Dashed horizontal limits mark SR >= 60%, BVR <= 15%,
and CR <= 20%. Outlined points in the BVR panel identify the smallest tested budgets
meeting all three criteria ({selected}). RA reuses the same trajectories at every
budget, so only its BVR changes. Execution cost averages all trials through goal,
physical collision, or timeout; budget violation does not terminate evaluation.
The SR and cost axes have nonzero lower bounds. Lines connect the given candidate
values as visual guides; no uncertainty intervals or continuous optimal budgets
are inferred. These manually specified values are for manuscript layout review,
not measured experimental evidence.

## 中文说明

三场景等权均值的预算响应（假设数据）。横轴是初始预算与场景参考预算之比；
1.00的均值与主表对应。三个水平阈值分别为SR≥60%、BVR≤15%、CR≤20%。
BVR面板的同色空心强调圈对应同时满足三项要求的最小已测试候选；
不代表连续最优预算、每个场景分别达标或有统计可靠性保证。

RA复用相同轨迹，所以SR、CR、Cost保持水平，仅BVR随阈值改变。
成本包括所有回合到达、物理碰撞或超时前的消耗；超预算不提前终止。
SR/Cost纵轴没有从零起算，刻度明确显示。折线只连接已有点，未做平滑、
未增加噪声或误差带。图面可用于正式版式，分享时应同时保留本图注的数据性质。

当前0.90/0.96/1.12的选择由输入数据重新计算；首次BVR通过余量仅为
0.3/0.1/0.6个百分点，不应表述为稳健达标。0.96至1.00的成功率变化保留
原假设，原始日志验证仍未完成。输入与逐点理由均在本项目data及reviews中。
"""


def draw(data_dir, style_path, output_dir, pilot=False):
    points, protocol, style, audit = load_inputs(data_dir,style_path)
    if output_dir.exists():
        raise FileExistsError("Use a new output directory to preserve prior renders")
    output_dir.mkdir(parents=True)
    plt.rcParams.update({"font.family":style["font_family"],"font.size":style["font_pt"],
                         "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none",
                         "svg.hashsalt":"budget-sweep-v1","axes.unicode_minus":False})
    order = ["SR","BVR"] if pilot else style["panel_order"]
    height = 48 if pilot else style["height_mm"]
    rows, columns = [1,2] if pilot else style.get("grid_shape",[2,2])
    if rows*columns != len(order):
        raise ValueError("Grid must contain exactly one axis per selected metric")
    fig, axes = plt.subplots(rows,columns,figsize=(style["width_mm"]/25.4,height/25.4),squeeze=False)
    margins = dict(style["panels"])
    if pilot:
        margins.update(bottom=.38,top=.76)
    fig.subplots_adjust(**margins)
    thresholds = {"SR":100*protocol["thresholds"]["SR_min"],
                  "BVR":100*protocol["thresholds"]["BVR_max"],
                  "CR":100*protocol["thresholds"]["CR_max"]}
    lookup = {(p["budget_ratio"],p["method"],p["metric"]):p for p in points}
    plotted = []
    methods = protocol["methods"]
    for ax, metric in zip(axes.flat,order):
        ax.set_facecolor("white")
        for side in ["top","right"]:
            ax.spines[side].set_visible(False)
        for side in ["left","bottom"]:
            ax.spines[side].set_color(style["axes_color"]); ax.spines[side].set_linewidth(.55)
        ax.grid(axis="y",color=style["grid_color"],lw=.45,zorder=0)
        ax.tick_params(axis="both",labelsize=style["tick_pt"],width=.5,length=2.5,pad=2,color=style["axes_color"])
        ax.set_xlim(style["x_limits"]); ax.set_xticks(style["x_ticks"])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x:.1f}"))
        ax.set_ylim(style["limits"][metric]); ax.set_yticks(style["ticks"][metric])
        if metric != "Cost":
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y,pos:f"{y:.0f}%"))
        ax.set_title(style["titles"][metric],fontsize=style["title_pt"],loc="left",pad=6,color="#26333D")
        ax.axvline(1,color="#D2D9DF",lw=.65,ls=(0,(2,2)),zorder=1)
        if metric in thresholds:
            limit = thresholds[metric]
            ax.axhline(limit,color="#737D84",lw=.7,ls=(0,(3,2.5)),zorder=2)
            ax.annotate(f"{limit:g}% limit",xy=(1.211,limit),xytext=(0,3),textcoords="offset points",
                        ha="right",va="bottom",fontsize=6.2,color="#5F6870",
                        bbox={"facecolor":"white","edgecolor":"none","pad":.2,"alpha":.9},zorder=7)
        for m in reversed(methods):
            records = [lookup[b,m,metric] for b in protocol["budget_ratios"]]
            xs = [p["budget_ratio"] for p in records]
            ys = [p["value"]*(1 if metric=="Cost" else 100) for p in records]
            line, = ax.plot(xs,ys,color=style["colors"][m],lw=style["line_width"],
                           marker=style["markers"][m],ms=style["marker_size"],
                           markeredgewidth=.5,ls=style["linestyles"][m],zorder=4)
            # Validate the actual Matplotlib artists against the input before saving.
            assert list(line.get_xdata()) == xs and list(line.get_ydata()) == ys
            for p,x,y in zip(records,line.get_xdata(),line.get_ydata()):
                plotted.append({"point_id":p["point_id"],"metric":metric,"method":m,
                                "x":float(x),"y_display":float(y),"value":p["value"]})
        if metric == "BVR":
            for m,row in audit["minimum_feasible"].items():
                ax.scatter([row["budget_ratio"]],[100*row["BVR"]],s=45,facecolors="none",
                           edgecolors=style["colors"][m],lw=.85,zorder=6)
    handles = [Line2D([0],[0],color=style["colors"][m],marker=style["markers"][m],
                      lw=style["line_width"],ms=3.2,ls=style["linestyles"][m],label=m) for m in methods]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(style.get("legend_x",.54),style["legend_y"]),
               ncol=3,frameon=False,fontsize=style["legend_pt"],handlelength=1.45,
               columnspacing=1.15,handletextpad=.4,borderaxespad=0)
    fig.text(style.get("xlabel_x",.56),.222 if pilot else style["xlabel_y"],r"Normalized initial budget $z_0/z_{\rm ref}$",
             ha="center",fontsize=7.2,color="#303C45")
    fig.text(style.get("summary_title_x",.54),.112 if pilot else style["summary_title_y"],"Smallest feasible tested budget",
             ha="center",fontsize=style["summary_pt"],color="#59636B")
    for x,m in zip(style.get("summary_values_x",[.24,.57,.85]),methods):
        selected = audit["minimum_feasible"].get(m)
        value = f"{selected['budget_ratio']:.2f}" if selected else "none"
        fig.text(x,.048 if pilot else style["summary_values_y"],f"{m}  {value}",ha="center",
                 fontsize=style["summary_pt"],color=style["colors"][m],fontweight="medium")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_boxes = []
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        text_boxes.append({"text":artist.get_text(),"bbox_px":list(box.bounds),"font_pt":artist.get_fontsize()})
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width+1 and box.y1 <= fig.bbox.height+1, "Text outside canvas: "+artist.get_text()
    base = output_dir/"budget_sweep"
    fig.savefig(base.with_suffix(".pdf"),metadata={"Title":"Budget response (assumed data)",
        "Subject":"Reviewed illustrative aggregate values; not empirical measurements", "CreationDate":None,"ModDate":None})
    fig.savefig(base.with_suffix(".svg"),metadata={"Title":"Budget response (assumed data)","Date":None})
    fig.savefig(base.with_suffix(".png"),dpi=style["dpi"],facecolor="white")
    plt.close(fig)
    if POPLER.exists():
        subprocess.run([str(POPLER),"-png","-singlefile","-r","220",str(base.with_suffix(".pdf")),str(output_dir/"pdf_preview")],check=True,capture_output=True)
    else:
        with fitz.open(base.with_suffix(".pdf")) as pdf:
            pdf[0].get_pixmap(matrix=fitz.Matrix(220/72,220/72)).save(output_dir/"pdf_preview.png")
    with fitz.open(base.with_suffix(".pdf")) as pdf:
        assert len(pdf)==1
        assert abs(pdf[0].rect.width*25.4/72-style["width_mm"]) < .01
        pdf_text = pdf[0].get_text()
        for m in methods:
            assert m in pdf_text
    (output_dir/"caption.md").write_text(caption(protocol,audit),encoding="utf-8")
    manifest = {"data_status":"hypothetical","empirical_validation":False,"pilot":pilot,
        "dimensions_mm":[style["width_mm"],height],"points_rendered":len(plotted),
        "panels":order,"plotted_points":plotted,"thresholds":thresholds,
        "summary":audit["minimum_feasible"],"text_boxes":text_boxes,
        "input_sha256":{str(p.relative_to(ROOT)):digest(p) for p in [data_dir/"reviewed_points.jsonl",data_dir/"plot_protocol.json",style_path]},
        "script_sha256":digest(__file__),"exports_sha256":{p.name:digest(p) for p in output_dir.iterdir() if p.is_file()},
        "readability_note":f"Physical width {style['width_mm']:g} mm; inspect at intended print size, not only enlarged preview"}
    (output_dir/"render_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(output_dir),"points_rendered":len(plotted),"dimensions_mm":manifest["dimensions_mm"],"passed":True}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",type=Path,default=ROOT/"data")
    ap.add_argument("--style",type=Path,default=ROOT/"config/style.json")
    ap.add_argument("--output-dir",required=True,type=Path)
    ap.add_argument("--pilot",action="store_true")
    ap.add_argument("--check-only",action="store_true")
    args = ap.parse_args()
    if args.check_only:
        p,proto,s,a = load_inputs(args.data_dir.resolve(),args.style.resolve())
        print(json.dumps({"passed":True,"points":len(p),"summary":a["minimum_feasible"]}))
    else:
        draw(args.data_dir.resolve(),args.style.resolve(),args.output_dir.resolve(),args.pilot)


if __name__ == "__main__":
    main()
