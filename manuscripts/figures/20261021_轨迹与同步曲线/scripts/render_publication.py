# -*- coding: utf-8 -*-
"""Render the paper layout from supplied records; never generate raw data."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from compute_metrics import ROOT,build,read_json,read_raw
from validate_data import validate,check_derived
from plot_style import load_data
from plot_trajectory import standalone as trajectory
from plot_metrics import standalone as metric,KINDS
from compose_figure import compose


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_caption(data,path):
    s,frames,events,summary=data
    budget=s["initial_budget"]
    alpha=s["methods"]["hd_mcra"]["alpha"]
    delta=s["methods"]["hd_mcra"]["delta_m"]
    event_defs="; ".join(f"{e['id']}: x = {e['x_m']:g} m" for e in s["events"])
    status=s["data_status"]
    disclosure=("当前数据为逐点编写的假设数据，用于审阅版式与论证关系，不是实测、策略输出或物理仿真结果。"
                if status=="assumed" else f"数据状态（来自输入配置）：{status}；来源须以配套日志记录为准。")
    en_status=("Assumed-data illustration; the records are manually authored, not measurements, policy outputs, or physical-simulation results. "
               if status=="assumed" else f"Input data status: {status}. See the accompanying provenance records. ")
    time_text="；".join(f"{s['methods'][m]['label']}："+"、".join(
        f"{r.event_id}={r.t_s:g} s" for r in events[events.method==m].itertuples()) for m in frames)
    both_within=all(not v["budget_violation"] for v in summary.values())
    limits=("本案例两方法均未超过预算，不能用其证明 α 降低预算违约率。" if both_within
            else "单次轨迹不能推断总体预算违约率，须结合多次实验统计。")
    zh_status="（当前为假设数据）" if status=="assumed" else ""
    unit_note=("图中 a.u. 表示示例单位；成本定义与论文主表单位尚未对齐，不能直接等同。"
               if s["cost"]["unit"]=="illustrative cost units" else "单位与成本定义取自输入配置。")
    text=f"""# 轨迹与同步曲线：配套图注

## 当前版本说明

{disclosure}图面采用论文排版；数据性质保留在本图注、输入配置、README、生成记录和 PDF 元数据中。

## 中文图注

同场景轨迹与同步诊断曲线{zh_status}。蓝色为 HD-MCRA，橙色为 RC-PPO，两者采用相同的冻结低层速度接口、起终点、机器人足迹和初始预算 {budget:g}。
(a) 实线表示执行轨迹，浅虚线表示相同命令理想跟踪下的名义轨迹。深灰为实体障碍，浅灰为计入机器人足迹的扩张区域，灰色点线在其外增加 δ={delta:g} m 的规划余量。局部图显示各方法经过 A 时的机器人足迹。
(b) 执行机身与实体障碍之间的间隙，零线表示接触边界。
(c) 执行与名义累计成本、共同预算，以及 HD-MCRA 的校正名义计费 (1+α)J_nom，其中 α={alpha:g}。
(d) 线速度跟踪误差的欧氏范数。A/B 对应各自首次采样越过指定位置的时刻（{event_defs}），不表示同时到达；各曲线在本方法首次终止时结束，跟踪误差绘至最后一个有动作记录的区间起点。

## English caption

{en_status}Same-scene trajectories and synchronized diagnostics for HD-MCRA (blue) and RC-PPO (orange), using a shared frozen low-level velocity interface, start, goal, footprint, and initial budget of {budget:g}. (a) Solid lines: executed trajectories; faint dashed lines: nominal trajectories under ideal tracking of the same commands. Dark gray: physical obstacles; pale gray: footprint-expanded regions; dotted boundaries: an additional planning margin δ={delta:g} m. The inset shows robot footprints at each method's event A. (b) Executed body-to-obstacle clearance; zero indicates contact. (c) Executed and nominal cumulative costs, the common budget, and HD-MCRA's calibrated nominal charge (1+α)J_nom, with α={alpha:g}. (d) Euclidean linear-velocity tracking error. A/B mark each method's own first sampled crossing ({event_defs}), rather than simultaneous arrivals. Each trace stops at its own first termination; tracking errors are available only at action-interval start times.

## 实验设置与结果解释

- 当前成本单位：{s['cost']['unit']}。{unit_note}成本函数和权重见输入 scenario.json；本次假设案例的背景见 data/versions/v2/protocol.md。
- 当前事件时刻：{time_text}。
- {limits}本图保留安全间隙、耗时和成本之间的取舍，不单独支持总体性能结论。
- 后续在新数据目录替换真实日志并重新计算；同时更新输入的数据来源、成本定义与配置。图注中的比较条件和论证边界也应随真实实验核对。
"""
    path.write_text(text,encoding="utf-8")


def render(source,out):
    source,out=Path(source).resolve(),Path(out).resolve()
    if out.exists():
        raise ValueError("Choose a new output directory to preserve reviewed outputs: "+str(out))
    s=read_json(source/"scenario.json")
    methods=["rcppo","hd_mcra"]
    inputs=[source/"scenario.json",source/"metadata.json"]+[source/f"raw_{m}.csv" for m in methods]
    before={str(p):sha(p) for p in inputs}
    # Validate supplied records before creating any output or calculating a plot.
    for m in methods:
        errors,_=validate(read_raw(source/f"raw_{m}.csv"),s,m)
        if errors:raise ValueError({m:errors})
    derived=out/"data";derived.mkdir(parents=True)
    for name in ["scenario.json","metadata.json"]:
        shutil.copy2(source/name,derived/name)
    build(source,derived,methods)
    for m in methods:
        _,rows=validate(read_raw(source/f"raw_{m}.csv"),s,m)
        errors=check_derived(derived/f"derived_{m}.csv",rows)
        if errors:raise ValueError({m:errors})
    data=load_data(derived)
    compose(data,out/"trajectory_sync",publication=True)
    trajectory(data,out/"panels/trajectory",publication=True)
    for k in KINDS:metric(data,k,out/"panels"/k,publication=True)
    write_caption(data,out/"caption.md")
    after={str(p):sha(p) for p in inputs}
    assert before==after,"Source input changed during rendering"
    comparisons={name:sha(source/name)==sha(derived/name)
                 for name in ["derived_rcppo.csv","derived_hd_mcra.csv","events.csv","summary.json"]
                 if (source/name).exists()}
    report=dict(data_status=s["data_status"],layout_version="publication_v4",source=str(source),
                source_sha256=before,source_unchanged=before==after,
                existing_derived_byte_equal=comparisons,validation_passed=True,
                outputs={str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.is_file()})
    (out/"render_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"output":str(out),"data_status":s["data_status"],"validated":True,
                      "source_unchanged":before==after,"existing_derived_byte_equal":comparisons},indent=2,ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data/versions/v2")
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args();render(args.data_dir,args.output_dir)
