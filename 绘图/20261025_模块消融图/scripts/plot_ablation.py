"""Publication-size, horizontal module-ablation chart from reviewed centers."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from check_ablation_data import read_json, read_lines, validate, summary

PROJECT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build(out, pilot=False, style_path=None):
    data_path = PROJECT / 'data/reviewed_points.jsonl'
    protocol_path = PROJECT / 'data/plot_protocol.json'
    style_path = style_path or PROJECT / 'config/style_horizontal.json'
    points, protocol, style = read_lines(data_path), read_json(protocol_path), read_json(style_path)
    reviews = read_lines(PROJECT / 'reviews/point_review.jsonl')
    errors = validate(points, protocol, reviews)
    if errors:
        raise ValueError(errors)
    values = {p['point_id']:p['value'] for p in points}
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': style['font_family'], 'font.size':style['font_pt'],
                         'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
                         'svg.hashsalt':'ablation-reviewed-centers', 'axes.unicode_minus':True,
                         'text.color':style['text'], 'axes.labelcolor':style['text'],
                         'xtick.color':style['muted'], 'ytick.color':style['text']})
    fig = plt.figure(figsize=(style['width_mm']/25.4, style['height_mm']/25.4), facecolor=style['background'])
    legend_style = style.get('legend', {})
    legend = [Line2D([],[],marker='o',ls='',color=style['colors']['FULL'],markersize=style['full_marker_pt'],label=legend_style.get('full','Full')),
              Line2D([],[],marker='o',ls='',markerfacecolor='white',markeredgecolor=style['muted'],markersize=style['ablated_marker_pt'],label=legend_style.get('ablated','Ablated'))]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=legend_style.get('anchor',(0.58,1.02)), ncol=2,
               frameon=False, fontsize=style['font_pt'], handletextpad=.45, columnspacing=legend_style.get('columnspacing',1.6), handlelength=.9)
    state_matrix = style.get('state_matrix')
    if state_matrix:
        # State identities follow the existing deletion definitions, not point values.
        expected_states = {'NO_ALPHA':[False,True], 'NO_DELTA':[True,False], 'ZERO':[False,False]}
        if state_matrix['states'] != expected_states:
            raise ValueError('Module-state matrix disagrees with deletion configurations')
        for x, label in zip(state_matrix['x'], [r'$\alpha$', r'$\delta$']):
            fig.text(x, state_matrix['header_y'], label, ha='center', va='center', fontsize=style['small_font_pt'])
    active = ['SR','BVR'] if pilot else protocol['metric_order']
    mapping = []
    numeric_labels = []
    axes = []
    for column, metric in enumerate(protocol['metric_order']):
        layout = style['axes']; panel = style['panels'][metric]
        ax = fig.add_axes([layout['left']+column*(layout['width']+layout['gap']), layout['bottom'],layout['width'],layout['height']])
        axes.append(ax)
        ax.set_xlim(panel['xlim']); ax.set_ylim(layout['ylim'])
        ax.set_xticks(panel['ticks']); ax.set_yticks(style['row_y'])
        ax.tick_params(axis='x',labelsize=style['small_font_pt'],length=2,width=style['axis_pt'],pad=3)
        ax.tick_params(axis='y',length=0,pad=4)
        ax.set_yticklabels([r'w/o $\alpha$',r'w/o $\delta$','w/o both'] if column==0 and not state_matrix else [])
        for tick,cfg in zip(ax.get_yticklabels(),protocol['row_order']):
            tick.set_color(style['colors'][cfg]);tick.set_fontsize(style['small_font_pt'])
        if 'title_y' in style:
            ax.text(.5,style['title_y'],panel['title'],transform=ax.transAxes,ha='center',va='center',
                    fontsize=style['title_font_pt'],fontweight=style.get('title_weight','medium'))
        else:
            ax.set_title(panel['title'],fontsize=style['title_font_pt'],fontweight='semibold',pad=5)
        ax.set_xlabel(panel['unit'],fontsize=style['small_font_pt'],labelpad=2)
        for side in ['top','left','right']:
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#A3AFBB');ax.spines['bottom'].set_linewidth(style['axis_pt'])
        ax.set_axisbelow(True); ax.grid(axis='x',color=style['grid'],linewidth=style['grid_pt'])
        for row,y in enumerate(style['row_y']):
            if 'row_band_alpha' in style:
                ax.axhspan(y-.30,y+.30,color=style['colors'][protocol['row_order'][row]],alpha=style['row_band_alpha'],zorder=0)
            else:
                ax.axhspan(y-.29,y+.29,color='#F5F7F9' if row%2==0 else '#FAFBFC',zorder=0)
            if column==0 and state_matrix:
                figure_y = layout['bottom']+(y-layout['ylim'][0])/(layout['ylim'][1]-layout['ylim'][0])*layout['height']
                cfg = protocol['row_order'][row]
                for x, enabled in zip(state_matrix['x'],state_matrix['states'][cfg]):
                    fig.text(x,figure_y,'✓' if enabled else '×',ha='center',va='center',
                             color=style['colors'][cfg],fontsize=style['font_pt'])
        if metric not in active:
            ax.text(.5,.5,'',transform=ax.transAxes)
            continue
        full = values['FULL_'+metric.upper()]
        if 'reference_labels' in style:
            ref = style['reference_labels']
            label = f"{ref['prefix']}{full:.1f}"
            ax.text(.5,ref['y'],label,transform=ax.transAxes,ha='center',va='center',
                    color=style['colors']['FULL'],fontsize=ref['font_pt'])
            numeric_labels.append({'kind':'reference','metric':metric,'value':full,'text':label})
        ax.axvline(full,color=style['colors']['FULL'],lw=.55,alpha=.23,ls=(0,(2,2)),zorder=1)
        for cfg,y in zip(protocol['row_order'],style['row_y']):
            val = values[cfg+'_'+metric.upper()]
            color = style['colors'][cfg]
            ax.plot([full,val],[y,y],color=color if style.get('connector_by_row') else style['connector'],
                    alpha=style.get('connector_alpha',1),lw=style['connector_pt'],solid_capstyle='round',zorder=2)
            # Ablated hollow markers preserve identity when centers almost overlap.
            ax.plot(val,y,marker=style['markers'][cfg],ms=style['ablated_marker_pt'],mfc='white',mec=color,mew=style['marker_edge_pt'],ls='',zorder=4)
            ax.plot(full,y,marker=style['markers']['FULL'],ms=style['full_marker_pt'],mfc=style['colors']['FULL'],mec='white',mew=.35,ls='',zorder=5)
            mapping.append({'metric':metric,'configuration':cfg,'full_value':full,'ablated_value':val,'difference':val-full,'row_y':y})
            if 'value_labels' in style:
                labels = style['value_labels']
                label = f'{val:.1f}'
                ax.annotate(label,(val,y),xytext=(0,labels['offset_pt']),textcoords='offset points',
                            ha='center',va='bottom',color=color,fontsize=labels['font_pt'])
                numeric_labels.append({'kind':'ablated','metric':metric,'configuration':cfg,'value':val,'text':label})
            for annotation in style['annotations']:
                if annotation['metric']!=metric or annotation['configuration']!=cfg:
                    continue
                label=f'{val-full:+.1f} pp'
                if 'offset_pt' in annotation:
                    ax.annotate(label,((full+val)/2,y),xytext=(0,annotation['offset_pt']),textcoords='offset points',
                                ha='center',va='top',color=color,fontsize=style['small_font_pt'])
                else:
                    ax.text((full+val)/2,y+.35,label,ha='center',va='center',color=color,fontsize=style['small_font_pt'],fontweight='medium')
                numeric_labels.append({'kind':'difference','metric':metric,'configuration':cfg,'value':val-full,'text':label})
    # Shared tracker and budget conditions are stated once in the companion caption.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_boxes = []
    for item in fig.findobj(matplotlib.text.Text):
        if not item.get_visible() or not item.get_text():
            continue
        box=item.get_window_extent(renderer)
        text_boxes.append({'text':item.get_text(),'font_pt':item.get_fontsize(),
                           'bbox_figure':[box.x0/fig.bbox.width,box.y0/fig.bbox.height,box.x1/fig.bbox.width,box.y1/fig.bbox.height]})
    base = 'ablation_pilot' if pilot else 'ablation_horizontal'
    for ext in ['pdf','svg','png']:
        metadata = {'Title':'Module ablation - hypothetical centers', 'Creator':'plot_ablation.py'} if ext=='pdf' else ({'Title':'Module ablation - hypothetical centers'} if ext=='svg' else {'Description':'Hypothetical centers; not experimental observations.'})
        fig.savefig(out/(base+'.'+ext),dpi=style['dpi'],facecolor=style['background'],metadata=metadata)
    manifest={'data_status':'hypothetical','active_metrics':active,'layout':'1x4','width_mm':style['width_mm'],'height_mm':style['height_mm'],
              'comparisons':mapping,'distinct_main_centers':len({('FULL',r['metric']) for r in mapping}|{(r['configuration'],r['metric']) for r in mapping}),
              'marker_instances':2*len(mapping),'inputs':{str(p.relative_to(PROJECT)):sha(p) for p in [data_path,protocol_path,style_path]},
              'numeric_labels':numeric_labels,'module_states':state_matrix['states'] if state_matrix else None,
              'script_sha256':sha(Path(__file__)),'text_boxes':text_boxes,'files':[base+'.'+e for e in ['pdf','svg','png']],
              'matplotlib_version':matplotlib.__version__,'python':sys.version.split()[0]}
    validation=validate(points,protocol,reviews,manifest)
    if validation:
        raise ValueError(validation)
    manifest['data_mapping_passed']=True
    (out/'render_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if not pilot:
        differences={(r['metric'],r['configuration']):r['difference'] for r in mapping}
        cost_budget_difference=differences[('BVR','NO_ALPHA')]
        collision_difference=differences[('CR','NO_DELTA')]
        state_en = ('The alpha/delta columns indicate enabled (check) or disabled (cross) margins; '
                    'full HD-MCRA enables both, and the both-disabled row is RC-PPO. '
                    'Header values are the full-method references; labels at hollow points are deletion values. ') if state_matrix else ''
        state_zh = ('左侧α/δ列中勾表示启用、叉表示关闭；完整HD-MCRA两者均启用，双叉行对应RC-PPO。'
                    '栏顶蓝色数值为完整方法参照，空心点旁为删除配置数值。') if state_matrix else ''
        deletion_en = '(both margins disabled is RC-PPO)' if state_matrix else '(w/o both is RC-PPO)'
        deletion_zh = '两者均关闭对应RC-PPO' if state_matrix else 'w/o both对应RC-PPO'
        caption=(
            '# Figure caption\n\n## English\n\n'
            'Component ablation at the common reference budget (all values are assumed data). '
            'Blue filled circles show full HD-MCRA; colored hollow markers show the indicated deletion variant '
            +deletion_en+', with the same frozen tracker. '+state_en+'Segments connect aggregate centers, not uncertainty '
            'intervals; each row reuses the same full-method reference. '
            f'The changes relative to the full method are {cost_budget_difference:+.1f} percentage points in BVR '
            f'without alpha and {collision_difference:+.1f} percentage points in CR without delta. '
            'Cost averages all trials up to termination and must be interpreted with SR. '
            'Axes use local ranges.\n\n## 中文\n\n'
            '相同参考预算下的模块消融（全部为假设数据）。蓝色实心圆为完整HD-MCRA，彩色空心点为对应删模块版本；'
            +deletion_zh+'，均使用同一冻结跟踪器。'+state_zh+'连接线表示汇总中心值差异，不是误差区间；'
            '每一行重复使用同一完整方法参照。'
            f'当前假设值中，去α的BVR变化为{cost_budget_difference:+.1f}个百分点，'
            f'去δ的CR变化为{collision_difference:+.1f}个百分点。'
            '成本统计所有试验截至终止的累计执行成本，需要结合成功率解释。坐标采用局部范围。\n\n'
            '## Reading notes\n\n'
            '- Panel order: SR, Cost, BVR, CR. Rates are percentages; highlighted differences use percentage points.\n'
            '- Three deletion rows are categories, not time or a parameter sweep.\n'
            '- Full and RC-PPO use equal-scene means at z0/zref = 1. Cost units remain draft units.\n'
            '- 12 segments and 24 markers reuse 16 distinct centers; no measured confidence intervals are supplied.\n'
            '- Single-margin settings need not improve monotonically over the no-margin setting.\n'
            '- MC and the joint-space interface comparison remain in the existing table.\n'
            '- Replace assumptions only with verified trial/seed/map data and matching statistical definitions.\n'
        )
        (out/'caption.md').write_text(caption,encoding='utf-8')
    plt.close(fig)
    print(json.dumps({'output':str(out),'mapping_passed':True,'comparisons':len(mapping),'markers':len(mapping)*2,'width_mm':style['width_mm'],'min_font_pt':min(t['font_pt'] for t in text_boxes)},ensure_ascii=False))
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--style',type=Path,help='Optional project style JSON; defaults to the original v1 style.')
    args=parser.parse_args()
    build(args.output.resolve(),args.pilot,args.style.resolve() if args.style else None)
