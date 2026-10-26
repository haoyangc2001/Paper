"""Audit the internal manuscript fixture; never reports empirical validation."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / 'CCDC/submission/manuscript_completion_20261026'
DATA = ROOT / 'CCDC/data/manuscript_completion_20261026'
TEX = ROOT / 'CCDC/latex/1020/main.tex'

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def baseline():
    import fitz
    import matplotlib
    import numpy
    target = REVIEW / 'baseline.json'
    if target.exists():
        raise RuntimeError('Baseline already exists; preserve the original snapshot.')
    tex = TEX.read_text(encoding='utf-8')
    assets = [(TEX.parent / p).resolve() for p in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', tex)]
    sources = [TEX, ROOT/'CCDC/output/1020/main.pdf', TEX.parent/'IEEEtran.cls', TEX.parent/'references.bib',
        ROOT/'CORL/scripts/generate_experiment_results_v3.py', ROOT/'CORL/data/experiment_results_rcppo_ra_v3.xlsx',
        ROOT/'绘图/20261023_多方法指标对比/data/mc_extension_v2/assumed_intervals.jsonl',
        ROOT/'绘图/20261023_多方法指标对比/data/mc_extension_v2/protocol.json',
        ROOT/'绘图/20261021_轨迹与同步曲线/data/versions/v2/scenario.json',
        ROOT/'绘图/20261025_模块消融图/data/baseline_points.json',
        ROOT/'绘图/20261025_裕度校准与独立检验图/data/paired_rollouts.csv'] + assets
    files = [{ 'path': p.relative_to(ROOT).as_posix(), 'sha256':sha(p), 'bytes':p.stat().st_size } for p in sources]
    doc = fitz.open(ROOT/'CCDC/output/1020/main.pdf')
    result = {'git_head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'git_status': subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        'pages':len(doc), 'submission_limit':8, 'figures':tex.count('\\begin{figure}'),
        'tables':tex.count('\\begin{table}'), 'tbd_count':tex.count('\\textsc{TBD}'),
        'python':sys.executable, 'numpy':numpy.__version__, 'matplotlib':matplotlib.__version__,
        'tex_compiler_exists':Path('D:/app/latex/texlive/2026/bin/windows/pdflatex.exe').is_file(),
        'files':files, 'data_status':'internal_synthetic_working_manuscript', 'empirical_validation':False}
    assert (result['figures'],result['tables'],result['pages'],result['tbd_count']) == (7,2,9,16)
    save(target,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['files','git_status']},ensure_ascii=False))

def contract():
    p=read_json(DATA/'protocol.json')
    s=p['statistics']; low=p['low_level']; high=p['high_level']
    assert s['trials_per_scene']==3*100 and s['trials_per_method']==900
    assert low['PD_hz']/low['policy_hz']==4
    assert low['policy_hz']*p['measurement']['dt_s']==10
    assert high['PPO']['parallel_envs']*high['PPO']['rollout_steps']*high['PPO']['updates']==20006400
    assert high['PPO']['parallel_envs']*high['PPO']['rollout_steps']%high['PPO']['minibatches']==0
    assert p['measurement']['observations']['dimension']==36+2+3+3+1+1
    assert low['observations']==3+3+3+3+12+12+12
    expected={'without_alpha':[67.2,19.8,22.4], 'without_delta':[64.8,26.3,13.8]}
    for name,c in s['ablation_counts'].items():
        assert [round(100*c[k]/c['total'],1) for k in ['success','collision','BVR']]==expected[name]
        assert c['success']+c['collision']<=c['total']
    assert all(v['reference_budget']>=3.6*30 for v in p['scenarios']['families'].values())
    rows=[json.loads(line) for line in (DATA/'inference_ledger.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len({(r['id'],r['revision']) for r in rows})==len(rows)
    result={'passed':True,'ledger_items':len(rows),'checks':['counts and rounding','input dimensions','integer clock ratios','training batch','budget lower bound','ledger uniqueness'],'empirical_validation':False}
    save(REVIEW/'contract_audit.json',result)
    print(json.dumps(result))

def interval(center, n, metric, count=None, sd=None):
    import math
    from statistics import NormalDist
    if metric in ['SR','CR','BVR']:
        assert count is not None and 0<=count<=n and abs(center-100*count/n)<1e-8
        z=NormalDist().inv_cdf(.975); p=count/n; den=1+z*z/n
        mid=(p+z*z/(2*n))/den
        half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        return [100*(mid-half),100*(mid+half)]
    assert sd>0 and metric in ['Cost','MC']
    # Two-sided 95% Student critical values, df=n-1 (fixed protocol sizes).
    critical={300:1.967929669,30:2.045229642,900:1.962606174}[n]
    half=critical*sd/math.sqrt(n)
    assert center-half>=0
    return [center-half,center+half]

def check_group(rows,n=300):
    import math
    r={v['metric']:v for v in rows}; cr=r['CR']['estimate']/100
    assert r['SR']['estimate']+r['CR']['estimate']<=100+1e-9
    assert r['MC']['estimate']<2, 'MC metres, not centimetres'
    assert r['MC']['sd']>=r['MC']['estimate']*math.sqrt(cr/(1-cr)), 'zero-contact mass imposes variance lower bound'
    for v in rows:
        lo,hi=interval(v['estimate'],n,v['metric'],v.get('count'),v.get('sd'))
        assert lo<=v['estimate']<=hi

def pilot():
    import copy,csv
    p=read_json(DATA/'pilot.json'); check_group(p['points'])
    caught=[]
    for kind,metric,key,value in [('arithmetic','SR','count',254),('units','MC','estimate',40),('event overlap','CR','estimate',20)]:
        bad=copy.deepcopy(p['points'])
        next(v for v in bad if v['metric']==metric)[key]=value
        try: check_group(bad)
        except AssertionError: caught.append(kind)
    assert len(caught)==3
    with (ROOT/'绘图/20261025_裕度校准与独立检验图/data/paired_rollouts.csv').open(encoding='utf-8-sig') as f:
        row=next(csv.DictReader(f))
    assert abs(float(row['J_exec'])/float(row['J_nom'])-1-float(row['cost_overrun']))<1e-8
    result={'points':[{**v,'CI95':interval(v['estimate'],300,v['metric'],v.get('count'),v.get('sd'))} for v in p['points']], 'negative_checks_detected':caught,'calibration_pair':row['pair_id'],'passed':True,'empirical_validation':False}
    save(REVIEW/'pilot_audit.json',result); print(json.dumps(result))

def main_results():
    import csv,math
    source=ROOT/'绘图/20261023_多方法指标对比/data/mc_extension_v2/assumed_intervals.jsonl'
    old=[json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    with (DATA/'continuous_sd.csv').open(encoding='utf-8-sig') as f:
        spreads={(r['method'],r['scenario']):r for r in csv.DictReader(f)}
    p=read_json(DATA/'protocol.json'); rows=[]
    for r in old:
        v=dict(r); metric=r['metric']; s=spreads[r['method'],r['scenario']]
        v['legacy_interval']=[r['lower'],r['upper']]; v['n']=300
        if metric in ['SR','CR','BVR']:
            v['count']=round(r['estimate']*3)
            note=f"中心保持；计数{v['count']}/300；两个端点由Wilson95%唯一计算，不再独立设定。"
        else:
            v['sd']=float(s[metric+'_sd']); note=s['cost_reason' if metric=='Cost' else 'mc_reason']+'；两个端点由t(299)均值区间计算。'
        v['lower'],v['upper']=interval(v['estimate'],300,metric,v.get('count'),v.get('sd'))
        v['source_status']='computed_from_synthetic'; v['review']=note
        rows.append(v)
    assert len(rows)==105
    groups={}
    for r in rows: groups.setdefault((r['method'],r['scenario']),[]).append(r)
    cross=[]
    for (method,scene),group in groups.items():
        check_group(group); d={r['metric']:r for r in group}
        budget=p['scenarios']['families'][scene]['reference_budget']
        # Necessary variance bound for the specified positive exceedance mass.
        b=d['BVR']['estimate']/100; mu=d['Cost']['estimate']; sd=d['Cost']['sd']
        assert sd >= max(0,budget-mu)*math.sqrt(b/(1-b))
        cross.append({'method':method,'scene':scene,'timeout_percent':100-d['SR']['estimate']-d['CR']['estimate'],'cost_tail_variance_feasible':True,'MC_collision_mass_feasible':True})
    means={m:{metric:sum(r['estimate'] for r in rows if r['method']==m and r['metric']==metric)/3 for metric in ['SR','Cost','CR','BVR','MC']} for m in dict.fromkeys(r['method'] for r in rows)}
    save(DATA/'main_results.json',{'data_status':'synthetic_sufficient_statistics','empirical_validation':False,'n_per_scene':300,'points':rows,'equal_scene_means':means,'group_checks':cross})
    save(REVIEW/'main_results_audit.json',{'passed':True,'centers':105,'endpoints':210,'table_means':35,'groups':21,'legacy_sha256':sha(source),'checks':cross,'empirical_validation':False})
    # Deterministic conversion of individually reviewed decisions into ledger records.
    ledger=DATA/'main_result_ledger.jsonl'
    required=['id','revision','category','source_status','source_paths','original_value','chosen_value','unit','statistical_population','denominator_or_weighting','reasoning','related_ids','checks','conflicts_and_resolution','paper_locations','data_consumers','replacement_instructions']
    decisions=[]
    for r in rows:
        group=[v for v in rows if v['method']==r['method'] and v['scenario']==r['scenario']]
        related=[v['id'] for v in rows if v['method']==r['method'] or (v['scenario']==r['scenario'] and v['metric']==r['metric'])]
        values=['main-'+r['id'],1,'main_comparison','computed_from_synthetic',[source.relative_to(ROOT).as_posix(),'continuous_sd.csv'],{'center':r['estimate'],'interval':r['legacy_interval']},{'center':r['estimate'],'interval':[r['lower'],r['upper']],'n':300,'count':r.get('count'),'sd':r.get('sd')},'m' if r['metric']=='MC' else ('a.u.' if r['metric']=='Cost' else '%'),r['method']+'/'+r['scenario'],'300 conditional trials; table equal scene weights',r['review'],related,['center unchanged','interval computed','same-group SR+CR<=100','cost-tail / clearance-zero-mass variance bounds'],'Old manual range retained; new synthetic CI has different semantics.',['fig:metrics_draft','tab:main_results'],['main_results.json','Fig2 plot'],'Replace counts/SD using real trials; recompute interval and table mean.']
        decisions.append(dict(zip(required,values)))
    ledger.write_text('\n'.join(json.dumps(d,ensure_ascii=False) for d in decisions)+'\n',encoding='utf-8')
    print('PASS:105 centers,210 endpoints,35 means,21 joint checks; old input unchanged.')

def calibration():
    import csv,numpy as np
    from collections import Counter
    path=ROOT/'绘图/20261025_裕度校准与独立检验图/data/paired_rollouts.csv'
    with path.open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    assert len(rows)==1200 and len({r['pair_id'] for r in rows})==1200
    for r in rows:
        jn,je,dur=map(float,[r['J_nom'],r['J_exec'],r['duration_s']])
        assert min(jn,je)>=3.6*dur and abs(je/jn-1-float(r['cost_overrun']))<1e-8
    out={'data_status':'computed_from_synthetic','empirical_validation':False,'source_sha256':sha(path),'groups':{}}
    for split,n,percell in [('calibration',800,40),('held_out',400,20)]:
        rr=[r for r in rows if r['split']==split]; assert len(rr)==n
        assert set(Counter(r['condition_id'] for r in rr).values())=={percell}
        a=np.array([[float(r['cost_overrun']),float(r['max_deviation_m'])] for r in rr])
        covered=(a<=np.array([.18,.12])+1e-12)
        counts=covered.sum(axis=0).tolist(); joint=int(covered.all(axis=1).sum())
        out['groups'][split]={'n':n,'median':np.median(a,axis=0).tolist(),'q90':np.quantile(a,.9,axis=0,method='inverted_cdf').tolist(),'covered':counts,'joint':joint,'CI95_coverage':[interval(100*k/n,n,'SR',count=k) for k in counts]}
        if split=='calibration':
            cells=[a[[i for i,r in enumerate(rr) if r['condition_id']==c]] for c in sorted({r['condition_id'] for r in rr})]
            rng=np.random.default_rng(20261026)
            boots=np.array([np.quantile(np.concatenate([c[rng.integers(0,len(c),len(c))] for c in cells]),.9,axis=0,method='inverted_cdf') for _ in range(2000)])
            out['quantile_CI95']=np.quantile(boots,[.025,.975],axis=0).T.tolist()
    assert out['groups']['calibration']['q90']==[.18,.12]
    assert out['groups']['calibration']['covered']==[727,722]
    assert out['groups']['held_out']['covered']==[353,346] and out['groups']['held_out']['joint']==336
    save(DATA/'calibration_results.json',out)
    print(json.dumps(out))

def ablation_budget():
    p=read_json(DATA/'protocol.json'); means=read_json(DATA/'main_results.json')['equal_scene_means']
    abpath=ROOT/'绘图/20261025_模块消融图/data/reviewed_points.jsonl'
    ab=[json.loads(l) for l in abpath.read_text(encoding='utf-8').splitlines() if l.strip()]
    ab=[r for r in ab if r['configuration'] in ['FULL','ZERO','NO_ALPHA','NO_DELTA']]
    bp=ROOT/'绘图/20261025_预算扫描图/data/reviewed_points.jsonl'
    old=[json.loads(l) for l in bp.read_text(encoding='utf-8').splitlines() if l.strip()]
    assert len(ab)==20 and len(old)==96
    rows=[]
    for r in old:
        v=dict(r); value=r['value']; v['legacy_value']=value
        if r['metric']!='Cost':
            v['count']=round(value*900); v['n']=900; v['value']=v['count']/900
            assert abs(100*value-100*v['value'])<=100/1800+1e-9
        if r['budget_ratio']==1:
            assert abs(v['value']-(means[r['method']][r['metric']]/(1 if r['metric']=='Cost' else 100)))<1e-9
        v['decision']='retain displayed value; exact probability from nearest compatible integer count' if r['metric']!='Cost' else 'retain mean proxy cost'
        rows.append(v)
    grid=sorted({r['budget_ratio'] for r in rows}); selected={}
    for m in ['HD-MCRA','RC-PPO','RA']:
        passing=[]
        for z in grid:
            d={r['metric']:r['value'] for r in rows if r['method']==m and r['budget_ratio']==z}
            assert d['SR']+d['CR']<=1
            if d['SR']>=.6 and d['CR']<=.2 and d['BVR']<=.15: passing.append(z)
        selected[m]=min(passing)
    assert selected=={'HD-MCRA':.9,'RC-PPO':.96,'RA':1.12}
    ra=[r for r in rows if r['method']=='RA']
    for metric in ['SR','Cost','CR']: assert len({r['value'] for r in ra if r['metric']==metric})==1
    assert all(a>=b for a,b in zip([r['value'] for r in ra if r['metric']=='BVR'],[r['value'] for r in ra if r['metric']=='BVR'][1:]))
    save(DATA/'ablation_budget_results.json',{'data_status':'synthetic','empirical_validation':False,'ablation':ab,'budget':rows,'smallest_passing_tested':selected,'ablation_exact_counts':p['statistics']['ablation_counts'],'old_budget_sha256':sha(bp)})
    changes=[{'id':r['point_id'],'old':r['legacy_value'],'new':r['value']} for r in rows if round(100*r['legacy_value'],1)!=round(100*r['value'],1) and r['metric']!='Cost']
    print(json.dumps({'passed':True,'ablation':20,'budget':96,'anchors':12,'selected':selected,'display_corrections':changes}))

def margin():
    import csv
    with (DATA/'margin_sweep.csv').open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    out=[]
    for r in rows:
        v={k:(int(r[k]) if k in ['n','success_count','collision_count','BVR_count'] else float(r[k])) for k in ['alpha','delta_m','n','success_count','collision_count','BVR_count','Cost','MC']}
        assert v['n']==900 and v['success_count']+v['collision_count']<=900
        for metric,key in [('SR','success_count'),('CR','collision_count'),('BVR','BVR_count')]:
            v[metric]=100*v[key]/v['n']
        v['reasoning']={m:r[m+'_reason'] for m in ['SR','Cost','CR','BVR','MC']}
        assert all(s.strip() for s in v['reasoning'].values())
        out.append(v)
    assert len({(r['alpha'],r['delta_m']) for r in out})==16
    means=read_json(DATA/'main_results.json')['equal_scene_means']
    for a,d,m in [(0,0,'RC-PPO'),(.18,.12,'HD-MCRA')]:
        v=next(r for r in out if r['alpha']==a and r['delta_m']==d)
        assert all(abs(v[k]-means[m][k])<1e-8 for k in ['SR','Cost','CR','BVR','MC'])
    save(DATA/'margin_sweep_results.json',{'data_status':'authored_example','empirical_validation':False,'training':'retrain all three high-level networks per setting, same frozen tracker and fixed training budget; three seeds;900 final trials at reference budget','rows':out})
    print('PASS:16 settings,80 metric-specific reasons; four existing anchors, no interpolated grid.')

def hardware():
    p=read_json(DATA/'protocol.json'); h=p['hardware']; n=h['n']
    assert sum(r['n'] for r in h['outcomes'])==30
    assert sum(r['collision'] for r in h['outcomes'])==1
    assert sum(r['BVR'] for r in h['outcomes'])==3
    assert h['route_m']/.6<h['horizon_s'] and h['reference_budget']>3.6*h['horizon_s']
    vals={'SR':(26,None,None),'CR':(1,None,None),'BVR':(3,None,None),'Cost':(None,184,44),'MC':(None,.11,.07)}
    result={}
    for k,(count,center,sd) in vals.items():
        mu=100*count/n if count is not None else center
        result[k]={'estimate':mu,'count':count,'sd':sd,'CI95':interval(mu,n,k,count,sd)}
    save(DATA/'hardware_results.json',{'data_status':'computed_from_authored_example','empirical_validation':False,'n':30,'outcomes':h['outcomes'],'metrics':result})
    for m,r in p['perturbation']['results'].items():
        assert r['success_count']+r['collision_count']<=900 and 0<=r['BVR_count']<=900
    print(json.dumps(result))

def result_review():
    for path in ['main_results.json','calibration_results.json','ablation_budget_results.json','margin_sweep_results.json','perturbation_results.json','hardware_results.json']:
        assert read_json(DATA/path)['empirical_validation'] is False
    main=read_json(DATA/'main_results.json')['equal_scene_means']
    sweep=read_json(DATA/'margin_sweep_results.json')['rows']; ab=read_json(DATA/'ablation_budget_results.json')['ablation']
    for a,d,c in [(0,.12,'NO_ALPHA'),(.18,0,'NO_DELTA')]:
        row=next(r for r in sweep if r['alpha']==a and r['delta_m']==d)
        for metric in ['SR','Cost','CR','BVR','MC']:
            old=next(r for r in ab if r['configuration']==c and r['metric']==metric)
            assert round(row[metric],old['display_decimals'])==old['display_value']
    for c,m in [('FULL','HD-MCRA'),('ZERO','RC-PPO')]:
        for row in [r for r in ab if r['configuration']==c]: assert abs(row['value']-main[m][row['metric']])<1e-8
    result={'passed':True,'main_centers':105,'main_endpoints':210,'ablation_centers':20,'budget_points':96,'margin_settings':16,'calibration_records':1200,'hardware_n':30,'empirical_validation':False,'conclusion_scope':'conditional descriptive synthetic internal draft; no significant-difference or formal safety claim'}
    save(REVIEW/'result_audit.json',result); print(json.dumps(result))

def wording():
    tex=TEX.read_text(encoding='utf-8')
    forbidden=['TBD','provisional','placeholder','draft count','planned margin','remain separate assumptions','await verified']
    found={k:[i+1 for i,line in enumerate(tex.splitlines()) if k.lower() in line.lower()] for k in forbidden}
    assert not any(found.values()),found
    retained=[]
    for i,line in enumerate(tex.splitlines(),1):
        if re.search(r'draft|assumed|illustrative',line,re.I):
            reason='stable TeX reference/label' if 'draft}' in line else ('deterministic model assumption' if 'assumed' in line else 'method schematic paths, not measurements')
            retained.append({'line':i,'text':line,'reason':reason})
    labels=set(re.findall(r'\\label\{([^}]+)\}',tex)); refs=set(re.findall(r'\\ref\{([^}]+)\}',tex))
    assert refs<=labels,refs-labels
    assert tex.count('\\begin{figure}')==7 and tex.count('\\begin{table}')==2
    assert '\\begin{figure*}' not in tex and '\\begin{table*}' not in tex
    assert tex.index('\\bibliography{')<tex.index('\\appendices') and '\\clearpage\n\\appendices' in tex
    save(REVIEW/'wording_audit.json',{'passed':True,'forbidden_hits':found,'retained':retained,'labels':len(labels),'references':len(refs),'empirical_validation':False})
    print('PASS:no visible writing placeholders;7 figures,2 tables,single-column;references resolved.')

def figure_inputs():
    import shutil
    name='manuscript_completion_20261026'
    main=ROOT/'绘图/20261023_多方法指标对比'; target=main/'data'/name
    result=read_json(DATA/'main_results.json')
    p=read_json(main/'data/mc_extension_v2/protocol.json')
    p.update(version=name,data_status='synthetic_sufficient_statistics',empirical_validation=False,confidence_level=.95,interval_kind='conditional_95CI_from_synthetic_counts_and_SD',purpose='Internal complete manuscript fixture, not empirical results',interval_note='Synthetic point estimates and sufficient statistics; Wilson95% for proportions and t(299)95% for continuous means; conditional on fixed policy bank. No empirical measurements.',simulation_caption='Scenario-level SR,Cost,CR,BVR,MC with conditional95% intervals;n=300 per scene;table equal-scene means.',unresolved_for_empirical_replacement=['Replace synthetic sufficient statistics with verified trial records and training configurations.'])
    for m in p['metrics']:
        if m['id']=='Cost':m.update(unit='a.u.',axis_label='(a.u.)')
    p['interpretation']=read_json(DATA/'protocol.json')['statistics']
    save(target/'protocol.json',p)
    (target/'assumed_intervals.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in result['points'])+'\n',encoding='utf-8')
    shutil.copy2(DATA/'continuous_sd.csv',target/'continuous_sd.csv')
    b=ROOT/'绘图/20261025_预算扫描图'; bt=b/'data'/name
    budget=read_json(DATA/'ablation_budget_results.json')['budget']
    p=read_json(b/'data/plot_protocol.json'); p.update(completion_version=name,trial_count=900,reference_budgets=[120,140,190],cost_definition='same proxy as manuscript Appendix B; a.u.',review_log=f'data/{name}/point_review.jsonl')
    save(bt/'plot_protocol.json',p);shutil.copy2(b/'data/baseline_points.json',bt/'baseline_points.json')
    previous=[json.loads(l) for l in (b/'reviews/point_review.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    latest={r['point_id']:r for r in previous}; appended=[]
    for v in budget:
        v['review_decision']='deterministic_correction' if v['value']!=v['legacy_value'] else 'retain'
        r=dict(latest[v['point_id']]); r.update(revision=r['revision']+1,supersedes_revision=r['revision'],adopted_value=v['value'],decision=v['review_decision'],reasoning=r['reasoning']+' 本轮补齐900分母：取最近可兼容整数计数，变化不超过半次试验；主表锚点和选中预算保持。',decision_basis='User-authorized consistency correction;synthetic counts,not recovered logs.')
        v['review_revision']=r['revision']; appended.append(r)
    (bt/'point_review.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in previous+appended)+'\n',encoding='utf-8')
    (bt/'reviewed_points.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in budget)+'\n',encoding='utf-8')
    print('Prepared new per-figure data versions; legacy data untouched.')

def figure_audit():
    import fitz
    name='manuscript_completion_20261026'
    pairs=[('绘图/20261023_多方法指标对比/output/'+name+'/metrics_comparison.pdf','metrics_comparison.pdf',[220,90]),('绘图/20261025_预算扫描图/output/'+name+'/budget_sweep.pdf','budget_sweep_horizontal.pdf',[180,50])]
    out=[]
    for source,target,size in pairs:
        a=ROOT/source;b=ROOT/'CCDC/images/1020'/target
        assert sha(a)==sha(b)
        doc=fitz.open(b); actual=[doc[0].rect.width*25.4/72,doc[0].rect.height*25.4/72]
        assert all(abs(v-w)<.01 for v,w in zip(actual,size))
        out.append({'source':source,'paper':str(b.relative_to(ROOT)),'sha256':sha(b),'size_mm':actual})
    ab=fitz.open(ROOT/'CCDC/images/1020/ablation_horizontal.pdf')
    assert abs(ab[0].rect.width*25.4/72-110)<.01 and abs(ab[0].rect.height*25.4/72-43.5)<.01
    baseline=read_json(REVIEW/'baseline.json')
    for f in baseline['files']:
        if f['path'].startswith('CCDC/images/1020/') and Path(f['path']).name not in [p[1] for p in pairs]:
            assert sha(ROOT/f['path'])==f['sha256']
    save(REVIEW/'figure_manifest.json',{'passed':True,'updated':out,'unchanged':'overview,trajectory,calibration,ablation,two historical images','empirical_validation':False})
    print('PASS:2 updated PDF pairs hash-identical;220x90 and180x50;ablation110x43.5;other assets unchanged.')

def table_values(tex,label):
    start=tex.index('\\label{'+label+'}');end=tex.index('\\end{table}',start)
    rows=[]
    for line in tex[start:end].splitlines():
        if '&' not in line or line.startswith('Method ') or line.startswith('Variant '):continue
        parts=line.split('&')
        vals=[]
        for cell in parts[1:]:
            text=re.sub(r'\\textbf\{([^}]+)\}',r'\1',cell)
            nums=re.findall(r'[-+]?\d+(?:\.\d+)?',text)
            assert len(nums)==1,(label,cell)
            vals.append(float(nums[0]))
        rows.append(vals)
    return rows

def full_audit():
    import math
    tex=TEX.read_text(encoding='utf-8');p=read_json(DATA/'protocol.json')
    main=read_json(DATA/'main_results.json');ab=read_json(DATA/'ablation_budget_results.json')
    metrics=['SR','Cost','CR','BVR','MC']
    methods=['HD-MCRA','RC-PPO','RA','PPO_beta_L','PPO_beta_H','CPPO_beta_L','CPPO_beta_H']
    expected=[[round(main['equal_scene_means'][m][k],2 if k=='MC' else 1) for k in metrics] for m in methods]
    assert table_values(tex,'tab:main_results')==expected
    abl_expected=[[round(next(r['value'] for r in ab['ablation'] if r['configuration']==c and r['metric']==k),2 if k=='MC' else 1) for k in metrics] for c in ['FULL','ZERO','NO_ALPHA','NO_DELTA']]
    assert table_values(tex,'tab:ablation_draft')==abl_expected
    for r in main['points']:
        expected_ci=interval(r['estimate'],300,r['metric'],r.get('count'),r.get('sd'))
        assert all(abs(r[k]-v)<1e-8 for k,v in zip(['lower','upper'],expected_ci))
    figure=read_json(ROOT/'绘图/20261023_多方法指标对比/output/manuscript_completion_20261026/render_manifest.json')
    lookup={r['id']:r for r in figure['plotted_points']}
    for r in main['points']:
        assert all(lookup[r['id']][k]==r[k] for k in ['estimate','lower','upper'])
    budget=read_json(ROOT/'绘图/20261025_预算扫描图/output/manuscript_completion_20261026/render_manifest.json')
    lookup={r['point_id']:r for r in budget['plotted_points']}
    assert all(lookup[r['point_id']]['value']==r['value'] for r in ab['budget'])
    required=['73.3\\%','9.0\\%','70.7\\%','14.0\\%','353/400','346/400','336/400','13.3\\%','6.7\\%','68.0\\%','95.0 a.u.','96.8 and 102.4','0.26 and 0.24','184.0 a.u.','167.6--200.4','0.084--0.136']
    assert all(x in tex for x in required),[x for x in required if x not in tex]
    # Preserve template and original bibliography; no formatting shortcut.
    base=read_json(REVIEW/'baseline.json')
    for f in base['files']:
        if Path(f['path']).name in ['IEEEtran.cls','references.bib']:
            assert sha(ROOT/f['path'])==f['sha256']
    result={'passed':True,'main_table_cells':35,'ablation_table_cells':20,'figure2_centers':105,'figure2_endpoints':210,'budget_points':96,'margin_settings':16,'calibration_records':1200,'prose_numeric_checks':len(required),'template_unchanged':True,'empirical_validation':False}
    save(REVIEW/'full_audit.json',result); print(json.dumps(result))

def replacement_dry_run():
    import copy,tempfile,importlib.util
    root=REVIEW/'tmp';root.mkdir(exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix='replacement_',dir=root))
    assert tmp.resolve().parent==root.resolve()
    main=read_json(DATA/'main_results.json'); original_sha=sha(TEX)
    rows=copy.deepcopy(main['points']);r=next(r for r in rows if r['method']=='HD-MCRA' and r['scenario']=='moderate' and r['metric']=='SR')
    r['count']-=3;r['estimate']=100*r['count']/300;r['lower'],r['upper']=interval(r['estimate'],300,'SR',count=r['count'])
    mean=sum(v['estimate'] for v in rows if v['method']=='HD-MCRA' and v['metric']=='SR')/3
    assert mean==73
    tex=TEX.read_text(encoding='utf-8').replace('73.3\\% success','73.0\\% success')
    tex=tex.replace('\\textbf{HD-MCRA} & \\textbf{73.3}', '\\textbf{HD-MCRA} & \\textbf{73.0}')
    (tmp/'main.tex').write_text(tex,encoding='utf-8')
    assert table_values(tex,'tab:main_results')[0][0]==73 and table_values(tex,'tab:ablation_draft')[0][0]==73
    assert '73.0\\% success' in tex
    budget=copy.deepcopy(read_json(DATA/'ablation_budget_results.json')['budget'])
    anchor=next(v for v in budget if v['method']=='HD-MCRA' and v['budget_ratio']==1 and v['metric']=='SR');anchor['value']=mean/100;anchor['count']=657
    sweep=copy.deepcopy(read_json(DATA/'margin_sweep_results.json'));v=next(v for v in sweep['rows'] if v['alpha']==.18 and v['delta_m']==.12);v['SR']=mean;v['success_count']=657
    project=ROOT/'绘图/20261023_多方法指标对比';protocol=read_json(project/'data/manuscript_completion_20261026/protocol.json');protocol['verify_legacy_centers']=False
    save(tmp/'protocol.json',protocol);(tmp/'assumed_intervals.jsonl').write_text('\n'.join(json.dumps(v,ensure_ascii=False) for v in rows)+'\n',encoding='utf-8')
    spec=importlib.util.spec_from_file_location('replacement_plot',project/'scripts/plot_metrics_comparison.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    pr,rr,lookup,report=mod.load_and_check(tmp)
    output=tmp/'render';output.mkdir();qa=mod.render(pr,lookup,output)
    drawn=next(v for v in qa['plotted_points'] if v['id']==r['id']);assert drawn['estimate']==84 and drawn['lower']==r['lower']
    assert sha(TEX)==original_sha
    save(REVIEW/'replacement_audit.json',{'passed':True,'temporary_directory':str(tmp),'mutation':'HD-MCRA/moderate SR255→252 of300,85→84%','propagation':['CI recalculated','actual Fig2 artists84%','main table73.0%','ablation full73.0%','abstract73.0%','budget reference657/900','margin full657/900'],'real_manuscript_unchanged':True,'empirical_validation':False})
    print('PASS:temporary SR change propagated to actual rendered figure,table,abstract and shared anchors; real manuscript unchanged.')

def layout():
    import fitz
    from PIL import Image,ImageOps,ImageDraw
    doc=fitz.open(ROOT/'CCDC/output/1020/main.pdf');out=REVIEW/'tmp/layout';out.mkdir(parents=True,exist_ok=True)
    pages=[]
    for i,page in enumerate(doc):
        path=out/f'page-{i+1:02}.png';page.get_pixmap(matrix=fitz.Matrix(1.6,1.6)).save(path)
        words=page.get_text('words');bounds=[w[:4] for w in words]
        clipped=[b for b in bounds if b[0]<0 or b[1]<0 or b[2]>page.rect.width+.5 or b[3]>page.rect.height+.5]
        assert not clipped
        pages.append({'page':i+1,'image':str(path),'characters':len(page.get_text()),'clipped_words':len(clipped),'head':page.get_text()[:90]})
    for i in range(0,len(doc),2):
        ims=[Image.open(out/f'page-{j+1:02}.png').convert('RGB') for j in range(i,min(i+2,len(doc)))];w=sum(im.width for im in ims)
        canvas=Image.new('RGB',(w,max(im.height for im in ims)+32),'#d9e0e6');draw=ImageDraw.Draw(canvas);x=0
        for j,im in enumerate(ims):canvas.paste(im,(x,32));draw.text((x+12,8),f'Page {i+j+1}',fill='black');x+=im.width
        canvas.save(out/f'spread-{i//2+1:02}.png')
    log=(ROOT/'CCDC/output/1020/main.log').read_text(encoding='utf-8',errors='replace')
    assert 'Overfull' not in log and 'undefined' not in log.lower() and 'Rerun to get cross-references right' not in log
    result={'pages':len(doc),'submission_limit':8,'within_limit':len(doc)<=8,'page_checks':pages,'underfull_warning_count':log.count('Underfull'),'overfull':0,'undefined_references':0,'visual_review':'pending','empirical_validation':False}
    save(REVIEW/'layout_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='page_checks'}))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['baseline','contract','pilot','main','calibration','ablation_budget','margin','hardware','results','wording','figure_inputs','figures','full','replacement','layout'])
    args=parser.parse_args()
    if args.mode=='baseline': baseline()
    elif args.mode=='contract': contract()
    elif args.mode=='pilot': pilot()
    elif args.mode=='main': main_results()
    elif args.mode=='calibration': calibration()
    elif args.mode=='ablation_budget': ablation_budget()
    elif args.mode=='margin': margin()
    elif args.mode=='hardware': hardware()
    elif args.mode=='results': result_review()
    elif args.mode=='wording': wording()
    elif args.mode=='figure_inputs': figure_inputs()
    elif args.mode=='figures': figure_audit()
    elif args.mode=='full': full_audit()
    elif args.mode=='replacement': replacement_dry_run()
    elif args.mode=='layout': layout()
