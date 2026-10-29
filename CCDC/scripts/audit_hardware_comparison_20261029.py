"""Read-only derivation and checks for authored hardware-comparison examples.

No experiment outcomes are generated here. --derive prints statistics computed
from the reviewed ledger; default verifies the saved result. No file is written.
"""
from pathlib import Path
from statistics import NormalDist, mean, stdev
import argparse
import hashlib
import json
import math
import re

from method_names import display_name

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'CCDC/data/hardware_comparison_20261029'
TEX = ROOT / 'CCDC/latex/1020/main.tex'
METHODS = ('HD-MCRA', 'RC-PPO')
OUTCOMES = ('success', 'localization_abort', 'timeout', 'physical_contact', 'manual_takeover')
ORDER = ('SR', 'Cost', 'CR', 'BVR', 'MC')
TC49 = 2.0095752371292397


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def wilson(k, n=50):
    assert isinstance(k, int) and 0 <= k <= n
    z = NormalDist().inv_cdf(.975)
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return [max(0., 100*(center-half)), min(100., 100*(center+half))]


def t_cdf(t, df, intervals=4000):
    # Independent numerical check of the stored df49 critical value.
    constant = math.exp(math.lgamma((df+1)/2)-math.lgamma(df/2))/math.sqrt(df*math.pi)
    step = t/intervals
    f = lambda x: constant*(1+x*x/df)**(-(df+1)/2)
    area = step/3*(f(0)+f(t)+sum((4 if i % 2 else 2)*f(i*step) for i in range(1, intervals)))
    return .5+area


def pilot():
    assert wilson(0)[0] == 0 and wilson(50)[1] == 100
    assert math.isclose(wilson(0)[1], 7.134759913319615, abs_tol=1e-9)
    z = NormalDist().inv_cdf(.975)
    half_case_lower = 100*(.5-z/(2*math.sqrt(50+z*z)))
    assert math.isclose(wilson(25)[0], half_case_lower, abs_tol=1e-10)
    assert math.isclose(wilson(25)[0]+wilson(25)[1], 100, abs_tol=1e-9)
    assert math.isclose(t_cdf(TC49, 49), .975, abs_tol=1e-10)
    assert math.isclose(t_cdf(12.706204736, 1), .975, abs_tol=1e-9)
    return {'passed': True, 'Wilson_0_50': wilson(0), 'Wilson_25_50': wilson(25),
            'Wilson_50_50': wilson(50), 't49': TC49, 't49_cdf': t_cdf(TC49, 49)}


def load_inputs():
    p = read(DATA/'protocol.json')
    rows = [json.loads(line) for line in (DATA/'inference_ledger.jsonl').read_text(encoding='utf-8').splitlines() if line]
    required = {'id','revision_of','method','metric_or_outcome','data_status','empirical_validation',
                'source_paths','old_value','new_value','unit','denominator','reasoning','related_entry_ids',
                'cross_metric_constraints','validation_checks','affected_consumers'}
    seen = {}
    inputs = {method: {} for method in METHODS}
    for row in rows:
        assert required <= row.keys() and row['id'] not in seen
        assert set(row['related_entry_ids']) <= seen.keys()
        if row['revision_of'] is not None:
            assert row['revision_of'] in seen
            previous = seen[row['revision_of']]
            assert row['method'] == previous['method']
            assert row['metric_or_outcome'] == previous['metric_or_outcome']
            assert row['old_value'] == previous['new_value']
        assert row['data_status'] == 'synthetic_example' and row['empirical_validation'] is False
        assert row['denominator'] == 50 and row['reasoning'] and row['validation_checks']
        seen[row['id']] = row
        inputs[row['method']][row['metric_or_outcome']] = (row['new_value'], row['id'])
    assert set(p['methods']) == set(METHODS) and p['total_instances'] == 100
    assert all(p['methods'][m]['n'] == 50 for m in METHODS)
    return p, inputs, rows


def derive():
    p, inputs, _ = load_inputs()
    methods = {}
    for method in METHODS:
        values = inputs[method]
        assert len(values) == 14
        outcomes = {name: values[name+'_count'][0] for name in OUTCOMES}
        cross = {name: values['BVR_'+name][0] for name in OUTCOMES}
        metrics = {}
        for metric, count in [('SR',outcomes['success']), ('CR',outcomes['physical_contact']), ('BVR',sum(cross.values()))]:
            metrics[metric] = {'estimate':100*count/50, 'count':count, 'CI95':wilson(count), 'unit':'percent'}
        for metric in ('Cost','MC'):
            mu, sd = values[metric+'_mean'][0], values[metric+'_sd'][0]
            half = TC49*sd/math.sqrt(50)
            metrics[metric] = {'estimate':mu, 'sd':sd, 'CI95':[mu-half,mu+half], 'unit':'a.u.' if metric=='Cost' else 'm'}
        methods[method] = {'n':50, 'outcomes':outcomes, 'budget_outcome_cross_counts':cross,
                           'metrics':{k:metrics[k] for k in ORDER},
                           'ledger_input_ids':{k:v[1] for k,v in values.items()}}
    differences = {k:methods['HD-MCRA']['metrics'][k]['estimate']-methods['RC-PPO']['metrics'][k]['estimate'] for k in ORDER}
    return {'schema_version':'1.0', 'result_revision':'v2', 'data_status':'synthetic_example', 'empirical_validation':False,
            'source':'individually authored counts and sufficient statistics in inference_ledger.jsonl',
            'intervals':{'rates':'Wilson95 n50', 'continuous':'Student95 df49 from synthetic mean/sample SD',
                         't_critical':TC49, 'interpretation':'illustrative marginal intervals, not measured uncertainty'},
            'methods':methods, 'differences_HD_minus_RC':differences,
            'difference_interpretation':'point differences only; paired joint outcomes unavailable; no significance claim'}


def feasible_cost(entry, threshold, tail):
    # Algebraic witness for moments/tail count, never a trial log or generator.
    n = entry['n']; k = entry['metrics']['BVR']['count']
    within_timeouts = entry['outcomes']['timeout'] - entry['budget_outcome_cross_counts']['timeout']
    timeout_floor = 3.6*60
    timeout_cost = (timeout_floor+threshold)/2
    low_count = sum(entry['outcomes'][name]-entry['budget_outcome_cross_counts'][name]
                    for name in ('physical_contact','localization_abort','manual_takeover'))
    high_count = entry['outcomes']['success']-entry['budget_outcome_cross_counts']['success']
    m = low_count+high_count
    assert n == k+within_timeouts+m and low_count > 0 and high_count > 0
    mu, sd = entry['metrics']['Cost']['estimate'], entry['metrics']['Cost']['sd']
    center = (n*mu-k*tail-within_timeouts*timeout_cost)/m
    between = k*(tail-mu)**2+within_timeouts*(timeout_cost-mu)**2+m*(center-mu)**2
    residual = (n-1)*sd**2-between
    assert residual >= 0
    gap = math.sqrt(residual*m/(low_count*high_count))
    low, high = center-high_count/m*gap, center+low_count/m*gap
    witness = [tail]*k+[timeout_cost]*within_timeouts+[low]*low_count+[high]*high_count
    assert min(witness) > 0 and low < high < threshold < tail
    # Nonexceeding successes get the high group; early failures the low group.
    # A 15m traversal must pay at least integral(3.6+3*v^2)dt >= 2*sqrt(10.8)*15.
    success_cost_floor = 2*math.sqrt(3.6*3)*15
    assert min(high,tail) >= success_cost_floor
    assert math.isclose(mean(witness), mu, abs_tol=1e-10) and math.isclose(stdev(witness), sd, abs_tol=1e-10)
    assert sum(x > threshold for x in witness) == k
    assert tail >= timeout_floor and timeout_floor <= timeout_cost <= threshold
    return {'passed':True, 'purpose':'mathematical witness only', 'tail_count':k, 'tail':tail,
            'within_timeout_count':within_timeouts, 'within_timeout_cost':timeout_cost,
            'early_failure_count':low_count, 'early_failure_cost':low,
            'within_success_count':high_count, 'within_success_cost':high,
            'success_cost_lower_bound':success_cost_floor}


def feasible_mc(entry):
    n = entry['n']; zeros = entry['metrics']['CR']['count']; m = n-zeros
    mu, sd = entry['metrics']['MC']['estimate'], entry['metrics']['MC']['sd']
    center = n*mu/m
    residual = (n-1)*sd**2+n*mu*mu-m*center*center
    assert residual >= 0
    # Search only for a moment witness after inputs are fixed, not for outcomes.
    candidates = []
    for high_count in range(1,m):
        low_count = m-high_count
        gap = math.sqrt(residual*m/(high_count*low_count))
        low, high = center-high_count/m*gap, center+low_count/m*gap
        if 0 < low < high <= .15:
            candidates.append((abs(high_count-m/2),high_count,low_count,low,high))
    assert candidates, 'No bounded moment witness for zero contacts and MC'
    _, high_count, low_count, low, high = min(candidates)
    witness = [0.]*zeros+[low]*low_count+[high]*high_count
    assert 0 < low < high <= .15
    assert math.isclose(mean(witness), mu, abs_tol=1e-12) and math.isclose(stdev(witness), sd, abs_tol=1e-12)
    return {'passed':True, 'purpose':'mathematical witness only', 'zeros':zeros,
            'low_count':low_count,'low':low,'high_count':high_count,'high':high,
            'geometry_bound_m':.15}


def preserved(p):
    rename_manifest = ROOT/'CCDC/output/1020/camra_rename_20261029/figure_manifest.json'
    if rename_manifest.exists():
        from rename_camra_figures import verify_preserved
        return verify_preserved(p)
    for item in p['files_preserved']:
        assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
    return {'passed':True,'files':len(p['files_preserved'])}


def history_checks(rows):
    original = (DATA/'inference_ledger.jsonl').read_bytes().splitlines(keepends=True)[:28]
    assert hashlib.sha256(b''.join(original)).hexdigest() == '84667b1cfac36d97064e47bd8e03c8bf08a4993098851e5236e5ccc7a34b8a1c'
    historical = read(DATA/'history/v1_hardware_comparison_results.json')
    for method in METHODS:
        inputs = {r['metric_or_outcome']:r['new_value'] for r in rows[:28] if r['method']==method}
        old = historical['methods'][method]
        assert all(old['outcomes'][name] == inputs[name+'_count'] for name in OUTCOMES)
        assert all(old['budget_outcome_cross_counts'][name] == inputs['BVR_'+name] for name in OUTCOMES)
        assert all(old['metrics'][metric][field] == inputs[metric+'_'+key]
                   for metric in ('Cost','MC') for field,key in [('estimate','mean'),('sd','sd')])
    assert len(rows) == 56 and all(r['revision_of'] for r in rows[28:])
    return {'passed':True,'original_entries_unchanged':28,'revision_entries':28,'v1_snapshot_matches':True}


def check_tex(result):
    text = TEX.read_text(encoding='utf-8')
    sec = text.split('\\label{sec:hardware_deployment}',1)[1].split('\\section{Limitations}',1)[0]
    table = sec.split('\\label{tab:hardware_comparison}',1)[1].split('\\end{table}',1)[0]
    for method, entry in result['methods'].items():
        label = display_name(method)
        line = next(line for line in table.splitlines() if label in line and '&' in line)
        cells = line.split('&')[1:]
        assert len(cells) == 5
        for metric, cell in zip(ORDER,cells):
            actual = float(re.search(r'\d+(?:\.\d+)?',cell).group())
            decimals = 2 if metric == 'MC' else 1
            assert actual == round(entry['metrics'][metric]['estimate'],decimals), (method,metric)
    assert 'Synthetic' in sec and '50' in sec
    assert sec.count('\\begin{table}') == 1 and '\\begin{table*}' not in text
    assert '\\begin{figure*}' not in text and text.count('\\label{tab:hardware_comparison}') == 1
    assert not re.search(r'26/30|n=30\b|out of 30|uses 30 trials|184\.0 a\.u\.|167\.6--200\.4|0\.084--0\.136',text)
    appendix = text.split('\\section{Experimental and Statistical Evaluation Details}',1)[1]
    assert '$n=50$' in appendix and '49 degrees of freedom' in appendix
    words = {0:'zero',1:'one',2:'two',3:'three',4:'four',5:'five',6:'six',7:'seven',8:'eight',9:'nine',10:'ten',11:'eleven'}
    for item in result['methods'].values():
        counts = item['outcomes']
        assert f"{counts['success']}/50" in sec
        assert f"{words[counts['localization_abort']]} localization aborts" in sec
        assert f"{words[counts['timeout']]} timeouts and {words[counts['physical_contact']]} contacts" in sec
    assert '43/50' not in sec and '40/50' not in sec
    # Body and appendix text intentionally use different display precision.
    for item in result['methods'].values():
        for metric in ORDER:
            digits = 3 if metric=='MC' else 1
            interval = '--'.join(f'{x:.{digits}f}' for x in item['metrics'][metric]['CI95'])
            assert interval in appendix, interval
        cross = item['budget_outcome_cross_counts']
        phrase = f"{words[cross['success']]} successful trials, {words[cross['timeout']]} timeouts, and {words[cross['physical_contact']]} "
        assert phrase in re.sub(r'\s+',' ',appendix), phrase
        assert f"{item['metrics']['Cost']['sd']:.1f}" in appendix
        assert f"{item['metrics']['MC']['sd']:.3f}" in appendix
    return {'passed':True,'table_cells':10,'figures':text.count('\\begin{figure}'),
            'tables':text.count('\\begin{table}'),'old_hardware_claims':0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--derive',action='store_true')
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--check-tex',action='store_true')
    args = parser.parse_args()
    small = pilot()
    if args.pilot:
        print(json.dumps(small,indent=2)); return
    result = derive()
    if args.derive:
        print(json.dumps(result,indent=2)); return
    p, _, rows = load_inputs()
    assert read(DATA/'hardware_comparison_results.json') == result
    checks = {}
    for method, entry in result['methods'].items():
        counts, cross = entry['outcomes'], entry['budget_outcome_cross_counts']
        assert sum(counts.values()) == 50 and all(isinstance(v,int) and v >= 0 for v in counts.values())
        assert all(isinstance(cross[k],int) and 0 <= cross[k] <= counts[k] for k in counts)
        for metric,item in entry['metrics'].items():
            assert 0 <= item['CI95'][0] <= item['estimate'] <= item['CI95'][1]
            if metric in ('SR','CR','BVR'):
                assert item['estimate'] == 2*item['count'] and item['CI95'][1] <= 100
        checks[method] = {'counts':True,'BVR_cross_counts':True,'intervals':True,
                          'cost_feasibility':feasible_cost(entry,230,250 if method=='HD-MCRA' else 255),
                          'MC_feasibility':feasible_mc(entry)}
    report = {'passed':True,'data_status':'synthetic_example','empirical_validation':False,
              'pilot':small,'ledger_entries':len(rows),'methods':checks,'preserved':preserved(p),
              'history':history_checks(rows)}
    if args.check_tex:
        report['manuscript'] = check_tex(result)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
