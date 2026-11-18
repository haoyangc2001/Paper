"""Finite critic budget decisions with an explicit, empirical one-step reserve."""
import math

GRID = (0.70, 0.85, 1.00, 1.15, 1.30)


def select_guarded_budget(query, e_ref_J, *, z_ref_J=550.0):
    """Query all five budgets before choosing; never consult trajectory labels."""
    if not math.isfinite(e_ref_J) or e_ref_J <= 0 or z_ref_J != 550.0:
        raise ValueError('invalid budget units or fixed scale')
    trace = []
    for k in GRID:
        z = k * e_ref_J
        vc, vh = map(float, query(z))
        finite = math.isfinite(vc) and math.isfinite(vh)
        w = max(vh, vc-z/z_ref_J) if finite else None
        trace.append({'k': k, 'budget_J': z, 'vc_norm': vc if math.isfinite(vc) else None,
                      'vh': vh if math.isfinite(vh) else None, 'W': w,
                      'raw_feasible': w is not None and w <= 0})
    result = {'selected_index': None, 'selected_k': None, 'selected_budget_J': None,
              'first_predicted_index': None, 'reason': None, 'trace': trace}
    if any(r['W'] is None for r in trace):
        result['reason'] = 'nonfinite'
    elif any(r['W'] == 0 for r in trace):
        result['reason'] = 'uncertain_zero'
    else:
        flags = [r['W'] < 0 for r in trace]
        if any(flags[i] and not flags[i+1] for i in range(4)):
            result['reason'] = 'non_monotonic'
        elif not any(flags):
            result['reason'] = 'no_feasible_grid_budget'
        else:
            j = flags.index(True)
            result['first_predicted_index'] = j
            if j == 4:
                result['reason'] = 'no_reserve_within_grid'
            else:
                result.update(selected_index=j+1, selected_k=GRID[j+1],
                              selected_budget_J=trace[j+1]['budget_J'], reason='selected')
    return result


def summarize_decisions(decisions, episodes, maps):
    """Audit actual selections independently, with non-vacuous denominators."""
    map_ids = {m['map_id'] for m in maps}
    if len(map_ids) != len(maps) or len(maps) != 9:
        raise ValueError('nine distinct registered maps required')
    if len(decisions) != 18 or len(episodes) != 90:
        raise ValueError('incomplete decision/trajectory coverage')
    summaries = {}
    for method in ('uncalibrated', 'full'):
        ds = {d['map_id']: d for d in decisions if d['method'] == method}
        rows = [e for e in episodes if e['method'] == method]
        keyed = {(e['map_id'], e['k']):e for e in rows}
        if set(ds) != map_ids or len(keyed) != len(rows) or set(keyed) != {(m,k) for m in map_ids for k in GRID}:
            raise ValueError('duplicated or incomplete method coverage')
        accepted = feasible_maps = correct = failures = efficiency_failures = 0
        reasons = {}
        levels = {m['level']:0 for m in maps}
        details = []
        confusion = dict(true_feasible=0, false_feasible=0, false_infeasible=0, true_infeasible=0)
        for m in maps:
            mid = m['map_id']
            d = ds[mid]
            actual = [keyed[mid,k]['episode']['feasible'] for k in GRID]
            feasible_maps += any(actual)
            for i, q in enumerate(d['decision']['trace']):
                pred, truth = q['raw_feasible'], actual[i]
                key = ('true_feasible' if truth else 'false_feasible') if pred else ('false_infeasible' if truth else 'true_infeasible')
                confusion[key] += 1
            choice = d['decision']['selected_index']
            reasons[d['decision']['reason']] = reasons.get(d['decision']['reason'],0)+1
            detail = {'map_id':mid, 'level':m['level'], 'any_actual_feasible':any(actual),
                      'actual_min_index':actual.index(True) if any(actual) else None,
                      'selected_index':choice, 'selected_k':None, 'correctly_accepted':False}
            if choice is not None:
                accepted += 1
                levels[m['level']] += 1
                row = keyed[mid,GRID[choice]]['episode']
                good = bool(actual[choice])
                correct += good
                failures += not good
                efficiency_ok = bool(any(actual) and choice <= actual.index(True)+1)
                efficiency_failures += not efficiency_ok
                detail.update(selected_k=GRID[choice], correctly_accepted=good, label=row['label'],
                              t_s=row['t_s'], danger_steps=row['danger_steps'],
                              budget_J=row['z0_J'], cost_model_J=row['cost_model_J'],
                              overspend_J=max(row['cost_model_J']-row['z0_J'],0),
                              overspend_fraction=max(row['cost_model_J']/row['z0_J']-1,0),
                              budget_utilization=row['cost_model_J']/row['z0_J'],
                              efficiency_pass=efficiency_ok,
                              excess_grid_steps=choice-actual.index(True) if any(actual) else None)
            details.append(detail)
        checks = {'accepted_safety_time_budget': failures == 0,
                  'nonvacuous_accepted_count': accepted >= 6,
                  'each_level_coverage': all(n >= 2 for n in levels.values()),
                  'feasible_map_recall': feasible_maps > 0 and 5*correct >= 4*feasible_maps,
                  'budget_efficiency': efficiency_failures == 0}
        summaries[method] = {'pass':all(checks.values()), 'checks':checks,
                             'maps':9, 'grid_rows':len(rows), 'accepted':accepted, 'rejected':9-accepted,
                             'actually_feasible_maps':feasible_maps, 'correctly_accepted':correct,
                             'accepted_failures':failures, 'efficiency_failures':efficiency_failures,
                             'feasible_map_recall':correct/feasible_maps if feasible_maps else None,
                             'false_rejections':sum(x['any_actual_feasible'] and x['selected_index'] is None for x in details),
                             'level_accepted':levels, 'decision_reasons':reasons,
                             'raw_W_confusion':confusion, 'details':details}
    return {'pass':all(s['pass'] for s in summaries.values()), 'methods':summaries,
            'independent_decisions':18, 'grid_episodes':90,
            'claim':'finite nominal research screen for the selected pair; no physical or population-risk guarantee'}
