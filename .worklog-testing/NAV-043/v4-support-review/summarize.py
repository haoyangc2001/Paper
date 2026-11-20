"""Summarize the new saved-data report only; no new scientific evaluation."""
from pathlib import Path
import json
import math
import os

ROOT = Path('/home/caohy/repositories/Paper')
r = json.loads((ROOT/'CaMRA/navigation/reports/nav043_v4_support_review_v1.json').read_text())
print('PANE', os.environ['TMUX_PANE'])
for fold in r['folds']:
    print('FOLD', fold['level'], 'command bounds', fold['training_command_min'], fold['training_command_max'])
    print('TEACHER', json.dumps(fold['teacher_RMSE_equal_case_weight']))
print('PROBES: kind, fold, outside axes, normalized nearest, teacher yaw RMSE, free yaw RMSE, mean innovation yaw, mean free error yaw, terminal heading degrees')
for row in r['cases']:
    if row['cohort'] == 'probe':
        s, e = row['command_support'], row['response_error']
        print(row['kind'], row['fold'], s['outside_per_axis'],
              s['normalized_nearest_training_distance_mean'],
              e['teacher_one_step_RMSE'][2], e['free_response_RMSE'][2],
              e['innovation_mean'][2], e['free_error_mean'][2], row['terminal_heading_error_deg'])
        print('Q_ACTUAL_AND_SAVED_FINAL20', row['measured_last_20_q_mean'], row['saved_free_last_20_q_mean'])
for cohort in ('navigation', 'probe'):
    subset = [row for row in r['cases'] if row['cohort'] == cohort]
    n = sum(row['command_support']['held_intervals'] for row in subset)
    outside = sum(row['command_support']['outside_axis_box_count'] for row in subset)
    one = [math.sqrt(sum(row['response_error']['intervals'] * row['response_error']['teacher_one_step_RMSE'][i]**2
                        for row in subset)/n) for i in range(3)]
    free = [math.sqrt(sum(row['response_error']['intervals'] * row['response_error']['free_response_RMSE'][i]**2
                         for row in subset)/n) for i in range(3)]
    print('COHORT', cohort, 'intervals', n, 'outside', outside, 'teacher', one, 'free', free)
print('MAX_CLOSURE', max(row['response_error']['recurrence_closure_max'] for row in r['cases']))
print('MAX_YAW_CLOSURE', max(abs(row['terminal_yaw_accounting_closure_rad']) for row in r['cases']))
