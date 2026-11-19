"""Copy the audited small outputs verbatim and summarize; no fit or replay."""
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys

ROOT = Path('/home/caohy/repositories/Paper')
HERE = ROOT/'.worklog-testing/NAV-043/affine-response-v2'
PYTHON = Path('/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python')
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert Path(sys.executable).resolve() == PYTHON.resolve()
report = json.loads((ROOT/'CaMRA/navigation/reports/nav043_affine_response_review_v2.json').read_text())
assert report['status'] == 'SAVED_AFFINE_EXECUTION_AUDIT_PASS'
assert report['actual_axis_solves'] == 9 and report['extra_verification_fits'] == 0
assert report['heldout_rows'] == 18 and len(report['output_index']) == 41
assert report['gate']['diagnostic_pass'] and not report['NAV043_passed'] and not report['G5_passed']
output = Path(report['launch']['output_root'])
finished = json.loads((output/'FINISHED.json').read_text())
destination = HERE/'saved-output'
destination.mkdir(exist_ok=False)
copied = []
for item in report['output_index']:
    source = Path(item['path'])
    data = source.read_bytes()
    assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']
    relative = source.relative_to(output)
    target = destination/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(data)
    assert target.read_bytes() == data
    copied.append({'original': str(source), 'copy': str(target.relative_to(ROOT)),
                   'sha256': item['sha256'], 'bytes': len(data)})
gate = report['gate']
summary = {
    'at': datetime.now().astimezone().isoformat(), 'host': socket.gethostname(),
    'pane': os.environ['TMUX_PANE'], 'python': sys.executable,
    'diagnostic_pass': gate['diagnostic_pass'],
    'new_RMSE_m': gate['new_RMSE_m'], 'old_RMSE_m': gate['old_RMSE_m'],
    'position_RMSE_reduction_pct': 100*(1-gate['new_RMSE_m']/gate['old_RMSE_m']),
    'new_terminal_heading_MAE_deg': math.degrees(gate['new_terminal_heading_MAE_rad']),
    'old_terminal_heading_MAE_deg': math.degrees(gate['old_terminal_heading_MAE_rad']),
    'heading_MAE_reduction_pct': 100*(1-gate['new_terminal_heading_MAE_rad']/gate['old_terminal_heading_MAE_rad']),
    'max_position_residual_m': max(row['metrics']['position_max_m'] for row in report['rows']),
    'levels': gate['levels'], 'actual_axis_solves': 9, 'models': 3, 'heldout_rows': 18,
    'saved_ticks': report['saved_measurement_ticks'], 'extra_verification_fits': 0,
    'probe_cpu_wall_s': finished['cpu_wall_s'],
    'wrapper_wall_s': report['exit']['wrapper_wall_s'],
    'parameter_ranges': {name: [min(s[name] for s in finished['solves']),
                               max(s[name] for s in finished['solves'])]
                         for name in ('a', 'b', 'd', 'condition')},
    'yaw_intercepts_per_fold': [{'heldout_level': s['level'], 'd_rad_s': s['d']}
                                for s in finished['solves'] if s['axis'] == 'yaw_rate'],
    'originals_retained': True, 'copied_files': copied,
    'protected_identities_checked': 295, 'raw_input_hashes_checked': 90,
    'new_business_episodes': 0, 'W_queries': 0, 'network_updates': 0, 'calibration_fits': 0,
    'deployment_model_selected': False, 'ninety_second_evidence': False,
    'NAV043_passed': False, 'G5_passed': False}
with (HERE/'result-summary.json').open('x') as stream:
    json.dump(summary, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({k:v for k,v in summary.items() if k != 'copied_files'}, ensure_ascii=False, indent=2))
print('41 saved output copies verified byte-for-byte; originals retained')
