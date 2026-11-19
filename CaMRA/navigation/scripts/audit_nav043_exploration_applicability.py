"""Describe margin applicability on reviewed saved budget-selected traces.

No inference, simulator, new episode, fitting or acceptance is performed.
The original physical-event review must finish first.
"""
from datetime import datetime
import os
from pathlib import Path
import socket
import sys

from audit_nav043_saved_applicability import (
    NAV, PYTHON, analyze_trial, load, sha, write_new,
)

REVIEW = NAV / 'reports/nav043_exploration_review_v1.json'
OUTPUT = NAV / 'reports/nav043_exploration_applicability_v1.json'
PROTOCOL = NAV / 'config/nav043_exploration_v1.json'


def main():
    assert socket.gethostname().split('.')[0] == 'tiger11'
    assert os.environ.get('TMUX') and sys.executable == PYTHON
    assert not OUTPUT.exists(), 'saved applicability report already exists'
    assert REVIEW.is_file(), 'original saved physical-event review is not finished'
    review = load(REVIEW)
    assert review['status'] == 'SAVED_EVIDENCE_REVIEW_NOT_G5'
    assert not review['NAV_043_pass'] and not review['G5_pass']
    index = review['raw_sources']
    for path, digest in index.items():
        assert sha(path) == digest, path
    protocol = load(PROTOCOL)
    assert sha(PROTOCOL) == review['protocol_sha256']
    assert sha(PROTOCOL) == Path(str(PROTOCOL) + '.sha256').read_text().strip()
    for path, digest in protocol['sealed_hashes'].items():
        assert sha(path) == digest, path
    pair_path = NAV / 'config/nav066_decision_prereg_v2.json'
    pair = load(pair_path)
    protected = {str(NAV / k): v for k, v in pair['implementation_hashes'].items()}
    protected.update(pair['protected_hashes'])
    protected.update({c['checkpoint']: c['checkpoint_sha256'] for c in pair['six_candidates']})
    for path, digest in protected.items():
        assert sha(path) == digest, path
    calibration_path = NAV / 'config/calibration_applicability_prereg_v1.json'
    geometry_path = NAV / 'config/geometry_margin_closedloop_v1.json'
    old_report_path = NAV / 'reports/nav043_saved_applicability_audit_v1.json'
    calibration, geometry = load(calibration_path), load(geometry_path)
    alpha, delta = (calibration['frozen_values'][k] for k in ('alpha', 'delta'))
    assert geometry['validity']['T_range_s'] == [5, 30]
    assert geometry['derivation']['fit_not_validation']
    rows, refusals = [], []
    for episode in review['episodes']:
        if episode['rejected']:
            refusals.append(episode)
            continue
        assert episode['complete_saved_physics_event_replay']
        trial = Path(episode['trial'])
        assert trial.parent.parent == Path(protocol['output_root'])
        analyzed = analyze_trial(trial, 'actual_state_W_selected', episode['level'],
                                 index, alpha, delta, geometry['table_q90_by_prefix_s'])
        assert analyzed['budget_mode'] == 'actual_state_guarded_W'
        assert analyzed['physics_ticks'] == episode['physics_ticks']
        analyzed['selected_k'] = episode['selected_k']
        analyzed['safe_timely_measured_budget_success'] = episode['safe_timely_measured_budget_success']
        analyzed['nominal_closed_loop_t_s'] = episode['nominal_t_s']
        analyzed['nominal_closed_loop_cost_J'] = episode['nominal_cost_J']
        rows.append(analyzed)
    paths = [REVIEW, PROTOCOL, pair_path, calibration_path, geometry_path,
             old_report_path, Path(__file__), NAV / 'scripts/audit_nav043_saved_applicability.py']
    report = {
        'schema': 'camra-nav043-exploratory-applicability-v1',
        'at': datetime.now().astimezone().isoformat(), 'task': 'NAV-043',
        'status': 'DESCRIPTIVE_BUDGET_SELECTED_MARGIN_REVIEW_NOT_G5',
        'source_hashes': {str(p): sha(p) for p in paths},
        'protected_identities_checked': len(protected),
        'frozen_alpha': alpha, 'frozen_delta_m': delta,
        'rows': rows, 'refusals': refusals, 'incomplete': review['incomplete'],
        'original_allocation_ledger': review['allocation_resource_ledger'],
        'new_applications': 0, 'new_physical_episodes': 0, 'new_nominal_episodes': 0,
        'W_queries': 0, 'training_updates': 0, 'calibration_fits': 0,
        'final_data_access': False, 'NAV043_passed': False, 'G5_passed': False,
        'limitations': [
            'Exposed integration maps, at most one execution per configuration; no population coverage claim.',
            'Same-command nominal reference and separately executed nominal closed loop are different quantities.',
            'Old rule table is fitted, domain limited and ends at 30s; numeric comparisons do not extend validity.',
            'Incomplete prefixes and refusals remain absent; no padding or alternative budget.',
            'No minimum feasible budget sweep, margin fitting, frozen-value change or causal attribution.',
        ],
    }
    # Detect concurrent edits to inputs before writing the new report.
    for path, digest in report['source_hashes'].items():
        assert sha(path) == digest, path
    write_new(OUTPUT, report)
    print({'status': report['status'], 'saved_trials': len(rows),
           'refusals': len(refusals), 'NAV043_passed': False, 'G5_passed': False}, flush=True)


if __name__ == '__main__':
    main()
