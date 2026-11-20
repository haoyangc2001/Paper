"""Final evidence and append-only plan verification, with no parameter fitting."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
EV = ROOT/'.worklog-testing/NAV-043/coupled-execution-v4'
sys.path.insert(0, str(ROOT/'CaMRA/navigation/scripts'))
import run_nav043_coupled_response as c
import nav043_motion_probe_recovery_guard as guard

c.context()
p = c.approved()
a = c.base.load(c.APPROVAL)
for ak, pk in [('environment', 'environment'), ('data_scope', 'data'), ('once_only', 'once_only')]:
    assert a[ak] == p[pk]
assert a['production_enabled'] is False and a['no_push'] and a['no_test_subagents']
assert c.base.sha(c.PROPOSAL) == 'f7b5ba4a39adc655f4cc06f2d30a7412e76ae749dc0efd8cc6aa92c813f9b7e7'
assert c.base.sha(c.READINESS) == '8dc215b35cd71b04a4d67bdebb05ac48db150cd842097f14cc0dc533720ee4c1'
identities = guard.protected_sources()
assert identities == dict(protected_identities=295, existing_raw_input_hashes=90, bundle_files=29)
for name in ('preflight', 'launcher', 'audit', 'close'):
    assert (EV/(name+'.exit')).read_text().strip() == '0'
assert (c.HERE/'preflight.exit').read_text().strip() == '1'
assert 'missing or changed pinned registration' in (c.HERE/'preflight.log').read_text()
report_path = ROOT/'CaMRA/navigation/reports/nav043_coupled_response_review_v4.json'
report, closed = c.base.load(report_path), c.base.load(EV/'close.json')
assert report['status'] == 'SAVED_COUPLED_RESPONSE_EXECUTION_AUDIT_PASS'
assert report['scientific_diagnostic_pass'] is False
assert closed['status'] == 'RETAINED_ANALYSIS_PASS_DIAGNOSTIC_FAIL'
assert report['actual_axis_solves'] == closed['actual_axis_solves'] == 9
assert report['extra_response_fits'] == closed['extra_parameter_solves'] == 0
assert sum(report['diagnostic_gate']['checks'].values()) == 1
assert closed['identities'] == identities
index = report['output_index']
assert len(index) == 83 and sum(i['bytes'] for i in index) == 5219988
output = Path(p['output_root'])
assert {str(f) for f in output.rglob('*') if f.is_file()} == {i['path'] for i in index}
assert not (output/'HELD.json').exists()
for item in index:
    source = Path(item['path'])
    assert source.stat().st_size == item['bytes'] and c.base.sha(source) == item['sha256']
    if source.suffix == '.json':
        copy = EV/'saved-small-output'/source.relative_to(output)
        assert copy.stat().st_size == item['bytes'] and c.base.sha(copy) == item['sha256']
assert len(list((EV/'saved-small-output').rglob('*.json'))) == 53
for name in ('launch.lock', 'launch.json', 'exit.json', 'run.log'):
    assert (c.HERE/name).read_bytes() == (EV/'saved-launcher'/name).read_bytes()
for name in ('wrapper_pid', 'child_pid'):
    assert not Path('/proc', str(report['launch'][name])).exists()
case_keys = ('candidate', 'affine', 'ideal', 'previous_v3')
primary = {}
for cohort in ('navigation', 'probe'):
    subset = [r for r in report['rows'] if r['cohort'] == cohort]
    primary[cohort] = {k: math.sqrt(sum(r[k]['position_squared_error_sum'] for r in subset)
                        /sum(r[k]['ticks'] for r in subset)) for k in case_keys}
    label = 'navigation' if cohort == 'navigation' else 'controlled'
    assert math.isclose(primary[cohort]['candidate'], report['diagnostic_gate']['groups'][label]['candidate_RMSE_m'])
path = 'doc/plan/20261109_camra_navigation_implementation.json'
source_commit = (EV/'approval-commit.txt').read_text().strip()
old = json.loads(subprocess.check_output(['git', 'show', source_commit+':'+path], cwd=ROOT, text=True))
now = c.base.load(ROOT/path)
assert {k: v for k, v in old.items() if k != 'tasks'} == {k: v for k, v in now.items() if k != 'tasks'}
for x, y in zip(old['tasks'], now['tasks'], strict=True):
    assert {k: v for k, v in x.items() if k != 'work_records'} == {k: v for k, v in y.items() if k != 'work_records'}
    assert y['work_records'][:len(x['work_records'])] == x['work_records']
    if x['id'] != 'NAV-043':
        assert x == y
    else:
        assert len(x['work_records']) == 65 and len(y['work_records']) == 66
states = {t['id']: t['status'] for t in now['tasks']}
assert sum(states.values()) == 55 and len(states) == 67
assert [t['id'] for t in now['tasks'] if not t['status']
        and all(states[d] for d in t['depends_on'])] == ['NAV-043']
validation = subprocess.run([sys.executable, '-B', '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', path, '--mode', 'update'], cwd=ROOT, text=True, capture_output=True)
assert validation.returncode == 0, validation.stdout+validation.stderr
subprocess.run(['git', 'diff', '--check'], cwd=ROOT, check=True)
diff_paths = subprocess.check_output(['git', 'diff', '--name-only', source_commit], cwd=ROOT, text=True).splitlines()
assert not any(f.startswith(('manuscripts/', 'RC-PPO/', 'HiRA/', 'CaMRA/locomotion/', 'CaMRA/calibration/')) or f.endswith(('.tex', '.bib')) for f in diff_paths)
c.base.write_new(EV/'final-verify.json', dict(status='FINAL_SAVED_EVIDENCE_AND_APPEND_ONLY_PLAN_PASS',
    pane=os.environ['TMUX_PANE'], approval_commit=source_commit, identities=identities,
    real_axis_solves=9, extra_solves=0, files=83, bytes=5219988, git_small_copies=53,
    external_npz=30, primary_tick_weighted_RMSE_m=primary,
    close_cohort_RMSE_is_supplementary_equal_case_weight=True,
    original_preflight_pin_format_failure_preserved=True,
    original_65_records_preserved=True, current_NAV043_records=66,
    all_task_definitions_and_statuses_preserved=True, plan_true=55, NAV043=False,
    execution_audit_pass=True, diagnostic_pass=False, G5_passed=False,
    validation=validation.stdout, source_pins_rechecked=True,
    output_index_and_small_copies_verified=True,
    source_hashes={str(f.relative_to(ROOT)):c.base.sha(f) for f in
        (EV/'final-verify.py', EV/'final-verify.sh', report_path,
         ROOT/'CaMRA/navigation/docs/nav043_coupled_response_result_20261120.md')}))
print('FINAL_SAVED_EVIDENCE_AND_APPEND_ONLY_PLAN_PASS; no new fit or execution', flush=True)
