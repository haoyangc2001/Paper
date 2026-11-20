"""Prepare reviewable proposal/readiness patches; never authorize or fit."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
sys.path.insert(0, str(ROOT/'CaMRA/navigation/scripts'))
import nav043_matched_support_v5 as m
import nav043_motion_probe_recovery_guard as guard
import run_nav043_coupled_response as old

m.context()
assert not m.APPROVAL.exists() and not (m.HERE/'launch.lock').exists()
output = '/pub/data/caohy/work/camra_navigation/nav043_matched_support_probe_v5'
assert not Path(output).exists() and not m.REVIEW.exists()
assert (m.HERE/'basic.exit').read_text().strip() == '0'
assert 'Ran 7 tests' in (m.HERE/'basic.log').read_text()
assert (m.HERE/'launch-basic.exit').read_text().strip() == '0'
assert 'Ran 4 tests' in (m.HERE/'launch-basic.log').read_text()
assert (ROOT/'.worklog-testing/NAV-043/v4-support-review/basic.exit').read_text().strip() == '0'
assert (ROOT/'.worklog-testing/NAV-043/v4-support-review/analyze.exit').read_text().strip() == '0'
identities = guard.protected_sources()
assert identities == dict(protected_identities=295, existing_raw_input_hashes=90, bundle_files=29)
original = old.approved()
review = m.base.load(m.base.NAV/'reports/nav043_coupled_response_review_v4.json')
for entry in review['output_index']:
    f = Path(entry['path'])
    assert f.stat().st_size == entry['bytes'] and m.base.sha(f) == entry['sha256']
assert len(review['output_index']) == 83
for name in ('wrapper_pid', 'child_pid'):
    assert not Path('/proc',str(review['launch'][name])).exists()
cases = m.diagonal.load_cases()
checks = []
for level, train, held, probes in m.partitions(cases):
    for kind in m.KINDS:
        for axis in range(3):
            x,y,w = m.design(train,kind,axis)
            singular = m.np.linalg.svd(x,compute_uv=False)
            rank = int(m.np.count_nonzero(singular > singular[0]*1.e-12))
            condition = float(singular[0]/singular[-1])
            assert rank == m.width(kind) and condition <= 1.e8
            checks.append(dict(level=level,structure=kind,axis=axis,rows=len(y),rank=rank,
                               columns=m.width(kind),condition=condition))

# Verify the real unapproved entrypoints reject before source inputs/output markers.
denials = []
env = os.environ.copy()
env['PYTHONPATH'] = str(m.base.NAV/'scripts')
for command in ([m.base.PYTHON,'-B',str(m.base.NAV/'scripts/nav043_matched_support_v5.py')],
                [m.base.PYTHON,'-B',str(m.base.NAV/'scripts/nav043_matched_support_v5.py'),'--audit'],
                [m.base.PYTHON,'-B',str(m.HERE/'preflight.py')]):
    denied = subprocess.run(command,cwd=ROOT,env=env,text=True,capture_output=True)
    assert denied.returncode == 1 and 'separate APPROVED' in denied.stderr
    denials.append(dict(command=command,exit_code=1,stderr=denied.stderr))
assert not m.APPROVAL.exists() and not (m.HERE/'launch.lock').exists() and not Path(output).exists()
for name in ('preflight.json','preflight.log','preflight.exit','launch.json','exit.json','run.log','wrapper.log'):
    assert not (m.HERE/name).exists()

new_sources = [
    'CaMRA/navigation/scripts/nav043_matched_support_v5.py',
    'CaMRA/navigation/scripts/analyze_nav043_v4_support.py',
    'CaMRA/navigation/reports/nav043_v4_support_review_v1.json',
    'CaMRA/navigation/reports/nav043_coupled_response_review_v4.json',
    'CaMRA/navigation/config/nav043_coupled_response_proposal_v4.json',
    'CaMRA/navigation/config/nav043_coupled_response_readiness_v4.json',
    'CaMRA/navigation/scripts/audit_nav043_coupled_response.py',
    'CaMRA/navigation/scripts/run_nav043_coupled_response.py',
    'CaMRA/navigation/scripts/nav043_motion_probe_recovery_guard.py']
source_hashes = dict(original['source_hashes'])
source_hashes.update({path:m.base.sha(ROOT/path) for path in new_sources})
proposal = dict(schema='nav043-matched-support-v5',status='PROPOSED',task_id='NAV-043',
    plan_version='1.5.0',production_enabled=False,approval_required=True,
    approval_path=str(m.APPROVAL.relative_to(ROOT)), readiness_path=str(m.READINESS.relative_to(ROOT)),
    output_root=output, limits=m.LIMITS,
    environment=dict(host='tiger11',tmux_required=True,python=m.base.PYTHON,CUDA_VISIBLE_DEVICES='',
                     OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1),
    question='With identical coverage of all twelve known probes, can either existing structure fit probes while predicting an omitted navigation map?',
    structures=list(m.KINDS),
    data=dict(total_exposed_cases=30,navigation_cases=18,probe_cases=12,
        measurement_ticks=182797,probe_warmup_ticks=2400,
        all_previously_exposed=True,new_independent_validation=False,
        original_probe_recovery_HELD_retained=True,original_v3_v4_failures_retained=True,
        nav_first_fitting_pair_excluded=True,probe_actual_warmup_q_required=True,
        partial_tail_excluded_from_fitting_but_retained_in_prediction=True),
    splits=dict(folds=3,train_each_fold=dict(navigation=12,probes=12),
        held_each_fold=dict(navigation=6,probes=0),
        in_sample_probe_predictions_each_fold=12,
        all_twelve_probes_in_training_each_fold=True,
        navigation_whole_map_held_together=True,all_failures_retained=True),
    fitting=dict(objective='original equal-cohort, equal-case, equal-row one-step weighted least squares',
        feature_versions=[m.diagonal.FEATURE_VERSION,m.coupled.FEATURE_VERSION],
        extra_features=False,parameter_sweep=False,regularization=False,
        diagonal_stability='all memory coefficients in [0,1)',
        coupled_stability='spectral radius of full A < 1',invalid_fit='save and stop; no clipping/retry'),
    interpretation=dict(probe_metrics='training diagnostics only, 36 rows per structure, correlated fold reuse',
        navigation_metrics='18 whole-map omitted but already exposed cases per structure',
        summaries='all cases and per map tick-weighted RMSE, terminal heading, prefixes and saved arrays',
        new_acceptance_gate=False,old_acceptance_thresholds_unchanged=True,
        automatic_model_selection=False,additional_all_data_fit=False,
        no_unique_physical_cause_claim=True,no_90second_or_G5_release=True),
    once_only=dict(durable_exclusive_lock=True,original_output_only=True,no_retry=True,
        expected_output_files=260,preflight_before_commit_before_launch=True,
        audit_reads_saved_outputs_without_extra_fits=True),
    no_test_subagents=True,no_push=True,source_hashes=source_hashes)
proposal_text=json.dumps(proposal,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
proposal_sha=m.base.hashlib.sha256(proposal_text.encode()).hexdigest()
support_files=[str((m.HERE/name).relative_to(ROOT)) for name in
    ('basic.py','basic.sh','basic.log','basic.exit','launch-basic.py','launch-basic.sh',
     'launch-basic.log','launch-basic.exit','preflight.py','preflight.sh','launch.sh','audit.sh','prepare.py')]
support_files += ['CaMRA/navigation/docs/nav043_v4_support_review_20261120.md']
ready=dict(schema='nav043-matched-support-readiness-v5',status='PREPARED_ONLY',
    proposal_sha256=proposal_sha,pane=os.environ['TMUX_PANE'],identities=identities,
    basic_tests=11,coefficient_solves_during_tests=0,known_coefficient_synthetic_pipeline=True,
    rank_checks=checks,missing_approval_denials=denials,source_cases=30,original_output_files_verified=83,
    fitting_or_predictions_on_real_data=0,applications=0,episodes=0,W_queries=0,
    new_independent_validation=False,APPROVED_present=False,
    source_hashes={path:m.base.sha(ROOT/path) for path in support_files})
ready_text=json.dumps(ready,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
ready_sha=m.base.hashlib.sha256(ready_text.encode()).hexdigest()
patch='*** Begin Patch\n'
for path,content in [(m.PROPOSAL,proposal_text),(Path(str(m.PROPOSAL)+'.sha256'),proposal_sha+'\n'),
                     (m.READINESS,ready_text),(Path(str(m.READINESS)+'.sha256'),ready_sha+'\n')]:
    assert not path.exists()
    patch+='*** Add File: '+str(path)+'\n'+''.join('+'+line+'\n' for line in content.splitlines())
patch+='*** End Patch\n'
Path('/tmp/nav043-matched-v5-registration.patch').write_text(patch)
print('PREPARED_ONLY_PATCH',proposal_sha,ready_sha,'rank checks',len(checks),'real fits 0',flush=True)
