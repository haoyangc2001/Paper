"""Render the finite PROPOSED registration, no fitting or approval."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT/'CaMRA/navigation'
sys.path.insert(0, str(NAV/'scripts'))
import nav043_supported_response as c

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

affine = json.loads((NAV/'config/nav043_affine_response_proposal_v2.json').read_text())
sources = ['CaMRA/navigation/scripts/nav043_supported_response.py',
           'CaMRA/navigation/scripts/launch_nav043_supported_response_once.py',
           'CaMRA/navigation/scripts/nav043_motion_response.py',
           'CaMRA/navigation/config/nav043_motion_response_proposal_v1.json',
           'CaMRA/navigation/config/motion_state_candidate_v1.json',
           'CaMRA/navigation/reports/nav043_motion_probe_recovery_held_review_v2.json',
           '.worklog-testing/NAV-043/motion-recovery-v2/held-raw-output-index.json',
           '.worklog-testing/NAV-043/motion-candidate-v3/support.json']
sources += [str(Path(affine['output_root'])/level/'model.json') for level in c.base.LEVELS]
value = dict(schema='nav043-supported-response-v3', status='PROPOSED',
    task_id='NAV-043', plan_version='1.5.0', production_enabled=False,
    approval_required=True, approval_path=str(c.APPROVAL.relative_to(ROOT)),
    readiness_path=str(c.READINESS.relative_to(ROOT)),
    output_root='/pub/data/caohy/work/camra_navigation/nav043_supported_response_probe_v3',
    affine_output_root=affine['output_root'], limits=c.LIMITS,
    environment=dict(host='tiger11', tmux_required=True, python=c.base.PYTHON,
                     CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1),
    model=dict(version=c.FEATURE_VERSION, state='completed 0.1s SE2 q, not instantaneous body velocity',
               equation='q_next_i = a_i*q_previous_i + C_i @ phi(u)',
               features=['u_x', 'u_y', 'u_yaw',
                         'sign(u_x)*max(abs(u_x)-0.1,0)',
                         'min(max(abs(u/[1,0.4,1]))/0.2,1)'],
               parameters_per_axis=6, zero_input='phi(0)=0, so q=0,u=0 stays zero',
               memory='diagonal a, require 0<=a<1 after one least squares solve',
               knot_status='prospective engineering choices based on exposed observations; not measured deadzone boundaries',
               action_domain='original full [-1,1] x [-0.4,0.4] x [-1,1], no action clipping/filtering or compensation',
               no_global_constant_intercept=True),
    data=dict(old_navigation_cases=18, controlled_cases=12, old_navigation_measurement_ticks=158797,
              probe_measurement_ticks=24000, probe_warmup_ticks=2400,
              role='previously exposed exploratory identification only, not independent unknown/final data',
              original_recovery_protocol='HELD remains HELD; only separately verified complete saved cases reused',
              include_failed_navigation=True, retries_independent=False,
              sample_unit='whole trace/case; correlated ticks are not independent repetitions',
              navigation_initial_q='unknown pre-window q: exclude first fitting pair; zero only for diagnostic free prediction',
              probe_initial_q='actual warmup 180/200 endpoints; include first measurement transition',
              partial_interval='exclude incomplete fitting pairs, retain all physics ticks in free prediction'),
    folds=dict(count=3, navigation='leave one whole map level out, all six repeated/method cases together',
               probes=[['M01','M04','M07','M10'], ['M02','M05','M08','M11'], ['M03','M06','M09','M12']],
               train_each=dict(navigation=12, controlled=8), heldout_each=dict(navigation=6, controlled=4),
               objective='0.5 navigation cohort + 0.5 controlled cohort; equal traces within each cohort, equal rows within trace',
               solves='exactly one ordinary weighted least squares per fold/axis, rcond=1e-12; at most nine',
               no_hyperparameter_scan=True, no_additional_all_data_fit=True, no_deployment_selection=True),
    diagnostics=dict(free_prediction='align initial pose/q once, then only fixed model and saved actual commands; no truth correction',
                     baselines='same-fold previously saved affine model and ideal command model; no baseline refitting',
                     saved='axis solutions/singular values, train/heldout identities, 30 per-case metrics, predictions and time prefixes',
                     fit_stop='rank<6, condition>1e8, nonfinite or a outside [0,1): save HELD and stop, no clipping/retry',
                     candidate_checks=[
                         'navigation aggregate and each of three maps position RMSE no worse than old affine',
                         'controlled position RMSE strictly lower than both old affine and ideal command',
                         'controlled terminal heading MAE no worse than old affine',
                         'M01 and M02 each position RMSE no worse than ideal command'],
                     numeric_tolerance=1.e-10,
                     pass_meaning='screening diagnostic only; not new alpha/delta, 90s applicability, navigation success, NAV043 or G5'),
    support=dict(path='.worklog-testing/NAV-043/motion-candidate-v3/support.json',
                 status='rank 6/6 on all nine weighted design matrices; condition about 51.1 to 88.3, no coefficient solves',
                 limitation='full column rank does not prove q Markov sufficiency or full action/state domain coverage'),
    protected=dict(controller='camra_go2_velocity_it2999_v2', identities=295,
                   old_raw_inputs=90, bundle_files=29, keep_old_failures=True,
                   production_horizon_s=45, candidate_horizon_s=90, final_access=False,
                   no_changes=['v2 package','power model','footprint','alpha','delta','old locks','RC-PPO objectives','old actor/critics']),
    once_only=dict(launcher='CaMRA/navigation/scripts/launch_nav043_supported_response_once.py',
                   persistent_lock='.worklog-testing/NAV-043/motion-candidate-v3/launch.lock',
                   refusal='missing approval/readiness, changed pins, existing output or lock; before any new data fitting/marker',
                   no_reuse_of_previous_allocations=True, no_automatic_extension=True),
    next_dependencies=['main-agent saved-output verification with no new solve',
                       'review diagnostic result and motion-state support before model selection or production integration',
                       'executable reference command program and frozen-power J evaluation, not L/90 or time doubling',
                       'separate new 90-second residual/physical scope, calibration review and affected learning/deployment gates'],
    source_hashes={p:sha(ROOT/p) for p in sources})
path = c.PROPOSAL
assert not path.exists() and not c.APPROVAL.exists()
body = json.dumps(value, ensure_ascii=False, indent=2)+'\n'
pin = hashlib.sha256(body.encode()).hexdigest()+'\n'
print('*** Begin Patch')
for target, text in [(path,body),(Path(str(path)+'.sha256'),pin)]:
    print('*** Add File: '+str(target.relative_to(ROOT)))
    for line in text.splitlines(): print('+'+line)
print('*** End Patch')
