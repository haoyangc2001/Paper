"""Seal the concrete recovery proposal and review, without running the pilot."""
import ast
from datetime import datetime
import json
from pathlib import Path
import subprocess

from nav067_sources import NAV, ROOT, OUTPUT, read_manifest, sha, write_new
from nav067_runtime import bind_local_locomotion

out=Path(__file__).resolve().parent
manifest, _, _ = read_manifest()
probe=json.loads((out/'probe-attempt1/RESULT.json').read_bytes())
assert probe['status']=='PASS_IMPORT_AND_CONFIG_ONLY'
assert probe['new_navigation_episodes']==probe['new_physics_steps']==0
assert json.loads((out/'probe-exit.json').read_bytes())['returncode']==0
assert '29 passed' in (out/'basic-tests.log').read_text()
origin=bind_local_locomotion(NAV)
assert origin['origin'].endswith('/locomotion/source/unitree_rl_lab/unitree_rl_lab/__init__.py')
paths=('scripts/nav067_runtime.py','scripts/evaluate_navigation_v3.py')
for rel in paths:
    ast.parse((NAV/rel).read_text(),filename=rel)
evidence=[OUTPUT/'manifest.json',OUTPUT/'L1_empty/STARTED.json',
          OUTPUT/'exit-L1_empty.json',out/'preflight.json',out/'probe-attempt1/RESULT.json']
proposal={
    'schema':'camra-nav067-startup-recovery-v2','status':'PROPOSED',
    'created_at':datetime.now().astimezone().isoformat(),'task':'NAV-067',
    'original_manifest_sha256':sha(OUTPUT/'manifest.json'),
    'change':'Bind the existing local locomotion source and persist exceptions before Kit cleanup; reuse the sealed physical simulator loop without modifying it.',
    'authorization_needed':'One explicit L1 startup-recovery exception to the original no-automatic-restart policy; four original unstarted L2/L3 slots remain available.',
    'limits':{'total_navigation_episodes':6,'horizon_s':45.,'warmup_s':1.,
              'training_updates':0,'W_queries':0,'automatic_retries':0},
    'slots':[{'level':'L1_empty','relative_output':'L1_empty/attempt2','methods':['uncalibrated','full'],
              'mode':'one explicit recovery of the two previously claimed slots'},
             {'level':'L2_single_detour','relative_output':'L2_single_detour','methods':['uncalibrated','full'],
              'mode':'two original unstarted slots'},
             {'level':'L3_multi_path','relative_output':'L3_multi_path','methods':['uncalibrated','full'],
              'mode':'two original unstarted slots'}],
    'original_attempt_policy':'Keep original L1 STARTED, logs, exit and quota claim unchanged. Never automatically retry any amended attempt or change its output path.',
    'source_hashes':{rel:sha(NAV/rel) for rel in paths},
    'evidence_hashes':{str(p):sha(p) for p in evidence},
    'unchanged':['original manifest and its four sealed sources','selected actors/critics',
                 'fixed v2 and calibration','45s/obs-v7 objective','map/budget identities',
                 '200/50/10Hz timing','joint mechanical work definition','old scientific FAIL'],
    'not_authorized':['extra repetitions','controlled physical counterexamples',
                      'shared framework/dependency edits','NAV-043 or final_*'],
    'failure_policy':'On any entry/runtime/trace error, persist ERROR and stop; no automatic retry or replacement. Performance failures remain in reports.'}
write_new(NAV/'config/nav067_startup_recovery_proposal_v2.json',proposal)
review={
    'task':'NAV-067','status':'PREPARED_RECOVERY_NOT_EXECUTED',
    'timestamp':datetime.now().astimezone().isoformat(),
    'basic_tests':29,'protected_hashes_checked':manifest['binding']['protected_hashes_checked'],
    'verified_defect':'Original launch PYTHONPATH cannot resolve unitree_rl_lab, while the repository source exists. The sealed simulator imports that module before creating the gym environment.',
    'original_exception_limit':'Original error was not persisted before finally: app.close(); no exact original exception traceback is available. Source/path evidence proves an import blocker, not every possible runtime issue.',
    'zero_physics_probe':{'status':probe['status'],'RESULT':str(out/'probe-attempt1/RESULT.json'),
                          'result_sha256':sha(out/'probe-attempt1/RESULT.json')},
    'plugin_error_limit':'The material plugin error still appears in the successful import/config probe. It was not repaired or demonstrated harmless for actual physics.',
    'local_source':origin,'proposal_path':'config/nav067_startup_recovery_proposal_v2.json',
    'proposal_sha256':sha(NAV/'config/nav067_startup_recovery_proposal_v2.json'),
    'production_entry':'scripts/evaluate_navigation_v3.py --level <original-level> --device cuda:0',
    'original_manifest_sha256':sha(OUTPUT/'manifest.json'),
    'new_navigation_episodes':0,'new_physics_steps':0,'new_training_updates':0,'new_W_queries':0,
    'NAV_067_status':False,'remaining':['Explicit one-time startup recovery decision.',
        'Actual six-pilot clean/event/command/reset/energy evidence.',
        'Separately allocated controlled cases and repeatability; no expansion in this proposal.']}
write_new(NAV/'reports/nav067_startup_recovery_preparation_v2.json',review)
print(json.dumps(review,ensure_ascii=False,indent=2))
