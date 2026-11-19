"""Prepare a reviewable amendment; never approve or execute it."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

from nav067_sources import NAV, ROOT, OUTPUT, read_manifest, sha, write_new

OUT = Path(__file__).resolve().parent
verification = json.loads((OUT/'verification.json').read_bytes())
assert verification['status'] == 'PASS_STATIC_AND_BASIC_ONLY'
assert '37 passed' in verification['tests_output']
manifest, _, _ = read_manifest()
refusal_command = [sys.executable,'-B',str(NAV/'scripts/evaluate_navigation_v4.py'),
                   '--level','L1_empty','--device','cuda:0']
refusal = subprocess.run(refusal_command,cwd=ROOT,stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,text=True)
with (OUT/'production-unapproved-refusal.log').open('x') as stream:
    stream.write(refusal.stdout)
assert refusal.returncode != 0 and 'explicit startup recovery amendment is absent' in refusal.stdout
old_path = NAV/'config/nav067_startup_recovery_proposal_v2.json'
old = json.loads(old_path.read_bytes())
proposal = dict(old)
proposal.update(
    schema='camra-nav067-startup-recovery-v3',
    created_at=datetime.now().astimezone().isoformat(),
    change='Bind local locomotion and persist pre-cleanup failures; independently copy the sealed simulator with RigidBodyView.get_transforms and initialize the missing direct-loop reset buffer.',
    supersedes_for_execution={'path':str(old_path.relative_to(NAV)),'sha256':sha(old_path),
        'old_proposal_preserved':True,'old_PROPOSED_was_not_approval':True},
    production_entry='scripts/evaluate_navigation_v4.py',
    source_hashes={rel:sha(NAV/rel) for rel in (
        'scripts/nav067_runtime.py','scripts/evaluate_navigation_v4.py',
        'scripts/nav067_simulation_v3.py','src/camra_navigation/envs/clean_v3.py')})
proposal['evidence_hashes'] = dict(old['evidence_hashes'])
proposal['evidence_hashes'].update({str(OUT/'verification.json'):sha(OUT/'verification.json'),
                                  str(old_path):sha(old_path)})
assert proposal['status'] == 'PROPOSED'
assert proposal['limits'] == old['limits']
assert proposal['slots'] == old['slots']
assert proposal['original_manifest_sha256'] == sha(OUTPUT/'manifest.json')
assert not (NAV/'config/nav067_startup_recovery_v3.json').exists()
new_path = NAV/'config/nav067_startup_recovery_proposal_v3.json'
write_new(new_path,proposal)
report = {
    'task':'NAV-067','status':'PREPARED_API_REPAIR_RECOVERY_NOT_EXECUTED',
    'timestamp':datetime.now().astimezone().isoformat(),
    'basic_tests':37,'protected_hashes_checked':295,
    'verified_defects':[
        'Sealed collider read invokes articulation-only get_root_transforms on a RigidBodyView; installed rigid-object data reads get_transforms.',
        'Sealed direct reset uses reset_buf before any gym.step; the installed ManagerBasedRLEnv only assigns it inside step.'],
    'installed_api_evidence':verification['installed_api_source_hashes'],
    'simulation_changes':verification['changed_simulator_sites'],
    'new_adapter':'src/camra_navigation/envs/clean_v3.py',
    'production_entry':'scripts/evaluate_navigation_v4.py',
    'verification_path':str(OUT/'verification.json'),'verification_sha256':sha(OUT/'verification.json'),
    'production_unapproved_refusal':{'command':refusal_command,'exit_code':refusal.returncode,
        'log':str(OUT/'production-unapproved-refusal.log'),'before_output_or_app':True},
    'proposal_path':str(new_path.relative_to(NAV)),'proposal_sha256':sha(new_path),
    'proposal_v2_preserved_sha256':sha(old_path),
    'original_four_sources_manifest_models_and_calibration_unchanged':True,
    'new_apps':0,'new_physics_steps':0,'new_navigation_episodes':0,
    'new_training_updates':0,'new_W_queries':0,
    'NAV_067_status':False,
    'limits':['Installed-source/static and basic evidence only; no actual physics acceptance.',
        'No exact original startup traceback; do not infer the first attempt reached either defect.',
        'Latest v3 proposal requires the same one-time L1 recovery decision; no extra episodes/repetitions.',
        'Controlled actual cases and repeatability still need separate allocation.']}
write_new(NAV/'reports/nav067_api_repair_preparation_v3.json',report)
print(json.dumps(report,ensure_ascii=False,indent=2))
