"""Freeze the explicitly authorized fresh r14 attempt with complete bindings."""
import datetime,hashlib,json
from pathlib import Path
from camra_navigation.algorithm.validation import verify_training_bindings
NAV=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 previous=NAV/'config/training_loop_prereg_r14_repair_v1.json'
 assert sha(previous)==Path(str(previous)+'.sha256').read_text().strip()
 p=json.loads(previous.read_bytes());p.update(version='r14.2.0',created_at=datetime.datetime.now().astimezone().isoformat(),status='frozen_before_repaired_attempt_2',task='NAV-064: authorized additional six fresh-init runs after implementation failure; consume reference reach gamma_2 and preserve scalar episode identity before reset.')
 p['authorization']['scope']='User explicitly approved the reviewed proposal: six new fresh-init runs x600 updates, prior six interrupted attempts retained separately; necessary local commits permitted, no remote push, no r15 or extra performance smoke.'
 p['runtime_contract']={'version':'r14-gamma-meta-v1','reach_schedule':'reference_gamma_2_half_horizon','run_prefix':'nav064_r14a2','metadata':'snapshot_before_reset_strict_audit','first100':'included_in_600'}
 p['run_registry']=[{'variant':m,'seed':s,'run_id':f'nav064_r14a2_{m}_s{s}'} for m in p['method']['variants'] for s in p['method']['seeds']]
 p['prior_attempts']={'prereg_sha256':sha(previous),'run_prefix':'nav064_r14','completed_updates':[65,62,64,59,59,59],'reason':'implementation mismatch, not infrastructure failure','retention':'all original logs/chunks/checkpoints retained; not formal evidence for corrected schedule'}
 p['resource']['override_basis']='User explicitly approved one additional six-run r14 repair attempt. Prior368 completed updates plus interrupted windows separately counted; another3600 updates/14709600 samples authorized. No wallclock cap, no r15.'
 p['resource']['r14_attempt_accounting']={'prior_runs':6,'prior_completed_updates':368,'new_runs':6,'new_updates':3600,'new_samples':14709600,'total_attempts_this_round':12}
 p['resource']['note']='Exactly six new runs, 600 updates each; first100 included. Preserve all preceding attempts; no additional performance smoke.'
 p['reporting']='reports/training_round_r14a2.json + reports/nav064_r14a2_closure.json + strict store audits; sr_feasible, energy/budget, safety and all raw/stability checks. Previous rounds only historical multi-factor comparisons.'
 p['stops']['infrastructure']='Preserve logs and report infrastructure fault; no automatic performance retry beyond explicitly authorized attempts.'
 p['round_acceptance']['required_additional_audits']+=['reach gamma_2 actual per-update/chunk consumption','strict pre-reset identity/budget agreement','all final checkpoints and current attempt registry identities']
 paths={'numerical_gate':'reports/gate_r14a2_attempt1.json','store_auditor':'scripts/audit_r14a2.py','development_evaluator':'scripts/eval_r14_repair.py','runner':'scripts/run_round14a2.py','reference_energy_report':p['reference_energy']['report']}
 for name,rel in paths.items():p['binding'][name]={'path':rel,'sha256':sha(NAV/rel)}
 for entry in p['binding'].values():entry['sha256']=sha(NAV/entry['path'])
 p['amendments'].append({'version':p['version'],'date':p['created_at'],'previous_prereg_sha256':sha(previous),'trigger':'User approved the concrete gamma/metadata repair and one additional fresh six-run attempt.','changes':'reach reference gamma_2 schedule consumed; scalar identity snapshot; strict auditor; explicit run registry.','unchanged':'math estimator obs-v5/rcppo-r14-v3, fixedv2, T45, budget sampling, 18x227,600updates, train/dev, last checkpoint, original closing thresholds.'})
 verify_training_bindings(p,NAV)
 target=NAV/'config/training_loop_prereg_r14_repair_v2.json'
 with target.open('x') as f:json.dump(p,f,ensure_ascii=False,indent=2);f.write('\n')
 with Path(str(target)+'.sha256').open('x') as f:f.write(sha(target)+'\n')
 print(target,sha(target),flush=True)
if __name__=='__main__':main()
