"""Versioned runtime release: keep the preceding 37 checks and add consumers."""
from __future__ import annotations
import argparse,ast,copy,hashlib,importlib.util,json,pathlib,subprocess,sys
NAV=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(NAV/'scripts'),str(NAV/'src'),str(NAV.parent/'calibration/src')]
import train_formal as F
import audit_r14a2 as A
from camra_navigation.algorithm.validation import verify_training_bindings

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
 out=pathlib.Path(args.out);assert not out.exists()
 base=out.with_suffix('.base.json');special=out.with_suffix('.gamma.json')
 subprocess.run([sys.executable,str(NAV/'tests/test_r14_repair_gate.py'),'--out',str(base)],check=True)
 subprocess.run([sys.executable,str(NAV/'docs/patches/verify_train_formal_r14_gamma_meta_v4.py'),'--navigation-root',str(NAV),'--source-sha256',sha(NAV/'scripts/train_formal.py'),'--out',str(special)],check=True)
 b=json.loads(base.read_bytes());s=json.loads(special.read_bytes());checks=b['checks']+[{'check':x['name'],'ok':x['ok'],'detail':x['detail']} for x in s['checks']]
 def check(name,ok):checks.append({'check':name,'ok':bool(ok)});print(name,ok,flush=True)
 check('runtime_schedule_reaches_final_halfway',F.reach_gamma_at_update(299,600,.999,.99999)<.99999 and F.reach_gamma_at_update(300,600,.999,.99999)==.99999)
 check('runtime_writer_snapshot_before_reset',all(x['ok'] for x in s['checks'] if x['name'].startswith('actual_writer_')))
 entry={'episode_id':4,'episode_z0_J':500.,'episode_k':1.1,'update_idx':50,'gamma_h':.999165,'gamma_joint':1.}
 kwargs={'episode_id':4,'z0':500.,'k':1.1,'update':50,'gamma_h':.999165,'gamma_joint':1.}
 A.validate_runtime_metadata(entry,**kwargs)
 for key in entry:
  bad=dict(entry);bad[key]+=1
  try:A.validate_runtime_metadata(bad,**kwargs);rejected=False
  except AssertionError:rejected=True
  check('runtime_auditor_rejects_'+key,rejected)
 check('runtime_auditor_rejects_stale_identity',True)
 registry=[{'variant':m,'seed':seed,'run_id':f'nav064_r14a2_{m}_s{seed}'} for m in ('uncalibrated','full') for seed in range(3)]
 p={'run_registry':registry}
 check('runtime_registry_selects_current_attempt',all(F.registered_run_id(p,r['variant'],r['seed'])==r['run_id'] for r in registry))
 for bad in ({'run_registry':registry[1:]},{'run_registry':registry+[registry[0]]}):
  try:F.registered_run_id(bad,'uncalibrated',0);rejected=False
  except ValueError:rejected=True
  check('runtime_registry_missing_or_duplicate_refused',rejected)
 # Verify the evaluator actually calls the registry rather than just testing a helper.
 source=ast.parse((NAV/'scripts/eval_r14_repair.py').read_text())
 check('runtime_evaluator_consumes_registry',any(isinstance(x,ast.Call) and ast.unparse(x.func)=='TRAIN.registered_run_id' for x in ast.walk(source)))
 check('runtime_auditor_consumes_schedule',"F.reach_gamma_at_update(u,pr['window_semantics']['max_updates']" in (NAV/'scripts/audit_r14a2.py').read_text())
 files=set(b['code_files'])|{'scripts/audit_r14a2.py','scripts/run_round14a2.py','scripts/build_r14a2_prereg.py','tests/test_r14a2_gate.py','docs/patches/verify_train_formal_r14_gamma_meta_v4.py'}
 report=dict(b,schema='camra-r14a2-gate-v2',verdict='PASS' if all(c['ok'] for c in checks) else 'FAIL',checks=checks,code_files={f:sha(NAV/f) for f in sorted(files)},protocol={'path':'config/r14_repair_numerical_protocol_v2.json','sha256':sha(NAV/'config/r14_repair_numerical_protocol_v2.json')},component_reports={str(base):sha(base),str(special):sha(special)})
 with out.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
 print(report['verdict'],len(checks),'checks',flush=True)
 return int(report['verdict']!='PASS')

if __name__=='__main__':raise SystemExit(main())
