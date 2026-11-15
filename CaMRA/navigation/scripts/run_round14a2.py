"""Run the one authorized repaired r14 attempt, then audit and evaluate it."""
from __future__ import annotations
import datetime,json,os,socket,subprocess,sys,time
from pathlib import Path

NAV=Path(__file__).resolve().parents[1]
ROOT=Path('/pub/data/caohy/work/camra_navigation')
PREREG='training_loop_prereg_r14_repair_v2.json'
PY='/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'

def main():
 assert os.environ.get('TMUX') and socket.gethostname().split('.')[0]=='tiger11'
 assert sys.executable==PY
 sys.path.insert(0,str(NAV/'scripts'))
 import train_formal as F
 p,pin=F.verify_prereg(PREREG)
 work=ROOT/'nav064_r14a2'
 for r in p['run_registry']:assert not (ROOT/r['run_id']).exists(),r['run_id']
 work.mkdir(exist_ok=False)
 def write(name,value):
  with (work/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
 write('start.json',{'time':datetime.datetime.now().astimezone().isoformat(),'host':socket.getfqdn(),'python':sys.executable,'cwd':str(NAV),'tmux':os.environ['TMUX'],'prereg_sha256':pin,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=NAV,text=True).strip(),'authorized_attempts':p['run_registry']})
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='6',PYTHONPATH=f'{NAV}/scripts:{NAV}/src:{NAV.parent}/calibration/src')
 processes=[];handles=[]
 for r in p['run_registry']:
  handle=(work/(r['run_id']+'.log')).open('x');handles.append(handle)
  cmd=[PY,'-u','scripts/train_formal.py','--run-id',r['run_id'],'--variant',r['variant'],'--seed',str(r['seed']),'--prereg-name',PREREG]
  proc=subprocess.Popen(cmd,cwd=NAV,env=env,stdout=handle,stderr=subprocess.STDOUT)
  processes.append((r,proc));print('START',proc.pid,r['run_id'],flush=True)
 write('pids.json',[dict(r,pid=proc.pid) for r,proc in processes])
 audit100=None;audit_handle=None
 def audit_command(n):return [PY,'-u','scripts/audit_r14a2.py','--updates',str(n),'--out',str(ROOT/f'nav064_r14a2_audit_u{n}')]
 def hold(reason):
  import signal
  for _,proc in processes:
   if proc.poll() is None:os.kill(proc.pid,signal.SIGSTOP)
  write('HELD.json',{'reason':reason,'time':datetime.datetime.now().astimezone().isoformat()})
  raise RuntimeError(reason)
 with (work/'progress.jsonl').open('x') as progress:
  while True:
   counts=[]
   for r,proc in processes:
    path=ROOT/r['run_id']/'train_log.jsonl';count=0
    if path.exists():
     for line in path.read_text().splitlines():
      try:row=json.loads(line)
      except json.JSONDecodeError:continue
      count+=row.get('kind')=='update'
    counts.append(count)
    if proc.poll() is not None and proc.returncode!=0:hold(f'training failure {r["run_id"]}: {proc.returncode}')
   row={'time':datetime.datetime.now().astimezone().isoformat(),'updates':counts}
   progress.write(json.dumps(row)+'\n');progress.flush();print(row,flush=True)
   if min(counts)>=100 and audit100 is None:
    audit_handle=(work/'audit_u100.log').open('x')
    audit100=subprocess.Popen(audit_command(100),cwd=NAV,env=env,stdout=audit_handle,stderr=subprocess.STDOUT)
   if audit100 is not None and audit100.poll() is not None and audit100.returncode!=0:hold('first100 raw-store audit failed')
   if all(proc.poll() is not None for _,proc in processes):break
   time.sleep(30)
 assert counts==[600]*6
 write('training_finished.json',{'runs':[dict(r,exit_code=proc.returncode) for r,proc in processes]})
 assert audit100 is not None and audit100.wait()==0
 with (work/'audit_u600.log').open('x') as log:
  subprocess.run(audit_command(600),cwd=NAV,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 out=NAV/'reports/training_round_r14a2.json'
 with (work/'development_eval.log').open('x') as log:
  subprocess.run([PY,'-u','scripts/eval_r14_repair.py','--prereg',PREREG,'--out',str(out)],cwd=NAV,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 dev=json.loads(out.read_bytes());audits=[json.loads((ROOT/f'nav064_r14a2_audit_u{n}'/'report.json').read_bytes()) for n in (100,600)]
 passed=dev['performance_pass'] and all(a['verdict']=='PASS' for a in audits)
 report={'owner':'NAV-064','prereg_sha256':pin,'verdict':'PASS' if passed else 'FAIL_DEVELOPMENT','performance':dev['summary'],'development_report':str(out),'audits':[str(ROOT/f'nav064_r14a2_audit_u{n}'/'report.json') for n in (100,600)],'all_training_updates':3600,'prior_invalid_attempt_completed_updates':368,'automatic_downstream_execution':False}
 with (NAV/'reports/nav064_r14a2_closure.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
 write('FINISHED.json',report);print(report,flush=True)
 for handle in handles:handle.close()
 if audit_handle:audit_handle.close()
 return 0 if passed else 2

if __name__=='__main__':raise SystemExit(main())
