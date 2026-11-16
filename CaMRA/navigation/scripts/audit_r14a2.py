"""Read-only audit of registered r14 stores; original manifests stay untouched."""
import argparse, collections, dataclasses, datetime, hashlib, json, math, os, socket, sys
from pathlib import Path
import numpy as np
import torch
NAV=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(NAV/'scripts'))
import train_formal as F
from camra_navigation import observation as O
from camra_navigation.algorithm import storage as S, networks as NW

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p):
 out=[]
 for s in Path(p).read_text().splitlines():
  try: out.append(json.loads(s))
  except json.JSONDecodeError: break
 return out

def checked(ok, msg):
 if not bool(ok): raise AssertionError(msg)
def close(a,b,msg,atol=1e-11,rtol=1e-10):
 checked(np.allclose(a,b,atol=atol,rtol=rtol),f'{msg}: max difference {np.max(np.abs(np.asarray(a)-np.asarray(b)))}')
def stats(values):
 a=np.asarray(values,dtype=float)
 return dict(zip(('min','p05','median','p95','max'),map(float,np.quantile(a,[0,.05,.5,.95,1]))))

def validate_runtime_metadata(entry, *, episode_id, z0, k, update, gamma_h, gamma_joint):
 checked(entry['episode_id']==episode_id,'stale episode identity')
 close(entry['episode_z0_J'],z0,'stale initial budget')
 close(entry['episode_k'],k,'stale budget draw')
 checked(entry['update_idx']==update,'incorrect update identity')
 close(entry['gamma_h'],gamma_h,'incorrect scheduled reach discount')
 close(entry['gamma_joint'],gamma_joint,'incorrect joint discount')

def run_audit(rid,variant,seed,pr,pin,updates,outdir):
 run=Path('/pub/data/caohy/work/camra_navigation')/rid
 logs=lines(run/'train_log.jsonl'); start=logs[0]
 ul=[r for r in logs if r['kind']=='update'][:updates]
 checked(len(ul)==updates,'updates incomplete')
 checked([r['update'] for r in ul]==list(range(updates)),'update sequence')
 checked(start['prereg_sha256']==pin,'run registration')
 checked(start['implementation_versions']==pr['implementation_versions'],'runtime versions')
 checked(start['obs_version']==pr['implementation_versions']['observation']
         and start['obs_version'] in ('obs-v5','obs-v6') and not start['smoke'],
         'observation/formal mode')
 checked(start['runtime']['host'].split('.')[0]=='tiger11' and bool(start['runtime']['tmux']),'runtime host/tmux')
 checked(start['runtime']['python']=='/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python','interpreter')
 maps=json.loads(Path(F.SPLIT).read_bytes())['splits']['train']['maps']
 map_ids=[m['map_id'] for m in maps]
 checked(start['data_access']['split_used']=='train' and start['data_access']['map_ids']==map_ids,'actual maps')
 for p,h in start['data_access']['inputs'].items(): checked(sha(p)==h,f'loaded input {p}')
 numeric_keys=['loss_policy','loss_cost','loss_reach','grad_norm_policy','grad_norm_cost','grad_norm_reach','entropy','clipped_ratio_fraction','adv_mean','adv_std','grad_adv_norm','grad_ent_norm']
 for r in ul:
  checked(r['n_samples']==4086 and r['epochs']==10 and r['minibatch_size']==512,'formal window/update')
  checked(r['grad_steps']+r.get('mask_skipped_minibatches',0)==80,'executed minibatches')
  for k in numeric_keys: checked(math.isfinite(r[k]),f'nonfinite update {r["update"]} {k}')
  checked(0<=r['clipped_ratio_fraction']<=1,'clip range')
 checked(any(r['clipped_ratio_fraction']>0 for r in ul),'PPO never clipped')
 cfg=S.ObjectiveConfig.from_contracts(str(NAV/pr['binding']['value_contract']['path']),r_goal_m=pr['environment']['r_goal_m'])
 cfg=dataclasses.replace(cfg,joint_min_recursion=True,h_semantics='v2_deep_well')
 est=json.loads((NAV/pr['binding']['value_contract']['path']).read_bytes())['estimator']
 for row in ul:
  expected_gamma=F.reach_gamma_at_update(row['update'],pr['window_semantics']['max_updates'],float(est['gamma_h_init']),float(est['gamma_h_final']))
  close(row['gamma_h'],expected_gamma,'update reach schedule')
  close(row['gamma_joint'],1.,'update joint gamma')
 checked(cfg.gamma_joint==cfg.gamma_cost==1,'joint/cost gamma')
 refs=json.loads((NAV/pr['reference_energy']['report']).read_bytes())
 erefs={r['map_id']:float(r['e_ref_J']) for r in refs['maps']}
 alpha=float(start['provenance']['alpha']); slices=O.field_slices()
 counters=collections.Counter(); diffs=collections.defaultdict(float); kinds=collections.Counter(); diagnostics=[]; kl=collections.defaultdict(list)
 replay_updates=[x for x in (50,300,550) if x<updates]
 spot_updates={0,updates-1,min(99,updates-1),min(299,updates-1),*replay_updates}
 replay_models={}; replay_after={}
 nav=json.loads((NAV/pr['binding']['navigation_contract']['path']).read_bytes())
 def model(state):
  ini=pr['action_init_fix']
  net=NW.build(nav,action_bounded=True,initial_forward_speed=ini['initial_forward_speed'],log_std_min=ini['log_std_min'],log_std_init=ini['log_std_init'],steering_gain=pr['steering_prior']['gain'])
  net.load_state_dict(state['model']); net.eval(); return net
 for u in replay_updates:
  ck=run/f'state_u{u}.pt'
  st=torch.load(ck,map_location='cpu',weights_only=False)
  checked(st['update_idx']==u-1 and st['prereg_sha256']==pin,'replay checkpoint identity')
  replay_models[u]=model(st)
  after=run/f'state_u{u+50}.pt'
  # Post-update policy at multiples of 50 supplies an empirical KL diagnostic.
  if after.exists() and u+50<=updates:
   replay_after[u+49]=model(torch.load(after,map_location='cpu',weights_only=False))
 canonical=outdir/f'{rid}_canonical_index.jsonl'
 with canonical.open('x') as idx:
  for env_i,mid in enumerate(map_ids):
   root=run/f'rollout_env{env_i}'/'rollout'
   entries=[e for e in lines(root/'manifest.jsonl') if int(e['global_step_end'])<updates*227]
   expected=0; last_ep=-1; last=None; z0=None; episode_steps=0
   rng=np.random.default_rng([seed,env_i]); ks=[]; ep_rows=[]
   diag_by_id={(r['env'],r['episode_id']):r for r in logs if r['kind']=='episode_diag'}
   for e in entries:
    checked(e['global_step_start']==expected,'store gap/duplicate')
    checked(e['map_id']==mid,'manifest map')
    path=root/e['file']; checked(sha(path)==e['file_sha256'],'file hash')
    with np.load(path) as f: a={k:f[k] for k in f.files}
    checked(S.content_hash(a)==e['content_sha256'],'content hash')
    for raw_name,v in a.items():
     if v.dtype.kind in 'fc': checked(np.isfinite(v).all(),f'nonfinite raw {path.name} {raw_name}')
    n=len(a['global_step']); checked(n==e['n_samples'],'chunk length')
    checked(np.array_equal(a['global_step'],np.arange(expected,expected+n)),'raw step continuity')
    checked(np.unique(a['episode_id']).size==1,'chunk mixes episodes')
    ep=int(a['episode_id'][0]); checked(ep in (last_ep,last_ep+1),'episode gap')
    if ep!=last_ep:
     checked(last is None or last['event']!=0,'episode reset without event')
     k=float(rng.uniform(.6,1.3)); z0=k*erefs[mid]*pr['reference_energy']['budget_k']*(1+alpha)
     close(a['z_j'][0],z0,'budget random stream')
     episode_steps=0; ks.append(k); ep_rows.append({'episode_id':ep,'k':k,'z0_J':z0})
    else:
     close(a['obs'][0],last['next_obs'],'slice observation continuity',atol=1e-6,rtol=1e-5)
     close(a['z_norm'][0],last['z_next'],'slice budget continuity')
     close(a['safety_y'][0],last['y_next'],'slice safety continuity')
    checked(e['global_step_end']//227==e['global_step_start']//227,'chunk spans update')
    u=int(e['global_step_start'])//227
    checked(np.all(a['map_index']==env_i),'raw map index')
    close(a['z_j']/550,a['z_norm'],'budget units')
    close(a['cost_j']/550,a['cost_norm'],'cost units')
    close(a['z_norm']-a['cost_norm'],a['z_norm_next'],'z_next=z-cost')
    close(a['obs'][:,slices['z_ratio']].reshape(-1),np.clip(a['z_norm'],-3,3),'budget observation',atol=1e-6,rtol=1e-5)
    close(a['obs'][:,slices['safety_y']].reshape(-1),a['safety_y'],'graded safety observation',atol=1e-6,rtol=1e-5)
    checked(np.all(a['safety_y_next']>=a['safety_y']),'safety history reversed')
    close(a['action_applied'],np.clip(a['action_raw'],-1,1)*np.array([1,.4,1]),'actual action',atol=1e-6,rtol=1e-5)
    checked(not a['action_invalid'].any(),'invalid action')
    close(a['obj_total_value'],np.maximum(a['value_reach'],a['value_cost']-a['z_norm']),'W baseline')
    close(a['obj_g_train'],np.maximum(a['obj_h'],-a['z_norm']),'zero additive costs')
    counters['negative_budget_samples']+=int(np.sum(a['z_norm']<0))
    counters['saturated_budget_samples']+=int(np.sum(np.abs(a['z_norm'])>3))
    counters['graded_safety_samples']+=int(np.sum((a['safety_y']>0)&(a['safety_y']<1)))
    counters['clipped_actions']+=int(a['action_clipped'].sum())
    counters['samples']+=n; counters['chunks']+=1
    checked(not np.any((a['terminated']|a['truncated'])[:-1]),'interior termination')
    checked(bool(a['terminated'][-1])!=bool(a['truncated'][-1]),'terminal flags')
    kind=e['terminal_kind']; kinds[kind]+=1
    if cfg.estimator_version == 'rcppo-feedback-v4':
     event_name=S.EVENT_NAMES[int(a['event_code'][-1])]
     expected_kind='truncation' if event_name=='none' else S.terminal_kind_for_event(event_name,cfg)
     checked(kind==expected_kind,'finite task ending vs sampling slice')
     checked(bool(a['terminated'][-1])==(kind!='truncation'),'feedback terminal flag')
     checked(F.training_sample_masks(a['safety_y'],pr,cfg)=={},'feedback all-state selection')
    gamma=F.reach_gamma_at_update(u,pr['window_semantics']['max_updates'],float(est['gamma_h_init']),float(est['gamma_h_final']))
    validate_runtime_metadata(e,episode_id=ep,z0=z0,k=k,update=u,gamma_h=gamma,gamma_joint=1.)
    if u in spot_updates:
     cfg_u=dataclasses.replace(cfg,gamma_h=gamma)
     rec=S.compute_objectives(a,cfg_u,terminal_kind=kind)
     for key,val in rec.items():
      if val.dtype.kind in 'biu': checked(np.array_equal(a[key],val),'discrete target '+key)
      else: close(a[key],val,'target rebuild '+key)
     counters['recomputed_chunks']+=1
    if u in replay_models:
     with torch.no_grad():
      x=torch.as_tensor(a['obs'],dtype=torch.float32); net=replay_models[u]
      lp=net.policy(x).log_prob(torch.as_tensor(a['action_raw'],dtype=torch.float32)).numpy()
      vc,vh=net.values(x)
     close(lp,a['log_prob'],'old policy log probability',atol=1e-5,rtol=1e-5)
     close(vc.numpy(),a['value_cost'],'sample cost critic',atol=1e-6,rtol=1e-5)
     close(vh.numpy(),a['value_reach'],'sample reach critic',atol=1e-6,rtol=1e-5)
     diffs['max_log_prob_replay_error']=max(diffs['max_log_prob_replay_error'],float(np.max(np.abs(lp-a['log_prob']))))
     counters['policy_replayed_samples']+=n
    if u in replay_after:
     with torch.no_grad():
      newlp=replay_after[u].policy(torch.as_tensor(a['obs'],dtype=torch.float32)).log_prob(torch.as_tensor(a['action_raw'],dtype=torch.float32)).numpy()
     delta=newlp.astype(float)-a['log_prob'].astype(float)
     checked(np.isfinite(delta).all(),'post-update log ratio')
     kl[u].extend((np.expm1(delta)-delta).tolist())
    event=int(a['event_code'][-1]); episode_steps+=n
    if event:
     d=diag_by_id[(env_i,ep)]
     checked(d['n_steps']==episode_steps and d['map_id']==mid,'episode diagnostic identity')
     close(d['z0_J'],z0,'episode initial budget log',atol=.0051,rtol=0)
     close(d['z_J_end'],a['z_norm_next'][-1]*550,'episode end budget log',atol=.0051,rtol=0)
     checked(S.EVENT_CODES[d['label']]==event,'terminal label')
     if event==3: checked(a['goal_distance_m_next'][-1]<=.5,'arrival event distance')
     counters['completed_episodes']+=1
    mismatch=(ep!=e['episode_id'] or abs(float(e['episode_z0_J'])-z0)>1e-7 or abs(float(e['episode_k'])-k)>1e-12)
    checked(not mismatch,'repaired manifest has stale metadata')
    counters['manifest_metadata_mismatches']+=int(mismatch)
    row={'store':str(root),'file':e['file'],'file_sha256':e['file_sha256'],'global_step_start':e['global_step_start'],'global_step_end':e['global_step_end'],'map_id':mid,'episode_id':ep,'episode_k':k,'episode_z0_J':z0,'terminal_kind':kind,'original_metadata_mismatch':mismatch,'source':'raw episode_id/z_j plus deterministic registered budget RNG; original manifest retained'}
    idx.write(json.dumps(row)+'\n')
    last={'next_obs':a['next_obs'][-1],'z_next':float(a['z_norm_next'][-1]),'y_next':float(a['safety_y_next'][-1]),'event':event}; last_ep=ep; expected+=n
   checked(expected==updates*227,f'incomplete env{env_i} samples {expected}')
   diagnostics.append({'env':env_i,'map_id':mid,'budget_draws':len(ks),'k_range':[min(ks),max(ks)],'k_mean':float(np.mean(ks)),'k_bins':np.histogram(ks,bins=np.linspace(.6,1.3,8))[0].tolist()})
   print(rid,env_i,expected,'checked',flush=True)
 if updates==600:
  ends=[r for r in logs if r['kind']=='run_end']; checked(len(ends)==1 and ends[0]['updates']==600,'run completion')
  st=torch.load(run/'state.pt',map_location='cpu',weights_only=False)
  checked(st['update_idx']==599 and st['prereg_sha256']==pin and st['implementation_versions']==pr['implementation_versions'],'final checkpoint identity')
  checked(all(torch.isfinite(v).all() for v in st['model'].values()),'final model nonfinite')
 checked(counters['samples']==updates*4086,'total sample count')
 return {'run_id':rid,'updates_audited':updates,'checks_pass':True,'counts':dict(counters),'target_kinds':dict(kinds),'numerical_differences':dict(diffs),'stability':{k:stats([r[k] for r in ul]) for k in numeric_keys},'empirical_post_update_kl':{str(u):{'n':len(v),'mean':float(np.mean(v)),**stats(v)} for u,v in kl.items()},'budget_coverage':diagnostics,'canonical_index':str(canonical),'canonical_index_sha256':sha(canonical),'metadata_validation':'Every chunk identity/budget/update/discount independently verified; no reconstructed replacement or sample deletion.'}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--updates',type=int,default=600); ap.add_argument('--out',required=True)
 ap.add_argument('--prereg',default='training_loop_prereg_r14_repair_v2.json'); args=ap.parse_args()
 checked(os.environ.get('TMUX'),'tmux required'); checked(socket.gethostname().split('.')[0]=='tiger11','host')
 torch.set_num_threads(1)
 out=Path(args.out); checked(not out.exists(),'output exists'); out.mkdir(parents=True)
 pr,pin=F.verify_prereg(args.prereg)
 report={'schema':'camra-r14a2-store-audit-v2','task':'NAV-064','time':datetime.datetime.now().astimezone().isoformat(),'prereg_sha256':pin,'script_sha256':sha(__file__),'updates_audited':args.updates,'runs':[],'verdict':'INCOMPLETE'}
 try:
  for variant in pr['method']['variants']:
   for seed in pr['method']['seeds']:
    rid=F.registered_run_id(pr,variant,seed)
    report['runs'].append(run_audit(rid,variant,seed,pr,pin,args.updates,out))
    (out/'progress.json').write_text(json.dumps(report,indent=2))
  report['verdict']='PASS'
 except Exception as exc:
  import traceback
  report['error']=repr(exc); report['traceback']=traceback.format_exc(); report['verdict']='FAIL'; raise
 finally:
  (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(report['verdict'],flush=True)
if __name__=='__main__': main()
