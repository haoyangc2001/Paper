"""Independent scalar oracles and production-consumer release gate for NAV-064.

Outputs are exclusively new versioned reports; historical FAILs are never edited.
"""
from __future__ import annotations
import argparse, dataclasses, hashlib, importlib.util, json, math, pathlib, sys, tempfile
from types import SimpleNamespace
import numpy as np
import torch
NAV=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(NAV/'src'),str(NAV.parent/'calibration/src')]
from camra_navigation import observation as O
from camra_navigation.algorithm import storage as S, targets as T, ppo as P
from camra_navigation.algorithm.validation import verify_training_bindings
CHECKS=[]
def check(name,ok,detail=None):
 CHECKS.append({'check':name,'ok':bool(ok),'detail':detail})
 print(('PASS ' if ok else 'FAIL ')+name,flush=True)
def equal(a,b):return np.allclose(a,b,atol=1e-12,rtol=1e-10)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def oracle_min(h,v,d,gamma,lam,force=False):
 # Enumerate each n-step Bellman target from scratch; no scan or rolling rows.
 out=[];N=len(h)-1
 for t in range(N):
  stops=[k for k in range(t,N) if d[k]]
  if stops and stops[0]==t:out.append(h[t]);continue
  stop=stops[0] if stops else N
  candidates=[];weights=[]
  for end in range(t+1,stop+1):
   val=v[end]
   for s in range(end-1,t-1,-1):
    cont=(1-gamma)*h[s]+gamma*val
    val=cont if force else min(h[s],cont)
   candidates.append(val)
   weights.append((1-lam)*lam**(end-t-1) if stops else lam**(end-t-1))
  if stops:
   val=h[stop]
   for s in range(stop-1,t-1,-1):
    cont=(1-gamma)*h[s]+gamma*val
    val=cont if force else min(h[s],cont)
   candidates.append(val);weights.append(lam**(stop-t))
  out.append(sum(x*w for x,w in zip(candidates,weights))/sum(weights))
 return np.array(out)
def oracle_cost(c,v,d,gamma,lam):
 out=[];N=len(c)
 for t in range(N):
  stop=next((s for s in range(t,N) if d[s]),N)
  if stop==t:out.append(0.);continue
  vals=[];weights=[];K=stop-t
  for n in range(1,K+1):
   total=sum(gamma**k*c[t+k] for k in range(n))
   tail=0. if t+n==stop and stop<N else v[t+n]
   vals.append(total+gamma**n*tail)
   weights.append(lam**(n-1) if n==K else (1-lam)*lam**(n-1))
  out.append(sum(x*w for x,w in zip(vals,weights)))
 return np.array(out)
def episode(dist,y=None,z0=1.,kind='success',bh=.3,bc=.2):
 N=len(dist)-1;c=np.full(N,.01);z=z0-np.arange(N+1)*.01
 ys=np.zeros(N+1) if y is None else np.array(y)
 return {'cost_norm':c,'z_norm':z[:-1],'z_ratio':z[:-1],
  'z_norm_next':z[1:],'z_ratio_next':z[1:],'value_cost':np.full(N,.15),
  'value_reach':np.full(N,.4),'safety_y':ys[:-1],'safety_y_next':ys[1:],
  'goal_distance_m':np.array(dist[:-1]),'goal_distance_m_next':np.array(dist[1:]),
  'terminated':np.array([False]*(N-1)+[kind!='truncation']),
  'truncated':np.array([False]*(N-1)+[kind=='truncation']),
  'bootstrap_value_cost':np.array([0.]*(N-1)+[bc]),
  'bootstrap_value_reach':np.array([0.]*(N-1)+[bh])}
def compute(a,kind='success',**kw):return S.compute_objectives(a,dataclasses.replace(CFG,**kw),terminal_kind=kind)
def startup_tests():
 with tempfile.TemporaryDirectory() as td:
  root=pathlib.Path(td)
  files={'nav.json':{'observation_version':'obs-v5','observation_spec':{'version':'obs-v5'}},'value.json':{'estimator':{'version':'rcppo-r14-v3','gamma_cost':1.,'gamma_joint':1.}},'protocol.json':{}}
  for n,v in files.items():(root/n).write_text(json.dumps(v))
  (root/'source.py').write_text('version = 1\n')
  versions={'observation':'obs-v5','estimator':'rcppo-r14-v3'}
  gate={'verdict':'PASS','implementation_versions':versions,'checks':[{'ok':True}], 'code_files':{'source.py':digest(root/'source.py')},'protocol':{'path':'protocol.json','sha256':digest(root/'protocol.json')}}
  (root/'gate.json').write_text(json.dumps(gate))
  binding={k:{'path':v,'sha256':digest(root/v)} for k,v in {'navigation_contract':'nav.json','value_contract':'value.json','numerical_gate':'gate.json'}.items()}
  prereg={'round':14,'binding':binding,'implementation_versions':versions,'objective_signal_fix':{'h_semantics':'v2_deep_well','joint_min_recursion':True,'joint_cost_weight':0.,'joint_cost_base_weight':0.}}
  verify_training_bindings(prereg,root);check('startup_matching_fixture_accepted',True)
  for variant in ('source','gate_fail','observation','missing','sha','estimator_flags','discount'):
   p=json.loads(json.dumps(prereg))
   if variant=='source':(root/'source.py').write_text('version = 2\n')
   if variant=='gate_fail':
    bad=dict(gate,verdict='FAIL');(root/'gate.json').write_text(json.dumps(bad));p['binding']['numerical_gate']['sha256']=digest(root/'gate.json')
   if variant=='observation':p['implementation_versions']['observation']='obs-v4'
   if variant=='missing':del p['binding']['value_contract']
   if variant=='sha':p['binding']['navigation_contract']['sha256']='0'*64
   if variant=='estimator_flags':p['objective_signal_fix']['joint_min_recursion']=False
   if variant=='discount':
    val={'estimator':{'version':'rcppo-r14-v3','gamma_cost':1.,'gamma_joint':.999}}
    (root/'value.json').write_text(json.dumps(val));p['binding']['value_contract']['sha256']=digest(root/'value.json')
   try:verify_training_bindings(p,root);rejected=False
   except (ValueError,KeyError):rejected=True
   check('startup_reject_'+variant,rejected)
   (root/'source.py').write_text('version = 1\n');(root/'gate.json').write_text(json.dumps(gate))
   (root/'value.json').write_text(json.dumps(files['value.json']))

def main():
 global CFG
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
 CFG=dataclasses.replace(S.ObjectiveConfig.from_contracts(str(NAV/'config/value_objective_contract_r14_repair_v1.json'),r_goal_m=.5),h_semantics='v2_deep_well',joint_min_recursion=True)
 for N in (1,2,7):
  for lam in (0.,.5,.95):
   h=np.ones(N+1);v=h.copy();d=np.zeros(N,bool)
   check(f'constant_N{N}_lambda{lam}',equal(T.min_gae_return(h,v,d,1.,lam),np.ones(N)))
 check('distinct_critic_two_step_manual',equal(T.min_gae_return(np.array([.8,.6,.4]),np.array([.7,.5,.2]),np.zeros(2),1.,.5),[.4,.2]))
 rng=np.random.default_rng(6414);cases=0;ok=True
 for N in (1,2,5):
  for stop in range(N+1):
   for lam in (0.,.5,.95):
    h=rng.uniform(-1,2,N+1);v=rng.uniform(-1,2,N+1);c=rng.uniform(0,.2,N);d=np.zeros(N,bool)
    if stop<N:d[stop]=True
    for gamma in (.8,1.):
     ok &= equal(T.min_gae_return(h,v,d,gamma,lam),oracle_min(h,v,d,gamma,lam))
     ok &= equal(T.cost_gae_return(c,v,d,gamma,lam),oracle_cost(c,v,d,gamma,lam))
     cases+=1
 check('independent_scalar_oracles_all_stops',ok,{'cases':cases})
 ref=T._ref_gae_reach4(1.,.95,np.ones((3,1)),np.ones((3,1)),np.zeros(2))[1][:,0]
 check('reference_common_constant_scan_horizon',equal(ref,[1.,1.]))
 for N in (1,4):
  a=episode(list(np.linspace(2.,.6,N))+[.2]);o=compute(a)
  check(f'first_arrival_after_action_N{N}',o['obj_target_total'][-1]<0 and equal(o['obj_h_terminal'],np.full(N,-3.)))
  b={k:v.copy() for k,v in a.items()};b['bootstrap_value_cost'][-1]=999;b['bootstrap_value_reach'][-1]=-999
  other=compute(b);check(f'true_success_ignores_learned_terminal_N{N}',all(equal(o[k],other[k]) for k in ('obj_target_cost','obj_target_reach','obj_target_total')))
 for name,y,z in [('light_violation',[0.,.2,.2],1.),('deep_violation',[0.,1.,1.],1.),('negative_budget',None,-.1)]:
  o=compute(episode([1.2,.7,.2],y,z));check(name+'_not_feasible',not o['obj_feasible'].any())
 a=episode([4.,3.,2.],kind='truncation',bh=-.2,bc=.2);o=compute(a,'truncation')
 b={k:v.copy() for k,v in a.items()};b['bootstrap_value_reach'][-1]=-.7
 oo=compute(b,'truncation');check('slice_actual_reach_bootstrap_consumed',not equal(o['obj_target_reach'],oo['obj_target_reach']) and not equal(o['obj_target_total'],oo['obj_target_total']) and equal(o['obj_target_cost'],oo['obj_target_cost']))
 b={k:v.copy() for k,v in a.items()};b['bootstrap_value_cost'][-1]=.6;oo=compute(b,'truncation');check('slice_actual_cost_bootstrap_consumed',not equal(o['obj_target_cost'],oo['obj_target_cost']))
 a=episode([4.,3.,2.],kind='failure');o=compute(a,'failure');check('failure_shared_stop_and_penalty',equal(o['obj_stop_index'],[2,2]) and equal(o['obj_target_cost'][-1],1.) and np.all(o['obj_target_total']>0))
 # gamma_h may change reach, but gamma_joint must be passed independently.
 a=episode([2.,1.,.2]);a['value_reach'][:]=0.;a['value_cost'][:]=0.
 g1=compute(a,gamma_h=.8);g2=compute(a,gamma_h=.99)
 check('production_gamma_joint_is_independent',equal(g1['obj_target_total'],g2['obj_target_total']) and not equal(g1['obj_target_reach'],g2['obj_target_reach']))
 g3=compute(a,gamma_joint=.8)
 check('production_gamma_joint_changes_target',not equal(g3['obj_target_total'],compute(a)['obj_target_total']))
 # Exercise the actual close_segment production consumer with these raw fields.
 spec=importlib.util.spec_from_file_location('train_formal_repair_gate',NAV/'scripts/train_formal.py');train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train);train.obj_cfg=CFG
 result,_=train.close_segment(SimpleNamespace(arrays=lambda:{k:v.copy() for k,v in a.items()}),'success',999.,-999.,env_done=True,label='goal_reached')
 check('production_close_segment_terminal_consumer',result['obj_target_total'][-1]<0)
 # Graded history must be visible; old encodings remain reconstructible.
 ego=O.EgoState(0.,0.,0.,0.,0.,0.);extent=O.MapExtent(-15.,15.,-15.,15.)
 task=O.TaskState(goal_x=1.,goal_y=0.,safety_y=.2,z_J=550.,time_remaining_s=20.)
 field=O.field_slices()['safety_y'];new=np.array(O.encode_observation(ego,[],extent,task,version='obs-v5'))[field];old=np.array(O.encode_observation(ego,[],extent,task,version='obs-v4'))[field]
 check('graded_observation_and_legacy_replay',equal(new,[.2]) and equal(old,[0.]))
 logits=torch.nn.Parameter(torch.zeros(2,dtype=torch.float64));opt=torch.optim.SGD([logits],lr=.1);old_log=torch.log_softmax(logits.detach(),0).clone();adv=torch.tensor([-1.,1.],dtype=torch.float64)
 for _ in range(3):
  opt.zero_grad();lp=torch.log_softmax(logits,0);loss,_=P.policy_loss(lp,old_log,adv,torch.zeros(2,dtype=torch.float64),dataclasses.replace(P.PPOConfig(),ent_coef=0.));loss.backward();opt.step()
 check('actual_minimizing_ppo_probability_update',float(torch.softmax(logits,0)[0].detach())>.5 and equal(old_log.numpy(),[-math.log(2)]*2))
 _,stats=P.policy_loss(torch.log(torch.tensor([2.,.5])),torch.zeros(2),torch.tensor([-1.,1.]),torch.zeros(2),P.PPOConfig())
 # Production PPO uses float32; retain the separately registered float32
 # tolerance instead of applying the scalar float64 oracle tolerance here.
 check('ppo_clipped_pessimistic_branch',np.allclose(stats['surrogate'].numpy(),[-1.2,.8],atol=1e-6,rtol=1e-5),stats['surrogate'].tolist())
 startup_tests()
 # Exercise the downstream environment constructor/trace API on a controlled
 # train-only stationary policy, without evaluating a development checkpoint.
 sys.path.insert(0,str(NAV/'scripts'))
 import eval_r14_repair as evaluation
 class Stationary:
  def act(self,x,deterministic=True):return torch.zeros((1,3)),None,None
 ctx=evaluation.E.build_rollout_ctx('uncalibrated')
 scene=evaluation.E.M.build('L1_empty',6414,ctx['design'])
 settings={'environment':{'t_max_s':45.,'r_goal_m':.5},'implementation_versions':{'observation':'obs-v5'},'stagnation':{'window_s':3.,'min_displacement_m':.02},'safety_continuity':{'enabled':True,'severity_floor':.2,'severity_ref_depth_m':.25}}
 result=evaluation.rollout(Stationary(),scene,ctx,500.,settings)
 check('evaluation_production_constructor_and_trace',result['label']=='stagnation' and len(result['trace'])>0 and not result['feasible'])
 files=sorted(str(p.relative_to(NAV)) for p in (NAV/'src').rglob('*.py'))
 files+=['scripts/train_formal.py','scripts/eval_r14_repair.py','scripts/eval_training_round.py','scripts/contract_margin_eval.py','tests/test_r14_repair_gate.py','config/navigation_contract_r14_repair_v2.json','config/value_objective_contract_r14_repair_v1.json']
 report={'schema':'camra-r14-repair-gate-v1','verdict':'PASS' if all(c['ok'] for c in CHECKS) else 'FAIL','implementation_versions':{'observation':'obs-v5','estimator':'rcppo-r14-v3'},'checks':CHECKS,'code_files':{f:digest(NAV/f) for f in files},'protocol':{'path':'config/r14_repair_numerical_protocol_v1.json','sha256':digest(NAV/'config/r14_repair_numerical_protocol_v1.json')},'reference_limitations':['Original JAX dependency remains outside this environment; no dependency installed. Common constant case uses the source-checked transcription; independent scalar enumeration is the primary oracle.','Distinct-baseline insertion, absorbing-failure continuation and explicit boundary table are recorded Go2 correctness adaptations; no claim of bitwise identity with all original JAX outputs.']}
 out=pathlib.Path(args.out)
 with out.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
 print(report['verdict'],len(CHECKS),'checks');return int(report['verdict']!='PASS')
if __name__=='__main__':raise SystemExit(main())
