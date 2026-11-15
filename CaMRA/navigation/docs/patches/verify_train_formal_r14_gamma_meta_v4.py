import argparse,ast,dataclasses,hashlib,importlib.util,json,pathlib,sys
from types import SimpleNamespace
import numpy as np
parser=argparse.ArgumentParser(description='Verify the isolated NAV-064 gamma/metadata repair; no performance sampling')
parser.add_argument('--navigation-root',required=True)
parser.add_argument('--out',required=True)
parser.add_argument('--source-sha256',required=True)
cli_args=parser.parse_args()
NAV=pathlib.Path(cli_args.navigation_root).resolve()
assert hashlib.sha256((NAV/'scripts/train_formal.py').read_bytes()).hexdigest()==cli_args.source_sha256
sys.path[:0]=[str(NAV/'scripts'),str(NAV/'src'),str(NAV.parent/'calibration/src')]
import train_formal as F
from camra_navigation.algorithm import storage as S
spec=importlib.util.spec_from_file_location('gate_fixture',NAV/'tests/test_r14_repair_gate.py'); G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)
checks=[]
def check(name,ok,detail=None):
 checks.append(dict(name=name,ok=bool(ok),detail=detail));print(checks[-1],flush=True)
value=json.loads((NAV/'config/value_objective_contract_r14_repair_v1.json').read_bytes());est=value['estimator']
cfg=dataclasses.replace(S.ObjectiveConfig.from_contracts(str(NAV/'config/value_objective_contract_r14_repair_v1.json'),r_goal_m=.5),joint_min_recursion=True,h_semantics='v2_deep_well')
source=(NAV/'scripts/train_formal.py').read_text();tree=ast.parse(source);main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
loop=next(x for x in main.body if isinstance(x,ast.For) and ast.unparse(x.target)=='update_idx')
schedule=next(x for x in loop.body if isinstance(x,ast.Assign) and any(isinstance(y,ast.Call) and isinstance(y.func,ast.Name) and y.func.id=='reach_gamma_at_update' for y in ast.walk(x)))
store_loop=next(x for x in ast.walk(main) if isinstance(x,ast.For) and ast.unparse(x.target)=='(identity, kind, arrays, env_done)')
globals_={'np':np,'S':S,'_dc':dataclasses,'reach_gamma_at_update':F.reach_gamma_at_update,'max_updates':600,'reach_estimator':est,'obj_cfg':cfg}
trace=[]
for u in (0,50,149,150,299,300,599):
 globals_['update_idx']=u
 exec(compile(ast.Module(body=[schedule],type_ignores=[]),'<actual production schedule>','exec'),globals_)
 gamma=globals_['obj_cfg'].gamma_h
 expected=min(.99999,.999+(.99999-.999)*(2*u/600))
 trace.append([u,gamma]);check(f'actual_schedule_update_{u}',abs(gamma-expected)<1e-15,[gamma,expected])
check('reach_contracts_and_joint_discount_preserved',all(0<g<1 for u,g in trace) and globals_['obj_cfg'].gamma_joint==1)
check('invalid_schedule_arguments_rejected',all(_ for _ in [True]))
for args in [(-1,600,.999,.99999),(600,600,.999,.99999),(0,0,.999,.99999),(0,600,.999,1.)]:
 try: F.reach_gamma_at_update(*args)
 except ValueError: pass
 else: checks[-1]['ok']=False
# Consume the scheduled value through actual close_segment and compare with the fixed-gamma case.
a=G.episode([4.,3.,2.],kind='truncation',bh=-.2,bc=.2)
F.obj_cfg=cfg
old,_=F.close_segment(SimpleNamespace(arrays=lambda:{k:v.copy() for k,v in a.items()}),'truncation',.2,-.2,env_done=False,label=None)
F.obj_cfg=globals_['obj_cfg']
new,_=F.close_segment(SimpleNamespace(arrays=lambda:{k:v.copy() for k,v in a.items()}),'truncation',.2,-.2,env_done=False,label=None)
check('scheduled_gamma_changes_production_reach_target',np.max(np.abs(old['obj_target_reach']-new['obj_target_reach']))>1e-8)
check('joint_and_cost_targets_unaffected_by_reach_schedule',np.array_equal(old['obj_target_total'],new['obj_target_total']) and np.array_equal(old['obj_target_cost'],new['obj_target_cost']))
# Execute the actual production write-loop AST with identities captured before two resets.
st=F.EpisodeState(0,'synthetic-map',5,41,{'z0_J':500.,'k':1.1})
id0=F.segment_identity(st);st.episode_z0={'z0_J':300.,'k':.7};st.reset_episode();id1=F.segment_identity(st)
st.episode_z0={'z0_J':200.,'k':.6};st.reset_episode()
class Store:
 def __init__(self): self.rows=[]
 def expected_chunk_index(self): return len(self.rows)
 def write_chunk(self,a,m): self.rows.append(dict(m))
store=Store();env={'np':np,'S':S,'window_segments':[(id0,'success',{'t':np.arange(2),'episode_id':np.zeros(2,dtype=int)},True),(id1,'success',{'t':np.arange(2),'episode_id':np.ones(2,dtype=int)},True)],'stores':[store],'obj_cfg':cfg,'update_idx':50}
exec(compile(ast.Module(body=[store_loop],type_ignores=[]),'<actual production chunk writer>','exec'),env)
check('actual_writer_keeps_old_episode_after_reset',store.rows[0]['episode_id']==0 and store.rows[1]['episode_id']==1)
check('actual_writer_keeps_old_budget_after_reset',store.rows[0]['episode_z0_J']==500 and store.rows[0]['episode_k']==1.1 and store.rows[1]['episode_z0_J']==300 and store.rows[1]['episode_k']==.7)
check('actual_writer_logs_update_and_discounts',all(r['update_idx']==50 and r['gamma_h']==cfg.gamma_h and r['gamma_joint']==1 for r in store.rows))
report={'owner':'NAV-064','schema':'camra-r14-gamma-meta-candidate-v2','verdict':'PASS' if all(r['ok'] for r in checks) else 'FAIL','checks':checks,'candidate_source':str(NAV/'scripts/train_formal.py'),'source_sha256':hashlib.sha256((NAV/'scripts/train_formal.py').read_bytes()).hexdigest(),'reference_schedule':'RC-PPO/rl/EC-EFPPO.py gamma_2 -> xs[:,1] -> training ent_gamma[1], saturates at half of total timesteps','scope':'Isolated source candidate and synthetic numerical checks only; no new performance run or changed active registration.'}
report['verification_source_sha256']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
report['schema']='camra-r14-gamma-meta-candidate-v4'
path=pathlib.Path(cli_args.out)
with path.open('x') as f: json.dump(report,f,indent=2)
assert report['verdict']=='PASS'
