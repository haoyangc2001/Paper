"""Register the user's confirmed one-time recovery; no simulator is started."""
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from zoneinfo import ZoneInfo

from nav067_sources import NAV, ROOT, OUTPUT, read_manifest, sha, write_new

HERE = Path(__file__).resolve().parent
PLAN = ROOT/'doc/plan/20261109_camra_navigation_implementation.json'
VALIDATOR = Path('/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py')
PROPOSAL = NAV/'config/nav067_startup_recovery_proposal_v3.json'
APPROVED = NAV/'config/nav067_startup_recovery_v3.json'
EXPECTED = '7fb223bdef1db1d884fb7973610f727c9cd62462bee9b2fcbaad261205083be7'

assert socket.gethostname().split('.')[0] == 'tiger11'
assert os.environ.get('TMUX_PANE') == '%587'
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
assert not (HERE/'preflight.json').exists()
assert not APPROVED.exists() and not Path(str(APPROVED)+'.sha256').exists()
assert sha(PROPOSAL) == EXPECTED
before = json.loads(PLAN.read_bytes())  # all tasks and historical records
assert before['document']['version'] == '1.4.1'
state = {t['id']:t['status'] for t in before['tasks']}
assert sum(state.values()) == 54
assert next(t['id'] for t in before['tasks'] if not t['status'] and all(state[d] for d in t['depends_on'])) == 'NAV-067'
for t in before['tasks']:
    for record in t['work_records']:
        assert all(k in record for k in ('timestamp','summary','actions','files_changed','commands','results','evidence','remaining_work','blockers'))
check = subprocess.run([sys.executable,'-B',str(VALIDATOR),str(PLAN),'--mode','update'],capture_output=True,text=True)
assert check.returncode == 0, check.stdout+check.stderr
manifest, _, _ = read_manifest()  # validates fixed v2, assets, 295 protected identities and both models
proposal = json.loads(PROPOSAL.read_bytes())
assert proposal['status'] == 'PROPOSED'
for rel, digest in proposal['source_hashes'].items():
    assert sha(NAV/rel) == digest, rel
for path, digest in proposal['evidence_hashes'].items():
    assert sha(path) == digest, path
for slot in proposal['slots']:
    assert not (OUTPUT/slot['relative_output']).exists(), slot
assert not Path('/proc/1232883').exists() and not Path('/proc/1297704').exists()
gpu = subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=index,uuid,name,memory.free','--format=csv,noheader,nounits'],text=True).strip()
assert int(gpu.rsplit(',',1)[1].strip()) >= 12000, gpu
tmux = subprocess.check_output(['tmux','display-message','-t',os.environ['TMUX_PANE'],'-p','#{session_name}:#{window_name} #{window_id} #{pane_id}'],text=True).strip()
now = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
with (HERE/'before-plan.json').open('xb') as stream:
    stream.write(PLAN.read_bytes())
approved = deepcopy(proposal)
approved.update(status='APPROVED',approved_at=now,
    approval={'source':'User reply 确认 to the immediately preceding request for the v3 recovery',
        'scope':'One L1 startup recovery plus the original unstarted L2/L3 slots, at most six 45s episodes; no automatic retry or extra repetitions.',
        'proposal_sha256':EXPECTED,'same_host_execution_exception':'Existing tiger11 direct-session authorization remains in force.'})
write_new(APPROVED,approved)
with Path(str(APPROVED)+'.sha256').open('x') as stream:
    stream.write(sha(APPROVED)+'\n')
task = next(t for t in before['tasks'] if t['id']=='NAV-067')
task['work_records'].append({
    'timestamp':now,
    'summary':'NAV-067/1.4.1恢复开工：用户确认v3一次L1恢复及原L2/L3，批准登记和资源核验完成',
    'actions':[
        '重读根AGENTS、legacy技能/执行及字段规则、完整解析计划和当前NAV-067/依赖/历史；NAV-066接受局限、旧科学FAIL及54/67保持。',
        '登记用户对已准备v3方案的明确确认，只恢复L1原两槽一次，并执行原L2/L3四槽；最多6个45秒回合、1秒warmup，失败停下，不自动重试。',
        '本项把已冻结两候选与固定v2实际接通并审计clean/真实命令/10-50-200Hz/复位/机械功；沿用同机执行例外，现场核验绝对解释器、GPU6及295项保护身份。',
        '新增独立APPROVED登记和pin，不改PROPOSED提案、原manifest/attempt1、actor/critic、目标/obs-v7、校准、冻结v2或共享框架；实际受控反例和重复性额度不在本次授权内。'],
    'files_changed':[str(APPROVED.relative_to(ROOT)),str(APPROVED.relative_to(ROOT))+'.sha256',str(PLAN.relative_to(ROOT)),str(HERE.relative_to(ROOT))],
    'commands':['tiger11/tmux '+tmux+'; absolute Python3.11.15 -B .worklog-testing/NAV-067/recovery-v3-execution/prepare.py','legacy validate_worklog.py <NAV plan> --mode update'],
    'results':['批准登记/pin已建立；295项保护身份、提案及旧证据hash匹配；原进程已终止且新输出未占用；GPU6可用。','本阶段尚未启动应用、采集回合、训练或W查询，NAV-067仍false。'],
    'evidence':[str(HERE.relative_to(ROOT))+'/preflight.json',str(PROPOSAL.relative_to(ROOT)),str(APPROVED.relative_to(ROOT))],
    'remaining_work':['在原tmux执行v4入口并逐level审计实际clean/命令/复位/子步/机械功，异常保留现场即停。','真实受控反例及同配置重复仍需独立额度；NAV-067全部验收前不进入NAV-043。'],
    'blockers':['真实受控反例和同配置重复尚无独立额度；本次确认仅解除原pilot一次启动恢复阻塞。']})
PLAN.write_text(json.dumps(before,ensure_ascii=False,indent=2)+'\n')
after = json.loads(PLAN.read_bytes())
original = json.loads((HERE/'before-plan.json').read_bytes())
for old,new in zip(original['tasks'],after['tasks']):
    assert {k:v for k,v in old.items() if k!='work_records'} == {k:v for k,v in new.items() if k!='work_records'}
    assert new['work_records'][:len(old['work_records'])] == old['work_records']
assert {k:v for k,v in original.items() if k!='tasks'} == {k:v for k,v in after.items() if k!='tasks'}
check = subprocess.run([sys.executable,'-B',str(VALIDATOR),str(PLAN),'--mode','update'],capture_output=True,text=True)
with (HERE/'plan-check.log').open('x') as stream:
    stream.write(check.stdout+check.stderr)
assert check.returncode == 0, check.stdout+check.stderr
report = {'status':'APPROVED_PREFLIGHT_ONLY','timestamp':now,'host':socket.getfqdn(),'python':sys.executable,'tmux':tmux,'gpu':gpu,
    'proposal_sha256':EXPECTED,'amendment_sha256':sha(APPROVED),'manifest_sha256':sha(OUTPUT/'manifest.json'),
    'protected_hashes_checked':manifest['binding']['protected_hashes_checked'],'tasks_read':len(after['tasks']),
    'historical_records_read':sum(len(t['work_records']) for t in original['tasks']),
    'old_task_fields_and_history_preserved':True,'new_episodes':0,'new_training_updates':0,'new_W_queries':0}
write_new(HERE/'preflight.json',report)
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
