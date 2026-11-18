"""Generate closing work records only after the real-data read-only audit passes."""
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT=Path('/home/caohy/repositories/Paper');NAV=ROOT/'CaMRA/navigation';HERE=Path(__file__).resolve().parent
assert socket.gethostname().split('.')[0]=='tiger11' and os.environ.get('TMUX')
read=lambda p:json.loads(p.read_bytes())
r=read(NAV/'reports/nav065_local_probe_v9_review.json')
assert r['integrity_pass'] and not r['local_gate_pass'] and not r['full_train_gate_pass']
assert r['optimizer_trials']==400 and r['v2_files_verified']==33
assert r['all_six_actors_unchanged'] and r['source_models_unchanged']
plan=ROOT/'doc/plan/20261109_camra_navigation_implementation.json'
before=json.loads(subprocess.check_output(['git','show',r['base_git']+':'+str(plan.relative_to(ROOT))],cwd=ROOT))
current=read(plan)
assert {k:v for k,v in before.items() if k!='tasks'}=={k:v for k,v in current.items() if k!='tasks'}
for old,new in zip(before['tasks'],current['tasks'],strict=True):
    if old['id']=='NAV-065':
        assert {k:v for k,v in old.items() if k!='work_records'}=={k:v for k,v in new.items() if k!='work_records'}
        assert new['work_records'][:-1]==old['work_records']
    else:assert old==new
summary=('用户确认v9后在tiger11/tmux完成一次full_s0副本25起点局部拟合，实际400更新、新回合/开发评分0。'
    '局部误放行13→4、成本符号错误11→4，但到达符号错误9→9、相对MAE69.86%→44.82%仍未到35%线；局部FAIL。'
    '诊断副本原90回合全时刻到达相对MAE15.65%→293.37%、旧新增72回合43.85%→140.95%，完整train仍FAIL。'
    '六原模型/actor及33项v2身份不变，原主预算18/18实际可行与起点识别保留；未替换部署模型。'
    'legacy主agent逐步/原门禁/CPU重载复核通过，无测试子agent；执行完整性不等于科学通过。'
    '原G4 FAIL、52/67、NAV-065=false和NAV-066未启动保持；本次尝试结束，不自动重启、扩展或加跑。')
record={'timestamp':datetime.now().astimezone().isoformat(),
    'summary':'NAV-065/1.3.1 v9收尾：400步局部诊断FAIL、全train估值退化，保留原模型',
    'actions':['执行已批准一次25起点Adam局部拟合至400步上限，保存诊断副本及逐步日志，原模型不替换。',
        '主agent重建来源train目标，复核原完整门禁、CPU重载、全部更新日志、401份局部预测与符号/误差、六原模型/actor和33项v2身份。',
        '保存首次验收脚本元组/JSON数组比较失败，规范化比较后只读重验通过，无重复优化；同步结果、交接与下游限制。'],
    'files_changed':['doc/plan/20261109_camra_navigation_implementation.json','AGENTS.md','CaMRA/navigation/README.md','CaMRA/navigation/docs/implementation_handoff.md',
        'CaMRA/navigation/config/nav065_local_probe_prereg_v9.json','CaMRA/navigation/config/nav065_local_probe_prereg_v9.json.sha256',
        'CaMRA/navigation/docs/nav065_local_probe_v9_review_20261118.md','CaMRA/navigation/reports/nav065_local_probe_v9_review.json',str(HERE.relative_to(ROOT))+'/'],
    'commands':['host=tiger11.ios.ac.cn；tmux=nav064-r14-repair-20261115:nav065-v9-execution；cwd='+str(NAV)+'；Python='+sys.executable,
        '<Python> -B -u scripts/probe_nav065_local.py --registration config/nav065_local_probe_prereg_v9.json',
        '<Python> -B -u '+str(HERE/'review.py'),'legacy validate_worklog.py doc/plan/20261109_camra_navigation_implementation.json --mode update'],
    'results':[summary,'固定原目标、W/beta、预算输入、架构、数据隔离和最终v5 train/G4；新版诊断入口实际复用原生产oracle/门禁。',
        '本次局部更新改善部分成本阻断，却没有修正9个reach分量符号；不能据此认定架构不可能，也不能说单一全量梯度冲突已被证明。',
        '无开发重评、新G4、部署包、预算搜索或物理执行；原NAV-064策略证据保留，下游继续阻塞。'],
    'evidence':['CaMRA/navigation/reports/nav065_local_probe_v9_review.json','CaMRA/navigation/docs/nav065_local_probe_v9_review_20261118.md',
        str(HERE/'context.json'),str(HERE/'process.json'),str(HERE/'review-attempt1.log'),r['evidence']['output']],
    'remaining_work':['先依据已存曲线/近邻目标分析局部拟合不足与全量退化，再审议有界后续方案；不自动新训练。',
        '六候选原完整train、development/G4及部署身份全部通过后才完成NAV-065并允许NAV-066。'],
    'blockers':['已批准400步尝试已结束，局部和全train均FAIL；不增加或转移额度、不降低门槛。']}
changes=[]
def change(path,old,new):
    changes.append('*** Update File: '+str(path)+'\n@@\n'+'\n'.join('-'+s for s in old.splitlines())+'\n'+'\n'.join('+'+s for s in new.splitlines())+'\n')
for path,prefix,link in [(NAV/'README.md','2026-11-18最新结果：','docs/'),(NAV/'docs/implementation_handoff.md','## 2026-11-18 NAV-065 v9结束：局部未过、全量估值退化\n\n','')]:
    heading=path.read_text().splitlines()[0]
    change(path,heading,heading+'\n\n'+prefix+summary+'\n\n详见[v9结果]('+link+'nav065_local_probe_v9_review_20261118.md)。以下为历史时点。')
anchor='## 前置状态：v2 选择与历史校准复核（2026-11-09）'
change(ROOT/'AGENTS.md',anchor,'### NAV-065最新补充（2026-11-18，局部诊断v9已结束）\n\n'+summary+'见[结果](CaMRA/navigation/docs/nav065_local_probe_v9_review_20261118.md)。\n\n'+anchor)
text=plan.read_text();start=text.index('"id": "NAV-065"');end=text.index('\n      ],\n   "test_reports": [',start)
anchor_start=text.rfind('\n  "blockers": []',start,end);assert anchor_start>start
old=text[anchor_start+1:end+len('\n      ],\n   "test_reports": [')]
new='  "blockers": []\n},\n'+json.dumps(record,ensure_ascii=False,indent=2)+'\n      ],\n   "test_reports": ['
change(plan,old,new)
with (HERE/'closing.patch').open('x') as f:f.write('*** Begin Patch\n'+''.join(changes)+'*** End Patch\n')
print('V9_CLOSING_PATCH_READY',flush=True)
