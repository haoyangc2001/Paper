"""Snapshot this active once-only run without touching its output or locks."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT/'CaMRA/navigation'
EV = ROOT/'.worklog-testing/NAV-043/timeout-extension-v1'
OUT = Path('/pub/data/caohy/work/camra_navigation/nav043_timeout_extension_v1')
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
def load(p): return json.loads(Path(p).read_bytes())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol = load(NAV/'config/nav043_timeout_extension_v1.json')
for name,digest in protocol['sealed_hashes'].items(): assert sha(name) == digest, name
pair = load(NAV/'config/nav066_decision_prereg_v2.json')
protected = {str(NAV/k):v for k,v in pair['implementation_hashes'].items()}
protected.update(pair['protected_hashes'])
protected.update({c['checkpoint']:c['checkpoint_sha256'] for c in pair['six_candidates']})
for name,digest in protected.items(): assert sha(name) == digest, name
owner = load(OUT/'OWNER.json')
proof = load(EV/'prelaunch-commit.json')
assert owner['git_head'] == proof['commit']
levels = []
for level in protocol['order']:
    app = OUT/level
    if not app.exists():
        levels.append({'level':level, 'started':False})
        continue
    proc = load(app/'PROCESS.json')
    row = {'level':level, 'started':True, 'process':proc,
        'pid_exists':(Path('/proc')/str(proc['pid'])).exists(),
        'markers':sorted(p.name for p in app.glob('*.json'))}
    if (app/'full/result.json').exists(): row['result'] = load(app/'full/result.json')
    levels.append(row)
snapshot = {'at':datetime.now().astimezone().isoformat(),
    'status':'RUNNING_SNAPSHOT_NOT_ACCEPTANCE', 'owner':owner, 'levels':levels,
    'protected_hashes_checked':len(protected), 'source_hashes_checked':len(protocol['sealed_hashes']),
    'root_finished':(OUT/'FINISHED.json').exists(), 'root_held':(OUT/'HELD.json').exists(),
    'NAV_043_pass':False, 'G5_pass':False}
with (EV/'execution-started.json').open('x') as stream:
    json.dump(snapshot, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
planpath = ROOT/'doc/plan/20261109_camra_navigation_implementation.json'
before = planpath.read_text()
plan = json.loads(before)
task = next(t for t in plan['tasks'] if t['id'] == 'NAV-043')
task['work_records'].append({'timestamp':snapshot['at'],
    'summary':'90秒执行延时诊断基本检查/预检通过，准备已提交14fda95，原一次物理包已在tmux启动',
    'actions':['%784首次合成超时检查被静止历史的停滞优先事件拦截；原源码/log保留，%786仅修正合成位移历史后15项通过，不改生产规则。',
        '%787实际tiger11/tmux/绝对解释器/GPU6预检通过，295保护身份与26来源通过，legacy计划校验通过。',
        '%788首次提交因Git忽略原日志退出，原失败保留；%789明确force白名单证据、原始日志/patch保留字节，准备提交14fda95。',
        '%790运行新launch.sh，原独立OWNER/持久锁及L1子进程已登记；结束后同一个shell自动执行主agent保存事件/能量/观测复核，无测试子agent。',
        '本次只测试现有actor在真实状态闭环中多给时间是否改善；保持原45秒观测和原预算规则，不把旧critic/裕度声明为90秒有效。'],
    'files_changed':['.worklog-testing/NAV-043/timeout-extension-v1',
        'doc/plan/20261109_camra_navigation_implementation.json','AGENTS.md',
        'CaMRA/navigation/docs/nav043_timeout_extension_20261120.md'],
    'commands':['%786: basic_v2.sh -> 15 MAIN_AGENT_TIMEOUT_BASIC_PASS',
        '%787: preflight.sh -> TIMEOUT_PREFLIGHT_PASS / 295 / GPU6=4MiB',
        '%789: commit_v2.py -> 14fda95', '%790: launch.sh -> OWNER and L1 PROCESS saved',
        'save_progress.py -> immutable execution-started snapshot, no new inference or physics'],
    'results':['已经启动不等于已到达或通过；此快照尚无完整三图结果，不能报告成功率。',
        '本次新范围最多3应用/3物理回合/15W/60000step/3小时，无自动重试。',
        '原模型、v2、alpha/delta及所有旧失败保持；NAV-043=false、1.5.0/55/67、G5未过。'],
    'evidence':['.worklog-testing/NAV-043/timeout-extension-v1/execution-started.json',
        '.worklog-testing/NAV-043/timeout-extension-v1/prelaunch-commit.json',
        '/pub/data/caohy/work/camra_navigation/nav043_timeout_extension_v1/OWNER.json'],
    'remaining_work':['检查原%790终态和保存复核报告，不重启；逐图分列90秒到达、安全、实际/内部预算和45秒前缀。',
        '若旧模型仍有局限，依据实际诊断确定后续；不默认训练、响应拟合或宣告通用性/G5。'],
    'blockers':[]})
after = json.dumps(plan, ensure_ascii=False, indent=2)+'\n'
a,b = before.splitlines(),after.splitlines()
i = 0
while i < min(len(a),len(b)) and a[i] == b[i]: i += 1
ea,eb = len(a),len(b)
while ea > i and eb > i and a[ea-1] == b[eb-1]: ea-=1;eb-=1
patch = '*** Begin Patch\n*** Update File: '+str(planpath)+'\n@@\n'
patch += ''.join(' '+x+'\n' for x in a[max(0,i-3):i])
patch += ''.join('-'+x+'\n' for x in a[i:ea])
patch += ''.join('+'+x+'\n' for x in b[i:eb])
patch += ''.join(' '+x+'\n' for x in a[ea:min(len(a),ea+3)])
patch += '*** End Patch\n'
(EV/'progress.patch').write_text(patch)
print(json.dumps(snapshot, ensure_ascii=False), flush=True)
