"""Fresh NAV-065 context, finite resource checks, and append-only opening patch."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
HERE = Path(__file__).resolve().parent
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
PYTHON = '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == PYTHON
raw = PLAN.read_bytes()
p = json.loads(raw)
states = {t['id']: t['status'] for t in p['tasks']}
assert len(states) == 67 and sum(states.values()) == 52
current = next(t for t in p['tasks'] if not t['status'] and all(states[x] for x in t['depends_on']))
assert current['id'] == 'NAV-065'
with (HERE / 'validation-before.log').open('x') as f:
    result = subprocess.run([PYTHON, '-B', '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', str(PLAN), '--mode', 'update'], stdout=f, stderr=subprocess.STDOUT)
assert result.returncode == 0
v2 = json.loads((ROOT / 'CaMRA/navigation/reports/nav064_terminal_tail_training_release_v1.json').read_bytes())['v2_verified_files']
for name, digest in v2.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
git = lambda *a: subprocess.check_output(['git', *a], cwd=ROOT, text=True).strip()
ssh = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=6', '192.168.20.11', 'hostname'], capture_output=True, text=True, timeout=10)
context = {'timestamp': datetime.now().astimezone().isoformat(), 'host': socket.getfqdn(), 'tmux': os.environ['TMUX'], 'cwd': str(ROOT), 'python': PYTHON, 'python_version': sys.version, 'git': git('rev-parse', 'HEAD'), 'git_status': git('status', '--short'), 'submodules': git('submodule', 'status'), 'plan': p, 'plan_sha256': hashlib.sha256(raw).hexdigest(), 'v2_files_verified': len(v2), 'ssh_returncode': ssh.returncode, 'ssh_output': ssh.stdout + ssh.stderr, 'ssh_exception': 'Existing explicit permission for direct current tiger11/tmux execution persists.', 'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.free', '--format=csv,noheader'], text=True)}
with (HERE / 'baseline.json').open('x') as f:
    json.dump(context, f, ensure_ascii=False, indent=2)
review = {k: v for k, v in p.items() if k != 'tasks'}
review['tasks'] = [{**{k: v for k, v in t.items() if k != 'work_records'}, 'work_record_summaries': [r['summary'] for r in t['work_records']]} for t in p['tasks']]
with (HERE / 'plan-reading.json').open('x') as f:
    json.dump(review, f, ensure_ascii=False, indent=2)
with (HERE / 'current-task-reading.json').open('x') as f:
    json.dump(current, f, ensure_ascii=False, indent=2)
record = {'timestamp': context['timestamp'], 'summary': 'NAV-065/1.3.1 v11开工：按legacy准备全train分层与正则化输出层验证', 'actions': ['重新读取根AGENTS、模块README、当前交接、legacy技能/执行规则及字段规范；完整解析主计划并运行update校验，按原数组确认NAV-065及已完成NAV-064依赖。', '核验tiger11/tmux、绝对解释器、GPU、Git/子模块、33项完整v2资源；沿用用户已明确批准的同机执行例外。'], 'files_changed': ['doc/plan/20261109_camra_navigation_implementation.json', str(HERE.relative_to(ROOT)) + '/'], 'commands': [PYTHON + ' -B ' + str(Path(__file__)), 'tmux=nav064-r14-repair-20261115:nav065-v11-preparation；cwd=' + str(ROOT)], 'results': ['目标仍为RC-PPO Phase II固定actor后估值，为后续价值预算搜索提供可信同目标双critic；NAV-065=false、52/67、原G4 FAIL。', '消费固定v8末次full_s0及其162条train/59877步，保留v9/v10诊断与失败；本次只读预检、隔离v11入口和主agent基础验证，不执行真实候选求解或梯度训练。', '拟把旧/新增轨迹全时刻、全部起点和原沿途边界样本共同纳入同一数值回归，并对相对源输出层改变量正则化；不换actor/body/架构、obs-v7、共同停止标签、W、beta=0、原v5 train/G4、冻结v2/校准。', '验收为全量样本接入、批准守卫、稳定性计算与原生产门禁调用正确，主agent基本测试及来源复核通过；交付单候选具体提案，不代表科学通过，不启动NAV-066。'], 'evidence': [str(HERE / 'baseline.json'), 'CaMRA/navigation/reports/nav065_head_probe_v10_review.json'], 'remaining_work': ['实现全量分层正则回归入口及只读预检、基本测试、独立提案、追加收尾与交接。'], 'blockers': ['v10已结束；新的真实候选输出层求解属于拟合，须评审具体数量/过程并明确授权，不挪用旧额度。']}
text = raw.decode()
start = text.index('"id": "NAV-065"')
end = text.index('\n      ],\n   "test_reports": [', start)
tail_start = text.rfind('\n  "blockers": [', start, end) + 1
tail = text[tail_start:end + len('\n      ],\n   "test_reports": [')]
new = tail[:tail.rfind('\n}')] + '\n},\n' + json.dumps(record, ensure_ascii=False, indent=2) + '\n      ],\n   "test_reports": ['
patch = '*** Begin Patch\n*** Update File: ' + str(PLAN) + '\n@@\n' + '\n'.join('-' + s for s in tail.splitlines()) + '\n' + '\n'.join('+' + s for s in new.splitlines()) + '\n*** End Patch\n'
with (HERE / 'opening.patch').open('x') as f:
    f.write(patch)
print(json.dumps({k: v for k, v in context.items() if k not in ('plan', 'gpu', 'git_status')}, ensure_ascii=False))
