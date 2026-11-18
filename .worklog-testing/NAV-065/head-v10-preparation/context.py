"""Record NAV-065 v10 read-only preparation context and append-only opening."""
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
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
p = json.loads(PLAN.read_bytes())
states = {t['id']: t['status'] for t in p['tasks']}
assert len(states) == 67 and sum(states.values()) == 52
assert next(t['id'] for t in p['tasks'] if not t['status'] and all(states[x] for x in t['depends_on'])) == 'NAV-065'
with (HERE / 'validation-before.log').open('x') as f:
    v = subprocess.run([sys.executable, '-B', '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', str(PLAN), '--mode', 'update'], stdout=f, stderr=subprocess.STDOUT)
assert v.returncode == 0
v2 = json.loads((ROOT / 'CaMRA/navigation/reports/nav064_terminal_tail_training_release_v1.json').read_bytes())['v2_verified_files']
for name, digest in v2.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
git = lambda *a: subprocess.check_output(['git', *a], cwd=ROOT, text=True).strip()
ssh = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=6', '192.168.20.11', 'hostname'], capture_output=True, text=True, timeout=10)
context = {'timestamp': datetime.now().astimezone().isoformat(), 'host': socket.getfqdn(), 'tmux': os.environ['TMUX'], 'cwd': str(ROOT), 'python': sys.executable, 'python_version': sys.version, 'git': git('rev-parse', 'HEAD'), 'git_status': git('status', '--short'), 'plan': p, 'v2_files_verified': len(v2), 'ssh_returncode': ssh.returncode, 'ssh_output': ssh.stdout + ssh.stderr, 'ssh_exception': 'Prior explicit same-host tiger11/tmux authorization persists.', 'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.free', '--format=csv,noheader'], text=True)}
with (HERE / 'baseline.json').open('x') as f:
    json.dump(context, f, ensure_ascii=False, indent=2)
record = {'timestamp': context['timestamp'], 'summary': 'NAV-065/1.3.1 v10开工：按legacy推进末层可拟合性诊断与最小改动准备', 'actions': ['重新完整读取AGENTS、legacy技能/执行规则/字段规范，完整解析主计划并校验update，按数组确认NAV-065及上游NAV-064。', '核验tiger11/tmux、绝对解释器、GPU、v2冻结33项和既有Git/会话；沿用用户已批准的同机执行例外。'], 'files_changed': ['doc/plan/20261109_camra_navigation_implementation.json', str(HERE.relative_to(ROOT)) + '/'], 'commands': [sys.executable + ' -B ' + str(Path(__file__)), 'tmux=nav064-r14-repair-20261115:nav065-v10-preparation；cwd=' + str(ROOT)], 'results': ['总目标保持RC-PPO Phase II固定actor后双critic估值，为NAV-066提供可信价值；当前NAV-065=false、52/67和原G4 FAIL。', '本轮授权用于既有train数据的只读几何/学习曲线诊断、隔离末层求解候选入口和主agent基本测试；不执行真实模型拟合、新回合或development评分。', '保持原cost/reach共同停止标签、W、beta=0、obs-v7、两独立critic及架构、六actor、固定v2/校准和最终v5 train/G4；只读分析不能证明泛化或部署通过。', '验收为来源/诊断可复算、批准守卫和生产原门禁接通、基础测试通过及追加交接；进一步真实末层求解须明确数量和新批准。'], 'evidence': [str(HERE / 'baseline.json'), 'CaMRA/navigation/reports/nav065_local_probe_v9_review.json'], 'remaining_work': ['完成v8/v9固定末层特征的数值条件诊断，准备一次不换架构的末层求解提案、隔离入口和基本测试。'], 'blockers': ['v9单次400步已结束；不重启、不挪用旧额度。真实候选末层求解仍是模型拟合，须具体评审批准。']}
text = PLAN.read_text()
start = text.index('"id": "NAV-065"')
end = text.index('\n      ],\n   "test_reports": [', start)
tail_start = text.rfind('\n  "blockers": [', start, end) + 1
tail = text[tail_start:end + len('\n      ],\n   "test_reports": [')]
new = tail[:tail.rfind('\n}')] + '\n},\n' + json.dumps(record, ensure_ascii=False, indent=2) + '\n      ],\n   "test_reports": ['
patch = '*** Begin Patch\n*** Update File: ' + str(PLAN) + '\n@@\n' + '\n'.join('-' + s for s in tail.splitlines()) + '\n' + '\n'.join('+' + s for s in new.splitlines()) + '\n*** End Patch\n'
with (HERE / 'opening.patch').open('x') as f:
    f.write(patch)
print(json.dumps({k: v for k, v in context.items() if k not in ('plan', 'gpu', 'git_status')}, ensure_ascii=False))
