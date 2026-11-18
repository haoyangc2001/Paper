"""Record approved v10 scope, verify resources, and generate registration patch."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
HERE = Path(__file__).resolve().parent
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
read = lambda p: json.loads(p.read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plan = read(PLAN)
states = {t['id']: t['status'] for t in plan['tasks']}
assert len(states) == 67 and sum(states.values()) == 52
task = next(t for t in plan['tasks'] if not t['status'] and all(states[k] for k in t['depends_on']))
assert task['id'] == 'NAV-065'
with (HERE / 'validation-before.log').open('x') as f:
    result = subprocess.run([sys.executable, '-B', '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', str(PLAN), '--mode', 'update'], stdout=f, stderr=subprocess.STDOUT)
assert result.returncode == 0
git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
assert git('rev-parse', 'HEAD') == 'eea3945d56f362c6097dc2f8a7a40a0ee7ce907a'
assert not git('diff', '--name-only') and not git('diff', '--cached', '--name-only')
review = read(NAV / 'reports/nav065_head_probe_v10_preparation_review.json')
for name, digest in review['evidence'].items():
    assert sha(ROOT / name) == digest, name
v2 = read(NAV / 'reports/nav064_terminal_tail_training_release_v1.json')['v2_verified_files']
for name, digest in v2.items():
    assert sha(Path(name)) == digest, name
ssh = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=6', '192.168.20.11', 'hostname'], capture_output=True, text=True, timeout=10)
context = {'timestamp': datetime.now().astimezone().isoformat(), 'host': socket.getfqdn(), 'tmux': os.environ['TMUX'],
    'cwd': str(Path.cwd()), 'python': sys.executable, 'python_version': sys.version, 'git': git('rev-parse', 'HEAD'),
    'git_status': git('status', '--short'), 'plan': plan, 'v2_verified_files': v2,
    'ssh_returncode': ssh.returncode, 'ssh_output': ssh.stdout + ssh.stderr,
    'ssh_exception': 'Prior explicit authorization for direct execution on current tiger11/tmux persists.',
    'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.free', '--format=csv,noheader'], text=True)}
with (HERE / 'baseline.json').open('x') as f:
    json.dump(context, f, ensure_ascii=False, indent=2)
sys.path.insert(0, str(NAV / 'scripts'))
import probe_nav065_head as H
proposal = read(H.PROPOSAL)
assert not (H.R.WORK / proposal['output_name']).exists()
registration = NAV / 'config/nav065_head_probe_prereg_v10.json'
assert not registration.exists() and not registration.with_suffix('.json.sha256').exists()
protocol = dict(proposal, approved=True, status='APPROVED',
    authorization_reference='用户在v10单模型副本、两个原输出层各求解一次/合计2次、新采样0、开发评分0的具体提案后明确回复“确认”；本次单次尝试，不授权追加、重试、参数扫描、开发评分或下游。',
    implementation_hashes={name: sha(NAV / name) for name in H.BOUND_INPUTS})
content = json.dumps(protocol, ensure_ascii=False, indent=2) + '\n'
pin = hashlib.sha256(content.encode()).hexdigest()
record = {'timestamp': context['timestamp'], 'summary': 'NAV-065/1.3.1 v10批准执行开工：固定一个副本，两个原输出层各一次求解',
    'actions': ['用户确认具体v10提案后重新读取AGENTS、legacy技能/规则、主计划及最新交接；核对NAV-064上游证据、NAV-065依赖和全部历史。',
        '现场核验tiger11/tmux、绝对Python3.11.15、GPU、v2冻结33项、源码和准备证据散列；SSH publickey拒绝沿用已有同机执行授权，不换主机。',
        '按原未批准提案生成独立批准登记、全实现散列和pin；原提案与旧v8/v9登记及产物保持原样。'],
    'files_changed': [str(registration.relative_to(ROOT)), str(registration.with_suffix('.json.sha256').relative_to(ROOT)), str(PLAN.relative_to(ROOT)), str(HERE.relative_to(ROOT)) + '/'],
    'commands': [sys.executable + ' -B -u ' + str(Path(__file__)), 'tmux=nav064-r14-repair-20261115:nav065-v10-execution; cwd=' + str(NAV)],
    'results': ['本项仍为RC-PPO固定actor后双critic估值诊断，为后续可靠价值预算搜索查明局部拟合和全量退化原因；NAV-065=false、52/67、原G4 FAIL不变。',
        '批准一个full_s0深拷贝，固定v8末次源与原25起点、actor和两个body；原两个输出层各一次float64最小改变量求解，总2次，转回原float32执行原局部及原v5全train检查。',
        '梯度更新0、新回合0、开发评分0；原目标/架构/W/beta/v2/评分不改。局部通过也不构成NAV-065科学通过，不生成部署包或执行NAV-066，不重用旧额度。',
        '双重验收为实际求解计数、原标签与生产前向/原门禁、CPU重载/原非head与六模型保护及全量影响复核；完成后交付诊断模型、报告、追加记录和Git提交。'],
    'evidence': [str(HERE / 'baseline.json'), 'CaMRA/navigation/reports/nav065_head_probe_v10_preparation_review.json', str(registration.relative_to(ROOT)), 'registration_sha256=' + pin],
    'remaining_work': ['应用批准登记和追加开工记录后执行一次原隔离入口；按legacy由主agent复核实际产物和科学限制，保留全部失败。'], 'blockers': []}
text = PLAN.read_text()
start = text.index('"id": "NAV-065"')
end = text.index('\n      ],\n   "test_reports": [', start)
tail_start = text.rfind('\n  "blockers": [', start, end) + 1
tail = text[tail_start:end + len('\n      ],\n   "test_reports": [')]
new = tail[:tail.rfind('\n}')] + '\n},\n' + json.dumps(record, ensure_ascii=False, indent=2) + '\n      ],\n   "test_reports": ['
patch = '*** Begin Patch\n'
for path, value in ((registration, content), (registration.with_suffix('.json.sha256'), pin + '\n')):
    patch += '*** Add File: ' + str(path) + '\n' + '\n'.join('+' + line for line in value.splitlines()) + '\n'
patch += '*** Update File: ' + str(PLAN) + '\n@@\n' + '\n'.join('-' + line for line in tail.splitlines()) + '\n' + '\n'.join('+' + line for line in new.splitlines()) + '\n*** End Patch\n'
with (HERE / 'approved-opening.patch').open('x') as f:
    f.write(patch)
print(json.dumps({k: v for k, v in context.items() if k not in ('plan', 'git_status', 'v2_verified_files')}, ensure_ascii=False))
print(json.dumps({'registration_pin': pin, 'implementation_files': len(H.BOUND_INPUTS), 'first_eligible': task['id'], 'v2_verified': len(v2)}, ensure_ascii=False))
