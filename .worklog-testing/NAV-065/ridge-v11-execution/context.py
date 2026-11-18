"""Verify fresh context and generate the explicitly approved v11 registration."""
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
    subprocess.run([sys.executable, '-B', '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', str(PLAN), '--mode', 'update'], stdout=f, stderr=subprocess.STDOUT, check=True)
git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
assert git('rev-parse', 'HEAD') == '5db5ce868b7c54c62a1f5edcc582fc3c976909cd'
assert not git('diff', '--name-only') and not git('diff', '--cached', '--name-only')
review = read(NAV / 'reports/nav065_ridge_probe_v11_preparation_review.json')
assert review['preparation_pass'] and review['basic_tests_passed'] == 38
for name, digest in review['evidence'].items():
    assert sha(ROOT / name) == digest, name
for name, digest in review['bound_input_hashes'].items():
    assert sha(NAV / name) == digest, name
v2 = read(NAV / 'reports/nav064_terminal_tail_training_release_v1.json')['v2_verified_files']
for name, digest in v2.items():
    assert sha(Path(name)) == digest, name
ssh = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=6', '192.168.20.11', 'hostname'], capture_output=True, text=True, timeout=10)
reading = [ROOT / p for p in ('AGENTS.md', 'README.md', 'CaMRA/README.md', 'CaMRA/navigation/README.md', 'CaMRA/calibration/README.md', 'CaMRA/locomotion/README.md', 'CaMRA/navigation/docs/implementation_handoff.md')]
reading += [Path('/home/caohy/.codex/skills/create-plan-worklog-legacy') / p for p in ('SKILL.md', 'references/execution-rules.md', 'references/schema.md')]
context = {'timestamp': datetime.now().astimezone().isoformat(), 'host': socket.getfqdn(), 'tmux': os.environ['TMUX'],
    'cwd': str(Path.cwd()), 'python': sys.executable, 'python_version': sys.version, 'git': git('rev-parse', 'HEAD'),
    'git_status': git('status', '--short'), 'submodules': git('submodule', 'status'), 'plan': plan,
    'read_sources_sha256': {str(p): sha(p) for p in reading}, 'v2_verified_files': v2,
    'ssh_returncode': ssh.returncode, 'ssh_output': ssh.stdout + ssh.stderr,
    'ssh_exception': 'Prior explicit authorization for direct execution on current tiger11/tmux persists.',
    'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.free', '--format=csv,noheader'], text=True)}
with (HERE / 'baseline.json').open('x') as f:
    json.dump(context, f, ensure_ascii=False, indent=2)
sys.path.insert(0, str(NAV / 'scripts'))
import probe_nav065_ridge as G
proposal = read(G.PROPOSAL)
assert not proposal['approved'] and not (G.R.WORK / proposal['output_name']).exists()
registration = NAV / 'config/nav065_ridge_probe_prereg_v11.json'
assert not registration.exists() and not registration.with_suffix('.json.sha256').exists()
protocol = dict(proposal, approved=True, status='APPROVED',
    authorization_reference='用户在v11已落地具体提案（一个模型副本、两个原输出层各拟合一次共2次、新采样0、开发评分0）后明确回复“确认”；只授权此一次尝试，不授权重试、扫描、追加、开发评分、六候选扩展或下游。',
    implementation_hashes={name: sha(NAV / name) for name in G.BOUND_INPUTS})
content = json.dumps(protocol, ensure_ascii=False, indent=2) + '\n'
pin = hashlib.sha256(content.encode()).hexdigest()
record = {'timestamp': context['timestamp'], 'summary': 'NAV-065/1.3.1 v11批准执行开工：一个副本、两次全量正则输出层求解',
    'actions': ['用户确认v11具体提案后重新完整读取AGENTS及适用文档、legacy规则、主计划和最新交接；按原数组核对NAV-065及NAV-064上游证据。',
        '现场核验tiger11/tmux、绝对解释器Python3.11.15、GPU、33项冻结v2、Git/子模块、38基本测试及准备证据散列；保留同机SSH执行授权，不切换主机。',
        '按原提案生成独立批准登记及完整实现hash/pin，原未批准提案和全部旧产物保留。'],
    'files_changed': [str(registration.relative_to(ROOT)), str(registration.with_suffix('.json.sha256').relative_to(ROOT)), str(PLAN.relative_to(ROOT)), str(HERE.relative_to(ROOT)) + '/'],
    'commands': [sys.executable + ' -B -u ' + str(Path(__file__)), 'tmux=nav064-r14-repair-20261115:nav065-v11-execution; cwd=' + str(NAV)],
    'results': ['本项继续RC-PPO固定actor后同目标双critic估值，为可靠价值预算搜索验证全轨迹共同拟合的稳定性；NAV-065=false、52/67、原G4 FAIL。',
        '固定v8末次full_s0及162条train/59877步，以原/新增全轨迹、162起点和866原边界行四组共同拟合；固定0.01正则、rcond和权重，不扫参数。',
        '只在一个深拷贝中对两个原输出层各求解一次、总2次；原actor/body/架构/obs-v7、共同停止目标、W/beta=0、原v5 train/G4及冻结v2不变。',
        '梯度更新/新回合/开发评分均0；通过也不自动扩展或部署。双重验收为计数、原生产门禁前后对比、CPU重载、仅四head变化、原六模型/v2保护及全量退化披露；主agent按legacy验收，无测试子agent。'],
    'evidence': [str(HERE / 'baseline.json'), 'CaMRA/navigation/reports/nav065_ridge_probe_v11_preparation_review.json', str(registration.relative_to(ROOT)), 'registration_sha256=' + pin],
    'remaining_work': ['应用批准登记、追加开工记录并本地提交后运行一次隔离入口；主agent复核实际产物、记录科学限制及交接。'], 'blockers': []}
text = PLAN.read_text()
start = text.index('"id": "NAV-065"')
sentinel = '\n      ],\n   "test_reports": ['
end = text.index(sentinel, start)
tail_start = text.rfind('\n  "blockers": [', start, end) + 1
tail = text[tail_start:end + len(sentinel)]
new = tail[:tail.rfind('\n}')] + '\n},\n' + json.dumps(record, ensure_ascii=False, indent=2) + sentinel
patch = '*** Begin Patch\n'
for path, value in ((registration, content), (registration.with_suffix('.json.sha256'), pin + '\n')):
    patch += '*** Add File: ' + str(path) + '\n' + '\n'.join('+' + line for line in value.splitlines()) + '\n'
patch += '*** Update File: ' + str(PLAN) + '\n@@\n' + '\n'.join('-' + line for line in tail.splitlines()) + '\n' + '\n'.join('+' + line for line in new.splitlines()) + '\n*** End Patch\n'
with (HERE / 'approved-opening.patch').open('x') as f:
    f.write(patch)
print(json.dumps({k: v for k, v in context.items() if k not in ('plan', 'git_status', 'v2_verified_files', 'read_sources_sha256')}, ensure_ascii=False))
print(json.dumps({'registration_pin': pin, 'implementation_files': len(G.BOUND_INPUTS), 'first_eligible': task['id'], 'v2_verified': len(v2)}, ensure_ascii=False))
