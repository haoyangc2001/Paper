"""Record read-only preparation context; never optimize a frozen candidate."""
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
assert not (HERE / 'baseline.json').exists()
p = json.loads(PLAN.read_bytes())
states = {t['id']: t['status'] for t in p['tasks']}
assert next(t['id'] for t in p['tasks'] if not t['status'] and all(states[x] for x in t['depends_on'])) == 'NAV-065'
v2 = json.loads((ROOT / 'CaMRA/navigation/reports/nav064_terminal_tail_training_release_v1.json').read_bytes())['v2_verified_files']
for name, digest in v2.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
git = lambda *a: subprocess.check_output(['git', *a], cwd=ROOT, text=True).strip()
baseline = {'timestamp': datetime.now().astimezone().isoformat(), 'host': socket.getfqdn(),
    'tmux': os.environ['TMUX'], 'cwd': str(ROOT), 'python': sys.executable,
    'version': sys.version, 'git': git('rev-parse', 'HEAD'),
    'status': git('status', '--short'), 'plan': p,
    'v2_files_verified': len(v2),
    'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.free', '--format=csv,noheader'], text=True),
    'ssh': 'BatchMode SSH denied publickey; existing user-authorized same-host tiger11/tmux exception applies.'}
with (HERE / 'baseline.json').open('x') as f:
    json.dump(baseline, f, ensure_ascii=False, indent=2)
record = {'timestamp': baseline['timestamp'],
    'summary': 'NAV-065/1.3.1 v9开工：按legacy准备局部可拟合性诊断，主agent基本验证',
    'actions': ['完整读取AGENTS/legacy规则及计划，校验update并按数组确认NAV-065为首个可执行项；保留全部既有修改。',
        '复核v8结果及参考论文Phase II、共同停止和价值正负判据；现场核验tiger11/tmux、绝对解释器与33项v2身份。'],
    'files_changed': ['doc/plan/20261109_camra_navigation_implementation.json', str(HERE.relative_to(ROOT)) + '/'],
    'commands': ['tiger11/tmux nav064-r14-repair-20261115:nav065-v9-preparation；' + sys.executable + ' -B ' + str(Path(__file__))],
    'results': ['总目标仍为固定v2下同actor双critic可靠判别预算，再交NAV-066；上游NAV-064保留。',
        '允许隔离诊断/候选入口/基础测试；不改六actor、原共同停止标签、W、beta、obs-v7、架构、原v5 train/G4及冻结系统。',
        'v8单次尝试已结束；本轮先验证来源、误判分型、相邻真实预算及可区分性，无正式模型更新、采样或开发评分。',
        '本项验收为诊断可复核、实现安全接通且基本测试通过；不表示NAV-065通过，下游仍受原G4 FAIL阻塞。'],
    'evidence': ['CaMRA/navigation/reports/nav065_pilot_refit_v8_review.json', str(HERE / 'baseline.json')],
    'remaining_work': ['完成隔离v9只读诊断、可执行小拟合提案、基本测试及收尾记录。'],
    'blockers': ['新的局部拟合数字额度须经具体评审确认；不挪用v8未执行的边界额度。']}
# Generate a narrow append patch; apply_patch performs the actual plan mutation.
needle = '        }\n      ],\n   "test_reports": ['
text = PLAN.read_text()
start = text.index('"id": "NAV-065"')
offset = text.index(needle, start)
addition = ',\n' + json.dumps(record, ensure_ascii=False, indent=2)
replacement = '        }' + addition + '\n      ],\n   "test_reports": ['
before = text[offset:offset+len(needle)]
patch = '*** Begin Patch\n*** Update File: ' + str(PLAN) + '\n@@\n'
patch += '\n'.join('-' + x for x in before.splitlines()) + '\n'
patch += '\n'.join('+' + x for x in replacement.splitlines()) + '\n*** End Patch\n'
with (HERE / 'opening.patch').open('x') as f:
    f.write(patch)
print(json.dumps({k: v for k, v in baseline.items() if k not in ('plan', 'status', 'gpu')}, ensure_ascii=False))
