"""Render append-only preparation acceptance and pending-scope record."""
import difflib
import json
from datetime import datetime
from pathlib import Path

p = Path('doc/plan/20261109_camra_navigation_implementation.json')
old = p.read_text()
j = json.loads(old)
t = next(t for t in j['tasks'] if t['id']=='NAV-043')
assert len(t['work_records']) == 54
r = dict(timestamp=datetime.now().astimezone().isoformat(),
    summary='NAV-043响应v3隔离准备：9矩阵支持及13基本检查通过，精确9次CPU诊断提案待批准，真实拟合0',
    actions=['新增低速连续分段/命令交叉响应/对角q记忆候选和一次性CPU入口，零q零指令保持零；不接旧生产链、不改冻结v2和RC-PPO目标。',
             '%705只读复用18导航/158797tick及12工况/24000测量tick，9加权设计矩阵秩6、条件数51.1～88.3，不求解目标参数。',
             '%706主agent13项合成/保护基本检查通过，3次合成轴求解；无专用测试子agent。首次静态提案生成误用无NumPy的系统Python，改用登记绝对解释器后生成；没有科学运行。',
             '封存PROPOSED a3b67ab3和PREPARED_ONLY 965fbf8e；%707核验来源、295/90/29身份、67任务定义/状态与53条旧记录前缀，实际缺批准入口exit1且未创建marker/输出。',
             '同步AGENTS、根/导航README及交接；原严格HELD/全部原数据/失败保留，未为退出账本重采。'],
    files_changed=['CaMRA/navigation/scripts/nav043_supported_response.py',
                   'CaMRA/navigation/scripts/launch_nav043_supported_response_once.py',
                   'CaMRA/navigation/tests/test_supported_response_v3.py',
                   'CaMRA/navigation/config/nav043_supported_response_proposal_v3.json',
                   'CaMRA/navigation/config/nav043_supported_response_readiness_v3.json',
                   'CaMRA/navigation/docs/nav043_supported_response_preparation_20261120.md',
                   '.worklog-testing/NAV-043/motion-candidate-v3/',
                   'AGENTS.md','README.md','CaMRA/navigation/README.md',
                   'CaMRA/navigation/docs/implementation_handoff.md',str(p)],
    commands=['tiger11/tmux %705: bash .worklog-testing/NAV-043/motion-candidate-v3/support.sh -> exit0',
              'tiger11/tmux %706: bash .worklog-testing/NAV-043/motion-candidate-v3/basic.sh -> exit0, 13 tests',
              'tiger11/tmux %707: bash .worklog-testing/NAV-043/motion-candidate-v3/verify.sh -> exit0, legacy VALID and diff check'],
    results=['局部工程准备通过：9个秩检查和13基本检查有效；参数可辨识仅指当前列秩，不能证明q马尔可夫性或全域有效。',
             '全局方法一致性保持，生产消费者0、旧45秒/actor双critic不改，90秒/obs-v9仍候选，不能据此给NAV-043/G5放行。',
             '独立提案复用30个已暴露病例，三折/最多9次CPU求解/1小时，新应用/回合/GPU/W/网络更新/校准0；不全量追加拟合、不扫描超参/不选部署模型。',
             'APPROVED、正式输出和launch.lock均不存在；本轮真实数据求解/新应用/导航/W/更新/校准0。',
             '计划1.5.0、55/67历史true、NAV-043=false；四项实际导航/适用性验收均未解除。'],
    evidence=['.worklog-testing/NAV-043/motion-candidate-v3/preflight.json',
              '.worklog-testing/NAV-043/motion-candidate-v3/support.json',
              '.worklog-testing/NAV-043/motion-candidate-v3/basic.log',
              '.worklog-testing/NAV-043/motion-candidate-v3/verify.json',
              '.worklog-testing/NAV-043/motion-candidate-v3/plan-update.log',
              'CaMRA/navigation/config/nav043_supported_response_proposal_v3.json',
              'CaMRA/navigation/config/nav043_supported_response_readiness_v3.json'],
    remaining_work=['收尾复核并本地提交全部准备/证据/计划，不推送。',
                    '向用户确认精确的新9次CPU求解/1小时范围；取得批准后独立登记、预检及先提交，再唯一运行并由主agent无额外求解复核保存输出。',
                    '依诊断结果评审模型/支持域，再具体制定可执行参考程序/J、独立90秒残余适用性及受影响训练/部署门禁。'],
    blockers=['旧采集批准不含新科学拟合，新PROPOSED尚无独立APPROVED；不能重启旧分配或自动借余量。',
              '仍缺新版实际导航/90秒残余适用性、生产接入及G5证据；不进入NAV-044。'])
start = old.index('[', old.index('"work_records":', old.index('"id": "NAV-043"')))
_, size = json.JSONDecoder().raw_decode(old[start:]); end = start + size - 1
record = '\n'.join('        '+s for s in json.dumps(r,ensure_ascii=False,indent=2).splitlines())
new = old[:end].rstrip()+',\n'+record+'\n      '+old[end:]
d = list(difflib.unified_diff(old.splitlines(True),new.splitlines(True)))
print('*** Begin Patch\n*** Update File: '+str(p))
for line in d[2:]: print('@@' if line.startswith('@@') else line, end='\n' if line.startswith('@@') else '')
print('*** End Patch')
