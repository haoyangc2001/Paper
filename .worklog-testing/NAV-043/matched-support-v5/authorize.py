"""Prepare the exact approval and append-only opening patch; no fit or launch."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import os
import subprocess
import sys

ROOT=Path('/home/caohy/repositories/Paper')
sys.path.insert(0,str(ROOT/'CaMRA/navigation/scripts'))
import nav043_matched_support_v5 as m

m.context()
p,ready=[m.diagonal.pin(x) for x in (m.PROPOSAL,m.READINESS)]
assert m.base.sha(m.PROPOSAL)=='71d5a31efe3890a713681d3e95600e87541563197e698c3a32446331a0e178d6'
assert m.base.sha(m.READINESS)=='364fe158b8bacedde45681137f8b4bb9e050560ead4ff5983a9962f8f1c0e982'
assert p['limits']==m.LIMITS
for mapping in (p['source_hashes'],ready['source_hashes']):
    for path,digest in mapping.items():
        assert m.base.sha(path)==digest,path
assert not m.APPROVAL.exists() and not Path(p['output_root']).exists() and not (m.HERE/'launch.lock').exists()
assert not m.REVIEW.exists()
dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).splitlines()
assert set(dirty)=={'?? .worklog-testing/NAV-043/matched-support-v5/'+x for x in ('opening.py','authorize.py')},dirty
assert 'AssertionError: ?? .worklog-testing/NAV-043/matched-support-v5/opening.py' in (m.HERE/'opening.log').read_text()
assert 'PRIOR_RECORD 69 ' in (m.HERE/'opening.log').read_text()
assert not (m.HERE/'opening.json').exists()
plan_path=ROOT/'doc/plan/20261109_camra_navigation_implementation.json'
plan=m.base.load(plan_path)
task=next(t for t in plan['tasks'] if t['id']=='NAV-043')
assert len(task['work_records'])==69 and not task['status']
states={t['id']:t['status'] for t in plan['tasks']}
assert sum(states.values())==55 and all(states[x] for x in task['depends_on'])
validation=subprocess.run([sys.executable,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',str(plan_path),'--mode','update'],cwd=ROOT,text=True,capture_output=True)
assert validation.returncode==0,validation.stdout+validation.stderr
now=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
approval=dict(schema='nav043-matched-support-approval-v5',status='APPROVED',task_id='NAV-043',
    plan_version='1.5.0',approved_at=now,user_response='可以  继续推进',
    authorization_context='Explicit current confirmation following the pending exact 18 CPU axis solves / 3600 seconds / old-data-only matched-support v5 proposal; no reuse of exhausted v4 allocation.',
    proposal_sha256=m.base.sha(m.PROPOSAL),readiness_sha256=m.base.sha(m.READINESS),
    limits=p['limits'],output_root=p['output_root'],environment=p['environment'],
    data_scope=p['data'],splits=p['splits'],fitting=p['fitting'],interpretation=p['interpretation'],
    once_only=p['once_only'],production_enabled=False,NAV043_passed=False,G5_passed=False,
    no_test_subagents=True,no_push=True)
text=json.dumps(approval,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
digest=m.base.hashlib.sha256(text.encode()).hexdigest()
record=dict(timestamp=now,summary='NAV-043/1.5.0同覆盖v5独立一次范围已确认，开工登记，先预检提交再执行',
    actions=['重新完整读取根AGENTS、legacy技能/执行规则、当前JSON全部任务及NAV-043的69条记录，核对README、交接、固定来源和当前依赖。',
      '用户在上轮明确18轴解/CPU1小时待确认后回复“可以  继续推进”，本次独立APPROVED绑定原提案71d5a31e/readiness364fe158，保留PROPOSED及原v3/v4失败。',
      '本项为运动模型适用性缺口提供相同数据覆盖的结构对照；三折各12导航+全部12短工况训练，6导航整图留出，原对角/耦合两结构同权重/时序。短工况仅训练内诊断，不当独立验证。',
      '%763开工脚本在已完成计划/来源核对后误将自己新增的opening.py视为外来改动；原源码/log/失败保留。本批准准备按精确允许文件集合核对，不改冻结模型、评分或封存执行源码。',
      '本次只允许一次6模型/18轴解/108保存指令预测，CPU累计3600秒；先实际预检并提交批准，再原launch.sh一次，最后主agent保存审阅，不安排测试子agent。'],
    files_changed=[str(m.APPROVAL.relative_to(ROOT)),str(m.APPROVAL.relative_to(ROOT))+'.sha256',str(plan_path.relative_to(ROOT)),'.worklog-testing/NAV-043/matched-support-v5/authorize.py','AGENTS.md','README.md','CaMRA/navigation/README.md','CaMRA/navigation/docs/implementation_handoff.md'],
    commands=['%763: matched-support-v5/opening.py，完整读取及来源核对，末尾自文件未跟踪断言失败，实际求解0',os.environ['TMUX_PANE']+': matched-support-v5/authorize.py -> 独立批准及追加开工patch，实际拟合0'],
    results=['原source pin全部一致，计划1.5.0、55/67、NAV-043=false，依赖NAV-066/067保持原状态。','本次不是NAV-043/G5通过；无生产消费者，不改RC-PPO、固定v2/功率/足迹/alpha/delta/锁、旧45秒actor双critic，90秒仍候选。','当前尚未启动新拟合；应用/物理与名义业务回合/W/网络更新/校准/GPU均0。'],
    evidence=['.worklog-testing/NAV-043/matched-support-v5/opening.log','.worklog-testing/NAV-043/matched-support-v5/authorize.json',str(m.APPROVAL.relative_to(ROOT)),str(m.PROPOSAL.relative_to(ROOT)),str(m.READINESS.relative_to(ROOT))],
    remaining_work=['实际预检、授权本地提交、原一次启动及全保存结果复核；108预测/260输出及18保存解均须核对。','按结果评审运动结构/支持域和长期目标；参考命令/J、独立90秒适用性及受影响学习/预算/物理门禁仍未完成。'],
    blockers=['本次只解除精确CPU对照范围授权阻塞，不授权新增物理、训练、全量部署拟合或签G5；实际适用性缺口保持。'])
task['work_records'].append(record)
new_plan=json.dumps(plan,ensure_ascii=False,indent=2)+'\n'
patch='*** Begin Patch\n'
for path,content in [(m.APPROVAL,text),(Path(str(m.APPROVAL)+'.sha256'),digest+'\n')]:
    assert not path.exists()
    patch+='*** Add File: '+str(path)+'\n'+''.join('+'+line+'\n' for line in content.splitlines())
old=plan_path.read_text().splitlines()
new=new_plan.splitlines()
import difflib
diff=list(difflib.unified_diff(old,new,n=3,lineterm=''))
patch+='*** Update File: '+str(plan_path)+'\n'
for line in diff[2:]:
    patch+=('@@' if line.startswith('@@') else line)+'\n'
updates=[('AGENTS.md','# Agent 接手说明','## 当前授权（2026-11-20，同覆盖v5一次CPU对照已确认）\n\n用户本轮回复“可以  继续推进”，独立APPROVED绑定原提案71d5a31e/readiness364fe158：原30已暴露病例、三折两结构、最多18轴解/CPU累计1小时、一次槽位。先实际预检并提交，再原matched-support-v5/launch.sh一次及主agent保存audit，不安排测试子agent、不推送。短工况全部为训练内诊断，导航整图留出但已暴露；不改旧门槛、不选部署模型或追加全量拟合。本时点真实求解0，应用/回合/W/更新/校准/GPU0；NAV-043=false、1.5.0/55/67、G5未过，固定v2/alpha/delta/45秒链/原失败保持。下方待确认为历史。'),
 ('README.md','# 四足机器人导航研发工作区','2026-11-20更新：用户已确认同覆盖v5一次CPU对照，独立批准后先实际预检/提交，再18轴解上限内运行与主agent保存复核；尚未求解，不新增回合或训练。NAV-043=false、55/67、G5未过；下方待确认为历史。'),
 ('CaMRA/navigation/README.md','# 上层最小成本到达避障导航器','2026-11-20更新：同覆盖v5一次范围已确认，独立APPROVED绑定原提案/支持清单，先实际预检并提交再原一次入口及保存审阅。当前尚未拟合；不新增物理/导航训练，不选部署模型；NAV-043=false、1.5.0/55/67、G5未过，旧失败保持。下方为历史。'),
 ('CaMRA/navigation/docs/implementation_handoff.md','# 上层导航实施交接','## 当前授权（2026-11-20，同覆盖v5已确认，尚未求解）\n\n用户明确回复“可以  继续推进”，独立APPROVED/pin绑定提案71d5a31e及readiness364fe158；先实际preflight.sh核验295/90/29身份、原83输出、资源和无旧进程/锁/输出，并本地提交，再原launch.sh一次，最后audit.sh保存复核。范围严格为原30已暴露病例、三折两原结构、18轴解/CPU累计3600秒，新应用/回合/W/网络更新/校准/GPU0；全部12短工况为训练内诊断，整图留出导航仍非新独立验证，不自动选模或追加全量拟合。\n\n%763开工末尾误把新opening.py当外来未跟踪文件，原源码/失败log保留；authorize.py只允许精确新增文件，未改封存执行源码或科学判定。后续结果以实际输出为准，不重复申请此次批准。计划1.5.0/55/67、NAV-043=false、G5未过，固定v2/alpha/delta/旧45秒链及旧FAIL/HELD保持。下方为历史。')]
for relative,title,header in updates:
    content=(ROOT/relative).read_text()
    assert content.startswith(title+'\n')
    patch+='*** Update File: '+str(ROOT/relative)+'\n@@\n '+title+'\n+\n'+''.join('+'+line+'\n' for line in header.splitlines())+' \n'
patch+='*** End Patch\n'
Path('/tmp/nav043-v5-authorization.patch').write_text(patch)
m.base.write_new(m.HERE/'authorize.json',dict(status='EXPLICIT_APPROVAL_PATCH_PREPARED',pane=os.environ['TMUX_PANE'],
    approval_sha256=digest,proposal_sha256=m.base.sha(m.PROPOSAL),readiness_sha256=m.base.sha(m.READINESS),
    prior_records=69,new_records=70,validation=validation.stdout,opening_self_file_false_positive_retained=True,
    real_fits=0,output_present=False,launch_present=False))
print('EXPLICIT_APPROVAL_PATCH_PREPARED',digest,flush=True)
