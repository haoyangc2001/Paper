"""Append result evidence and verify the final local commit. Never fit or push."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import difflib
import json
import os
import subprocess
import sys

ROOT=Path('/home/caohy/repositories/Paper')
sys.path.insert(0,str(ROOT/'CaMRA/navigation/scripts'))
import nav043_matched_support_v5 as m
import nav043_motion_probe_recovery_guard as guard

m.context()
p=m.approved()
closed=m.base.load(m.HERE/'close.json')
assert closed['status']=='MATCHED_SUPPORT_V5_ARCHIVE_AND_SCOPE_PASS'
assert closed['actual_axis_solves']==18 and closed['extra_parameter_solves']==0
report=m.base.load(m.REVIEW)
assert m.base.sha(m.REVIEW)==closed['review_sha256']
assert (m.HERE/'audit.exit').read_text().strip()=='0'
plan_name='doc/plan/20261109_camra_navigation_implementation.json'
plan_path=ROOT/plan_name
plan=m.base.load(plan_path)
task=next(t for t in plan['tasks'] if t['id']=='NAV-043')
doc_name='CaMRA/navigation/docs/nav043_matched_support_result_20261120.md'

if sys.argv[1:] == ['--record']:
    assert len(task['work_records'])==70 and not task['status']
    task['work_records'].append(dict(timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        summary='NAV-043/1.5.0同覆盖v5一次CPU对照与保存复核结束，短工况改善但长导航未见整体优势',
        actions=['用户本轮确认后独立APPROVED pin54e59adf绑定原提案71d5a31e/readiness364fe158；%765实际预检通过，295/90/29身份、30病例及原83输出核对，拟合0。',
            '%766首次提交检查因原始opening.log末尾空行失败，原脚本/log保留；%767只对原始日志保留字节，所有非日志文件继续严格diff检查，批准/预检先提交53f7aca，工作区干净。',
            '%768原一次launch.sh完成6模型/18轴解/108保存指令预测，拟合11.001秒、包装11.205秒，FINISHED、退出0、无HELD/重试。',
            '%769原audit.sh主agent核对18保存解法方程/谱、108全预测数组与指标/角色、260完整输出及启动账本通过；额外拟合0，无测试子agent。',
            '%770保存归档/来源复核通过，152小JSON逐字节另存Git，108NPZ外部原件/完整hash索引保持；汇总各工况和导航前缀，未做额外预测拟合或生产变更。',
            '更新结果文档与交接，保留原门槛/旧FAIL/HELD，不设新PASS门槛、不选部署模型、不追加全量拟合。按legacy核验追加记录和全局一致性后提交，不推送。'],
        files_changed=[str(m.REVIEW.relative_to(ROOT)),doc_name,'.worklog-testing/NAV-043/matched-support-v5',plan_name,'AGENTS.md','README.md','CaMRA/navigation/README.md','CaMRA/navigation/docs/implementation_handoff.md'],
        commands=['%765: matched-support-v5/preflight.sh -> APPROVED_MATCHED_SUPPORT_PREFLIGHT_PASS / exit0',
            '%767: prelaunch-commit-v2.py -> 53f7aca / WORKTREE_CLEAN',
            '%768: matched-support-v5/launch.sh -> 18 axis solves / FINISHED / exit0',
            '%769: matched-support-v5/audit.sh -> SAVED_MATCHED_SUPPORT_EXECUTION_AUDIT_PASS / exit0',
            '%770: matched-support-v5/close.py -> MATCHED_SUPPORT_V5_ARCHIVE_AND_SCOPE_PASS',
            'finish.py --commit在tmux核验71记录/全部任务定义状态/来源/归档并完成本地提交'],
        results=['同覆盖下短工况训练内位置RMSE：对角0.245565m、耦合0.149317m；末端航向MAE7.6829°/5.9273°。36记录/结构来自12工况三折复用，不作独立验证。',
            '留出导航18条位置RMSE1.015271m/1.026430m，末端航向MAE14.6483°/13.5504°；L2耦合稍好、L1/L3位置稍差，无整体择优部署依据。',
            '同18条导航5/10/20/30秒前缀位置RMSE约0.068/0.137/0.320/0.572m及0.067/0.136/0.330/0.592m；45秒仅13条，不能直接同分母比较。训练内M02/M03仍有明显残差，不能仅归因覆盖或唯一物理机制。',
            '执行完整性通过且全部260原件/12388563字节保持；新应用/物理与名义业务回合/W/网络更新/校准/GPU0。固定v2/旧alpha/delta/功率/足迹/锁/45秒actor双critic和RC-PPO不变，无生产消费者。',
            '本次一次18解已用完，不借旧额度或重试；NAV-043=false、1.5.0/55/67、G5未过，90秒/obs-v9仍候选。'],
        evidence=[str(m.REVIEW.relative_to(ROOT)),doc_name,'.worklog-testing/NAV-043/matched-support-v5/close.json','.worklog-testing/NAV-043/matched-support-v5/launch.json','.worklog-testing/NAV-043/matched-support-v5/exit.json','.worklog-testing/NAV-043/matched-support-v5/prelaunch-commit-v1-failed.log','.worklog-testing/NAV-043/matched-support-v5/final.json'],
        remaining_work=['优先依据保存输出评审长期预测目标、低速/转向状态和结构，再冻结单一候选/独立验证协议，不默认加数据、复杂结构或上层重训。',
            '参考命令与J、独立90秒残余适用性、受影响训练/价值预算/物理链及NAV-043自身预登记放行标准尚未满足。'],
        blockers=['现有训练内改善不能作为长导航/90秒适用性或G5放行；无新拟合、物理或校准配额，不自动扩展本次已结束范围。']))
    old=plan_path.read_text().splitlines()
    new=(json.dumps(plan,ensure_ascii=False,indent=2)+'\n').splitlines()
    patch='*** Begin Patch\n*** Update File: '+str(plan_path)+'\n'
    for line in list(difflib.unified_diff(old,new,n=3,lineterm=''))[2:]:
        patch+=('@@' if line.startswith('@@') else line)+'\n'
    header='## 当前结果（2026-11-20，同覆盖v5对照已保存复核）\n\n批准/实际预检先提交53f7aca，%768一次完成6模型/18轴解/108保存指令预测，拟合11.001秒、退出0、无HELD；%769/%770主agent保存复核/归档通过，额外求解0，无测试子agent。同数据下短工况训练内位置RMSE对角/耦合0.246/0.149米，留出导航1.015/1.026米，长导航没有整体优势；短工况三折复用不当独立验证。260原件及152小JSON/108外置NPZ索引保留，旧FAIL/HELD保持。见[结果与后续依据](CaMRA/navigation/docs/nav043_matched_support_result_20261120.md)。\n\n本次单次18解范围已结束，不默认重试、加拟合或训练。下一优先评审长期目标/低速转向状态，再明确新候选协议；参考J、独立90秒及受影响生产门禁仍未完成。新增应用/业务回合/W/网络更新/校准/GPU0，固定v2/alpha/delta/旧45秒链不改；NAV-043=false、1.5.0/55/67、G5未过。下方为历史。'
    titles={'AGENTS.md':'# Agent 接手说明','README.md':'# 四足机器人导航研发工作区','CaMRA/navigation/README.md':'# 上层最小成本到达避障导航器','CaMRA/navigation/docs/implementation_handoff.md':'# 上层导航实施交接'}
    for name,title in titles.items():
        text=header
        if name=='CaMRA/navigation/README.md':
            text=text.replace('CaMRA/navigation/docs/','docs/')
        if name=='CaMRA/navigation/docs/implementation_handoff.md':
            text=text.replace('CaMRA/navigation/docs/','')
        assert (ROOT/name).read_text().startswith(title+'\n')
        patch+='*** Update File: '+str(ROOT/name)+'\n@@\n '+title+'\n+\n'+''.join('+'+line+'\n' for line in text.splitlines())+' \n'
    patch+='*** End Patch\n'
    Path('/tmp/nav043-v5-result-record.patch').write_text(patch)
    print('RESULT_RECORD_PATCH_PREPARED 70 -> 71',flush=True)
elif sys.argv[1:] == ['--commit']:
    assert len(task['work_records'])==71 and not task['status']
    prior=json.loads(subprocess.check_output(['git','show','HEAD:'+plan_name],cwd=ROOT,text=True))
    assert {k:v for k,v in prior.items() if k!='tasks'}=={k:v for k,v in plan.items() if k!='tasks'}
    for a,b in zip(prior['tasks'],plan['tasks'],strict=True):
        assert {k:v for k,v in a.items() if k!='work_records'}=={k:v for k,v in b.items() if k!='work_records'}
        assert b['work_records'][:len(a['work_records'])]==a['work_records']
        if a['id']=='NAV-043':
            assert len(a['work_records'])==70
        else:
            assert a==b
    assert sum(t['status'] for t in plan['tasks'])==55
    for entry in report['output_index']:
        path=Path(entry['path'])
        assert path.stat().st_size==entry['bytes'] and m.base.sha(path)==entry['sha256']
        if path.suffix=='.json':
            assert m.base.sha(m.HERE/'saved-small-output'/path.relative_to(Path(p['output_root'])))==entry['sha256']
    assert guard.protected_sources()==closed['identities']
    hits=subprocess.run(['rg','-l','nav043_matched_support_v5','CaMRA/navigation','--glob','*.py'],cwd=ROOT,text=True,capture_output=True)
    assert hits.returncode==0
    assert hits.stdout.splitlines()==['CaMRA/navigation/scripts/nav043_matched_support_v5.py'],hits.stdout
    launched=m.base.load(m.HERE/'launch.json')
    assert all(not Path('/proc',str(launched[key])).exists() for key in ('wrapper_pid','child_pid'))
    validation=subprocess.run([sys.executable,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',str(plan_path),'--mode','update'],cwd=ROOT,text=True,capture_output=True)
    assert validation.returncode==0,validation.stdout+validation.stderr
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
    paths=['AGENTS.md','README.md','CaMRA/navigation/README.md','CaMRA/navigation/docs/implementation_handoff.md',plan_name,doc_name,str(m.REVIEW.relative_to(ROOT))]
    prefix=str(m.HERE.relative_to(ROOT))+'/'
    changed=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines()
    assert all(x in paths or x.startswith(prefix) for x in changed),changed
    m.base.write_new(m.HERE/'final.json',dict(status='MATCHED_SUPPORT_V5_APPEND_ONLY_AND_GLOBAL_REVIEW_PASS',pane=os.environ['TMUX_PANE'],
        original_records=70,current_records=71,all_task_definitions_and_states_preserved=True,
        validation=validation.stdout,source_pins_rechecked=True,output_files_verified=260,small_JSON_verified=152,
        protected_identities=closed['identities'],no_production_consumers=True,extra_parameter_solves=0,
        NAV043=False,G5_passed=False,prelaunch_commit=closed['prelaunch_commit']))
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
    subprocess.run(['git','add','-f','--',str(m.HERE.relative_to(ROOT))],cwd=ROOT,check=True)
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
    assert set(paths).issubset(staged)
    assert all(x in paths or x.startswith(prefix) for x in staged),staged
    assert not any(x.endswith(('.npz','.pyc')) or '__pycache__' in x for x in staged)
    subprocess.run(['git','diff','--cached','--check','--',*[x for x in staged if not x.endswith('.log')]],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','NAV-043: preserve matched-support v5 results and long-horizon limitations'],cwd=ROOT,check=True)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
    assert not status,status
    print('RESULT_LOCAL_COMMIT',revision,'FILES',len(staged),'WORKTREE_CLEAN',flush=True)
else:
    raise SystemExit('use --record or --commit; neither repeats fits or audit')
