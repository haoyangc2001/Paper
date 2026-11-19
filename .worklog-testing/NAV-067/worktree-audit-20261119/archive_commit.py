"""Preserve historical evidence, archive large raw reports, commit the missing small files."""
import ast
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'CaMRA/navigation/scripts'))
from nav067_sources import load, sha, write_new

def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()

assert os.environ.get('TMUX_PANE')=='%599'
baseline=load(ROOT/'.worklog-testing/NAV-067/remaining-recovery-v2-execution/initial-untracked-inventory.json')
archive=Path('/pub/data/caohy/work/camra_navigation/historical_worktree_archive_20261119_v1')
archive.mkdir(exist_ok=True)
tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'))
rows=[]
small=[]
excluded=[]
checks={'JSON_parsed':0,'Python_syntax_parsed':0,'historical_bytes_unchanged':0}
format_failures=[]
sensitive=[]
patterns=[re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),re.compile(r'\b(?:sk-proj-|ghp_|github_pat_)[A-Za-z0-9_-]{24,}')]
for item in baseline['files']:
    name=item['path']
    if name in tracked:
        continue
    path=ROOT/name
    assert name.startswith(('.worklog-testing/NAV-065/','.worklog-testing/NAV-066/','.worklog-testing/NAV-067/','CaMRA/navigation/reports/','doc/plan/20261109_camra_navigation_implementation.json.bak_20261115')),name
    assert path.is_file() and sha(path)==item['sha256'],name
    checks['historical_bytes_unchanged']+=1
    content=path.read_text()
    for pattern in patterns:
        if pattern.search(content):
            sensitive.append(name)
    if path.suffix=='.json':
        try:
            value=json.loads(content)
            checks['JSON_parsed']+=1
        except json.JSONDecodeError as error:
            format_failures.append({'path':name,'kind':'historical non-JSON or empty failure artifact','error':str(error),'sha256':item['sha256']})
    elif path.suffix=='.py':
        ast.parse(content,filename=name)
        checks['Python_syntax_parsed']+=1
    row=dict(item)
    if item['bytes']>10_000_000 or '.bak_' in path.name:
        destination=archive/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():
            with path.open('rb') as src,destination.open('xb') as dest:
                shutil.copyfileobj(src,dest,1024*1024)
        assert sha(destination)==item['sha256'],name
        row.update(disposition='EXTERNAL_ARCHIVE_WITH_TRACKED_INDEX_ORIGINAL_RETAINED',archive_path=str(destination))
        excluded.append(name)
    else:
        row.update(disposition='COMMIT_HISTORICAL_EVIDENCE_UNCHANGED')
        small.append(name)
    rows.append(row)
assert not sensitive,{'potential_sensitive_files':sensitive}
assert len(excluded)==8,excluded
manifest={'schema':'camra-historical-untracked-archive-v1','at':now(),'task':'NAV-067',
    'source_inventory':'.worklog-testing/NAV-067/remaining-recovery-v2-execution/initial-untracked-inventory.json',
    'scope':'Previously untracked historical NAV-064/065/066/067 evidence only; not a new scientific result or frozen-source change.',
    'archive_root':str(archive),'files':rows,'counts':{'reviewed':len(rows),'commit_small':len(small),'external_archive':len(excluded)},
    'checks':checks,'historical_format_failures_preserved':format_failures,'all_originals_retained':True,'large_raw_reports':7,'backup_snapshots_externally_archived':1,
    'no_helper_or_historical_launcher_executed':True,'new_training':0,'W_queries':0,'push':False}
write_new(archive/'manifest.json',manifest)
index=ROOT/'CaMRA/navigation/reports/historical_worktree_archive_20261119_v1.json'
write_new(index,manifest)
write_new(HERE/'verification.json',{'at':now(),'status':'HISTORICAL_ARCHIVE_INTEGRITY_PASS_FORMAT_FAILURES_RETAINED','counts':manifest['counts'],'checks':checks,'historical_format_failures_preserved':format_failures,
    'archive_manifest_sha256':sha(archive/'manifest.json'),'tracked_index_sha256':sha(index),'all_source_bytes_preserved':True,
    'note':'Syntax/JSON checks validate storage integrity only; historical failures remain failures.'})
ignore=ROOT/'.gitignore'
before=ignore.read_text()
addition='\n# NAV历史大体积原始明细及旧备份：保留原件，外部归档和hash见\n# CaMRA/navigation/reports/historical_worktree_archive_20261119_v1.json\n'+''.join('/'+name+'\n' for name in excluded)
last=before.splitlines()[-1]
patch='*** Begin Patch\n*** Update File: '+str(ignore)+'\n@@\n '+last+'\n'+''.join('+'+s+'\n' for s in addition.splitlines())+'*** End Patch\n'
subprocess.run(['apply_patch'],input=patch,text=True,cwd=ROOT,check=True)
doc=ROOT/'CaMRA/navigation/docs/historical_untracked_archive_20261119.md'
doc_text='工作区未跟踪材料已逐项核对。此前主要实现已提交，但部分历史报告、核查脚本和失败证据遗漏了Git归档，本次补齐。\n\n'+str(len(small))+'份小文件按原字节补交，包含历史失败、脚本、工作记录补丁及计划快照；它们是历史证据，不能作为新生产入口执行。7份70–94 MB逐步明细和1份旧计划备份已逐字节复制到`'+str(archive)+'`，原文件保留，只对这8条精确路径新增Git忽略规则。Git保存完整路径、大小、SHA256和外部归档位置，见[归档索引](../reports/historical_worktree_archive_20261119_v1.json)。\n\n原始明细不直接放入Git，避免单次增加约567 MB；汇总、引用、原实验源码及全部失败继续保留。需要恢复时从索引指定归档路径复制到原相对路径并核对SHA256，不改旧锁。未删除原文件、未执行旧脚本、未改旧结果、未新增训练或查询、未推送。\n'
subprocess.run(['apply_patch'],input='*** Begin Patch\n*** Add File: '+str(doc)+'\n'+''.join('+'+s+'\n' for s in doc_text.splitlines())+'*** End Patch\n',text=True,cwd=ROOT,check=True)
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
assert not subprocess.check_output(['git','diff','--name-only','--','manuscripts','*.tex','*.bib','RC-PPO','HiRA'],cwd=ROOT,text=True).strip()
paths=small+['.gitignore',str(index.relative_to(ROOT)),str(doc.relative_to(ROOT)),str(Path(__file__).relative_to(ROOT)),str((HERE/'verification.json').relative_to(ROOT))]
subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
# Historical patches/logs are preserved byte-for-byte; whitespace checks apply
# to source-facing docs, index, and ignore policy, not the old raw evidence.
subprocess.run(['git','diff','--cached','--check','--','.gitignore',str(doc.relative_to(ROOT)),str(index.relative_to(ROOT))],cwd=ROOT,check=True)
subprocess.run(['git','commit','-m','chore(nav): archive missing historical evidence and index large reports'],cwd=ROOT,check=True)
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
print(json.dumps({'commit':head,'counts':manifest['counts'],'checks':checks,'archive':str(archive),'source_bytes_preserved':True},ensure_ascii=False),flush=True)
