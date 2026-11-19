import ast
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess

from nav067_sources import NAV, ROOT, OUTPUT, read_manifest, sha, write_new

out=ROOT/'.worklog-testing/NAV-067/entry-v1'
manifest,prereg,nets=read_manifest()
assert len(manifest['maps'])==3 and len(manifest['candidates'])==2
assert len({r['level'] for r in manifest['maps']})==3
assert all(r['seed']==3001 and r['z0_J']>0 and r['budget_mode']=='common_budget' for r in manifest['maps'])
assert '24 passed' in Path('/tmp/nav067_binding_tests_v2.log').read_text()
base=json.loads(subprocess.check_output(['git','show','HEAD:doc/plan/20261109_camra_navigation_implementation.json'],cwd=ROOT))
plan=json.loads((ROOT/'doc/plan/20261109_camra_navigation_implementation.json').read_bytes())
for a,b in zip(base['tasks'],plan['tasks'],strict=True):
    assert a['id']==b['id'] and a['status']==b['status'] and a['depends_on']==b['depends_on']
    assert b['work_records'][:len(a['work_records'])]==a['work_records']
    if a['id']!='NAV-067': assert a==b
paths=['scripts/nav067_sources.py','scripts/evaluate_navigation_v2.py','tests/test_nav067_binding.py']
for rel in paths: ast.parse((NAV/rel).read_text(),filename=rel)
for source,name in [('/tmp/nav067_binding_tests_v2.log','basic-tests.log'),
                    ('/tmp/nav067_pilot_prepare_v1.log','prepare.log'),
                    ('/tmp/nav067_binding_identity.log','legacy-format-failure.log')]:
    target=out/name
    assert not target.exists()
    shutil.copyfile(source,target)
report={'task':'NAV-067','status':'BOUND_READY_FOR_PHYSICAL_PILOT_NOT_ACCEPTED',
        'timestamp':datetime.now().astimezone().isoformat(),
        'basic_tests':24,'binding':manifest['binding'],
        'manifest_path':str(OUTPUT/'manifest.json'),'manifest_sha256':sha(OUTPUT/'manifest.json'),
        'new_sources':{p:sha(NAV/p) for p in paths},
        'selected_pair':[{k:c[k] for k in ('variant','seed','run_id','checkpoint_sha256')} for c in manifest['candidates']],
        'pilot_maps':[{'map_id':r['map_id'],'common_z0_J':r['z0_J']} for r in manifest['maps']],
        'allocated_physical_episodes':6,'executed_physical_episodes':0,'W_queries':0,'training_updates':0,
        'legacy_protocol_format_issue':'Original pinned file contains arithmetic in an unrelated JSON total. Retained unchanged; only unique pinned clean_6_episodes scalar is read, without expression evaluation.',
        'limitations':['No real Isaac pilot yet.','Ground/material/reset/command/energy readback requires actual execution.',
                      'Controlled actual cases and same-configuration repetition still need an explicit independent allocation.',
                      'NAV-067 remains false; NAV-043 and all final_* remain unopened.']}
write_new(NAV/'reports/nav067_pilot_binding_v1.json',report)
print(json.dumps(report,ensure_ascii=False,indent=2))
