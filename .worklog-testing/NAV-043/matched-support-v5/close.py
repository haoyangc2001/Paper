"""Archive and summarize audited outputs, without additional parameter solving."""
from pathlib import Path
import os
import subprocess
import sys

ROOT=Path('/home/caohy/repositories/Paper')
sys.path.insert(0,str(ROOT/'CaMRA/navigation/scripts'))
import nav043_matched_support_v5 as m
import nav043_motion_probe_recovery_guard as guard

m.context()
p=m.approved()
assert (m.HERE/'audit.exit').read_text().strip()=='0'
assert (m.HERE/'wrapper.exit').read_text().strip()=='0'
review=m.base.load(m.REVIEW)
assert review['status']=='SAVED_MATCHED_SUPPORT_EXECUTION_AUDIT_PASS'
assert review['actual_axis_solves']==18 and review['extra_fits']==0
output=Path(p['output_root'])
finished=m.base.load(output/'FINISHED.json')
launched=m.base.load(m.HERE/'launch.json')
exited=m.base.load(m.HERE/'exit.json')
for name in ('wrapper_pid','child_pid'):
    assert not Path('/proc',str(launched[name])).exists(),name
assert exited['finished'] and not exited['held'] and exited['exit_code']==0
assert not (output/'HELD.json').exists()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()=='53f7aca31e19eda4165d4cdcbf296f840272cf5e'
assert subprocess.check_output(['git','show','HEAD:'+str(m.APPROVAL.relative_to(ROOT))],cwd=ROOT)==m.APPROVAL.read_bytes()
identities=guard.protected_sources()
assert identities==dict(protected_identities=295,existing_raw_input_hashes=90,bundle_files=29)
old=m.base.load(m.base.NAV/'reports/nav043_coupled_response_review_v4.json')
for entry in old['output_index']:
    path=Path(entry['path'])
    assert path.stat().st_size==entry['bytes'] and m.base.sha(path)==entry['sha256']
assert len(old['output_index'])==83
assert len(review['output_index'])==260
copied=0
for entry in review['output_index']:
    path=Path(entry['path'])
    assert path.stat().st_size==entry['bytes'] and m.base.sha(path)==entry['sha256']
    if path.suffix=='.json':
        target=m.HERE/'saved-small-output'/path.relative_to(output)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:
            stream.write(path.read_bytes())
        assert m.base.sha(target)==entry['sha256']
        copied+=1
assert copied==152
assert len(list((m.HERE/'saved-small-output').rglob('*.json')))==152
rows=finished['rows']
probe_groups={}
prefix_groups={}
for kind in m.KINDS:
    subset=[r for r in rows if r['structure']==kind]
    probes=[r for r in subset if r['evaluation_role']=='in_sample_probe']
    nav=[r for r in subset if r['evaluation_role']=='held_navigation']
    assert len(probes)==36 and len(nav)==18
    probe_groups[kind]={}
    for probe in sorted({r['kind'] for r in probes}):
        group=[r for r in probes if r['kind']==probe]
        assert len(group)==3
        probe_groups[kind][probe]=dict(correlated_fold_rows=3,unique_cases=1,
            position_RMSE_m=float(m.np.sqrt(sum(r['metrics']['position_squared_error_sum'] for r in group)/sum(r['metrics']['ticks'] for r in group))),
            terminal_heading_MAE_deg=float(m.np.degrees(m.np.mean([abs(r['metrics']['heading_terminal_rad']) for r in group]))))
    prefix_groups[kind]={}
    for seconds in ('5','10','20','30','45'):
        group=[r['prefixes'][seconds] for r in nav if seconds in r['prefixes']]
        if group:
            prefix_groups[kind][seconds]=dict(available_rows=len(group),
                position_RMSE_m=float(m.np.sqrt(sum(x['position_squared_error_sum'] for x in group)/sum(x['ticks'] for x in group))),
                terminal_heading_MAE_deg=float(m.np.degrees(m.np.mean([abs(x['heading_terminal_rad']) for x in group]))))
stability={}
for level in m.base.LEVELS:
    for kind in m.KINDS:
        model=m.base.load(output/level/kind/'model.json')
        stability[level+'/'+kind]=model['stability']
closed=dict(status='MATCHED_SUPPORT_V5_ARCHIVE_AND_SCOPE_PASS',pane=os.environ['TMUX_PANE'],
    prelaunch_commit='53f7aca31e19eda4165d4cdcbf296f840272cf5e',identities=identities,
    actual_axis_solves=18,extra_parameter_solves=0,new_applications=0,new_business_episodes=0,
    W_queries=0,actor_critic_updates=0,calibration_fits=0,GPU_used=False,
    fitted_cpu_wall_s=finished['cpu_wall_s'],wrapper_wall_s=exited['wrapper_wall_s'],
    original_cases=30,descriptive_prediction_rows=108,output_files=260,
    output_bytes=sum(x['bytes'] for x in review['output_index']),copied_small_JSON=152,
    external_NPZ=108,old_v4_outputs_preserved=83,processes_exited=True,
    review_sha256=m.base.sha(m.REVIEW),descriptive_summary=finished['descriptive_summary'],
    in_sample_probe_groups=probe_groups,held_navigation_prefixes=prefix_groups,stability=stability,
    old_FAIL_and_HELD_unchanged=True,new_PASS_gate=False,deployment_model_selected=False,
    independent_validation=False,NAV043=False,G5_passed=False)
m.base.write_new(m.HERE/'close.json',closed)
print('MATCHED_SUPPORT_V5_ARCHIVE_AND_SCOPE_PASS',flush=True)
print(m.base.json.dumps({k:closed[k] for k in ('fitted_cpu_wall_s','output_files','output_bytes','copied_small_JSON','external_NPZ','descriptive_summary','in_sample_probe_groups','held_navigation_prefixes')},ensure_ascii=False,indent=2),flush=True)
