"""Print new proposal/readiness patches; never authorize or execute recovery."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path('/home/caohy/repositories/Paper')
NAV=ROOT/'CaMRA/navigation'
E=ROOT/'.worklog-testing/NAV-043/motion-recovery-v2'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text())
def rendered(obj): return json.dumps(obj,ensure_ascii=False,indent=2)+'\n'
old=load(NAV/'config/nav043_motion_probe_proposal_v1.json')
failure=load(NAV/'reports/nav043_motion_probe_first_failure_v1.json')
assert failure['status']=='FAILED_CAPTURE_RETAINED'
assert (E/'basic.exit').read_text().strip()=='0'
suites=ET.parse(E/'basic.xml').getroot().findall('testsuite')
assert sum(int(s.attrib['tests']) for s in suites)==20
assert all(int(s.attrib[k])==0 for s in suites for k in ('failures','errors','skipped'))
sources=dict(old['source_hashes'])
for name in ('config/nav043_motion_probe_proposal_v1.json',
             'config/nav043_motion_probe_approval_v1.json',
             'config/nav043_motion_probe_readiness_v1.json',
             'reports/nav043_motion_probe_first_failure_v1.json'):
    path=NAV/name;sources[str(path.relative_to(ROOT))]=sha(path)
p={**old,'schema':'nav043-motion-probe-recovery-proposal-v2',
   'approval_path':'CaMRA/navigation/config/nav043_motion_probe_recovery_approval_v2.json',
   'purpose':'One isolated recovery of the failed capture: original M01 retained, recapture M01 then unstarted M02-M12. No navigation, fit, training or calibration.',
   'execution_readiness':'ISOLATED_IMPLEMENTATION_BASIC_CPU_CHECKED_NOT_APPROVED',
   'output_root':str(Path(old['output_root'])/'recovery_v2'),
   'limits':{**old['limits'],'gpu_wall_s':6300,
             'all_explicit_simulation_step_calls':31105,
             'setup_reset_reserve_step_calls':4705},
   'recovery':{'source_proposal_sha256':sha(NAV/'config/nav043_motion_probe_proposal_v1.json'),
      'first_failure_review_sha256':sha(NAV/'reports/nav043_motion_probe_first_failure_v1.json'),
      'original_output_root':old['output_root'],
      'prior_applications_started':1,'prior_controlled_cases_started':1,
      'prior_completed_cases':0,'prior_explicit_calls':895,
      'prior_reported_launcher_wall_s':failure['launcher_exit']['wall_s'],
      'failed_slot_wall_charge_s':900,
      'wall_charge_basis':'Conservatively charge 900s from original OWNER time to saved review confirming both PIDs absent (less than 900s); original cleanup wall gap is not called zero.',
      'cumulative_applications_cap':2,'cumulative_controlled_cases_started_cap':13,
      'cumulative_explicit_step_cap':32000,'cumulative_wall_cap_s':7200,
      'M01_identity':'original incomplete M01 retained; recovery_v2/M01 is its single second attempt',
      'remaining_cases':'M02-M12 exactly once, unchanged order and commands',
      'no_further_retry':True},
   'capture':{**old['capture'],
      'clock_v2':'Save raw actual simulator time and integer current_time_step_index at reset/every substep. Check consecutive index and registered float32/float64 callback representations; nominal J/q dt stays .005.',
      'terminal_v2':'Save primary application ERROR independently; parent HELD and process_cleanup retain cleanup timeouts and unconfirmed process state without fabricating missing child records.'},
   'stop_rules':{**old['stop_rules'],
      'wall_clock':'At most one additional application, 6300s including startup/cleanup, plus prior slot charged 900s; cumulative cap 7200s unchanged. No automatic retry or alternate root.',
      'resource_guard':'New explicit cap31105 and reserve4705, plus prior895 within cumulative32000. Native internal initialization steps remain unobservable, not zero.'},
   'implementation_before_launch':[
      'Obtain separate recovery APPROVED binding exact proposal/readiness, environment/schedule/limits/recovery; preserve original proposal/approval/data/failure.',
      'Recheck fixed tiger11/tmux/Python/GPU6 UUID, protected inputs and original PID absence; commit approval before one launch.',
      'Use only run_nav043_motion_probe_recovery_v2.py and the sealed recovery_v2 root, retain first failure and stop.',
      'Main agent reviews saved actual clock/index/state/command/q/J/accounting; basic CPU preparation is not physical or G5 acceptance.'],
   'source_hashes':sources}
p_path=NAV/'config/nav043_motion_probe_recovery_proposal_v2.json'
p_text=rendered(p);p_sha=hashlib.sha256(p_text.encode()).hexdigest()
support=load(NAV/'config/nav043_motion_probe_readiness_v1.json')['source_hashes'].copy()
new_names=[
 'src/camra_navigation/envs/simulation_clock_v2.py',
 'src/camra_navigation/envs/motion_capture_v2.py',
 'scripts/nav043_process_cleanup_v2.py',
 'scripts/nav043_motion_probe_recovery_guard.py',
 'scripts/nav043_motion_probe_recovery_simulation.py',
 'scripts/run_nav043_motion_probe_recovery_v2.py',
 'scripts/audit_nav043_motion_probe_recovery_v2.py',
 'tests/test_motion_recovery_v2.py']
for name in new_names:
    path=NAV/name;support[str(path.relative_to(ROOT))]=sha(path)
for name in ('basic.sh','basic.log','basic.xml','basic.exit','launch-once.sh',
             'prepare-documents.py','verify.py','verify.sh'):
    path=E/name;support[str(path.relative_to(ROOT))]=sha(path)
timer=Path(failure['clock']['installed_timer_source'])
assert sha(timer)==failure['clock']['installed_timer_sha256']
support[str(timer)]=sha(timer)
r={'schema':'nav043-motion-probe-recovery-readiness-v2','status':'PREPARED_ONLY',
   'proposal_sha256':p_sha,
   'basic_verification':{'count':20,'evidence':str((E/'basic.xml').relative_to(ROOT)),
      'pane':'%688','main_agent_only':True,'physical_execution_proven':False},
   'source_hashes':support,'approval_exists':False,
   'new_physical_applications':0,'NAV043_passed':False,'G5_passed':False}
r_path=NAV/'config/nav043_motion_probe_recovery_readiness_v2.json'
r_text=rendered(r);r_sha=hashlib.sha256(r_text.encode()).hexdigest()
assert not (NAV/'config/nav043_motion_probe_recovery_approval_v2.json').exists()
assert not Path(p['output_root']).exists() and not (E/'launch.lock').exists()
files={p_path:p_text,Path(str(p_path)+'.sha256'):p_sha+'\n',
       r_path:r_text,Path(str(r_path)+'.sha256'):r_sha+'\n'}
print('*** Begin Patch')
for path,content in files.items():
    assert not path.exists()
    print('*** Add File: '+str(path.relative_to(ROOT)))
    print(''.join('+'+line+'\n' for line in content.splitlines()),end='')
print('*** End Patch')
