"""Read-only candidate-clock and source-identity audit; no model solves."""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import xml.etree.ElementTree as ET

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parents[1]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def load(p):
    return json.loads(Path(p).read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--plan-before',type=Path,required=True)
    args = parser.parse_args()
    if socket.gethostname().split('.')[0]!='tiger11' or not os.environ.get('TMUX'):
        raise ValueError('requires tiger11/tmux')
    path = NAV/'config/navigation_horizon_v2.json'
    if sha(path)!=Path(str(path)+'.sha256').read_text().strip():
        raise ValueError('horizon protocol pin mismatch')
    p = load(path)
    for source,digest in p['source_hashes'].items():
        if sha(ROOT/source)!=digest:
            raise ValueError('changed old source: '+source)
    code = NAV/'scripts/nav043_motion_response.py'
    spec = importlib.util.spec_from_file_location('response_probe',code)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    response_protocol = module.verify_protocol()
    module.require_approval(response_protocol)
    before = load(args.plan_before)
    after = load(ROOT/'doc/plan/20261109_camra_navigation_implementation.json')
    if len(before['tasks'])!=len(after['tasks']):
        raise ValueError('task count changed')
    for old,new in zip(before['tasks'],after['tasks']):
        if {k:v for k,v in old.items() if k!='work_records'}!={k:v for k,v in new.items() if k!='work_records'}:
            raise ValueError('historical task definition/status changed')
        if new['work_records'][:len(old['work_records'])]!=old['work_records']:
            raise ValueError('historical work records changed')
    junit = ROOT/'.worklog-testing/NAV-043/horizon90-v1/basic.xml'
    suite = ET.parse(junit).getroot().find('testsuite')
    if int(suite.attrib['tests'])!=17 or any(int(suite.attrib[k]) for k in ('failures','errors','skipped')):
        raise ValueError('basic check evidence not passing')
    from camra_navigation.horizon import PROFILE
    PROFILE.require_binding(p['time_binding'])
    if [PROFILE.step_cap(dt) for dt in (.1,.02,.005)] != [900,4500,18000]:
        raise ValueError('clock cap mismatch')
    candidate_sources = ['src/camra_navigation/horizon.py','src/camra_navigation/envs/horizon90.py',
                         'tests/test_horizon90.py','scripts/audit_nav043_horizon90.py']
    output = {'status':'BASIC_INTERFACE_AND_IDENTITY_PASS_ONLY',
              'at':datetime.now().astimezone().isoformat(),'host':socket.gethostname(),
              'pane':os.environ.get('TMUX_PANE'),'python':sys.executable,
              'profile_sha256':sha(path),'profile':p['time_binding'],
              'design':p['deadline_design'],'protected_identities_checked':295,
              'saved_raw_identities_checked':sum(len(t['hashes']) for t in response_protocol['traces']),
              'old_plan_sha256':sha(args.plan_before),'plan_before_path':str(args.plan_before),
              'historical_tasks_and_records_preserved':True,'historical_completed_count':sum(t['status'] for t in after['tasks']),
              'basic_tests_passed':17,'basic_junit_sha256':sha(junit),
              'candidate_implementation_hashes':{s:sha(NAV/s) for s in candidate_sources},
              'candidate_production_launch_and_models_validated':False,
              'new_physical_or_nominal_business_episodes':0,'W_queries':0,'actor_critic_updates':0,
              'basic_fixture_calls':{'scalar_nav_steps':2,'vector_env_nav_steps':2,
                                    'physical_event_monitor_active_ticks':2,'physical_event_monitor_inactive_ticks':2,
                                    'Isaac_simulator_steps':0},
              'NAV043_passed':False,'G5_passed':False}
    with args.out.open('x') as stream:
        json.dump(output,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


if __name__=='__main__':
    main()
