"""CPU-only approval, identity and explicit-step guards for motion probe v1.

Importing this module creates no simulator, output, approval or CUDA context.
The original PROPOSED document is immutable; readiness is never an approval.
"""
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
NAV = ROOT/'CaMRA/navigation'
PROPOSAL = NAV/'config/nav043_motion_probe_proposal_v1.json'
APPROVAL = NAV/'config/nav043_motion_probe_approval_v1.json'
READINESS = NAV/'config/nav043_motion_probe_readiness_v1.json'
EVIDENCE = ROOT/'.worklog-testing/NAV-043/motion-launch-v1'
PROPOSAL_PIN = '9eb3caf00523a65d4848dc07e11a8fcafab87d8a01fa6c33a0459398ba7444bb'
BUNDLE = Path('/pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2')


def load(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def persist(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def pinned(path):
    if sha(path) != Path(str(path)+'.sha256').read_text().strip():
        raise ValueError('changed pinned document: '+str(path))
    return load(path)


def approved():
    p = pinned(PROPOSAL)
    if sha(PROPOSAL) != PROPOSAL_PIN or p['status'] != 'PROPOSED':
        raise ValueError('original proposal identity changed')
    # Deliberately first: a missing approval must not create launch markers,
    # scan GPU resources, import Isaac, or consume a physical application.
    if not APPROVAL.is_file():
        raise ValueError('separate exact APPROVED registration is missing')
    a = pinned(APPROVAL)
    r = pinned(READINESS)
    if (a.get('schema') != 'nav043-motion-probe-approval-v1'
            or a.get('status') != 'APPROVED'
            or a.get('proposal_sha256') != PROPOSAL_PIN
            or a.get('readiness_sha256') != sha(READINESS)
            or a.get('output_root') != p['output_root']
            or a.get('limits') != p['limits']
            or a.get('environment') != p['environment']
            or a.get('schedule') != p['schedule']):
        raise ValueError('approval does not bind exact scope, sources and limits')
    if (r.get('status') != 'PREPARED_ONLY'
            or r.get('proposal_sha256') != PROPOSAL_PIN):
        raise ValueError('readiness does not bind original proposal')
    for group in (p['source_hashes'], r['source_hashes']):
        for name, digest in group.items():
            if sha(ROOT/name) != digest:
                raise ValueError('changed supporting source: '+name)
    return p, a, r


def registered_context(p):
    e = p['environment']
    if (socket.gethostname().split('.')[0] != e['host'] or not os.environ.get('TMUX')
            or Path(sys.executable).resolve() != Path(e['python']).resolve()
            or os.environ.get('CUDA_VISIBLE_DEVICES') != str(e['physical_gpu_index'])):
        raise ValueError('requires registered tiger11/tmux/Python/physical GPU6')
    spec = importlib.util.find_spec('isaaclab')
    if spec is None or spec.origin != e['isaaclab_origin']:
        raise ValueError('actual installed IsaacLab differs')
    plan = load(ROOT/'doc/plan/20261109_camra_navigation_implementation.json')
    states = {t['id']: t['status'] for t in plan['tasks']}
    ready = next(t['id'] for t in plan['tasks'] if not t['status']
                 and all(states[d] for d in t['depends_on']))
    if plan['document']['version'] != p['plan_version'] or ready != 'NAV-043':
        raise ValueError('registered NAV-043 is not eligible')
    return {'host':socket.getfqdn(), 'python':sys.executable,
            'tmux_pane':os.environ.get('TMUX_PANE'), 'isaaclab_origin':spec.origin}


def protected_sources():
    # Read only identities of already exposed inputs, not actor/critic tensors.
    import nav043_motion_response as old
    protocol = old.verify_protocol()
    frozen = load(ROOT/'CaMRA/locomotion/docs/frozen/camra_go2_velocity_it2999_v2.frozen_manifest.json')
    for name, item in frozen['integrity']['files'].items():
        path = BUNDLE/name
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('changed whole frozen bundle: '+name)
    return {'protected_identities':len(old.protected_identities(protocol)),
            'existing_raw_input_hashes':sum(len(t['hashes']) for t in protocol['traces']),
            'bundle_files':len(frozen['integrity']['files'])}


def gpu_ready(p):
    e = p['environment']
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
        '--format=csv,noheader,nounits'], text=True)
    matches = [[v.strip() for v in row.split(',')] for row in rows.splitlines()
               if row.split(',')[0].strip() == str(e['physical_gpu_index'])]
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader'], text=True)
    if (len(matches) != 1 or matches[0][1] != e['physical_gpu_uuid']
            or int(matches[0][2]) > 100 or e['physical_gpu_uuid'] in apps):
        raise ValueError('registered GPU unavailable; no replacement')
    return {'registered_gpu':matches[0], 'compute_apps':apps}


def refuse_conflicts(p, launcher):
    if Path(p['output_root']).exists() or (EVIDENCE/'launch.lock').exists():
        raise ValueError('existing output/attempt; do not retry or change root')
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = (entry/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        if str(launcher).encode() in args:
            raise ValueError('another launcher/child exists: '+entry.name)


class ExplicitStepBudget:
    """Count installed SimulationContext.step calls, including setup/reset.

    Journal each attempted call before executing it and its return afterward.
    Native internal initialization steps remain explicitly unobservable.
    No application is created here; installation happens before gym.make.
    """
    def __init__(self, stream, *, total_cap=32000, reserve_cap=5600,
                 case_ids=tuple(f'M{i:02}' for i in range(1,13)), ticks_per_case=2200):
        self.stream = stream
        self.total_cap, self.reserve_cap = total_cap, reserve_cap
        self.case_ids, self.ticks_per_case = tuple(case_ids), ticks_per_case
        self.calls = self.returned = self.reserve = self.case_ticks = self.finished_cases = 0
        self.case_id = None
        self.in_flight = False

    def event(self, kind, **extra):
        self.stream.write(json.dumps({'kind':kind, 'calls':self.calls,
            'returned':self.returned, 'reserve':self.reserve,
            'case_id':self.case_id, 'case_ticks':self.case_ticks, **extra})+'\n')
        self.stream.flush()

    def begin_case(self, case_id):
        if (self.case_id is not None or self.in_flight
                or self.finished_cases >= len(self.case_ids)
                or case_id != self.case_ids[self.finished_cases]):
            raise ValueError('case order, duplicate attempt or incomplete previous step')
        self.case_id, self.case_ticks = case_id, 0
        self.event('case_begin')

    def finish_case(self):
        if self.case_id is None or self.in_flight or self.case_ticks != self.ticks_per_case:
            raise ValueError('incomplete case; no replacement or retry')
        self.event('case_finished')
        self.finished_cases += 1
        self.case_id, self.case_ticks = None, 0

    def call(self, original, *args, **kwargs):
        if self.in_flight:
            raise ValueError('previous explicit physical call did not return')
        if (self.calls >= self.total_cap
                or (self.case_id is None and self.reserve >= self.reserve_cap)
                or (self.case_id is not None and self.case_ticks >= self.ticks_per_case)):
            self.event('cap_refused')
            raise ValueError('explicit physical step allocation exhausted')
        self.calls += 1
        if self.case_id is None:
            self.reserve += 1
        else:
            self.case_ticks += 1
        self.in_flight = True
        self.event('step_called')
        result = original(*args, **kwargs)
        self.returned += 1
        self.in_flight = False
        self.event('step_returned')
        return result

    def snapshot(self):
        return {'explicit_step_calls':self.calls, 'returned_calls':self.returned,
                'reserve_calls':self.reserve, 'completed_cases':self.finished_cases,
                'case_id':self.case_id, 'case_ticks':self.case_ticks,
                'in_flight':self.in_flight,
                'native_internal_initialization_steps':'unobservable; not declared zero'}
