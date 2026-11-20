"""Isolated recovery-v2 guards. Import does not authorize a second app."""
import os
from datetime import datetime
from pathlib import Path
import nav043_motion_probe_guard as original
from nav043_motion_probe_guard import (
    ROOT, NAV, BUNDLE, load, sha, persist, pinned, registered_context,
    protected_sources, gpu_ready, ExplicitStepBudget)

PROPOSAL = NAV/'config/nav043_motion_probe_recovery_proposal_v2.json'
APPROVAL = NAV/'config/nav043_motion_probe_recovery_approval_v2.json'
READINESS = NAV/'config/nav043_motion_probe_recovery_readiness_v2.json'
EVIDENCE = ROOT/'.worklog-testing/NAV-043/motion-recovery-v2'


def approved():
    if not APPROVAL.is_file():
        raise ValueError('separate exact recovery APPROVED registration is missing')
    p, a, r = pinned(PROPOSAL), pinned(APPROVAL), pinned(READINESS)
    source_proposal, _, _ = original.approved()
    if (p.get('status') != 'PROPOSED' or a.get('status') != 'APPROVED'
            or a.get('schema') != 'nav043-motion-probe-recovery-approval-v2'
            or a.get('proposal_sha256') != sha(PROPOSAL)
            or a.get('readiness_sha256') != sha(READINESS)
            or r.get('status') != 'PREPARED_ONLY'
            or r.get('proposal_sha256') != sha(PROPOSAL)
            or p['environment'] != source_proposal['environment']
            or p['schedule'] != source_proposal['schedule']
            or any(a.get(k) != p[k] for k in
                   ('output_root','limits','environment','schedule','recovery'))):
        raise ValueError('recovery approval does not bind exact sources/scope')
    if (p['limits']['applications'] != 1
            or p['limits']['all_explicit_simulation_step_calls'] != 31105
            or p['limits']['setup_reset_reserve_step_calls'] != 4705
            or p['limits']['gpu_wall_s'] != 6300
            or p['limits']['automatic_retries'] != 0
            or p['recovery']['cumulative_applications_cap'] != 2
            or p['recovery']['cumulative_explicit_step_cap'] != 32000
            or p['recovery']['cumulative_wall_cap_s'] != 7200
            or p['recovery']['failed_slot_wall_charge_s'] != 900):
        raise ValueError('recovery resource arithmetic differs')
    failure = load(NAV/'reports/nav043_motion_probe_first_failure_v1.json')
    if failure['explicit_calls_and_returns'] != 895 or failure['cases_completed'] != 0:
        raise ValueError('first failed allocation evidence differs')
    original_root = Path(source_proposal['output_root'])
    owner, child = load(original_root/'OWNER.json'), load(original_root/'PROCESS.json')
    elapsed_bound = (datetime.fromisoformat(failure['at'])
                     - datetime.fromisoformat(owner['at'])).total_seconds()
    if (not failure['both_registered_processes_absent'] or not 0 < elapsed_bound <= 900
            or any((Path('/proc')/str(pid)).exists() for pid in (owner['pid'],child['pid']))):
        raise ValueError('original process absence/900s conservative wall charge not verified')
    if Path(p['output_root']).parent != Path(source_proposal['output_root']):
        raise ValueError('recovery is not isolated under the retained original root')
    for group in (p['source_hashes'],r['source_hashes']):
        for name,digest in group.items():
            if sha(ROOT/name) != digest:
                raise ValueError('changed supporting recovery source: '+name)
    return p,a,r


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
