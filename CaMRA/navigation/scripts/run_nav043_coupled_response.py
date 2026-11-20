"""Finite response-v4 diagnostic, gated before fitting inputs or markers."""
from pathlib import Path
import os
import signal
import time
import numpy as np
import nav043_coupled_response as core

base=core.base
context=core.old.context
FEATURE_VERSION=core.FEATURE_VERSION
LIMITS=dict(core.old.LIMITS)
HERE=base.ROOT/'.worklog-testing/NAV-043/coupled-response-v4'
PROPOSAL=base.NAV/'config/nav043_coupled_response_proposal_v4.json'
APPROVAL=base.NAV/'config/nav043_coupled_response_approval_v4.json'
READINESS=base.NAV/'config/nav043_coupled_response_readiness_v4.json'
load_cases=core.old.load_cases
folds=core.old.folds
diagnostic_gate=core.old.diagnostic_gate
check_model=core.check_model


def child_command():
    return [base.PYTHON, '-B', '-c',
            'import run_nav043_coupled_response as r; r.run(r.approved()["output_root"])']


def approved():
    if not APPROVAL.is_file():
        raise ValueError('response-v4 finite scientific fits require separate APPROVED registration')
    p,a,ready=[core.old.pin(path) for path in (PROPOSAL,APPROVAL,READINESS)]
    if (p['schema']!='nav043-coupled-response-v4' or p['status']!='PROPOSED'
            or p['limits']!=LIMITS or a.get('status')!='APPROVED'
            or a.get('proposal_sha256')!=base.sha(PROPOSAL)
            or a.get('readiness_sha256')!=base.sha(READINESS)
            or a.get('limits')!=LIMITS or a.get('output_root')!=p['output_root']
            or ready['status']!='PREPARED_ONLY'
            or ready['proposal_sha256']!=base.sha(PROPOSAL)):
        raise ValueError('approval does not bind exact v4 diagnostic and support')
    for group in (p['source_hashes'],ready['source_hashes']):
        for path,digest in group.items():
            if base.sha(path)!=digest:
                raise ValueError('changed diagnostic source: '+path)
    return p


def reference_rows():
    report=base.load(base.NAV/'reports/nav043_supported_response_review_v3.json')
    if report['status']!='SAVED_SUPPORTED_RESPONSE_EXECUTION_AUDIT_PASS':
        raise ValueError('unverified reference output')
    for item in report['output_index']:
        path=Path(item['path'])
        if path.stat().st_size!=item['bytes'] or base.sha(path)!=item['sha256']:
            raise ValueError('changed original v3 reference file')
    return {row['trial']:row for row in report['rows']}


def make_row(case,theta,reference):
    t=case.trace
    row=dict(cohort=case.cohort,**t.identity,initial_q=case.initial_q.tolist(),
             initialization='actual warmup q' if case.cohort=='probe'
             else 'zero; missing historical warmup q')
    for key in ('trial','cohort','level','method','kind','initial_q','initialization'):
        if row[key]!=reference[key]:
            raise ValueError('baseline and candidate case alignment differs')
    poses,responses=core.free_replay(case,theta)
    row['candidate']=base.metrics(t,poses,responses,t.ticks)
    row['candidate_prefixes']={str(s):base.metrics(t,poses,responses,s*200)
                               for s in (5,10,20,30,45) if s*200<=t.ticks}
    for key in ('affine','affine_prefixes','ideal'):
        row[key]=reference[key]
    row['previous_v3']=reference['candidate']
    row['previous_v3_prefixes']=reference['candidate_prefixes']
    return row,(poses,responses)


def design(cases,axis):
    x,y,weights=core.design(cases)
    return x,y[:,axis],weights


def run(output):
    context()
    p=approved()
    if Path(output)!=Path(p['output_root']) or not (HERE/'launch.lock').is_file():
        raise ValueError('requires registered once-only launcher')
    output=Path(output)
    output.mkdir(exist_ok=False)
    started=time.monotonic()
    solves,rows=[],[]
    def timeout(*_):
        raise TimeoutError('CPU wall limit reached')
    previous=signal.signal(signal.SIGALRM,timeout)
    signal.alarm(LIMITS['cpu_wall_s'])
    try:
        base.write_new(output/'STARTED.json',dict(proposal_sha256=base.sha(PROPOSAL),
            approval_sha256=base.sha(APPROVAL),pane=os.environ['TMUX_PANE'],pid=os.getpid()))
        reference=reference_rows()
        for level,train,held in folds(load_cases()):
            folder=output/level
            folder.mkdir()
            x,y,_=core.design(train)
            theta=[]
            for axis,name in enumerate(base.AXES):
                ordinal=len(solves)+1
                if ordinal>LIMITS['axis_solves']:
                    raise ValueError('solve allocation exhausted')
                marker=dict(ordinal=ordinal,axis=name,rows=len(x))
                base.write_new(folder/(name+'-solve-started.json'),marker)
                solves.append(dict(level=level,**marker))
                fitted=core.solve_axis(x,y[:,axis])
                solves[-1].update(fitted)
                base.write_new(folder/(name+'-solve-result.json'),solves[-1])
                if not fitted['valid']:
                    raise ValueError('invalid solution; no clipping or retry')
                theta.append(fitted['theta'])
            theta=np.asarray(theta)
            stability=check_model(theta)
            base.write_new(folder/'model.json',dict(theta=theta.tolist(),stability=stability,
                training=[case.key for case in train],heldout=[case.key for case in held],
                feature_version=FEATURE_VERSION,deployment_model=False))
            for case in held:
                row,(poses,responses)=make_row(case,theta,reference[case.key])
                stem=case.trace.identity['kind']+'-'+case.trace.identity['method']
                np.savez(folder/(stem+'.npz'),predicted=poses,responses=responses)
                base.write_new(folder/(stem+'.json'),row)
                rows.append(row)
        load_cases()
        reference_rows()
        approved()
        base.write_new(output/'FINISHED.json',dict(status='EXPOSED_DIAGNOSTIC_ONLY',
            solves=solves,rows=rows,gate=diagnostic_gate(rows),cpu_wall_s=time.monotonic()-started,
            limits=LIMITS,extra_all_data_fit=False,calibration_passed=False))
    except BaseException as error:
        base.write_new(output/'HELD.json',dict(error=repr(error),solves=solves,
            heldout_rows=len(rows),cpu_wall_s=time.monotonic()-started,NAV043_passed=False))
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,previous)
