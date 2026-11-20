"""Main-agent basic checks; exactly three synthetic solves, no real inputs."""
from pathlib import Path
from contextlib import ExitStack
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import nav043_coupled_response as c
import run_nav043_coupled_response as run
import launch_nav043_coupled_response_once as launcher
import audit_nav043_coupled_response as audit


def case(name,cohort='navigation',ticks=45):
    trace=c.base.Trace(dict(trial=name,level=c.base.LEVELS[0],kind='baseline',method='full'),
        np.zeros(3),np.zeros((ticks,3)),np.zeros((ticks,3)),
        np.zeros(((ticks+19)//20,3)),np.arange(ticks//20*3).reshape(-1,3)/10.)
    return c.old.Case(trace,cohort,np.array([.2,0.,0.]))


class CoupledResponseTests(unittest.TestCase):
    def test_features_origin_continuity_and_frozen_domain(self):
        np.testing.assert_array_equal(c.features(np.zeros((1,3))),np.zeros((1,8)))
        for value in (.1,.2):
            p=c.features(np.array([[value-1.e-9,0,.01],[value+1.e-9,0,.01]]))
            self.assertLess(np.max(np.abs(p[1]-p[0])),2.e-8)
        for u in ([[1.001,0,0]],[[0,.401,0]],[[0,0,1.001]],[[np.nan,0,0]]):
            with self.assertRaises(ValueError):
                c.features(u)

    def test_non_linear_terms_separate_forward_speed_and_turn(self):
        u=np.array([[.2,0,.3],[.4,0,-.3],[0,0,.3]])
        np.testing.assert_allclose(c.features(u)[:,5:],[[.04,.06,.06],[.16,-.12,.12],[0,0,0]])

    def test_case_weights_previous_full_q_and_partial_pair(self):
        nav=case('n');probe=case('p','probe',ticks=60)
        x,y,w=c.design([nav,probe])
        self.assertEqual(x.shape,(4,11));self.assertEqual(y.shape,(4,3))
        self.assertAlmostEqual(w[0],.5);self.assertAlmostEqual(w[1:].sum(),.5)
        np.testing.assert_allclose(x[0,:3]/np.sqrt(w[0]),nav.trace.observed_twists[0])
        np.testing.assert_allclose(x[1,:3]/np.sqrt(w[1]),probe.initial_q)

    def test_exact_three_synthetic_solves(self):
        rng=np.random.default_rng(123)
        x=rng.normal(size=(60,11))
        theta=np.zeros((3,11));theta[:,:3]=np.eye(3)*.5
        theta[:,3:]=rng.normal(size=(3,8))*.01
        for axis in range(3):
            solved=c.solve_axis(x,x@theta[axis])
            self.assertTrue(solved['valid'])
            np.testing.assert_allclose(solved['theta'],theta[axis],atol=1.e-13)

    def test_full_memory_stability_and_origin(self):
        theta=np.zeros((3,11));theta[:,:3]=[[.5,.2,0],[-.2,.5,0],[0,0,.9]]
        self.assertAlmostEqual(c.check_model(theta)['spectral_radius'],.9)
        theta[2,2]=1.
        with self.assertRaises(ValueError):
            c.check_model(theta)
        theta[:,:3]=np.eye(3)*.5;theta[:,3:]=1.
        item=case('origin',ticks=61);item.initial_q[:]=0.
        poses,response=c.free_replay(item,theta)
        np.testing.assert_array_equal(poses,0.);np.testing.assert_array_equal(response,0.)

    def test_cross_memory_free_replay_keeps_tail_and_no_truth_correction(self):
        theta=np.zeros((3,11));theta[:,:3]=[[.5,0,0],[.1,.5,0],[0,0,.5]]
        item=case('tail')
        poses,responses=c.free_replay(item,theta)
        np.testing.assert_allclose(responses,[[.1,.02,0],[.05,.02,0],[.025,.015,0]])
        self.assertEqual(poses.shape,(45,3))
        item.trace.actual[:]=999.
        np.testing.assert_array_equal(c.free_replay(item,theta)[0],poses)

    def test_rank_support_no_target_solver(self):
        with patch.object(np.linalg,'lstsq',side_effect=AssertionError('no fit')):
            self.assertTrue(c.rank_support(np.eye(11))['valid'])
            self.assertFalse(c.rank_support(np.zeros((15,11)))['valid'])

    def test_missing_approval_before_data_or_marker(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with patch.object(run,'APPROVAL',root/'approval'),patch.object(run,'HERE',root),patch.object(run,'context'),patch.object(run,'load_cases',side_effect=AssertionError('no data')):
                for entry in (launcher.main,audit.main):
                    with self.assertRaisesRegex(ValueError,'APPROVED'):
                        entry()
            self.assertEqual(list(root.iterdir()),[])

    def test_once_launcher_refuses_consumed_slot_and_incomplete_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);output=root/'output';output.mkdir()
            with patch.object(run,'context'),patch.object(run,'HERE',root),patch.object(run,'approved',return_value={'output_root':str(output)}),patch.object(launcher.subprocess,'Popen',side_effect=AssertionError('no child')):
                with self.assertRaises(ValueError):
                    launcher.main()
                with self.assertRaisesRegex(ValueError,'no complete'):
                    audit.main()
                output.rmdir();(root/'launch.lock').write_text('consumed')
                with self.assertRaises(ValueError):
                    launcher.main()

    def test_audit_normal_equations_no_solver_and_corrupt_rejection(self):
        x=np.vstack((np.diag(np.arange(1.,12.)),np.zeros((11,11))))
        theta=np.arange(11)/100.
        y=x@theta+np.r_[np.zeros(11),np.ones(11)*.1]
        record=dict(theta=theta.tolist(),singular_values=list(np.arange(11.,0.,-1.)),rows=22,rank=11,valid=True,residual_sum=[.11])
        with patch.object(np.linalg,'lstsq',side_effect=AssertionError('no fit')):
            self.assertEqual(audit.verify_solution(x,y,record)['normal_residual_inf'],0.)
        record['theta'][1]+=.1
        with self.assertRaises(ValueError):
            audit.verify_solution(x,y,record)

    def test_synthetic_production_output_and_full_audit_without_extra_fit(self):
        rng=np.random.default_rng(91)
        theta=rng.normal(size=(3,11))*.02
        theta[:,:3]=np.diag([.5,.3,.2])
        cases=[]
        for i in range(30):
            cohort='navigation' if i<18 else 'probe'
            level=c.base.LEVELS[i//6] if i<18 else 'controlled'
            kind=('baseline','repeat','W_selected')[i%3] if i<18 else 'M%02d'%(i-17)
            method='full' if i%6<3 else 'uncalibrated'
            commands=rng.uniform(-1,1,(10,3))*np.array([1,.4,1])
            initial=np.zeros(3);previous=initial.copy();observed=[]
            for command in commands:
                previous=theta[:,:3]@previous+theta[:,3:]@c.features(command[None])[0]
                observed.append(previous.copy())
            trace=c.base.Trace(dict(trial='synthetic-'+str(i),cohort=cohort,level=level,kind=kind,method=method),
                np.zeros(3),np.zeros((200,3)),np.repeat(commands,20,axis=0),commands,np.array(observed))
            # Cohort lives on Case, not Trace.identity, as in actual saved cases.
            del trace.identity['cohort']
            cases.append(c.old.Case(trace,cohort,initial))
        reference={}
        for item in cases:
            t=item.trace;poses,responses=c.base.free_replay(t)
            metrics=c.base.metrics(t,poses,responses,t.ticks)
            reference[item.key]=dict(cohort=item.cohort,**t.identity,initial_q=item.initial_q.tolist(),
                initialization='actual warmup q' if item.cohort=='probe' else 'zero; missing historical warmup q',
                affine=metrics,affine_prefixes={},ideal=metrics,candidate=metrics,candidate_prefixes={})
        ordinal=0
        def witnessed_solution(x,y):
            nonlocal ordinal
            axis=ordinal%3;ordinal+=1
            residual=x@theta[axis]-y
            self.assertLess(np.max(np.abs(residual)),1.e-14)
            return dict(theta=theta[axis].tolist(),residual_sum=[float(residual@residual)],
                rank=11,singular_values=np.linalg.svd(x,compute_uv=False).tolist(),valid=True)
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            root=Path(folder);output=root/'output';proposal=root/'proposal';approval=root/'approval'
            proposal.write_text('{}');approval.write_text('{}');readiness=root/'readiness';readiness.write_text('{}')
            p=dict(output_root=str(output))
            for name,value in [('HERE',root),('PROPOSAL',proposal),('APPROVAL',approval),('READINESS',readiness)]:
                stack.enter_context(patch.object(run,name,value))
            stack.enter_context(patch.object(run,'context'))
            stack.enter_context(patch.object(run,'approved',return_value=p))
            stack.enter_context(patch.object(run,'load_cases',return_value=cases))
            stack.enter_context(patch.object(run,'reference_rows',return_value=reference))
            stack.enter_context(patch.object(c,'solve_axis',side_effect=witnessed_solution))
            stack.enter_context(patch.object(np.linalg,'lstsq',side_effect=AssertionError('no extra fit')))
            stack.enter_context(patch.object(audit,'REPORT',root/'report.json'))
            (root/'launch.lock').write_text('123')
            run.run(output)
            self.assertEqual(ordinal,9)
            self.assertEqual(len(list(output.rglob('*.json'))),53)
            c.base.write_new(root/'launch.json',dict(wrapper_pid=123,child_pid=os.getpid(),
                command=run.child_command(),pane=os.environ['TMUX_PANE'],output_root=str(output),
                proposal_sha256=c.base.sha(proposal),approval_sha256=c.base.sha(approval)))
            c.base.write_new(root/'exit.json',dict(exit_code=0,error=None,retry_performed=False,
                finished=True,held=False,wrapper_wall_s=3600.))
            audit.main()
            self.assertEqual(c.base.load(root/'report.json')['heldout_cases'],30)
            self.assertEqual(ordinal,9)


if __name__=='__main__':
    unittest.main(verbosity=2)
