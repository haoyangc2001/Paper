"""Main-agent basic checks, including a known-coefficient full saved-output fixture.

The fixture intercepts all fitting calls: zero real or synthetic coefficient solves.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path('/home/caohy/repositories/Paper')
sys.path.insert(0, str(ROOT/'CaMRA/navigation/scripts'))
import nav043_matched_support_v5 as m


def cases():
    rng = np.random.default_rng(20261120)
    values = []
    identities = [dict(trial=f'{level}-{i}', level=level, method='synthetic', kind=f'nav{i}')
                  for level in m.base.LEVELS for i in range(6)]
    identities += [dict(trial=f'probe{i}', level='controlled', method='synthetic', kind=f'M{i:02}')
                   for i in range(1, 13)]
    for identity in identities:
        u = rng.uniform(-1., 1., size=(40, 3))*[1., .4, 1.]
        q, pose = np.zeros(3), np.zeros(3)
        observed, actual = [], []
        for command in u:
            q = .2*q+.3*command
            observed.append(q.copy())
            for _ in range(20):
                pose = m.base.step(pose, q, .005)
                actual.append(pose.copy())
        trace = m.base.Trace(identity, np.zeros(3), np.array(actual),
                            np.repeat(u, 20, axis=0), u, np.array(observed))
        values.append(m.diagonal.Case(trace, 'probe' if identity['level'] == 'controlled'
                                      else 'navigation', np.zeros(3)))
    return values


class Basic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = cases()

    def test_partition_roles_and_all_known_probe_support(self):
        parts = m.partitions(self.cases)
        self.assertEqual(len(parts), 3)
        for _, train, held, probes in parts:
            self.assertEqual((len(train), len(held), len(probes)), (24, 6, 12))
            self.assertTrue({c.key for c in train}.isdisjoint(c.key for c in held))
            self.assertTrue({c.key for c in probes}.issubset(c.key for c in train))
            self.assertTrue(all(c.cohort == 'navigation' for c in held))

    def test_probe_omission_and_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            m.partitions(self.cases[:-1])
        with self.assertRaises(ValueError):
            m.partitions(self.cases[:-1]+self.cases[:1])

    def test_matched_cohort_and_case_weights(self):
        train = m.partitions(self.cases)[0][1]
        expected = []
        for cohort in ('navigation', 'probe'):
            for case in train:
                if case.cohort == cohort:
                    expected.extend([.5/12/len(case.pairs()[2])]*len(case.pairs()[2]))
        for kind in m.KINDS:
            x, y, w = m.design(train, kind, 0)
            self.assertEqual(x.shape, (len(y), m.width(kind)))
            np.testing.assert_allclose(w, expected, rtol=0., atol=0.)
            self.assertAlmostEqual(w.sum(), 1.)

    def test_unstable_models_and_mislabeled_probe_rejected(self):
        for kind in m.KINDS:
            theta = np.zeros((3, m.width(kind)))
            theta[0, 0] = 1.01
            with self.assertRaises(ValueError):
                m.check_model(theta, kind)
        with self.assertRaisesRegex(ValueError, 'training status'):
            m.row(self.cases[-1], np.zeros((3, 11)), 'coupled_v4', 'held_navigation')

    def test_missing_approval_before_loading_or_markers(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            with patch.object(m, 'APPROVAL', folder/'missing.json'), patch.object(m, 'HERE', folder), \
                    patch.object(m.diagonal, 'load_cases', side_effect=AssertionError('loaded too early')):
                for call in (m.approved, m.launch, m.run, m.audit):
                    with self.assertRaisesRegex(ValueError, 'separate APPROVED'):
                        call()
                self.assertEqual(list(folder.iterdir()), [])

    def test_existing_attempt_rejected_before_child(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder/'launch.lock').write_text('1\n')
            with patch.object(m, 'HERE', folder), \
                    patch.object(m, 'approved', return_value={'output_root':str(folder/'output')}), \
                    patch.object(m.subprocess, 'Popen', side_effect=AssertionError('must not spawn')):
                with self.assertRaisesRegex(ValueError, 'already attempted'):
                    m.launch()

    def test_full_synthetic_pipeline_and_audit_with_known_coefficients(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            here, output = folder/'evidence', folder/'output'
            here.mkdir()
            (here/'launch.lock').write_text('123\n')
            proposal, approval = folder/'proposal.json', folder/'approval.json'
            proposal.write_text('{}\n')
            approval.write_text('{}\n')
            counters = dict(diagonal_v3=0, coupled_v4=0)
            def known(kind, x, y):
                axis = counters[kind] % 3
                counters[kind] += 1
                theta = np.zeros(m.width(kind))
                theta[0 if kind == 'diagonal_v3' else axis] = .2
                theta[axis+(1 if kind == 'diagonal_v3' else 3)] = .3
                np.testing.assert_allclose(x@theta, y, rtol=1.e-12, atol=1.e-12)
                singular = np.linalg.svd(x, compute_uv=False)
                return dict(theta=theta.tolist(), rank=m.width(kind), valid=True,
                    residual_sum=[float(np.sum((x@theta-y)**2))], singular_values=singular.tolist())
            with patch.object(m, 'HERE', here), patch.object(m, 'PROPOSAL', proposal), \
                    patch.object(m, 'APPROVAL', approval), patch.object(m, 'REVIEW', folder/'review.json'), \
                    patch.object(m, 'approved', return_value={'output_root':str(output)}), \
                    patch.object(m.diagonal, 'load_cases', return_value=self.cases), \
                    patch.object(m.diagonal, 'fit_axis', side_effect=lambda x,y:known('diagonal_v3',x,y)), \
                    patch.object(m.coupled, 'solve_axis', side_effect=lambda x,y:known('coupled_v4',x,y)), \
                    patch.object(np.linalg, 'lstsq', side_effect=AssertionError('no coefficient solves')):
                m.run()
                final = m.base.load(output/'FINISHED.json')
                self.assertEqual(len(final['solves']), 18)
                self.assertEqual(len(final['rows']), 108)
                self.assertFalse(final['model_selected'])
                self.assertTrue(all(r['evaluation_role']=='in_sample_probe'
                    for r in final['rows'] if r['cohort']=='probe'))
                started = m.base.load(output/'STARTED.json')
                m.base.write_new(here/'launch.json', dict(wrapper_pid=123, child_pid=started['pid'],
                    command=m.child_command(), pane=started['pane'], output_root=str(output),
                    proposal_sha256=m.base.sha(proposal), approval_sha256=m.base.sha(approval)))
                m.base.write_new(here/'exit.json', dict(exit_code=0,error=None,retry_performed=False,
                    finished=True,held=False,wrapper_wall_s=final['cpu_wall_s']+1.))
                m.audit()
                audited = m.base.load(m.REVIEW)
                self.assertEqual(audited['status'], 'SAVED_MATCHED_SUPPORT_EXECUTION_AUDIT_PASS')
                self.assertEqual(len(audited['output_index']), 260)
                self.assertEqual(counters, dict(diagonal_v3=9,coupled_v4=9))
                with self.assertRaises(ValueError):
                    m.audit()


if __name__ == '__main__':
    m.context()
    unittest.main(verbosity=2)
