"""Basic synthetic correctness tests, without reading physical fitting data."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import nav043_supported_response as c
import launch_nav043_supported_response_once as launcher


def case(key, cohort='navigation', level=None, kind='baseline', ticks=45):
    n = ticks//20
    trace = c.base.Trace(dict(trial=key, level=level or c.base.LEVELS[0],
        method='full', kind=kind), np.zeros(3), np.zeros((ticks, 3)),
        np.zeros((ticks, 3)), np.zeros(((ticks+19)//20, 3)),
        np.arange(n*3, dtype=float).reshape(n, 3)/10.)
    return c.Case(trace, cohort, np.array([-.01, .002, .003]))


class SupportedResponseTests(unittest.TestCase):
    def test_origin_and_continuity(self):
        np.testing.assert_array_equal(c.features(np.zeros((1, 3))), np.zeros((1, 5)))
        for u in (.1, .2):
            a = c.features(np.array([[u-1.e-9, 0, 0], [u+1.e-9, 0, 0]]))
            self.assertLess(np.max(np.abs(a[1]-a[0])), 2.e-8)

    def test_frozen_domain_and_invalid_commands(self):
        for axis in range(3):
            for sign in (-1., 1.):
                u = np.zeros((1, 3)); u[0, axis] = sign*np.float32(c.DOMAIN[axis])
                c.features(u)
        for u in ([[1.001, 0, 0]], [[0, .401, 0]], [[0, 0, 1.001]], [[float('nan'), 0, 0]]):
            with self.assertRaises(ValueError):
                c.features(np.array(u))

    def test_navigation_does_not_fabricate_initial_fit_pair(self):
        item = case('old')
        prev, command, target = item.pairs()
        self.assertEqual(len(target), 1)
        np.testing.assert_array_equal(prev[0], item.trace.observed_twists[0])
        np.testing.assert_array_equal(target[0], item.trace.observed_twists[1])

    def test_probe_uses_real_warmup_q(self):
        item = case('probe', 'probe')
        prev, command, target = item.pairs()
        self.assertEqual(len(target), 2)
        np.testing.assert_array_equal(prev[0], item.initial_q)

    def test_cohort_and_case_weighting(self):
        items = [case('n1', ticks=60), case('n2', ticks=100),
                 case('p', 'probe', ticks=80)]
        x, y, weights = c.design(items, 0)
        self.assertAlmostEqual(weights[:2].sum(), .25)
        self.assertAlmostEqual(weights[2:6].sum(), .25)
        self.assertAlmostEqual(weights[6:].sum(), .5)
        self.assertAlmostEqual(weights.sum(), 1.)

    def test_folds_keep_whole_cases_and_cover_each_once(self):
        items = [case(level+str(i), level=level) for level in c.base.LEVELS for i in range(6)]
        items += [case('M%02d'%i, 'probe', kind='M%02d'%i) for i in range(1, 13)]
        seen = []
        for level, train, held in c.folds(items):
            self.assertFalse({x.key for x in train} & {x.key for x in held})
            self.assertTrue(all(x.trace.identity['level'] != level for x in train if x.cohort == 'navigation'))
            seen += [x.key for x in held]
        self.assertEqual(sorted(seen), sorted(x.key for x in items))
        with self.assertRaises(ValueError):
            c.folds(items[:-1]+[items[0]])

    def test_rank_audit_does_not_solve_targets(self):
        with patch.object(np.linalg, 'lstsq', side_effect=AssertionError('no fit')):
            self.assertTrue(c.rank_support(np.eye(6))['valid'])
            self.assertFalse(c.rank_support(np.zeros((8, 6)))['valid'])

    def test_exact_synthetic_solve_and_unstable_rejection(self):
        # Exactly three synthetic solves in this suite; never real trajectories.
        x = np.vstack((np.eye(6), np.eye(6)*2))
        theta = np.array([.5, .2, .1, -.03, .3, -.05])
        fitted = c.fit_axis(x, x@theta)
        self.assertTrue(fitted['valid'])
        np.testing.assert_allclose(fitted['theta'], theta, atol=1.e-14)
        theta[0] = 1.01
        self.assertFalse(c.fit_axis(x, x@theta)['valid'])
        self.assertFalse(c.fit_axis(np.zeros((8, 6)), np.zeros(8))['valid'])

    def test_free_prediction_uses_memory_keeps_partial_tail(self):
        item = case('partial')
        item.initial_q = np.array([.2, 0., 0.])
        theta = np.zeros((3, 6)); theta[:, 0] = .5
        poses, responses = c.free_replay(item, theta)
        self.assertEqual(poses.shape, (45, 3))
        np.testing.assert_allclose(responses[:, 0], [.1, .05, .025])
        self.assertAlmostEqual(poses[-1, 0], .015625)
        item.trace.actual[:] = 999.
        poses2, _ = c.free_replay(item, theta)
        np.testing.assert_array_equal(poses, poses2)

    def test_zero_input_has_no_affine_drift(self):
        item = case('origin', ticks=2000); item.initial_q[:] = 0.
        theta = np.ones((3, 6)); theta[:, 0] = [.2, .5, .9]
        poses, responses = c.free_replay(item, theta)
        np.testing.assert_array_equal(poses, 0.)
        np.testing.assert_array_equal(responses, 0.)

    def test_missing_approval_stops_before_input_or_markers(self):
        with tempfile.TemporaryDirectory() as folder:
            here = Path(folder)
            with patch.object(c, 'APPROVAL', here/'approval.json'), patch.object(c, 'HERE', here), \
                    patch.object(c, 'context'), patch.object(c, 'load_cases', side_effect=AssertionError('no data')):
                with self.assertRaisesRegex(ValueError, 'APPROVED'):
                    launcher.main()
            self.assertEqual(list(here.iterdir()), [])

    def test_once_launcher_refuses_existing_output_and_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            here = Path(folder); output = here/'output'; output.mkdir()
            with patch.object(c, 'context'), patch.object(c, 'HERE', here), \
                    patch.object(c, 'approved', return_value={'output_root': str(output)}), \
                    patch.object(launcher.subprocess, 'Popen', side_effect=AssertionError('no child')):
                with self.assertRaises(ValueError):
                    launcher.main()
                output.rmdir(); (here/'launch.lock').write_text('consumed')
                with self.assertRaises(ValueError):
                    launcher.main()

    def test_gate_cannot_drop_cases(self):
        with self.assertRaises(ValueError):
            c.diagnostic_gate([])


if __name__ == '__main__':
    unittest.main(verbosity=2)
