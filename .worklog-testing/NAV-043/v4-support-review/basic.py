"""Small main-agent checks of support/recurrence math; no coefficient solving."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path('/home/caohy/repositories/Paper')
sys.path.insert(0, str(ROOT/'CaMRA/navigation/scripts'))
import analyze_nav043_v4_support as a


class Basic(unittest.TestCase):
    def test_support_endpoints_and_extrapolation(self):
        v = a.support([[0, 0, 0], [.4, 0, 1]], [[.4, 0, 1], [-1, 0, 0], [0, .4, 0]])
        self.assertEqual(v['outside_axis_box_count'], 2)
        self.assertEqual(v['outside_per_axis'], [1, 1, 0])
        self.assertFalse(v['full_convex_hull_or_state_support_verified'])

    def test_axis_box_is_not_joint_support(self):
        v = a.support([[0, 0, 0], [1, .4, 1]], [[1, 0, 0]])
        self.assertEqual(v['outside_axis_box_count'], 0)
        self.assertEqual(v['normalized_nearest_training_distance_mean'], 1.)

    def test_exact_recurrence_and_signed_error(self):
        theta = np.zeros((3, 11))
        theta[:, :3] = .5*np.eye(3)
        theta[:, 3:6] = .3*np.eye(3)
        previous = np.array([[.2, -.1, .3], [.4, .1, .2]])
        command = np.array([[.4, .1, .2], [.2, -.1, -.2]])
        actual = previous+.02
        free_previous = previous+.1
        predicted = .5*free_previous+.3*command
        v = a.accounting(theta, previous, command, actual, predicted, free_previous)
        self.assertLess(v['recurrence_closure_max'], 1.e-14)
        np.testing.assert_allclose(v['free_error_mean'],
                                   np.array(v['innovation_mean'])+.05)

    def test_saved_prediction_corruption_rejected(self):
        theta = np.zeros((3, 11))
        z = np.zeros((1, 3))
        with self.assertRaisesRegex(ValueError, 'recurrence'):
            a.accounting(theta, z, z, z, z+.01, z)

    def test_invalid_alignment_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            a.support([[float('nan'), 0, 0]], [[0, 0, 0]])
        z = np.zeros((1, 3))
        with self.assertRaisesRegex(ValueError, 'aligned'):
            a.accounting(np.zeros((3, 11)), z, z, z, z, np.zeros((2, 3)))

    def test_fits_forbidden(self):
        with self.assertRaisesRegex(RuntimeError, 'must not fit'):
            a.forbidden_fit()


if __name__ == '__main__':
    a.saved.context()
    unittest.main(verbosity=2)
