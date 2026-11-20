"""Small synthetic integrity checks; no parameter solves or physical inputs."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_nav043_supported_response as audit


def witness():
    x = np.vstack((np.diag(np.arange(1., 7.)), np.zeros((6, 6))))
    theta = np.array([.5, .2, -.1, .03, .04, -.02])
    y = x@theta+np.r_[np.zeros(6), np.full(6, .1)]
    record = dict(theta=theta.tolist(), singular_values=list(np.arange(6., 0., -1.)),
                  rows=12, rank=6, valid=True, residual_sum=[.06])
    return x, y, record


class SupportedAuditTests(unittest.TestCase):
    def test_witness_needs_no_parameter_solver(self):
        with ExitStack() as stack:
            for name in ('lstsq', 'solve', 'pinv'):
                stack.enter_context(patch.object(np.linalg, name, side_effect=AssertionError('no fit')))
            checked = audit.verify_solution(*witness())
        self.assertEqual(checked['normal_residual_inf'], 0.)
        self.assertAlmostEqual(checked['residual_squared_sum'], .06)

    def test_corrupt_saved_parameters_spectrum_residual_rejected(self):
        x, y, original = witness()
        for key, value in [('theta', [1., .2, -.1, .03, .04, -.02]),
                           ('theta', [.5, .3, -.1, .03, .04, -.02]),
                           ('singular_values', [7., 5., 4., 3., 2., 1.]),
                           ('residual_sum', [.07])]:
            with self.subTest(key=key, value=value):
                record = deepcopy(original); record[key] = value
                with self.assertRaises(ValueError):
                    audit.verify_solution(x, y, record)

    def test_missing_approval_precedes_data_and_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(audit.c, 'context'), patch.object(audit.c, 'APPROVAL', root/'approval'), \
                    patch.object(audit, 'REPORT', root/'review'), \
                    patch.object(audit.c, 'load_cases', side_effect=AssertionError('no data')):
                with self.assertRaisesRegex(ValueError, 'APPROVED'):
                    audit.main()
            self.assertEqual(list(root.iterdir()), [])

    def test_missing_or_held_output_precedes_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(audit.c, 'context'), \
                    patch.object(audit.c, 'approved', return_value={'output_root': str(root)}), \
                    patch.object(audit.c, 'load_cases', side_effect=AssertionError('no data')):
                with self.assertRaisesRegex(ValueError, 'no complete'):
                    audit.main()
                (root/'FINISHED.json').write_text('{}')
                (root/'HELD.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'no complete'):
                    audit.main()

    def test_saved_full_prediction_and_partial_tail_integrity(self):
        poses, responses = np.zeros((45, 3)), np.zeros((3, 3))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'synthetic.npz'
            np.savez(path, predicted=poses, responses=responses)
            audit.verify_prediction(path, poses, responses)
            for bad in (poses[:-1], np.ones_like(poses), np.full_like(poses, np.nan)):
                np.savez(path, predicted=bad, responses=responses)
                with self.assertRaises(ValueError):
                    audit.verify_prediction(path, poses, responses)


if __name__ == '__main__':
    unittest.main(verbosity=2)
