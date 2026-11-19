"""Basic synthetic/negative checks only; no fit on the eighteen saved traces."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/nav043_motion_response.py'
SPEC = importlib.util.spec_from_file_location('nav043_motion_response', SOURCE)
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


def fixture(path, ticks=45):
    initial = np.array([1., 2., math.pi - .02])
    commands = np.repeat([[.3, -.1, .4]], ticks, axis=0)
    pose = initial.copy()
    rows = []
    for i, command in enumerate(commands):
        pose = m.step(pose, command, .005)
        saved = pose.copy()
        saved[2] = m.wrap(saved[2])
        rows.append({'physics_tick': i + 1, 'env_id': 0, 't_s': (i + 1) * .005,
                     'state': saved.tolist(), 'applied': command.tolist(),
                     'label': 'timeout' if i == ticks - 1 else None})
    nav = [{'physics_tick': i, 'navigation_step': i // 20,
            'command': [commands[i].tolist()]} for i in range(0, ticks, 20)]
    result = {'method': 'full', 'map_id': 'synthetic', 'physics_steps': ticks,
              't_s': ticks * .005, 'terminal': rows[-1], 'warmup': {
                  'physics_steps': 200, 'state_at_window_start': [
                      dict(zip(('x', 'y', 'yaw'), initial.tolist()))]}}
    for name, data in [('result.json', result), ('substeps.json', rows), ('navigation.json', nav)]:
        (path / name).write_text(json.dumps(data))
    return {'trial': str(path), 'method': 'full', 'map_id': 'synthetic',
            'level': 'L1_empty', 'kind': 'baseline'}


class SE2Tests(unittest.TestCase):
    def test_straight_with_rotated_body_axes(self):
        np.testing.assert_allclose(m.step([0., 0., math.pi / 2], [1., .2, 0.], 2.),
                                   [-.4, 2., math.pi / 2], atol=1.e-12)

    def test_analytic_quarter_circle(self):
        np.testing.assert_allclose(m.step([0., 0., 0.], [1., 0., 1.], math.pi / 2),
                                   [1., 1., math.pi / 2], atol=1.e-12)

    def test_log_exp_roundtrip_small_and_finite_turn(self):
        for rate in (0., 1.e-8, -.8, .9):
            start = np.array([3., -2., .7])
            command = np.array([.4, -.2, rate])
            np.testing.assert_allclose(m.interval_twist(start, m.step(start, command, .1)),
                                       command, atol=1.e-12)

    def test_yaw_wrap_crossing(self):
        start = np.array([0., 0., math.pi - .01])
        end = m.step(start, [.2, .1, .5], .1)
        end[2] = m.wrap(end[2])
        np.testing.assert_allclose(m.interval_twist(start, end), [.2, .1, .5], atol=1.e-12)

    def test_ambiguous_rotation_refused(self):
        with self.assertRaises(ValueError):
            m.interval_twist(np.zeros(3), np.array([0., 0., 2.]))


class TraceTests(unittest.TestCase):
    def test_partial_interval_kept_out_of_fitting(self):
        with tempfile.TemporaryDirectory() as folder:
            entry = fixture(Path(folder))
            trace = m.read_trace(entry)
            self.assertEqual(trace.ticks, 45)
            self.assertEqual(len(trace.observed_twists), 2)
            self.assertEqual(len(trace.interval_commands), 3)
            np.testing.assert_allclose(trace.observed_twists, np.tile([.3, -.1, .4], (2, 1)), atol=1.e-12)
            ideal, _ = m.free_replay(trace)
            np.testing.assert_allclose(ideal[:, :2], trace.actual[:, :2], atol=1.e-12)

    def test_broken_clock_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            entry = fixture(path)
            rows = m.load(path / 'substeps.json')
            rows[3]['physics_tick'] = 3
            (path / 'substeps.json').write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, 'physics tick'):
                m.read_trace(entry)

    def test_hidden_midinterval_command_change_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            entry = fixture(path)
            rows = m.load(path / 'substeps.json')
            rows[5]['applied'][0] += .01
            (path / 'substeps.json').write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, 'command not held'):
                m.read_trace(entry)

    def test_no_future_state_reset_during_free_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            trace = m.read_trace(fixture(Path(folder)))
            parameters = np.array([[0., .8], [0., .8], [0., .8]])
            before, _ = m.free_replay(trace, parameters)
            trace.actual[:] = 123.
            after, _ = m.free_replay(trace, parameters)
            np.testing.assert_array_equal(before, after)

    def test_unknown_initial_response_is_zero_not_first_observation(self):
        with tempfile.TemporaryDirectory() as folder:
            trace = m.read_trace(fixture(Path(folder)))
            parameters = np.array([[.5, .5], [.5, .5], [.5, .5]])
            _, responses = m.free_replay(trace, parameters)
            np.testing.assert_allclose(responses[0], .5 * trace.interval_commands[0])
            row = m.compare_trace(trace, parameters)
            self.assertFalse(row['prefixes']['5']['complete'])


class SolveTests(unittest.TestCase):
    def test_synthetic_positive_stable_model_one_solve(self):
        x = np.array([[1., 0.], [0., 1.], [1., 1.], [2., -1.]])
        with patch.object(m.np.linalg, 'lstsq', wraps=np.linalg.lstsq) as solve:
            result = m.fit_axis(x, x @ [.4, .5])
            self.assertEqual(solve.call_count, 1)
        self.assertTrue(result['valid'])
        np.testing.assert_allclose([result['a'], result['b']], [.4, .5], atol=1.e-12)

    def test_nonstable_model_refused_without_clipping(self):
        x = np.array([[1., 0.], [0., 1.], [1., 1.]])
        result = m.fit_axis(x, x @ [1.2, .5])
        self.assertFalse(result['valid'])
        self.assertGreater(result['a'], 1.)

    def test_unidentifiable_axis_refused(self):
        result = m.fit_axis(np.ones((5, 2)), np.ones(5))
        self.assertFalse(result['valid'])
        self.assertEqual(result['rank'], 1)


class GateTests(unittest.TestCase):
    @staticmethod
    def rows(errors=(.8, .8, .8)):
        return [{'level': level, 'terminal': {
            'ideal': {'position_squared_error_sum': 100., 'ticks': 100},
            'response': {'position_squared_error_sum': 100. * error**2, 'ticks': 100}}}
            for level, error in zip(m.LEVELS, errors) for _ in range(6)]

    def test_improvement_requires_every_map_not_worse(self):
        self.assertFalse(m.grouped_gate(self.rows((.1, .1, 1.01)))['diagnostic_pass'])

    def test_no_aggregate_improvement_refused(self):
        self.assertFalse(m.grouped_gate(self.rows((1., 1., 1.)))['diagnostic_pass'])

    def test_missing_failed_trace_cannot_be_dropped(self):
        with self.assertRaises(ValueError):
            m.grouped_gate(self.rows()[:-1])

    def test_diagnostic_pass_never_passes_g5(self):
        gate = m.grouped_gate(self.rows())
        self.assertTrue(gate['diagnostic_pass'])
        self.assertFalse(gate['NAV043_passed'])
        self.assertFalse(gate['G5_passed'])

    def test_missing_approval_refuses_before_loading_or_solving(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(m, 'APPROVAL', Path(folder) / 'missing.json'):
                with patch.object(m, 'verify_protocol', return_value={}):
                    with patch.object(m.socket, 'gethostname', return_value='tiger11'):
                        with patch.dict(m.os.environ, {'TMUX': 'synthetic'}):
                            with patch.object(m.sys, 'executable', m.PYTHON):
                                with patch.object(m.sys, 'argv', ['probe', '--run']):
                                    with patch.object(m, 'read_trace') as read:
                                        with patch.object(m, 'fit_axis') as solve:
                                            with self.assertRaisesRegex(ValueError, 'not approved'):
                                                m.main()
                                            read.assert_not_called()
                                            solve.assert_not_called()


if __name__ == '__main__':
    unittest.main()
