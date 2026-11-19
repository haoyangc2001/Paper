"""Basic API counterexamples only; no Isaac application or physics."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from camra_navigation.envs import clean_v3 as C


def test_direct_loop_initializes_missing_gym_step_buffer_without_stepping():
    env = SimpleNamespace(episode_length_buf=torch.zeros(2, dtype=torch.long))
    assert C.ensure_direct_reset_buffer(env)
    assert env.reset_buf.dtype == torch.bool
    assert env.reset_buf.tolist() == [False, False]


def test_existing_gym_buffer_is_preserved_for_normal_reset():
    buf = torch.tensor([True, False])
    env = SimpleNamespace(episode_length_buf=torch.zeros(2), reset_buf=buf)
    assert not C.ensure_direct_reset_buffer(env)
    assert env.reset_buf is buf


@pytest.mark.parametrize('buf', [torch.zeros(1, dtype=torch.bool), torch.zeros(2)])
def test_invalid_reset_buffer_is_rejected(buf):
    env = SimpleNamespace(episode_length_buf=torch.zeros(2), reset_buf=buf)
    with pytest.raises(ValueError, match='reset buffer'):
        C.ensure_direct_reset_buffer(env)


def test_subclass_initializes_before_invoking_sealed_reset(monkeypatch):
    env = SimpleNamespace(episode_length_buf=torch.zeros(1, dtype=torch.long))
    adapter = C.IsaacSubstepAdapter.__new__(C.IsaacSubstepAdapter)
    adapter.u = env
    def sealed_reset(self, *, start, yaw):
        self.u.reset_buf.zero_()
        return {'start':start, 'yaw':yaw}
    monkeypatch.setattr(C._V2Adapter, 'reset', sealed_reset)
    proof = adapter.reset(start=(1., 2.), yaw=.3)
    assert proof['direct_reset_buffer_initialized']
    assert proof['reset_buffer_after_reset'] == [False]


def test_rigid_body_view_needs_only_its_actual_get_transforms_api():
    class RigidBodyView:
        def get_transforms(self):
            return torch.tensor([[1., 2., 3., 0., 0., 0., 1.]])
    body = SimpleNamespace(root_physx_view=RigidBodyView())
    np.testing.assert_array_equal(C.rigid_body_transform(body), [1, 2, 3, 0, 0, 0, 1])


@pytest.mark.parametrize('pose', [[1., 2., 3.], [float('nan'), 0, 0, 0, 0, 0, 1]])
def test_missing_actual_pose_evidence_is_rejected(pose):
    view = SimpleNamespace(get_transforms=lambda:torch.tensor([pose]))
    with pytest.raises(ValueError, match='rigid-body transform'):
        C.rigid_body_transform(SimpleNamespace(root_physx_view=view))
