"""Isaac 0.47.2 API repair; the sealed clean_v2 implementation stays intact."""
import numpy as np
import torch

from .clean_v2 import configure_clean  # re-export the unchanged clean contract
from .clean_v2 import IsaacSubstepAdapter as _V2Adapter


def ensure_direct_reset_buffer(env):
    """The manager-based gym step normally creates this buffer on its first call.

    The direct substep loop never calls gym.step, so initialize its local
    bookkeeping explicitly, without executing a hidden physical step.
    """
    created = not hasattr(env, 'reset_buf')
    if created:
        env.reset_buf = torch.zeros_like(env.episode_length_buf, dtype=torch.bool)
    if env.reset_buf.shape != env.episode_length_buf.shape or env.reset_buf.dtype != torch.bool:
        raise ValueError('invalid direct-loop reset buffer')
    return created


def rigid_body_transform(body, env_index=0):
    """Read the actual RigidBodyView transform, in its native xyz/xyzw format."""
    pose = body.root_physx_view.get_transforms()[env_index].detach().cpu().numpy().copy()
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError('invalid actual rigid-body transform')
    return pose


class IsaacSubstepAdapter(_V2Adapter):
    def reset(self, *, start, yaw):
        created = ensure_direct_reset_buffer(self.u)
        proof = super().reset(start=start, yaw=yaw)
        proof['direct_reset_buffer_initialized'] = created
        proof['reset_buffer_after_reset'] = self.u.reset_buf.detach().cpu().tolist()
        return proof
