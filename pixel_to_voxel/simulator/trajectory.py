"""Analytic (scripted) object trajectories.

Deterministic, closed-form motion — no physics engine. Each function returns an
(N, 3) array of world-space object-centre positions, one row per frame, which is
also saved as the ground-truth ``object_positions.npy``.
"""

import numpy as np


def parabola(p0, v0, num_frames, duration, g=(0.0, 0.0, -9.81)):
    """Projectile arc: p(t) = p0 + v0 t + 0.5 g t^2, sampled over [0, duration].

    With the default Z-up gravity this is a ballistic parabola, a natural test
    motion for a "flying object". ``g`` can be zeroed for straight-line motion.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    ts = np.linspace(0.0, duration, num_frames)[:, None]      # (N,1)
    return p0 + v0 * ts + 0.5 * g * ts ** 2


def linear(p0, p1, num_frames):
    """Constant-velocity straight line from p0 to p1 (inclusive)."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    s = np.linspace(0.0, 1.0, num_frames)[:, None]
    return p0 + (p1 - p0) * s
