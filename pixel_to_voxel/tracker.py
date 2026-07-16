"""Single-target tracking of the flying object's centre.

Triangulates per-camera motion-mask centroids into a 3D centre point — a
sub-voxel measurement independent of the voxel grid resolution — and filters
it with a constant-acceleration Kalman filter. The filtered state gives the
position, velocity, and the derived speed / heading / climb rate.

Heading is a compass-style bearing of the horizontal velocity in the
gravity-aligned world frame: 0 deg = world +Y ("grid north"), clockwise,
090 deg = world +X. With self-calibrated extrinsics, "grid north" is
rig-relative, not true north.

Frame-differencing masks mark the union of the object's silhouette at two
consecutive frames, so a centroid measurement effectively refers to about half
a frame earlier than its timestamp; at 30 fps this sub-voxel lag is accepted.
"""

import numpy as np

from . import settings


def triangulate_center(centroids, cameras):
    """DLT-triangulate one world point from per-camera pixel centroids.

    centroids -- {port: (u, v)} undistorted pixel centroid, from >= 2 cameras
    cameras   -- {port: {"camera_matrix": 3x3 K, "extrinsic": 4x4 world->cam}}
    Returns the (3,) world-frame centre.
    """
    rows = []
    for port, (u, v) in centroids.items():
        K = np.asarray(cameras[port]["camera_matrix"], dtype=np.float64)
        extrinsic = np.asarray(cameras[port]["extrinsic"], dtype=np.float64)
        P = K @ extrinsic[:3, :]
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.stack(rows))
    X = vt[-1]
    return X[:3] / X[3]


class KalmanTracker:
    """Constant-acceleration Kalman filter over the object centre.

    State: [position(3), velocity(3), acceleration(3)] in world coordinates.
    Feed ``update(measurement, timestamp)`` once per measured frame; missed
    frames need no special handling — the next update's larger dt widens the
    prediction accordingly.
    """

    def __init__(self, measurement_noise=None, process_noise=None):
        self.r = (measurement_noise if measurement_noise is not None
                  else settings.TRACKER_MEASUREMENT_NOISE)
        self.q = (process_noise if process_noise is not None
                  else settings.TRACKER_PROCESS_NOISE)
        self.x = None
        self.P = None
        self.last_time = None
        self.n_updates = 0

    def update(self, measurement, timestamp):
        z = np.asarray(measurement, dtype=np.float64)
        if self.x is None:
            # First measurement pins the position; velocity/acceleration start
            # unknown with generous uncertainty and converge over a few frames.
            self.x = np.zeros(9)
            self.x[:3] = z
            self.P = np.diag([self.r ** 2] * 3 + [50.0 ** 2] * 3 + [20.0 ** 2] * 3)
            self.last_time = float(timestamp)
            self.n_updates = 1
            return

        dt = float(timestamp) - self.last_time
        if dt <= 0:
            return
        self.last_time = float(timestamp)

        # Predict: per-axis constant-acceleration model, white-jerk noise
        F1 = np.array([[1.0, dt, 0.5 * dt * dt],
                       [0.0, 1.0, dt],
                       [0.0, 0.0, 1.0]])
        Q1 = self.q * np.array([
            [dt ** 5 / 20.0, dt ** 4 / 8.0, dt ** 3 / 6.0],
            [dt ** 4 / 8.0, dt ** 3 / 3.0, dt ** 2 / 2.0],
            [dt ** 3 / 6.0, dt ** 2 / 2.0, dt],
        ])
        F = np.kron(F1, np.eye(3))
        x = F @ self.x
        P = F @ self.P @ F.T + np.kron(Q1, np.eye(3))

        # Update with the position measurement (H selects the first 3 states)
        S = P[:3, :3] + (self.r ** 2) * np.eye(3)
        gain = P[:, :3] @ np.linalg.inv(S)
        self.x = x + gain @ (z - x[:3])
        self.P = P - gain @ P[:3, :]
        self.n_updates += 1

    @property
    def ready(self):
        """True once enough updates have converged the velocity estimate."""
        return self.n_updates >= settings.TRACKER_MIN_UPDATES

    @property
    def position(self):
        return None if self.x is None else self.x[:3]

    @property
    def velocity(self):
        return None if self.x is None else self.x[3:6]

    @property
    def speed(self):
        return None if self.x is None else float(np.linalg.norm(self.x[3:6]))

    @property
    def heading_deg(self):
        """Bearing of horizontal travel: 0 = +Y, clockwise, 090 = +X."""
        if self.x is None:
            return None
        return float(np.degrees(np.arctan2(self.x[3], self.x[4])) % 360.0)

    @property
    def climb_rate(self):
        return None if self.x is None else float(self.x[5])
