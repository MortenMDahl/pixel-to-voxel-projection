"""Tracking of flying objects' centres (multi-target, see docs/adr/0003).

Per-camera motion-mask centroids ("detections") are associated to predicted
targets camera-by-camera, triangulated into 3D centre points — sub-voxel
measurements independent of the voxel grid resolution — and filtered per
target with a constant-acceleration Kalman filter. The filtered state gives
position, velocity, and the derived speed / heading / climb rate.

Association is track-oriented global-nearest-neighbour: each target's
prediction is projected into every camera that can see it and matched to
detections by pixel distance (Hungarian assignment); leftover detections can
spawn new tentative targets via pairwise epipolar gating, vetoed when a third
camera should see the candidate point but detects nothing there. Targets seen
by fewer than two cameras coast ballistically on prediction.

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

    def predict_position(self, timestamp):
        """Ballistic position prediction at ``timestamp`` (no state change)."""
        dt = float(timestamp) - self.last_time
        return self.x[:3] + self.x[3:6] * dt + 0.5 * self.x[6:9] * dt * dt


def hungarian(cost):
    """Minimum-cost assignment for a small dense cost matrix.

    Classic potentials/augmenting-path algorithm, O(n^3) — trivially fast for
    the <=15x15 matrices association produces, and avoids a scipy dependency.
    Returns [(row, col), ...] covering every row (or column, whichever is
    fewer); the caller filters out gated (large-cost) pairs afterwards.
    """
    cost = np.asarray(cost, dtype=np.float64)
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T
    n, m = cost.shape
    INF = 1e18
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    assigned = np.zeros(m + 1, dtype=int)     # column -> 1-based row
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        assigned[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = assigned[j0]
            delta, j1 = INF, 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[assigned[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if assigned[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            assigned[j0] = assigned[j1]
            j0 = j1
    pairs = [(assigned[j] - 1, j - 1) for j in range(1, m + 1) if assigned[j]]
    return [(c, r) for r, c in pairs] if transposed else pairs


def fundamental_matrix(K_a, extrinsic_a, K_b, extrinsic_b):
    """F with x_b^T F x_a = 0 for homogeneous pixel coordinates."""
    T = np.asarray(extrinsic_b) @ np.linalg.inv(np.asarray(extrinsic_a))
    R, t = T[:3, :3], T[:3, 3]
    tx = np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])
    return np.linalg.inv(np.asarray(K_b)).T @ tx @ R @ np.linalg.inv(np.asarray(K_a))


def epipolar_distance_px(F, pt_a, pt_b):
    """Symmetric point-to-epipolar-line distance in pixels."""
    xa = np.array([pt_a[0], pt_a[1], 1.0])
    xb = np.array([pt_b[0], pt_b[1], 1.0])
    line_b = F @ xa
    line_a = F.T @ xb
    d_b = abs(xb @ line_b) / max(np.hypot(line_b[0], line_b[1]), 1e-12)
    d_a = abs(xa @ line_a) / max(np.hypot(line_a[0], line_a[1]), 1e-12)
    return 0.5 * (d_a + d_b)


def merge_close_detections(detections, radius_px=None):
    """Collapse per-camera detections closer than ``radius_px`` into their mean.

    Frame differencing marks the union of an object's silhouette at two
    consecutive frames; a mover faster than its own diameter per frame splits
    into leading/trailing blobs a few pixels apart. Merged, they yield one
    centroid per object instead of debris that spawns phantom twins.
    """
    if radius_px is None:
        radius_px = settings.DETECTION_MERGE_PX
    merged = {}
    for port, dets in detections.items():
        clusters = []
        for det in dets:
            for cluster in clusters:
                center = np.mean(cluster, axis=0)
                if np.hypot(center[0] - det[0], center[1] - det[1]) < radius_px:
                    cluster.append(det)
                    break
            else:
                clusters.append([det])
        merged[port] = [tuple(np.mean(c, axis=0)) for c in clusters]
    return merged


class _Target:
    """One tracked object: a Kalman filter plus lifecycle bookkeeping."""

    def __init__(self, target_id, kalman, timestamp):
        self.id = target_id
        self.filter = kalman
        self.last_seen = timestamp
        self.confirmed = False
        self.cameras = []          # ports with a matched detection this frame


class MultiTargetTracker:
    """Track-oriented multi-target manager over per-camera detections.

    cameras -- {port: {"camera_matrix": 3x3 K, "extrinsic": 4x4 world->cam}},
    fixed for the tracker's lifetime. Image sizes are inferred from the
    principal point (cx, cy) — exact for the simulator, and close enough for
    real calibrated cameras where the gates are tens of pixels wide.
    """

    _GATE_LARGE = 1e6      # cost placeholder for gated-out pairs

    def __init__(self, cameras):
        self.cameras = {port: {"camera_matrix": np.asarray(c["camera_matrix"], dtype=np.float64),
                               "extrinsic": np.asarray(c["extrinsic"], dtype=np.float64)}
                        for port, c in cameras.items()}
        self.image_size = {port: (2.0 * c["camera_matrix"][0, 2],
                                  2.0 * c["camera_matrix"][1, 2])
                           for port, c in self.cameras.items()}
        ports = sorted(self.cameras)
        self.pairs = [(a, b) for i, a in enumerate(ports) for b in ports[i + 1:]]
        self.F = {(a, b): fundamental_matrix(self.cameras[a]["camera_matrix"],
                                             self.cameras[a]["extrinsic"],
                                             self.cameras[b]["camera_matrix"],
                                             self.cameras[b]["extrinsic"])
                  for a, b in self.pairs}
        lo = np.asarray(settings.VOXEL_GRID_MIN, dtype=np.float64)
        hi = np.asarray(settings.VOXEL_GRID_MAX, dtype=np.float64)
        margin = 0.25 * (hi - lo)
        self._world_lo, self._world_hi = lo - margin, hi + margin
        self.targets = []
        self._next_id = 1

    def _project(self, port, point, margin=10.0):
        """Pixel projection of a world point, or None if the camera can't see it."""
        cam = self.cameras[port]
        p = cam["extrinsic"][:3, :3] @ point + cam["extrinsic"][:3, 3]
        if p[2] <= 0:
            return None
        uvw = cam["camera_matrix"] @ p
        u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]
        width, height = self.image_size[port]
        if -margin <= u < width + margin and -margin <= v < height + margin:
            return (u, v)
        return None

    def step(self, detections, timestamp):
        """Advance one frame. detections: {port: [(u, v), ...]}.

        Returns {"confirmed": [ids], "deleted": [ids]} for event logging.
        """
        events = {"confirmed": [], "deleted": []}
        detections = merge_close_detections(detections)
        used = {port: set() for port in detections}
        matches = {id(t): {} for t in self.targets}

        # 1. Per-camera assignment of detections to predicted targets
        for port, dets in detections.items():
            if not dets:
                continue
            visible = []
            for target in self.targets:
                pixel = self._project(port, target.filter.predict_position(timestamp))
                if pixel is not None:
                    visible.append((target, pixel))
            if not visible:
                continue
            cost = np.full((len(visible), len(dets)), self._GATE_LARGE)
            for i, (_, pixel) in enumerate(visible):
                for j, det in enumerate(dets):
                    d = np.hypot(pixel[0] - det[0], pixel[1] - det[1])
                    if d < settings.TARGET_GATE_PX:
                        cost[i, j] = d
            for i, j in hungarian(cost):
                if cost[i, j] < settings.TARGET_GATE_PX:
                    matches[id(visible[i][0])][port] = dets[j]
                    used[port].add(j)

        # 2. Update matched targets; coast the rest
        for target in self.targets:
            m = matches[id(target)]
            target.cameras = sorted(m)
            if len(m) >= 2:
                target.filter.update(triangulate_center(m, self.cameras), timestamp)
                target.last_seen = timestamp
                if (not target.confirmed
                        and target.filter.n_updates >= settings.TARGET_CONFIRM_UPDATES):
                    target.confirmed = True
                    events["confirmed"].append(target.id)
            elif len(m) == 1:
                # Seen by one camera: alive, but coasting (no triangulation)
                target.last_seen = timestamp

        # 3. Spawn tentative targets from leftover detections (epipolar pairing)
        candidates = []
        for a, b in self.pairs:
            for i in range(len(detections.get(a, []))):
                if i in used[a]:
                    continue
                for j in range(len(detections.get(b, []))):
                    if j in used[b]:
                        continue
                    d = epipolar_distance_px(self.F[(a, b)],
                                             detections[a][i], detections[b][j])
                    if d < settings.EPIPOLAR_GATE_PX:
                        candidates.append((d, a, i, b, j))
        occupied_points = [t.filter.predict_position(timestamp) for t in self.targets]
        for d, a, i, b, j in sorted(candidates, key=lambda c: c[0]):
            if i in used[a] or j in used[b]:
                continue
            point = triangulate_center({a: detections[a][i], b: detections[b][j]},
                                       self.cameras)
            if not (np.all(point >= self._world_lo) and np.all(point <= self._world_hi)):
                continue
            # Detections near an existing target — or near a spawn from this
            # very step — are its debris (e.g. the trailing blob when frame
            # differencing splits a fast mover in two), not a new object
            if any(np.linalg.norm(point - p) < settings.SPAWN_SUPPRESSION_RADIUS_M
                   for p in occupied_points):
                continue
            if self._vetoed_by_third_camera(point, (a, b), detections):
                continue
            occupied_points.append(point)
            used[a].add(i)
            used[b].add(j)
            kalman = KalmanTracker()
            kalman.update(point, timestamp)
            target = _Target(self._next_id, kalman, timestamp)
            target.cameras = [a, b]
            self._next_id += 1
            self.targets.append(target)

        # 4. Delete targets unseen for too long
        remaining = []
        for target in self.targets:
            if timestamp - target.last_seen > settings.TARGET_DELETE_AFTER_S:
                if target.confirmed:
                    events["deleted"].append(target.id)
            else:
                remaining.append(target)
        self.targets = remaining
        return events

    def _vetoed_by_third_camera(self, point, spawn_ports, detections):
        """A candidate is a likely ghost if another camera should see it but
        has no detection anywhere near its projection."""
        for port in self.cameras:
            if port in spawn_ports:
                continue
            pixel = self._project(port, point, margin=-10.0)
            if pixel is None:
                continue
            near = any(np.hypot(pixel[0] - d[0], pixel[1] - d[1])
                       < 3.0 * settings.TARGET_GATE_PX
                       for d in detections.get(port, []))
            if not near:
                return True
        return False

    def state_list(self):
        """Confirmed targets as JSON-able dicts for the dashboard."""
        state = []
        for target in self.targets:
            if not target.confirmed:
                continue
            f = target.filter
            state.append({
                "id": target.id,
                "position": [round(float(c), 2) for c in f.position],
                "velocity": [round(float(c), 2) for c in f.velocity],
                "speed": round(f.speed, 2),
                "heading": round(f.heading_deg, 1),
                "climb": round(f.climb_rate, 2),
                "cameras": list(target.cameras),
            })
        return state
