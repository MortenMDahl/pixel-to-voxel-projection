"""Extrinsics self-calibration from the tracked flying object.

The cameras point at the sky at 50 m+ range, which forces a baseline of metres
— far too wide for a calibration board. Instead the flying object itself is the
calibration target (see docs/adr/0001): during a calibration session the object
is flown through the shared view several times ("calibration passes"), each
camera records the time-stamped centroids of its motion masks (a "track"), and
the solve recovers each camera's pose from the paired tracks:

    time-paired centroids -> essential matrix -> relative rotation + direction
    measured baseline (tape/GPS)              -> metric scale
    ballistic acceleration of the passes      -> gravity -> world +Z up

World frame: origin at the reference camera's optical centre, +Z opposing
gravity, metres. Falls back to the reference camera's own frame when no
ballistic fit is possible. Extrinsics are written as 4x4 world->camera
matrices (``extrinsics_{port}.npy``), the same format the simulator exports.

Run an interactive calibration session (needs >=2 cameras and existing
intrinsics) with::

    python -m pixel_to_voxel.camera_extrinsics

Raw tracks are saved to ``calibration_data/`` too, so the solve can be re-run
later without re-flying anything::

    python -m pixel_to_voxel.camera_extrinsics --solve-only
"""

import argparse
import sys
import time

import cv2 as cv
import numpy as np

from . import settings
from . import camera_calibration


# ---------------------------------------------------------------------------
# Loading / saving
# ---------------------------------------------------------------------------

def load_extrinsics(ports):
    """Load the 4x4 world->camera extrinsic matrix for every port.

    Returns {port: (4, 4) ndarray}, or None if any camera's file is missing —
    projection needs the pose of *every* camera to be meaningful.
    """
    extrinsics = {}
    try:
        for port in ports:
            extrinsics[port] = np.load(f"{settings.CALIBRATION_DATA_PATH}extrinsics_{port}.npy")
    except FileNotFoundError:
        return None
    return extrinsics


def save_extrinsics(extrinsics):
    for port, matrix in extrinsics.items():
        np.save(f"{settings.CALIBRATION_DATA_PATH}extrinsics_{port}.npy", matrix)


def save_tracks(tracks):
    """Persist per-camera tracks as (N, 4) arrays of [pass_id, t, u, v]."""
    for port, track in tracks.items():
        np.save(f"{settings.CALIBRATION_DATA_PATH}tracks_{port}.npy", np.asarray(track, dtype=np.float64))


def load_tracks(ports):
    tracks = {}
    try:
        for port in ports:
            tracks[port] = np.load(f"{settings.CALIBRATION_DATA_PATH}tracks_{port}.npy")
    except FileNotFoundError:
        return None
    return tracks


def save_baselines(baselines):
    """Persist {port: metres} distances from the reference camera."""
    rows = np.array([[port, dist] for port, dist in baselines.items()], dtype=np.float64)
    np.save(f"{settings.CALIBRATION_DATA_PATH}baselines.npy", rows)


def load_baselines():
    try:
        rows = np.load(f"{settings.CALIBRATION_DATA_PATH}baselines.npy")
    except FileNotFoundError:
        return None
    return {int(port): float(dist) for port, dist in rows}


# ---------------------------------------------------------------------------
# Solving (pure math, testable without cameras)
# ---------------------------------------------------------------------------

def pair_tracks(track_a, track_b, tolerance=None):
    """Time-pair two (N, 4) [pass_id, t, u, v] tracks.

    Per pass, track_b's pixels are linearly interpolated onto track_a's
    timestamps, so the pairing is exact even when the cameras' frames
    interleave, and a constant timestamp offset between the cameras cancels
    to first order. Samples whose nearest track_b neighbour is further than
    ``tolerance`` seconds away (dropout holes, pass edges) are discarded.

    Returns (pts_a (M,2), pts_b (M,2), times (M,), pass_ids (M,)).
    """
    if tolerance is None:
        tolerance = settings.TRACK_PAIR_TOLERANCE
    track_a = np.asarray(track_a, dtype=np.float64)
    track_b = np.asarray(track_b, dtype=np.float64)
    pts_a, pts_b, times, pass_ids = [], [], [], []
    for pass_id in np.intersect1d(track_a[:, 0], track_b[:, 0]):
        a = track_a[track_a[:, 0] == pass_id]
        b = track_b[track_b[:, 0] == pass_id]
        if len(b) < 2:
            continue
        a = a[np.argsort(a[:, 1])]
        b = b[np.argsort(b[:, 1])]
        t_a, t_b = a[:, 1], b[:, 1]
        # Keep only track_a samples bracketed by track_b with a close neighbour
        nearest = np.searchsorted(t_b, t_a)
        gap = np.minimum(np.abs(t_b[np.clip(nearest, 0, len(t_b) - 1)] - t_a),
                         np.abs(t_b[np.clip(nearest - 1, 0, len(t_b) - 1)] - t_a))
        ok = (t_a >= t_b[0]) & (t_a <= t_b[-1]) & (gap <= tolerance)
        if not ok.any():
            continue
        interp = np.column_stack([np.interp(t_a[ok], t_b, b[:, 2]),
                                  np.interp(t_a[ok], t_b, b[:, 3])])
        pts_a.append(a[ok, 2:4])
        pts_b.append(interp)
        times.append(t_a[ok])
        pass_ids.append(a[ok, 0])
    if not pts_a:
        empty = np.empty((0, 2))
        return empty, empty.copy(), np.empty(0), np.empty(0)
    return (np.vstack(pts_a), np.vstack(pts_b),
            np.concatenate(times), np.concatenate(pass_ids))


def normalize_pixels(pixels, K):
    """Undistorted pixels -> normalized image coordinates via K^-1."""
    pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    return cv.undistortPoints(pts, np.asarray(K, dtype=np.float64), None).reshape(-1, 2)


def solve_relative_pose(pts_a, pts_b, K_a, K_b, pixel_threshold=None):
    """Recover camera b's pose relative to camera a from paired pixels.

    Returns (R, t_unit, inliers) with x_b = R @ x_a + t (OpenCV convention);
    ``t_unit`` has unit norm (scale is unobservable), ``inliers`` is a boolean
    mask over the input pairs that survived RANSAC + cheirality.
    """
    if pixel_threshold is None:
        pixel_threshold = settings.EXTRINSICS_RANSAC_THRESHOLD
    if len(pts_a) < 15:
        raise ValueError(f"Only {len(pts_a)} paired samples; need at least 15 to solve.")

    na = normalize_pixels(pts_a, K_a).reshape(-1, 1, 2)
    nb = normalize_pixels(pts_b, K_b).reshape(-1, 1, 2)
    # The RANSAC threshold is specified in normalized coordinates
    mean_focal = (K_a[0][0] + K_a[1][1] + K_b[0][0] + K_b[1][1]) / 4.0
    method = getattr(cv, "USAC_MAGSAC", cv.RANSAC)
    E, mask = cv.findEssentialMat(na, nb, np.eye(3), method=method,
                                  prob=0.9999, threshold=pixel_threshold / mean_focal)
    if E is None:
        raise ValueError("Essential matrix estimation failed; are the passes degenerate?")
    E = E[:3, :3]

    # RANSAC returns a minimal 5-point model; refit on all inliers (8-point,
    # then projected back onto the essential manifold) for a far less noisy
    # estimate — this matters in the weakly-conditioned wide-baseline regime.
    inliers = mask.ravel().astype(bool)
    if inliers.sum() >= 8:
        F, _ = cv.findFundamentalMat(na[inliers], nb[inliers], cv.FM_8POINT)
        if F is not None:
            U, _, Vt = np.linalg.svd(F[:3, :3])
            E = U @ np.diag([1.0, 1.0, 0.0]) @ Vt

    _, R, t, mask = cv.recoverPose(E, na, nb, np.eye(3), mask=mask)
    return R, t.ravel(), mask.ravel().astype(bool)


def triangulate_pairs(pts_a, pts_b, K_a, K_b, R, t):
    """Triangulate paired pixels into 3D points in camera a's frame."""
    P_a = np.asarray(K_a, dtype=np.float64) @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P_b = np.asarray(K_b, dtype=np.float64) @ np.hstack([R, np.reshape(t, (3, 1))])
    X = cv.triangulatePoints(P_a, P_b, np.asarray(pts_a, dtype=np.float64).T,
                             np.asarray(pts_b, dtype=np.float64).T)
    return (X[:3] / X[3]).T


def _project_pixels(K, R, t, points):
    """Project camera-a-frame points into a camera at pose (R, t) with intrinsics K."""
    cam = points @ R.T + t
    uvw = cam @ np.asarray(K, dtype=np.float64).T
    return uvw[:, :2] / uvw[:, 2:3]


def refine_ballistic(pts_a, pts_b, times, pass_ids, K_a, K_b, R, t_unit, baseline):
    """Ballistic bundle adjustment: refine the relative pose using the physics
    of the calibration passes.

    The epipolar solve leaves a weakly-observable rotation/depth trade-off in
    the wide-baseline, distant-points regime. Modelling each pass as quadratic
    motion X(tau) = p0 + v0*tau + 0.5*a*tau^2 with one shared acceleration and
    minimizing pixel reprojection in BOTH cameras over (rotation, translation
    direction, a, per-pass p0/v0, inter-camera time offset) pins that direction
    down. The time offset models the constant exposure-latency difference
    between the cameras, which timestamps alone cannot reveal but a moving
    target makes observable. Levenberg-Marquardt with numeric Jacobians; the
    translation length stays fixed at ``baseline``.

    Returns (R, t_unit, accel, time_offset, rms_px), or None if refinement
    failed to improve on the initial pose.
    """
    passes = np.unique(pass_ids)
    pass_index = np.searchsorted(passes, pass_ids)
    tau = np.empty_like(times)
    for k, p in enumerate(passes):
        sel = pass_ids == p
        tau[sel] = times[sel] - times[sel].min()

    # Initial structure and motion coefficients from the epipolar pose
    X0 = triangulate_pairs(pts_a, pts_b, K_a, K_b, R, t_unit * baseline)
    p0s, v0s, accels = [], [], []
    for p in passes:
        sel = pass_ids == p
        coeffs = np.polyfit(tau[sel], X0[sel], 2)      # rows: tau^2, tau, 1
        p0s.append(coeffs[2])
        v0s.append(coeffs[1])
        accels.append(2.0 * coeffs[0])

    w0 = cv.Rodrigues(np.asarray(R, dtype=np.float64))[0].ravel()
    theta0 = np.arccos(np.clip(t_unit[2], -1.0, 1.0))
    phi0 = np.arctan2(t_unit[1], t_unit[0])
    params = np.concatenate([w0, [theta0, phi0], np.mean(accels, axis=0), [0.0],
                             np.concatenate(p0s), np.concatenate(v0s)])
    n_passes = len(passes)
    observed = np.concatenate([np.asarray(pts_a).ravel(), np.asarray(pts_b).ravel()])

    def residuals(p):
        R_p = cv.Rodrigues(p[:3])[0]
        st, ct = np.sin(p[3]), np.cos(p[3])
        t_p = baseline * np.array([st * np.cos(p[4]), st * np.sin(p[4]), ct])
        accel = p[5:8]
        p0 = p[9:9 + 3 * n_passes].reshape(n_passes, 3)
        v0 = p[9 + 3 * n_passes:].reshape(n_passes, 3)
        # Camera b observes the trajectory shifted by its latency offset p[8]
        tau_b = (tau + p[8])[:, None]
        X_a = (p0[pass_index] + v0[pass_index] * tau[:, None]
               + 0.5 * accel * tau[:, None] ** 2)
        X_b = p0[pass_index] + v0[pass_index] * tau_b + 0.5 * accel * tau_b ** 2
        proj = np.concatenate([
            _project_pixels(K_a, np.eye(3), np.zeros(3), X_a).ravel(),
            _project_pixels(K_b, R_p, t_p, X_b).ravel(),
        ])
        return proj - observed

    def jacobian(p, r0):
        J = np.empty((len(r0), len(p)))
        for j in range(len(p)):
            eps = 1e-6 * max(1.0, abs(p[j]))
            probe = p.copy()
            probe[j] += eps
            J[:, j] = (residuals(probe) - r0) / eps
        return J

    r = residuals(params)
    cost = r @ r
    initial_cost = cost
    lam = 1e-3
    for _ in range(30):
        J = jacobian(params, r)
        JTJ = J.T @ J
        JTr = J.T @ r
        improved = False
        for _ in range(8):
            damped = JTJ + lam * np.diag(np.diag(JTJ).clip(min=1e-12))
            try:
                step = np.linalg.solve(damped, -JTr)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            candidate = params + step
            r_new = residuals(candidate)
            cost_new = r_new @ r_new
            if cost_new < cost:
                relative_drop = 1.0 - cost_new / cost if cost > 0 else 1.0
                params, r, cost = candidate, r_new, cost_new
                lam = max(lam / 3.0, 1e-9)
                improved = True
                break
            lam *= 10.0
        if not improved:
            break
        if cost < 1e-12 or relative_drop < 1e-9:
            break

    if cost >= initial_cost:
        return None
    R_ref = cv.Rodrigues(params[:3])[0]
    st, ct = np.sin(params[3]), np.cos(params[3])
    t_ref = np.array([st * np.cos(params[4]), st * np.sin(params[4]), ct])
    accel = params[5:8]
    rms_px = float(np.sqrt(cost / (len(r) / 2.0)))
    return R_ref, t_ref, accel, float(params[8]), rms_px


def fit_gravity(points, times, pass_ids):
    """Fit the acceleration of each calibration pass and average them.

    ``points`` are (M, 3) triangulated positions with matching ``times`` and
    ``pass_ids``. Returns (accel (3,) mean acceleration vector, g_est float)
    or (None, None) if no pass has enough samples for a quadratic fit.
    """
    accels = []
    for pass_id in np.unique(pass_ids):
        sel = pass_ids == pass_id
        if sel.sum() < settings.MIN_PASS_SAMPLES:
            continue
        t = times[sel]
        if t.max() - t.min() < 0.2:
            continue
        coeffs = np.polyfit(t - t.min(), points[sel], 2)   # (3, 3): rows are t^2, t, 1
        accels.append(2.0 * coeffs[0])
    if not accels:
        return None, None
    accel = np.mean(accels, axis=0)
    return accel, float(np.linalg.norm(accel))


def gravity_alignment(accel):
    """Rotation taking reference-camera coordinates to the gravity-aligned
    world frame: rows are the world axes expressed in camera coordinates.

    World +Z opposes the fitted acceleration (up); world +X is the camera's
    image-right projected onto the horizontal plane, so the frame stays
    intuitive when the camera points near-vertically.
    """
    up = -np.asarray(accel, dtype=np.float64)
    up /= np.linalg.norm(up)
    x = np.array([1.0, 0.0, 0.0]) - up[0] * up   # camera +X made horizontal
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0.0, 1.0, 0.0]) - up[1] * up
    x /= np.linalg.norm(x)
    y = np.cross(up, x)                          # right-handed: X x Y = Z
    return np.stack([x, y, up], axis=0)


def assemble_extrinsics(ref_port, relative_poses, R_world_from_ref):
    """Build 4x4 world->camera matrices from reference-relative poses.

    ``relative_poses`` maps port -> (R, t_metric) with x_cam = R @ x_ref + t.
    The world origin is the reference camera's optical centre.
    """
    extrinsics = {}
    ref = np.eye(4)
    ref[:3, :3] = R_world_from_ref.T             # world -> reference camera
    extrinsics[ref_port] = ref
    for port, (R, t) in relative_poses.items():
        M = np.eye(4)
        M[:3, :3] = R @ R_world_from_ref.T
        M[:3, 3] = t
        extrinsics[port] = M
    return extrinsics


def solve_extrinsics(tracks, intrinsics, baselines, use_gravity=True):
    """Full solve: tracks + intrinsics + measured baselines -> extrinsics.

    tracks     -- {port: (N, 4) [pass_id, t, u, v]} undistorted-pixel tracks
    intrinsics -- {port: 3x3 K}
    baselines  -- {port: metres from the reference camera} for every
                  non-reference port
    Returns (extrinsics {port: 4x4 world->camera}, diagnostics dict).
    """
    ports = sorted(tracks.keys())
    ref = ports[0]
    diagnostics = {"reference_port": ref, "pairs": {}, "gravity_aligned": False,
                   "g_est": None}

    relative_poses = {}
    ref_accel = None
    ref_gravity_data = None
    for port in ports[1:]:
        pts_r, pts_p, times, pass_ids = pair_tracks(tracks[ref], tracks[port])
        R, t_unit, inliers = solve_relative_pose(pts_r, pts_p,
                                                 intrinsics[ref], intrinsics[port])
        baseline = float(baselines[port])
        # The epipolar pose only initializes; the ballistic bundle adjustment
        # resolves the rotation/depth ambiguity of the wide-baseline regime.
        refined = refine_ballistic(pts_r[inliers], pts_p[inliers], times[inliers],
                                   pass_ids[inliers], intrinsics[ref],
                                   intrinsics[port], R, t_unit, baseline)
        rms_px = None
        time_offset = None
        if refined is not None:
            R, t_unit, accel, time_offset, rms_px = refined
            if ref_accel is None:
                ref_accel = accel
        t = t_unit * baseline
        relative_poses[port] = (R, t)
        diagnostics["pairs"][port] = {
            "paired_samples": int(len(pts_r)),
            "inlier_fraction": float(inliers.mean()),
            "passes": int(len(np.unique(pass_ids))),
            "refined_rms_px": rms_px,
            "time_offset_s": time_offset,
        }
        if ref_gravity_data is None:
            points = triangulate_pairs(pts_r[inliers], pts_p[inliers],
                                       intrinsics[ref], intrinsics[port], R, t)
            ref_gravity_data = (points, times[inliers], pass_ids[inliers])

    R_world_from_ref = np.eye(3)
    if use_gravity:
        # Prefer the acceleration recovered by the bundle adjustment; fall back
        # to fitting the triangulated points when refinement didn't run.
        accel, g_est = ref_accel, None
        if accel is not None:
            g_est = float(np.linalg.norm(accel))
        elif ref_gravity_data is not None:
            accel, g_est = fit_gravity(*ref_gravity_data)
        if accel is not None:
            R_world_from_ref = gravity_alignment(accel)
            diagnostics["gravity_aligned"] = True
            diagnostics["g_est"] = g_est

    return assemble_extrinsics(ref, relative_poses, R_world_from_ref), diagnostics


def report_diagnostics(diagnostics):
    """Print the solve diagnostics, warning on the known failure signals."""
    for port, stats in diagnostics["pairs"].items():
        print(f"Camera {diagnostics['reference_port']} <-> {port}: "
              f"{stats['paired_samples']} paired samples across {stats['passes']} passes, "
              f"{stats['inlier_fraction']:.0%} inliers")
        if stats["refined_rms_px"] is not None:
            print(f"  ballistic refinement: {stats['refined_rms_px']:.2f} px reprojection RMS, "
                  f"inter-camera latency {stats['time_offset_s'] * 1000.0:+.1f} ms")
        else:
            print("  WARNING: ballistic refinement did not improve the epipolar pose; "
                  "extrinsics come from the (noisier) essential-matrix solve alone.")
        if stats["passes"] < 2:
            print("  WARNING: a single pass is planar and can degrade the solve; "
                  "record passes along different paths.")
        if stats["inlier_fraction"] < 0.5:
            print("  WARNING: low inlier fraction; check camera sync and detections.")
    if diagnostics["gravity_aligned"]:
        g = diagnostics["g_est"]
        print(f"Fitted ballistic acceleration: {g:.2f} m/s^2 "
              f"(expected {settings.GRAVITY_MS2})")
        if abs(g - settings.GRAVITY_MS2) > settings.GRAVITY_TOLERANCE * settings.GRAVITY_MS2:
            print("  WARNING: fitted |g| is far from 9.81 — the baseline measurement, "
                  "camera sync, or tracking is likely off. Extrinsics scale is suspect.")
    else:
        print("No gravity alignment (no usable ballistic pass); "
              "world frame = reference camera frame.")


# ---------------------------------------------------------------------------
# Interactive calibration session
# ---------------------------------------------------------------------------

def mask_centroid(mask):
    """Centroid (u, v) of the largest plausible blob in a binary mask, or None."""
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv.contourArea)
    if cv.contourArea(largest) < settings.EXTRINSICS_MIN_CONTOUR_AREA:
        return None
    moments = cv.moments(largest)
    if moments["m00"] == 0:
        return None
    return (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])


def detect_centroid(gray_new, gray_old):
    """Motion-mask centroid of the largest blob, or None if nothing plausible."""
    diff = cv.absdiff(gray_new, gray_old)
    _, mask = cv.threshold(diff, settings.PIXEL_NOISE_THRESHOLD, 255, cv.THRESH_BINARY)
    return mask_centroid(mask)


def prompt_baselines(ports):
    """Ask for the measured reference<->camera distances, offering saved values."""
    saved = load_baselines() or {}
    baselines = {}
    ref = sorted(ports)[0]
    for port in sorted(ports)[1:]:
        default = saved.get(port)
        suffix = f" [{default}]" if default is not None else ""
        answer = input(f"Measured distance between camera {ref} and camera {port} "
                       f"in metres{suffix}: ").strip()
        if answer:
            baselines[port] = float(answer)
        elif default is not None:
            baselines[port] = default
        else:
            sys.exit("A measured baseline is required to fix the metric scale.")
    return baselines


def run_capture_session(ports, calibration_data):
    """Live capture: undistort, detect, and record time-stamped tracks.

    Keys (with any camera window focused):
        p — start / stop a calibration pass (only active passes are recorded)
        q — end the session
    Returns {port: list of [pass_id, t, u, v]}.
    """
    from .main import CameraStream   # imported lazily to avoid a module cycle

    streams = {port: CameraStream(port).start() for port in ports}
    gray_old = {port: None for port in ports}
    tracks = {port: [] for port in ports}
    trails = {port: [] for port in ports}
    recording = False
    pass_id = -1

    print("Fly the object through the shared view. 'p' starts/stops a pass "
          "(do several, along different paths), 'q' ends the session.")
    try:
        while True:
            now = time.perf_counter()
            for port in ports:
                frame = streams[port].read()
                if frame is None:
                    continue
                frame = cv.undistort(frame, calibration_data[port]["camera_matrix"],
                                     calibration_data[port]["dist_coeffs"])
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                if gray_old[port] is not None:
                    centroid = detect_centroid(gray, gray_old[port])
                    if centroid is not None:
                        trails[port] = (trails[port] + [centroid])[-30:]
                        if recording:
                            tracks[port].append([pass_id, now, centroid[0], centroid[1]])
                gray_old[port] = gray

                for u, v in trails[port]:
                    cv.circle(frame, (int(round(u)), int(round(v))), 3, (0, 0, 255), -1)
                state = f"PASS {pass_id} RECORDING" if recording else "idle"
                cv.putText(frame, f"{state} | samples: {len(tracks[port])}",
                           (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv.imshow(f"Extrinsics camera {port}", frame)

            key = cv.waitKey(1) & 0xFF
            if key == ord("p"):
                recording = not recording
                if recording:
                    pass_id += 1
                    print(f"Recording pass {pass_id}...")
                else:
                    print(f"Pass {pass_id} stopped.")
            elif key == ord("q"):
                break
    finally:
        for stream in streams.values():
            stream.stop()
        cv.destroyAllWindows()
    return tracks


def main():
    parser = argparse.ArgumentParser(
        description="Self-calibrate camera extrinsics from a tracked flying object.")
    parser.add_argument("--solve-only", action="store_true",
                        help="Skip capture; re-run the solve on tracks saved by a "
                             "previous session.")
    args = parser.parse_args()

    ports = camera_calibration.list_camera_ports()
    if len(ports) < 2:
        sys.exit(f"{len(ports)} ports found, need at least 2.")

    calibration_data = camera_calibration.load_calibration_data()
    if calibration_data is None:
        sys.exit("Intrinsics not found — run python -m pixel_to_voxel.camera_calibration first.")

    if args.solve_only:
        tracks = load_tracks(ports)
        if tracks is None:
            sys.exit("No saved tracks found; run a capture session first.")
    else:
        baselines = prompt_baselines(ports)
        save_baselines(baselines)
        tracks = run_capture_session(ports, calibration_data)
        tracks = {port: np.asarray(track, dtype=np.float64).reshape(-1, 4)
                  for port, track in tracks.items()}
        save_tracks(tracks)   # saved before solving so a failed solve loses nothing

    baselines = load_baselines()
    if baselines is None:
        sys.exit("No saved baselines found.")

    intrinsics = {port: calibration_data[port]["camera_matrix"] for port in ports}
    try:
        extrinsics, diagnostics = solve_extrinsics(tracks, intrinsics, baselines)
    except ValueError as e:
        sys.exit(f"Solve failed: {e}")

    report_diagnostics(diagnostics)
    save_extrinsics(extrinsics)
    print(f"Extrinsics written to {settings.CALIBRATION_DATA_PATH} "
          f"for cameras {sorted(extrinsics.keys())}.")


if __name__ == "__main__":
    main()
