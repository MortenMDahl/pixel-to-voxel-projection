"""Ground-truth check for extrinsics self-calibration (no rendering, no cameras).

Simulates the real calibration procedure at operating range: a wide-baseline
sky-pointing rig watches a ballistic object cross the shared view on several
calibration passes. The passes are projected into pixel tracks (optionally with
noise, camera clock offset, and frame dropout), fed to solve_extrinsics(), and
the recovered geometry is compared against the rig's ground truth.

The solver's world frame (origin at camera 0, +Z from the gravity fit) matches
the rig's world only up to a yaw about Z — yaw is inherently unobservable — so
comparisons align that single gauge freedom first.

Run as a plain script (matching the existing test convention):

    python tests/test_camera_extrinsics.py
"""

import os
import sys

import numpy as np

sys.path.append(os.getcwd())

from pixel_to_voxel.camera_extrinsics import solve_extrinsics
from pixel_to_voxel.simulator.rig import CameraRig
from pixel_to_voxel.simulator import trajectory

BASELINE = 14.0          # metres between the two cameras
FPS = 30.0
FRAMES_PER_PASS = 75

# Calibration passes: ballistic arcs crossing the shared view along different
# paths (a single arc is planar and would degenerate the essential matrix).
PASSES = [
    dict(p0=(-15.0, 75.0, 35.0), v0=(12.0, 3.0, 6.0)),
    dict(p0=(25.0, 60.0, 50.0), v0=(-14.0, 6.0, -1.0)),
    dict(p0=(5.0, 90.0, 25.0), v0=(2.0, -8.0, 14.0)),
]
HELD_OUT = dict(p0=(-10.0, 85.0, 40.0), v0=(10.0, -4.0, 5.0))
PASS_DURATION = 2.5


def build_rig():
    return CameraRig.from_positions(
        positions=[(0.0, 0.0, 0.0), (BASELINE, 0.0, 0.0)],
        target=(7.0, 70.0, 45.0),
        width=640, height=480, fov_deg=60.0,
    )


def project_track(cam, points, times, pass_id, rng=None, noise_px=0.0,
                  time_offset=0.0, time_jitter=0.0, dropout=0.0):
    """Project world points into one camera's (N, 4) [pass, t, u, v] track,
    keeping only samples the camera actually sees."""
    pixels, in_front = cam.project(points)
    inside = (in_front
              & (pixels[:, 0] >= 0) & (pixels[:, 0] < cam.width)
              & (pixels[:, 1] >= 0) & (pixels[:, 1] < cam.height))
    pixels, t = pixels[inside], np.asarray(times)[inside]
    if rng is not None:
        if noise_px:
            pixels = pixels + rng.normal(0.0, noise_px, pixels.shape)
        t = t + time_offset + (rng.normal(0.0, time_jitter, t.shape) if time_jitter else 0.0)
        if dropout:
            keep = rng.random(len(pixels)) > dropout
            pixels, t = pixels[keep], t[keep]
    return np.column_stack([np.full(len(pixels), pass_id), t, pixels])


def make_tracks(rig, **noise):
    """Tracks for both cameras over all calibration passes."""
    rng = np.random.default_rng(noise.pop("seed", 0)) if noise else None
    tracks = {0: [], 1: []}
    for pass_id, spec in enumerate(PASSES):
        points = trajectory.parabola(spec["p0"], spec["v0"],
                                     num_frames=FRAMES_PER_PASS, duration=PASS_DURATION)
        times = pass_id * 30.0 + np.linspace(0.0, PASS_DURATION, FRAMES_PER_PASS)
        for cam in rig.cameras:
            cam_noise = noise if cam.id == 1 else {k: noise[k] for k in ("noise_px",) if k in noise}
            tracks[cam.id].append(project_track(cam, points, times, pass_id,
                                                rng=rng, **cam_noise))
    return {port: np.vstack(parts) for port, parts in tracks.items()}


def rotation_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def yaw_align(from_pts, to_pts):
    """Best rotation about Z taking ``from_pts`` onto ``to_pts`` (shared origin/up)."""
    cross = np.sum(from_pts[:, 0] * to_pts[:, 1] - from_pts[:, 1] * to_pts[:, 0])
    dot = np.sum(from_pts[:, 0] * to_pts[:, 0] + from_pts[:, 1] * to_pts[:, 1])
    theta = np.arctan2(cross, dot)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def triangulate_world(extrinsics, rig, points_world):
    """Project a ground-truth trajectory and triangulate it back through the
    *recovered* extrinsics; returns points in the solver's world frame."""
    import cv2 as cv
    pix = [cam.project(points_world)[0] for cam in rig.cameras]
    P = [np.asarray(rig.cameras[i].K) @ extrinsics[i][:3, :] for i in range(2)]
    X = cv.triangulatePoints(P[0], P[1], pix[0].T, pix[1].T)
    return (X[:3] / X[3]).T


def run_case(name, tracks, rig, tol_rot_deg, tol_c1z, tol_g, tol_rms,
             expected_dt=None):
    intrinsics = {cam.id: cam.K for cam in rig.cameras}
    extrinsics, diag = solve_extrinsics(tracks, intrinsics, {1: BASELINE})

    for port, stats in diag["pairs"].items():
        print(f"[{name}] pair 0<->{port}: {stats['paired_samples']} samples, "
              f"{stats['inlier_fraction']:.0%} inliers, {stats['passes']} passes, "
              f"refined RMS {stats['refined_rms_px']} px, "
              f"latency {stats['time_offset_s']} s")
        if expected_dt is not None:
            assert abs(stats["time_offset_s"] - expected_dt) < 0.003, (
                f"[{name}] latency estimate {stats['time_offset_s']} "
                f"vs injected {expected_dt}")

    # Relative rotation between the cameras is world-frame independent.
    R_rel = extrinsics[1][:3, :3] @ extrinsics[0][:3, :3].T
    R_rel_gt = rig.cameras[1].R @ rig.cameras[0].R.T
    rot_err = rotation_angle_deg(R_rel @ R_rel_gt.T)

    # Camera 1's centre in the solver world: length is fixed by the baseline,
    # and gravity alignment must make the (physically level) baseline horizontal.
    C1 = -extrinsics[1][:3, :3].T @ extrinsics[1][:3, 3]
    g_est = diag["g_est"]

    # End-to-end: triangulate a held-out trajectory through the recovered
    # extrinsics and compare to ground truth after fixing the yaw gauge.
    gt = trajectory.parabola(HELD_OUT["p0"], HELD_OUT["v0"],
                             num_frames=FRAMES_PER_PASS, duration=PASS_DURATION)
    recovered = triangulate_world(extrinsics, rig, gt)
    Rz = yaw_align(gt, recovered)
    rms = float(np.sqrt(np.mean(np.sum((recovered - gt @ Rz.T) ** 2, axis=1))))

    print(f"[{name}] rotation error: {rot_err:.3f} deg | baseline length: "
          f"{np.linalg.norm(C1):.3f} m | C1 z: {C1[2]:+.3f} m | "
          f"g: {g_est:.2f} m/s^2 | held-out RMS: {rms:.2f} m")

    assert diag["gravity_aligned"], f"[{name}] gravity alignment did not run"
    assert rot_err < tol_rot_deg, f"[{name}] rotation error {rot_err}"
    assert abs(np.linalg.norm(C1) - BASELINE) < 1e-6, f"[{name}] baseline length {np.linalg.norm(C1)}"
    assert abs(C1[2]) < tol_c1z, f"[{name}] baseline not horizontal: C1 z = {C1[2]}"
    assert abs(g_est - 9.81) < tol_g * 9.81, f"[{name}] fitted g {g_est}"
    assert rms < tol_rms, f"[{name}] held-out trajectory RMS {rms}"


def main():
    rig = build_rig()

    # Sanity: the passes must actually cross the shared view.
    for spec in PASSES + [HELD_OUT]:
        pts = trajectory.parabola(spec["p0"], spec["v0"],
                                  num_frames=FRAMES_PER_PASS, duration=PASS_DURATION)
        for cam in rig.cameras:
            pix, valid = cam.project(pts)
            inside = (valid & (pix[:, 0] >= 0) & (pix[:, 0] < cam.width)
                      & (pix[:, 1] >= 0) & (pix[:, 1] < cam.height))
            assert inside.mean() > 0.6, f"Pass {spec} barely visible from camera {cam.id}"

    run_case("noiseless", make_tracks(rig), rig,
             tol_rot_deg=0.05, tol_c1z=0.05, tol_g=0.01, tol_rms=0.3)

    # Camera 1's stamps run 4 ms late, so its pixels correspond to 4 ms
    # earlier along the trajectory: the solver should report dt ~ -0.004.
    run_case("noisy", make_tracks(rig, seed=0, noise_px=0.4, time_offset=0.004,
                                  time_jitter=0.002, dropout=0.1), rig,
             tol_rot_deg=0.5, tol_c1z=0.5, tol_g=0.08, tol_rms=3.0,
             expected_dt=-0.004)

    print("PASS: self-calibrated extrinsics match the rig ground truth.")


if __name__ == "__main__":
    main()
