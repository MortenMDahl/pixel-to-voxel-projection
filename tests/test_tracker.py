"""Ground-truth check for the object tracker (no rendering, no cameras).

Projects the simulator's default ballistic arc into both default cameras,
triangulates the (noisy) pixel centroids back with tracker.triangulate_center,
and runs the constant-acceleration Kalman filter on them. The filtered state
must recover position, speed, heading, and climb rate of the analytic
trajectory — including across a mid-flight measurement gap.

Run as a plain script (matching the existing test convention):

    python tests/test_tracker.py
"""

import os
import sys

import numpy as np

sys.path.append(os.getcwd())

from pixel_to_voxel import settings
from pixel_to_voxel.tracker import KalmanTracker, triangulate_center
from pixel_to_voxel.simulator.rig import CameraRig
from pixel_to_voxel.simulator import trajectory

GRAVITY = np.array([0.0, 0.0, -9.81])
SETTLE_FRAMES = 20          # skip the filter's convergence transient
GAP = range(80, 96)         # frames with no measurement (detection dropout)
GAP_RECOVERY = 5            # frames allowed to re-converge after the gap


def build_scene():
    rig = CameraRig.from_positions(
        positions=settings.SIM_CAMERA_POSITIONS,
        target=settings.SIM_LOOK_AT,
        width=settings.SIM_IMAGE_WIDTH,
        height=settings.SIM_IMAGE_HEIGHT,
        fov_deg=settings.SIM_FOV_DEG,
    )
    cameras = {cam.id: {"camera_matrix": cam.K, "extrinsic": cam.extrinsic_4x4()}
               for cam in rig.cameras}
    positions = trajectory.parabola(
        p0=settings.SIM_TRAJECTORY_P0, v0=settings.SIM_TRAJECTORY_V0,
        num_frames=settings.SIM_NUM_FRAMES,
        duration=settings.SIM_TRAJECTORY_DURATION)
    times = np.linspace(0.0, settings.SIM_TRAJECTORY_DURATION, settings.SIM_NUM_FRAMES)
    return rig, cameras, positions, times


def main():
    rig, cameras, positions, times = build_scene()
    v0 = np.asarray(settings.SIM_TRAJECTORY_V0, dtype=np.float64)

    # Noiseless triangulation must reproduce the point to numerical precision
    exact = {cam.id: tuple(cam.project(positions[0])[0][0]) for cam in rig.cameras}
    err = np.linalg.norm(triangulate_center(exact, cameras) - positions[0])
    assert err < 1e-6, f"Noiseless triangulation error {err}"

    rng = np.random.default_rng(0)
    track = KalmanTracker()
    errors = {"pos": [], "speed": [], "heading": [], "climb": []}
    for i, (point, t) in enumerate(zip(positions, times)):
        if i in GAP:
            continue
        centroids = {}
        for cam in rig.cameras:
            pix, valid = cam.project(point)
            assert valid[0], f"Frame {i} behind camera {cam.id}; check the scene"
            centroids[cam.id] = tuple(pix[0] + rng.normal(0.0, 0.3, 2))
        track.update(triangulate_center(centroids, cameras), t)

        if i < SETTLE_FRAMES or (GAP.start <= i < GAP.stop + GAP_RECOVERY):
            continue
        v_true = v0 + GRAVITY * t
        errors["pos"].append(np.linalg.norm(track.position - point))
        errors["speed"].append(abs(track.speed - np.linalg.norm(v_true)))
        heading_true = np.degrees(np.arctan2(v_true[0], v_true[1])) % 360.0
        heading_err = abs(track.heading_deg - heading_true)
        errors["heading"].append(min(heading_err, 360.0 - heading_err))
        errors["climb"].append(abs(track.climb_rate - v_true[2]))

    worst = {key: max(vals) for key, vals in errors.items()}
    print(f"Evaluated frames: {len(errors['pos'])} (gap of {len(GAP)} frames mid-flight)")
    print("Worst errors | position: {pos:.2f} m | speed: {speed:.2f} m/s | "
          "heading: {heading:.2f} deg | climb: {climb:.2f} m/s".format(**worst))

    assert track.ready
    assert worst["pos"] < 1.0, f"Position error {worst['pos']}"
    assert worst["speed"] < 1.5, f"Speed error {worst['speed']}"
    assert worst["heading"] < 5.0, f"Heading error {worst['heading']}"
    assert worst["climb"] < 1.5, f"Climb-rate error {worst['climb']}"
    print("PASS: tracker recovers position, speed, heading, and climb from noisy centroids.")


if __name__ == "__main__":
    main()
