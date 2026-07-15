"""Ground-truth geometry check for the simulator (no rendering required).

Validates the core promise of the simulator: the exported intrinsics K and
extrinsics [R|t] are mutually consistent, so a 3D point projected into two
cameras can be triangulated back to its original location. This exercises the
exact math the pixel-to-voxel projection stage depends on.

Run as a plain script (matching the existing test convention):

    python tests/test_simulator_geometry.py
"""

import os
import sys

import numpy as np

sys.path.append(os.getcwd())

from pixel_to_voxel.simulator.rig import CameraRig
from pixel_to_voxel.simulator import trajectory


def triangulate(cameras, pixels):
    """Linear DLT triangulation of one point from N cameras.

    ``pixels`` is a list of (u, v) observations aligned with ``cameras``.
    Returns the recovered 3D world point.
    """
    rows = []
    for cam, (u, v) in zip(cameras, pixels):
        P = cam.projection_matrix()          # 3x4
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    A = np.stack(rows, axis=0)
    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    return X[:3] / X[3]


def main():
    rig = CameraRig.from_positions(
        positions=[(0.0, -4.0, 1.2), (3.5, -2.0, 1.2)],
        target=(0.0, 0.0, 0.5),
        width=640, height=480, fov_deg=60.0,
    )

    positions = trajectory.parabola(
        p0=(-1.5, 0.0, 0.2), v0=(1.5, 0.0, 3.5), num_frames=30, duration=0.9)

    max_err = 0.0
    in_view = 0
    for pt in positions:
        pixels = []
        visible = True
        for cam in rig.cameras:
            px, valid = cam.project(pt)
            u, v = px[0]
            # Point must be in front of the camera and inside the image.
            if not valid[0] or not (0 <= u < cam.width and 0 <= v < cam.height):
                visible = False
            pixels.append((u, v))
        if not visible:
            continue
        in_view += 1
        recovered = triangulate(rig.cameras, pixels)
        max_err = max(max_err, np.linalg.norm(recovered - pt))

    print(f"Frames observed by both cameras: {in_view}/{len(positions)}")
    print(f"Max triangulation error: {max_err:.2e} m")

    assert in_view >= len(positions) // 2, "Trajectory largely out of view; check rig setup."
    assert max_err < 1e-6, f"Triangulation error too large: {max_err}"
    print("PASS: intrinsics/extrinsics are self-consistent.")


if __name__ == "__main__":
    main()
