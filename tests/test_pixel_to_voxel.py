"""Visual-hull carving check for pixel_to_voxel() (no rendering, no cameras).

Feeds pixel_to_voxel() ideal synthetic masks — a filled disc per camera at the
exact projection of a small sphere flying along the default trajectory — built
from the simulator's ground-truth rig geometry. The occupied voxels must track
the sphere: with only two views the hull is a lens-shaped intersection of two
tangent cones, so individual voxels may sit a couple of radii from the centre,
but the centroid has to stay close to it.

Run as a plain script (matching the existing test convention):

    python tests/test_pixel_to_voxel.py
"""

import os
import sys

import cv2 as cv
import numpy as np

sys.path.append(os.getcwd())

from pixel_to_voxel import settings
from pixel_to_voxel.main import pixel_to_voxel
from pixel_to_voxel.simulator.rig import CameraRig
from pixel_to_voxel.simulator import trajectory

SPHERE_RADIUS = 0.15  # metres, matching the renderer's default object


def silhouette_mask(cam, center_world, radius):
    """Binary mask with the projected silhouette of a sphere: a filled disc at
    the projected centre, pixel radius fx * r / depth (small-angle model)."""
    mask = np.zeros((cam.height, cam.width), dtype=np.uint8)
    pixels, valid = cam.project(center_world)
    if not valid[0]:
        return mask
    depth = (cam.R @ np.asarray(center_world, dtype=np.float64) + cam.t)[2]
    pixel_radius = max(int(round(cam.K[0, 0] * radius / depth)), 1)
    u = int(round(float(pixels[0][0])))
    v = int(round(float(pixels[0][1])))
    cv.circle(mask, (u, v), pixel_radius, 255, thickness=-1)
    return mask


def main():
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
        p0=settings.SIM_TRAJECTORY_P0,
        v0=settings.SIM_TRAJECTORY_V0,
        num_frames=10,
        duration=settings.SIM_TRAJECTORY_DURATION,
    )

    worst_centroid_err = 0.0
    worst_spread = 0.0
    for point in positions:
        masks = {cam.id: silhouette_mask(cam, point, SPHERE_RADIUS)
                 for cam in rig.cameras}
        assert all(m.any() for m in masks.values()), "Sphere out of view; check rig setup."

        voxels = pixel_to_voxel(masks, cameras)
        assert len(voxels) > 0, f"No voxels recovered at {point}"

        worst_centroid_err = max(worst_centroid_err,
                                 np.linalg.norm(voxels.mean(axis=0) - point))
        worst_spread = max(worst_spread,
                           np.linalg.norm(voxels - point, axis=1).max())

    print(f"Worst centroid error over {len(positions)} frames: {worst_centroid_err:.3f} m")
    print(f"Worst occupied-voxel distance from true centre: {worst_spread:.3f} m")

    assert worst_centroid_err < 0.075, f"Centroid strayed from the object: {worst_centroid_err}"
    assert worst_spread < 0.5, f"Voxels far outside the two-view hull: {worst_spread}"

    # Empty masks must carve away everything
    empty = {cam.id: np.zeros((cam.height, cam.width), np.uint8) for cam in rig.cameras}
    assert len(pixel_to_voxel(empty, cameras)) == 0, "Empty masks left occupied voxels."

    print("PASS: pixel_to_voxel() recovers the object location from exact masks.")


if __name__ == "__main__":
    main()
