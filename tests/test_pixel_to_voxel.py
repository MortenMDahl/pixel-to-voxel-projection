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
from pixel_to_voxel.simulator.rig import (Camera, CameraRig, intrinsic_matrix,
                                          look_at_extrinsic)
from pixel_to_voxel.simulator import trajectory

SPHERE_RADIUS = settings.SIM_OBJECT_RADIUS  # match the simulated object


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

    # Tolerances scale with the grid: ~2 voxels of centroid slack, and a
    # spread bound covering the two-view hull's lens-shaped elongation
    # (~2.5x the object radius at the rig's ~50 deg baseline angle) plus
    # half a voxel diagonal of discretisation.
    assert worst_centroid_err < 2.0 * settings.VOXEL_SIZE, \
        f"Centroid strayed from the object: {worst_centroid_err}"
    assert worst_spread < 3.0 * SPHERE_RADIUS + settings.VOXEL_SIZE, \
        f"Voxels far outside the two-view hull: {worst_spread}"

    # Empty masks must carve away everything
    empty = {cam.id: np.zeros((cam.height, cam.width), np.uint8) for cam in rig.cameras}
    assert len(pixel_to_voxel(empty, cameras)) == 0, "Empty masks left occupied voxels."

    print("PASS: pixel_to_voxel() recovers the object location from exact masks.")


def divergent_rig():
    """Three cameras aimed at different points — the expanded-FOV rig shape."""
    positions = [(-10.0, -48.0, 1.5), (0.0, -48.0, 1.5), (10.0, -48.0, 1.5)]
    targets = [(-20.0, 0.0, 20.0), (0.0, 0.0, 20.0), (20.0, 0.0, 20.0)]
    cameras = []
    for i, (pos, target) in enumerate(zip(positions, targets)):
        R, t = look_at_extrinsic(pos, target)
        K = intrinsic_matrix(settings.SIM_IMAGE_WIDTH, settings.SIM_IMAGE_HEIGHT,
                             settings.SIM_FOV_DEG)
        cameras.append(Camera(id=i, width=settings.SIM_IMAGE_WIDTH,
                              height=settings.SIM_IMAGE_HEIGHT, K=K, R=R, t=t,
                              eye=np.asarray(pos)))
    return CameraRig(cameras)


def test_partial_visibility():
    """The visibility-aware rule on a divergent rig (see docs/adr/0003):
    voxels supported by every camera that sees them survive with only two
    views, and a contradicting third view kills two-view ghosts."""
    rig = divergent_rig()
    cameras = {cam.id: {"camera_matrix": cam.K, "extrinsic": cam.extrinsic_4x4()}
               for cam in rig.cameras}

    def masks_for(point, supporting_ids):
        masks = {}
        for cam in rig.cameras:
            if cam.id in supporting_ids:
                masks[cam.id] = silhouette_mask(cam, point, SPHERE_RADIUS)
                assert masks[cam.id].any(), f"camera {cam.id} cannot see {point}"
            else:
                masks[cam.id] = np.zeros((cam.height, cam.width), np.uint8)
        return masks

    # A point outside camera 0's view but seen by cameras 1 and 2: occupied.
    # (The old every-camera rule would have carved it for lacking cam-0 support.)
    edge_point = np.array([14.0, 10.0, 20.0])
    pix, valid = rig.cameras[0].project(edge_point)
    u, v = pix[0]
    assert not (valid[0] and 0 <= u < rig.cameras[0].width
                and 0 <= v < rig.cameras[0].height), "point unexpectedly in cam 0 view"
    voxels = pixel_to_voxel(masks_for(edge_point, {1, 2}), cameras)
    assert len(voxels), "two-camera support in the overlap zone left no voxels"
    # Cameras 1 and 2 subtend only ~10 deg at this point, so the two-view hull
    # is a long thin lens (~2r/sin(theta) ~ 5 m); its centroid stays centred.
    centroid_err = np.linalg.norm(voxels.mean(axis=0) - edge_point)
    spread = np.linalg.norm(voxels - edge_point, axis=1).max()
    assert centroid_err < 2.0, f"edge-zone hull centroid off by {centroid_err} m"
    assert spread < 6.0, f"voxels far outside the narrow-baseline hull: {spread}"

    # A ghost: cameras 0 and 1 'support' a point that camera 2 also sees —
    # but camera 2 shows background there, so the intersection must be carved.
    ghost_point = np.array([-5.0, 10.0, 20.0])
    for cam in rig.cameras:
        pix, valid = cam.project(ghost_point)
        u, v = pix[0]
        assert valid[0] and 0 <= u < cam.width and 0 <= v < cam.height, \
            f"ghost point must be visible to camera {cam.id}"
    voxels = pixel_to_voxel(masks_for(ghost_point, {0, 1}), cameras)
    near_ghost = (np.linalg.norm(voxels - ghost_point, axis=1) < 3.0).sum() if len(voxels) else 0
    assert near_ghost == 0, f"{near_ghost} ghost voxels survived a contradicting view"

    print("PASS: visibility-aware carving handles the divergent rig (edge zones + ghost veto).")


if __name__ == "__main__":
    main()
    test_partial_visibility()
