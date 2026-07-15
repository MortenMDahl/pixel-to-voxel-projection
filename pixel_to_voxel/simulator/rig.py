"""Camera rig geometry for the synthetic simulator.

Pure NumPy — no rendering dependencies — so the camera math (intrinsics,
extrinsics, projection) can be imported and unit-tested without pyrender.

Conventions
-----------
* World frame is **Z-up** (gravity points along -Z), which suits a flying /
  projectile object.
* Cameras are ideal pinholes: zero distortion. This means the ``dist_coeffs``
  we export are all zeros, so ``cv.undistort`` in ``main.py`` is a safe no-op.
* Camera axes follow the **OpenCV** convention: +X right, +Y down, +Z forward
  (looking into the scene). This matches ``cv.calibrateCamera`` / ``cv.undistort``
  so the exported matrices drop straight into the existing pipeline.
"""

import os
from dataclasses import dataclass

import numpy as np

# Converts an OpenCV camera-to-world pose (X right, Y down, Z forward) into the
# OpenGL/pyrender convention (X right, Y up, Z backward). See renderer.py.
_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def intrinsic_matrix(width, height, fov_deg):
    """Build a 3x3 pinhole intrinsic matrix K from a horizontal field of view.

    Square pixels (fx == fy) and a centered principal point are assumed, which
    is exactly what an ideal virtual camera produces.
    """
    fov_rad = np.deg2rad(fov_deg)
    fx = (width / 2.0) / np.tan(fov_rad / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def look_at_extrinsic(eye, target, world_up=(0.0, 0.0, 1.0)):
    """Return world->camera rotation R (3x3) and translation t (3,) for a camera
    at ``eye`` looking at ``target``, in OpenCV camera axes.

    X_camera = R @ X_world + t
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    world_up = np.asarray(world_up, dtype=np.float64)

    z = target - eye                      # camera +Z looks toward the target
    z /= np.linalg.norm(z)

    x = np.cross(z, world_up)             # camera +X points right
    if np.linalg.norm(x) < 1e-8:
        # Forward is parallel to world_up; fall back to an alternate up vector.
        x = np.cross(z, np.array([0.0, 1.0, 0.0]))
    x /= np.linalg.norm(x)

    y = np.cross(z, x)                    # camera +Y points down (right-handed)

    R = np.stack([x, y, z], axis=0)       # rows are the camera axes in world coords
    t = -R @ eye
    return R, t


@dataclass
class Camera:
    """A single synthetic pinhole camera with exact known geometry."""

    id: int
    width: int
    height: int
    K: np.ndarray            # 3x3 intrinsics
    R: np.ndarray            # 3x3 world->camera rotation (OpenCV axes)
    t: np.ndarray            # 3, world->camera translation
    eye: np.ndarray          # 3, camera centre in world coords

    @property
    def dist_coeffs(self):
        """Ideal pinhole: no lens distortion."""
        return np.zeros(5, dtype=np.float64)

    def extrinsic_4x4(self):
        """World->camera transform as a 4x4 matrix (OpenCV convention)."""
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = self.R
        M[:3, 3] = self.t
        return M

    def cam2world(self):
        """Camera->world transform (inverse of the extrinsic), OpenCV convention."""
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = self.R.T
        M[:3, 3] = self.eye
        return M

    def pyrender_pose(self):
        """Camera->world pose in the OpenGL convention pyrender expects.

        pyrender cameras look down -Z with +Y up, whereas we work in OpenCV
        axes (+Z forward, +Y down). Post-multiplying by ``_CV_TO_GL`` flips the
        Y and Z axes so the rendered image matches the exported intrinsics.
        This coordinate flip is the single most common source of silent errors
        in Blender/pyrender synthetic-data pipelines, so it is centralised here.
        """
        return self.cam2world() @ _CV_TO_GL

    def projection_matrix(self):
        """3x4 projection matrix P = K [R | t] mapping world points to pixels."""
        return self.K @ np.hstack([self.R, self.t.reshape(3, 1)])

    def project(self, points_world):
        """Project (N,3) world points to (N,2) pixel coordinates.

        Returns pixels and a boolean mask of points that lie in front of the
        camera (positive depth). Points behind the camera are still returned but
        flagged invalid.
        """
        points_world = np.atleast_2d(np.asarray(points_world, dtype=np.float64))
        cam = (self.R @ points_world.T).T + self.t          # (N,3) in camera frame
        depth = cam[:, 2]
        uvw = (self.K @ cam.T).T                             # (N,3)
        with np.errstate(divide="ignore", invalid="ignore"):
            pixels = uvw[:, :2] / uvw[:, 2:3]
        return pixels, depth > 0


class CameraRig:
    """A collection of synthetic cameras that all observe the same scene."""

    def __init__(self, cameras):
        self.cameras = list(cameras)

    @classmethod
    def from_positions(cls, positions, target, width, height, fov_deg,
                       world_up=(0.0, 0.0, 1.0)):
        """Build a rig from a list of camera ``positions`` all aimed at ``target``.

        Cameras are given integer ids 0, 1, 2, ... matching the pipeline's
        per-camera file naming.
        """
        K = intrinsic_matrix(width, height, fov_deg)
        cameras = []
        for i, pos in enumerate(positions):
            R, t = look_at_extrinsic(pos, target, world_up)
            cameras.append(Camera(
                id=i, width=width, height=height,
                K=K.copy(), R=R, t=t, eye=np.asarray(pos, dtype=np.float64),
            ))
        return cls(cameras)

    def save(self, output_dir):
        """Write exact ground-truth calibration to ``output_dir`` as .npy files.

        Intrinsics use the *same* file names as ``camera_calibration.py`` so they
        can be copied into ``calibration_data/`` and consumed unchanged:

            camera_matrix_{id}.npy   3x3 intrinsics
            dist_coeffs_{id}.npy     zeros (ideal pinhole)

        Extrinsics are the ground truth the (still-stubbed) ``camera_extrinsics``
        stage is meant to recover, written as 4x4 world->camera matrices:

            extrinsics_{id}.npy      4x4 world->camera (OpenCV convention)
        """
        os.makedirs(output_dir, exist_ok=True)
        for cam in self.cameras:
            np.save(os.path.join(output_dir, f"camera_matrix_{cam.id}.npy"), cam.K)
            np.save(os.path.join(output_dir, f"dist_coeffs_{cam.id}.npy"), cam.dist_coeffs)
            np.save(os.path.join(output_dir, f"extrinsics_{cam.id}.npy"), cam.extrinsic_4x4())
