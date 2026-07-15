"""Synthetic multi-camera simulator for the pixel-to-voxel pipeline.

Generates time-synchronised RGB streams of a scripted moving object from two or
more virtual pinhole cameras, and exports exact ground-truth intrinsics and
extrinsics. Use it to develop and validate the projection stage without physical
cameras.

Layout
------
* ``rig``        — camera intrinsics/extrinsics + calibration export (NumPy only)
* ``trajectory`` — analytic object motion (parabola, line)
* ``renderer``   — pyrender offscreen rendering (optional ``[sim]`` extras)
* ``stream``     — ``SimulatedStream``, a drop-in for ``main.CameraStream``

Run ``python -m pixel_to_voxel.simulator`` to generate a dataset.
"""

from .rig import Camera, CameraRig, intrinsic_matrix, look_at_extrinsic
from . import trajectory
from .stream import SimulatedStream

__all__ = [
    "Camera",
    "CameraRig",
    "intrinsic_matrix",
    "look_at_extrinsic",
    "trajectory",
    "SimulatedStream",
]
