"""Inter-camera extrinsics (world->camera poses) for the projection stage.

Only *loading* is implemented so far. The ``extrinsics_{id}.npy`` files (4x4
world->camera matrices, OpenCV convention) currently come from the simulator's
ground-truth export; copy them into ``calibration_data/`` alongside the
intrinsics to enable voxel projection in ``main.py``.
"""

import numpy as np

from . import settings


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


# TODO: Implement camera extrinsic calculation for real cameras (recover these
# matrices from mutual views instead of relying on the simulator's export).
