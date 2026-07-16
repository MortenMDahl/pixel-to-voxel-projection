"""CLI entry point: generate a synthetic multi-camera dataset.

    python -m pixel_to_voxel.simulator [options]

Writes, under the output directory:
    camera_matrix_{id}.npy   3x3 intrinsics   (same names as calibration_data/)
    dist_coeffs_{id}.npy     zeros (ideal pinhole)
    extrinsics_{id}.npy      4x4 world->camera ground truth
    object_positions.npy     (N,3) ground-truth object centre per frame
    cam{id}/frame_{i:04d}.png rendered frames

Rendering needs the optional extras:  pip install -e ".[sim]"
The calibration/geometry export works with NumPy alone (use --no-render).
"""

import argparse
import os

import numpy as np

from .. import settings as S
from .rig import CameraRig
from . import trajectory


def build_default_rig():
    return CameraRig.from_positions(
        positions=S.SIM_CAMERA_POSITIONS,
        target=S.SIM_LOOK_AT,
        width=S.SIM_IMAGE_WIDTH,
        height=S.SIM_IMAGE_HEIGHT,
        fov_deg=S.SIM_FOV_DEG,
    )


def build_default_trajectories(num_frames):
    """(num_objects, num_frames, 3) ground-truth positions from settings."""
    return np.stack([
        trajectory.parabola(p0=spec["p0"], v0=spec["v0"], num_frames=num_frames,
                            duration=S.SIM_TRAJECTORY_DURATION)
        for spec in S.SIM_TRAJECTORIES])


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic multi-camera dataset.")
    parser.add_argument("-o", "--output", default=S.SIMULATION_DATA_PATH,
                        help="Output directory (default: %(default)s)")
    parser.add_argument("-n", "--frames", type=int, default=S.SIM_NUM_FRAMES,
                        help="Number of frames per camera (default: %(default)s)")
    parser.add_argument("--no-render", action="store_true",
                        help="Only export calibration + trajectory, skip rendering "
                             "(does not require the pyrender extras).")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    rig = build_default_rig()
    positions = build_default_trajectories(args.frames)

    # Always export exact ground truth (pure NumPy, no heavy deps).
    # object_positions.npy is (num_objects, num_frames, 3).
    rig.save(args.output)
    np.save(os.path.join(args.output, "object_positions.npy"), positions)
    print(f"Wrote calibration for {len(rig.cameras)} cameras + ground truth for "
          f"{positions.shape[0]} objects x {positions.shape[1]} frames to {args.output}")

    if args.no_render:
        print("Skipping rendering (--no-render).")
        return

    # Import the renderer lazily so --no-render works without pyrender installed.
    from .renderer import render_sequence
    print("Rendering frames (this needs the [sim] extras)...")
    render_sequence(rig, positions, args.output, object_radius=S.SIM_OBJECT_RADIUS)
    for cam in rig.cameras:
        print(f"  cam{cam.id}: {positions.shape[1]} frames -> "
              f"{os.path.join(args.output, f'cam{cam.id}')}")
    print("Done.")


if __name__ == "__main__":
    main()
