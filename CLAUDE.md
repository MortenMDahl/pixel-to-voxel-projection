# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Reconstruct a voxel-based 3D model of a flying object from a pair (or more) of synchronized cameras. The pipeline is: calibrate each camera → undistort live frames → isolate the moving object via frame differencing → project the surviving 2D pixels into a shared 3D voxel space.

**This is an early work-in-progress and is not in a working state.** Key pieces are still stubs — `camera_extrinsics.py` (inter-camera pose) and `pixel_to_voxel()` in `main.py` (the actual projection) are unimplemented, so there is no working end-to-end flow yet.

## Commands

```bash
# Install the package (editable) plus deps (opencv-python, numpy)
pip install -e .

# Run camera calibration (interactive prompts, needs webcams + chessboard)
python -m pixel_to_voxel.camera_calibration

# Run the main projection app (interactive; requires >=2 cameras)
python -m pixel_to_voxel.main

# "Test" suite: currently a single import smoke-check, run as a plain script
python tests/test_imports.py

# Simulator: generate a synthetic multi-camera dataset (no physical cameras)
pip install -e ".[sim]"                          # rendering extras (pyrender, trimesh)
python -m pixel_to_voxel.simulator               # render frames + export ground truth
python -m pixel_to_voxel.simulator --no-render   # export calibration only (NumPy-only, no extras)
python tests/test_simulator_geometry.py          # validates K/extrinsics consistency (no rendering)
```

There is no configured test runner (no pytest config), linter, or build step beyond setuptools. `tests/test_imports.py` only verifies the package imports; it is not a pytest test.

Both entry points are **interactive** (they call `input()` and open camera device windows), so they cannot run unattended in this environment — reason about them by reading the code rather than executing them.

## Architecture

Everything lives in the `pixel_to_voxel/` package. There are two stages that communicate only through `.npy` files on disk:

1. **Calibration stage** (`camera_calibration.py`) — Discovers cameras by probing `cv.VideoCapture(0..9)` (`list_camera_ports`), captures chessboard photos into `calibration_images/`, finds chessboard corners, runs `cv.calibrateCamera` per camera, and writes per-camera intrinsics to `calibration_data/`. Cameras are identified by parsing the filename convention `calibration_{camera_id}_{photo_num}.png`.

2. **Projection stage** (`main.py`) — `CameraStream` wraps each `cv.VideoCapture` in a background thread that continuously overwrites `self.frame`. `main()` loads calibration data (offering to run calibration if missing), then per frame: undistort → grayscale → `absdiff` against the previous frame → threshold (`settings.PIXEL_NOISE_THRESHOLD`) into a motion mask → `bitwise_and`. The masked pixels are meant to feed `pixel_to_voxel(frames)`, which is an empty stub.

`settings.py` is the single source of configuration (chessboard geometry, paths, calibration criteria, noise threshold). All other modules import it as `from . import settings`.

`camera_extrinsics.py` is a stub (`# TODO`). Camera **extrinsics** (relative pose between cameras) are required before any real multi-view voxel projection can work, but are not yet computed — `calibrateCamera`'s per-view rvecs/tvecs are relative to the chessboard, not between cameras.

**Simulator** (`pixel_to_voxel/simulator/`) — a synthetic stand-in for physical cameras, used to develop/validate the projection stage. It renders 2+ time-synchronized virtual pinhole cameras observing a scripted (analytic, not physics-driven) moving object, and exports **exact** ground truth: intrinsics reuse the calibration file names (`camera_matrix_{id}.npy`, `dist_coeffs_{id}.npy` — zeros, ideal pinhole) so they drop into `calibration_data/`; extrinsics are written as 4x4 world→camera matrices (`extrinsics_{id}.npy`), which is the ground truth `camera_extrinsics.py` should eventually recover. World frame is **Z-up**; cameras use **OpenCV axes** (+Z forward, +Y down). Submodules: `rig` (camera geometry + export, NumPy-only), `trajectory` (parabola/line), `renderer` (pyrender offscreen, optional `[sim]` extras, lazily imported), `stream.SimulatedStream` (drop-in for `main.CameraStream`, serves rendered frames from disk via `read()`/`stop()`). The OpenCV↔OpenGL camera-axis flip is centralized in `Camera.pyrender_pose()`.

## Conventions

- Package-relative imports only (`from . import settings`); run modules with `python -m pixel_to_voxel.<module>`, not as loose scripts.
- `calibration_images/` and `calibration_data/` are git-ignored except for `.gitkeep`; generated `.npy` and image files are never committed.
