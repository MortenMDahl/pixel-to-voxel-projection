# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Reconstruct a voxel-based 3D model of a flying object from a pair (or more) of synchronized cameras. The pipeline is: calibrate each camera → undistort live frames → isolate the moving object via frame differencing → project the surviving 2D pixels into a shared 3D voxel space.

**This is an early work-in-progress.** `pixel_to_voxel()` (visual-hull space carving) and extrinsics self-calibration (`camera_extrinsics.py`) are implemented; the remaining gap to a real end-to-end run is operational (an actual calibration session with physical cameras). Without `extrinsics_{port}.npy` in `calibration_data/` (from a session or the simulator's ground-truth export), `main.py` runs but skips voxel projection.

Domain vocabulary lives in `CONTEXT.md`; architectural decisions in `docs/adr/`.

## Commands

```bash
# Install the package (editable) plus deps (opencv-python, numpy)
pip install -e .

# Run camera calibration (interactive prompts, needs webcams + chessboard)
python -m pixel_to_voxel.camera_calibration

# Extrinsics self-calibration session (interactive; needs >=2 cameras + intrinsics)
python -m pixel_to_voxel.camera_extrinsics
python -m pixel_to_voxel.camera_extrinsics --solve-only   # re-solve saved tracks

# Run the main app: serves the web dashboard at http://localhost:8321 (requires >=2 cameras)
python -m pixel_to_voxel.main
python -m pixel_to_voxel.main --sim            # same dashboard, driven by simulation_output/ (no cameras)
python -m pixel_to_voxel.main --sim --no-browser   # don't auto-open the browser

# "Test" suite: currently a single import smoke-check, run as a plain script
python tests/test_imports.py

# Simulator: generate a synthetic multi-camera dataset (no physical cameras)
pip install -e ".[sim]"                          # rendering extras (pyrender, trimesh)
python -m pixel_to_voxel.simulator               # render frames + export ground truth
python -m pixel_to_voxel.simulator --no-render   # export calibration only (NumPy-only, no extras)
python tests/test_simulator_geometry.py          # validates K/extrinsics consistency (no rendering)
python tests/test_pixel_to_voxel.py              # validates voxel carving against exact synthetic masks (no rendering)
python tests/test_camera_extrinsics.py           # validates extrinsics self-calibration on synthetic far-range tracks (no rendering)
python tests/test_dashboard.py                   # dashboard endpoints + full --sim pipeline served over HTTP vs ground truth
python tests/test_tracker.py                     # validates triangulation + Kalman state estimate against the analytic arc (no rendering)
python tests/test_multi_target.py                # association/identity/handover/ghosts on a divergent 3-camera rig (no rendering)
```

There is no configured test runner (no pytest config), linter, or build step beyond setuptools. `tests/test_imports.py` only verifies the package imports; it is not a pytest test.

The entry points cannot run unattended in this environment: calibration/extrinsics call `input()` and need physical cameras, and `main.py` serves the dashboard until Ctrl+C (with `--sim` it needs no cameras, but still runs forever). Reason about them by reading the code; the dashboard's data path is covered headlessly by `tests/test_dashboard.py`.

## Architecture

Everything lives in the `pixel_to_voxel/` package. There are two stages that communicate only through `.npy` files on disk:

1. **Calibration stage** (`camera_calibration.py`) — Discovers cameras by probing `cv.VideoCapture(0..9)` (`list_camera_ports`), captures chessboard photos into `calibration_images/`, finds chessboard corners, runs `cv.calibrateCamera` per camera, and writes per-camera intrinsics to `calibration_data/`. Cameras are identified by parsing the filename convention `calibration_{camera_id}_{photo_num}.png`.

2. **Projection stage** (`main.py`) — `CameraStream` wraps each `cv.VideoCapture` in a background thread that continuously overwrites `self.frame`. `main()` loads calibration data (offering to run calibration if missing), then per frame: undistort → grayscale → `absdiff` against the previous frame → threshold (`settings.PIXEL_NOISE_THRESHOLD`) into a motion mask. The per-camera masks feed `pixel_to_voxel(masks, cameras)`: visual-hull space carving over a world-space voxel grid (bounds/voxel size in `settings.py`, centres cached by `voxel_grid_centers()`). The rule is **visibility-aware** (the rig's cameras point in different directions with pairwise overlap): a voxel is occupied when every camera *that can see it* has mask support at its projection and at least two cameras can see it — a seeing camera showing background carves it, which also erases two-camera ghosts wherever a third view has coverage. Per-voxel camera visibility is static and cached (`_carve_tables`); the function returns the (M, 3) world coordinates of occupied voxel centres. Projection runs only when every port has an extrinsic (from `camera_extrinsics.load_extrinsics()` or the sim dataset); otherwise masks are still shown and projection is skipped. `--sim` drives the identical loop from `simulation_output/` (`load_simulation()`: `SimulatedStream`s + ground-truth calibration, paced at ~30 fps). The sim sequence plays **once**; afterwards the voxel scene and trail stay inspectable, and SPACE replays from the start (rewinds streams, clears frame-diff state and trail).

3. **Tracking** (`tracker.py`, see `docs/adr/0003`) — multi-target (up to ~10), N-camera-native, independent of the voxel grid. Per-camera motion-mask centroids (`camera_extrinsics.mask_centroids`, all blobs) are merged within `DETECTION_MERGE_PX` (frame differencing splits fast movers into leading/trailing blobs), then `MultiTargetTracker.step()` associates them to predicted targets camera-by-camera (pixel gating + hand-rolled `hungarian()`), triangulates targets matched in ≥2 cameras (`triangulate_center`), and updates each target's `KalmanTracker` (constant-acceleration, 9-state). Leftover detections spawn tentative targets via epipolar gating (`fundamental_matrix`/`epipolar_distance_px`), suppressed near existing/same-step targets and vetoed by a contradicting third camera. Lifecycle: confirm after `TARGET_CONFIRM_UPDATES` triangulated updates, delete after `TARGET_DELETE_AFTER_S` unseen; a target seen by one camera coasts ballistically (its per-frame `cameras` property tells the UI whether it is triangulating or coasting). `step()` returns confirmed/deleted ids for the event log; `state_list()` feeds the dashboard. Sim mode timestamps measurements on the dataset's own timebase, real mode on `time.perf_counter()`.

4. **Interface** (`dashboard.py` + `web/`, see `docs/adr/0002`) — the UI is a web dashboard served by an embedded stdlib `ThreadingHTTPServer` (default port 8321, browser auto-opens; LAN-accessible). The pipeline `publish()`es latest state under a condition variable; HTTP threads only read. Feeds: `/stream/{port}.mjpg` (camera panes: undistorted frame | motion mask, detections marked), `/events` (SSE JSON: `targets` list — id, position, velocity, speed, heading, climb, and the `cameras` currently observing each — plus voxel centres capped at 2000, grid, fps, events, sim progress, `run_id` for client resets), `POST /api/restart` (sim replay). The page (`web/index.html` + `static/app.js`) renders a target table (colour swatch, id, cams, speed, heading, altitude, climb; click to select, first confirmed auto-selected) with a detail card for the selection (speed m/s + km/h, heading + cardinal, climb, altitude, position, and a cams chip: green ≥2 triangulating / amber 1 / grey coasting), a three.js orbit 3D scene (vendored r165, Z-up, instanced height-coloured voxel cubes, per-target trails and velocity arrows in per-id palette colours, selected highlighted), speed/altitude sparkline charts following the selection, an event log (`target N confirmed/lost`), and sim controls. Ctrl+C quits the process; there are no OpenCV windows anymore. `render()` is pure (image out, no window) so it's testable headlessly; only `attach()`/`update()` touch windows.

`settings.py` is the single source of configuration (chessboard geometry, paths, calibration criteria, noise threshold). All other modules import it as `from . import settings`.

`camera_extrinsics.py` implements **extrinsics self-calibration from the tracked object** (see `docs/adr/0001`): an interactive session records time-stamped motion-mask centroid tracks of a flying object from all cameras (several passes), then solves — essential matrix (init) → ballistic bundle adjustment refining rotation, translation direction, shared acceleration, and inter-camera latency. A measured camera-to-camera baseline (prompted, persisted to `baselines.npy`) fixes metric scale; the fitted acceleration gravity-aligns the world frame (origin at the reference camera, +Z up) and sanity-checks scale (|a| ≈ 9.81). Targets 50 m+ operation with wide (metres) baselines, where a calibration board is physically inadequate. Raw tracks are persisted (`tracks_{port}.npy`) so `--solve-only` can re-run the solve without re-flying. The solve functions are pure math, testable without cameras.

**Simulator** (`pixel_to_voxel/simulator/`) — a synthetic stand-in for physical cameras, used to develop/validate the projection stage. It renders 2+ time-synchronized virtual pinhole cameras observing scripted (analytic, not physics-driven) moving objects — by default a far-range scene matching the mission profile: wide ~48 m baseline viewed from ~50 m, three 0.8 m objects on crossing 6 s ballistic arcs (`SIM_TRAJECTORIES`; one dips below ground early, exercising target loss). `object_positions.npy` is (num_objects, num_frames, 3). It exports **exact** ground truth: intrinsics reuse the calibration file names (`camera_matrix_{id}.npy`, `dist_coeffs_{id}.npy` — zeros, ideal pinhole) so they drop into `calibration_data/`; extrinsics are written as 4x4 world→camera matrices (`extrinsics_{id}.npy`), which is the ground truth `camera_extrinsics.py` should eventually recover. World frame is **Z-up**; cameras use **OpenCV axes** (+Z forward, +Y down). Submodules: `rig` (camera geometry + export, NumPy-only), `trajectory` (parabola/line), `renderer` (pyrender offscreen, optional `[sim]` extras, lazily imported), `stream.SimulatedStream` (drop-in for `main.CameraStream`, serves rendered frames from disk via `read()`/`stop()`). The OpenCV↔OpenGL camera-axis flip is centralized in `Camera.pyrender_pose()`.

## Conventions

- Package-relative imports only (`from . import settings`); run modules with `python -m pixel_to_voxel.<module>`, not as loose scripts.
- `calibration_images/` and `calibration_data/` are git-ignored except for `.gitkeep`; generated `.npy` and image files are never committed.
