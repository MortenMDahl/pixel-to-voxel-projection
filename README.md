# Pixel to Voxel Projection

Track flying objects in 3D with two or more ordinary cameras pointed at the sky.

Synchronized cameras isolate moving objects by frame differencing, carve the
surviving pixels into a shared voxel grid (a visual hull), and triangulate each
object's centre into a live **multi-target track** — position, speed, heading,
and climb rate per target — presented on a built-in web dashboard with a
rotatable 3D view.

The whole system runs end-to-end against a bundled synthetic simulator (no
hardware needed). With real cameras, the one extra step is an extrinsics
calibration session — no calibration board required: the system self-calibrates
from a thrown object (see [ADR 0001](docs/adr/0001-self-calibrate-extrinsics-from-tracked-object.md)).

## Quick start (no cameras needed)

```bash
pip install -e ".[sim]"                # core + rendering extras (pyrender, trimesh)
python -m pixel_to_voxel.simulator    # render a synthetic 3-object dataset
python -m pixel_to_voxel.main --sim   # open the dashboard on the dataset
```

The browser opens `http://localhost:8321`: camera panes with motion masks,
a three.js voxel scene (drag to rotate), and a target table tracking three
simulated objects crossing a ~45 m ballistic arc seen from ~50 m.

## How it works

1. **Intrinsics** (`camera_calibration.py`) — classic per-camera chessboard
   calibration (`cv.calibrateCamera`), saved to `calibration_data/`.
2. **Extrinsics self-calibration** (`camera_extrinsics.py`) — at 50 m+ range
   the camera baseline is metres wide, so a hand-held board cannot calibrate
   it. Instead, fly/throw an object through the shared view a few times: the
   time-paired motion-mask centroids feed an essential-matrix solve refined by
   a ballistic bundle adjustment (rotation, translation direction, shared
   acceleration, inter-camera latency). A tape-measured camera distance fixes
   metric scale; the fitted acceleration recovers gravity, aligning the world
   frame (+Z up) and sanity-checking scale (|a| ≈ 9.81 m/s²).
3. **Detection** — undistort, frame-difference, threshold into motion masks;
   every plausible blob centroid becomes a detection.
4. **Voxel carving** (`pixel_to_voxel()` in `main.py`) — a voxel is occupied
   when every camera that can see it has mask support and at least two can see
   it, so the hull works even when cameras point in different directions to
   expand coverage ([ADR 0003](docs/adr/0003-multi-target-association-and-visibility-aware-carving.md)).
5. **Multi-target tracking** (`tracker.py`) — detections are associated to
   predicted targets per camera (Hungarian assignment + epipolar spawning with
   ghost suppression), each target filtered by a constant-acceleration Kalman
   filter. Targets seen by fewer than two cameras coast ballistically; each
   target reports which cameras currently observe it.
6. **Dashboard** (`dashboard.py` + `web/`, [ADR 0002](docs/adr/0002-web-dashboard-as-interface.md)) —
   an embedded stdlib HTTP server streams camera frames (MJPEG) and state
   (Server-Sent Events) to a single-page UI: target table with per-target
   colours and camera-count accuracy chips, detail card, speed/altitude
   charts, event log, and sim playback controls. Works from any device on the
   LAN.

## Using real cameras (2+)

```bash
pip install -e .
python -m pixel_to_voxel.camera_calibration   # chessboard intrinsics, per camera
python -m pixel_to_voxel.camera_extrinsics    # calibration session: fly a few passes
python -m pixel_to_voxel.main                 # dashboard on live cameras
```

The extrinsics session shows live detections, records time-stamped tracks
(`p` starts/stops a pass, `q` ends), asks for the measured camera-to-camera
distance, and writes `extrinsics_{port}.npy`. Tracks are persisted, so
`--solve-only` re-runs the solve without re-flying. Keep the cameras rigidly
mounted afterwards — moving a camera invalidates its extrinsics.

## Tests

Script-style, all headless:

```bash
python tests/test_imports.py                # package smoke test
python tests/test_simulator_geometry.py     # rig K/extrinsics consistency
python tests/test_pixel_to_voxel.py         # carving vs exact synthetic masks (+ divergent-rig rules)
python tests/test_camera_extrinsics.py      # self-calibration vs ground truth at 60-100 m
python tests/test_tracker.py                # Kalman state estimate vs the analytic arc
python tests/test_multi_target.py           # association, identity, handover, ghosts (3-camera rig)
python tests/test_dashboard.py              # HTTP endpoints + full --sim pipeline vs ground truth
```

## Layout

- `pixel_to_voxel/` — the package: `settings.py` (all configuration),
  `camera_calibration.py`, `camera_extrinsics.py`, `main.py` (pipeline +
  carving), `tracker.py`, `dashboard.py`, `web/` (UI + vendored three.js),
  `simulator/` (synthetic cameras, trajectories, renderer, streams)
- `CONTEXT.md` — domain glossary; `docs/adr/` — architectural decisions
- `calibration_data/`, `calibration_images/`, `simulation_output/` —
  generated artifacts (git-ignored)

## Requirements

Python 3.9+, `opencv-python`, `numpy` (installed by `pip install -e .`).
The simulator's renderer needs the `[sim]` extras; everything else — including
all tests except dataset generation — runs without them.
