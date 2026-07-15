# Self-calibrate extrinsics from the tracked object

The system targets flying objects at 50 m+ range, which forces a camera baseline of
metres — a hand-held calibration board cannot be seen well by all cameras at once at
that separation, and single-shot board pose accuracy is far below what triangulation
at range demands. Instead, the flying object itself is the calibration target:
time-paired motion-mask centroids from synchronized cameras (several "calibration
passes" along different paths) feed an essential-matrix solve, which initializes a
ballistic bundle adjustment — quadratic motion with a shared acceleration, refined
jointly with rotation, translation direction, and the inter-camera latency offset.
A tape/GPS-measured baseline fixes metric scale; the fitted acceleration recovers
gravity, aligning the world frame (origin at the reference camera, +Z up) and
cross-checking the scale (|a| should be ~9.81 m/s²).

## Considered Options

- **Hand-held chessboard + solvePnP** — physically infeasible at wide baseline; close-range only.
- **cv.stereoCalibrate** — same board limitation, and the world frame would not be gravity-aligned.
- **Astrometric (star) calibration + GPS** — most accurate at extreme range; deferred, the
  extrinsics file format would not change.

## Consequences

- Extrinsics quality depends on flying good calibration passes (a single pass is planar
  and degenerate; passes should cross the shared view along different paths).
- Gravity alignment needs at least one unpowered/ballistic pass; otherwise the world
  frame silently falls back to the reference camera's frame.
- The ballistic refinement is what delivers usable accuracy in the wide-baseline
  regime; the essential matrix alone leaves a rotation/depth ambiguity worth ~1° or
  metres of triangulation error at 70 m.
