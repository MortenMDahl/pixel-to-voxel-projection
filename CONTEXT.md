# Pixel-to-Voxel

Reconstructs the 3D position and coarse voxel occupancy of a flying object observed by two or more synchronized cameras pointed at the sky.

## Language

**World frame**:
The single gravity-aligned 3D coordinate frame everything is expressed in: origin at camera 0's optical centre, +Z pointing up (opposing gravity), units in metres.
_Avoid_: global frame, reference frame, board frame

**Extrinsics**:
A camera's 4x4 world→camera rigid transform. Maps world-frame points into that camera's OpenCV-convention axes (+Z forward, +Y down).
_Avoid_: camera pose (that usually means the inverse, camera→world)

**Intrinsics**:
A camera's 3x3 pinhole matrix K plus lens distortion coefficients, calibrated per camera with the chessboard.
_Avoid_: camera matrix (ambiguous with the 3x4 projection matrix)

**Baseline**:
The physically measured straight-line distance between two cameras' optical centres. Fixes the metric scale of the reconstruction.
_Avoid_: camera distance, separation

**Calibration pass**:
One traversal of a flying object through the shared field of view, recorded synchronously by all cameras, used as input for extrinsics self-calibration. Several passes along different paths are needed.
_Avoid_: calibration flight, throw

**Track**:
The time-ordered sequence of one camera's object centroids (pixel coordinates), extracted from its motion masks. Time-paired tracks from two cameras form the correspondences for self-calibration.
_Avoid_: trajectory (reserved for the 3D path)

**Trajectory**:
The object's path through the world frame in 3D. Scripted analytically in the simulator; recovered by triangulation from real tracks.
_Avoid_: track (2D, per camera), arc

**Target**:
The single flying object currently being tracked — what the dashboard reports on. There is at most one; multi-target tracking is out of scope for now.
_Avoid_: object (in UI copy), blob

**State estimate**:
The tracker's filtered description of the object at an instant: world-frame centre position and velocity, from which speed, heading, and climb rate are derived.
_Avoid_: track state, solution

**Heading**:
Compass-style bearing of the object's horizontal travel in the world frame: 0° = world +Y ("grid north", rig-relative — not true north), clockwise, 090° = world +X.
_Avoid_: course, azimuth

**Motion mask**:
The per-frame binary image marking pixels that changed beyond the noise threshold — the object's frame-differenced silhouette.
_Avoid_: foreground mask, diff

**Visual hull**:
The voxel-grid intersection of all cameras' back-projected motion-mask cones; a voxel is occupied only if every camera's mask supports it.
_Avoid_: point cloud, reconstruction (too generic)
