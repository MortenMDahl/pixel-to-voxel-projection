# Web dashboard as the system's interface

The interface is a browser dashboard served by an HTTP server embedded in the
pipeline process (`dashboard.py`, Python stdlib only): camera panes stream as
MJPEG, pipeline state (target track, voxels, events, sim progress) streams as
Server-Sent Events, and the 3D voxel scene renders client-side with a vendored
three.js (no CDN, no build step). The pipeline thread publishes latest-state
under a lock; HTTP threads only read, so a slow client can never stall
processing. The previous OpenCV-window display (`voxel_viewer.py`) was removed
rather than kept as a fallback — one display path, no drift.

Chosen over a native Qt app (heavyweight dependency, slower polish) and a Dear
PyGui dashboard (limited theming) because it adds zero Python dependencies,
gives GPU 3D interaction decoupled from pipeline rate, and works from any
device on the LAN — which suits field operation of a sky-pointing rig.

## Consequences

- The UI needs a browser; there is no windowed fallback.
- Anyone on the local network can view the dashboard and trigger sim restarts
  (fine for a hobby field tool; revisit if that changes).
- three.js is pinned as a vendored file (`web/static/three.module.min.js`,
  r165) and upgraded manually.
