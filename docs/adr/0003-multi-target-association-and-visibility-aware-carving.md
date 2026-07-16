# Multi-target association and visibility-aware carving on a divergent rig

The rig will grow to 3+ cameras aimed in *different* directions with pairwise
overlap, expanding the total field of view — so no camera sees everything, and
up to ~10 targets must be tracked simultaneously. Two coupled decisions follow:

**Association** is track-oriented global-nearest-neighbour: each target's
prediction is projected into every camera that can see it and matched to
detections by pixel distance (hand-rolled Hungarian, no scipy); only leftover
detections may spawn new tentative targets, via pairwise epipolar gating.
Ghost pairings are suppressed by three mechanisms: per-camera detections are
merged within ~25 px first (frame differencing splits a fast mover into
leading/trailing blobs that would otherwise spawn phantom twins and cross-pair
into ghosts), spawns are rejected near existing or same-step targets, and a
third camera that should see the candidate but detects nothing vetoes it.
Targets confirm after several triangulated updates and die after ~1.5 s unseen;
a target seen by fewer than two cameras coasts ballistically (single-view
corridors between overlap zones), exposed to the UI via its ``cameras``
property. Chosen over JPDA/MHT-class probabilistic association, which two 640px
cameras' detection quality does not justify at this target count.

**Carving** became visibility-aware: a voxel is occupied when every camera that
can *see* it supports it and at least two can see it. The previous
all-cameras rule would shrink the voxel volume to the intersection of all
frustums — nearly empty on a divergent rig. Per-voxel camera visibility is
static, so it is precomputed once (carving got faster, not slower).

## Consequences

- With exactly two cameras, crossing targets can still swap identities or
  briefly ghost — the geometric veto needs a third view. Accepted at 2-camera
  scale; adding cameras improves it with zero redesign.
- Two real objects within ~25 px in one camera merge into one detection for
  those frames (they were unresolvable debris otherwise).
- Coasting targets drift until re-acquired (metres over seconds); the
  dashboard's per-target camera count is the honesty signal.
- Bearing-only single-camera filter updates are a known future upgrade that
  would shrink corridor drift.
