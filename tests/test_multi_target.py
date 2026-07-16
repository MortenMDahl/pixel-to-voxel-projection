"""Ground-truth checks for multi-target tracking (no rendering, no cameras).

A divergent three-camera rig (overlapping but differently-aimed views, the
planned expanded-FOV configuration) watches three crossing ballistic objects.
Each camera only reports detections for objects inside its own view, so the
scenario exercises association, identity persistence through crossings, ghost
suppression, camera handover across the expanded field, per-target camera
visibility, and deletion.

Run as a plain script (matching the existing test convention):

    python tests/test_multi_target.py
"""

import itertools
import os
import sys

import numpy as np

sys.path.append(os.getcwd())

from pixel_to_voxel import settings
from pixel_to_voxel.tracker import MultiTargetTracker, hungarian
from pixel_to_voxel.simulator.rig import CameraRig, look_at_extrinsic, intrinsic_matrix
from pixel_to_voxel.simulator.rig import Camera
from pixel_to_voxel.simulator import trajectory

FPS = 30.0
DURATION = 4.0
FRAMES = int(DURATION * FPS) + 1

# Divergent rig: three cameras along a line, aimed at spread-out points so the
# combined field is wide and only neighbours overlap strongly.
CAMERA_POSITIONS = [(-10.0, -48.0, 1.5), (0.0, -48.0, 1.5), (10.0, -48.0, 1.5)]
CAMERA_TARGETS = [(-20.0, 0.0, 20.0), (0.0, 0.0, 20.0), (20.0, 0.0, 20.0)]

# Crossing arcs, kept low enough (apex ~26 m) that they never rise above the
# cameras' vertical field of view; the first sweeps the whole expanded field
# left-to-right (handover).
OBJECTS = [
    dict(p0=(-35.0, 0.0, 10.0), v0=(16.5, 0.0, 18.0)),
    dict(p0=(30.0, 10.0, 8.0), v0=(-15.0, -1.5, 18.5)),
    dict(p0=(-5.0, -5.0, 5.0), v0=(3.0, 3.0, 19.0)),
]


def build_rig():
    cameras = []
    for i, (pos, target) in enumerate(zip(CAMERA_POSITIONS, CAMERA_TARGETS)):
        R, t = look_at_extrinsic(pos, target)
        K = intrinsic_matrix(settings.SIM_IMAGE_WIDTH, settings.SIM_IMAGE_HEIGHT,
                             settings.SIM_FOV_DEG)
        cameras.append(Camera(id=i, width=settings.SIM_IMAGE_WIDTH,
                              height=settings.SIM_IMAGE_HEIGHT, K=K, R=R, t=t,
                              eye=np.asarray(pos)))
    return CameraRig(cameras)


def visible_detections(rig, points, rng, noise_px=0.3):
    """Per-camera noisy detections for the objects each camera can see."""
    detections = {}
    truth_visibility = {}
    for cam in rig.cameras:
        dets = []
        seen = []
        for obj_index, point in enumerate(points):
            pix, valid = cam.project(point)
            u, v = pix[0]
            if valid[0] and 0 <= u < cam.width and 0 <= v < cam.height:
                dets.append((u + rng.normal(0, noise_px), v + rng.normal(0, noise_px)))
                seen.append(obj_index)
        order = rng.permutation(len(dets))          # association must not rely on order
        detections[cam.id] = [dets[k] for k in order]
        truth_visibility[cam.id] = {seen[k] for k in order}
    return detections, truth_visibility


def test_hungarian():
    rng = np.random.default_rng(0)
    for _ in range(60):
        n, m = (int(x) for x in rng.integers(1, 6, size=2))
        cost = rng.random((n, m)) * 10
        pairs = hungarian(cost)
        got = sum(cost[i, j] for i, j in pairs)
        k = min(n, m)
        best = min(
            sum(cost[r, c] for r, c in zip(rows, cols))
            for rows in itertools.permutations(range(n), k)
            for cols in itertools.permutations(range(m), k))
        assert abs(got - best) < 1e-9, f"hungarian {got} vs brute force {best}"
    print("PASS: hungarian assignment matches brute force.")


def main():
    test_hungarian()

    rig = build_rig()
    cameras = {cam.id: {"camera_matrix": cam.K, "extrinsic": cam.extrinsic_4x4()}
               for cam in rig.cameras}
    tracker = MultiTargetTracker(cameras)
    rng = np.random.default_rng(1)

    tracks = np.stack([trajectory.parabola(o["p0"], o["v0"], FRAMES, DURATION)
                       for o in OBJECTS])
    times = np.linspace(0.0, DURATION, FRAMES)

    confirmed_ids = []
    id_to_object = {}
    cameras_over_life = {}
    triangulated_errors = []
    coasting_errors = []
    visibility_checks = 0
    for k in range(FRAMES):
        detections, truth_vis = visible_detections(rig, tracks[:, k], rng)
        events = tracker.step(detections, times[k])
        confirmed_ids.extend(events["confirmed"])

        for target in tracker.targets:
            if not target.confirmed:
                continue
            position = target.filter.position
            if target.id not in id_to_object:
                # Bind each new confirmed target to its nearest true object
                distances = np.linalg.norm(tracks[:, k] - position, axis=1)
                id_to_object[target.id] = int(np.argmin(distances))
            obj = id_to_object[target.id]
            error = np.linalg.norm(position - tracks[obj, k])
            # While >=2 cameras triangulate, the estimate must be tight; in
            # single-view coasting (the chosen edge-of-field behaviour, which
            # the per-target `cameras` property exposes) it may drift until
            # re-acquisition, but identity must hold.
            if len(target.cameras) >= 2:
                triangulated_errors.append(error)
            else:
                coasting_errors.append(error)
            cameras_over_life.setdefault(target.id, set()).update(target.cameras)
            # The reported cameras must be a subset of the cameras that truly
            # see the object (association can miss, never invent)
            if target.cameras:
                assert set(target.cameras) <= {p for p, seen in truth_vis.items()
                                               if obj in seen}, \
                    f"target {target.id} claims cameras it cannot have"
                visibility_checks += 1

    # Exactly the three real objects were confirmed — no ghosts, none missed
    assert len(confirmed_ids) == 3, f"confirmed {confirmed_ids}, expected 3 targets"
    assert sorted(id_to_object.values()) == [0, 1, 2], \
        f"targets bound to objects {id_to_object} — identity mixup"

    worst_identity = max(triangulated_errors)
    worst_coasting = max(coasting_errors) if coasting_errors else 0.0
    assert worst_identity < 3.0, f"triangulated identity drift {worst_identity:.2f} m (swap?)"
    assert worst_coasting < 10.0, f"coasting drift {worst_coasting:.2f} m before re-acquisition"

    # Handover: the sweeping object was observed by the left camera early and
    # the right camera late, under one persistent id
    sweeper = next(tid for tid, obj in id_to_object.items() if obj == 0)
    assert {0, 2} <= cameras_over_life[sweeper], \
        f"no handover: target {sweeper} only saw cameras {cameras_over_life[sweeper]}"

    # Deletion: with no more detections, every target is dropped
    deleted = []
    for i in range(1, 12):
        events = tracker.step({port: [] for port in cameras},
                              times[-1] + 0.2 * i)
        deleted.extend(events["deleted"])
    assert sorted(deleted) == sorted(id_to_object.keys()), \
        f"deleted {deleted}, expected all of {list(id_to_object)}"
    assert not tracker.targets

    print(f"Confirmed targets: {len(id_to_object)} (objects {sorted(id_to_object.values())})")
    print(f"Worst identity error: {worst_identity:.2f} m over "
          f"{len(triangulated_errors)} triangulated target-frames "
          f"(+{worst_coasting:.2f} m across {len(coasting_errors)} coasting frames); "
          f"camera-visibility checks: {visibility_checks}")
    print(f"Handover cameras for sweeping target: {sorted(cameras_over_life[sweeper])}")
    print("PASS: multi-target association, identity, handover, visibility, deletion.")


if __name__ == "__main__":
    main()
