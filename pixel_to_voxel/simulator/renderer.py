"""pyrender-based offscreen renderer for the synthetic multi-camera scene.

This is the only module that needs the heavy optional dependencies
(``pyrender`` + ``trimesh``). They are imported lazily inside the functions so
that ``rig.py`` / ``trajectory.py`` — and the geometry tests — keep working with
just NumPy installed.

Install the rendering extras with::

    pip install -e ".[sim]"
"""

import os

import numpy as np
import cv2 as cv


def _require_pyrender():
    """Import pyrender/trimesh lazily with a helpful error if they're missing."""
    try:
        import trimesh
        import pyrender
    except ImportError as e:  # pragma: no cover - depends on optional install
        raise ImportError(
            "The simulator renderer needs 'pyrender' and 'trimesh'. "
            "Install them with:  pip install -e \".[sim]\""
        ) from e
    return trimesh, pyrender


def _build_scene(object_radius, object_color):
    """Create the static scene (ground + scenery + lights) and the moving object.

    Returns ``(scene, object_node)``. The static content gives frame differencing
    a stable background, and the moving object is the only thing that changes
    between frames — exactly what ``main.py``'s ``absdiff`` step isolates.
    """
    trimesh, pyrender = _require_pyrender()

    scene = pyrender.Scene(
        ambient_light=np.array([0.3, 0.3, 0.3]),
        bg_color=np.array([0.05, 0.05, 0.08]),
    )

    # Ground plane (a thin, wide box centred at the world origin, top at z=0),
    # sized so the far-range cameras never see past its edge.
    ground = trimesh.creation.box(extents=[300.0, 300.0, 0.5])
    ground.apply_translation([0.0, 0.0, -0.25])
    ground.visual.vertex_colors = np.array([90, 90, 100, 255], dtype=np.uint8)
    scene.add(pyrender.Mesh.from_trimesh(ground, smooth=False))

    # A few static coloured boxes so the background has texture/features,
    # building-sized to stay visible at ~50 m.
    scenery = [
        ([25.0, 12.0, 0.0], [8.0, 8.0, 6.0], [180, 120, 60]),
        ([-30.0, 18.0, 0.0], [10.0, 10.0, 8.0], [60, 140, 170]),
        ([10.0, 32.0, 0.0], [6.0, 6.0, 10.0], [150, 160, 70]),
    ]
    for pos, extents, color in scenery:
        box = trimesh.creation.box(extents=extents)
        box.apply_translation([pos[0], pos[1], extents[2] / 2.0])
        box.visual.vertex_colors = np.array(color + [255], dtype=np.uint8)
        scene.add(pyrender.Mesh.from_trimesh(box, smooth=False))

    # Directional light from above so the moving object is well lit.
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    light_pose = np.eye(4)
    light_pose[:3, 3] = [0.0, 0.0, 5.0]
    scene.add(light, pose=light_pose)

    # The moving object: a coloured sphere added at the origin; its pose is
    # updated per frame by the caller.
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=object_radius)
    sphere.visual.vertex_colors = np.array(list(object_color) + [255], dtype=np.uint8)
    object_node = scene.add(pyrender.Mesh.from_trimesh(sphere, smooth=True))

    return scene, object_node


def render_sequence(rig, positions, output_dir, object_radius=0.15,
                    object_color=(220, 40, 40)):
    """Render every camera in ``rig`` for every frame in ``positions``.

    Writes ``output_dir/cam{id}/frame_{i:04d}.png`` (BGR PNGs, matching what the
    OpenCV pipeline expects from a real camera). All cameras observe the same
    object position per frame, so the streams are perfectly time-synchronised.
    """
    _, pyrender = _require_pyrender()

    scene, object_node = _build_scene(object_radius, object_color)

    # Register one pyrender camera node per rig camera; we switch the scene's
    # active camera before each render.
    cam_nodes = []
    for cam in rig.cameras:
        pyr_cam = pyrender.IntrinsicsCamera(
            fx=cam.K[0, 0], fy=cam.K[1, 1], cx=cam.K[0, 2], cy=cam.K[1, 2],
            zfar=1000.0,   # default 100 would clip the far-range scene
        )
        node = scene.add(pyr_cam, pose=cam.pyrender_pose())
        cam_nodes.append(node)
        os.makedirs(os.path.join(output_dir, f"cam{cam.id}"), exist_ok=True)

    # A single renderer sized to the (shared) image resolution.
    w, h = rig.cameras[0].width, rig.cameras[0].height
    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)

    try:
        for frame_idx, pos in enumerate(positions):
            # Move the object to this frame's ground-truth position.
            obj_pose = np.eye(4)
            obj_pose[:3, 3] = pos
            scene.set_pose(object_node, obj_pose)

            for cam, node in zip(rig.cameras, cam_nodes):
                scene.main_camera_node = node
                color, _ = renderer.render(scene)       # color is RGB uint8
                bgr = cv.cvtColor(color, cv.COLOR_RGB2BGR)
                path = os.path.join(
                    output_dir, f"cam{cam.id}", f"frame_{frame_idx:04d}.png")
                cv.imwrite(path, bgr)
    finally:
        renderer.delete()
