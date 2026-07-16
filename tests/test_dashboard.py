"""Headless checks for the web dashboard and the full --sim pipeline behind it.

Starts the embedded server on an ephemeral port and asserts over real HTTP —
the page and static assets load, the MJPEG stream delivers a JPEG, the SSE
feed carries the pipeline state, and the restart endpoint round-trips. The
end-to-end part drives 20 frames of the actual sim pipeline (load dataset ->
undistort -> mask -> carve -> track -> publish) and validates the tracked
state as served to the browser against the dataset's exported ground truth.

Run as a plain script (matching the existing test convention):

    python tests/test_dashboard.py
"""

import glob
import json
import os
import sys
from http.client import HTTPConnection

import cv2 as cv
import numpy as np

sys.path.append(os.getcwd())

from pixel_to_voxel import settings
from pixel_to_voxel.dashboard import Dashboard, serializable_state
from pixel_to_voxel.camera_extrinsics import mask_centroids
from pixel_to_voxel.main import load_simulation, pixel_to_voxel
from pixel_to_voxel.tracker import MultiTargetTracker

GRID = {"min": list(settings.VOXEL_GRID_MIN),
        "max": list(settings.VOXEL_GRID_MAX),
        "size": settings.VOXEL_SIZE}


def get(port, path, timeout=5):
    conn = HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp.status, body


def read_sse_state(port):
    """Connect to /events and return the first pushed state as a dict."""
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", "/events")
    resp = conn.getresponse()
    assert resp.status == 200
    try:
        while True:
            line = resp.fp.readline()
            assert line, "SSE stream closed without data"
            if line.startswith(b"data: "):
                return json.loads(line[len(b"data: "):])
    finally:
        conn.close()


def read_mjpeg_frame(port, stream_port):
    """Read the first JPEG out of an MJPEG stream."""
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", f"/stream/{stream_port}.mjpg")
    resp = conn.getresponse()
    assert resp.status == 200
    assert "multipart/x-mixed-replace" in resp.getheader("Content-Type", "")
    try:
        # Part headers, blank line, then JPEG bytes
        length = None
        while True:
            line = resp.fp.readline()
            assert line, "MJPEG stream closed without a frame"
            if line.lower().startswith(b"content-length:"):
                length = int(line.split(b":")[1])
            if line == b"\r\n" and length is not None:
                return resp.fp.read(length)
    finally:
        conn.close()


def test_server_endpoints():
    dashboard = Dashboard(ports=[0, 1])
    port = dashboard.start(host="127.0.0.1", port=0)
    try:
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        cv.rectangle(frame, (20, 15), (60, 45), (0, 200, 255), -1)
        dashboard.add_event("unit test started")
        dashboard.publish(
            {0: frame, 1: frame},
            serializable_state({"active": False}, True, 30.0,
                               np.array([[1.0, 2.0, 3.0]]), [], GRID, 0))

        status, body = get(port, "/")
        assert status == 200 and b"Pixel-to-Voxel" in body, "index page broken"
        status, body = get(port, "/static/app.js")
        assert status == 200 and b"EventSource" in body, "app.js broken"
        status, body = get(port, "/static/three.module.min.js")
        assert status == 200 and len(body) > 100_000, "three.js missing"
        status, _ = get(port, "/static/nope.js")
        assert status == 404
        status, _ = get(port, "/static/../dashboard.py")
        assert status == 404, "path traversal not rejected"

        state = read_sse_state(port)
        assert state["voxel_count"] == 1 and state["ports"] == [0, 1]
        assert state["events"][-1]["text"] == "unit test started"

        jpeg = read_mjpeg_frame(port, 0)
        assert jpeg[:2] == b"\xff\xd8", "not a JPEG frame"
        assert cv.imdecode(np.frombuffer(jpeg, np.uint8), cv.IMREAD_COLOR).shape == (60, 80, 3)

        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", "/api/restart")
        assert conn.getresponse().status == 204
        conn.close()
        assert dashboard.pop_restart() is True
        assert dashboard.pop_restart() is False
        print("PASS: dashboard endpoints (page, static, SSE, MJPEG, restart).")
    finally:
        dashboard.stop()


def test_sim_pipeline_through_dashboard():
    frame_files = glob.glob(os.path.join(settings.SIMULATION_DATA_PATH, "cam*", "frame_*.png"))
    if not frame_files:
        print("SKIP: no rendered simulator dataset "
              "(run python -m pixel_to_voxel.simulator to create one).")
        return

    ports, calibration_data, streams = load_simulation()
    truth = np.load(os.path.join(settings.SIMULATION_DATA_PATH, "object_positions.npy"))
    total = max(len(stream.frame_paths) for stream in streams.values())
    sim_dt = settings.SIM_TRAJECTORY_DURATION / max(total - 1, 1)

    dashboard = Dashboard(ports, sim=True)
    http_port = dashboard.start(host="127.0.0.1", port=0)
    tracker = MultiTargetTracker(calibration_data)
    gray_old = {port: None for port in ports}
    frames_read = 20
    carved_any = False
    try:
        for i in range(frames_read):
            masks, detections, display = {}, {}, {}
            for port in ports:
                frame = streams[port].read()
                frame = cv.undistort(frame, calibration_data[port]["camera_matrix"],
                                     calibration_data[port]["dist_coeffs"])
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                if gray_old[port] is not None:
                    diff = cv.absdiff(gray, gray_old[port])
                    _, mask = cv.threshold(diff, settings.PIXEL_NOISE_THRESHOLD, 255,
                                           cv.THRESH_BINARY)
                    masks[port] = mask
                    detections[port] = mask_centroids(mask)
                gray_old[port] = gray
                display[port] = frame
            voxels = np.empty((0, 3))
            if len(masks) == len(ports):
                voxels = pixel_to_voxel(masks, calibration_data)
                carved_any = carved_any or len(voxels) > 0
            if detections:
                tracker.step(detections, i * sim_dt)
            sim_info = {"active": True, "frame": i + 1, "total": total, "ended": False}
            dashboard.publish(display,
                              serializable_state(sim_info, True, 30.0, voxels,
                                                 tracker.state_list(), GRID, 0))

        assert carved_any, "No frame produced occupied voxels from the sim dataset"

        # Validate what the browser would receive
        state = read_sse_state(http_port)
        assert state["sim"]["frame"] == frames_read
        assert state["grid"]["size"] == settings.VOXEL_SIZE
        targets = state["targets"]
        assert len(targets) == truth.shape[0], \
            f"served {len(targets)} targets, dataset has {truth.shape[0]} objects"

        # Each served target must sit on its own true object with the right
        # velocity direction (this drives the 3D arrows), seen by both cameras.
        last = frames_read - 1
        claimed = set()
        for target in targets:
            errors = np.linalg.norm(truth[:, last] - np.array(target["position"]), axis=1)
            obj = int(np.argmin(errors))
            assert obj not in claimed, "two targets bound to the same object"
            claimed.add(obj)
            v_true = (np.asarray(settings.SIM_TRAJECTORIES[obj]["v0"])
                      + np.array([0.0, 0.0, -9.81]) * last * sim_dt)
            v_served = np.array(target["velocity"], dtype=np.float64)
            cos_dir = (v_served @ v_true) / (np.linalg.norm(v_served) * np.linalg.norm(v_true))
            print(f"target {target['id']} -> object {obj}: position error "
                  f"{errors[obj]:.2f} m, speed {target['speed']:.1f}, "
                  f"cameras {target['cameras']}, cos(dir) {cos_dir:.3f}")
            assert errors[obj] < 2.5, f"target {target['id']} off by {errors[obj]} m"
            assert cos_dir > 0.98, f"velocity direction off: cos={cos_dir:.3f}"
            assert target["cameras"] == ports, \
                f"expected both cameras observing, got {target['cameras']}"

        jpeg = read_mjpeg_frame(http_port, ports[0])
        assert jpeg[:2] == b"\xff\xd8"
        print("PASS: --sim pipeline serves multi-target state and frames over HTTP.")
    finally:
        for stream in streams.values():
            stream.stop()
        dashboard.stop()


def main():
    test_server_endpoints()
    test_sim_pipeline_through_dashboard()


if __name__ == "__main__":
    main()
