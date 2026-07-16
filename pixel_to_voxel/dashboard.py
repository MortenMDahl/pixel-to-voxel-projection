"""Embedded web dashboard: the system's user interface (see docs/adr/0002).

``main.py`` publishes every pipeline frame into a ``Dashboard``; a background
``ThreadingHTTPServer`` serves the single-page UI (``pixel_to_voxel/web/``)
plus the live data feeds:

    /                       the dashboard page
    /static/...             app JS/CSS and the vendored three.js
    /stream/{port}.mjpg     per-camera MJPEG stream (annotated frames)
    /events                 Server-Sent Events: JSON state per pipeline frame
    POST /api/restart       request a sim replay (the main loop polls a flag)

The pipeline thread only *writes* the latest state under a condition variable;
HTTP handler threads only *read* — a slow or stuck client can never stall
processing, it just misses frames.
"""

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2 as cv
import numpy as np

WEB_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

# Serialization caps: keep a single SSE message comfortably small
MAX_VOXELS_SENT = 2000
MAX_EVENTS_KEPT = 50

_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
}


class _QuietServer(ThreadingHTTPServer):
    """Clients dropping mid-stream (closed tab, aborted MJPEG) are routine —
    don't print a traceback for every one."""

    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


class Dashboard:
    """Latest-state bus between the pipeline loop and the HTTP handlers."""

    def __init__(self, ports, sim=False):
        self.ports = list(ports)
        self.sim = sim
        self._cond = threading.Condition()
        self._frames = {}          # port -> latest JPEG bytes
        self._state = {}           # latest JSON-serializable state
        self._events = []          # [{"id", "t", "text"}], newest last
        self._event_id = 0
        self._seq = 0
        self._restart_requested = False
        self.running = False
        self._server = None
        self._thread = None

    # -- pipeline side --------------------------------------------------

    def publish(self, frames, state):
        """Store the latest camera frames (BGR ndarrays) and state dict."""
        encoded = {}
        for port, frame in frames.items():
            ok, jpeg = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                encoded[port] = jpeg.tobytes()
        with self._cond:
            self._frames.update(encoded)
            state = dict(state)
            state["ports"] = self.ports
            state["events"] = list(self._events)
            self._state = state
            self._seq += 1
            self._cond.notify_all()

    def add_event(self, text):
        with self._cond:
            self._event_id += 1
            self._events.append({"id": self._event_id, "t": time.time(), "text": text})
            self._events = self._events[-MAX_EVENTS_KEPT:]

    def pop_restart(self):
        """True once per POST /api/restart; the caller performs the reset."""
        with self._cond:
            requested = self._restart_requested
            self._restart_requested = False
            return requested

    # -- HTTP side helpers ------------------------------------------------

    def wait_for_update(self, last_seq, timeout=1.0):
        """Block until the state advances past ``last_seq`` (or timeout)."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq or not self.running,
                                timeout=timeout)
            return self._seq

    def snapshot_state(self):
        with self._cond:
            return self._seq, dict(self._state)

    def snapshot_frame(self, port):
        with self._cond:
            return self._seq, self._frames.get(port)

    # -- server lifecycle --------------------------------------------------

    def start(self, host="0.0.0.0", port=8321):
        """Serve in a daemon thread; returns the actual bound port."""
        dashboard = self

        class Handler(_DashboardHandler):
            pass

        Handler.dashboard = dashboard
        self._server = _QuietServer((host, port), Handler)
        self.running = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self._server.server_address[1]

    def stop(self):
        with self._cond:
            self.running = False
            self._cond.notify_all()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


def serializable_state(sim_info, projection_enabled, fps, voxels, targets, grid,
                       run_id):
    """Build the JSON-able state dict published each frame.

    ``targets`` is the already-JSON-able list from MultiTargetTracker.state_list():
    per confirmed target its id, position, velocity, speed, heading, climb, and
    the ``cameras`` currently observing it (fewer than 2 means it is coasting).
    """
    voxel_list = np.asarray(voxels, dtype=np.float64).reshape(-1, 3)
    sent = voxel_list
    if len(sent) > MAX_VOXELS_SENT:
        sent = sent[:: len(sent) // MAX_VOXELS_SENT + 1]
    return {
        "run_id": run_id,
        "sim": sim_info,
        "projection": bool(projection_enabled),
        "fps": None if fps is None else round(float(fps), 1),
        "voxel_count": int(len(voxel_list)),
        "voxels": [[round(float(c), 2) for c in row] for row in sent],
        "grid": grid,
        "targets": list(targets or []),
    }


class _DashboardHandler(BaseHTTPRequestHandler):
    dashboard = None   # bound by Dashboard.start()
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):   # silence per-request console noise
        pass

    # -- routes -------------------------------------------------------------

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_file(os.path.join(WEB_ROOT, "index.html"))
        elif self.path.startswith("/static/"):
            name = os.path.normpath(self.path[len("/static/"):]).replace("\\", "/")
            if name.startswith(".") or "/" in name:
                self.send_error(404)
                return
            self._send_file(os.path.join(WEB_ROOT, "static", name))
        elif self.path == "/events":
            self._serve_events()
        elif self.path.startswith("/stream/") and self.path.endswith(".mjpg"):
            try:
                port = int(self.path[len("/stream/"):-len(".mjpg")])
            except ValueError:
                self.send_error(404)
                return
            self._serve_mjpeg(port)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/restart":
            with self.dashboard._cond:
                self.dashboard._restart_requested = True
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self.send_error(404)

    # -- implementations ------------------------------------------------

    def _send_file(self, path):
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type",
                         _MIME.get(os.path.splitext(path)[1], "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        seq = None
        try:
            while self.dashboard.running:
                if seq is not None:
                    new_seq = self.dashboard.wait_for_update(seq)
                    if new_seq == seq:
                        continue           # timeout heartbeat; nothing new
                seq, state = self.dashboard.snapshot_state()
                if not state:
                    time.sleep(0.05)
                    continue
                payload = json.dumps(state, separators=(",", ":"))
                self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionError, OSError):
            pass

    def _serve_mjpeg(self, port):
        boundary = "pixeltovoxelframe"
        self.send_response(200)
        self.send_header("Content-Type",
                         f"multipart/x-mixed-replace; boundary={boundary}")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        seq = None
        try:
            while self.dashboard.running:
                if seq is not None:
                    new_seq = self.dashboard.wait_for_update(seq)
                    if new_seq == seq:
                        continue
                seq, jpeg = self.dashboard.snapshot_frame(port)
                if jpeg is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(
                    f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionError, OSError):
            pass
