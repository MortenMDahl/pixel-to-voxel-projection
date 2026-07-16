import argparse
import glob
import os
import sys
import threading
import time
import webbrowser

import cv2 as cv
import numpy as np

from . import settings
from . import camera_calibration
from . import camera_extrinsics
from .dashboard import Dashboard, serializable_state
from .tracker import KalmanTracker, triangulate_center

class CameraStream:
    def __init__(self, port):
        self.stream = cv.VideoCapture(port)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                break
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# Voxel-centre coordinates are fixed by settings, so they are built only once
_voxel_centers = None

def voxel_grid_centers():
    """World coordinates (N, 3) of every voxel centre in the grid spanned by
    settings.VOXEL_GRID_MIN/MAX with edge length settings.VOXEL_SIZE."""
    global _voxel_centers
    if _voxel_centers is None:
        size = float(settings.VOXEL_SIZE)
        axes = [np.arange(low + size / 2.0, high, size)
                for low, high in zip(settings.VOXEL_GRID_MIN, settings.VOXEL_GRID_MAX)]
        grid = np.meshgrid(*axes, indexing="ij")
        _voxel_centers = np.stack(grid, axis=-1).reshape(-1, 3)
    return _voxel_centers

def pixel_to_voxel(masks, cameras):
    """Carve the voxel grid down to the voxels supported by every camera's
    motion mask (visual-hull space carving).

    A voxel survives only if its centre projects onto a nonzero mask pixel in
    every camera, so the result is the intersection of the back-projected mask
    cones; voxels behind a camera or outside its image are treated as empty.
    Masks come from undistorted frames, so the ideal pinhole model applies.

    masks   -- {port: binary (H, W) mask, nonzero where motion was detected}
    cameras -- {port: {"camera_matrix": 3x3 K, "extrinsic": 4x4 world->camera}}
    Returns the (M, 3) world coordinates of the occupied voxel centres.
    """
    centers = voxel_grid_centers()
    occupied = np.ones(len(centers), dtype=bool)

    for port, mask in masks.items():
        # Only voxels that survived every previous camera need re-testing
        live = np.flatnonzero(occupied)
        if live.size == 0:
            break

        K = np.asarray(cameras[port]["camera_matrix"], dtype=np.float64)
        extrinsic = np.asarray(cameras[port]["extrinsic"], dtype=np.float64)
        cam_points = centers[live] @ extrinsic[:3, :3].T + extrinsic[:3, 3]

        supported = np.zeros(live.size, dtype=bool)
        # Only voxels in front of the camera can project into the image
        front = np.flatnonzero(cam_points[:, 2] > 0)
        if front.size:
            uvw = cam_points[front] @ K.T
            pixels = uvw[:, :2] / uvw[:, 2:3]
            u = np.rint(pixels[:, 0]).astype(np.intp)
            v = np.rint(pixels[:, 1]).astype(np.intp)
            height, width = mask.shape[:2]
            inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            supported[front[inside]] = mask[v[inside], u[inside]] > 0

        occupied[live] = supported

    return centers[occupied]

def load_simulation(directory=None):
    """Ports, calibration (with ground-truth extrinsics), and looping frame
    streams from a simulator dataset directory."""
    from .simulator.stream import SimulatedStream   # keeps the hardware path lean
    if directory is None:
        directory = settings.SIMULATION_DATA_PATH
    cam_dirs = sorted(d for d in glob.glob(os.path.join(directory, "cam*"))
                      if os.path.isdir(d) and os.path.basename(d)[3:].isdigit())
    if not cam_dirs:
        sys.exit(f"No simulator dataset in '{directory}'; "
                 "run python -m pixel_to_voxel.simulator first.")
    ports, calibration_data, streams = [], {}, {}
    for cam_dir in cam_dirs:
        port = int(os.path.basename(cam_dir)[3:])
        ports.append(port)
        calibration_data[port] = {
            "camera_matrix": np.load(os.path.join(directory, f"camera_matrix_{port}.npy")),
            "dist_coeffs": np.load(os.path.join(directory, f"dist_coeffs_{port}.npy")),
            "extrinsic": np.load(os.path.join(directory, f"extrinsics_{port}.npy")),
        }
        streams[port] = SimulatedStream.from_directory(cam_dir)
    return ports, calibration_data, streams

def setup_cameras():
    """Ports, calibration (plus extrinsics when available), and live camera
    streams for a physical rig."""
    ports = camera_calibration.list_camera_ports()
    if ports is None:
        sys.exit("No ports found.")
    if len(ports) < 2:
        sys.exit(f"{len(ports)} ports found, need at least 2.")

    calibration_data = camera_calibration.load_calibration_data()
    if calibration_data is None:
        print("Calibration data not found.")
        do_calibration = input("Do you want to calibrate the cameras? (y/n)")
        if do_calibration == "y":
            camera_calibration.main()
            calibration_data = camera_calibration.load_calibration_data()
            if calibration_data is None:
                sys.exit("Calibration data STILL not found.")
        else:
            sys.exit("Exiting.")

    # Without extrinsics the camera views still run; the dashboard just
    # reports that projection is disabled.
    extrinsics = camera_extrinsics.load_extrinsics(ports)
    if extrinsics is None:
        print("Extrinsics not found, voxel projection disabled. "
              "Run python -m pixel_to_voxel.camera_extrinsics to calibrate.")
    else:
        for port in ports:
            calibration_data[port]["extrinsic"] = extrinsics[port]

    streams = {port: CameraStream(port) for port in ports}
    return ports, calibration_data, streams

def main(sim=False, browser=True):
    if sim:
        ports, calibration_data, streams = load_simulation()
        frame_period = 1.0 / 30.0    # pace the recorded sequence at ~30 fps
    else:
        ports, calibration_data, streams = setup_cameras()
        frame_period = 0.005         # cameras own the rate; just avoid spinning
    for stream in streams.values():
        stream.start()

    have_extrinsics = all("extrinsic" in calibration_data[port] for port in ports)

    dashboard = Dashboard(ports, sim=sim)
    url = f"http://localhost:{dashboard.start()}"
    print(f"Dashboard running at {url}  (Ctrl+C to quit)")
    if browser:
        webbrowser.open(url)
    dashboard.add_event("pipeline started" + (" (simulation)" if sim else ""))
    if not have_extrinsics:
        dashboard.add_event("extrinsics missing - voxel projection disabled")

    grid_info = {"min": list(settings.VOXEL_GRID_MIN),
                 "max": list(settings.VOXEL_GRID_MAX),
                 "size": settings.VOXEL_SIZE}
    total_frames = None
    sim_dt = None
    if sim:
        total_frames = max(len(stream.frame_paths) for stream in streams.values())
        # Sim measurements use the dataset's own timebase, not wall clock
        sim_dt = settings.SIM_TRAJECTORY_DURATION / max(total_frames - 1, 1)
    frames_played = 0
    run_id = 0
    track = KalmanTracker()
    track_announced = False
    one_camera_announced = False
    ended_announced = False
    fps = None

    # Previous grayscale frame per port, for frame differencing
    gray_old = {port: None for port in ports}

    try:
        while True:
            loop_start = time.perf_counter()
            masks = {}
            centroids = {}
            display_frames = {}
            got_frame = False
            for port in ports:
                raw = streams[port].read()
                if raw is None:
                    continue
                got_frame = True

                # Undistort the frame
                frame = cv.undistort(raw, calibration_data[port]["camera_matrix"], calibration_data[port]["dist_coeffs"])

                # Convert to grayscale
                gray_new = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                # Create mask with the difference of the old and new frames
                # (on the first frame there is no previous frame to diff against)
                mask = None
                if gray_old[port] is not None:
                    diff = cv.absdiff(gray_new, gray_old[port])
                    _, mask = cv.threshold(diff, settings.PIXEL_NOISE_THRESHOLD, 255, cv.THRESH_BINARY)
                    masks[port] = mask
                    # Detected object centroid, marked in the camera pane
                    centroid = camera_extrinsics.mask_centroid(mask)
                    if centroid is not None:
                        centroids[port] = centroid
                        cv.drawMarker(frame,
                                      (int(round(centroid[0])), int(round(centroid[1]))),
                                      (0, 255, 255), cv.MARKER_CROSS, 20, 2)

                # Update the old frame
                gray_old[port] = gray_new

                # Camera pane for the dashboard: clean frame | motion mask
                mask_pane = mask if mask is not None else np.zeros_like(gray_new)
                display_frames[port] = np.hstack(
                    [frame, cv.cvtColor(mask_pane, cv.COLOR_GRAY2BGR)])

            # Carve this frame's motion masks into the shared voxel grid
            voxels = np.empty((0, 3))
            if have_extrinsics and len(masks) == len(ports):
                voxels = pixel_to_voxel(masks, calibration_data)

            if sim and got_frame:
                frames_played = min(frames_played + 1, total_frames)

            # Track the object centre from the triangulated mask centroids
            if have_extrinsics and got_frame and len(centroids) >= 2:
                now = (frames_played - 1) * sim_dt if sim else time.perf_counter()
                track.update(triangulate_center(centroids, calibration_data), now)

            # Event-log transitions
            if track.ready and not track_announced:
                dashboard.add_event("track acquired")
                track_announced = True
            if got_frame and len(centroids) == 1 and track_announced:
                if not one_camera_announced:
                    dashboard.add_event("warning: only one camera detecting")
                    one_camera_announced = True
            elif len(centroids) >= 2:
                one_camera_announced = False
            if sim and not got_frame and frames_played and not ended_announced:
                dashboard.add_event("sequence ended")
                ended_announced = True

            sim_info = {"active": False}
            if sim:
                sim_info = {"active": True, "frame": frames_played,
                            "total": total_frames,
                            "ended": bool(frames_played and not got_frame)}
            dashboard.publish(display_frames,
                              serializable_state(sim_info, have_extrinsics, fps,
                                                 voxels, track, grid_info, run_id))

            # A dashboard restart request rewinds the sim sequence
            if sim and dashboard.pop_restart():
                for stream in streams.values():
                    stream.reset()
                gray_old = {port: None for port in ports}
                track = KalmanTracker()
                frames_played = 0
                run_id += 1
                track_announced = False
                one_camera_announced = False
                ended_announced = False
                dashboard.add_event("sequence restarted")

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, frame_period - elapsed))
            loop_time = time.perf_counter() - loop_start
            instant_fps = 1.0 / max(loop_time, 1e-6)
            fps = instant_fps if fps is None else 0.9 * fps + 0.1 * instant_fps
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        for stream in streams.values():
            stream.stop()
        dashboard.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live pixel-to-voxel projection, served as a web dashboard.")
    parser.add_argument("--sim", action="store_true",
                        help="Run on the simulator dataset in simulation_output/ "
                             "instead of physical cameras.")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't open the dashboard in a browser automatically.")
    args = parser.parse_args()
    main(sim=args.sim, browser=not args.no_browser)
