import cv2 as cv
import numpy as np
import threading
import sys

from . import settings
from . import camera_calibration
from . import camera_extrinsics

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

def main():
    # Attempt to open ports
    ports = camera_calibration.list_camera_ports()
    if ports is None:
        sys.exit("No ports found.")
    if len(ports) < 2:
        sys.exit(f"{len(ports)} ports found, need at least 2.")

    # Attempt to load calibration data
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

    # Load extrinsics (world->camera poses); without them the motion masks are
    # still shown, but projection into the voxel grid is skipped.
    extrinsics = camera_extrinsics.load_extrinsics(ports)
    if extrinsics is None:
        print("Extrinsics not found, voxel projection disabled.")
    else:
        for port in ports:
            calibration_data[port]["extrinsic"] = extrinsics[port]

    # Open camera stream threads, keyed by port
    streams = {port: CameraStream(port).start() for port in ports}

    # Previous grayscale frame per port, for frame differencing
    gray_old = {port: None for port in ports}

    # Main loop
    while True:
        raw_frames = {}
        for port in ports:
            # Overwrite frames every loop
            raw_frames[port] = streams[port].read()
            cv.imshow(f"Frame {port}", raw_frames[port])
        if cv.waitKey(1) & 0xFF == ord("q"):
            # Stop the streams and break the loop
            for stream in streams.values():
                stream.stop()
            break

        masks = {}
        for port in ports:
            # Undistort the frame
            frame = cv.undistort(raw_frames[port], calibration_data[port]["camera_matrix"], calibration_data[port]["dist_coeffs"])

            # Convert to grayscale
            gray_new = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # On the first frame there is no previous frame to diff against
            if gray_old[port] is None:
                gray_old[port] = gray_new
                continue

            # Create mask with the difference of the old and new frames
            diff = cv.absdiff(gray_new, gray_old[port])
            _, mask = cv.threshold(diff, settings.PIXEL_NOISE_THRESHOLD, 255, cv.THRESH_BINARY)
            masks[port] = mask

            # Update the old frame
            gray_old[port] = gray_new

            # Apply the mask to the frame
            resulting_frame = cv.bitwise_and(frame, frame, mask=mask)
            cv.imshow(f"Resulting frame {port}", resulting_frame)

        # Carve this frame's motion masks into the shared voxel grid
        if extrinsics is not None and len(masks) == len(ports):
            voxels = pixel_to_voxel(masks, calibration_data)
            status = f"Occupied voxels: {len(voxels)}"
            if len(voxels):
                status += "   centroid: ({:+.2f}, {:+.2f}, {:+.2f})".format(*voxels.mean(axis=0))
            # \r keeps the per-frame report on one updating console line
            print("\r" + status.ljust(70), end="", flush=True)

    print()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()