import cv2 as cv
import numpy as np
import threading
import sys

from . import settings
from . import camera_calibration

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

def pixel_to_voxel(frames):

    pass

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

            # Update the old frame
            gray_old[port] = gray_new

            # Apply the mask to the frame
            resulting_frame = cv.bitwise_and(frame, frame, mask=mask)
            cv.imshow(f"Resulting frame {port}", resulting_frame)

        # TODO: Pixel to voxel projection

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()