import cv2 as cv
import numpy as np
import threading
import sys

import settings
import camera_calibration

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
    
    # Open camera stream threads
    streams = []
    for port in ports:
        streams.append(CameraStream(port).start())

    # Store and display frames
    frames = []
    while True:
        for i in range(len(streams)):
            # Overwrite frames every loop
            frames[i] = streams[i].read()
            cv.imshow(f"Frame {i}", frames[i])
        if cv.waitKey(1) & 0xFF == ord("q"):
            # Stop the streams and break the loop
            for stream in streams:
                stream.stop()
            break
        
        #TODO: Add image processing here

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()