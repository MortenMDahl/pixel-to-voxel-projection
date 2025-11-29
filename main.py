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

    
    ports = camera_calibration.list_camera_ports()

    if len(ports) < 2:
        sys.exit(f"{len(ports)} ports found, need at least 2.")

    streams = []
    for port in ports:
        streams.append(CameraStream(port).start())
    
    

if __name__ == "__main__":
    main()