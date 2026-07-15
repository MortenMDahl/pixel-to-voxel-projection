"""Drop-in replacement for ``main.CameraStream`` backed by rendered frames.

``main.py`` only relies on a stream exposing ``read()`` (latest frame) and
``stop()``. ``SimulatedStream`` mirrors that interface but serves pre-rendered
frames from disk in order, so the differencing / projection loop can be driven
by simulated data without any physical cameras.
"""

import glob
import os

import cv2 as cv


class SimulatedStream:
    """Serve a directory of rendered frames one at a time via ``read()``."""

    def __init__(self, frame_paths, loop=False):
        self.frame_paths = list(frame_paths)
        if not self.frame_paths:
            raise ValueError("SimulatedStream got no frames.")
        self.loop = loop
        self.index = 0
        self.stopped = False

    @classmethod
    def from_directory(cls, directory, loop=False):
        """Build a stream from ``directory/frame_*.png`` (sorted by name)."""
        paths = sorted(glob.glob(os.path.join(directory, "frame_*.png")))
        return cls(paths, loop=loop)

    def start(self):
        # Present for parity with CameraStream; nothing to spin up.
        return self

    def read(self):
        """Return the current frame (BGR ndarray), then advance.

        Returns ``None`` once the sequence is exhausted (unless ``loop=True``),
        which callers can use as an end-of-stream signal.
        """
        if self.stopped or self.index >= len(self.frame_paths):
            if self.loop and not self.stopped:
                self.index = 0
            else:
                return None
        frame = cv.imread(self.frame_paths[self.index])
        self.index += 1
        return frame

    def stop(self):
        self.stopped = True
