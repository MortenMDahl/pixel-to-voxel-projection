import cv2 as cv

# Camera Calibration Settings

# Calibration chessboard settings
CHESSBOARD_PATTERN_SIZE = (7, 6)
CHESSBOARD_SQUARE_SIZE = 15 #mm

# Path to sample images for calibration
CALIBRATION_IMAGES_PATH = "calibration_images/*.png"
CALIBRATION_DATA_PATH = "calibration_data/"

# Calibration criteria
# (type, max_iter, epsilon)
CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Noise threshold for pixel changes
PIXEL_NOISE_THRESHOLD = 25


# Simulator Settings (synthetic multi-camera dataset generation)

# Where generated frames + ground-truth calibration are written
SIMULATION_DATA_PATH = "simulation_output/"

# Default rendered image size and horizontal field of view (degrees)
SIM_IMAGE_WIDTH = 640
SIM_IMAGE_HEIGHT = 480
SIM_FOV_DEG = 60.0

# Default number of frames per camera in a generated sequence
SIM_NUM_FRAMES = 60

# Default camera positions (world coords, Z-up) and the point they all look at.
# Two cameras placed to both observe the default trajectory arc.
SIM_CAMERA_POSITIONS = [
    (0.0, -4.0, 1.2),
    (3.5, -2.0, 1.2),
]
SIM_LOOK_AT = (0.0, 0.0, 0.5)

# Default scripted trajectory: a ballistic parabola (Z-up gravity).
SIM_TRAJECTORY_P0 = (-1.5, 0.0, 0.2)   # start position
SIM_TRAJECTORY_V0 = (1.5, 0.0, 3.5)    # initial velocity
# Duration is kept short enough that the ballistic arc stays above the ground
# plane (z >= 0) for every frame, so the object is never occluded by the floor.
SIM_TRAJECTORY_DURATION = 0.7          # seconds