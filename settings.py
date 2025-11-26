import cv2 as cv

# Camera Calibration Settings

# Current operating mode
SAMPLE_MODE = True

# Number of cameras
NUMBER_OF_CAMERAS = 2

# Calibration chessboard settings
CHESSBOARD_PATTERN_SIZE = (7, 6)
CHESSBOARD_SQUARE_SIZE = 15 #mm

# Path to sample images for calibration
SAMPLE_IMAGES_PATH = "sample_calibration_images/*.png"

# Calibration criteria
# (type, max_iter, epsilon)
CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
