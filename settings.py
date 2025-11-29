import cv2 as cv

# Camera Calibration Settings

# Calibration chessboard settings
CHESSBOARD_PATTERN_SIZE = (7, 6)
CHESSBOARD_SQUARE_SIZE = 15 #mm

# Path to sample images for calibration
CALIBRATION_IMAGES_PATH = "calibration_images/*.jpg"
CALIBRATION_DATA_PATH = "calibration_data/"

# Calibration criteria
# (type, max_iter, epsilon)
CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Noise threshold for pixel changes
PIXEL_NOISE_THRESHOLD = 25