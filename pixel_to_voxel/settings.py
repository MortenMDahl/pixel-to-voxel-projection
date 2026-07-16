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


# Extrinsics Self-Calibration Settings

# Smallest motion-mask blob (in pixels of contour area) accepted as the object
EXTRINSICS_MIN_CONTOUR_AREA = 5

# Max timestamp gap (seconds) when pairing track samples between two cameras
TRACK_PAIR_TOLERANCE = 0.02

# Epipolar inlier threshold for the essential-matrix RANSAC, in pixels
EXTRINSICS_RANSAC_THRESHOLD = 1.5

# Gravity magnitude used for the scale sanity-check, and the relative
# mismatch of the fitted ballistic acceleration that triggers a warning
GRAVITY_MS2 = 9.81
GRAVITY_TOLERANCE = 0.15

# A calibration pass needs at least this many paired samples to join the
# gravity fit
MIN_PASS_SAMPLES = 8


# Object Tracking Settings

# Std-dev (metres) assumed for triangulated centre measurements
TRACKER_MEASUREMENT_NOISE = 0.5

# White-jerk process-noise intensity driving the constant-acceleration model;
# larger tracks maneuvers faster but smooths less
TRACKER_PROCESS_NOISE = 20.0

# Filter updates required before speed/heading are displayed
TRACKER_MIN_UPDATES = 5


# Voxel Grid Settings (pixel-to-voxel projection stage)

# Axis-aligned bounds of the reconstruction volume, in world coordinates
# (Z-up; metres when using the simulator). Sized to cover the simulator's
# default trajectory with margin. Voxels outside any camera's view are
# treated as empty, so the volume should lie inside the shared view frustum.
VOXEL_GRID_MIN = (-30.0, -15.0, 0.0)
VOXEL_GRID_MAX = (30.0, 15.0, 50.0)

# Edge length of a single cubic voxel, in the same units (0.5 -> 120x60x100).
# Keep it below the object's diameter, or the carve can miss voxel centres.
VOXEL_SIZE = 0.5


# Simulator Settings (synthetic multi-camera dataset generation)

# Where generated frames + ground-truth calibration are written
SIMULATION_DATA_PATH = "simulation_output/"

# Default rendered image size and horizontal field of view (degrees)
SIM_IMAGE_WIDTH = 640
SIM_IMAGE_HEIGHT = 480
SIM_FOV_DEG = 60.0

# Default number of frames per camera in a generated sequence (~30 fps)
SIM_NUM_FRAMES = 180

# Radius of the simulated flying object (metres). Must exceed half a voxel,
# or the visual hull can miss every voxel centre at range.
SIM_OBJECT_RADIUS = 0.4

# Default camera positions (world coords, Z-up) and the point they all look at.
# A wide-baseline pair ~50 m south of the scene, watching a high ballistic arc
# (matching the far-range mission the pipeline targets).
SIM_CAMERA_POSITIONS = [
    (0.0, -48.0, 1.5),
    (36.0, -33.0, 1.5),
]
SIM_LOOK_AT = (0.0, 0.0, 20.0)

# Default scripted trajectory: a ballistic parabola (Z-up gravity) rising to
# ~45 m over 6 seconds while crossing ~48 m horizontally.
SIM_TRAJECTORY_P0 = (-24.0, 0.0, 0.5)   # start position
SIM_TRAJECTORY_V0 = (8.0, 0.0, 29.4)    # initial velocity
# Duration is kept short enough that the ballistic arc stays above the ground
# plane (z >= 0) for every frame, so the object is never occluded by the floor.
SIM_TRAJECTORY_DURATION = 6.0           # seconds