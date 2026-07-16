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

# Multi-Target Settings

# Most detections kept per camera per frame (largest blobs first)
MAX_DETECTIONS_PER_CAMERA = 15

# Detections closer than this (pixels) are merged into one before
# association: frame differencing splits a fast mover into leading/trailing
# blobs about one body-length apart, which must not become phantom twins
DETECTION_MERGE_PX = 25.0

# Pixel gate when assigning detections to a predicted target in one camera
TARGET_GATE_PX = 40.0

# Symmetric epipolar-distance gate (pixels) for pairing unassigned detections
# between two cameras when spawning a new tentative target
EPIPOLAR_GATE_PX = 8.0

# Lifecycle ("balanced"): a tentative target is confirmed after this many
# triangulated updates, and any target is deleted after this long unseen by
# every camera (coasting on prediction does not count as seen)
TARGET_CONFIRM_UPDATES = 5
TARGET_DELETE_AFTER_S = 1.5

# No new target may spawn within this distance (metres) of an existing one:
# leftover detections near a target are its own debris — e.g. the trailing
# blob when frame differencing splits a fast mover into two blobs — not a
# separate object
SPAWN_SUPPRESSION_RADIUS_M = 5.0


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

# Scripted trajectories: ballistic parabolas (Z-up gravity), one flying object
# each. The default trio crosses mid-air to exercise multi-target association;
# the second dips below the ground shortly before the end, exercising target
# deletion ("lost") while the others fly the full duration.
SIM_TRAJECTORIES = [
    {"p0": (-24.0, 0.0, 0.5), "v0": (8.0, 0.0, 29.4)},
    {"p0": (24.0, 10.0, 1.0), "v0": (-8.0, -2.0, 28.0)},
    {"p0": (0.0, -8.0, 0.3), "v0": (1.0, 3.0, 27.5)},
]
SIM_TRAJECTORY_DURATION = 6.0           # seconds

# First-trajectory aliases, used where a single object is enough (tests,
# extrinsics examples)
SIM_TRAJECTORY_P0 = SIM_TRAJECTORIES[0]["p0"]
SIM_TRAJECTORY_V0 = SIM_TRAJECTORIES[0]["v0"]