import sys
import os

# Add current directory to sys.path to ensure we can import the package
sys.path.append(os.getcwd())

try:
    import pixel_to_voxel.main
    import pixel_to_voxel.camera_calibration
    import pixel_to_voxel.settings
    print("Imports successful")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)
