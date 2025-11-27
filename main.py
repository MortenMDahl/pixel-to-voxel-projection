import cv2 as cv
import numpy as np
import settings
import camera_calibration

# TODO: Load calibration data
def load_calibration_data():
    ports = camera_calibration.list_camera_ports()
    calibration_data = {}
    for i in range(0, len(ports)):
        calibration_data[i] = {
            "camera_matrix": np.load(f"calibration_data/camera_matrix_{i}.npy"),
            "dist_coeffs": np.load(f"calibration_data/dist_coeffs_{i}.npy"),
            "rotation_vectors": np.load(f"calibration_data/rotation_vectors_{i}.npy"),
            "translation_vectors": np.load(f"calibration_data/translation_vectors_{i}.npy")
        }
    return calibration_data


def main():
    pass

if __name__ == "__main__":
    main()