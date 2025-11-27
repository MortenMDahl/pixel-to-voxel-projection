# Pixel to Voxel Projection

This project aims to create a voxel-based 3D model of real-world flying objects using a pair of synchronized cameras. It involves calibrating the cameras to understand their intrinsic and extrinsic parameters, which are then used to project 2D pixel data into 3D voxel space.

# WORK IN PROGRESS
The code is in no way shape or form in a working state, and is currently under development.

## Prerequisites

- Python 3.12+
- OpenCV (`opencv-python`)
- NumPy (`numpy`)

## Installation

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install opencv-python numpy
   ```

## Configuration

The `settings.py` file contains configuration constants:
- **Camera Settings**: Number of cameras, chessboard pattern size, and square size.
- **Paths**: Directories for storing calibration images and output data.
- **Calibration Criteria**: Parameters for the iterative calibration algorithm.

## Usage

### 1. Camera Calibration

Before running the main projection logic, you must calibrate the cameras to ensure accurate 3D reconstruction.

Run the calibration script:
```bash
python camera_calibration.py
```

The script is interactive and guides you through the process:
1. **Image Management**: It detects existing images in `calibration_images/` and asks if you want to delete them or keep them.
2. **Image Capture**: If you choose to take new photos, it will count down and capture a series of images (default: 20) from all detected cameras. Ensure the chessboard pattern is visible in different angles and positions.
3. **Calibration**: The script processes the images to find chessboard corners.
4. **Output**: 
   - It calculates the Camera Matrix, Distortion Coefficients, Rotation Vectors, and Translation Vectors.
   - These parameters are saved as `.npy` files in the `calibration_data/` directory for each camera (e.g., `camera_matrix_0.npy`).

### 2. Main Application

(Currently under development)
The `main.py` script will serve as the entry point for the real-time projection system, utilizing the calibration data generated in the previous step.

## Project Structure

- `camera_calibration.py`: Handles image capture and camera calibration logic.
- `settings.py`: Central configuration file.
- `main.py`: Main application entry point.
- `calibration_images/`: Directory where captured calibration photos are stored.
- `calibration_data/`: Directory where calculated calibration matrices are saved.