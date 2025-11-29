import cv2 as cv
import numpy as np
import glob
import os
import settings
import time

# List all available camera ports
def list_camera_ports():
    ports = []
    # Run through all ports and check if they are open
    for i in range(10):
        cap = cv.VideoCapture(i)
        if cap.isOpened():
            ports.append(i)
            cap.release()
    return ports

# Take calibration photos using each of the cameras
def take_calibration_photos():
    ports = list_camera_ports()
    if len(ports) < 2:
        print(f"{len(ports)} ports found, need at least 2.")
        return
    
    print("Taking calibration photos. Keep the chessboard in view of all cameras, and move it around to cover different angles. ")
    print("Starting in... ")
    for k in range(5):
        print(5-k)
        time.sleep(1)
    
    for i in ports:
        print(f"Taking photos for camera {i}")
        cap = cv.VideoCapture(i)
        for j in range(20):
            print(f"Photo {j+1}/20")
            ret, frame = cap.read()
            if ret:
                cv.imwrite(f"calibration_images/calibration_{i}_{j}.png", frame)
            else:
                print(f"Camera {i} not found")
        cap.release()

# Calibrate each camera based on the calibration photos
def generate_calibration_data_from_image(image):

    objpoints = []
    imgpoints = []

    # Create a matrix of object points, later to be related to the image points in order to calibrate the cameras
    objp = np.zeros((settings.CHESSBOARD_PATTERN_SIZE[0]*settings.CHESSBOARD_PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:settings.CHESSBOARD_PATTERN_SIZE[0], 0:settings.CHESSBOARD_PATTERN_SIZE[1]].T.reshape(-1, 2)

    # Find the chessboard corners in the image
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, settings.CHESSBOARD_PATTERN_SIZE, None)
    # If the corners are found, further improve the accuracy of the corners and add them to the list
    if ret:
        subcorners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        objpoints.append(objp)
        imgpoints.append(subcorners)
        print(f"Chessboard corners found in image {image}")
    else:
        print(f"Chessboard corners not found in image {image}")
    return objpoints, imgpoints

def load_calibration_data():
    ports = list_camera_ports()
    calibration_data = {}
    try:
        for i in range(0, len(ports)):
            calibration_data[i] = {
                "camera_matrix": np.load(f"calibration_data/camera_matrix_{i}.npy"),
                "dist_coeffs": np.load(f"calibration_data/dist_coeffs_{i}.npy"),
                "rotation_vectors": np.load(f"calibration_data/rotation_vectors_{i}.npy"),
                "translation_vectors": np.load(f"calibration_data/translation_vectors_{i}.npy")
            }
        return calibration_data
    except FileNotFoundError:
        print("Calibration data not found.")
        return
    except Exception as e:
        print(e)
        return

def main():
    
    # Find existing images
    found_images = glob.glob(settings.CALIBRATION_IMAGES_PATH)

    print(f"Found {len(found_images)} images for calibration.")
    
    reset = input("Do you want to take new calibration photos? (y/n) ")
    delete_images = input("Do you want to delete the current calibration images? (y/n) ")
    delete_data = input("Do you want to delete the current calibration data? (y/n) ")

    # Delete data if requested
    if delete_data == "y":
        data_files = glob.glob(f"{settings.CALIBRATION_DATA_PATH}/*.npy")
        for file in data_files:
            os.remove(file)
        print("Calibration data deleted.")

    # Delete images if requested
    if delete_images == "y":
        for image in found_images:
            os.remove(image)
        found_images = [] # Clear the list after deletion
        print("Calibration images deleted.")

    # Take new photos if requested
    if reset == "y":
        take_calibration_photos()
        print("Done taking calibration photos.")
        # Refresh the list of images
        found_images = glob.glob(settings.CALIBRATION_IMAGES_PATH)

    if not found_images:
        print("No images found for calibration.")
        return

    # Identify unique cameras
    # Assumes filename format: "calibration_{camera_id}_{photo_num}.png" (or .jpg)
    camera_ids = set()
    for image_path in found_images:
        filename = os.path.basename(image_path)
        parts = filename.split("_")
        if len(parts) >= 2:
            camera_ids.add(parts[1])
        else:
            print(f"Invalid filename format: {filename}")
    
    print(f"Identified cameras: {camera_ids}")

    # Prepare data structures
    all_objpoints = {cam_id: [] for cam_id in camera_ids}
    all_imgpoints = {cam_id: [] for cam_id in camera_ids}

    # Process images
    for image in found_images:
        print(f"Processing {image}...")
        objpoints, imgpoints = generate_calibration_data_from_image(image)
        
        if objpoints:
            filename = os.path.basename(image)
            parts = filename.split("_")
            if len(parts) >= 2:
                cam_id = parts[1]
                if cam_id in all_objpoints:
                    all_objpoints[cam_id].extend(objpoints)
                    all_imgpoints[cam_id].extend(imgpoints)

    # Calibrate and save the data
    for cam_id in camera_ids:
        if all_objpoints[cam_id] and all_imgpoints[cam_id]:
            print(f"Calibrating camera {cam_id}...")
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
                all_objpoints[cam_id], 
                all_imgpoints[cam_id], 
                settings.CHESSBOARD_PATTERN_SIZE, 
                None, 
                None
            )
            
            if ret:
                print(f"Camera {cam_id} calibrated successfully!")
                print("Camera matrix:", camera_matrix)
                print("Distortion coefficients:", dist_coeffs)
                
                # Save the calibration data
                np.save(f"{settings.CALIBRATION_DATA_PATH}camera_matrix_{cam_id}.npy", camera_matrix)
                np.save(f"{settings.CALIBRATION_DATA_PATH}dist_coeffs_{cam_id}.npy", dist_coeffs)
                np.save(f"{settings.CALIBRATION_DATA_PATH}rvecs_{cam_id}.npy", rvecs)
                np.save(f"{settings.CALIBRATION_DATA_PATH}tvecs_{cam_id}.npy", tvecs)
            else:
                print(f"Calibration failed for camera {cam_id}.")
        else:
            print(f"Not enough data to calibrate camera {cam_id}.")

if __name__ == "__main__":
    main()