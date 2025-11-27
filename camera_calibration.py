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


def main():

    all_objpoints = {}
    all_imgpoints = {}
    
    found_images = glob.glob(settings.SAMPLE_IMAGES_PATH)
    print(f"Found {len(found_images)} images for calibration.")
    reset = input("Do you want to take new calibration photos? (y/n)")
    delete = input("Do you want to delete the current calibration images? (y/n)")
    if delete == "y":
        for image in found_images:
            os.remove(image)
    if reset == "y":
        take_calibration_photos()
        print("Done taking calibration photos. Proceeding to creating calibration data.")
    
    if not found_images:
        print("No images found for calibration.")
        return

    camera_identifiers = []
    for image in found_images:
        # Get camera number from image name
        camera_id = image.split("_")[1]
        camera_identifiers.append(camera_id)
        # If the camera number is not in the dictionary, add it
        if camera_id not in all_objpoints:
            all_objpoints[camera_id] = []
            all_imgpoints[camera_id] = []
    
    for image in found_images:
        print(f"Processing {image}...")
        objpoints, imgpoints = generate_calibration_data_from_image(image)
        # If the calibration data is found, add it to the dictionary
        if objpoints:
            # Get camera number from image name and add the calibration data to the dictionary relating correct camera data
            camera_identifier = image.split("_")[1]
            all_objpoints[camera_identifier].extend(objpoints)
            all_imgpoints[camera_identifier].extend(imgpoints)
    
    if all_objpoints:
        # Calibrate the cameras
        for camera_id in camera_identifiers:
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(all_objpoints[camera], all_imgpoints[camera], settings.CHESSBOARD_PATTERN_SIZE, None, None)
            print("Camera calibrated!")
            print("Camera matrix:", camera_matrix)
            print("Distortion coefficients:", dist_coeffs)
            print("Rotation vectors:", rvecs)
            print("Translation vectors:", tvecs)
        # Save the calibration data
        np.save(f"{settings.CALIBRATION_DATA_PATH}camera_matrix_" + camera_id + ".npy", camera_matrix)
        np.save(f"{settings.CALIBRATION_DATA_PATH}dist_coeffs_" + camera_id + ".npy", dist_coeffs)
        np.save(f"{settings.CALIBRATION_DATA_PATH}rvecs_" + camera_id + ".npy", rvecs)
        np.save(f"{settings.CALIBRATION_DATA_PATH}tvecs_" + camera_id + ".npy", tvecs)
    else:
        print("No chessboard corners found in any images.")

if __name__ == "__main__":
    main()