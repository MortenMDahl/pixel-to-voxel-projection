import cv2 as cv
import numpy as np
import glob
import os
import settings

def list_camera_ports():
    ports = []
    # Run through all ports and check if they are open
    for i in range(10):
        cap = cv.VideoCapture(i)
        if cap.isOpened():
            ports.append(i)
            cap.release()
    return ports

def calibrate_cameras():
    # Take 20 photos using each of the two cameras and find the chessboard corners
    ports = list_camera_ports()
    if len(ports) < 2:
        print(f"{len(ports)} ports found, need at least 2.")
        return
    
    print("Taking calibration photos. Keep the chessboard in view of both cameras, and move it around to cover different angles. ")
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

def calibrate_camera_from_image(image):
    # Do a camera calibration based on a set of sample images

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
    else:
        print("Chessboard not found")
    
    return objpoints, imgpoints


def main():

    if not settings.SAMPLE_MODE:
        print("Camera calibration not implemented for real-time cameras just yet.")
        return
    else:
        all_objpoints = []
        all_imgpoints = []
        
        found_images = glob.glob(settings.SAMPLE_IMAGES_PATH)
        
        if not found_images:
            print("No images found for calibration.")
            return

        for image in found_images:
            print(f"Processing {image}...")
            objpoints, imgpoints = calibrate_camera_from_image(image)
            # calibrate_camera_from_image returns lists, so we extend
            if objpoints:
                all_objpoints.extend(objpoints)
                all_imgpoints.extend(imgpoints)
        
        if all_objpoints:
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(all_objpoints, all_imgpoints, settings.CHESSBOARD_PATTERN_SIZE, None, None)
            print("Camera calibrated!")
            print("Camera matrix:", camera_matrix)
            print("Distortion coefficients:", dist_coeffs)
            print("Rotation vectors:", rvecs)
            print("Translation vectors:", tvecs)
        else:
            print("No chessboard corners found in any images.")

if __name__ == "__main__":
    main()