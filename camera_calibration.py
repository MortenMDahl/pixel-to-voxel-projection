import cv2 as cv
import numpy as np
import glob
import settings

def calibrate_cameras():
    #TODO: Implement camera calibration using two cameras for real
    # Take a photo using each of the two cameras and find the chessboard corners
    for i in range(settings.NUMBER_OF_CAMERAS):
        img = cv.VideoCapture(i)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, settings.CHESSBOARD_PATTERN_SIZE, None)

        if ret:
            subcorners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1))
            cv.drawChessboardCorners(img, settings.CHESSBOARD_PATTERN_SIZE, subcorners, ret)
            cv.imshow(f"img_{i}", img)
            cv.waitKey(0)
        else:
            print(f"Camera {i} not found")

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
    objpoints = []
    imgpoints = []
    for image in glob.glob(settings.SAMPLE_IMAGES_PATH):
        objpoints, imgpoints = calibrate_camera_from_image(image)
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, settings.CHESSBOARD_PATTERN_SIZE, None, None)
        print("Camera calibrated!")
        print("Camera matrix:", camera_matrix)
        print("Distortion coefficients:", dist_coeffs)
        print("Rotation vectors:", rvecs)
        print("Translation vectors:", tvecs)

if __name__ == "__main__":
    main()