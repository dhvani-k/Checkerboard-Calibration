# Checkerboard-Calibration

This project provides a set of functions to calibrate a camera using a checkerboard pattern. It includes functions to calculate the image coordinates of the checkerboard corners, find the world coordinates of the corners, estimate the intrinsic parameters (focal length and principal point), and calculate the extrinsic parameters (rotation matrix and translation vector).

## Requirements
The following libraries are required to run the functions in this project:

* numpy
* cv2 (OpenCV)
* os

Please make sure these libraries are installed before running the code.

## Usage
The project consists of the following functions:

- findRot_xyz2XYZ(alpha: float, beta: float, gamma: float) -> np.ndarray: This function calculates the rotation matrix from the XYZ coordinate system to the xyz coordinate system based on the given rotation angles.

- findRot_XYZ2xyz(alpha: float, beta: float, gamma: float) -> np.ndarray: This function calculates the rotation matrix from the xyz coordinate system to the XYZ coordinate system based on the given rotation angles.

- find_corner_img_coord(image: np.ndarray) -> np.ndarray: This function takes an input image and returns the image coordinates of the 32 corners of the checkerboard. The image coordinate is defined such that the top-left corner is (0, 0) and the bottom-right corner of the image is (N, M), where N is the width and M is the height of the image.

- find_corner_world_coord(img_coord: np.ndarray) -> np.ndarray: This function returns the world coordinates of the 32 corners of the checkerboard. The world coordinates are in the form of (x, y, z) and are given in millimeters.

- find_intrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[float, float, float, float]: This function calculates the intrinsic parameters (focal length and principal point) based on the image coordinates and world coordinates of the 32 corners.

- find_extrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: This function calculates the extrinsic parameters (rotation matrix and translation vector) based on the image coordinates, world coordinates, and intrinsic parameters.

To use these functions, import the necessary libraries and call the functions with the required input parameters. 

Please note that you need to provide the input image containing the checkerboard. The image should be in the BGR format.

## Additional Notes
The function findRot_xyz2XYZ and findRot_XYZ2xyz are helper functions to calculate the rotation matrices for the XYZ and xyz coordinate systems. They are internally used by the find_intrinsic and find_extrinsic functions.

The find_intrinsic function uses the perform_calibration function internally to estimate the intrinsic parameters. This function is not included in the provided code, but you can implement it if needed.

The find_extrinsic function also uses the perform_calibration function to estimate the intrinsic parameters. Make sure to provide the correct implementation of this
