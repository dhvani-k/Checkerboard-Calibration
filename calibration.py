import numpy as np
from typing import List, Tuple
import cv2
import os

from cv2 import cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners


# task1

def findRot_xyz2XYZ(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles along x, y and z axis respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from xyz to XYZ.

    '''
    rot_xyz2XYZ = np.eye(3).astype(float)

    # Your implementation
    # add condition to check if angles are between 0 and 90
    if alpha < 0 or alpha > 90:
        print("Angle alpha is out of range (0, 90) degrees")
        exit()

    if beta < 0 or beta > 90:
        print("Angle beta is out of range (0, 90) degrees")
        exit()

    if gamma < 0 or gamma > 90:
        print("Angle gamma is out of range (0, 90) degrees")
        exit()

    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)

    cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
    cos_beta, sin_beta = np.cos(beta), np.sin(beta)
    cos_gamma, sin_gamma = np.cos(gamma), np.sin(gamma)

    R1 = np.array(
        ((cos_alpha, -sin_alpha, 0), (sin_alpha, cos_alpha, 0), (0, 0, 1)))  # rotation around z-axis with alpha
    R2 = np.array(((1, 0, 0), (0, cos_beta, -sin_beta), (0, sin_beta, cos_beta)))  # rotation around x-axis with beta
    R3 = np.array(
        ((cos_gamma, -sin_gamma, 0), (sin_gamma, cos_gamma, 0), (0, 0, 1)))  # rotation around z axis with gamma

    R12 = np.matmul(R1, R2)

    rot_xyz2XYZ = np.matmul(R12, R3)

    return rot_xyz2XYZ


def findRot_XYZ2xyz(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles of the 3 step respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from XYZ to xyz.

    '''
    rot_XYZ2xyz = np.eye(3).astype(float)

    # Your implementation
    if alpha < 0 or alpha > 90:
        print("Angle alpha is out of range (0, 90) degrees")
        exit()

    if beta < 0 or beta > 90:
        print("Angle beta is out of range (0, 90) degrees")
        exit()

    if gamma < 0 or gamma > 90:
        print("Angle gamma is out of range (0, 90) degrees")
        exit()

    alpha = np.deg2rad(-alpha)
    beta = np.deg2rad(-beta)
    gamma = np.deg2rad(-gamma)

    cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
    cos_beta, sin_beta = np.cos(beta), np.sin(beta)
    cos_gamma, sin_gamma = np.cos(gamma), np.sin(gamma)

    R1 = np.array(
        ((cos_alpha, -sin_alpha, 0), (sin_alpha, cos_alpha, 0), (0, 0, 1)))  # rotation around z-axis with alpha
    R2 = np.array(((1, 0, 0), (0, cos_beta, -sin_beta), (0, sin_beta, cos_beta)))  # rotation around x-axis with beta
    R3 = np.array(
        ((cos_gamma, -sin_gamma, 0), (sin_gamma, cos_gamma, 0), (0, 0, 1)))  # rotation around z axis with gamma

    R32 = np.matmul(R3, R2)

    rot_XYZ2xyz = np.matmul(R32, R1)

    return rot_XYZ2xyz


"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above "findRot_xyz2XYZ()" and "findRot_XYZ2xyz()" functions are the only 2 function that will be called in task1.py.
"""


# Your functions for task1


# --------------------------------------------------------------------------------------------------------------
# task2:

def find_corner_img_coord(image: np.ndarray) -> np.ndarray:
    '''
    Args: 
        image: Input image of size MxNx3. M is the height of the image. N is the width of the image. 3 is the channel of the image.
    Return:
        A numpy array of size 32x2 that represents the 32 checkerboard corners' pixel coordinates. 
        The pixel coordinate is defined such that the of top-left corner is (0, 0) and the bottom-right corner of the image is (N, M). 
    '''
    img_coord = np.zeros([32, 2], dtype=float)

    # Your implementation
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (4, 9), None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        corners_left = corners2[0:16]
        corners_right = corners2[20:]

        img_coord = np.concatenate((corners_left, corners_right))

        # Draw and display the corners
        image = cv2.drawChessboardCorners(image, (4, 4), corners_left, ret)
        image = cv2.drawChessboardCorners(image, (4, 4), corners_right, ret)
        # cv2.imshow('image', image)
        # cv2.waitKey(20000)
        cv2.imwrite(os.path.join("test.png"), image)

    return img_coord


def find_corner_world_coord(img_coord: np.ndarray) -> np.ndarray:
    '''
    You can output the world coord manually or through some algorithms you design. Your output should be the same order with img_coord.
    Args: 
        img_coord: The image coordinate of the corners. Note that you do not required to use this as input, 
        as long as your output is in the same order with img_coord.
    Return:
        A numpy array of size 32x3 that represents the 32 checkerboard corners' pixel coordinates. 
        The world coordinate or each point should be in form of (x, y, z). 
        The axis of the world coordinate system are given in the image. The output results should be in milimeters.
    '''
    world_coord = np.zeros([32, 3], dtype=float)

    # Your implementation
    world_coord = np.array(
        [
            [[0, 40, 10]],
            [[0, 40, 20]],
            [[0, 40, 30]],
            [[0, 40, 40]],
            [[0, 30, 10]],
            [[0, 30, 20]],
            [[0, 30, 30]],
            [[0, 30, 40]],
            [[0, 20, 10]],
            [[0, 20, 20]],
            [[0, 20, 30]],
            [[0, 20, 40]],
            [[0, 10, 10]],
            [[0, 10, 20]],
            [[0, 10, 30]],
            [[0, 10, 40]],
            [[10, 0, 10]],
            [[10, 0, 20]],
            [[10, 0, 30]],
            [[10, 0, 40]],
            [[20, 0, 10]],
            [[20, 0, 20]],
            [[20, 0, 30]],
            [[20, 0, 40]],
            [[30, 0, 10]],
            [[30, 0, 20]],
            [[30, 0, 30]],
            [[30, 0, 40]],
            [[40, 0, 10]],
            [[40, 0, 20]],
            [[40, 0, 30]],
            [[40, 0, 40]],

        ])
    #print(world_coord)

    return world_coord


def find_intrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[float, float, float, float]:
    '''
    Use the image coordinates and world coordinates of the 32 point to calculate the intrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        fx, fy: Focal length. 
        (cx, cy): Principal point of the camera (in pixel coordinate).
    '''


    fx: float = 0
    fy: float = 0
    cx: float = 0
    cy: float = 0

    # Your implementation
    P = perform_calibration(img_coord, world_coord)

    M = P[0:3,0:3]

    R, Q = np.linalg.qr(M)

    K = R/float(R[2,2])

    if K[0,0] < 0:
        K[:,0] = -1*K[:,0]

    if K[1,1] < 0:
        K[:,1] = -1*K[:,1]

    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]

    return fx, fy, cx, cy


def find_extrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Use the image coordinates, world coordinates of the 32 point and the intrinsic parameters to calculate the extrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        R: The rotation matrix of the extrinsic parameters. It is a 3x3 numpy array.
        T: The translation matrix of the extrinsic parameters. It is a 1-dimensional numpy array with length of 3.
    '''

    R = np.eye(3).astype(float)
    T = np.zeros(3, dtype=float)

    # Your implementation
    P = perform_calibration(img_coord, world_coord)

    M = P[0:3,0:3]

    R, Q = np.linalg.qr(M)

    K = R/float(R[2,2])

    if K[0,0] < 0:
        K[:,0] = -1*K[:,0]
        Q[0,:] = -1*Q[0,:]

    if K[1,1] < 0:
        K[:,1] = -1*K[:,1]
        Q[1,:] = -1*Q[1,:]

    P_mat = np.dot(K,Q)

    P_scaled = (P_mat[0,0]*P)/float(P[0,0])

    T = np.dot(np.linalg.inv(K), P_scaled[:,3])

    return R, T


"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above 4 functions are the only ones that will be called in task2.py.
"""


# Your functions for task2

def perform_calibration(x, X):

    x = x.reshape([2, 32])
    X = X.reshape([3, 32])

    xt = np.transpose(x)
    Xt = np.transpose(X)

    Xt = np.hstack((Xt, np.ones((32, 1))))
    np_zero_arr = np.array((0, 0, 0, 0))

    M = np.array((64, 12))
    for i in range(0, 32):
        arr1 = np.hstack((np_zero_arr, -Xt[i], xt[i][1] * Xt[i]))
        arr1 = np.reshape(arr1, (1, 12))
        arr2 = np.hstack((Xt[i], np_zero_arr, -xt[i][0] * Xt[i]))
        arr2 = np.reshape(arr2, (1, 12))
        if i == 0:
            M = np.vstack((arr1, arr2))

        else:
            M = np.vstack((M, arr1, arr2))

    u, s, vt = np.linalg.svd(M)
    v = np.transpose(vt)
    p = v[:, 11]

    P = p.reshape((3, 4))
    return P

# ---------------------------------------------------------------------------------------------------------------------


