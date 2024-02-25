# Copyright (c) 2024 Yonwoo Choi
# All rights reserved.
# This code is part of my research at Seoul National University.
# Redistribution and use in source and binary forms, with or without
# modification, are not permitted without direct permission from the author.

import numpy as np 
from scipy.optimize import least_squares
from utils.triangulation import Rotation2Quaternion
from utils.triangulation import Quaternion2Rotation

def construct_camera_matrix(R, C):
    """
    Construct camera projection matrix from rotation matrix and camera center.
    s
    Parameters:
    - R: Rotation matrix (3x3)
    - C: Camera center (3,)
    
    Returns:
    - Projection matrix (3x4)
    """
    return np.hstack([R, -R @ C[:, np.newaxis]])

def SetupBundleAdjustment(P, X, track):
    """
    Prepares variables for bundle adjustment optimization.
    
    Args:
    - P (ndarray): Camera poses of shape (K, 3, 4).
    - X (ndarray): 3D points of shape (J, 3).
    - track (ndarray): Track matrix of shape (K, J, 2) containing 2D point observations.
    
    Returns:
    - tuple: Contains initial guess for optimization (z), 2D points (b), 
             sparse indicator matrix (S), camera and point indices for each measurement.
    """
    n_cameras, n_points = P.shape[0], X.shape[0]
    n_projs = np.sum(track[:, :, 0] != -1)
    
    # Initialize variables
    b = np.zeros((2 * n_projs,))
    S = np.zeros((2 * n_projs, 7 * n_cameras + 3 * n_points), dtype=bool)
    camera_index, point_index = [], []
    
    # Populate b and S, and collect indices
    k = 0
    for i in range(n_cameras):
        for j in range(n_points):
            if track[i, j, 0] != -1:
                S[2*k : 2*(k+1), 7*i : 7*(i+1)] = True
                S[2*k : 2*(k+1), 7*n_cameras+3*j : 7*n_cameras+3*(j+1)] = True
                b[2*k : 2*(k+1)] = track[i, j]
                camera_index.append(i)
                point_index.append(j)
                k += 1
    
    # Convert lists to numpy arrays for efficiency
    camera_index, point_index = np.array(camera_index), np.array(point_index)
    
    # Prepare optimization variable z (camera poses + 3D points)
    z = np.zeros((7 * n_cameras + 3 * n_points,))
    for i, P_i in enumerate(P):
        R, C = P_i[:, :3], -P_i[:, :3].T @ P_i[:, 3]
        q = Rotation2Quaternion(R)
        z[7*i : 7*(i+1)] = np.concatenate([C, q])
    z[7*n_cameras:] = X.ravel()

    return z, b, S, camera_index, point_index
    

def MeasureReprojection(z, b, n_cameras, n_points, camera_index, point_index):
    """
    Calculates the reprojection error for each observation.
    
    Args:
    - z (ndarray): Optimization variables (camera poses + 3D points).
    - b (ndarray): 2D measured points.
    - camera_index (ndarray): Indices of cameras for each measurement.
    - point_index (ndarray): Indices of 3D points for each measurement.
    
    Returns:
    - ndarray: Reprojection errors for all measurements.
    """
    # Initialize the array f to have the same shape as b but filled with zeros. 
    # b contains the observed image coordinates of the 3D points.
    f = np.zeros_like(b)
    # Iterate over each observation. `k` is the observation index,
    # `i` is the index of the camera, and `j` is the index of the 3D point in this observation.
    for k, (i, j) in enumerate(zip(camera_index, point_index)):
        # Extract the parameters for the i-th camera and the j-th 3D point from the optimization variable z.
        # For the camera, this includes its position and quaternion-based orientation.
        # For the 3D point, this simply includes its coordinates.
        p = z[7*i : 7*(i+1)]
        X = z[7*len(camera_index)+3*j : 7*len(camera_index)+3*(j+1)]
        # Normalize the quaternion to ensure it represents a valid rotation,
        # then convert it to a rotation matrix.
        R = Quaternion2Rotation(p[3:] / np.linalg.norm(p[3:]))
        # Compute the predicted image point by first transforming the 3D point into the camera's coordinate frame,
        # then performing perspective division to get the normalized image coordinates.
        proj = R @ (X - p[:3])
        proj = proj / proj[2]  # Perspective division by the z-coordinate to get normalized image coordinates
        # Assign the computed x and y coordinates to the corresponding positions in the array f.
        # These are the predicted image points for the current observation.
        f[2*k : 2*(k+1)] = proj[:2]
    # Return the difference between the observed image points (b) and the predicted image points (f).
    # This difference is essentially the reprojection error for each observation.
    return b - f

def UpdatePosePoint(z, n_cameras, n_points):
    """
    Update camera poses and 3D points from optimization variables.
    
    Parameters:
    - z: Optimization variable
    - n_cameras: Number of cameras
    - n_points: Number of 3D points
    
    Returns:
    - P_new: Updated camera poses
    - X_new: Updated 3D points
    """
    # Initialize an array to hold the new camera poses
    P_new = np.empty((n_cameras, 3, 4))
    # Iterate through each camera to update its pose
    for i in range(n_cameras):
        # Extract the optimization parameters for the current camera
        # This includes its position (C) and orientation (quaternion q)
        p = z[7 * i:7 * (i + 1)]
        # The last four elements of p are the quaternion representing rotation
        q = p[3:]
        # Convert the quaternion to a rotation matrix
        R = Quaternion2Rotation(q)
        # The first three elements of p represent the camera position
        C = p[:3]
        # Construct the camera matrix for the current camera using the rotation matrix and position
        # The function `construct_camera_matrix` is assumed to create a 3x4 camera matrix from R and C
        P_new[i, :, :] = construct_camera_matrix(R, C)

    # The remaining elements in z after extracting camera parameters are the 3D points
    # Reshape these elements to form the updated set of 3D points
    X_new = z[7 * n_cameras:].reshape((-1, 3))

    # Return the updated camera poses and 3D points
    return P_new, X_new

def RunBundleAdjustment(P, X, track):
    """
    Perform bundle adjustment to refine camera poses and 3D points.

    Parameters:
    ----------
    P : ndarray, shape (K, 3, 4)
        Initial camera poses.
    X : ndarray, shape (J, 3)
        Initial 3D points.
    track : ndarray, shape (K, J, 2)
        Tracks of 2D points across images.

    Returns:
    -------
    P_new : ndarray, shape (K, 3, 4)
        Refined camera poses.
    X_new : ndarray, shape (J, 3)
        Refined 3D points.
    """
    n_cameras, n_points = P.shape[0], X.shape[0]

    # Prepare initial parameters for optimization
    z0, b, S, camera_index, point_index = SetupBundleAdjustment(P, X, track)

    # Perform least squares optimization
    res = least_squares(
        lambda x: MeasureReprojection(x, b, n_cameras, n_points, camera_index, point_index),
        z0,
        jac_sparsity=S,
        verbose=2
    )

    # Extract optimized parameters
    z_opt = res.x
    # Calculate initial and final reprojection errors for comparison
    initial_error = MeasureReprojection(z0, b, n_cameras, n_points, camera_index, point_index)
    final_error = MeasureReprojection(z_opt, b, n_cameras, n_points, camera_index, point_index)
    print(f'Reprojection error improved from {np.linalg.norm(initial_error)} to {np.linalg.norm(final_error)}.')
    # Update poses and points based on optimized parameters
    P_new, X_new = UpdatePosePoint(z_opt, n_cameras, n_points)

    return P_new, X_new