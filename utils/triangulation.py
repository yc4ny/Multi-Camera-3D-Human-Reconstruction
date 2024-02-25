# Copyright (c) 2024 Yonwoo Choi
# All rights reserved.
# This code is part of my research at Seoul National University.
# Redistribution and use in source and binary forms, with or without
# modification, are not permitted without direct permission from the author.

import numpy as np

def LinearTriangulation(K1, K2, C1, R1, C2, R2, x1, x2):
    """
    Perform linear triangulation of corresponding points in two images.
    
    Args:
    - K1, K2: The camera intrinsic matrices for the first and second camera, respectively.
    - C1, C2: The camera centers for the first and second camera, respectively.
    - R1, R2: The rotation matrices for the first and second camera, respectively.
    - x1, x2: Corresponding points in the first and second image, respectively.
    
    Returns:
    - A numpy array of shape (N, 4) containing the homogeneous coordinates of the triangulated points.
    """
    # Convert camera centers to column vectors
    C1 = np.reshape(C1, (3, 1))
    C2 = np.reshape(C2, (3, 1))

    # Compute projection matrices for each camera
    P1 = K1 @ np.hstack((R1, C1))
    P2 = K2 @ np.hstack((R2, C2))

    all_X = [] # Initialize list to store triangulated points
    for i in range(x1.shape[0]):
        # Skip points with NaN values
        if np.isnan(x1[i][0]) or np.isnan(x2[i][0]):
            all_X.append(np.zeros(4))
            continue

        # Construct matrix A for solving the homogeneous system Ax = 0
        A = np.vstack([(x1[i, 1] * P1[2, :] - P1[1, :]),
                       (P1[0, :] - x1[i, 0] * P1[2, :]),
                       (x2[i, 1] * P2[2, :] - P2[1, :]),
                       (P2[0, :] - x2[i, 0] * P2[2, :])])

        # Solve for x using SVD
        _, _, vt = np.linalg.svd(A)
        X = vt[-1]
        all_X.append(X / X[-1])  # Ensure last component is 1

    return np.array(all_X)

def Triangulation_nl(point3d, I, R1, C1, R2, C2, x1, x2):
    """
    Non-linear refinement of 3D point positions using iterative optimization.
    
    Args:
    - point3d: Initial 3D points obtained from linear triangulation.
    - I: Identity matrix representing camera intrinsic parameters.
    - R1, R2: Rotation matrices for the two cameras.
    - C1, C2: Translation vectors for the two cameras.
    - x1, x2: Corresponding points in the two images.
    
    Returns:
    - Refined 3D points after non-linear optimization.
    """
    p1 = I @ np.hstack((R1, C1))
    p2 = I @ np.hstack((R2, C2))
    lamb = 0.1  # Regularization parameter
    n_iter = 100  # Number of iterations
    X_new = point3d.copy()

    for i in range(point3d.shape[0]):
        pt = point3d[i, :]
        if pt[0] == 0:  # Skip points initialized to zero
            continue
        for _ in range(n_iter):
            pt_h = np.append(pt, 1)  # Convert to homogeneous coordinates
            proj1 = p1 @ pt_h
            proj1 /= proj1[2]  # Convert to inhomogeneous coordinates
            proj2 = p2 @ pt_h
            proj2 /= proj2[2]

            # Compute Jacobians
            dfdX1 = ComputePointJacobian(pt_h, R1, C1)
            dfdX2 = ComputePointJacobian(pt_h, R2, C2)

            # Compute Hessians and error gradients
            H1 = dfdX1.T @ dfdX1 + lamb * np.eye(3)
            H2 = dfdX2.T @ dfdX2 + lamb * np.eye(3)
            J1 = dfdX1.T @ (x1[i, :] - proj1[:2])
            J2 = dfdX2.T @ (x2[i, :] - proj2[:2])

            # Skip update if Hessians are singular
            if np.linalg.det(H1) == 0 or np.linalg.det(H2) == 0:
                continue

            # Compute update step and update point
            delta_pt = np.linalg.inv(H1) @ J1 + np.linalg.inv(H2) @ J2
            pt += delta_pt[:3]

        X_new[i, :] = pt

    return X_new

def ComputePointJacobian(X, R, C):
    """
    Compute the Jacobian of the projection of a 3D point with respect to the point coordinates.
    
    Args:
    - X: The 3D point in homogeneous coordinates.
    - R: The rotation matrix of the camera.
    - C: The camera center (translation vector).
    
    Returns:
    - The Jacobian matrix of the projection.
    """
    x = R @ X[:3] + C  # Project point using R and C
    u, v, w = x  # Unpack projected point
    du_dc = R[0, :]
    dv_dc = R[1, :]
    dw_dc = R[2, :]

    # Compute Jacobian components
    dfdX = np.vstack([((w * du_dc - u * dw_dc) / w**2),
                      ((w * dv_dc - v * dw_dc) / w**2)])
    return dfdX

def reprojectionError(openpose, projectionMatrix, keypoint3d):
    """
    Calculates the average reprojection error between 2D keypoints detected by OpenPose and
    the reprojection of 3D keypoints using a projection matrix.

    Parameters:
    - openpose: np.array, detected 2D keypoints in the image space.
    - projectionMatrix: np.array, the camera projection matrix.
    - keypoint3d: np.array, the 3D keypoints.

    Returns:
    - float, the average reprojection error.
    """
    # Augment 3D keypoints with ones for homogeneous coordinates
    keypoint3d_homo = np.hstack((keypoint3d, np.ones((keypoint3d.shape[0], 1))))
    # Project 3D points to 2D using the projection matrix
    reproject = (projectionMatrix @ keypoint3d_homo.T).T
    # Convert to non-homogeneous coordinates
    reproject[:, 0] /= reproject[:, 2]
    reproject[:, 1] /= reproject[:, 2]
    # Filter out invalid projections and original points
    valid = (reproject[:, 0] != 0) & (reproject[:, 1] != 0) & \
            (openpose[:, 0] != 0) & (openpose[:, 1] != 0)
    # Calculate distances between original and reprojected points only for valid points
    if np.any(valid):
        dist = np.linalg.norm(openpose[valid] - reproject[valid, :2], axis=1)
        return np.mean(dist)
    else:
        return 0

def reprojectedPoints(projectionMatrix, keypoint3d):
    """
    Reprojects 3D keypoints to 2D space using a projection matrix.

    Parameters:
    - projectionMatrix: np.array, the camera projection matrix.
    - keypoint3d: np.array, the 3D keypoints.

    Returns:
    - np.array, reprojected 2D keypoints.
    """
    # Same implementation as in reprojectionError for reprojecting points
    keypoint3d_homo = np.hstack((keypoint3d, np.ones((keypoint3d.shape[0], 1))))
    reproject = (projectionMatrix @ keypoint3d_homo.T).T
    reproject[:, 0] /= reproject[:, 2]
    reproject[:, 1] /= reproject[:, 2]
    # Set NaN values to zero
    reproject[np.isnan(reproject)] = 0
    return reproject[:, :2]

def Quaternion2Rotation(q):
    """
    Converts a quaternion to a rotation matrix.

    Parameters:
    - q: np.array, quaternion represented as [w, x, y, z].

    Returns:
    - np.array, the corresponding rotation matrix (3x3).
    """
    w, x, y, z = q
    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    return R

def Rotation2Quaternion(R):
    """
    Converts a rotation matrix to a quaternion.

    Parameters:
    - R: np.array, the rotation matrix (3x3).

    Returns:
    - np.array, the corresponding quaternion [w, x, y, z].
    """
    tr = R.trace()
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2  # S=4*qw
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) & (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
    return np.array([qw, qx, qy, qz])

def homogeneous_2D(point2D):
    """
    Converts a point from homogeneous to Cartesian coordinates in 2D.

    Parameters:
    - point2D: np.array, a 2D point in homogeneous coordinates.

    Returns:
    - np.array, the point in Cartesian coordinates.
    """
    # Assuming the input is already in homogeneous form ([x, y, w])
    return point2D[:2] / point2D[2]

def cartesian_3D_to_homogeneous(point3D):
    """
    Converts a 3D point in Cartesian coordinates to homogeneous coordinates.

    Parameters:
    - point3D: np.array, a 3D point in Cartesian coordinates.

    Returns:
    - np.array, the point in homogeneous coordinates.
    """
    return np.append(point3D, 1)

def homogeneous_cartesian(homo): 
    """
    Converts points from homogeneous to Cartesian coordinates.

    Parameters:
    - homo: np.array, an array of points in homogeneous coordinates with shape (n, 4),
            where n is the number of points.

    Returns:
    - np.array, the points converted to Cartesian coordinates with shape (n, 3).
    """
    # Ensure the input is a NumPy array for vectorized operations
    homo = np.array(homo)

    # Avoid division by zero by setting any zero in the fourth dimension to one
    # This prevents changing the original points but ensures no division by zero error
    homo[:, 3] = np.where(homo[:, 3] == 0, 1, homo[:, 3])
    
    # Vectorized operation to divide each x, y, z by w (the fourth element)
    # This converts all points to Cartesian coordinates efficiently
    cartesian = homo[:, :3] / homo[:, 3, np.newaxis]  # np.newaxis ensures correct broadcasting

    return cartesian
