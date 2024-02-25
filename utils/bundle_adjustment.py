import numpy as np 
from scipy.optimize import least_squares
from utils.stereo_calib import Rotation2Quaternion
from utils.stereo_calib import Quaternion2Rotation

def SetupBundleAdjustment(P, X, track):
    """
    Setup bundle adjustment
    Parameters
    ----------
    P : ndarray of shape (K, 3, 4)
        Set of reconstructed camera poses
    X : ndarray of shape (J, 3)
        Set of reconstructed 3D points
    track : ndarray of shape (K, J, 2)
        Tracks for the reconstructed cameras
    Returns
    -------
    z : ndarray of shape (7K+3J,)
        The optimization variable that is made of all camera poses and 3D points
    b : ndarray of shape (2M,)
        The 2D points in track, where M is the number of 2D visible points
    S : ndarray of shape (2M, 7K+3J)
        The sparse indicator matrix that indicates the locations of Jacobian computation
    camera_index : ndarray of shape (M,)
        The index of camera for each measurement
    point_index : ndarray of shape (M,)
        The index of 3D point for each measurement
    """
    n_cameras = P.shape[0]
    n_points = X.shape[0]

    n_projs = np.sum(track[:,:,0] != -1)
    b = np.zeros((2*n_projs,))
    S = np.zeros((2*n_projs, 7*n_cameras+3*n_points), dtype=bool)
    k = 0
    camera_index = []
    point_index = []
    for i in range(n_cameras):
        for j in range(n_points):
            if track[i, j, 0] != -1:
                if i not in (0, 1):
                    S[2*k : 2*(k+1), 7*i : 7*(i+1)] = 1
                S[2*k : 2*(k+1), 7*n_cameras+3*j : 7*n_cameras+3*(j+1)] = 1
                b[2*k : 2*(k+1)] = track[i, j, :]
                camera_index.append(i)
                point_index.append(j)
                k += 1
    camera_index = np.asarray(camera_index)
    point_index = np.asarray(point_index)
    
    z = np.zeros((7*n_cameras+3*n_points,))
    for i in range(n_cameras):
        R = P[i, :, :3]
        C = -R.T @ P[i, :, 3]
        q = Rotation2Quaternion(R)
        p = np.concatenate([C, q])
        z[7*i : 7*(i+1)] = p
    # for i in range(n_points):
    #     z[7*n_cameras+3*i : 7*n_cameras+3*(i+1)] = X[i, :]
    z[7*n_cameras:] = X.ravel()

    return z, b, S, camera_index, point_index
    

def MeasureReprojection(z, b, n_cameras, n_points, camera_index, point_index):
    """
    Evaluate the reprojection error
    Parameters
    ----------
    z : ndarray of shape (7K+3J,)
        Optimization variable
    b : ndarray of shape (2M,)
        2D measured points
    n_cameras : int
        Number of cameras
    n_points : int
        Number of 3D points
    camera_index : ndarray of shape (M,)
        Index of camera for each measurement
    point_index : ndarray of shape (M,)
        Index of 3D point for each measurement
    Returns
    -------
    err : ndarray of shape (2M,)
        The reprojection error
    """
    n_projs = camera_index.shape[0]
    f = np.zeros((2*n_projs,))
    for k, (i, j) in enumerate(zip(camera_index, point_index)):
        p = z[7*i : 7*(i+1)]
        X = z[7*n_cameras+3*j : 7*n_cameras+3*(j+1)]
        R = Quaternion2Rotation(p[3:] / np.linalg.norm(p[3:]))
        C = p[:3]
        proj = R @ (X - C)
        proj = proj / proj[2]
        f[2*k : 2*(k+1)] = proj[:2]
    err = b - f

    return err


def UpdatePosePoint(z, n_cameras, n_points):
    """
    Update the poses and 3D points
    Parameters
    ----------
    z : ndarray of shape (7K+3J,)
        Optimization variable
    n_cameras : int
        Number of cameras
    n_points : int
        Number of 3D points
    Returns
    -------
    P_new : ndarray of shape (K, 3, 4)
        The set of refined camera poses
    X_new : ndarray of shape (J, 3)
        The set of refined 3D points
    """
    P_new = np.empty((n_cameras, 3, 4))
    for i in range(n_cameras):
        p = z[7*i : 7*(i+1)]
        q = p[3:]
        R = Quaternion2Rotation(q / np.linalg.norm(q))
        C = p[:3]
        P_new[i,:,:] = R @ np.hstack([np.eye(3), -C[:,np.newaxis]]) 

    X_new = np.reshape(z[7*n_cameras:], (-1,3))

    return P_new, X_new


def RunBundleAdjustment(P, X, track):
    """
    Run bundle adjustment
    Parameters
    ----------
    P : ndarray of shape (K, 3, 4)
        Set of reconstructed camera poses
    X : ndarray of shape (J, 3)
        Set of reconstructed 3D points
    track : ndarray of shape (K, J, 2)
        Tracks for the reconstructed cameras
    Returns
    -------
    P_new : ndarray of shape (K, 3, 4)
        The set of refined camera poses
    X_new : ndarray of shape (J, 3)
        The set of refined 3D points
    """
    n_cameras = P.shape[0]
    n_points = X.shape[0]

    z0, b, S, camera_index, point_index  = SetupBundleAdjustment(P, X, track)

    res = least_squares(
        lambda x : MeasureReprojection(x, b, n_cameras, n_points, camera_index, point_index),
        z0,
        jac_sparsity=S,
        verbose=2
    )
    z = res.x

    err0 = MeasureReprojection(z0, b, n_cameras, n_points, camera_index, point_index)
    err = MeasureReprojection(z, b, n_cameras, n_points, camera_index, point_index)
    print('Reprojection error {} -> {}'.format(np.linalg.norm(err0), np.linalg.norm(err)))

    P_new, X_new = UpdatePosePoint(z, n_cameras, n_points)

    return P_new, X_new