import numpy as np

def LinearTriangulation(K1,K2, C1, R1, C2, R2, x1, x2):

    I = np.identity(3)
    C1 = np.reshape(C1, (3, 1))
    C2 = np.reshape(C2, (3, 1))

    P1 = np.matmul(K1, np.column_stack((R1,C1)))
    P2 = np.matmul(K2, np.column_stack((R2,C2)))

    p1T = P1[0,:].reshape(1,4)
    p2T = P1[1,:].reshape(1,4)
    p3T = P1[2,:].reshape(1,4)

    p_dash_1T = P2[0,:].reshape(1,4)
    p_dash_2T = P2[1,:].reshape(1,4)
    p_dash_3T = P2[2,:].reshape(1,4)

    all_X = []
    for i in range(x1.shape[0]):
        if np.isnan(x1[i][0]) == True or np.isnan(x2[i][0])==True:
            all_X.append(([0,0,0,0]))
            continue 

        x = x1[i,0]
        y = x1[i,1]
        x_dash = x2[i,0]
        y_dash = x2[i,1]


        A = []
        A.append((y * p3T) -  p2T)
        A.append(p1T -  (x * p3T))
        A.append((y_dash * p_dash_3T) -  p_dash_2T)
        A.append(p_dash_1T -  (x_dash * p_dash_3T))

        A = np.array(A).reshape(4,4)

        _, _, vt = np.linalg.svd(A)
        v = vt.T
        x = v[:,-1]
        all_X.append(x)
    return np.array(all_X)

def Triangulation_nl(point3d,I,R1,C1, R2, C2, x1, x2):
    p1 = np.matmul(I,np.column_stack((R1,C1)))
    p2 = np.matmul(I,np.column_stack((R2,C2)))
    lamb = 0.1
    n_iter = 100
    X_new = point3d.copy()

    for i in range(0,point3d.shape[0]):
        pt = point3d[i,:]
        if pt[0] == 0:
            continue
        for j in range(n_iter):
            pt = np.append(pt,1)
            proj1 = np.matmul(p1,pt)
            proj1 = proj1[:2] / proj1[2]
            proj2 = np.matmul(p2,pt)
            proj2 = proj2[:2] / proj2[2]

            dfdX1 = ComputePointJacobian(pt, R1,C1)
            dfdX2 = ComputePointJacobian(pt, R2,C2)

            H1 = dfdX1.T @ dfdX1 + lamb * np.eye(3)
            H2 = dfdX2.T @ dfdX2 + lamb * np.eye(3)
            J1 = dfdX1.T @ (x1[i,:] - proj1)
            J2 = dfdX2.T @ (x2[i,:] - proj2)
            if np.linalg.det(H1) == 0 or np.linalg.det(H2) == 0:
                continue
            delta_pt = np.linalg.inv(H1) @ J1 + np.linalg.inv(H2) @ J2
            pt = pt[:3]
            pt += delta_pt

        X_new[i,:] = pt
    return X_new

def ComputePointJacobian(X, R,C):
    x = np.matmul(np.column_stack((R,C)), X)
    u = x[0]
    v = x[1]
    w = x[2]
    du_dc = R[0, :]
    dv_dc = R[1, :]
    dw_dc = R[2, :]

    dfdX = np.stack([
        (w * du_dc - u * dw_dc) / (w**2),
        (w * dv_dc - v * dw_dc) / (w**2)
    ], axis=0)

    return dfdX

def reprojectionError(openpose, projectionMatrix, keypoint3d): 
    #Reprojection to original image space
    reproject = np.zeros((openpose.shape[0],2))
    ones = np.ones((openpose.shape[0],1))
    keypoint3d = np.column_stack((keypoint3d,ones))
    for i in range (0,openpose.shape[0]):  
        if keypoint3d[i][0] == 0:
            reproject[i][0] = 0 
            reproject[i][1] = 1
            continue 
        point2D = np.matmul(projectionMatrix ,keypoint3d[i])
        homogeneous_2D(point2D)
        reproject[i][0] = point2D[0]
        reproject[i][1] = point2D[1]
        reproject[np.isnan(reproject)] = 0 
    
    #Calculating Reprojection error 
    sum_error = 0; 
    numPoints = 0;
    for i in range (0,openpose.shape[0]):
        if reproject[i][0]==0 or reproject[i][1]==0 or openpose[i][0] == 0 or openpose[i][1] ==0:
             continue 
        dist = np.linalg.norm(openpose[i]-reproject[i])
        sum_error = sum_error + dist 
        numPoints += 1 

    return sum_error/numPoints

def reprojectedPoints(projectionMatrix, keypoint3d): 
    reproject = np.zeros((keypoint3d.shape[0],2))
    ones = np.ones((keypoint3d.shape[0],1))
    keypoint3d = np.column_stack((keypoint3d,ones))
    for i in range (0,keypoint3d.shape[0]):  
        if keypoint3d[i][0] == 0:
            continue
        point2D = np.matmul(projectionMatrix ,keypoint3d[i])
        homogeneous_2D(point2D)
        reproject[i][0] = point2D[0]
        reproject[i][1] = point2D[1]
        reproject[np.isnan(reproject)] = 0 
    return reproject 

def Quaternion2Rotation(q):
    """
    Convert a quaternion to rotation matrix
    Parameters
    ----------
    q : ndarray of shape (4,)
        Unit quaternion (w, x, y, z)
    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    """
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    R = np.empty([3, 3])
    R[0, 0] = 1 - 2 * y**2 - 2 * z**2
    R[0, 1] = 2 * (x*y - z*w)
    R[0, 2] = 2 * (x*z + y*w)

    R[1, 0] = 2 * (x*y + z*w)
    R[1, 1] = 1 - 2 * x**2 - 2 * z**2
    R[1, 2] = 2 * (y*z - x*w)

    R[2, 0] = 2 * (x*z - y*w)
    R[2, 1] = 2 * (y*z + x*w)
    R[2, 2] = 1 - 2 * x**2 - 2 * y**2

    return R

def Rotation2Quaternion(R):
    """
    Convert a rotation matrix to quaternion
    Parameters
    ----------
    R : ndarray of shape (3, 3)
        Rotation matrix
    Returns
    -------
    q : ndarray of shape (4,)
        The unit quaternion (w, x, y, z)
    """
    q = np.empty([4,])

    tr = np.trace(R)
    if tr < 0:
        i = R.diagonal().argmax()
        j = (i + 1) % 3
        k = (j + 1) % 3

        q[i] = np.sqrt(1 - tr + 2 * R[i, i]) / 2
        q[j] = (R[j, i] + R[i, j]) / (4 * q[i])
        q[k] = (R[k, i] + R[i, k]) / (4 * q[i])
        q[3] = (R[k, j] - R[j, k]) / (4 * q[i])
    else:
        q[3] = np.sqrt(1 + tr) / 2
        q[0] = (R[2, 1] - R[1, 2]) / (4 * q[3])
        q[1] = (R[0, 2] - R[2, 0]) / (4 * q[3])
        q[2] = (R[1, 0] - R[0, 1]) / (4 * q[3])

    q /= np.linalg.norm(q)
    # Rearrange (x, y, z, w) to (w, x, y, z)
    q = q[[3, 0, 1, 2]]

    return q

def homogeneous_2D(point2D): 
    point2D[0] = point2D[0]/point2D[2]
    point2D[1] = point2D[1]/point2D[2]
    point2D[2] = point2D[2]/point2D[2]
    return point2D

def homogeneous_cartesian(homo): 
    for j in range (homo.shape[0]): 
        if homo[j][0] == 0:
            continue
        homo[j][0] = homo[j][0] / homo[j][3]
        homo[j][1] = homo[j][1] / homo[j][3]
        homo[j][2] = homo[j][2] / homo[j][3]
        homo[j][3] = homo[j][3] / homo[j][3]   
    
    return homo
