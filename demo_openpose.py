# Copyright (c) 2024 Yonwoo Choi
# All rights reserved.
# This code is part of my research at Seoul National University.
# Redistribution and use in source and binary forms, with or without
# modification, are not permitted without direct permission from the author.

import numpy as np
import os
import visualizer.vis_openpose as viz
import utils.triangulation as stereo
import utils.concat_openpose_imgs as concat_openpose_imgs

if __name__ == '__main__':
    
    # Initialize Camera Calibration Parameters 
    C1_Intrinsic = np.array([
        [1793.40866450411, 0, 1941.82280218514], 
        [0, 1784.34883485341, 1079.49235904473],
        [0., 0., 1.]
    ])

    C1_radialDistortion = np.array([
        [-0.247779937548622],
        [0.0637764128399176]
    ])

    C2_Intrinsic = np.array([
        [1795.70365788380, 0, 1877.62446127849],
        [0, 1789.66009806022, 1118.82747427708],
        [0., 0., 1.]
    ])
    
    C2_radialDistortion = np.array([
        [-0.239496853407498],
        [0.0546352045342115]
    ])

    C1_Rotation = np.eye(3)  # Identity matrix for default rotation
    C2_Rotation = np.array([
        [0.911619921535167,-0.124455570407734,0.391739619715318],
        [0.0735810919149552,0.987075830182677,0.142362664985526 ],
        [-0.404394537021483,-0.100956012518640, 0.908995567625892]
    ])

    C1_Translation = np.zeros((3,1))  # Zero vector for default translation
    C2_Translation = np.array([
        [-1520.23416825357],
        [-191.344404566789],
        [331.725213364265]
    ])
    
    # Projection Matrix = K[R|t]
    C1_ProjectionMatrix = np.hstack((C1_Intrinsic, np.dot(C1_Intrinsic, np.hstack((C1_Rotation, C1_Translation)))))
    C2_ProjectionMatrix = np.hstack((C2_Intrinsic, np.dot(C2_Intrinsic, np.hstack((C2_Rotation, C2_Translation)))))

    # Count number of frames to process 
    dir_path = r'C1/C1_keypoints'
    count = len([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))])

    # Triangulation using DLT
    keypoint_results_dlt = []
    total_error_dlt = 0
    error_track_dlt = []

    for i in range(1, count + 1):
        error = 0
        C1_op = viz.loadOpenPose(f'C1/C1_keypoints/{i}_keypoints.json')[:,:2]
        C2_op = viz.loadOpenPose(f'C2/C2_keypoints/{i}_keypoints.json')[:,:2]

        keypoint3d = stereo.LinearTriangulation(C1_Intrinsic, C2_Intrinsic, C1_Translation, C1_Rotation, C2_Translation, C2_Rotation, C1_op, C2_op)
        keypoint3d = stereo.homogeneous_cartesian(keypoint3d)[:,:3]
        keypoint_results_dlt.append(keypoint3d)

        C1_reprojected = stereo.reprojectedPoints(C1_ProjectionMatrix, keypoint3d)
        C2_reprojected = stereo.reprojectedPoints(C2_ProjectionMatrix, keypoint3d)

        viz.visualizeReprojectionError(C1_op, C1_reprojected, f'C1/C1_undistorted/{i}.jpg', f'C1/C1_dlt/{i}_reproject.jpg')
        viz.visualizeReprojectionError(C2_op, C2_reprojected, f'C2/C2_undistorted/{i}.jpg', f'C2/C2_dlt/{i}_reproject.jpg')

        error = (stereo.reprojectionError(C1_op, C1_ProjectionMatrix, keypoint3d) + stereo.reprojectionError(C2_op, C2_ProjectionMatrix, keypoint3d)) / 2
        error_track_dlt.append(error)
        total_error_dlt += error

        print('Frame:', i)
        print('Error:', error)

    print("DLT Total Error:", total_error_dlt / count)
    np.save('keypoints_3d/dlt.npy', keypoint_results_dlt, True, True)
    np.save('concat/dlt_error.npy', error_track_dlt, True, True)

    # Triangulation with non-linear optimization
    keypoint_results_nl = []
    total_error_nl = 0
    error_track_nl = []

    for i in range(1, count + 1):
        error = 0
        C1_op = viz.loadOpenPose(f'C1/C1_keypoints/{i}_keypoints.json')[:,:2]
        C2_op = viz.loadOpenPose(f'C2/C2_keypoints/{i}_keypoints.json')[:,:2]

        keypoint3d = stereo.LinearTriangulation(C1_Intrinsic, C2_Intrinsic, C1_Translation, C1_Rotation, C2_Translation, C2_Rotation, C1_op, C2_op)
        keypoint3d = stereo.homogeneous_cartesian(keypoint3d)[:,:3]

        nonlinear3d = stereo.Triangulation_nl(keypoint3d, C1_ProjectionMatrix, C2_ProjectionMatrix, C1_op, C2_op)
        keypoint_results_nl.append(nonlinear3d)

        C1_reprojected = stereo.reprojectedPoints(C1_ProjectionMatrix, nonlinear3d)
        C2_reprojected = stereo.reprojectedPoints(C2_ProjectionMatrix, nonlinear3d)

        viz.visualizeReprojectionError(C1_op, C1_reprojected, f'C1/C1_undistorted/{i}.jpg', f'C1/C1_nonlinear/{i}_reproject.jpg')
        viz.visualizeReprojectionError(C2_op, C2_reprojected, f'C2/C2_undistorted/{i}.jpg', f'C2/C2_nonlinear/{i}_reproject.jpg')

        error = (stereo.reprojectionError(C1_op, C1_ProjectionMatrix, nonlinear3d) + stereo.reprojectionError(C2_op, C2_ProjectionMatrix, nonlinear3d)) / 2
        error_track_nl.append(error)
        total_error_nl += error

        print('Frame:', i)
        print('Error:', error)

    print("Non-linear Optimization Total Error:", total_error_nl / count)
    np.save('keypoints_3d/nl.npy', keypoint_results_nl, True, True)
    np.save('concat/nl_error.npy', error_track_nl, True, True)

    # Concat Images 
    concat_openpose_imgs.concat()
