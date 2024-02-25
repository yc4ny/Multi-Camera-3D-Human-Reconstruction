# Copyright (c) 2024 Yonwoo Choi
# All rights reserved.
# This code is part of my research at Seoul National University.
# Redistribution and use in source and binary forms, with or without
# modification, are not permitted without direct permission from the author.

import os
import cv2
import numpy as np
import scipy.io

# Function to calculate the average of non-NaN x or y points.
# Needed for drawing with OpenCV (Only allows INT)
def point_average(points, axis):
    """
    Calculate the average value of non-NaN points along a specified axis.

    :param points: numpy array of points
    :param axis: axis along which to calculate the average (0 for x, 1 for y)
    :return: average value along the specified axis
    """
    valid_points = points[~np.isnan(points[:, axis]), axis]
    if valid_points.size == 0:
        return 0
    return np.mean(valid_points)

# Process images from 5 cameras
for a in range(1, 6):
    dir_path = f'checkerboard/C{a}_keypoints'
    file_names = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    count = len(file_names)
    
    for i in range(1, count):
        print(f"Processing Image {i}....")
        img_path = f"checkerboard/C{a}_undistort/{i}.jpg"
        img = cv2.imread(img_path)
        mat_path = f"checkerboard/C{a}_keypoints/{i}.mat"
        mat = scipy.io.loadmat(mat_path)['imagePoints']

        # Skip frames with insufficient keypoints or small checkerboard patterns
        if mat.size == 0 or mat.shape[0] < 45 or \
           np.ptp(mat, axis=0)[0] < 200 or np.ptp(mat, axis=0)[1] < 400:
            cv2.imwrite(f"checkerboard/C{a}_drawCheckerboard/{i}.jpg", img)
            continue

        # Compare the current frame with the previous one (if exists) to check for large movements
        should_draw = True
        if i > 1:
            prev_mat = scipy.io.loadmat(f"checkerboard/C{a}_keypoints/{i-1}.mat")['imagePoints']
            if abs(point_average(mat, 0) - point_average(prev_mat, 0)) > 100 or \
               abs(point_average(mat, 1) - point_average(prev_mat, 1)) > 100:
                should_draw = False

        # Draw keypoints if the current frame passed all checks
        if should_draw:
            for j in range(mat.shape[0]):
                if not np.isnan(mat[j]).any():
                    cv2.circle(img, (int(mat[j][0]), int(mat[j][1])), 12, (0, 255, 0), -1)
        cv2.imwrite(f"checkerboard/C{a}_drawCheckerboard/{i}.jpg", img)
