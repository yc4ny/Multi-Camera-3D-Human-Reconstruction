# Copyright (c) 2024 Yonwoo Choi
# All rights reserved.
# This code is part of my research at Seoul National University.
# Redistribution and use in source and binary forms, with or without
# modification, are not permitted without direct permission from the author.

import numpy as np 
import cv2
import json 
import matplotlib.pyplot as plt

def loadOpenPose(jsonFile):
    """
    Loads OpenPose data from a JSON file and reshapes it into a (25, 3) array.

    Parameters:
    - jsonFile: Path to the JSON file containing OpenPose data.

    Returns:
    - openpose: A numpy array of shape (25, 3), where each row represents a keypoint (x, y, confidence).
    """
    with open(jsonFile) as f:
        data = json.load(f)
        keypoints = data['people'][0]['pose_keypoints_2d']
        openpose = np.reshape(keypoints, (25, 3))
    return openpose

def view3Djoints(keypoints3D):
    """
    Visualizes 3D joints in world space using matplotlib.

    Parameters:
    - keypoints3D: A numpy array of shape (N, 3), where N is the number of keypoints,
                    and each row represents a keypoint (x, y, z).
    """
    fig = plt.figure(figsize=(18, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Define connections between keypoints in a list of tuples (start, end)
    connections = [(0, 1), (1, 8), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (8, 11), (11, 12),
                   (12, 13), (8, 14), (14, 15), (15, 16), (0, 17), (0, 18), (14, 19), (19, 20), 
                   (14, 21), (11, 22), (22, 23), (11, 24)]

    # Plot keypoints
    for x, y, z in keypoints3D:
        ax.scatter(y, z, x)

    # Draw connections
    for start, end in connections:
        if not (keypoints3D[start][1] == 0 and keypoints3D[start][0] in [15, 16, 17, 18]):
            ax.plot3D([keypoints3D[start][1], keypoints3D[end][1]],
                      [keypoints3D[start][2], keypoints3D[end][2]],
                      [keypoints3D[start][3], keypoints3D[end][3]])

    plt.show()

def visualizeReprojectionError(openpose,reprojectedPoints,image, saveName):
    """
    Visualize the reprojection error by drawing lines between original and reprojected keypoints
    on the given image and saving the result.
    
    Parameters:
    - openpose: np.array, original keypoints detected by OpenPose in the format (x, y, confidence).
    - reprojected_points: np.array, reprojected keypoints in the format (x, y).
    - image: str, path to the input image file.
    - save_name: str, path to save the image with visualization.
    """
    img = cv2.imread(image)
    
    # Define pairs of keypoints for drawing lines, considering visibility and existence.
    keypoint_pairs = [
        (0, 15), (0, 16), (15, 17), (16, 18), (0, 1), (1, 2),
        (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8),
        (8, 9), (9, 10), (10, 11), (11, 24), (11, 22), (22, 23),
        (12, 13), (13, 14), (14, 21), (14, 19), (19, 20)
    ]

    # Draw lines for each pair if both keypoints are visible
    for start, end in keypoint_pairs:
        if np.all(openpose[[start, end], :2] != 0, axis=1).all():
            img = cv2.line(img, tuple(openpose[start, :2].astype(int)), tuple(openpose[end, :2].astype(int)), (0, 128, 255), 10)
        if np.all(reprojectedPoints[[start, end], :] != 0, axis=1).all():
            img = cv2.line(img, tuple(reprojectedPoints[start, :].astype(int)), tuple(reprojectedPoints[end, :].astype(int)), (0, 165, 0), 5)

    # Draw circles for original and reprojected keypoints
    for i in range(24):
        if np.any(openpose[i, :2] != 0):
            img = cv2.circle(img, tuple(openpose[i, :2].astype(int)), 10, (0, 0, 255), -1)
        if np.any(reprojectedPoints[i, :] != 0):
            img = cv2.circle(img, tuple(reprojectedPoints[i, :].astype(int)), 10, (0, 255, 0), -1)

    # Save the image with drawn lines and keypoints
    cv2.imwrite(saveName, img)