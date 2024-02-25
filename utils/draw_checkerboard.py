import os
import cv2
import numpy as np 
import scipy.io 
from utils.concat_checkerboard import concat

def pointAverage_x(mat):
    sum = 0
    elements = 0
    for i in range(mat.shape[0]):
        if np.isnan(mat[i][0]):
            continue 
        sum = sum + mat[i][0]
        elements += 1
    if elements == 0:
        return 0
    return sum/elements

def pointAverage_y(mat):
    sum = 0
    elements = 0
    for i in range(mat.shape[0]):
        if np.isnan(mat[i][1]):
            continue 
        sum = sum + mat[i][1]
        elements += 1
    return sum/elements
    


# 5 Cameras: Able to adjust camera by range
for a in range(1,6):
    #Count number of frames to draw on
    dir_path = r'checkerboard/C'+ str(a)+'_keypoints'
    count = 0
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
    #
    for i in range (1,count):
        bool = True
        print("Image " + str(i) + "....")
        img = cv2.imread("checkerboard/C" + str(a) + "_undistort/" + str(i) + ".jpg")
        mat = scipy.io.loadmat("checkerboard/C" + str(a) + "_keypoints/" + str(i) + ".mat")
        mat = mat['imagePoints']

        if mat.size == 0 or mat.shape[0] <45 or (np.nanmax(mat,0) - np.nanmin(mat,0))[0] <200 or (np.nanmax(mat,0) - np.nanmin(mat,0))[1]<400: 
            cv2.imwrite("checkerboard/C" + str(a) + "_drawCheckerboard/" +str(i) + ".jpg", img)
            continue
        
        if i >1: 
            mat_before = scipy.io.loadmat("checkerboard/C" + str(a) + "_keypoints/" + str(i-1) + ".mat")
            mat_before = mat_before['imagePoints']
            if abs(pointAverage_x(mat)-pointAverage_x(mat_before)) > 100 or abs(pointAverage_y(mat)-pointAverage_y(mat_before))>100:
                bool = False

        if bool is False:
            cv2.imwrite("checkerboard/C" + str(a) + "_drawCheckerboard/" +str(i) + ".jpg", img)
        if bool is True:
            for j in range (mat.shape[0]):
                if np.isnan(mat[j][0]) or np.isnan(mat[j][1]) :
                    continue
                check = cv2.circle(img, (int(mat[j][0]), int(mat[j][1])), 12, (0,255,0),-1)
                cv2.imwrite("checkerboard/C" + str(a) + "_drawCheckerboard/" +str(i) + ".jpg", check)

    
