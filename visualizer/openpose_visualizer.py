import numpy as np 
import cv2
import json 
import matplotlib.pyplot as plt

# Load openpose to correct format (x,y,c) c:confidence
def loadOpenPose(jsonFile):
    with open(jsonFile) as f:
        d = json.load(f)
        openpose = d['people'][0]['pose_keypoints_2d']
        openpose = np.reshape(openpose, (25, 3))
    return openpose

# View 3D Joint in World Space  
def view3Djoints(keypoints3D): 
    fig = plt.figure(figsize=(18,7))
    ax= fig.add_subplot(111, projection = '3d')
    for i in range (0,24):
        if keypoints3D[i][1]==0 and keypoints3D[i][0] is 15 or 16 or 17 or 18:
            continue
        ax.scatter(keypoints3D[i][1],keypoints3D[i][2], keypoints3D[i][3])
    ax.plot3D([keypoints3D[0][1],keypoints3D[1][1]], [keypoints3D[0][2],keypoints3D[1][2]],[keypoints3D[0][3],keypoints3D[1][3]])
    ax.plot3D([keypoints3D[1][1],keypoints3D[8][1]], [keypoints3D[1][2],keypoints3D[8][2]],[keypoints3D[1][3],keypoints3D[8][3]])
    ax.plot3D([keypoints3D[1][1],keypoints3D[2][1]], [keypoints3D[1][2],keypoints3D[2][2]],[keypoints3D[1][3],keypoints3D[2][3]])
    ax.plot3D([keypoints3D[2][1],keypoints3D[5][1]], [keypoints3D[2][2],keypoints3D[5][2]],[keypoints3D[2][3],keypoints3D[5][3]])    
    ax.plot3D([keypoints3D[2][1],keypoints3D[3][1]], [keypoints3D[2][2],keypoints3D[3][2]],[keypoints3D[2][3],keypoints3D[3][3]])   
    ax.plot3D([keypoints3D[5][1],keypoints3D[6][1]], [keypoints3D[5][2],keypoints3D[6][2]],[keypoints3D[5][3],keypoints3D[6][3]])     
    ax.plot3D([keypoints3D[3][1],keypoints3D[4][1]], [keypoints3D[3][2],keypoints3D[4][2]],[keypoints3D[3][3],keypoints3D[4][3]])     
    ax.plot3D([keypoints3D[7][1],keypoints3D[6][1]], [keypoints3D[7][2],keypoints3D[6][2]],[keypoints3D[7][3],keypoints3D[6][3]])     
    ax.plot3D([keypoints3D[8][1],keypoints3D[11][1]], [keypoints3D[8][2],keypoints3D[11][2]],[keypoints3D[8][3],keypoints3D[11][3]])  
    ax.plot3D([keypoints3D[8][1],keypoints3D[14][1]], [keypoints3D[8][2],keypoints3D[14][2]],[keypoints3D[8][3],keypoints3D[14][3]])  
    ax.plot3D([keypoints3D[20][1],keypoints3D[14][1]], [keypoints3D[20][2],keypoints3D[14][2]],[keypoints3D[20][3],keypoints3D[14][3]])  
    ax.plot3D([keypoints3D[23][1],keypoints3D[11][1]], [keypoints3D[23][2],keypoints3D[11][2]],[keypoints3D[23][3],keypoints3D[11][3]])  
  
    plt.show()

def visualizeReprojectionError(openpose,reprojectedPoints,image, saveName):
    # openpose = np.delete(openpose, np.s_[-1:],axis=1)
    img = cv2.imread(image)
    if openpose[0][0] == 0 or openpose[0][1] ==0 or  openpose[15][0] == 0 or openpose[15][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(openpose[0][0]),int(openpose[0][1])),(int(openpose[15][0]),int(openpose[15][1])),  (0,128,255),10)
    if openpose[0][0] == 0 or openpose[0][1] ==0 or  openpose[16][0] == 0 or openpose[16][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(openpose[0][0]),int(openpose[0][1])),(int(openpose[16][0]),int(openpose[16][1])),  (0,128,255),10)
    if openpose[15][0] == 0 or openpose[15][1] ==0 or  openpose[17][0] == 0 or openpose[17][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(openpose[15][0]),int(openpose[15][1])),(int(openpose[17][0]),int(openpose[17][1])),  (0,128,255),10)
    if openpose[16][0] == 0 or openpose[16][1] ==0 or  openpose[18][0] == 0 or openpose[18][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(openpose[16][0]),int(openpose[16][1])),(int(openpose[18][0]),int(openpose[18][1])), (0,128,255),10)
    if openpose[0][0] == 0 or openpose[0][1] ==0 or  openpose[1][0] == 0 or openpose[1][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(openpose[0][0]),int(openpose[0][1])),(int(openpose[1][0]),int(openpose[1][1])),  (0,128,255),10)
    if openpose[1][0] == 0 or openpose[1][1] ==0 or  openpose[2][0] == 0 or openpose[2][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(openpose[1][0]),int(openpose[1][1])),(int(openpose[2][0]),int(openpose[2][1])),  (0,128,255),10)
    if openpose[1][0] == 0 or openpose[1][1] ==0 or  openpose[5][0] == 0 or openpose[5][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(openpose[1][0]),int(openpose[1][1])),(int(openpose[5][0]),int(openpose[5][1])),  (0,128,255), 10)
    if openpose[2][0] == 0 or openpose[2][1] ==0 or  openpose[3][0] == 0 or openpose[3][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(openpose[2][0]),int(openpose[2][1])),(int(openpose[3][0]),int(openpose[3][1])),  (0,128,255),10)

    if openpose[3][0] == 0 or openpose[3][1] ==0 or  openpose[4][0] == 0 or openpose[4][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(openpose[3][0]),int(openpose[3][1])),(int(openpose[4][0]),int(openpose[4][1])), (0,128,255),10)
    if openpose[5][0] == 0 or openpose[5][1] ==0 or  openpose[6][0] == 0 or openpose[6][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(openpose[5][0]),int(openpose[5][1])),(int(openpose[6][0]),int(openpose[6][1])),  (0,128,255),10)
    if openpose[6][0] == 0 or openpose[6][1] ==0 or  openpose[7][0] == 0 or openpose[7][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(openpose[6][0]),int(openpose[6][1])),(int(openpose[7][0]),int(openpose[7][1])),  (0,128,255),10)
    if openpose[1][0] == 0 or openpose[1][1] ==0 or  openpose[8][0] == 0 or openpose[8][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(openpose[1][0]),int(openpose[1][1])),(int(openpose[8][0]),int(openpose[8][1])), (0,128,255),10)
    if openpose[8][0] == 0 or openpose[8][1] ==0 or  openpose[9][0] == 0 or openpose[9][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(openpose[8][0]),int(openpose[8][1])),(int(openpose[9][0]),int(openpose[9][1])),  (0,128,255),10)
    if openpose[8][0] == 0 or openpose[8][1] ==0 or  openpose[12][0] == 0 or openpose[12][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(openpose[8][0]),int(openpose[8][1])),(int(openpose[12][0]),int(openpose[12][1])),  (0,128,255),10)
    if openpose[9][0] == 0 or openpose[9][1] ==0 or  openpose[10][0] == 0 or openpose[10][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(openpose[9][0]),int(openpose[9][1])),(int(openpose[10][0]),int(openpose[10][1])),  (0,128,255),10)
    if openpose[10][0] == 0 or openpose[10][1] ==0 or  openpose[11][0] == 0 or openpose[11][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(openpose[10][0]),int(openpose[10][1])),(int(openpose[11][0]),int(openpose[11][1])),  (0,128,255),10)
    if openpose[11][0] == 0 or openpose[11][1] ==0 or  openpose[24][0] == 0 or openpose[24][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(openpose[11][0]),int(openpose[11][1])),(int(openpose[24][0]),int(openpose[24][1])),  (0,128,255),10)
    if openpose[11][0] == 0 or openpose[11][1] ==0 or  openpose[22][0] == 0 or openpose[22][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(openpose[11][0]),int(openpose[11][1])),(int(openpose[22][0]),int(openpose[22][1])), (0,128,255),10)
    if openpose[22][0] == 0 or openpose[22][1] ==0 or  openpose[23][0] == 0 or openpose[23][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(openpose[22][0]),int(openpose[22][1])),(int(openpose[23][0]),int(openpose[23][1])),  (0,128,255),10)
    if openpose[12][0] == 0 or openpose[12][1] ==0 or openpose[13][0] == 0 or openpose[13][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(openpose[12][0]),int(openpose[12][1])),(int(openpose[13][0]),int(openpose[13][1])),  (0,128,255),10)
    if openpose[13][0] == 0 or openpose[13][1] ==0 or  openpose[14][0] == 0 or openpose[14][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(openpose[13][0]),int(openpose[13][1])),(int(openpose[14][0]),int(openpose[14][1])),  (0,128,255),10)
    if openpose[14][0] == 0 or openpose[14][1] ==0 or  openpose[21][0] == 0 or openpose[21][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(openpose[14][0]),int(openpose[14][1])),(int(openpose[21][0]),int(openpose[21][1])),  (0,128,255),10)
    if openpose[14][0] == 0 or openpose[14][1] ==0 or openpose[19][0] == 0 or openpose[19][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(openpose[14][0]),int(openpose[14][1])),(int(openpose[19][0]),int(openpose[19][1])), (0,128,255),10)
    if openpose[19][0] == 0 or openpose[19][1] ==0 or  openpose[20][0] == 0 or openpose[20][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(openpose[19][0]),int(openpose[19][1])),(int(openpose[20][0]),int(openpose[20][1])), (0,128,255),10)

    if reprojectedPoints[0][0] == 0 or reprojectedPoints[0][1] ==0 or  reprojectedPoints[15][0] == 0 or reprojectedPoints[15][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(reprojectedPoints[0][0]),int(reprojectedPoints[0][1])),(int(reprojectedPoints[15][0]),int(reprojectedPoints[15][1])), (0,165,0),5)
    if reprojectedPoints[0][0] == 0 or reprojectedPoints[0][1] ==0 or  reprojectedPoints[16][0] == 0 or reprojectedPoints[16][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(reprojectedPoints[0][0]),int(reprojectedPoints[0][1])),(int(reprojectedPoints[16][0]),int(reprojectedPoints[16][1])), (0,165,0),5)
    if reprojectedPoints[15][0] == 0 or reprojectedPoints[15][1] ==0 or  reprojectedPoints[17][0] == 0 or reprojectedPoints[17][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(reprojectedPoints[15][0]),int(reprojectedPoints[15][1])),(int(reprojectedPoints[17][0]),int(reprojectedPoints[17][1])), (0,165,0),5)
    if reprojectedPoints[16][0] == 0 or reprojectedPoints[16][1] ==0 or  reprojectedPoints[18][0] == 0 or reprojectedPoints[18][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(reprojectedPoints[16][0]),int(reprojectedPoints[16][1])),(int(reprojectedPoints[18][0]),int(reprojectedPoints[18][1])), (0,165,0),5)
    if reprojectedPoints[0][0] == 0 or reprojectedPoints[0][1] ==0 or  reprojectedPoints[1][0] == 0 or reprojectedPoints[1][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(reprojectedPoints[0][0]),int(reprojectedPoints[0][1])),(int(reprojectedPoints[1][0]),int(reprojectedPoints[1][1])), (0,165,0),5)
    if reprojectedPoints[1][0] == 0 or reprojectedPoints[1][1] ==0 or  reprojectedPoints[2][0] == 0 or reprojectedPoints[2][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(reprojectedPoints[1][0]),int(reprojectedPoints[1][1])),(int(reprojectedPoints[2][0]),int(reprojectedPoints[2][1])), (0,165,0),5)
    if reprojectedPoints[1][0] == 0 or reprojectedPoints[1][1] ==0 or  reprojectedPoints[5][0] == 0 or reprojectedPoints[5][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(reprojectedPoints[1][0]),int(reprojectedPoints[1][1])),(int(reprojectedPoints[5][0]),int(reprojectedPoints[5][1])), (0,165,0),5)
    if reprojectedPoints[2][0] == 0 or reprojectedPoints[2][1] ==0 or  reprojectedPoints[3][0] == 0 or reprojectedPoints[3][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(reprojectedPoints[2][0]),int(reprojectedPoints[2][1])),(int(reprojectedPoints[3][0]),int(reprojectedPoints[3][1])),(0,165,0),5)

    if reprojectedPoints[3][0] == 0 or reprojectedPoints[3][1] ==0 or  reprojectedPoints[4][0] == 0 or reprojectedPoints[4][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(reprojectedPoints[3][0]),int(reprojectedPoints[3][1])),(int(reprojectedPoints[4][0]),int(reprojectedPoints[4][1])),(0,165,0),5)
    if reprojectedPoints[5][0] == 0 or reprojectedPoints[5][1] ==0 or  reprojectedPoints[6][0] == 0 or reprojectedPoints[6][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(reprojectedPoints[5][0]),int(reprojectedPoints[5][1])),(int(reprojectedPoints[6][0]),int(reprojectedPoints[6][1])), (0,165,0),5)
    if reprojectedPoints[6][0] == 0 or reprojectedPoints[6][1] ==0 or  reprojectedPoints[7][0] == 0 or reprojectedPoints[7][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(reprojectedPoints[6][0]),int(reprojectedPoints[6][1])),(int(reprojectedPoints[7][0]),int(reprojectedPoints[7][1])), (0,165,0),5)
    if reprojectedPoints[1][0] == 0 or reprojectedPoints[1][1] ==0 or  reprojectedPoints[8][0] == 0 or reprojectedPoints[8][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(reprojectedPoints[1][0]),int(reprojectedPoints[1][1])),(int(reprojectedPoints[8][0]),int(reprojectedPoints[8][1])),(0,165,0),5)
    if reprojectedPoints[8][0] == 0 or reprojectedPoints[8][1] ==0 or  reprojectedPoints[9][0] == 0 or reprojectedPoints[9][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(reprojectedPoints[8][0]),int(reprojectedPoints[8][1])),(int(reprojectedPoints[9][0]),int(reprojectedPoints[9][1])), (0,165,0),5)
    if reprojectedPoints[8][0] == 0 or reprojectedPoints[8][1] ==0 or  reprojectedPoints[12][0] == 0 or reprojectedPoints[12][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(reprojectedPoints[8][0]),int(reprojectedPoints[8][1])),(int(reprojectedPoints[12][0]),int(reprojectedPoints[12][1])), (0,165,0),5)
    if reprojectedPoints[9][0] == 0 or reprojectedPoints[9][1] ==0 or  reprojectedPoints[10][0] == 0 or reprojectedPoints[10][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(reprojectedPoints[9][0]),int(reprojectedPoints[9][1])),(int(reprojectedPoints[10][0]),int(reprojectedPoints[10][1])), (0,165,0),5)
    if reprojectedPoints[10][0] == 0 or reprojectedPoints[10][1] ==0 or  reprojectedPoints[11][0] == 0 or reprojectedPoints[11][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(reprojectedPoints[10][0]),int(reprojectedPoints[10][1])),(int(reprojectedPoints[11][0]),int(reprojectedPoints[11][1])), (0,165,0),5)
    if reprojectedPoints[11][0] == 0 or reprojectedPoints[11][1] ==0 or  reprojectedPoints[24][0] == 0 or reprojectedPoints[24][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(reprojectedPoints[11][0]),int(reprojectedPoints[11][1])),(int(reprojectedPoints[24][0]),int(reprojectedPoints[24][1])), (0,165,0),5)
    if reprojectedPoints[11][0] == 0 or reprojectedPoints[11][1] ==0 or  reprojectedPoints[22][0] == 0 or reprojectedPoints[22][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(reprojectedPoints[11][0]),int(reprojectedPoints[11][1])),(int(reprojectedPoints[22][0]),int(reprojectedPoints[22][1])),(0,165,0),5)
    if reprojectedPoints[22][0] == 0 or reprojectedPoints[22][1] ==0 or  reprojectedPoints[23][0] == 0 or reprojectedPoints[23][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(reprojectedPoints[22][0]),int(reprojectedPoints[22][1])),(int(reprojectedPoints[23][0]),int(reprojectedPoints[23][1])), (0,165,0),5)
    if reprojectedPoints[12][0] == 0 or reprojectedPoints[12][1] ==0 or  reprojectedPoints[13][0] == 0 or reprojectedPoints[13][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(reprojectedPoints[12][0]),int(reprojectedPoints[12][1])),(int(reprojectedPoints[13][0]),int(reprojectedPoints[13][1])), (0,165,0),5)
    if reprojectedPoints[13][0] == 0 or reprojectedPoints[13][1] ==0 or  reprojectedPoints[14][0] == 0 or reprojectedPoints[14][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(reprojectedPoints[13][0]),int(reprojectedPoints[13][1])),(int(reprojectedPoints[14][0]),int(reprojectedPoints[14][1])), (0,165,0),5)
    if reprojectedPoints[14][0] == 0 or reprojectedPoints[14][1] ==0 or  reprojectedPoints[21][0] == 0 or reprojectedPoints[21][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(reprojectedPoints[14][0]),int(reprojectedPoints[14][1])),(int(reprojectedPoints[21][0]),int(reprojectedPoints[21][1])), (0,165,0),5)
    if reprojectedPoints[14][0] == 0 or reprojectedPoints[14][1] ==0 or  reprojectedPoints[19][0] == 0 or reprojectedPoints[19][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(reprojectedPoints[14][0]),int(reprojectedPoints[14][1])),(int(reprojectedPoints[19][0]),int(reprojectedPoints[19][1])), (0,165,0),5)
    if reprojectedPoints[19][0] == 0 or reprojectedPoints[19][1] ==0 or  reprojectedPoints[20][0] == 0 or reprojectedPoints[20][1] ==0:
        pass
    else:
        img = cv2.line(img,(int(reprojectedPoints[19][0]),int(reprojectedPoints[19][1])),(int(reprojectedPoints[20][0]),int(reprojectedPoints[20][1])),(0,165,0),5)

    for i in range (0,24):
        if openpose[i][0] == 0 or openpose[i][1] ==0:
            continue
        img = cv2.circle(img, (int(openpose[i][0]),int(openpose[i][1])),10,(0,0,255),-1)
    for i in range (0,24):
        if reprojectedPoints[i][0] == 0 or reprojectedPoints[i][1] ==0:
            continue
        img = cv2.circle(img, (int(reprojectedPoints[i][0]),int(reprojectedPoints[i][1])),10,(0,255,0),-1)

    cv2.imwrite(saveName,img)