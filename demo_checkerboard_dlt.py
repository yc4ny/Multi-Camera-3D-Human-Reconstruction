import numpy as np
import utils.triangulation as stereo 
import utils.bundle_adjustment as bundle
import os
import visualizer.vis_openpose as viz
import scipy.io 
import cv2
import time
from tqdm import tqdm


if __name__ == '__main__':
    #Initialize Camera Calibration Parameters 
    Intrinsic = np.array([
        [1769.60561310104, 0, 1927.08704019384], 
        [0,1763.89532833387, 1064.40054933721],
        [0.,             0.,                1.]
        ])

    radialDistortion = np.array([
        [-0.244052127306437],
        [0.0597008096110524]
    ])

    C1_Rotation = np.array([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
        ])

    C1_Translation = np.array([
        [0], 
        [0], 
        [0]
        ])
    
    C2_Rotation = np.array([
        [0.300148282316866, -0.178854539023674, 0.936974952969856],
        [0.213312172002903, 0.969974532135580, 0.116821762885889],
        [-0.929735944178582, 0.164804310862890, 0.329288039903318]
        ])

    C2_Translation = np.array([
        [-1553.27690612998], 
        [-99.8704006528866], 
        [1125.21211587601]
        ])

    C3_Rotation = np.array([
        [-0.753112996400152, -0.117579644646254, 0.647298881366286],
        [0.320722899793417, 0.793447914809945, 0.517278675407997],
        [-0.574419390516447, 0.597172867475249, -0.559845452022344]
        ])

    C3_Translation = np.array([
        [-1135.29518429509], 
        [-902.726453136737], 
        [2343.99645746765]
        ])

    C4_Rotation = np.array([
        [-0.569257197341602, 0.278175027692455, -0.773669759809110],
        [-0.374093119078187, 0.750327055646885, 0.545035455564709],
        [0.732120605865685, 0.599689889470031, -0.323065713027990]
        ])

    C4_Translation = np.array([
        [1356.34864363282], 
        [-1165.67330423141], 
        [1960.83185883102]
        ])

    C5_Rotation = np.array([
        [0.516834948460620, 0.233842426156420, -0.823528600462053],
        [-0.273558397873549, 0.956647137631025, 0.0999602771866104],
        [0.811201232125892, 0.173620199837594, 0.558398233526744]
        ])

    C5_Translation = np.array([
        [1548.24347614776], 
        [-362.806008137686], 
        [816.572353445145]
        ])

    
    # Projection Matrix = K[R|t]
    C1_ProjectionMatrix = np.matmul(Intrinsic, np.column_stack((C1_Rotation,C1_Translation)))
    C2_ProjectionMatrix = np.matmul(Intrinsic, np.column_stack((C2_Rotation,C2_Translation)))
    C3_ProjectionMatrix = np.matmul(Intrinsic, np.column_stack((C3_Rotation,C3_Translation)))
    C4_ProjectionMatrix = np.matmul(Intrinsic, np.column_stack((C4_Rotation,C4_Translation)))
    C5_ProjectionMatrix = np.matmul(Intrinsic, np.column_stack((C5_Rotation,C5_Translation)))
    

    projectionMatrix = ["",C1_ProjectionMatrix, C2_ProjectionMatrix,C3_ProjectionMatrix,C4_ProjectionMatrix,C5_ProjectionMatrix ]
    rotation = ["",C1_Rotation,C2_Rotation,C3_Rotation,C4_Rotation,C5_Rotation, ]
    translation = ["",C1_Translation,C2_Translation,C3_Translation,C4_Translation,C5_Translation ]
    
    # Count number of frames to process 
    dir_path = r'input_data/checkerboard/C1_drawCheckerboard'
    count = 0
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1

    # Triangulation using DLT
    # Array to store triangulated 3d keypoints 
    keypoint_results = []
    # Temperary variable to keep track of reprojection error 
    total_error = 0;
    # List of reprojection error each frame, used for putting reprojection error on image using cv2.putText 
    error_track = []

    C1_on = np.load('input_data/checkerboard_on/C1.npy')
    C2_on = np.load('input_data/checkerboard_on/C2.npy')
    C3_on = np.load('input_data/checkerboard_on/C3.npy')
    C4_on = np.load('input_data/checkerboard_on/C4.npy')
    C5_on = np.load('input_data/checkerboard_on/C5.npy')

    valid = 0 
    for i in tqdm(range(1,count)):
        #Int variable to keep track of reprojection error 
        error = 0

        C1_key = scipy.io.loadmat("input_data/checkerboard/C1_keypoints/" + str(i) + ".mat")
        C1_key = C1_key['imagePoints']
        C2_key = scipy.io.loadmat("input_data/checkerboard/C2_keypoints/" + str(i) + ".mat")
        C2_key = C2_key['imagePoints']
        C3_key = scipy.io.loadmat("input_data/checkerboard/C3_keypoints/" + str(i) + ".mat")
        C3_key = C3_key['imagePoints']
        C4_key = scipy.io.loadmat("input_data/checkerboard/C4_keypoints/" + str(i) + ".mat")
        C4_key = C4_key['imagePoints']
        C5_key = scipy.io.loadmat("input_data/checkerboard/C5_keypoints/" + str(i) + ".mat")
        C5_key = C5_key['imagePoints']

        key_2d = [" ",C1_key, C2_key, C3_key, C4_key, C5_key]


        img_1 = cv2.imread("input_data/checkerboard/C1_drawCheckerboard/" +str(i) + ".jpg")
        img_2 = cv2.imread("input_data/checkerboard/C2_drawCheckerboard/"+ str(i) + ".jpg")
        img_3 = cv2.imread("input_data/checkerboard/C3_drawCheckerboard/"+ str(i) + ".jpg")
        img_4 = cv2.imread("input_data/checkerboard/C4_drawCheckerboard/"+ str(i) + ".jpg")
        img_5 = cv2.imread("input_data/checkerboard/C5_drawCheckerboard/"+ str(i) + ".jpg")

        index = []

        #Check which cameras are on for finding usable points for triangulation 
        for k in range (1,6):
            on = np.load('input_data/checkerboard_on/C' + str(k) +'.npy')
            if on[i][0] == 1: 
                index.append(k)
        
        if len(index) < 2: 
            print("Frame " + str(i))
            print("Detected from only 1 camera!")
            cv2.imwrite("input_data/checkerboard/dlt_C1/" +str(i) + ".jpg", img_1)
            cv2.imwrite("input_data/checkerboard/dlt_C2/" +str(i) + ".jpg", img_2)
            cv2.imwrite("input_data/checkerboard/dlt_C3/" +str(i) + ".jpg", img_3)    
            cv2.imwrite("input_data/checkerboard/dlt_C4/" +str(i) + ".jpg", img_4)
            cv2.imwrite("input_data/checkerboard/dlt_C5/" +str(i) + ".jpg", img_5)
            error_track.append(0)
            continue

        #Non Linear Optimization 
        keypoint3d = stereo.LinearTriangulation(Intrinsic,Intrinsic, translation[index[0]], rotation[index[0]], translation[index[1]], rotation[index[1]], key_2d[index[0]], key_2d[index[1]])
        keypoint3d = stereo.homogeneous_cartesian(keypoint3d)
        keypoint3d_nl= keypoint3d[:,:3]
        # keypoint3d_nl = stereo.Triangulation_nl(keypoint3d, Intrinsic, rotation[index[0]],translation[index[0]],rotation[index[1]],translation[index[1]],key_2d[index[0]], key_2d[index[1]])
        keypoint_results.append(keypoint3d_nl)

        # #Bundle Adjustment
        # track = []
        # track = np.append(track,key_2d[index[0]])
        # track = np.append(track,key_2d[index[1]])
        # track = np.reshape(track, (2,54,2))
        # P = []
        # P = np.append(P, np.column_stack((rotation[index[0]],translation[index[0]])))
        # P = np.append(P, np.column_stack((rotation[index[1]],translation[index[1]])))
        # P = np.reshape(P, (2,3,4))

        # P_new, X_new = bundle.RunBundleAdjustment(P,keypoint3d,track)

        C1_3d = stereo.reprojectedPoints(C1_ProjectionMatrix, keypoint3d_nl)
        C2_3d = stereo.reprojectedPoints(C2_ProjectionMatrix, keypoint3d_nl)
        C3_3d = stereo.reprojectedPoints(C3_ProjectionMatrix, keypoint3d_nl)
        C4_3d = stereo.reprojectedPoints(C4_ProjectionMatrix, keypoint3d_nl)
        C5_3d = stereo.reprojectedPoints(C5_ProjectionMatrix, keypoint3d_nl)

        
        for j in range (C1_3d.shape[0]):
            check = cv2.circle(img_1, (int(C1_3d[j][0]), int(C1_3d[j][1])), 12, (0,0,255),-1)
            cv2.imwrite("input_data/checkerboard/dlt_C1/" +str(i) + ".jpg", check)
        
        for j in range (C2_3d.shape[0]):
            check2 = cv2.circle(img_2, (int(C2_3d[j][0]), int(C2_3d[j][1])), 12, (0,0,255),-1)
            cv2.imwrite("input_data/checkerboard/dlt_C2/"+str(i) + ".jpg", check2)

        for j in range (C3_3d.shape[0]):
            check2 = cv2.circle(img_3, (int(C3_3d[j][0]), int(C3_3d[j][1])), 12, (0,0,255),-1)
            cv2.imwrite("input_data/checkerboard/dlt_C3/"+str(i) + ".jpg", check2)

        for j in range (C4_3d.shape[0]):
            check2 = cv2.circle(img_4, (int(C4_3d[j][0]), int(C4_3d[j][1])), 12, (0,0,255),-1)
            cv2.imwrite("input_data/checkerboard/dlt_C4/"+str(i) + ".jpg", check2)

        for j in range (C5_3d.shape[0]):
            check2 = cv2.circle(img_5, (int(C5_3d[j][0]), int(C5_3d[j][1])), 12, (0,0,255),-1)
            cv2.imwrite("input_data/checkerboard/dlt_C5/"+str(i) + ".jpg", check2)

        notEmpty = 0

        if C1_key.shape[0] ==54:
            error = stereo.reprojectionError(C1_key, C1_ProjectionMatrix, keypoint3d_nl)
            notEmpty = notEmpty + 1 
        
        if C2_key.shape[0] ==54:
            error = stereo.reprojectionError(C2_key, C2_ProjectionMatrix, keypoint3d_nl)
            notEmpty = notEmpty + 1 

        if C3_key.shape[0] == 54:
            error = stereo.reprojectionError(C3_key, C3_ProjectionMatrix, keypoint3d_nl)
            notEmpty = notEmpty + 1 

        if C4_key.shape[0] == 54:
            error = stereo.reprojectionError(C4_key, C4_ProjectionMatrix, keypoint3d_nl)
            notEmpty = notEmpty + 1 

        if C5_key.shape[0] == 54:
            error = stereo.reprojectionError(C5_key, C5_ProjectionMatrix, keypoint3d_nl)
            notEmpty = notEmpty + 1 

        valid = valid+1
        averageError = error / notEmpty
        error_track.append(averageError)
        # Set a variable to calculate the total average reprojection error in the future. 
        total_error += averageError
        print('Frame ' + str(i))
        print('Error: ' + str(averageError))

    print("Total Average Error: " + str(total_error/valid))
    np.save('output_3d/dlt_keypoints.npy', keypoint_results, True, True)
    np.save('output_3d/dlt_error.npy',error_track, True, True)

        

        

    















    

