import numpy as np
import utils.triangulation as stereo
import os
import visualizer.openpose_visualizer as viz
import utils.concat_openpose_imgs as concat_openpose_imgs 

if __name__ == '__main__':
    
    #Initialize Camera Calibration Parameters 
    C1_Intrinsic = np.array([
        [1793.40866450411, 0, 1941.82280218514], 
        [0, 1784.34883485341, 1079.49235904473],
        [0.,             0.,                1.]
        ])

    C1_radialDistortion = np.array([
        [-0.247779937548622],
        [0.0637764128399176]
    ])

    C2_Intrinsic = np.array([
        [1795.70365788380, 0, 1877.62446127849],
        [0, 1789.66009806022, 1118.82747427708],
        [0.,              0.,               1.]
        ])
    
    C2_radialDistortion = np.array([
        [-0.239496853407498],
        [0.0546352045342115]
    ])

    C1_Rotation = np.array([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
        ])

    C2_Rotation = np.array([
    [0.911619921535167,-0.124455570407734,0.391739619715318],
    [0.0735810919149552,0.987075830182677,0.142362664985526 ],
    [-0.404394537021483,-0.100956012518640, 0.908995567625892]
        ])

    C1_Translation = np.array([
        [0], 
        [0], 
        [0]
        ])

    C2_Translation = np.array([
        [-1520.23416825357],
        [-191.344404566789],
        [331.725213364265]
        ])
    
    # Projection Matrix = K[R|t]
    C1_ProjectionMatrix = np.matmul(C1_Intrinsic, np.column_stack((C1_Rotation,C1_Translation)))
    C2_ProjectionMatrix = np.matmul(C2_Intrinsic, np.column_stack((C2_Rotation,C2_Translation)))

    # Count number of frames to process 
    dir_path = r'C1/C1_keypoints'
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

    # In range of every frame,
    # Example of path to keypoints: C1/C1/keypoints/1_keypoints.json
    for i in range (1,count): 

        #Int variable to keep track of reprojection error 
        error = 0

        #C1_op : Camera 1 Openpose Keypoints (count,25,3) - BODY_25 Format
        #C2_op : Camera 2 
        C1_op = viz.loadOpenPose('C1/C1_keypoints/'+str(i)+'_keypoints.json')
        C2_op = viz.loadOpenPose('C2/C2_keypoints/'+str(i)+'_keypoints.json')
        #Slice to obtain just the keypoints (x,y) also known as removing the confidence z values 
        C1_op = C1_op[:,:2]
        C2_op = C2_op[:,:2]

        # Perform Triangulation Linearly using Direct Linear Transform (DLT)
        #       Input:    Camera Parameters, Openpose 2D Joints 
        #       Output:   Homogeneous 3D Keypoints (x,y,z,w)
        keypoint3d = stereo.LinearTriangulation(C1_Intrinsic,C2_Intrinsic, C1_Translation, C1_Rotation, C2_Translation, C2_Rotation, C1_op, C2_op)
        # Homogeneous -> Cartesian (x/w,y/w,z/w,1)
        keypoint3d = stereo.homogeneous_cartesian(keypoint3d)
        keypoint3d = keypoint3d[:,:3]
        keypoint_results.append(keypoint3d)

        # Compare Openpose Joints vs Triangulated Joints and save comparison images
        viz.visualizeReprojectionError(C1_op, stereo.reprojectedPoints(C1_ProjectionMatrix, keypoint3d),'C1/C1_undistorted/'+str(i)+'.jpg','C1/C1_dlt/'+str(i)+'_reproject.jpg')
        viz.visualizeReprojectionError(C2_op, stereo.reprojectedPoints(C2_ProjectionMatrix, keypoint3d),'C2/C2_undistorted/'+str(i)+'.jpg','C2/C2_dlt/'+str(i)+'_reproject.jpg')
        
        # Calculate Reproejction Error (L2 Distance between openpose and triangulated joints) for C1,C2 then compute the average.
        error = ( stereo.reprojectionError(C1_op, C1_ProjectionMatrix, keypoint3d) + stereo.reprojectionError(C2_op,C2_ProjectionMatrix, keypoint3d) ) /2
        # Make sure to save the error for later use 
        error_track.append(error)
        # Set a variable to calculate the total average reprojection error in the future. 
        total_error += error
        print('Frame: ' + str(i))
        print('Error: ' + str(error))
    print("Dlt Total Error: " + str(total_error/count))
    np.save('keypoints_3d/dlt.npy', keypoint_results, True, True)
    np.save('concat/dlt_error.npy',error_track, True, True)


    # Triangulation with non-linear optimization
    keypoint_results = []
    total_error = 0
    error_track = []

    for i in range (1,count): 

        error = 0

        C1_op = viz.loadOpenPose('C1/C1_keypoints/'+str(i)+'_keypoints.json')
        C2_op = viz.loadOpenPose('C2/C2_keypoints/'+str(i)+'_keypoints.json')
        C1_op = C1_op[:,:2]
        C2_op = C2_op[:,:2]

        #Find initial 3d keypoints
        keypoint3d = stereo.LinearTriangulation(C1_Intrinsic,C2_Intrinsic, C1_Translation, C1_Rotation, C2_Translation, C2_Rotation, C1_op, C2_op)
        keypoint3d = stereo.homogeneous_cartesian(keypoint3d)
        keypoint3d = keypoint3d[:,:3]

        #Non linear optimization
        nonlinear3d = stereo.Triangulation_nl(keypoint3d, C1_ProjectionMatrix,C2_ProjectionMatrix,C1_op,C2_op)
        keypoint_results.append(nonlinear3d)

        viz.visualizeReprojectionError(C1_op, stereo.reprojectedPoints(C1_ProjectionMatrix, nonlinear3d),'C1/C1_undistorted/'+str(i)+'.jpg','C1/C1_nonlinear/'+str(i)+'_reproject.jpg')
        viz.visualizeReprojectionError(C2_op, stereo.reprojectedPoints(C2_ProjectionMatrix, nonlinear3d),'C2/C2_undistorted/'+str(i)+'.jpg','C2/C2_nonlinear/'+str(i)+'_reproject.jpg')
        error = ( stereo.reprojectionError(C1_op, C1_ProjectionMatrix, nonlinear3d) + stereo.reprojectionError(C1_op, C1_ProjectionMatrix, nonlinear3d) ) /2
        error_track.append(error)
        total_error += error

        print('Frame: ' + str(i))
        print('Error: ' + str(error))       
    print("Non Linear Optimization Total Error: " + str(total_error/count))
    np.save('keypoints_3d/nl.npy', keypoint_results, True, True)
    np.save('concat/nl_error.npy',error_track, True, True)

    # Concat Images 
    concat_openpose_imgs.concat()
       









    

