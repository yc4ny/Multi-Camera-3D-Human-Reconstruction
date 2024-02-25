import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from camera_pose_visualizer import CameraPoseVisualizer

if __name__ == '__main__':
    # Create Visualizer 
    #   Input:  Range of x,y,z 
    visualizer = CameraPoseVisualizer([-4000, 4000], [-4000, 4000], [-4000, 4000])

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

    # Extrinsic = [R|t]
    C1_Extrinsic = np.column_stack((C1_Rotation,C1_Translation)) 
    C2_Extrinsic = np.column_stack((C2_Rotation,C2_Translation)) 

    # Visualize in 3D Space 
    #   Input:  Extrinsic Matrix (4,4) 
    #           Color: color or list of rgba tuples
    #           Scaled Focal Length: z-axis length of frame body of camera 
    visualizer.extrinsic2pyramid(C1_Extrinsic, 'r', 10)
    visualizer.extrinsic2pyramid(C2_Extrinsic, 'g', 10)

    # Show visualizer 
    visualizer.show()