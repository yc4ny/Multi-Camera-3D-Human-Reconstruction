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
        [0., 0., 1.],
        [0., 0., 0.]
        ])

    C1_Translation = np.array([
        [0], 
        [0], 
        [0],
        [1]
        ])
    
    C2_Rotation = np.array([
        [0.300148282316866, -0.178854539023674, 0.936974952969856],
        [0.213312172002903, 0.969974532135580, 0.116821762885889],
        [-0.929735944178582, 0.164804310862890, 0.329288039903318],
        [0., 0., 0.]
        ])

    C2_Translation = np.array([
        [-1553.27690612998], 
        [-99.8704006528866], 
        [1125.21211587601],
        [1]
        ])

    C3_Rotation = np.array([
        [-0.753112996400152, -0.117579644646254, 0.647298881366286],
        [0.320722899793417, 0.793447914809945, 0.517278675407997],
        [-0.574419390516447, 0.597172867475249, -0.559845452022344],
        [0., 0., 0.]
        ])

    C3_Translation = np.array([
        [-1135.29518429509], 
        [-902.726453136737], 
        [2343.99645746765],
        [1]
        ])

    C4_Rotation = np.array([
        [-0.569257197341602, 0.278175027692455, -0.773669759809110],
        [-0.374093119078187, 0.750327055646885, 0.545035455564709],
        [0.732120605865685, 0.599689889470031, -0.323065713027990],
        [0., 0., 0.]
        ])

    C4_Translation = np.array([
        [1356.34864363282], 
        [-1165.67330423141], 
        [1960.83185883102],
        [1]
        ])

    C5_Rotation = np.array([
        [0.516834948460620, 0.233842426156420, -0.823528600462053],
        [-0.273558397873549, 0.956647137631025, 0.0999602771866104],
        [0.811201232125892, 0.173620199837594, 0.558398233526744],
        [0., 0., 0.]
        ])

    C5_Translation = np.array([
        [1548.24347614776], 
        [-362.806008137686], 
        [816.572353445145],
        [1]
        ])

    # Extrinsic = [R|t]
    C1_Extrinsic = np.column_stack((C1_Rotation,C1_Translation)) 
    C2_Extrinsic = np.column_stack((C2_Rotation,C2_Translation)) 
    C3_Extrinsic = np.column_stack((C3_Rotation,C3_Translation)) 
    C4_Extrinsic = np.column_stack((C4_Rotation,C4_Translation)) 
    C5_Extrinsic = np.column_stack((C5_Rotation,C5_Translation)) 

    # Visualize in 3D Space 
    #   Input:  Extrinsic Matrix (4,4) 
    #           Color: color or list of rgba tuples
    #           Scaled Focal Length: z-axis length of frame body of camera 
    visualizer.extrinsic2pyramid(C1_Extrinsic, 'r', 600)
    visualizer.extrinsic2pyramid(C2_Extrinsic, 'g', 600)
    visualizer.extrinsic2pyramid(C3_Extrinsic, 'c', 600)
    visualizer.extrinsic2pyramid(C4_Extrinsic, 'y', 600)
    visualizer.extrinsic2pyramid(C5_Extrinsic, 'b', 600)


    # Show visualizer 
    visualizer.show()