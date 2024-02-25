import open3d as o3d 
import numpy as np 

keypoint = np.load("output_3d/nl_keypoints.npy")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(keypoint)
o3d.io.write_point_cloud('output_3d/nl_keypoints.ply', pcd)

vis = o3d.visualization.Visualizer()
vis.create_window()

# geometry is the point cloud used in your animaiton
geometry = o3d.geometry.PointCloud()
vis.add_geometry(geometry)

for i in range(keypoint.shape[0]):
    # now modify the points of your geometry
    # you can use whatever method suits you best, this is just an example
    geometry.points = keypoint[i]
    vis.update_geometry(geometry)
    vis.poll_events()
    vis.update_renderer()
    