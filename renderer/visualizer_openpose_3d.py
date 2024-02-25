import open3d as o3d 
import numpy as np 

dlt_keypoint = np.load("output_3d/nl_keypoints.npy")

line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(dlt_keypoint[0])
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([line_set])

vis = o3d.visualization.Visualizer()
vis.create_window()
for j in range (0,851):
    line_set = o3d.geometry.LineSet()   
    line_set.points = o3d.utility.Vector3dVector(dlt_keypoint[j])
    vis.add_geometry(line_set)
    ctr = vis.get_view_control()
    ctr.change_field_of_view(60.0)
    ctr.set_front(  [ 0.60128001750265858, -0.38316853081412899, 0.70117345753083415 ])
    ctr.set_up([ 0.072805529159330667, -0.84759919121340532, -0.52561865071381453 ])
    ctr.set_zoom(2.0)
    ctr.set_lookat([ 581.09208917074909, -370.90876972586221, 2657.3605125068325 ])
    vis.update_geometry(line_set)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image("3dkeypoint_viz/4/" + str(j) + "_output.jpg")
    vis.remove_geometry(line_set)


vis.destroy_window()



# {
# 	"class_name" : "ViewTrajectory",
# 	"interval" : 29,
# 	"is_loop" : false,
# 	"trajectory" : 
# 	[
# 		{
# 			"boundingbox_max" : [ 610.12040994403708, 406.96750337211319, 3192.6585567195716 ],
# 			"boundingbox_min" : [ 37.608649435065736, -1209.1806232983683, 2222.0660315545933 ],
# 			"field_of_view" : 60.0,
# 			"front" : [ 0.60128001750265858, -0.38316853081412899, 0.70117345753083415 ],
# 			"lookat" : [ 581.09208917074909, -370.90876972586221, 2657.3605125068325 ],
# 			"up" : [ 0.072805529159330667, -0.84759919121340532, -0.52561865071381453 ],
# 			"zoom" : 2.0
# 		}
# 	],
# 	"version_major" : 1,
# 	"version_minor" : 0
# }

# {
# 	"class_name" : "ViewTrajectory",
# 	"interval" : 29,
# 	"is_loop" : false,
# 	"trajectory" : 
# 	[
# 		{
# 			"boundingbox_max" : [ 610.12040994403708, 406.96750337211319, 3192.6585567195716 ],
# 			"boundingbox_min" : [ 37.608649435065736, -1209.1806232983683, 2222.0660315545933 ],
# 			"field_of_view" : 60.0,
# 			"front" : [ -0.921633374745366, -0.34427386941773164, 0.17907379873004997 ],
# 			"lookat" : [ 581.09208917074909, -370.90876972586221, 2657.3605125068325 ],
# 			"up" : [ 0.22129159103425, -0.84533125545739984, -0.486256208478973 ],
# 			"zoom" : 2.0
# 		}
# 	],
# 	"version_major" : 1,
# 	"version_minor" : 0
# }

# {
# 	"class_name" : "ViewTrajectory",
# 	"interval" : 29,
# 	"is_loop" : false,
# 	"trajectory" : 
# 	[
# 		{
# 			"boundingbox_max" : [ 610.12040994403708, 406.96750337211319, 3192.6585567195716 ],
# 			"boundingbox_min" : [ 37.608649435065736, -1209.1806232983683, 2222.0660315545933 ],
# 			"field_of_view" : 60.0,
# 			"front" : [ 0.19536752216433198, -0.47264769902124026, 0.85932280540743033 ],
# 			"lookat" : [ 581.09208917074909, -370.90876972586221, 2657.3605125068325 ],
# 			"up" : [ 0.20241367107892361, -0.83790958072961408, -0.5068887849241448 ],
# 			"zoom" : 2.0
# 		}
# 	],
# 	"version_major" : 1,
# 	"version_minor" : 0
# }