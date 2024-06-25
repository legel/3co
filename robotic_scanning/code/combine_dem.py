from robotics import Iris
robot = Iris()
point_cloud_names = ["auto-0-0.ply", "auto-1-0.ply"]
focus_stacked_cloud = robot.combine_point_clouds(point_cloud_names, from_object=False)
focus_stacked_cloud.average_xyz_positions_with_outlier_removal(new_scan_index=0)
focus_stacked_cloud.export_to_npy(project_name="combineddd", from_tensor=True)
focus_stacked_cloud.save_as_ply(filename="stacked.ply", from_tensor=True)
