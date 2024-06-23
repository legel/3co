from geometry import PointCloud

project_tag = "forest_crystal"
number_of_projects = 5

clouds = []
projects = []
commands = [ {"x": 1.40, "y": 0.1, "z": 1.0, "pitch": -90.0, "yaw": -90.0}, 
			 {"x": 1.45, "y": 0.1, "z": 1.0, "pitch": -90.0, "yaw": -90.0}, 
			 {"x": 1.50, "y": 0.1, "z": 1.0, "pitch": -90.0, "yaw": -90.0},
			 {"x": 1.55, "y": 0.1, "z": 1.0, "pitch": -90.0, "yaw": -90.0},
			 {"x": 1.60, "y": 0.1, "z": 1.0, "pitch": -90.0, "yaw": -90.0}]

for project_number in range(1, number_of_projects+1):
	project_name = "{}_{}".format(project_tag,project_number)
	projects.append(project_name)



	# This loading is not in a great situation, partially because we're just stacking complexity with the saved HDR multi-scan data structure (not to mention how big that is)
	# At this point, taking the step of combining the individual points for each HDR scan is starting to make sense.
	# This can technically serve as a measure of outlier control, and incentivizes the collection of several scans.
	# This may also be a good time to implement a quick auto focus (x1-4), auto exposure (x1-4)
	# ...
	# Auto commander.move(x,y,z,pitch,yaw) is clearly within grasp - minus obstacles...
	#   Suppose, first of all z = 0 is redefined based on wherever platform is, and it is illegal to move there
	#   Suppose, we set an optimum Gaussian distribution for the desired distance and angle between N percent of points seen in current view, and 100 - N percent of points that will be new in held-out view
	#   Using the unit vector transforms, we then minimize the deviation for (1) distance to all N points and (2) angle to all N points vs. ideal, while also minimizing the current robot pose distance from current pose, and forcing all N points to be contiguous and edgewise with a single edge vector
	# 	(System can use previously estimated focus distances as priors for new probing)

	cloud = PointCloud(filename="{}-0.npy".format(project_name), project_name=project_name, ignore_dimensions=True, reindex_into_tensor=True, clip_outer_n_pixels=100) #  reindex_into_tensor=True, 
	clouds.append(cloud)

source_cloud = clouds[0]

for a in range(number_of_projects-1):
	b = a + 1
	target_cloud = clouds[b]
	command_a = commands[a]
	command_b = commands[b]
	print("Command A: {}, Command B: {}".format(command_a, command_b))
	source_cloud.transform_by_robot_commands(start_command=command_a, end_command=command_b)
	source_cloud.add_point_cloud(target_cloud)
	source_cloud.reindex_into_tensor() # first of all, because every "scan" is actually already a conglomerate, we either need to pre-combine points before this, or don't reindex...
	break

#source_cloud.reindex_into_tensor()
source_cloud.save_as_ply("{}_transformed_initial_estimates.ply".format(project_tag), from_tensor=True)


# Implement Open3D ICP for final scan alignment
# Save a color map which is separate from geometry map:
#	Color normalization by distance from projector is accounted for
#	Only colors from +- 10 degrees normal are used 