import subprocess
import sys
import numpy as np
from ram import *
from kill import *
from robot_path_optimizer import *
import scipy

np.set_printoptions(threshold=sys.maxsize, suppress=True, linewidth= 1000)

project_name = "v8"
batch_size = 25

current_pose_index = 0
robot_poses = generate_robot_poses(radius_count = 2, radius_min = 0.40, radius_max = 0.85, pitch_count = 2, pitch_min_value = -30, pitch_max_value = -75, yaw_count = 2, yaw_min_value = -40.0, yaw_max_value = 40.0)
robot_poses = plan_path_to_minimize_distance_traveled(robot_poses)

number_of_poses = robot_poses.shape[0]
number_of_batches = math.ceil(number_of_poses / batch_size)

approximate_marker_origin = np.asarray([0.28, 0.0, -0.07])

for batch_index in range(number_of_batches):

	min_index = batch_index * batch_size
	if batch_index + batch_size > batch_size:
		max_index = batch_size
	else:
		max_index = batch_index + batch_size
	poses = robot_poses[min_index : max_index]

	print("Planning to scan batches:")
	print(poses)

	# try:
	# 	kill_command(command_substring_to_purge="ray::IDLE")
	# 	kill_command(command_substring_to_purge="scan_allocator.py")
	# except:
	# 	pass

	first_pose_xyz = robot_poses[0, 0:3]
	#print(approximate_marker_origin)

	initial_distance = scipy.spatial.distance.euclidean(first_pose_xyz, approximate_marker_origin)

	poses_joined = "SPACE".join(["{}_{}_{}_{}_{}".format(x,y,z,pitch,yaw) for x,y,z,pitch,yaw in zip(poses[:,0], poses[:, 1], poses[:, 2], poses[:, 3], poses[:, 4])])

	# print("Planning to scan SPACE")
	# print(poses_joined)

	# print("Starting pose index")
	# print(min_index)

	# print("starting_scan_distance")
	# print(initial_distance)

	# print("Project name")
	# print(project_name)

	subprocess.call(["python3", "scan_allocator.py",
					"robot_poses={}".format(poses_joined), 
					"current_pose_index={}".format(min_index), 
					"starting_scan_distance={}".format(initial_distance), 
					"project_name={}".format(project_name) ])
		
	# if min_index % 100 == 0 and min_index != 0:
	# 	subprocess.call(["python3", "recalibrate_robot_position.py"])