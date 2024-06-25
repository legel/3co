import subprocess
import sys
import numpy as np
from ram import *
from kill import *
from commander import *
from robotics import *
from geometry import *
from pympler import tracker, asizeof
from operator import itemgetter
import numpy as np
import time
import sys
import os
import psutil
import scipy

radius_count = 2
radius_min = 0.30
radius_max = 1.25
pitch_count = 2
pitch_min_value = -15
pitch_max_value = -85
yaw_count = 2
yaw_min_value = -30.0
yaw_max_value = 30.0

robot_pitch_standard_deviation = 0.5
robot_yaw_standard_deviation = 0.5
robot_x_standard_deviation = 0.01
robot_y_standard_deviation = 0.01
robot_z_standard_deviation = 0.01
number_of_dimensions_per_pose = 5

scans_per_perspective = 1
min_exposure_time = 1.0
max_exposure_time = 2.0
exposure_time_count = 2
exposure_time_multipliers = np.linspace(min_exposure_time, max_exposure_time, exposure_time_count, endpoint=True)

total_scans_to_take = radius_count * pitch_count * yaw_count * scans_per_perspective * exposure_time_count
total_poses_to_see = radius_count * pitch_count * yaw_count

np.set_printoptions(threshold=sys.maxsize, suppress=True, linewidth= 1000)

def save_positions_with_corresponding_scan_data_to_log(robot_poses, scan_data):
	try:
		print("Saving output of scan {} to file {}".format(scan_index, "{}.tsv".format(project_name)), flush=True)
		with open("{}.tsv".format(project_name), "a") as outputs:
			#outputs.write("ROBOT X\tROBOT Y\tROBOT Z\tROBOT PITCH\tROBOT YAW\tPX1\tPY1\tPZ1\tPX2\tPY2\tPZ2\tPX3\tPY3\tPZ3\tPX4\tPY4\tPZ4\n")
			if robot_scans_valid[scan_index]:
				pose = robot_poses[scan_index, :]
				block = calibration_blocks[scan_index, :, :]
				robot_x = pose[0]
				robot_y = pose[1]
				robot_z = pose[2]
				robot_pitch = pose[3]
				robot_yaw = pose[4]
				outputs.write("{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(robot_x, robot_y, robot_z, robot_pitch, robot_yaw))
				for corner_point in range(number_of_corner_points_to_measure):
					point = block[corner_point,:]
					point_x = point[0]
					point_y = point[1]
					point_z = point[2]
					outputs.write("\t{:.3f}\t{:.3f}\t{:.3f}".format(point_x, point_y, point_z))
				outputs.write("\n")
	except Exception as error_message:
		print(error_message)
		print("\nFailure to save output, moving on...\n")

def safe_division(n, d, default):
    return n / d if d else default

def generate_robot_poses():
	robot_poses = np.zeros((total_poses_to_see, number_of_dimensions_per_pose))

	iterations = 0

	radii = np.linspace(radius_min, radius_max, radius_count, endpoint=True)
	pitchs = np.linspace(pitch_min_value, pitch_max_value, pitch_count, endpoint=True)
	yaws = np.linspace(yaw_min_value, yaw_max_value, yaw_count, endpoint=True)

	radius_range = np.linspace(radius_min, radius_max, radius_count, endpoint=True)

	for sweep_index, radius in enumerate(radius_range):
		if sweep_index % 2 == 0.0:
			vertical_motion_direction = 1
		else:
			vertical_motion_direction = -1

		if vertical_motion_direction:
			pitch_min = pitch_min_value
			pitch_max = pitch_max_value
		else:
			pitch_min = pitch_max_value
			pitch_max = pitch_min_value
		pitch_range = np.linspace(pitch_min, pitch_max, pitch_count, endpoint=True)

		for pitch_index, pitch in enumerate(pitch_range):
			z = 0.084 + abs(math.sin(math.radians(pitch)) * radius)
			z_offset_for_new_charuco_position_relative_to_block = -0.1
			z += z_offset_for_new_charuco_position_relative_to_block
			pitch_stretcher = 0 #pitch_min_value #1.75*pitch_min_value 
			xy_radius = math.cos(math.radians(pitch - pitch_stretcher)) * radius 

			if pitch_index % 2 == 0.0:
				horizontal_motion_direction = 1
			else:
				horizontal_motion_direction = -1

			yaw_min = yaw_min_value * horizontal_motion_direction
			yaw_max = yaw_max_value * horizontal_motion_direction 
			
			yaw_range = np.linspace(yaw_min, yaw_max, yaw_count, endpoint=True)

			for yaw in yaw_range:
				x = xy_radius
				y = -0.125

				sign = safe_division(yaw, abs(yaw), default=1)
				xy_hypotenuse = 2 * xy_radius * math.sin(0.5*math.radians(yaw))
				angle_b = math.asin(safe_division(xy_radius * math.sin(math.radians(yaw)), xy_hypotenuse, default=1))
				angle_b_inverse = math.radians(90) - angle_b
				delta_x = -1 * abs((xy_hypotenuse * math.sin(angle_b_inverse)))
				delta_y = -1 * (xy_hypotenuse * math.sin(angle_b))

				x = x + delta_x
				y = y + delta_y

				x_offset_for_new_charuco_position_relative_to_block = 0.15
				x += x_offset_for_new_charuco_position_relative_to_block

				robot_poses[iterations, 0] = round(x,6)
				robot_poses[iterations, 1] = round(y,6)
				robot_poses[iterations, 2] = round(z,6)
				robot_poses[iterations, 3] = round(pitch,3)
				robot_poses[iterations, 4] = round(yaw,3)

				iterations += 1

				#print("({}) r={:.3f}m, r_xy={:.3f}m, pitch={:.3f}°, yaw {:.3f}°, x={:.3f}m, y={:.3f}m, z={:.3f}m".format(iterations, radius, xy_radius, pitch, yaw, x, y, z))

	return robot_poses


def plan_path_to_minimize_distance_traveled():
	robot_poses = generate_robot_poses()

	start_time = time.time()

	number_of_poses_to_sort = robot_poses.shape[0]

	# initialize history of distance traveled, and new sequence of robot paths
	total_distance_to_travel = np.zeros((number_of_poses_to_sort))

	# compute matrix of pairwise 3D distances between all points 
	xyz_1 = robot_poses[:,0:3].copy()
	xyz_2 = robot_poses[:,0:3].copy()
	original_pairwise_distances = scipy.spatial.distance.cdist(xyz_1, xyz_2, 'euclidean')

	sum_of_distances = np.zeros((number_of_poses_to_sort))

	best_path = robot_poses
	best_path_distance = 100000.0

	for next_closest_pose_index_start in range(number_of_poses_to_sort):
		next_closest_pose_index = next_closest_pose_index_start
		distance_minimized_path_of_robot_poses = np.zeros((number_of_poses_to_sort, number_of_dimensions_per_pose))
		distance_minimized_path_of_robot_poses_indices = np.zeros((number_of_poses_to_sort))

		pairwise_distances = original_pairwise_distances.copy()
		remaining_pose_indices = np.asarray([i for i in range(number_of_poses_to_sort)])
		distance_minimized_path_of_robot_poses[0,:] = robot_poses[next_closest_pose_index, :]


		# # delete this selected pose from options to select from in the future 
		# remaining_pose_indices = np.delete(remaining_pose_indices, next_closest_pose_index)
		# pairwise_distances = np.delete(pairwise_distances, next_closest_pose_index, axis=1)

		#print("Initial index to start from: {}".format(next_closest_pose_index))

		#print("Initial remaining pose indices")
		#print(remaining_pose_indices)

		#print("Initial pairwise:")
		#print(pairwise_distances)

		for next_pose in range(1, number_of_poses_to_sort):
			# select row from matrix with distances for this position
			distances_from_this_pose = pairwise_distances[next_closest_pose_index, :]

			# find minimum distance index
			indices_of_non_zero_distances = np.nonzero(distances_from_this_pose)[0]

			#print("Non-zero indices: {}".format(indices_of_non_zero_distances))

			nonzero_distances = distances_from_this_pose[indices_of_non_zero_distances]
			#print("Non-zero distances: {}".format(nonzero_distances))

			minimum_nonzero_distance = np.min(nonzero_distances)
			#print("Minimum non-zero distance: {}".format(minimum_nonzero_distance))

			minimum_nonzero_distance_index = np.where(distances_from_this_pose == minimum_nonzero_distance)[0][0]
			#print("Minimum non-zero distance index: {}".format(minimum_nonzero_distance_index))

			# convert index from shrunk set of options to original set
			#minimum_nonzero_distance_original_index = remaining_pose_indices[minimum_nonzero_distance_index]
			#print("Minimum non-zero distance original index: {}".format(minimum_nonzero_distance_original_index))

			# update distance to travel
			total_distance_to_travel[next_pose] = minimum_nonzero_distance

			# update poses to travel to in new schedule
			distance_minimized_path_of_robot_poses[next_pose,:] = robot_poses[minimum_nonzero_distance_index, :]
			distance_minimized_path_of_robot_poses_indices[next_pose] = minimum_nonzero_distance_index

			# delete the current pose from options to select from in the future 
			#remaining_pose_indices = np.delete(remaining_pose_indices, next_closest_pose_index)
			pairwise_distances[:, next_closest_pose_index] = np.zeros((number_of_poses_to_sort))
			pairwise_distances[next_closest_pose_index, :] = np.zeros((number_of_poses_to_sort))

			#pairwise_distances = np.delete(pairwise_distances, next_closest_pose_index, axis=1)
			#pairwise_distances = np.delete(pairwise_distances, next_closest_pose_index, axis=1)

			# print("Updated remaining_pose_indices:")
			# print(remaining_pose_indices)

			#print("Updated pairwise distances:")
			#print(pairwise_distances)

			# update next pose to look out from and find closest distance
			next_closest_pose_index = minimum_nonzero_distance_index
			#print("Next closest pose index:")
			#print(next_closest_pose_index)

			# if next_pose == len(number_of_poses_to_sort-1):
			# 	remaining_pose_indices = np.delete(remaining_pose_indices, next_closest_pose_index)
			# 	pairwise_distances = np.delete(pairwise_distances, next_closest_pose_index, axis=1)


		#print(pairwise_distances)

		#print(total_distance_to_travel)

		#print(distance_minimized_path_of_robot_poses)
		sum_of_distance = total_distance_to_travel.sum() 
		print("Starting from index {}, total distance to travel: {:.3f} meters".format(next_closest_pose_index_start, sum_of_distance))
		sum_of_distances[next_closest_pose_index_start] = sum_of_distance
		if sum_of_distance < best_path_distance:
			best_path_distance = sum_of_distance
			best_path = distance_minimized_path_of_robot_poses
			best_indices = distance_minimized_path_of_robot_poses_indices
			best_starting_index = next_closest_pose_index_start

	#best_starting_index = np.where(sum_of_distances == np.min(sum_of_distances))
	print("\nActual path to follow:")
	print(best_path)
	print("Best path to follow:")
	print(best_indices)
	print("Best starting index at {}".format(best_starting_index))
	end_time = time.time()
	print("Total distance to travel: {} meters".format(best_path_distance))
	print("\n{} seconds to compute all pose distance minimization".format(end_time-start_time))

	return robot_poses