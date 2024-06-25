from robotics import *
from commander import *
from geometry import *
from localization import *
from scipy.optimize import minimize, Bounds
from scipy.spatial.distance import pdist, squareform
from cpd.coherent_point_drift import estimate_transformation_from_a_to_b
from scanning import *
import numpy as np
import sys
import time
import cv2
import math

def optimize_3d_representation(xyz_estimates, plane, charuco_points_found):
	a = plane.a
	b = plane.b
	c = plane.c
	d = plane.d

	xyz_1 = xyz_estimates.flatten().tolist()
	#point_to_point_distance_guess = 32.0

	initial_guess = xyz_1[:9]
	#initial_guess.append(point_to_point_distance_guess)

	#number_of_points = xyz_estimates.shape[0]
	#square_length_difference_guess = 0.475
	#initial_guess = [square_length_difference_guess]

	#original_xyz = xyz_estimates
	#xyz_estimates = xyz_estimates.flatten()
	#initial_guess.extend(xyz_estimates)

	#print("\n Initial Guess: {}".format(initial_guess))

	#print("\n\n\nINPUT DATA: {}\n\n\n".format(xyz_1))

	def compute_xyz_gridpoints(x0, y0, z0, x_right, y_right, z_right, x_below, y_below, z_below, found_points):
		point_to_point_distance_estimate = 32.000
		number_of_columns_per_row = 14
		number_of_rows_per_plane = 3

		all_xyz_points = np.zeros((number_of_columns_per_row*number_of_rows_per_plane, 3))
		refined_found_xyz_gridpoints = np.zeros((len(found_points), 3))

		column_positions = np.linspace(0, point_to_point_distance_estimate*(number_of_columns_per_row-1), number_of_columns_per_row)
		row_positions = np.linspace(0, point_to_point_distance_estimate*(number_of_rows_per_plane-1), number_of_rows_per_plane)

		# define relative direction along plane in right direction
		x_right_vector = x_right - x0
		y_right_vector = y_right - y0
		z_right_vector = z_right - z0
		right_vector_magnitude = math.sqrt(x_right_vector**2 + y_right_vector**2 + z_right_vector**2)
		x_right_vector_norm = x_right_vector / right_vector_magnitude
		y_right_vector_norm = y_right_vector / right_vector_magnitude
		z_right_vector_norm = z_right_vector / right_vector_magnitude

		# define relive direction along plane in below direction 
		x_below_vector = x_below - x0
		y_below_vector = y_below - y0
		z_below_vector = z_below - z0
		below_vector_magnitude = math.sqrt(x_below_vector**2 + y_below_vector**2 + z_below_vector**2)
		x_below_vector_norm = x_below_vector / below_vector_magnitude
		y_below_vector_norm = y_below_vector / below_vector_magnitude
		z_below_vector_norm = z_below_vector / below_vector_magnitude

		# model all points on the board, including those not actually seen
		for row_1 in range(number_of_rows_per_plane):
			row_1_position = row_positions[row_1]
			for column_1 in range(number_of_columns_per_row):
				column_1_position = column_positions[column_1]
				point_1_index = int(row_1*number_of_columns_per_row + column_1)
				all_xyz_points[point_1_index][0] = x0 + x_right_vector*column_1_position + x_below_vector*row_1_position
				all_xyz_points[point_1_index][1] = y0 + y_right_vector*column_1_position + y_below_vector*row_1_position
				all_xyz_points[point_1_index][2] = z0 + z_right_vector*column_1_position + z_below_vector*row_1_position


		# create a matrix for the points that were actually seen, for error calculation
		xyz_points_in_view = np.zeros((len(found_points),3))
		for i, point_id_on_plane in enumerate(found_points):
			xyz_points_in_view[i][0] = all_xyz_points[point_id_on_plane][0]
			xyz_points_in_view[i][1] = all_xyz_points[point_id_on_plane][1]				
			xyz_points_in_view[i][2] = all_xyz_points[point_id_on_plane][2]

		return all_xyz_points, xyz_points_in_view


	def compute_distance_matrix(points_found):
		square_length = 32
		number_of_planes = 12
		number_of_points_per_plane = 42
		number_of_columns_per_row = 14
		number_of_rows_per_plane = 3

		distance_matrix = np.zeros((number_of_points_per_plane, number_of_points_per_plane))

		column_positions = np.linspace(0, square_length*(number_of_columns_per_row-1), number_of_columns_per_row)
		row_positions = np.linspace(0, square_length*(number_of_rows_per_plane-1), number_of_rows_per_plane)

		# make a distance matrix genrally for all points on a single board
		for row_1 in range(number_of_rows_per_plane):
		    row_1_position = row_positions[row_1]
		    for column_1 in range(number_of_columns_per_row):
		        column_1_position = column_positions[column_1]
		        point_1_index = int(row_1*number_of_columns_per_row + column_1)
		        for row_2 in range(number_of_rows_per_plane):
		            row_2_position = row_positions[row_2]
		            for column_2 in range(number_of_columns_per_row):
		                column_2_position = column_positions[column_2]
		                point_2_index = int(row_2*number_of_columns_per_row + column_2)
		                distance = math.sqrt((row_2_position - row_1_position)**2 + (column_2_position-column_1_position)**2)
		                distance_matrix[point_1_index, point_2_index] = round(distance,4)

		# make a distance matrix only for those charuco points found
		points_found_for_this_plane = len(points_found)
		found_charuco_distance_matrix = np.zeros((points_found_for_this_plane,points_found_for_this_plane))
		for i, relative_charuco_id_1 in enumerate(points_found):
			for j, relative_charuco_id_2 in enumerate(points_found):
				distance_1_2 = distance_matrix[relative_charuco_id_1][relative_charuco_id_2]
				found_charuco_distance_matrix[i][j] = distance_1_2
				#print("Global plane {} with {:.3f}x + {:.3f}y + {:.3f}z + {:.3f} = 0, relative ID {} (global ID {}): {:.3f}mm to relative ID {} (global ID {})".format(global_plane_number, plane.a,plane.b,plane.c,plane.d,relative_charuco_id_1, global_id_1, distance_1_2, relative_charuco_id_2, global_id_2))

		return found_charuco_distance_matrix


	def optimize_model(hypothesis, args):
		x0, y0, z0, x_right, y_right, z_right, x_below, y_below, z_below = hypothesis
		found_points = args[0]
		#print("x0: {:.3f}, y0: {:.3f}, z0:{:.3f}, xR: {:.3f}, yR: {:.3f}, zR: {:.3f}, xL: {:.3f}, yL: {:.3f}, zL: {:.3f}".format(x0, y0, z0, x_right, y_right, z_right, x_below, y_below, z_below))
		all_xyz_points, xyz_points_in_view = compute_xyz_gridpoints(x0, y0, z0, x_right, y_right, z_right, x_below, y_below, z_below, found_points)

		#distance_matrix = compute_distance_matrix(square_guess, points_found)

		xyz_1 = args[1]

		xyz_1 = np.asarray(xyz_1).reshape(len(found_points),3)


		x1 = xyz_1[:,0]
		y1 = xyz_1[:,1]
		z1 = xyz_1[:,2]

		x2 = xyz_points_in_view[:,0]
		y2 = xyz_points_in_view[:,1]
		z2 = xyz_points_in_view[:,2]

		a = args[2][0]
		b = args[2][1]
		c = args[2][2]
		d = args[2][3]

		#distance_matrix = args[2]
		# original_xyz = args[2]

		# x1 = original_xyz[:,0]
		# y1 = original_xyz[:,1]
		# z1 = original_xyz[:,2]

		# number_of_points = args[0]
		# points_found = args[3]

		# square_guess = 32.0 + hypothesis[-1]
		# #print("Square length guess = {:6f}mm given difference hypothesis of {}".format(square_guess, hypothesis[-1]))

		distance_matrix = compute_distance_matrix(found_points)

		# xyz_hypotheses = hypothesis[:-1].reshape(number_of_points,3)
		# #print("New XYZ hypothesis: {}".format(xyz_hypotheses))

		# x2 = xyz_hypotheses[:,0]
		# y2 = xyz_hypotheses[:,1]
		# z2 = xyz_hypotheses[:,2]

		# a = args[1][0]
		# b = args[1][1]
		# c = args[1][2]
		# d = args[1][3]

		# #distance_matrix = args[2]
		# original_xyz = args[2]

		# x1 = original_xyz[:,0]
		# y1 = original_xyz[:,1]
		# z1 = original_xyz[:,2]

		#print("\n\nNew (x,y,z) estimates:\n{}".format(hypothesis))
		#print("Original (x,y,z) estimates:\n{}".format(original_xyz))

		# MEASURE CLOSENESS TO ORIGINAL MEASUREMENTS
		x_delta = (x2-x1)**2
		y_delta = (y2-y1)**2
		z_delta = (z2-z1)**2
		differences_from_original_position = np.sqrt(x_delta + y_delta + z_delta)
		average_position_difference = np.average(differences_from_original_position)
		#sum_of_differences_from_original_position = differences_from_original_position.sum()
		#print("Differences from original position: {}".format(differences_from_original_position))
		#print("Average difference from original position: {:.5f}mm".format(average_position_difference))

		# MEASURE DISTANCE MATRIX
		hypothesis_xyz_distances = squareform(pdist(xyz_points_in_view, 'euclidean'))
		original_xyz_distances = squareform(pdist(xyz_1, 'euclidean'))

		#print("\nNew distance matrix: {}".format(hypothesis_xyz_distances))
		#print("Original distance matrix: {}".format(original_xyz_distances))
		#print("Perfect distance matrix: {}".format(distance_matrix))
		differences_from_distance_matrix = np.sqrt((hypothesis_xyz_distances - distance_matrix)**2)
		original_differences_from_distance_matrix = np.sqrt((original_xyz_distances - distance_matrix)**2)

		average_deviation_from_ideal_distance = np.average(differences_from_distance_matrix)
		original_average_deviation_from_ideal_distance = np.average(original_differences_from_distance_matrix)

		#print("Subtraction of distance matrices: {}".format(differences_from_distance_matrix))
		#sum_of_differences_from_distance_matrix = differences_from_distance_matrix.sum()

		#print("Average deviation from expected distance: {:.5f}mm".format(average_deviation_from_ideal_distance))
		#print("Original average deviation from expected distance: {:.5f}mm".format(original_average_deviation_from_ideal_distance))


		# MEASURE PLANAR DEVIATION
		hypothesis_differences_from_plane = np.sqrt((a*x2 + b*y2 + c*z2 + d)**2)
		original_differences_from_plane = np.sqrt((a*x1 + b*y1 + c*z1 + d)**2)
		#print("\nNew planar deviation: {}".format(hypothesis_differences_from_plane))
		#print("Original planar deviation: {}".format(original_differences_from_plane))
		sum_of_hypothesis_differences_from_plane = hypothesis_differences_from_plane.sum()
		sum_of_original_differences_from_plane = original_differences_from_plane.sum()
		#print("Sum of new planar deviation: {}".format(sum_of_hypothesis_differences_from_plane))    	
		#print("Sum of original planar deviation: {}".format(sum_of_original_differences_from_plane))    	

		# PENALIZE DEVIATION FROM BASELINE PRINT ESTIMATE
		#baseline_estimate = 32.000
		#deviation_from_print_ideal_distance = abs(baseline_estimate - point_to_point_distance_estimate)

		objective_error = 10*average_position_difference + sum_of_hypothesis_differences_from_plane #+ 2.5*deviation_from_print_ideal_distance
		#weighted_objective_error = sum_of_differences_from_original_position + sum_of_differences_from_distance_matrix + sum_of_hypothesis_differences_from_plane
		print("\rError {:.4f} (Planar {:.3f}, Position Deviation {:.3f}mm)                    ".format(objective_error, sum_of_hypothesis_differences_from_plane, average_position_difference), end='', flush=True)
		return objective_error

	def unit_norm(hypothesis):
		x0, y0, z0, x1, y1, z1, x2, y2, z2 = hypothesis
		distance_1 = np.sqrt((x0-x1)**2 + (y0-y1)**2 + (z0-z1)**2)
		distance_2 = np.sqrt((x0-x2)**2 + (y0-y2)**2 + (z0-z2)**2)
		return distance_1 - distance_2

	def unit_length(hypothesis):
		x0, y0, z0, x1, y1, z1, x2, y2, z2 = hypothesis
		distance_1 = np.sqrt((x0-x1)**2 + (y0-y1)**2 + (z0-z1)**2)
		return distance_1 - 32.000

	def perpendicular(hypothesis):
		r = 32.000
		x0, y0, z0, x1, y1, z1, x2, y2, z2 = hypothesis
		hypotenuse = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
		return hypotenuse - math.sqrt(2.0) * r

	# constrain the hypotheses to be of same length
	#cons = ({'type': 'eq', 'fun': unit_norm}, {'type': 'eq', 'fun': unit_length}, {'type': 'eq', 'fun': perpendicular})
	# constraints=cons,

	# x0, y0, z0, x-right, y-right, z-right, x-below, y-below, z-below
	#bounds = [(-1000.0, 1000.0), (-1000.0, 1000.0), (0.0, 1000.0), (-1000.0, 1000.0), (-1000.0, 1000.0), (0.0, 1000.0), (-1000.0, 1000.0), (-1000.0, 1000.0), (0.0, 1000.0)]

	sol = minimize(fun=optimize_model, x0=initial_guess, args=[charuco_points_found, xyz_1, [a,b,c,d]], options={'maxiter': 1000}, method='BFGS') # , constraints=cons # , ,  
	#print(sol)
	x0, y0, z0, x_right, y_right, z_right, x_below, y_below, z_below = sol.x
	all_xyz_points, xyz_points_in_view = compute_xyz_gridpoints(x0, y0, z0, x_right, y_right, z_right, x_below, y_below, z_below, charuco_points_found)

	print()
	return all_xyz_points, xyz_points_in_view


def optimize_transformation(rotation0, translation0, source_points, target_points):
	current_step = 0
	current_error = None

	def average_transformed_point_to_point_distance(rotation, translation, source_points, target_points):
		transformed_source_points = np.dot(source_points, rotation) + translation
		average_point_to_point_difference = np.average(np.abs(transformed_source_points - target_points)) * 1000
		return average_point_to_point_difference

	def squeeze(hypothesis, args):
		new_rotation = hypothesis[:9]
		new_rotation = np.asarray(new_rotation).reshape(3,3)
		new_translation = hypothesis[9:]
		new_translation = np.asarray(new_translation)
		source_points = args[0]
		target_points = args[1]
		average_point_to_point_difference = average_transformed_point_to_point_distance(rotation=new_rotation, translation=new_translation, source_points=source_points, target_points=target_points)
		#print("New rotation: {}".format(new_rotation))
		#print("New translation: {}".format(new_translation))
		#global current_step
		#current_step += 1
		#if average_point_to_point_difference != current_error:
		#current_error = average_point_to_point_difference
		return average_point_to_point_difference

	average_point_to_point_difference = average_transformed_point_to_point_distance(rotation=rotation0, translation=translation0, source_points=source_points, target_points=target_points)
	#print("Original average point-to-point distance = {:.1f} microns".format(average_point_to_point_difference))
	rotation0 = rotation0.flatten().tolist()
	translation0 = translation0.flatten().tolist()
	initial_guess = rotation0 + translation0
	sol = minimize(fun=squeeze, x0=initial_guess, args=[source_points, target_points], options={'maxiter': 1000}, method='BFGS')
	transformation = sol.x
	final_rotation = transformation[:9]
	final_rotation = np.asarray(final_rotation).reshape(3,3)

	final_translation = transformation[9:]
	final_translation = np.asarray(final_translation)

	average_point_to_point_difference = average_transformed_point_to_point_distance(rotation=final_rotation, translation=final_translation, source_points=source_points, target_points=target_points)
	print("Average point-to-point alignment error = {:.1f} microns".format(average_point_to_point_difference))
	return final_rotation, final_translation


pitchs = [i*15 - 90.0 for i in range(10)]
xs = [0.11 for _ in pitchs]
ys = [-0.09 for _ in pitchs]
zs = [0.56 for _ in pitchs]
yaws = [90.0 for _ in pitchs]

project_names = ["pitch_registration_{}".format(i) for i in range(10)]

axes_to_recalibrate = []
multi_batch_recalibrate = False
new_scan = False
new_plane_finder = False
new_calibot_extraction = False
add_superpoints = True
extrinsic_transform = False
exit_after_one = False
estimate_transforms_and_combine_point_clouds = True
point_cloud_to_add_project_name = "pitch_registration"

if new_scan:
	robot = Robot()


if new_calibot_extraction:

	standard_deviation_history_per_corner_id = {}
	scan_number = 0
	for x,y,z,pitch,yaw,project_name in zip(xs,ys,zs,pitchs,yaws,project_names):
		scan_number += 1
		if scan_number <= 0:
			continue
		if scan_number > 10:
			sys.exit(0)

		if new_scan:
			print("{} - (x={},y={},z={},pitch={},yaw={})".format(project_name,x,y,z,pitch,yaw))
			commander.move({"x":x, "y":y, "z":z, "pitch":pitch, "yaw": yaw})
			time.sleep(10.0)
			if multi_batch_recalibrate:
				for axis in axes_to_recalibrate:
					commander.calibrate(axis)				
			points = robot.scan(project_name=project_name, distance=555.0, exposure_time=75.0)
		else:
			points = PointCloud(filename="{}-0-0.ply".format(project_name), project_name=project_name)

		if new_plane_finder:
			points_to_downsample = points.copy()
			points_to_downsample.downsample(reduce_by_1_divided_by_n=1)
			point_cloud_file = points_to_downsample.preprocess_points_for_plane_finding()
			finder = PlaneFinder(project_name=project_name, path_to_point_cloud_file=point_cloud_file)
		else:	
			plane_data_file = "{}_plane_data.csv".format(project_name)
			finder = PlaneFinder(project_name=project_name, path_to_plane_file=plane_data_file)

		planes = finder.best_planes
		# for i, plane in enumerate(finder.best_planes):
		# 	planes[]
		# 	print(plane.normal_vector())

		image_name = "{}-0-0.png".format(project_name)

		first_time_localization = True
		show_final_detections=False
		charuco_id_to_planes, charuco_id_to_plane_point_index, plane_distance_matrix, global_plane_to_global_points = get_charuco_id_to_plane_data()

		if first_time_localization:
			corner_coordinates, corner_ids, camera_intrinsics_matrix  = detect_saddle_points(image_name, show_aruco_detections=False,  show_final_detections=show_final_detections, undistort = False, optimize_camera_matrix=False, recrop=False)
			with open('{}_charuco_ids.npy'.format(project_name), 'wb') as output_file:
				np.save(output_file, corner_ids)
			with open('{}_charuco_coordinates.npy'.format(project_name), 'wb') as output_file:
				np.save(output_file, corner_coordinates)
		else:
			with open('{}_charuco_ids.npy'.format(project_name), 'rb') as input_file:
				corner_ids = np.load(input_file)
			with open('{}_charuco_coordinates.npy'.format(project_name), 'rb') as input_file:
				corner_coordinates = np.load(input_file)

		#min_point_to_point_distance, max_point_to_point_distance, avg_point_to_point_distance = points.average_neighboring_point_distance()

		# pursue optimization where minimize = math.sqrt((x_1 - x_2) ** 2 + ...) + sum_of_deviation_from_distance_matrix + deviation_from_plane_equation
		number_of_points = corner_ids.shape[0]
		#print("{} feature points found".format(number_of_points))
		point_to_plane = {}
		estimated_x = {}
		estimated_y = {}
		estimated_z = {}

		found_global_charuco_points_by_plane = {}
		found_relative_charuco_points_by_plane = {}

		for plane_number, plane in enumerate(planes):
			found_global_charuco_points_by_plane[plane_number] = []
			found_relative_charuco_points_by_plane[plane_number] = []

		for cooordinate, corner_id in zip(corner_coordinates, corner_ids):
			u = 2048 - cooordinate[0][0]
			v = cooordinate[0][1]
			#print("ID {}: sub-pixel ({:.3f},{:.3f}), plane {}".format(corner_id[0], u,v, charuco_id_to_planes[corner_id[0]]))
			charuco_id = corner_id[0]
			closest_point = points.get_closest_point_by_subpixel_row_column(row=u, column=v)
			x = closest_point.x[0]
			y = closest_point.y[0]
			z = closest_point.z[0]
			pixel_row = int(closest_point.pixel_row[0])
			pixel_column = int(closest_point.pixel_column[0])

			points.add_points(x=x, y=y, z=z, red=0, green=255, blue=0)
			points.add_superpoint(x=x, y=y, z=z, red=255, green=0, blue=125, sphere_radius=0.25, superpoint_samples=200)
			points.add_superpoint(x=x, y=y, z=z, red=100, green=0, blue=0, sphere_radius=3, superpoint_samples=1000)

			for plane_number, plane in enumerate(planes):
				if plane.point_exists(pixel_row, pixel_column):
					#perpendicular_distance = plane.perpendicular_error_for_point(x,y,z)
					# #closest_x, closest_y, closest_z = plane.find_closest_point_on_plane(x,y,z)
					# distance = math.sqrt((x-closest_x)**2 + (y-closest_y)**2 + (z-closest_z)**2)
					#print("({},{}) ({:.3f},{:.3f},{:.3f}) on plane {}: ({:.3f},{:.3f},{:.3f}) ({:.3f} mm)".format(pixel_row, pixel_column, x,y,z, plane_number, closest_x, closest_y, closest_z, distance))

					#print("ID {} at ({},{}) ({:.3f},{:.3f},{:.3f}) on plane {}: ({:.2f} microns)".format(charuco_id, pixel_row, pixel_column, x,y,z, plane_number, perpendicular_distance * 1000))
					estimated_x[charuco_id] = x
					estimated_y[charuco_id] = y
					estimated_z[charuco_id] = z

					found_global_charuco_points_by_plane[plane_number].append(charuco_id)
					relative_point_index = charuco_id_to_plane_point_index[charuco_id] 
					found_relative_charuco_points_by_plane[plane_number].append(relative_point_index)

					# points.add_points(x=closest_x, y=closest_y, z=closest_z, red=0, green=0, blue=255)
					# points.add_superpoint(x=closest_x, y=closest_y, z=closest_z, red=0, green=255, blue=0, sphere_radius=0.25, superpoint_samples=200)
					# points.add_superpoint(x=closest_x, y=closest_y, z=closest_z, red=100, green=0, blue=255, sphere_radius=3, superpoint_samples=1000)

			# points.add_points(x=x, y=y, z=z, red=0, green=255, blue=0)
			# points.add_superpoint(x=x, y=y, z=z, red=255, green=0, blue=125, sphere_radius=0.25, superpoint_samples=200)
			# points.add_superpoint(x=x, y=y, z=z, red=100, green=0, blue=0, sphere_radius=3, superpoint_samples=1000)

		ground_truth_coordinates_file_name = "{}_3d_coordinates.csv".format(project_name)
		unique_global_planes = []
		with open(ground_truth_coordinates_file_name, "w") as output_file:

			ground_truth_coordinates = []
			for plane_number, charuco_points_found in found_relative_charuco_points_by_plane.items():

				points_found_for_this_plane = len(charuco_points_found)
				xyz_estimates = np.zeros((points_found_for_this_plane,3))
				xyz_estimates_global_plane = np.zeros((points_found_for_this_plane,1), dtype=np.int16)

				plane = planes[plane_number]
				found_charuco_distance_matrix = np.zeros((points_found_for_this_plane,points_found_for_this_plane))
				global_points_found = []
				for i, relative_charuco_id_1 in enumerate(charuco_points_found):
					global_id_1 = found_global_charuco_points_by_plane[plane_number][i]
					global_points_found.append(global_id_1)
					global_plane_number = charuco_id_to_planes[global_id_1]
					xyz_estimates[i,0] = estimated_x[global_id_1]
					xyz_estimates[i,1] = estimated_y[global_id_1]
					xyz_estimates[i,2] = estimated_z[global_id_1]
					xyz_estimates_global_plane[i] = global_plane_number
					if global_plane_number not in unique_global_planes:
						unique_global_planes.append(global_plane_number)
					for j, relative_charuco_id_2 in enumerate(charuco_points_found):
						distance_1_2 = plane_distance_matrix[relative_charuco_id_1][relative_charuco_id_2]
						global_id_2 = found_global_charuco_points_by_plane[plane_number][j]
						#found_charuco_distance_matrix[i][j] = distance_1_2
					print("Global plane {} with {:.3f}x + {:.3f}y + {:.3f}z + {:.3f} = 0, relative ID {} (global ID {}): v1 estimate (x={:.3f}, y={:.3f}, z={:.3f})".format(global_plane_number, plane.a,plane.b,plane.c,plane.d,relative_charuco_id_1, global_id_1, xyz_estimates[i,0], xyz_estimates[i,1], xyz_estimates[i,2]))
				global_points_for_this_plane = global_plane_to_global_points[global_plane_number]
				show_all_points = False
				if len(xyz_estimates) > 0:
					all_xyz_points, xyz_points_in_view = optimize_3d_representation(xyz_estimates=xyz_estimates, plane=plane, charuco_points_found=charuco_points_found)
					if show_all_points:
						for i, global_point in zip(range(all_xyz_points.shape[0]), global_points_for_this_plane):
							optimized_x = all_xyz_points[i,0]
							optimized_y = all_xyz_points[i,1]
							optimized_z = all_xyz_points[i,2]
							points.add_points(x=optimized_x, y=optimized_y, z=optimized_z, red=0, green=0, blue=255)
							points.add_superpoint(x=optimized_x, y=optimized_y, z=optimized_z, red=0, green=255, blue=0, sphere_radius=0.25, superpoint_samples=200)
							points.add_superpoint(x=optimized_x, y=optimized_y, z=optimized_z, red=100, green=0, blue=255, sphere_radius=3, superpoint_samples=1000)
							output_file.write("{},{},{},{}\n".format(global_point, optimized_x, optimized_y, optimized_z))				
					else:			
						for i, global_point in zip(range(points_found_for_this_plane), global_points_found):
							optimized_x = xyz_points_in_view[i,0]
							optimized_y = xyz_points_in_view[i,1]
							optimized_z = xyz_points_in_view[i,2]
							points.add_points(x=optimized_x, y=optimized_y, z=optimized_z, red=0, green=0, blue=255)
							points.add_superpoint(x=optimized_x, y=optimized_y, z=optimized_z, red=0, green=255, blue=0, sphere_radius=0.25, superpoint_samples=200)
							points.add_superpoint(x=optimized_x, y=optimized_y, z=optimized_z, red=100, green=0, blue=255, sphere_radius=3, superpoint_samples=1000)
							output_file.write("{},{},{},{}\n".format(global_point, optimized_x, optimized_y, optimized_z))				
		print("\n\nSaved ground truth (x,y,z) coordinates extracted for this scan as {}".format(ground_truth_coordinates_file_name))

		visualization = "{}_with_superpoints.ply".format(project_name)
		points.save_as_ply(visualization)

if estimate_transforms_and_combine_point_clouds:
	clouds_to_process = 10
	point_clouds = []
	for i, project_name in enumerate(project_names):
		if i == clouds_to_process:
			break
		points = PointCloud(filename="{}-0-0.ply".format(project_name), project_name=project_name)
		point_clouds.append(points)

	source_cloud = point_clouds[0]

	for base, project_name in zip(range(0,9), project_names):
		source = base
		target = base + 1
		if target == clouds_to_process:
			break
		points_source_filepath = "/home/sense/3cobot/pitch_registration_{}_3d_coordinates.csv".format(base)
		points_target_filepath = "/home/sense/3cobot/pitch_registration_{}_3d_coordinates.csv".format(target)
		rotation, translation, source_points, target_points = estimate_transformation_from_a_to_b(points_source_filepath=points_source_filepath, points_target_filepath=points_target_filepath)
		rotation, translation = optimize_transformation(rotation0=rotation, translation0=translation, source_points=source_points, target_points=target_points)
		target_cloud = point_clouds[target]
		source_cloud.transform(rotation,translation)
		source_cloud.add_point_cloud(target_cloud)

	chained_clouds = "combined_clouds_from_{}.ply".format(project_names[0])
	source_cloud.save_as_ply(chained_clouds)
