import cv2
import open3d
import numpy as np
import sys
from PIL import Image
import os
from os import listdir
from os.path import isfile, join
from scipy.optimize import minimize, Bounds
from scipy.spatial.distance import pdist, squareform
import math

def optimize_raw_point_to_point(rotation0, translation0, source_points_file_1, target_points_file_2):
	current_step = 0
	current_error = None

	cloud_1 = o3d.io.read_point_cloud(source_points_file_1)
	pc_tree = o3d.geometry.KDTreeFlann(pc)

	def average_transformed_point_to_point_distance(rotation, translation, source_points, target_points):
		transformed_source_points = np.dot(source_points, rotation) + translation
		average_point_to_point_difference = np.average(np.abs(transformed_source_points - target_points)) * 1000
		return average_point_to_point_difference

	def objective_function_evaluation(hypothesis, args):
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
		#print(average_point_to_point_difference)
		return average_point_to_point_difference

	average_point_to_point_difference = average_transformed_point_to_point_distance(rotation=rotation0, translation=translation0, source_points=source_points, target_points=target_points)
	#print("Original average point-to-point distance = {:.1f} microns".format(average_point_to_point_difference))
	rotation0 = rotation0.flatten().tolist()
	translation0 = translation0.flatten().tolist()
	initial_guess = rotation0 + translation0
	sol = minimize(fun=objective_function_evaluation, x0=initial_guess, args=[source_points, target_points], options={'maxiter': 1000}, method='BFGS')
	transformation = sol.x
	final_rotation = transformation[:9]
	final_rotation = np.asarray(final_rotation).reshape(3,3)

	final_translation = transformation[9:]
	final_translation = np.asarray(final_translation)

	final_average_point_to_point_difference = average_transformed_point_to_point_distance(rotation=final_rotation, translation=final_translation, source_points=source_points, target_points=target_points)
	print("Average point-to-point alignment error = {:.1f} microns (after original error of {:.1f})".format(final_average_point_to_point_difference, average_point_to_point_difference))
	return final_rotation, final_translation




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

	def objective_function_evaluation(hypothesis, args):
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
		#print(average_point_to_point_difference)
		return average_point_to_point_difference

	average_point_to_point_difference = average_transformed_point_to_point_distance(rotation=rotation0, translation=translation0, source_points=source_points, target_points=target_points)
	#print("Original average point-to-point distance = {:.1f} microns".format(average_point_to_point_difference))
	rotation0 = rotation0.flatten().tolist()
	translation0 = translation0.flatten().tolist()
	initial_guess = rotation0 + translation0
	sol = minimize(fun=objective_function_evaluation, x0=initial_guess, args=[source_points, target_points], options={'maxiter': 1000}, method='BFGS')
	transformation = sol.x
	final_rotation = transformation[:9]
	final_rotation = np.asarray(final_rotation).reshape(3,3)

	final_translation = transformation[9:]
	final_translation = np.asarray(final_translation)

	final_average_point_to_point_difference = average_transformed_point_to_point_distance(rotation=final_rotation, translation=final_translation, source_points=source_points, target_points=target_points)
	print("Average point-to-point alignment error = {:.1f} microns (after original error of {:.1f})".format(final_average_point_to_point_difference, average_point_to_point_difference))
	return final_rotation, final_translation



def combine_images(project_name, preprocessing="rotate_left_90_degrees", dual_exposure_mode=False, ambient_image_index = 4, save_16_bit_image=False):
	# R,G,B are the first 3 patterns captured, respectively
	# Read the dark (ambient lighting) image. Note: this is the 5th image in the sequence, not the 4th

	if dual_exposure_mode:
		for counter_index, init_counter in enumerate([0, 51]):
			red_image_path = "/home/sense/3cobot/{}-{}.png".format(project_name, init_counter)
			green_image_path = "/home/sense/3cobot/{}-{}.png".format(project_name, init_counter+1)
			blue_image_path = "/home/sense/3cobot/{}-{}.png".format(project_name, init_counter+2)
			ambient_image_path = "/home/sense/3cobot/{}-{}.png".format(project_name, init_counter+ambient_image_index)

			red_image = cv2.imread(red_image_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
			green_image = cv2.imread(green_image_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
			blue_image = cv2.imread(blue_image_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
			ambient_image = cv2.imread(ambient_image_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)

			width = red_image.shape[0]
			height = red_image.shape[1]

			# subtract the ambient lighting image from the color images
			color_channels = np.zeros((width,height,3), np.float32)

			# vis2 = cv2.CreateMat(3, 1, cv2.CV_32FC3)
			# color_channels = cv2.fromarray(raw_color_channels)

			# convert the color channels to 8-bit to create a 24-bit RGB image. You could possibly make a 16-bit/channel color image, or have 10-bit/channel, but you will have an issue finding viewers to display it!
			color_channels[:,:,0] = cv2.absdiff(red_image, ambient_image) 
			color_channels[:,:,1] = cv2.absdiff(green_image, ambient_image) 
			color_channels[:,:,2] = cv2.absdiff(blue_image, ambient_image) 

			relative_colors = color_channels / 1024
			relative_colors = relative_colors * 2**8

			color_channels = relative_colors.astype('uint8')
			pil_image = Image.fromarray(color_channels) #, 'I;8'

			output_image_name = "{}_{}.png".format(project_name, counter_index)

			# remove leftover images
			for i in range(init_counter, init_counter+29): # number of raw images to clean up
				os.remove("/home/sense/3cobot/{}-{}.png".format(project_name, i))

			if preprocessing == "rotate_left_90_degrees":
				pil_image = pil_image.transpose(Image.ROTATE_90)

			pil_image.save(output_image_name) #,bits=8

	else:
		for counter_index, init_counter in enumerate([0]):
			red_image_path = "/home/sense/3cobot/{}-{}.png".format(project_name, init_counter)
			green_image_path = "/home/sense/3cobot/{}-{}.png".format(project_name, init_counter+1)
			blue_image_path = "/home/sense/3cobot/{}-{}.png".format(project_name, init_counter+2)
			ambient_image_path = "/home/sense/3cobot/{}-{}.png".format(project_name, init_counter+ambient_image_index)

			red_image = cv2.imread(red_image_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
			green_image = cv2.imread(green_image_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
			blue_image = cv2.imread(blue_image_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
			ambient_image = cv2.imread(ambient_image_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)

			width = red_image.shape[0]
			height = red_image.shape[1]

			# subtract the ambient lighting image from the color images
			color_channels = np.zeros((width,height,3), np.float32)

			# vis2 = cv2.CreateMat(3, 1, cv2.CV_32FC3)
			# color_channels = cv2.fromarray(raw_color_channels)

			# convert the color channels to 8-bit to create a 24-bit RGB image. You could possibly make a 16-bit/channel color image, or have 10-bit/channel, but you will have an issue finding viewers to display it!
			color_channels[:,:,0] = cv2.absdiff(red_image, ambient_image) 
			color_channels[:,:,1] = cv2.absdiff(green_image, ambient_image) 
			color_channels[:,:,2] = cv2.absdiff(blue_image, ambient_image) 

			relative_colors_8bit = color_channels / 1024  # 1024 is 10-bit integer representation of color, i.e. colors from scanner are integers between 1-1024
			relative_colors_8bit = relative_colors_8bit * 2**8 # 2^8, i.e. 256, is the 8-bit color representation that most image viewers require


			# 8-bit
			color_channels_8bit = relative_colors_8bit.astype('uint8')
			pil_image = Image.fromarray(color_channels_8bit) #, 'I;8'
			output_image_name_8bit = "{}.png".format(project_name)


			#if ambient_image_index != 3:
				# remove leftover images
			for i in range(init_counter, init_counter+29): # number of raw images to clean up
				try:
					os.remove("/home/sense/3cobot/{}-{}.png".format(project_name, i))
				except FileNotFoundError:
					break


			if preprocessing == "rotate_left_90_degrees":
				pil_image = pil_image.transpose(Image.ROTATE_90)

			pil_image.save(output_image_name_8bit) #,bits=8


			if save_16_bit_image:
				# 16-bit 
				color_channels_normalized_raw = color_channels / 1024
				#pil_image = Image.fromarray(color_channels_16bit) #, 'I;8'
				output_npy_image_10bit = "{}_color.npy".format(project_name)
				with open(output_npy_image_10bit, 'wb') as output_file:
					np.save(output_file, color_channels_normalized_raw)

	if save_16_bit_image:
		return output_npy_image_10bit
	else:
		return output_image_name_8bit

def combine_all_images_with_tag(project_directory = "/home/sense/3cobot", tag = "focus_at_1300mm_", total_views = 9):
	all_files_in_directory = os.listdir(project_directory)
	files_matching_tag = [f for f in all_files_in_directory if tag in f]
	print(files_matching_tag)
	view_indices = [int(f.split(tag)[1].split("_")[0].split("view")[1]) for f in files_matching_tag]

	view_index_to_perspective = {}
	for view_index, file_name in zip(view_indices, files_matching_tag):
		if not view_index_to_perspective.get(view_index, False):
			perspective_header = "_".join(file_name.split(".png")[0].split("_")[:-1])
			print(perspective_header)
			if "pitch" in perspective_header and "yaw" in perspective_header:
				view_index_to_perspective[view_index] = perspective_header

	ordered_project_names = []
	for view_index in range(1, total_views+1):
		ordered_project_names.append(view_index_to_perspective[view_index])

	for project_name in ordered_project_names:
		print(project_name)
		combine_images(project_name=project_name)

if __name__ == "__main__":
	combine_images("1300mm_on_origin")
	#combine_all_images_with_tag()