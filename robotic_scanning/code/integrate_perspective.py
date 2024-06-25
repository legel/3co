from robotics import *
from commander import *
from geometry import *
from localization import *
from registration import optimize_3d_representation, optimize_transformation
from cpd.coherent_point_drift import estimate_transformation_from_a_to_b
from scanning import *
import numpy as np
import sys
import time
import cv2
import math

min_unit_position = -3
max_unit_position = 4

project_tag = "x_unit_deviation"
xs = [0.975 + i*0.01 for i in range(min_unit_position,max_unit_position)]
ys = [-0.09 for _ in range(min_unit_position,max_unit_position)]
zs = [0.61 for _ in range(min_unit_position,max_unit_position)]
pitchs = [-90.0 for _ in range(min_unit_position,max_unit_position)]
yaws = [90.0 for _ in range(min_unit_position,max_unit_position)]
x_unit_deviations = [project_tag, xs, ys, zs, pitchs, yaws]

project_tag = "y_unit_deviation"
xs = [0.975 for _ in range(min_unit_position,max_unit_position)]
ys = [-0.09 + i*0.01 for i in range(min_unit_position,max_unit_position)]
zs = [0.61 for _ in range(min_unit_position,max_unit_position)]
pitchs = [-90.0 for _ in range(min_unit_position,max_unit_position)]
yaws = [90.0 for _ in range(min_unit_position,max_unit_position)]
y_unit_deviations = [project_tag, xs, ys, zs, pitchs, yaws]

project_tag = "z_unit_deviation"
xs = [0.975 for _ in range(min_unit_position,max_unit_position)]
ys = [-0.09 for _ in range(min_unit_position,max_unit_position)]
zs = [0.61 + i*0.01 for i in range(min_unit_position,max_unit_position)]
pitchs = [-90.0 for _ in range(min_unit_position,max_unit_position)]
yaws = [90.0 for _ in range(min_unit_position,max_unit_position)]
z_unit_deviations = [project_tag, xs, ys, zs, pitchs, yaws]

project_tag = "yaw_unit_deviation"
xs = [0.975 for _ in range(min_unit_position,max_unit_position)]
ys = [-0.09 for _ in range(min_unit_position,max_unit_position)]
zs = [0.61 for _ in range(min_unit_position,max_unit_position)]
pitchs = [-90.0 for _ in range(min_unit_position,max_unit_position)]
yaws = [90.0 + i*1.00 for i in range(min_unit_position,max_unit_position)]
yaw_unit_deviations = [project_tag, xs, ys, zs, pitchs, yaws]

min_unit_position = 0
max_unit_position = 7

project_tag = "pitch_unit_deviation"
xs = [0.975 for _ in range(min_unit_position,max_unit_position)]
ys = [-0.09 for _ in range(min_unit_position,max_unit_position)]
zs = [0.61 for _ in range(min_unit_position,max_unit_position)]
pitchs = [-90.0 + i*1.00 for i in range(min_unit_position,max_unit_position)]
yaws = [90.0 for _ in range(min_unit_position,max_unit_position)]
pitch_unit_deviations = [project_tag, xs, ys, zs, pitchs, yaws]


all_projects_to_run = [x_unit_deviations, y_unit_deviations, z_unit_deviations, pitch_unit_deviations, yaw_unit_deviations]
axis_labels = ["x","y","z","pitch","yaw"]
for project_index, project_to_run in enumerate(all_projects_to_run):
	axis_label = axis_labels[project_index]
	project_tag, xs, ys, zs, pitchs, yaws = project_to_run

	#project_tag = "yaw_unit_degree"
	project_names = ["{}_{}".format(project_tag, i) for i in range(len(xs))]

	rescan_all = False

	axes_to_recalibrate = []
	multi_batch_recalibrate = False

	if rescan_all:
		new_scan = True
		new_plane_finder = True
		new_calibot_extraction = True
		add_superpoints = True
	else:
		new_scan = False
		new_plane_finder = False
		new_calibot_extraction = False
		add_superpoints = False

	extrinsic_transform = False
	exit_after_one = False
	estimate_transforms_and_combine_point_clouds = True

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
				print("{} - moving robot to (x={},y={},z={},pitch={},yaw={}) then sleeping for 10 seconds".format(project_name,x,y,z,pitch,yaw))
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
				points_to_downsample.downsample(reduce_by_1_divided_by_n=5)
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

		negative_to_positive_rotation_matrices = np.zeros((3,3), dtype=np.float32)
		positive_to_negative_rotation_matrices = np.zeros((3,3), dtype=np.float32)
		negative_to_positive_translation_matrices = np.zeros((3), dtype=np.float32)
		positive_to_negative_translation_matrices = np.zeros((3), dtype=np.float32)

		clouds_to_process = len(xs)
		point_clouds = []
		reds = [0,0,255]
		greens = [0,255,0]
		blues = [255,0,0]
		for i in range(clouds_to_process):
			#if i == clouds_to_process:
			#	break
			#data_project_name = "ikea_chair_2_{}".format(i)
			data_project_name = "{}_{}".format(project_tag,i)
			points = PointCloud(filename="{}-0-0.ply".format(data_project_name), project_name=data_project_name, focus_on_center_pixels=True, row_crop=600, column_crop=200)
			#points.set_all_points_to_color(red=reds[i], green=greens[i], blue=blues[i])
			point_clouds.append(points)

		source_cloud = point_clouds[0]

		all_rotations = []
		all_translations = []


		# negative to positive
		for base, project_name in zip(range(0,9), project_names):
			source = base
			target = base + 1
			if target == clouds_to_process:
				break
			points_source_filepath = "/home/sense/3cobot/{}_{}_3d_coordinates.csv".format(project_tag, base)
			points_target_filepath = "/home/sense/3cobot/{}_{}_3d_coordinates.csv".format(project_tag, target)
			rotation, translation, source_points, target_points = estimate_transformation_from_a_to_b(points_source_filepath=points_source_filepath, points_target_filepath=points_target_filepath)
			rotation, translation = optimize_transformation(rotation0=rotation, translation0=translation, source_points=source_points, target_points=target_points)
			print("From {} ({}) to {} ({}):\nRotation: {}\nTranslation: {}".format(project_name, points_source_filepath, "...{}".format(target),points_target_filepath, rotation, translation))

			target_cloud = point_clouds[target]
			source_cloud.transform(rotation,translation)
			source_cloud.add_point_cloud(target_cloud)

			all_rotations.append(rotation)
			all_translations.append(translation)

		chained_clouds = "combined.ply".format(project_names[0])
		source_cloud.save_as_ply(chained_clouds)

		average_rotation = np.average(all_rotations, axis=0)
		average_translation = np.average(all_translations, axis=0)

		negative_to_positive_rotation_matrices[:,:] = average_rotation
		negative_to_positive_translation_matrices[:] = average_translation

		with open('/home/sense/3cobot/{}_increasing_rotation_matrix.npy'.format(axis_label), 'wb') as output_file:
			np.save(output_file, negative_to_positive_rotation_matrices)

		with open('/home/sense/3cobot/{}_increasing_translation_matrix.npy'.format(axis_label), 'wb') as output_file:
			np.save(output_file, negative_to_positive_translation_matrices)

		print("\n\n\nFor {}:".format(project_tag))
		print("Average rotation per unit: {}".format(average_rotation))
		print("Average translation per unit: {}\n\n\n".format(average_translation))


		# positive to negative
		for base in range(clouds_to_process-1,-1):
			source = base
			target = base - 1
			if target == -1:
				break
			points_source_filepath = "/home/sense/3cobot/{}_{}_3d_coordinates.csv".format(project_tag, base)
			points_target_filepath = "/home/sense/3cobot/{}_{}_3d_coordinates.csv".format(project_tag, target)
			rotation, translation, source_points, target_points = estimate_transformation_from_a_to_b(points_source_filepath=points_source_filepath, points_target_filepath=points_target_filepath)
			rotation, translation = optimize_transformation(rotation0=rotation, translation0=translation, source_points=source_points, target_points=target_points)
			print("From {} ({}) to {} ({}):\nRotation: {}\nTranslation: {}".format(project_name, points_source_filepath, "...{}".format(target),points_target_filepath, rotation, translation))

			target_cloud = point_clouds[target]
			source_cloud.transform(rotation,translation)
			source_cloud.add_point_cloud(target_cloud)

			all_rotations.append(rotation)
			all_translations.append(translation)

		chained_clouds = "combined.ply"
		source_cloud.save_as_ply(chained_clouds)

		average_rotation = np.average(all_rotations, axis=0)
		average_translation = np.average(all_translations, axis=0)

		positive_to_negative_rotation_matrices[:,:] = average_rotation
		positive_to_negative_translation_matrices[:] = average_translation

		with open('/home/sense/3cobot/{}_decreasing_rotation_matrix.npy'.format(axis_label), 'wb') as output_file:
			np.save(output_file, positive_to_negative_rotation_matrices)

		with open('/home/sense/3cobot/{}_decreasing_translation_matrix.npy'.format(axis_label), 'wb') as output_file:
			np.save(output_file, positive_to_negative_translation_matrices)

		print("\n\n\nFor {}:".format(project_tag))
		print("Average rotation per unit: {}".format(average_rotation))
		print("Average translation per unit: {}\n\n\n".format(average_translation))