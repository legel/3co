from commander import *
from robotics import *
from geometry import *
from pympler import tracker, asizeof
from operator import itemgetter
#from memory_profiler import profile
import numpy as np
import time
import sys
import os
import psutil

# py spherical_calibrator.py radius_count=1 radius_min=0.45 radius_max=0.45 pitch_count=1 pitch_min_value=-15 pitch_max_value=-15 yaw_count=1 yaw_min_value=-20.0 yaw_max_value=-20.0
try:
	radius_count = int(sys.argv[1].split("=")[1]) #1
	radius_min = float(sys.argv[2].split("=")[1]) #0.45
	radius_max = float(sys.argv[3].split("=")[1]) #0.45
	pitch_count = int(sys.argv[4].split("=")[1]) #1
	pitch_min_value = float(sys.argv[5].split("=")[1]) #-15
	pitch_max_value = float(sys.argv[6].split("=")[1]) #-15
	yaw_count = int(sys.argv[7].split("=")[1]) #1
	yaw_min_value = float(sys.argv[8].split("=")[1]) #-20.0
	yaw_max_value = float(sys.argv[9].split("=")[1]) #20.0
	number_of_perspectives = int(sys.argv[10].split("=")[1]) #20.0

	project_name = "6d_calibration_v7_90_degrees_hardline"

	np.set_printoptions(threshold=sys.maxsize)
except Exception as error_message:
	print(error_message)
	print("\nFailure to initialize arguments, moving on...\n")

def safe_division(n, d, default):
    return n / d if d else default

def how_much_ram_is_being_used():
	''' Memory usage in GB '''

	def get_size(bytes, suffix="B"):
	    """
	    Scale bytes to its proper format
	    e.g:
	        1253656 => '1.20MB'
	        1253656678 => '1.17GB'
	    """
	    factor = 1024
	    for unit in ["", "K", "M", "G", "T", "P"]:
	        if bytes < factor:
	            return f"{bytes:.2f}{unit}{suffix}"
	        bytes /= factor


	with open('/proc/self/status') as f:
		memusage = f.read().split('VmRSS:')[1].split('\n')[0][:-3]

	svmem = psutil.virtual_memory()
	print(f"\n\nRAM Total Virtual Use: {svmem.percent}%")
	swap = psutil.swap_memory()
	print(f"RAM Total Swap Use: {swap.percent}%")

	print("\n{:.2f} GB of RAM currently being used by this program\n\n".format(int(memusage.strip()) / (1000.0 * 1000.0)), flush=True)

try:
	start_time = time.time()

	robot_pitch_standard_deviation = 0.5
	robot_yaw_standard_deviation = 0.5
	robot_x_standard_deviation = 0.01
	robot_y_standard_deviation = 0.01
	robot_z_standard_deviation = 0.01

	total_views_to_see = radius_count * pitch_count * yaw_count #robot_pitch_count*robot_yaw_count*robot_x_count*robot_y_count*robot_z_count

	print("\n\nPreparing to see {} perspectives:".format(total_views_to_see))

	number_of_corner_points_to_measure = 8
	number_of_dimensions_per_point = 3
	number_of_dimensions_per_pose = 5
	scans_per_perspective = 1

	min_exposure_time = 1.0
	max_exposure_time = 1.5
	exposure_time_count = 2
	exposure_time_multipliers = np.linspace(min_exposure_time, max_exposure_time, exposure_time_count, endpoint=True)

	calibration_blocks = np.zeros((total_views_to_see * scans_per_perspective * exposure_time_count, number_of_corner_points_to_measure, number_of_dimensions_per_point))
	robot_scans_valid = np.zeros((total_views_to_see * scans_per_perspective * exposure_time_count))
	robot_poses = np.zeros((total_views_to_see * scans_per_perspective * exposure_time_count, number_of_dimensions_per_pose))

	scan_distance = radius_min * 1000.0
	views = 0
	scan_reset_frequency_for_robot = 10
	vibration_minimizing_delay = 40.0

except Exception as error_message:
	print(error_message)
	print("\nFailure to initialize globals, moving on...\n")

def save_to_log(robot_poses, calibration_blocks, scan_index):
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

try:
	radius_range = np.linspace(radius_min, radius_max, radius_count, endpoint=True)

	iterations = 0
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

				number_of_perspectives +=1
				iterations += 1

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

				print("({}) r={:.3f}m, r_xy={:.3f}m, pitch={:.3f}°, yaw {:.3f}°, x={:.3f}m, y={:.3f}m, z={:.3f}m".format(iterations, radius, xy_radius, pitch, yaw, x, y, z))

				continue

				how_much_ram_is_being_used()

				for scan_number_for_perspective in range(scans_per_perspective):

					x_mean = x
					y_mean = y
					z_mean = z
					pitch_mean = pitch
					yaw_mean = yaw

					for exposure_time_index, exposure_time_multiplier in enumerate(exposure_time_multipliers):
						try:

							if views % scan_reset_frequency_for_robot == 0:

								if views > 0:
									robot.die()
									del robot
									time.sleep(5.0) # a moment of silence

								robot = Robot()
								how_much_ram_is_being_used()


							views +=1
 
							if scan_number_for_perspective == 0 and exposure_time_index == 0:
								x = np.random.normal(loc=x_mean, scale=robot_x_standard_deviation)
								y = np.random.normal(loc=y_mean, scale=robot_y_standard_deviation)
								z = np.random.normal(loc=z_mean, scale=robot_z_standard_deviation)
								pitch = np.random.normal(loc=pitch_mean, scale=robot_pitch_standard_deviation)
								yaw = np.random.normal(loc=yaw_mean, scale=robot_yaw_standard_deviation)
								robot.move(x=x, y=y, z=z, pitch=pitch, yaw=yaw, sleep_for_dampening_vibration=0.0)

							if views == 1:
								print("Delaying {} seconds to dampen vibrations".format(vibration_minimizing_delay))
								time.sleep(vibration_minimizing_delay)
								points = robot.scan(distance = scan_distance, exposure_time_multiplier=exposure_time_multiplier)
								robot_poses[views-1, 0] = x
								robot_poses[views-1, 1] = y
								robot_poses[views-1, 2] = z
								robot_poses[views-1, 3] = pitch
								robot_poses[views-1, 4] = yaw
							else:
								processing_delay = 0.0
								start_processing_time = time.time()
								points.downsample(reduce_by_1_divided_by_n=3)
								point_cloud_file = points.preprocess_points_for_plane_finding()
								points.delete_ply_file()
								planes = PlaneFinder(path_to_point_cloud_file=point_cloud_file)
								xs, ys, zs = find_block_corners(point_cloud_filename=planes.point_cloud_filename, project_scan_name="{}_scan{}".format(project_name,views))
								if type(xs) != type(None):
									robot_scans_valid[views-2] = 1
									scan_distance = np.average(zs[0:8])
									print("Extracted scan distance of {:.3f}mm".format(scan_distance))
									calibration_blocks[views-2, :, 0] = xs[0:number_of_corner_points_to_measure]
									calibration_blocks[views-2, :, 1] = ys[0:number_of_corner_points_to_measure]
									calibration_blocks[views-2, :, 2] = zs[0:number_of_corner_points_to_measure]
									save_to_log(robot_poses, calibration_blocks, views-2)


								end_processing_time = time.time()
								processing_delay = end_processing_time - start_processing_time
								print("{:.3f} seconds in processing".format(processing_delay))
								if processing_delay < vibration_minimizing_delay and scan_number_for_perspective == 0 and exposure_time_index == 0:
									print("Delaying {} additional seconds to dampen vibrations".format(vibration_minimizing_delay-processing_delay))
									time.sleep(vibration_minimizing_delay - processing_delay)

								del points 
								points = robot.scan(distance = scan_distance, exposure_time_multiplier=exposure_time_multiplier)
								robot_poses[views-1, 0] = x
								robot_poses[views-1, 1] = y
								robot_poses[views-1, 2] = z
								robot_poses[views-1, 3] = pitch
								robot_poses[views-1, 4] = yaw

								del planes

							if views == total_views_to_see*scans_per_perspective*exposure_time_count and scan_number_for_perspective == scans_per_perspective - 1 and exposure_time_index == exposure_time_count - 1:
								points.downsample(reduce_by_1_divided_by_n=3)
								point_cloud_file = points.preprocess_points_for_plane_finding()
								points.delete_ply_file()
								planes = PlaneFinder(path_to_point_cloud_file=point_cloud_file)
								xs, ys, zs = find_block_corners(point_cloud_filename=planes.point_cloud_filename, project_scan_name="{}_scan{}".format(project_name,views))
								if type(xs) != type(None):
									robot_scans_valid[views-1] = 1
									scan_distance = np.average(zs[0:8])
									print("Extracted scan distance of {:.3f}mm".format(scan_distance))
									calibration_blocks[views-1, :, 0] = xs[0:number_of_corner_points_to_measure]
									calibration_blocks[views-1, :, 1] = ys[0:number_of_corner_points_to_measure]
									calibration_blocks[views-1, :, 2] = zs[0:number_of_corner_points_to_measure]
									save_to_log(robot_poses, calibration_blocks, views-1)

								del planes

							print("\nPERSPECTIVE {}, TRIAL {}, EXPOSURE TIME MULTIPLIER {:.1f}, BATCH TOTAL {}: x: {:.3f}mm, y: {:.3f}mm, z: {:.3f}mm, pitch: {:.3f} degrees, yaw: {:.3f} degrees".format(number_of_perspectives, scan_number_for_perspective + 1, exposure_time_multiplier, views, x,y,z,pitch,yaw), flush=True)

						except Exception as error_message:
							print(error_message)
							print("\nAfter above error, moving on to next scan...\n")
	os._exit(0)
	end_time = time.time()
	print("Finished in {:.2f} minutes".format((end_time-start_time)/60.0))

except Exception as error_message:
	print(error_message)
	print("\nFailure during perspective calculation, moving on to next batch...\n")

try:
	robot.die()
	del robot
	os._exit(0)
except Exception as error_message:
	print(error_message)
	print("\nFailure to kill robot, moving on...\n")