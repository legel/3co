from robotics import *
from ram import *
import numpy as np
from aws_file_uploader import *

min_exposure_time = 0.5
max_exposure_time = 2.0
exposure_time_count = 3
scans_per_pose = 1

# see charuco_calibration_orchestrator.py
try:
	robot_poses_unparsed = sys.argv[1].split("=")[1].split("SPACE")
	number_of_poses_to_scan = len(robot_poses_unparsed)
	robot_poses = []
	for robot_pose in robot_poses_unparsed:
		x,y,z,pitch,yaw = robot_pose.split("_")
		robot_poses.append([float(x),float(y),float(z),float(pitch),float(yaw)])
	starting_pose_index = int(sys.argv[2].split("=")[1])
	starting_scan_distance = float(sys.argv[3].split("=")[1]) * 1000
	project_name = sys.argv[4].split("=")[1]

except Exception as error_message:
	print(error_message)
	print("\nFailure to initialize arguments, moving on...\n")

np.set_printoptions(threshold=sys.maxsize, suppress=True, linewidth= 1000)

def save_to_log(robot_pose, scan_identifier):
	try:
		print("Saving output of scan {} to file {}".format(scan_identifier, "{}_pose_to_scan.tsv".format(project_name)), flush=True)
		with open("{}_pose_to_scan.tsv".format(project_name), "a") as outputs:
			#outputs.write("ROBOT X\tROBOT Y\tROBOT Z\tROBOT PITCH\tROBOT YAW\tPX1\tPY1\tPZ1\tPX2\tPY2\tPZ2\tPX3\tPY3\tPZ3\tPX4\tPY4\tPZ4\n")
			robot_x = robot_pose[0]
			robot_y = robot_pose[1]
			robot_z = robot_pose[2]
			robot_pitch = robot_pose[3]
			robot_yaw = robot_pose[4]
			outputs.write("{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{}\n".format(robot_x, robot_y, robot_z, robot_pitch, robot_yaw, scan_identifier))
	except Exception as error_message:
		print(error_message)
		print("\nFailure to save output, moving on...\n")


def scan_poses(robot_poses, scans_per_pose, min_exposure_time, max_exposure_time, exposure_time_count, starting_pose_index, starting_scan_distance):
	current_pose_index = starting_pose_index 
	scans_in_this_batch = 0
	scan_distance = starting_scan_distance
	scan_reset_frequency_for_robot = 10
	vibration_minimizing_delay = 20.0

	exposure_time_multipliers = np.linspace(min_exposure_time, max_exposure_time, exposure_time_count, endpoint=True)
	total_scans_to_take = number_of_poses_to_scan * scans_per_pose * exposure_time_count

	#how_much_ram_is_being_used()

	for robot_pose in robot_poses:
		x = robot_pose[0]
		y = robot_pose[1]
		z = robot_pose[2]
		pitch = robot_pose[3]
		yaw = robot_pose[4]
		for scan_number_for_pose in range(scans_per_pose):
			for exposure_time_index, exposure_time_multiplier in enumerate(exposure_time_multipliers):
				#try:
				if scans_in_this_batch % scan_reset_frequency_for_robot == 0:
					if scans_in_this_batch > 0:
						robot.die()
						del robot
					robot = Robot() #x=0.0, y=0.0, z=0.5, pitch=-90, yaw=0)
					how_much_ram_is_being_used()

				scans_in_this_batch +=1

				if scan_number_for_pose == 0 and exposure_time_index == 0:
					#move({'x': x, 'y': y, 'z': z, 'pitch': pitch, 'yaw': yaw})
					robot.move(x=x, y=y, z=z, pitch=pitch, yaw=yaw, sleep_for_dampening_vibration=0.0)
					print("Delaying {} seconds to dampen vibrations".format(vibration_minimizing_delay))
					time.sleep(vibration_minimizing_delay)

				print("PERSPECTIVE {}, TRIAL {}, EXPOSURE TIME MULTIPLIER {:.1f}, BATCH TOTAL {}: x: {:.3f}mm, y: {:.3f}mm, z: {:.3f}mm, pitch: {:.3f} degrees, yaw: {:.3f} degrees".format(current_pose_index, scan_number_for_pose + 1, exposure_time_multiplier, scans_in_this_batch, x,y,z,pitch,yaw), flush=True)

				scan_identifier = "{}_{}".format(project_name, current_pose_index*scans_per_pose * exposure_time_count + exposure_time_count*scan_number_for_pose + exposure_time_index)
				points = robot.scan(distance = scan_distance, exposure_time_multiplier=exposure_time_multiplier, dual_exposure_mode=True, dual_exposure_multiplier=4.0, project_name=scan_identifier)
				points.delete_ply_file()
				if len(points.z) > 0:
					scan_distance = np.average(points.z)
					print("Updating scan distance to average distance in FOV: {:.3f}mm".format(scan_distance))

					save_to_log([x,y,z,pitch,yaw], scan_identifier)
					upload_file(file_name="{}.png".format(scan_identifier), bucket="3co".format(project_name), upload_path="{}/{}".format(project_name, "{}.png".format(scan_identifier)))
					upload_file(file_name="{}.csv".format(scan_identifier), bucket="3co".format(project_name), upload_path="{}/{}".format(project_name, "{}.csv".format(scan_identifier)))
					upload_file(file_name="{}_pose_to_scan.tsv".format(project_name), bucket="3co".format(project_name), upload_path="{}_pose_to_scan.tsv".format(project_name))

				del points

				# except Exception as error_message:
				# 	print(error_message)
				# 	print("\nAfter above error, moving on to next scan...\n")
		current_pose_index += 1

scan_poses(robot_poses=robot_poses, scans_per_pose=scans_per_pose, min_exposure_time=min_exposure_time, max_exposure_time=max_exposure_time, exposure_time_count=exposure_time_count, starting_pose_index=starting_pose_index, starting_scan_distance=starting_scan_distance)