#from robotics import *
#from scanning import *
#from geometry import *
#from commander import *
import numpy as np
#import pickle
#import pandas as pd
#import time
#from pprint import pprint
import math
#from os import listdir, path, getcwd
import sys

def roll_correction(roll, x, y, z):
	# roll an angle in degrees
	# x, y, z are each numpy arrays of float values
	roll = math.radians(roll)
	roll_origin = np.arctan2(x, y)
	radius = np.sqrt(x**2 + y**2)
	x = radius * np.sin(roll + roll_origin)
	y = radius * np.cos(roll + roll_origin)
	z = z
	return x,y,z

def pitch_correction(pitch, x, y, z):
	# pitch an angle in degrees
	# x, y, z are each numpy arrays of float values
	pitch = math.radians(pitch)
	pitch_origin = np.arctan2(x, z)
	radius = np.sqrt(x**2 + z**2)
	x = radius * np.sin(pitch + pitch_origin)
	y = y
	z = radius * np.cos(pitch + pitch_origin)
	return x,y,z
	
def yaw_correction(yaw, x, y, z):
	# yaw an angle in degrees
	# x, y, z are each numpy arrays of float values
	yaw = math.radians(yaw)
	yaw_origin = np.arctan2(z, y)
	radius = np.sqrt(y**2 + z**2)
	x = x
	y = radius * np.cos(yaw + yaw_origin)
	z = radius * np.sin(yaw + yaw_origin)
	return x,y,z


def calculate_displacement(xs,ys,zs):
	x_difference = xs[1] - xs[0]
	y_difference = ys[1] - ys[0]
	z_difference = zs[1] - zs[0]
	return math.sqrt(x_difference**2 + y_difference**2 + z_difference**2)


### 

### NOTES: 
### 1. The scanner pitch measurement using a calibration plane may be legitimate, but it requires definite identification of the proper plane.
### 2. The scanner roll measurement and the scanner yaw measurement is a sinusoidal function of robot pitch, and probably impossible to isolate outside of this fact
### 3. The scanner yaw measurement is also influenced by robot yaw measurement sinusoidally
### 4. Probably for experimentation stability purposes it is best to move to a 1 or 2 block setup, where 3 planes of each block are identified in each scan; and therefore 8 points on the block
### 5. To speed this up, consider how the planes change homogeneously at high detail, and therefore, consider downsampling the entire point cloud by skipping e.g. 19 out of 20 pixels in every row and column
### 6. Theoretically, if a coordinate system is fixed and an object is moving in it, and at least 3 points are localized, then 6-D canonical transformation can be localized with ease
### 7. "...Euclidian rotation error in centimeters" ... am I overdoing this? 
### 8. Most likely we need to pose the problem as a minimization one where the (x,y,z,pitch,yaw,roll) of the scanner are estimated as a function of the robot (pitch, yaw) and motion of calibration points
### 9. This is not so bad - just find a way to normalize the state space for smooth linear regression
### 10. Godspeed


### https://arxiv.org/abs/2002.10838
### 
###


# start_time = time.time()
# np.set_printoptions(threshold=sys.maxsize)

# calibrated_pitches = []
# calibration_runs = 1

# robot = Robot()

# #calibrate('yaw')
# #calibrate('pitch')

# start_yaw = 20
# end_yaw = -180
# yaw_increment = -20
# start_pitch = -90
# end_pitch = -85
# pitch_increment = -1
# #move({'x': 0.0, 'y': 0.0, 'z': 0.45, 'pitch': start_pitch, 'yaw': start_yaw})

# for calibration_run in range(calibration_runs):

# 	min_block_distance = 390
# 	max_block_distance = 450

# 	min_marker_distance = 450
# 	max_marker_distance = 470

# 	min_block_distance

# 	target_distance = 475

# 	robot_yaws = [0] #[angle for angle in range(start_yaw, end_yaw, yaw_increment)]
# 	robot_pitches = [pitch for pitch in range(-85, -60, 1)]

# 	print("Robot yaw angles: {}".format(robot_yaws))
# 	print("Robot pitch angles: {}".format(robot_pitches))

# 	n_yaws = len(robot_yaws)
# 	n_pitches = len(robot_pitches)

# 	scanner_yaws = np.zeros((n_yaws, n_pitches))
# 	scanner_pitches = np.zeros((n_yaws, n_pitches))
# 	scanner_rolls = np.zeros((n_yaws, n_pitches))

# 	scanner_xs = np.zeros((n_yaws, n_pitches))
# 	scanner_ys = np.zeros((n_yaws, n_pitches))
# 	scanner_zs = np.zeros((n_yaws, n_pitches))

# 	calibration_block_displacements = np.zeros((n_yaws, n_pitches))

# 	x1s = np.zeros((n_yaws, n_pitches))
# 	y1s = np.zeros((n_yaws, n_pitches))
# 	z1s = np.zeros((n_yaws, n_pitches))

# 	x2s = np.zeros((n_yaws, n_pitches))
# 	y2s = np.zeros((n_yaws, n_pitches))
# 	z2s = np.zeros((n_yaws, n_pitches))

# 	scanner_yaw_origin = 0.0
# 	scanner_pitch_origin = 0.0
# 	scanner_roll_origin = 0.0
# 	for robot_yaw_index, robot_yaw in enumerate(robot_yaws):
# 		for robot_pitch_index, robot_pitch in enumerate(robot_pitches):

# 			robot.move(x=0.15, y=0.0, z=0.45, pitch=robot_pitch, yaw=robot_yaw, sleep_for_dampening_vibration=2.5)
# 			block_1_points = robot.scan(distance = target_distance)
# 			block_2_points = block_1_points.copy()

# 			block_1_points.filter(min_z=min_block_distance, max_z=max_block_distance)
# 			ply_1 = block_1_points.preprocess_points_for_plane_finding()
# 			block_1_top_plane = PlaneFinder(path_to_point_cloud_file=ply_1).best_planes[0]

# 			scanner_pitch_1 = math.degrees(math.atan(block_1_top_plane.a))
# 			scanner_yaw_1 = math.degrees(math.atan(block_1_top_plane.b))

# 			block_2_points.filter(min_z=min_marker_distance, max_z=max_marker_distance)
# 			ply_2 = block_2_points.preprocess_points_for_plane_finding()
# 			block_2_top_plane = PlaneFinder(path_to_point_cloud_file=ply_2).best_planes[0]

# 			scanner_pitch_2 = math.degrees(math.atan(block_2_top_plane.a))
# 			scanner_yaw_2 = math.degrees(math.atan(block_2_top_plane.b))

# 			block_1_center_point = block_1_top_plane.average_point() # requires entire surface to be visible for all scans!
# 			block_2_center_point = block_2_top_plane.average_point()

# 			block_1_xs = block_1_top_plane.points[:,0]
# 			block_1_ys = block_1_top_plane.points[:,1]
# 			block_1_zs = block_1_top_plane.points[:,2]

# 			distances = np.sqrt((block_1_xs - block_2_center_point.x)**2 + (block_1_ys - block_2_center_point.y)**2 + (block_1_zs - block_2_center_point.z)**2)
# 			minimum_distance_point_index = np.where(distances == np.amin(distances))
# 			block_1_corner_point = Point(block_1_xs[minimum_distance_point_index][0], block_1_ys[minimum_distance_point_index][0], block_1_zs[minimum_distance_point_index][0])

# 			a = block_1_center_point
# 			b = block_1_corner_point
# 			# c = Point(a.x, b.y, a.z)
# 			# d = Point(a.x, b.y, b.z)
# 			# e = Point(b.x, b.y, a.z)
# 			# f = Point(0.0, a.y, 0.0)
# 			# g = Point(0.0, a.y, a.z)
# 			# h = Point(0.0, 0.0, 0.0)

# 			print("BLOCK 1 CENTER: ({:.3f},{:.3f},{:.3f})".format(a.x, a.y, a.z))
# 			print("BLOCK 1 CORNER: ({:.3f},{:.3f},{:.3f})".format(b.x, b.y, b.z))
# 			print("BLOCK 1 CENTER-CORNER DISPLACEMENT: {:.3f}mm".format(a.distance(b)))

# 			xs = np.asarray([block_1_center_point.x, block_1_corner_point.x])
# 			ys = np.asarray([block_1_center_point.y, block_1_corner_point.y])
# 			zs = np.asarray([block_1_center_point.z, block_1_corner_point.z])

# 			displacement_original_measured = calculate_displacement(xs,ys,zs)

# 			print("\n\n(X,Y,Z) of BLOCK 1 original measured: ({:.3f}, {:.3f}, {:.3f})".format(xs[0],ys[0],zs[0]))
# 			print("(X,Y,Z) of BLOCK 2 original measured: ({:.3f}, {:.3f}, {:.3f})".format(xs[1],ys[1],zs[1]))
# 			print("Block 1 - Block 2 3D displacement original measured: {:.6f}mm".format(displacement_original_measured))


# 			scanner_roll = math.degrees(np.arctan2(xs[1] - xs[0], ys[1] - ys[0]))
# 			scanner_yaw = (scanner_yaw_1 + scanner_yaw_2) / 2.0  #math.degrees(np.arctan2(zs[1] - zs[0], ys[1] - ys[0]))
# 			scanner_pitch = (scanner_pitch_1 + scanner_pitch_2) / 2.0 # math.degrees(np.arctan2(xs[1] - xs[0], zs[1] - zs[0]))

# 			xs, ys, zs = roll_correction(roll=scanner_roll, x=xs, y=ys, z=zs)
# 			displacement_after_roll_correction = calculate_displacement(xs,ys,zs)

# 			print("\n\n(X,Y,Z) of BLOCK 1 after correcting scanner roll by {:.3f} degrees: ({:.3f}, {:.3f}, {:.3f})".format(scanner_roll, xs[0],ys[0],zs[0]))
# 			print("(X,Y,Z) of BLOCK 2 after correcting scanner roll by {:.3f} degrees: ({:.3f}, {:.3f}, {:.3f})".format(scanner_roll, xs[1],ys[1],zs[1]))
# 			print("Block 1 - Block 2 3D displacement after scanner roll correction of: {:.6f}mm".format(displacement_after_roll_correction))


# 			xs, ys, zs = yaw_correction(yaw=scanner_yaw, x=xs, y=ys, z=zs)
# 			displacement_after_yaw_correction = calculate_displacement(xs,ys,zs)

# 			print("\n\n(X,Y,Z) of BLOCK 1 after correcting scanner yaw by {:.3f} degrees: ({:.3f}, {:.3f}, {:.3f})".format(scanner_yaw, xs[0],ys[0],zs[0]))
# 			print("(X,Y,Z) of BLOCK 2 after correcting scanner yaw by {:.3f} degrees: ({:.3f}, {:.3f}, {:.3f})".format(scanner_yaw, xs[1],ys[1],zs[1]))
# 			print("Block 1 - Block 2 3D displacement after scanner yaw correction: {:.6f}mm\n\n".format(displacement_after_yaw_correction))

# 			xs, ys, zs = pitch_correction(pitch=scanner_pitch, x=xs, y=ys, z=zs)
# 			displacement_after_pitch_correction = calculate_displacement(xs,ys,zs)

# 			print("\n\n(X,Y,Z) of BLOCK 1 after correcting scanner pitch by {:.3f} degrees: ({:.3f}, {:.3f}, {:.3f})".format(scanner_pitch, xs[0],ys[0],zs[0]))
# 			print("(X,Y,Z) of BLOCK 2 after correcting scanner pitch by {:.3f} degrees: ({:.3f}, {:.3f}, {:.3f})".format(scanner_pitch, xs[1],ys[1],zs[1]))
# 			print("Block 1 - Block 2 3D displacement after scanner pitch correction: {:.6f}mm".format(displacement_after_pitch_correction))

# 			scanner_pitches[robot_yaw_index, robot_pitch_index] = scanner_pitch
# 			scanner_rolls[robot_yaw_index, robot_pitch_index] = scanner_roll
# 			scanner_yaws[robot_yaw_index, robot_pitch_index] = scanner_yaw

# 			x1s[robot_yaw_index, robot_pitch_index] = xs[0]
# 			y1s[robot_yaw_index, robot_pitch_index] = ys[0]
# 			z1s[robot_yaw_index, robot_pitch_index] = zs[0]

# 			x2s[robot_yaw_index, robot_pitch_index] = xs[0]
# 			y2s[robot_yaw_index, robot_pitch_index] = ys[0]
# 			z2s[robot_yaw_index, robot_pitch_index] = zs[0]

# 			scanner_xs[robot_yaw_index, robot_pitch_index] = -1 * xs[0] # where the first calibration point is permanently defined as (0,0,0)
# 			scanner_ys[robot_yaw_index, robot_pitch_index] = -1 * ys[0] 
# 			scanner_zs[robot_yaw_index, robot_pitch_index] = -1 * zs[0] 

# 			del block_1_points, block_2_points, block_1_top_plane, block_2_top_plane, distances, block_1_xs, block_1_ys, block_1_zs

# robot.die()


# print("\n\nScanner Roll vs. Robot Yaw & Robot Pitch")
# robot_pitch_presentation = ["ROBOT YAW ANGLE"]
# for robot_pitch in robot_pitches:
# 		robot_pitch_presentation.append("{:.1f}° ROBOT PITCH".format(robot_pitch))
# print("\t".join(robot_pitch_presentation))

# for robot_yaw_index, robot_yaw in enumerate(robot_yaws):
# 	scanner_roll_presentation = ["{:.3f}".format(robot_yaw)]
# 	for robot_pitch_index, robot_pitch in enumerate(robot_pitches):
# 		scanner_roll_presentation.append("{:.3f}".format(scanner_rolls[robot_yaw_index, robot_pitch_index]))
# 	print("\t".join(scanner_roll_presentation))


# print("\n\nScanner Pitch vs. Robot Yaw & Robot Pitch")
# robot_pitch_presentation = ["ROBOT YAW ANGLE"]
# for robot_pitch in robot_pitches:
# 		robot_pitch_presentation.append("{:.1f}° ROBOT PITCH".format(robot_pitch))
# print("\t".join(robot_pitch_presentation))

# for robot_yaw_index, robot_yaw in enumerate(robot_yaws):
# 	scanner_pitch_presentation = ["{:.3f}".format(robot_yaw)]
# 	for robot_pitch_index, robot_pitch in enumerate(robot_pitches):
# 		scanner_pitch_presentation.append("{:.3f}".format(scanner_pitches[robot_yaw_index, robot_pitch_index]))
# 	print("\t".join(scanner_pitch_presentation))


# print("\n\nScanner Yaw vs. Robot Yaw & Robot Pitch")
# robot_pitch_presentation = ["ROBOT YAW ANGLE"]
# for robot_pitch in robot_pitches:
# 		robot_pitch_presentation.append("{:.1f}° ROBOT PITCH".format(robot_pitch))
# print("\t".join(robot_pitch_presentation))

# for robot_yaw_index, robot_yaw in enumerate(robot_yaws):
# 	scanner_yaw_presentation = ["{:.3f}".format(robot_yaw)]
# 	for robot_pitch_index, robot_pitch in enumerate(robot_pitches):
# 		scanner_yaw_presentation.append("{:.3f}".format(scanner_yaws[robot_yaw_index, robot_pitch_index]))
# 	print("\t".join(scanner_yaw_presentation))


# print("\n\nScanner X vs. Robot Yaw & Robot Pitch")
# robot_pitch_presentation = ["ROBOT YAW ANGLE"]
# for robot_pitch in robot_pitches:
# 		robot_pitch_presentation.append("{:.1f}° ROBOT PITCH".format(robot_pitch))
# print("\t".join(robot_pitch_presentation))

# for robot_yaw_index, robot_yaw in enumerate(robot_yaws):
# 	x_presentation = ["{:.3f}".format(robot_yaw)]
# 	for robot_pitch_index, robot_pitch in enumerate(robot_pitches):
# 		x_presentation.append("{:.3f}".format(scanner_xs[robot_yaw_index, robot_pitch_index]))
# 	print("\t".join(x_presentation))


# print("\n\nScanner Y vs. Robot Yaw & Robot Pitch")
# robot_pitch_presentation = ["ROBOT YAW ANGLE"]
# for robot_pitch in robot_pitches:
# 		robot_pitch_presentation.append("{:.1f}° ROBOT PITCH".format(robot_pitch))
# print("\t".join(robot_pitch_presentation))

# for robot_yaw_index, robot_yaw in enumerate(robot_yaws):
# 	y_presentation = ["{:.3f}".format(robot_yaw)]
# 	for robot_pitch_index, robot_pitch in enumerate(robot_pitches):
# 		y_presentation.append("{:.3f}".format(scanner_ys[robot_yaw_index, robot_pitch_index]))
# 	print("\t".join(y_presentation))


# print("\n\nScanner Z vs. Robot Yaw & Robot Pitch")
# robot_pitch_presentation = ["ROBOT YAW ANGLE"]
# for robot_pitch in robot_pitches:
# 		robot_pitch_presentation.append("{:.1f}° ROBOT PITCH".format(robot_pitch))
# print("\t".join(robot_pitch_presentation))

# for robot_yaw_index, robot_yaw in enumerate(robot_yaws):
# 	z_presentation = ["{:.3f}".format(robot_yaw)]
# 	for robot_pitch_index, robot_pitch in enumerate(robot_pitches):
# 		z_presentation.append("{:.3f}".format(scanner_zs[robot_yaw_index, robot_pitch_index]))
# 	print("\t".join(z_presentation))

# end_time = time.time()
# print("\n\nInvested {:.1f} minutes into this".format((end_time-start_time)/60.0))