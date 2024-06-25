from robotics import *
from scanning import PointCloud
import numpy as np
import pickle
import pandas as pd
import time
from pprint import pprint
import math
import sys

start_time = time.time()

robot = Robot()
#robot.calibrate()

target_distance = 465
measurements_to_make = 5
#points = PointCloud(filename="1586123676432.ply")

standard_deviation_history = {}
axes_to_measure = ['yaw']

#robot.calibrate_axis('x')
#robot.move(x=0.0, y=0.0, z=0.65, pitch=-90.000, yaw=-90.000)

for axis in axes_to_measure:
	x_centers = []
	y_centers = []
	z_centers = []
	standard_deviation_history[axis] = {}
	for experiment in range(measurements_to_make):
		robot.calibrate_axis('yaw')
		robot.move(yaw=0)
		time.sleep(10.0)

		points = robot.scan(distance = target_distance)
		#points.filter(min_z=min_block_distance, max_z=max_block_distance)
		#print("{:,} points of the block top surface".format(len(points.x)))

		x_offset = np.average(points.x)
		y_offset = np.average(points.y)
		z_offset = np.average(points.z)

		print("\n\nEstimated block center point (relative to scanner): x = {:.3f}, y = {:.3f}, z = {:.3f}\n\n".format(x_offset, y_offset, z_offset))
		x_centers.append(x_offset)
		y_centers.append(y_offset)
		z_centers.append(z_offset)

	x_std = np.std(x_centers)
	y_std = np.std(y_centers)
	z_std = np.std(z_centers)
	three_dim_std = math.sqrt(x_std**2 + y_std**2 + z_std**2)

	standard_deviation_history[axis]['x'] = x_std
	standard_deviation_history[axis]['y'] = y_std
	standard_deviation_history[axis]['z'] = z_std
	standard_deviation_history[axis]['3D'] = three_dim_std

	print("\n\nFor {}-axis, standard deviations in 3D measurement is {:.3f}mm\nx = {:.3f}, y = {:.3f}, z = {:.3f}\n\n".format(axis, three_dim_std, x_std, y_std, z_std))

pprint(standard_deviation_history)
end_time = time.time()
print("({:.1f} seconds)".format(end_time-start_time))