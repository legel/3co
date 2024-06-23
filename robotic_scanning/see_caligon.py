import sys
from commander import *
from robotics import *
from math import radians, sin, cos
import time
import random

x = 0.95
y = -0.2
z = 0.6
pitch = -80
yaw = 0

print("\n\n\nMoving robot back to initial position\n\n\n")
commander.move({'x': x, 'y': y, 'z': z, 'pitch': pitch, 'yaw': yaw})

focus_distance = 800
trials_per_batch = 30

min_pitch = -90
max_pitch = -70
min_yaw = -175
max_yaw = 175
min_x = 0.45
max_x = 1.45
min_y = -0.65
max_y = 0.25
min_z = 0.55
max_z = 0.65

unit_vector_distance = 0.15
unit_angle_distance = 15

views = int(sys.argv[1])

random.seed(views)

print("\n\n\nFocusing robot at {}mm\n\n\n".format(focus_distance))
robot = Robot(distance=focus_distance)

for sample in range(trials_per_batch):
	random_x = random.uniform(min_x, max_x)
	random_y = random.uniform(min_y, max_y)
	random_z = random.uniform(min_z, max_z)
	random_pitch = random.uniform(min_pitch, max_pitch)
	random_yaw = random.uniform(min_yaw, max_yaw)

	delta_x = random_x - x
	delta_y = random_y - y
	delta_z = random_z - z
	delta_pitch = random_pitch - pitch
	delta_yaw = random_yaw - yaw

	delta_x_scaled = (delta_x / (abs(delta_x) + abs(delta_y) + abs(delta_z))) * unit_vector_distance
	delta_y_scaled = (delta_y / (abs(delta_x) + abs(delta_y) + abs(delta_z))) * unit_vector_distance
	delta_z_scaled = (delta_z / (abs(delta_x) + abs(delta_y) + abs(delta_z))) * unit_vector_distance

	delta_pitch = random_pitch - pitch
	delta_yaw = random_yaw - yaw

	delta_pitch_scaled = (delta_pitch / (abs(delta_pitch) + abs(delta_yaw))) * unit_angle_distance
	delta_yaw_scaled = (delta_yaw / (abs(delta_pitch) + abs(delta_yaw))) * unit_angle_distance

	x = x + delta_x_scaled
	y = y + delta_y_scaled
	z = z + delta_z_scaled
	pitch = pitch + delta_pitch_scaled
	yaw = yaw + delta_yaw_scaled

	if x > max_x:
		x = max_x
	if x < min_x:
		x = min_x
	if y > max_y:
		y = max_y
	if y < min_y:
		y = min_y
	if z > max_z:
		z = max_z
	if z < min_z:
		z = min_z
	if pitch > max_pitch:
		pitch = max_pitch
	if pitch < min_pitch:
		pitch = min_pitch
	if yaw > max_yaw:
		yaw = max_yaw
	if yaw < min_yaw:
		yaw = min_yaw

	print("VIEW {}, X {:.3f}m, Y {:.3f}m, Z {:.3f}m, YAW {:.2f}°, PITCH {:.2f}°".format(views, x, y, z, yaw, pitch))
	commander.move({'x': x, 'y': y, 'z': z, 'pitch': pitch, 'yaw': yaw})

	print("Sleeping off those nasty vibrations")
	time.sleep(15.0)

	robot.scan(process_point_cloud=True, project_name="reconstruction_v1_view{}_x{:.3f}_y{:.3f}_z{:.3f}_yaw{:.2f}_pitch{:.2f}".format(views, x, y, z, yaw, pitch)) 

	views += 1