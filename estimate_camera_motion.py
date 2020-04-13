import math
import random
import numpy as np
import sys
from os import listdir
import scipy.optimize as optimize

class Point():
    def __init__(self, x, y, z, h=None, v=None):
        self.x = x
        self.y = y
        self.z = z
        self.xyz = [self.x, self.y, self.z]
        self.h = h # pixel horizontal position
        self.v = v # pixel vertical position
        self.validate()
        
    def validate(self):
        if self.x == 0.0 and self.y == 0.0 and self.z == 0.0:
            self.valid = False
        else:
            self.valid = True
            
    def distance(self, point):
        return math.sqrt((self.x - point.x)**2 + (self.y - point.y)**2 + (self.z - point.z)**2)


class Vector():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Perspective():
    def __init__(self, robot_x, robot_y, robot_z, robot_pitch, robot_yaw):
        self.x = robot_x
        self.y = robot_y
        self.z = robot_z
        self.pitch = robot_pitch
        self.yaw = robot_yaw

class Block():
	def __init__(self, points, perspective):
		self.p0 = points[0]
		self.p1 = points[1]
		self.p2 = points[2]
		self.p3 = points[3]
		self.p4 = points[4]
		self.p5 = points[5]
		self.p6 = points[6]
		self.p7 = points[7]
		self.view = perspective

	def distance(self, another_block):
		p0_distance = (self.p0 - another_block.p0)**2
		p1_distance = (self.p1 - another_block.p1)**2
		p2_distance = (self.p2 - another_block.p2)**2
		p3_distance = (self.p3 - another_block.p3)**2 
		return p0_distance + p1_distance + p2_distance + p3_distance

	def optimize_transform_to_another_block(self, another_block):
        initial_error = self.distance(another_block)
        
        x, y, z = self.points[:, 0], self.points[:, 1], self.points[:, 2]
        initial_guess = [self.a, self.b, self.c, self.d] # use previous plane equation as initial guess
        
        def model(params, xyz):
            a, b, c, d = params
            x, y, z = xyz
            length_squared = a**2 + b**2 + c**2
            if length_squared > 0:
                return ((a * x + b * y + c * z + d) ** 2 / length_squared).sum() 
            else:
                return 1000.0

        def unit_length(params):
            a, b, c, d = params
            return a**2 + b**2 + c**2 - 1

        # constrain the vector perpendicular to the plane be of unit length
        cons = ({'type': 'eq', 'fun': unit_length})
        sol = optimize.minimize(model, initial_guess, args=[x, y, z], constraints=cons)
        self.a = tuple(sol.x)[0]
        self.b = tuple(sol.x)[1]
        self.c = tuple(sol.x)[2]
        self.d = tuple(sol.x)[3]
        return tuple(sol.x)
    


point_cloud_directory = sys.argv[1]

all_corner_points_files = [f for f in listdir(point_cloud_directory) if "_corner_points.txt" in f]
print("{} valid sets of corner points found".format(len(all_corner_points_files)))

blocks = []

for corner_points_file in all_corner_points_files:
	block_identifier = corner_points_file.replace("_corner_points.txt", "")

	robot_x = float(block_identifier.split("x_")[1].split("_y")[0])
	robot_y = float(block_identifier.split("y_")[1].split("_z")[0])
	robot_z = float(block_identifier.split("z_")[1].split("_theta")[0])
	robot_pitch = float(block_identifier.split("theta_")[1].split("_phi")[0])
	robot_yaw = float(block_identifier.split("phi_")[1])

	camera_perspective = Perspective(robot_x, robot_y, robot_z, robot_pitch, robot_yaw)

	with open("{}/{}".format(point_cloud_directory, corner_points_file), "r") as lines:
		points = []
		for line in lines:
			x = float(line.split(",")[0])
			y = float(line.split(",")[1])
			z = float(line.split(",")[2])
			points.append(Point(x,y,z))

	b = Block(points=points, perspective=camera_perspective)
	blocks.append(b)


# add index of scan to start of filename immediately after project name, e.g. "full-calibration_1_x=..."
# then sort below by that
for b in blocks:
	print("\nROBOT: (x={:.2f},y={:.2f},z={:.2f},pitch={:.2f},yaw={:.2f})".format(b.view.x, b.view.y, b.view.z, b.view.pitch, b.view.yaw))
	print("BLOCK: P0=({:.2f},{:.2f},{:.2f}) P1=({:.2f},{:.2f},{:.2f}) P2=({:.2f},{:.2f},{:.2f}) P3=({:.2f},{:.2f},{:.2f})".format(b.p0.x, b.p0.y, b.p0.z, b.p1.x, b.p1.y, b.p1.z, b.p2.x, b.p2.y, b.p2.z, b.p3.x, b.p3.y, b.p3.z))

b1 = blocks[2]
b2 = blocks[3]

# x_translation_initial_estimate = b2.x - b1.x
# y_translation_initial_estimate = b2.y - b1.y
# z_translation_initial_estimate = b2.z - b1.z

