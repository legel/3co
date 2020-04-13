import math
from PIL import Image
from scipy.spatial import ConvexHull
from pprint import pprint
import random
import logging
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sys
import ray
import multiprocessing
from itertools import zip_longest

def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

cpus = multiprocessing.cpu_count() 
print("\nLaunching 8 corner point recognition algorithm on {} CPUs".format(cpus))
ray.init(num_cpus=cpus, logging_level=logging.WARNING)

mm_per_inch = 25.4

rows = 2280 # vertical pixels from scanner
columns = 1824 # horizontal pixels from scanner

point_cloud_directory = "/".join(sys.argv[1].split("/")[0:-1])
point_cloud_filename = sys.argv[1].split("/")[-1]
point_cloud_identifier = point_cloud_filename.replace("_hvxyz.csv", "")
point_cloud_filename = "{}/{}_plane_data.csv".format(point_cloud_directory, point_cloud_identifier)

#timestamp = "1582652957"
#planes_visualization = Image.open("planes_{}_abcd.png".format(timestamp))
#planes_visualization_pixels = planes_visualization.load()

class Vector():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

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

        
class Plane():
    def __init__(self, a, b, c, d):
    	self.a = a
    	self.b = b 
    	self.c = c
    	self.d = d 
    	self.points = []

points_of_planes = [[] for i in range(3)]

print("\nReading plane data from {}".format(point_cloud_identifier))

all_planes = []
total_planes = 0
with open("{}".format(point_cloud_filename), "r") as lines:
	for line in lines:
		if "PLANE" in line:
			plane_index = total_planes #int(line.split("PLANE ")[1].split(":")[0])
			a = float(line.split(": a=")[1].split(",b=")[0])
			b = float(line.split("b=")[1].split(",c=")[0])
			c = float(line.split("c=")[1].split(",d=")[0])
			d = float(line.split("d=")[1].split("\n")[0])
			print("PLANE {}: a={:.4f}, b={:.4f}, c={:.4f}, d={:.4f}".format(total_planes, a, b ,c, d))
			total_planes += 1
			all_planes.append(Plane(a=a, b=b, c=c, d=d))
		elif "," in line:
			h = int(line.rstrip("\n").split(",")[0])
			v = int(line.rstrip("\n").split(",")[1])
			x = float(line.rstrip("\n").split(",")[2])
			y = float(line.rstrip("\n").split(",")[3])
			z = float(line.rstrip("\n").split(",")[4])
			p = Point(x=x, y=y, z=z, h=h, v=v)
			points_of_planes[plane_index].append(p)

for i, points_of_plane in enumerate(points_of_planes):
	try:
		all_planes[i].points = points_of_plane
	except IndexError:
		print("Unable to find a plane at index {}, which probably means that not all planes were fitted for this block; moving on, this scan invalid".format(i))
		raise("4 valid planes not found")

@ray.remote
def compute_average_distance(points):
	pairwise_distances = []
	total_points = float(len(points)*len(points))
	for point_1 in points:
		for point_2 in points:
			if type(point_1) != type(None) and type(point_2) != type(None):
				pairwise_distances.append(point_1.distance(point_2) / mm_per_inch)
			else:
				total_points -= 1
	average_distance_for_cpu = sum(pairwise_distances) / total_points
	return average_distance_for_cpu

#convex_hull_distances_per_plane = [[] for i in range(3)]
#topological_sample_size_per_plane = 5000
sample_size_as_percent_of_number_of_plane_points = 0.20
all_distances_per_plane = [[] for i in range(3)]
average_distances = {0: 0, 1: 0, 2:0}
max_distances = {0: 0, 1: 0, 2:0}
max_distance_points = []
surfaces = {0: None, 1: None, 2: None}

for plane_number in [0,1,2]:
	max_distance_points.append([])
	max_distance = 0
	sample_size = sample_size_as_percent_of_number_of_plane_points * len(points_of_planes[plane_number])
	sample_of_points = random.sample(points_of_planes[plane_number], int(sample_size))
	print("\nComputing pairwise distances for {:,} points in plane {} to find average distance, corresponding to a topological metric".format(len(sample_of_points), plane_number))
	points_per_cpu = math.ceil(len(sample_of_points) / cpus)
	points_to_process_for_cpus = grouper(points_per_cpu, sample_of_points)
	parallel_cpu_computations = [compute_average_distance.remote(points_to_process_for_cpu) for points_to_process_for_cpu in points_to_process_for_cpus]
	parallel_cpu_results = ray.get(parallel_cpu_computations)
	average_distances[plane_number] = sum(parallel_cpu_results) / float(len(parallel_cpu_results))
	print("For plane {}, average distance of {:.4f}\"".format(plane_number, average_distances[plane_number])) # for points (h,v)=(x,y,z) at p1: ({},{})=({:.3f},{:.3f},{:.3f}) and p2: ({},{})=({:.3f},{:.3f},{:.3f})".format(plane_number, max_distance, point_a.h, point_a.v, point_a.x, point_a.y, point_a.z, point_b.h, point_b.v, point_b.x, point_b.y, point_b.z))

average_distances_sorted_keys = sorted(average_distances, key=average_distances.get, reverse=True)
plane_surface_identification = {"2\"x3\"": None, "2\"x3\"": None, "2\"x3\"": None}

print("")
for key, surface in zip(average_distances_sorted_keys, ["2\"x3\"", "1\"x3\"", "1\"x2\""]):
	average_distance = average_distances[key]
	if surface == "2\"x3\"":
		if average_distance > math.sqrt(13) + 0.1:
			print("For surface 2\"x3\" the average distance {:.4f} inches is longer than SQRT(3^2 + 2^2) = 3.6055; data needs further filtering or plane recognition wrong".format(average_distance))
			raise("Average distance too large for this surface")
		else:
			print("PLANE {} is identified as a 2\"x3\" surface with average distance {:.2f}\" between points".format(key, average_distance))
			plane_surface_identification["2\"x3\""] = all_planes[key]
	elif surface == "1\"x3\"":
		if average_distance > math.sqrt(10) + 0.1:
			print("For surface 1\"x3\" the average distance {:.4f} inches is longer than SQRT(3^2 + 1^2) = 3.1622; data needs further filtering or plane recognition wrong".format(average_distance))
			raise("Average distance too large for this surface")
		else:
			print("PLANE {} is identified as a 1\"x3\" surface with average distance {:.2f}\" between points".format(key, average_distance))
			plane_surface_identification["1\"x3\""] = all_planes[key]
	elif surface == "1\"x2\"":
		if average_distance > math.sqrt(5) + 0.1:
			print("For surface 1\"x2\" the average distance {:.4f} is longer than SQRT(2^2 + 1^2) = 2.2360; data needs further filtering or plane recognition wrong".format(average_distance))
			raise("Average distance too large for this surface")
		else:
			print("PLANE {} is identified as a 1\"x2\" surface with average distance {:.2f}\" between points".format(key, average_distance))
			plane_surface_identification["1\"x2\""] = all_planes[key]

p1 = all_planes[0]
p2 = all_planes[1]
p3 = all_planes[2]

abstract_1 = -1 * p3.d - (p3.c / (p1.c - p2.c)) * (p2.d - p1.d)
abstract_2 = -1 * p3.a - p3.c * (p2.a - p1.a) / (p1.c - p2.c)
abstract_3 = p3.b + p3.c * (p2.b - p1.b) / (p1.c - p2.c)
abstract_4 = p1.c / (p1.c - p2.c)
abstract_5 = abstract_4 * (p2.a - p1.a)
abstract_6 = abstract_4 * (p2.b - p1.b)
abstract_7 = abstract_4 * (p2.d - p1.d)
abstract_8 = abstract_6 * abstract_1 / abstract_3
abstract_9 = abstract_6 * abstract_2 / abstract_3
abstract_10 = p1.b * abstract_1 / abstract_3
abstract_11 = p1.b * abstract_2 / abstract_3

x_s = (-1 * abstract_10 - abstract_8 - abstract_7 - p1.d) / (p1.a + abstract_11 + abstract_5 + abstract_9)
y_s = (abstract_1 + abstract_2 * x_s) / abstract_3
z_s = (x_s * (p2.a - p1.a) + y_s * (p2.b - p1.b) + (p2.d - p1.d)) / (p1.c - p2.c)

def angle_between(plane_1, plane_2):
	s = plane_1.a * plane_2.a + plane_1.b * plane_2.b + plane_1.c * plane_2.c
	t = math.sqrt(plane_1.a ** 2 + plane_1.b ** 2 + plane_1.c ** 2)
	u = math.sqrt(plane_2.a ** 2 + plane_2.b ** 2 + plane_2.c ** 2)
	s = s / (t * u)
	angle = (180.0 / math.pi) * (math.acos(s))
	return angle

print("")
for plane_a_surface, plane_b_surface in [["1\"x2\"", "1\"x3\""], ["1\"x2\"", "2\"x3\""], ["1\"x3\"", "2\"x3\""]]:
	plane_a = plane_surface_identification[plane_a_surface]
	plane_b = plane_surface_identification[plane_b_surface]
	angle = angle_between(plane_a, plane_b)
	print("{} and {} planes are angled {:.2f} degrees relative to each other".format(plane_a_surface, plane_b_surface, angle))

print("\nIntersection of 3 planes calculated at (x,y,z)=({:.4f},{:.4f},{:.4f})".format(round(x_s,5), round(y_s,5), round(z_s,5)))

def point_at_distance_from_origin_along_plane_intersection(plane_1_surface, plane_2_surface, origin, distance):
	p1 = plane_surface_identification[plane_1_surface]
	p2 = plane_surface_identification[plane_2_surface]

	abstract_1 = (p2.a / p1.a) * (-1 * p1.b) + p2.b
	abstract_2 = (p2.a * p1.d / p1.a) - p2.d
	abstract_3 = (p2.a / p1.a) * p1.c - p2.c
	abstract_4 = abstract_2 / abstract_1
	abstract_5 = abstract_3 / abstract_1
	abstract_6 = -1 * (p1.d / p1.a) - (p1.b / p1.a) * abstract_4
	abstract_7 = -1 * (p1.c / p1.a) - (p1.b / p1.a) * abstract_5
	abstract_8 = abstract_6 - origin.x
	abstract_9 = abstract_4 - origin.y
	abstract_10 = abstract_8 ** 2 + abstract_9 ** 2 + origin.z ** 2
	abstract_11 = 2 * (abstract_8 * abstract_7 + abstract_9 * abstract_5 - origin.z)
	abstract_12 = abstract_7 ** 2 + abstract_5 ** 2 + 1
	abstract_13 = abstract_10 - distance ** 2

	t_possibility_1 = (-1 * abstract_11 + math.sqrt(abstract_11**2 - 4 * abstract_12 * abstract_13)) / (2 * abstract_12)
	t_possibility_2 = (-1 * abstract_11 - math.sqrt(abstract_11**2 - 4 * abstract_12 * abstract_13)) / (2 * abstract_12)

	y_possibility_1 = (p1.d * (p2.a / p1.a) + t_possibility_1 * ((p2.a / p1.a) * p1.c - p2.c) - p2.d) /  ((p2.a / p1.a) * -1 * p1.b + p2.b)
	y_possibility_2 = (p1.d * (p2.a / p1.a) + t_possibility_2 * ((p2.a / p1.a) * p1.c - p2.c) - p2.d) /  ((p2.a / p1.a) * -1 * p1.b + p2.b)

	x_possibility_1 = (-1 * p1.d - p1.c * t_possibility_1 - p1.b * y_possibility_1) / p1.a
	x_possibility_2 = (-1 * p1.d - p1.c * t_possibility_2 - p1.b * y_possibility_1) / p1.a

	z_possibility_1 = t_possibility_1
	z_possibility_2 = t_possibility_2

	# get a few sample points to reference position of points in plane relative to origin
	sample_point_from_p1 = random.sample(p1.points, 1)[0]
	sample_point_from_p2 = random.sample(p1.points, 1)[0]
	sampled_x = (sample_point_from_p1.x + sample_point_from_p2.x) / 2.0
	sampled_y = (sample_point_from_p1.y + sample_point_from_p2.y) / 2.0
	sampled_z = (sample_point_from_p1.z + sample_point_from_p2.z) / 2.0
	sampled_point = Point(sampled_x, sampled_y, sampled_z)

	possible_point_1 = Point(x_possibility_1, y_possibility_1, z_possibility_1)
	possible_point_2 = Point(x_possibility_2, y_possibility_2, z_possibility_2)

	sampled_distance_from_possibility_1 = possible_point_1.distance(sampled_point)
	sampled_distance_from_possibility_2 = possible_point_2.distance(sampled_point)

	# print("\nEquation for line at the intersection of {} and {} planes:".format(plane_1_surface, plane_2_surface))
	# print("{:.2f}(x-{:.2f}) + {:.2f}(y-{:.2f}) + {:.2f}(z-{:.2f}) = 0".format(a, origin.x, b, origin.y, c, origin.z))

	if sampled_distance_from_possibility_1 < sampled_distance_from_possibility_2:
		x = possible_point_1.x
		y = possible_point_1.y
		z = possible_point_1.z
		distance_from_origin = round(possible_point_1.distance(origin) / 25.4, 1)
	else:
		x = possible_point_2.x
		y = possible_point_2.y
		z = possible_point_2.z
		distance_from_origin = round(possible_point_2.distance(origin) / 25.4, 1)

	print("Point ({:.2f},{:.2f},{:.2f}) is {:.4f} inches from ({:.4f},{:.4f},{:.2f}) along the intersection of the {} and {} surfaces".format(x,y,z,distance_from_origin,origin.x, origin.y, origin.z, plane_1_surface, plane_2_surface))

	return Point(x,y,z)


def parallel_plane_from_point_to_point(existing_surface, point_on_existing_plane, point_on_new_plane):
	existing_plane = plane_surface_identification[existing_surface]
	new_d = -1 * (existing_plane.a * point_on_new_plane.x + existing_plane.b * point_on_new_plane.y + existing_plane.c * point_on_new_plane.z)
	parallel_plane = Plane(a=existing_plane.a, b=existing_plane.b, c=existing_plane.c, d=new_d)
	translation = Vector(point_on_new_plane.x - point_on_existing_plane.x, point_on_new_plane.y - point_on_existing_plane.y, point_on_new_plane.z - point_on_existing_plane.z)
	parallel_plane.points = [Point(p.x + translation.x, p.y + translation.y , p.z + translation.z) for p in existing_plane.points]
	print("\nFrom {} plane with original equation {:.2f}x + {:.2f}y + {:.2f}z + {:.2f} = 0...".format(existing_surface, existing_plane.a, existing_plane.b, existing_plane.c, existing_plane.d))
	print("Given that ({:.2f},{:.2f},{:.2f}) exists in a parallel plane, equation for plane {} opposite derived: {:.2f}x + {:.2f}y + {:.2f}z + {:.2f} = 0 (i.e. same slope, different intercept)".format(point_on_new_plane.x, point_on_new_plane.y, point_on_new_plane.z, surface, parallel_plane.a, parallel_plane.b, parallel_plane.c, parallel_plane.d))
	return parallel_plane

point_0 = Point(x_s, y_s, z_s)
point_1 = point_at_distance_from_origin_along_plane_intersection(plane_1_surface="1\"x3\"", plane_2_surface="1\"x2\"", origin=point_0, distance=1 * 25.4)
point_2 = point_at_distance_from_origin_along_plane_intersection(plane_1_surface="2\"x3\"", plane_2_surface="1\"x2\"", origin=point_0, distance=2 * 25.4)
point_3 = point_at_distance_from_origin_along_plane_intersection(plane_1_surface="2\"x3\"", plane_2_surface="1\"x3\"", origin=point_0, distance=3 * 25.4)

plane_1x2_opposite = parallel_plane_from_point_to_point(existing_surface="1\"x2\"", point_on_existing_plane=point_0, point_on_new_plane=point_3)
plane_1x3_opposite = parallel_plane_from_point_to_point(existing_surface="1\"x3\"", point_on_existing_plane=point_0, point_on_new_plane=point_2)
plane_2x3_opposite = parallel_plane_from_point_to_point(existing_surface="2\"x3\"", point_on_existing_plane=point_0, point_on_new_plane=point_1)

plane_surface_identification["1\"x2\" opposite"] = plane_1x2_opposite
plane_surface_identification["1\"x3\" opposite"] = plane_1x3_opposite
plane_surface_identification["2\"x3\" opposite"] = plane_2x3_opposite

point_4 = point_at_distance_from_origin_along_plane_intersection(plane_1_surface="1\"x2\" opposite", plane_2_surface="1\"x3\"", origin=point_3, distance=1 * 25.4)
point_5 = point_at_distance_from_origin_along_plane_intersection(plane_1_surface="1\"x2\" opposite", plane_2_surface="2\"x3\"", origin=point_3, distance=2 * 25.4)
point_6 = point_at_distance_from_origin_along_plane_intersection(plane_1_surface="1\"x2\" opposite", plane_2_surface="1\"x3\" opposite", origin=point_5, distance=1 * 25.4)
point_7 = point_at_distance_from_origin_along_plane_intersection(plane_1_surface="1\"x2\"", plane_2_surface="1\"x3\" opposite", origin=point_2, distance=1 * 25.4)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
names = ["point_0", "point_1", "point_2", "point_3", "point_4", "point_5", "point_6", "point_7"]

all_block_points = [point_0, point_1, point_2, point_3, point_4, point_5, point_6, point_7]

plane_colors = ['r','g','b']
xs = []
ys = []
zs = []
colors = []
markers = []

block_point_colors = ['yellow' for i in range(len(all_block_points))]
for block_point_index, block_point in enumerate(all_block_points):
	xs.append(block_point.x)
	ys.append(block_point.y)
	zs.append(block_point.z)
	colors.append(block_point_colors[block_point_index])
	#markers.append('o')

for plane_number, plane in enumerate(all_planes):
	for point in plane.points[:1000]:
		xs.append(point.x)
		ys.append(point.y)
		zs.append(point.z)
		colors.append(plane_colors[plane_number])
		#markers.append('o')

ax.scatter(xs, ys, zs, c=colors, marker='o', ) 

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

for i, txt in enumerate(names):
	ax.text(xs[i], ys[i], zs[i], txt, None)

print("")

with open("{}/{}_corner_points.txt".format(point_cloud_directory, point_cloud_identifier), "w") as output_file:
	for point, name in zip(all_block_points, names):
		print("{} = Point({:.5f},{:.5f},{:.5f})".format(name, point.x, point.y, point.z))
		output_file.write("{:.5f},{:.5f},{:.5f}\n".format(point.x, point.y, point.z))

print("\nSaving visualization of 8 corners points to {}/{}_corner_point_visualization.png".format(point_cloud_directory, point_cloud_identifier))

#plt.figure(num=1, figsize=(50, 50), dpi=1000, facecolor='w', edgecolor='k')
fig = plt.gcf()
fig.set_size_inches(40.0, 40.0)
plt.savefig("{}/{}_corner_point_visualization.png".format(point_cloud_directory, point_cloud_identifier), dpi=100)

#plt.show()

# Scanning proof -> Initial calibration dataset
# Spend 1-2 hours collecting 10-20 samples, with (x,y,z,pitch,yaw) calibrations

# Test the code on new data, iterate

#planes_visualization.save('{}_convex_hull_visualization_plane_1.png'.format(timestamp))
#print("")