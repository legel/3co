import numpy as np
import bpy, bmesh
from bpy import context
import sys
import time 
import os
argv = sys.argv

# convert OpenCV point cloud XML into .ply

try:
	argv = argv[argv.index("--") + 1:] 
except:
	print("No arguments provided, run like so:")
	print("blender -P export_xml_to_ply.py -- cloud_points.xml")
	raise

current_directory = os.getcwd()
bpy.ops.preferences.addon_install(overwrite=True, filepath='{}/point_cloud_visualizer.py'.format(current_directory))
bpy.ops.preferences.addon_enable(module='point_cloud_visualizer')

from point_cloud_visualizer import BinPlyPointCloudWriter

launch_time = int(time.time())
current_directory = os.getcwd()
folder_directory = argv[0].split("/")[0]
xml_filename = argv[0]
filename_stub = xml_filename.split(".xml")[0]

xs = []
ys = []
zs = []
reds = []
greens = []
blues = []
with open(xml_filename, "r") as lines:
	lines.readline()
	for line in lines:
		split_line = line.rstrip("\n").split(",")
		x = split_line[2]
		y = split_line[3]
		z = split_line[4]
		red = split_line[5]
		green = split_line[6]
		blue = split_line[7]
		xs.append(float(x))
		ys.append(float(y))
		zs.append(float(z))
		reds.append(int(red))
		greens.append(int(green))
		blues.append(int(blue))

xs = np.array(xs, dtype=np.float32)
ys = np.array(ys, dtype=np.float32)
zs = np.array(zs, dtype=np.float32)
reds = np.array(reds, dtype=np.uint8)
greens = np.array(greens, dtype=np.uint8)
blues = np.array(blues, dtype=np.uint8)

dt = (('x', xs.dtype.str, ),
      ('y', ys.dtype.str, ),
      ('z', zs.dtype.str, ), 
      ('red', reds.dtype.str, ),
      ('green', greens.dtype.str, ),
      ('blue', blues.dtype.str, ), 
     )

number_of_points = len(xs)

point_cloud_data = np.empty(number_of_points, dtype=list(dt), )

point_cloud_data['x'] = xs
point_cloud_data['y'] = ys
point_cloud_data['z'] = zs
point_cloud_data['red'] = reds
point_cloud_data['green'] = greens
point_cloud_data['blue'] = blues

w = BinPlyPointCloudWriter("{}_point_cloud.ply".format(filename_stub), point_cloud_data)