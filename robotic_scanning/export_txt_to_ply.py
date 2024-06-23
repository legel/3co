import numpy as np
import bpy, bmesh
from bpy import context
import sys
import time 
import os
argv = sys.argv

# convert list of points into .ply
# .txt file input structure:
#x1 y1 z1\n
#x2 y2 z2\n
# ...
#xN yN zN\n
#

try:
	argv = argv[argv.index("--") + 1:] 
except:
	print("No arguments provided, run like so:")
	print("blender -b -P /home/sense/3cobot/export_txt_to_ply.py -- /home/sense/3cobot/cloud_points.txt")
	raise

red = 255
green = 255
blue = 255

current_directory = os.getcwd()
bpy.ops.preferences.addon_install(overwrite=True, filepath='{}/point_cloud_visualizer.py'.format(current_directory))
bpy.ops.preferences.addon_enable(module='point_cloud_visualizer')

from point_cloud_visualizer import BinPlyPointCloudWriter

launch_time = int(time.time())
current_directory = os.getcwd()
folder_directory = argv[0].split("/")[0]
filename = argv[0]
filename_stub = filename.split(".txt")[0]

xs = []
ys = []
zs = []
reds = []
greens = []
blues = []

with open(filename, "r") as lines:
	for line in lines:
		if " " in line:
			split_line = line.rstrip("\n").split(" ")
			x = split_line[0]
			y = split_line[1]
			z = split_line[2]
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