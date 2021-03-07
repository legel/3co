import numpy as np
import bpy, bmesh
from bpy import context
import sys
import os
argv = sys.argv
try:
	argv = argv[argv.index("--") + 1:] 
except:
	print("No arguments provided, run like so:")
	print("blender -P convert_ply_to_csv.py -- some.ply")
	raise

current_directory = os.getcwd()
print("Current directory: {}".format(current_directory))
folder_directory = argv[0].split("/")[0]
ply_filename_to_convert_to_csv = argv[0]
filename_minus_type = ply_filename_to_convert_to_csv.replace(".ply", "").split("/")[-1]

bpy.ops.import_mesh.ply(filepath=ply_filename_to_convert_to_csv)
obj = bpy.data.objects[filename_minus_type] 
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode='EDIT')
total_vertices = len(obj.data.vertices)
rows = 2280
columns = 1824 
invalid = 0
valid = 0

with open("{}/{}_hvxyz.csv".format(current_directory, filename_minus_type), "w") as output_file:
	for row in range(rows):
		for column in range(columns):
			vertex_index = row * columns + column
			vertex = obj.data.vertices[vertex_index]
			if vertex.co.x == 0.0 and vertex.co.y == 0.0 and vertex.co.z == 0.0:
				invalid += 1
			else:
				valid += 1
				print("{} valid and {} invalid".format(valid, invalid))
			co = obj.matrix_world @ vertex.co
			output_file.write("{},{},{},{},{}\n".format(row, column, co.x, co.y, co.z))

print("File written to {}_hvxyz.csv".format(filename_minus_type))

sys.exit(0)