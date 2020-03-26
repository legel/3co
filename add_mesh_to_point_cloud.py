import numpy as np
import bpy, bmesh
from bpy import context
import sys
import time 
import os
argv = sys.argv
try:
	argv = argv[argv.index("--") + 1:] 
except:
	print("No arguments provided, run like so:")
	print("blender -P add_mesh_to_point_cloud.py -- some.ply")
	raise

current_directory = os.getcwd()
bpy.ops.preferences.addon_install(overwrite=True, filepath='{}/point_cloud_visualizer.py'.format(current_directory))
bpy.ops.preferences.addon_enable(module='point_cloud_visualizer')

from point_cloud_visualizer import PlyPointCloudReader

launch_time = int(time.time())
current_directory = os.getcwd()
folder_directory = argv[0].split("/")[0]
ply_filename = argv[0]
filename_minus_type = ply_filename.replace(".ply", "").split("/")[-1]

point_cloud = PlyPointCloudReader(ply_filename).points # see https://raw.githubusercontent.com/uhlik/bpy/master/space_view3d_point_cloud_visualizer.py

# bpy.ops.import_mesh.ply(filepath=ply_filename)
# obj = bpy.data.objects[filename_minus_type] 
# obj.select_set(True)

point_cloud_mesh = bpy.data.meshes.new("point_cloud_mesh")
point_cloud_object = bpy.data.objects.new("point_cloud_object", point_cloud_mesh)
bpy.context.collection.objects.link(point_cloud_object)
bpy.context.view_layer.objects.active = point_cloud_object
point_cloud_object.select_set( state = True, view_layer = None)

point_cloud_mesh = bpy.context.object.data
bm = bmesh.new()
bm.from_mesh(point_cloud_mesh)
color_layer = bm.loops.layers.color.new("color")

if point_cloud_mesh.vertex_colors.active is None:
	point_cloud_mesh.vertex_colors.new()

vertices = np.column_stack((point_cloud['x'], point_cloud['y'], point_cloud['z'], ))
vertices = vertices.astype(np.float32)
vertex_objects = []

valid_points = 0
invalid_points = 0
for vertex in vertices:
	if np.isnan(vertex[0]):
		invalid_points += 1
		vertex_object = bm.verts.new((0.0, 0.0, 0.0))
		vertex_objects.append(vertex_object)		
	else:
		valid_points += 1
		vertex_object = bm.verts.new((vertex[0], vertex[1], vertex[2]))
		vertex_objects.append(vertex_object)

print("{} valid points and {} invalid points (removed from point cloud)".format(valid_points, invalid_points))

rows = 2280 # vertical_pixels
columns = 1824 # horizontal_pixels
invalid = 0
valid = 0

max_edge_distance = 1.00 # millimeters

class Point():
	def __init__(self, vertex, row, column):
		self.vertex = vertex
		if vertex.co.x == 0.0 and vertex.co.y == 0.0 and vertex.co.z == 0.0:
			self.valid = False
		else:
			self.valid = True
		self.co = vertex.co
		self.x = self.co.x
		self.y = self.co.y
		self.z = self.co.z
		self.row = row
		self.column = column

	def distance(self, other_vertex):
		a = np.array((self.x, self.y, self.z))
		b = np.array((other_vertex.x, other_vertex.y, other_vertex.z))
		return np.linalg.norm(a - b)


for row in range(rows):
	for column in range(columns):
		if row+1 <= rows-1 and column+1 <= columns-1:
			vertex_index_a = row * columns + column
			vertex_index_b = row * columns + (column+1)
			vertex_index_c = (row+1) * columns + column

			vertex_a = vertex_objects[vertex_index_a]
			vertex_b = vertex_objects[vertex_index_b]
			vertex_c = vertex_objects[vertex_index_c]

			point_a = Point(vertex=vertex_a, row=row, column=column)
			point_b = Point(vertex=vertex_b, row=row, column=column+1)
			point_c = Point(vertex=vertex_c, row=row+1, column=column)

			if point_a.valid and point_b.valid and point_c.valid:
				if point_a.distance(point_b) < max_edge_distance and point_a.distance(point_c) < max_edge_distance and point_b.distance(point_c) < max_edge_distance:
					bm.edges.new( [vertex_a, vertex_b] )
					bm.edges.new( [vertex_a, vertex_c] )
					bm.edges.new( [vertex_b, vertex_c] )

colors_of_faces = []

for row in range(rows):
	for column in range(columns):
		if row+1 <= rows-1 and column+1 <= columns-1:
			vertex_index_a = row * columns + column
			vertex_index_b = row * columns + (column+1)
			vertex_index_c = (row+1) * columns + column
			vertex_index_d = (row+1) * columns + (column+1)

			vertex_a = vertex_objects[vertex_index_a]
			vertex_b = vertex_objects[vertex_index_b]
			vertex_c = vertex_objects[vertex_index_c]
			vertex_d = vertex_objects[vertex_index_d]

			point_a = Point(vertex=vertex_a, row=row, column=column)
			point_b = Point(vertex=vertex_b, row=row, column=column+1)
			point_c = Point(vertex=vertex_c, row=row+1, column=column)			
			point_d = Point(vertex=vertex_d, row=row+1, column=column)			

			if point_a.valid and point_b.valid and point_c.valid and point_d.valid:
				if point_a.distance(point_b) < max_edge_distance and point_a.distance(point_c) < max_edge_distance and point_b.distance(point_c) < max_edge_distance and point_c.distance(point_d) < max_edge_distance and point_b.distance(point_d) < max_edge_distance:
					bm.faces.new( [vertex_a, vertex_b, vertex_c])
					bm.faces.new( [vertex_b, vertex_c, vertex_d])

					vertex_a_red = point_cloud['red'][vertex_index_a] / 255.0
					vertex_b_red = point_cloud['red'][vertex_index_b] / 255.0
					vertex_c_red = point_cloud['red'][vertex_index_c] / 255.0
					vertex_d_red = point_cloud['red'][vertex_index_d] / 255.0

					vertex_a_green = point_cloud['green'][vertex_index_a] / 255.0
					vertex_b_green = point_cloud['green'][vertex_index_b] / 255.0
					vertex_c_green = point_cloud['green'][vertex_index_c] / 255.0
					vertex_d_green = point_cloud['green'][vertex_index_d] / 255.0

					vertex_a_blue = point_cloud['blue'][vertex_index_a] / 255.0
					vertex_b_blue = point_cloud['blue'][vertex_index_b] / 255.0
					vertex_c_blue = point_cloud['blue'][vertex_index_c] / 255.0
					vertex_d_blue = point_cloud['blue'][vertex_index_d] / 255.0


					color_a = [vertex_a_red, vertex_a_green, vertex_a_blue, 1.0]
					color_b = [vertex_b_red, vertex_b_green, vertex_b_blue, 1.0]
					color_c = [vertex_c_red, vertex_c_green, vertex_c_blue, 1.0]
					color_d = [vertex_d_red, vertex_d_green, vertex_d_blue, 1.0]

					colors_of_faces.append([color_a, color_b, color_c])
					colors_of_faces.append([color_b, color_c, color_d])

point_cloud_mesh.update()

# assign colors of faces to mesh object
face_index = 0
for face in bm.faces:
	color_index = 0
	for loop in face.loops:
		loop[color_layer] = colors_of_faces[face_index][color_index]
		color_index += 1
	face_index += 1

bm.to_mesh(point_cloud_mesh)  
bm.free()

try:
	bpy.context.scene.update() 
except:
	bpy.context.view_layer.update()

bpy.ops.export_mesh.ply(filepath="{}/{}.ply".format(current_directory, launch_time), check_existing=False)
print("Exported {} with mesh and colored faces based on average of vertex colors".format("{}/{}.ply".format(current_directory, launch_time)))

# sys.exit(0)