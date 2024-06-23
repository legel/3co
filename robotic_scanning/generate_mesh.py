import os
import bpy, bmesh
import sys
import math
import random
import numpy as np
import time
from pprint import pprint

recompute_exp_xyz = False
recompute_global_indices = False
recompute_nonnan_vertices = False
recompute_edges_and_faces = False

total_pixel_rows = 2280 #2048
total_pixel_cols = 1824 #2048

max_distance_between_edges = 1.0 # mm

blender_dir = os.path.dirname(bpy.data.filepath)
if not blender_dir in sys.path:
    sys.path.append(blender_dir)

from geometry import PointCloud

argv = sys.argv
try:
	argv = argv[argv.index("--") + 1:] 
except:
	print("No arguments provided, run like so:")
	print("blender -P generate_mesh.py -- some.npy")
	raise

folder_directory = argv[0].split("/")[0]
filename = argv[0]
file_tag = filename.split(".")[0]
expanded_vertices_file_name = "{}_expanded_vertices.npy".format(file_tag)

print(filename)
cloud = PointCloud(filename=filename, ignore_dimensions=True, reindex_into_tensor=True, clip_outer_n_pixels=100, camera_coordinate_system="projector_coordinate_system")


print("Loaded cloud with {} points".format(len(cloud.x)))
unique_data = np.unique(cloud.cloud_tensor.shape[0], return_counts=True)
number_of_scans = unique_data[1][0]
print("{} scans".format(number_of_scans))

scan = cloud.cloud_tensor[0,:,:,:]
xyz = scan[:,:,0:3]
rgb = scan[:,:,3:6]

# t = scan[1000,1000,:]
# x = t[0]
# y = t[1]
# z = t[2]
# r = t[0]
# g = t[1]
# b = t[2]

# print("e.g. at (750,750) we have (x,y,z)=({},{},{}) with (r,g,b)=({},{},{})".format(x,y,z,r,g,b))

# define an expanded set of vertices which includes virtual vertices between existing ones, in order to expand faces from each existing vertex
total_expanded_rows = total_pixel_rows * 2 - 1
total_expanded_cols = total_pixel_cols * 2 - 1

# global variables that we will make, and then feed into Blender
mesh_vertices = []
local_nonnan_index_to_existing_mesh_vertices = {}
local_nonnan_index_to_mesh_vertices_index = {}
mesh_faces = []
mesh_colors = []

if recompute_exp_xyz:
	start_expanded_vertex_computation_time = time.time()
	exp_xyz = np.full(shape=(total_expanded_rows, total_expanded_cols, 3), fill_value=np.nan, dtype=np.float32)

	for exp_row in range(total_expanded_rows):
		if exp_row % 2 == 0:
			even_row = True
		else:
			even_row = False

		if exp_row % 100 == 0:
			print("ROW {}".format(exp_row))

		for exp_col in range(total_expanded_cols):
			# determine if row and column are even or odd; this tells whether the pixel is based on a real value or derived
			if exp_col % 2 == 0:
				even_col = True
			else:
				even_col = False

			if even_row and even_col:
				# we've hit an existing pixel, so all we need to do is look up its values
				row = int(exp_row / 2)
				col = int(exp_col / 2)
				exp_xyz[exp_row, exp_col, :] = xyz[row, col, :]

			elif even_row and not even_col:
				# we've hit a pixel row that exists, but our pixel column is in between existing cols
				row = int(exp_row / 2)
				col_1 = math.floor((exp_col / 2))
				col_2 = math.ceil((exp_col / 2))
				xyz_1 = xyz[row, col_1,:]
				xyz_2 = xyz[row, col_2,:]
				xyz_avg = np.average([xyz_1,xyz_2],axis=0)
				if ~np.isnan(xyz_avg[0]):
					#print("Virtual point: {}".format(xyz_avg))
					exp_xyz[exp_row, exp_col, :] = xyz_avg			

			elif not even_row and even_col:
				# we've hit a pixel row that exists, but our pixel column is in between existing cols
				col = int(exp_col / 2)
				row_1 = math.floor((exp_row / 2))
				row_2 = math.ceil((exp_row / 2))

				xyz_1 = xyz[row_1, col,:]
				xyz_2 = xyz[row_2, col,:]
				xyz_avg = np.average([xyz_1,xyz_2],axis=0)
				if ~np.isnan(xyz_avg[0]):
					# print("Row 1 = {}, Row 2 = {}, shape of xyz = {}".format(row_1, row_2, xyz.shape))
					# print("Point above: {}".format(xyz_1))
					# print("Virtual point from odd row: {}".format(xyz_avg))
					# print("Point below: {}".format(xyz_2))
					exp_xyz[exp_row, exp_col, :] = xyz_avg

			else:
				row_1 = math.floor((exp_row / 2))
				row_2 = math.ceil((exp_row / 2))
				col_1 = math.floor((exp_col / 2))
				col_2 = math.ceil((exp_col / 2))
				xyz_1 = xyz[row_1, col_1,:]
				xyz_2 = xyz[row_1, col_2,:]
				xyz_3 = xyz[row_2, col_1,:]
				xyz_4 = xyz[row_2, col_2,:]
				xyz_avg = np.average([xyz_1,xyz_2,xyz_3,xyz_4],axis=0)
				if ~np.isnan(xyz_avg[0]):
					#print("Virtual point: {}".format(xyz_avg))
					exp_xyz[exp_row, exp_col, :] = xyz_avg

	end_expanded_vertex_computation_time = time.time()
	with open(expanded_vertices_file_name, 'wb') as output_file:
		np.save(output_file, exp_xyz)
	print("{} seconds to compute expanded vertices".format(end_expanded_vertex_computation_time - start_expanded_vertex_computation_time))
else:
	with open(expanded_vertices_file_name, 'rb') as input_file:
		exp_xyz = np.load(input_file)

random_samples = 100
print("Here are {} samples from the data:".format(random_samples))
for i in range(random_samples):
	random_row = random.randint(0,total_expanded_rows-1)
	random_col = random.randint(0,total_expanded_cols-1)
	r_x = exp_xyz[random_row, random_col, 0]
	r_y = exp_xyz[random_row, random_col, 1]
	r_z = exp_xyz[random_row, random_col, 2]
	print("({}) for row {}, col {}: x,y,z = ({},{},{})".format(i,random_row,random_col,r_x,r_y,r_z))

if recompute_global_indices:
	global_expanded_indices = np.zeros(shape=(total_expanded_rows, total_expanded_cols), dtype=np.uint32)
	for exp_row in range(total_expanded_rows):
		for exp_col in range(total_expanded_cols):
			global_index = exp_row * total_expanded_cols + exp_col
			global_expanded_indices[exp_row, exp_col] = global_index
	with open("global_expanded_indices.npy", 'wb') as output_file:
		np.save(output_file, global_expanded_indices)
else:
	with open("global_expanded_indices.npy", 'rb') as input_file:
		global_expanded_indices = np.load(input_file)

print("Example global index for max row, max column: {:,}".format(global_expanded_indices[total_expanded_rows-1,total_expanded_cols-1]))

# saving non-NaN vertices in preparation to toss them into Blender
nonnan_vertices_filename = "{}_nonnan_vertices.npy".format(file_tag)
global_indices_to_nonnan_vertices_filename = "{}_global_indices_to_nonnan_vertices.npy".format(file_tag)

if recompute_nonnan_vertices:
	# make vertices in blender
	print("Saving non-NaN vertices")
	start_blender_vertex_making_time = time.time()
	nonnan_vertices = []
	global_indices_to_nonnan_vertices = []
	global_index = 0
	for exp_row in range(total_expanded_rows):
		if exp_row % 100 == 0:
			print("ROW {}".format(exp_row))
		for exp_col in range(total_expanded_cols):
			this_xyz = exp_xyz[exp_row,exp_col,:]
			x = this_xyz[0]
			y = this_xyz[1]
			z = this_xyz[2]
			if ~np.isnan(x):
				nonnan_vertices.append(this_xyz)
				global_indices_to_nonnan_vertices.append(global_index)
				#vertex_object = bm.verts.new((x, y, z))
				#blender_vertices.append(vertex_object)
			global_index += 1

	nonnan_vertices = np.asarray(nonnan_vertices)
	with open(nonnan_vertices_filename, 'wb') as output_file:
		np.save(output_file, nonnan_vertices)

	global_indices_to_nonnan_vertices = np.asarray(global_indices_to_nonnan_vertices)
	with open(global_indices_to_nonnan_vertices_filename, 'wb') as output_file:
		np.save(output_file, global_indices_to_nonnan_vertices)

	end_blender_vertex_making_time = time.time()
	print("{:.1f} seconds to make {:,} Blender vertices".format(end_blender_vertex_making_time - start_blender_vertex_making_time, len(nonnan_vertices)))
else:
	with open(nonnan_vertices_filename, 'rb') as input_file:
		nonnan_vertices = np.load(input_file)
	with open(global_indices_to_nonnan_vertices_filename, 'rb') as input_file:
		global_indices_to_nonnan_vertices = np.load(input_file)

print("Loaded {} non-NaN expanded vertices".format(len(nonnan_vertices)))
print("Example of first 10 non-NaN indices, in terms of their global expanded index: {}".format(global_indices_to_nonnan_vertices[:10]))
last_nonnan_index = global_indices_to_nonnan_vertices[-1]
global_vertices = dict(map(reversed, enumerate(global_indices_to_nonnan_vertices)))
last_nonnan_local_index = global_vertices.get(last_nonnan_index)
print("Looked up global index {} as existing with value {}".format(last_nonnan_index, last_nonnan_local_index))
print("There the (x,y,z) value is ({})".format(nonnan_vertices[last_nonnan_local_index]))
print("Here are the first 100 global to local indices:")
for i, (key, value) in enumerate(global_vertices.items()):
	print("({}) {} -> {}".format(i, key, value))
	if i > 100:
		break
print("There are {} total vertices with non-NaN values".format(len(global_vertices)))

faces_made = 0
n_faces_to_make = 1000000

def make_faces_in_blender(mesh_vertices, mesh_faces, mesh_colors):
	global file_tag

	# prepare objects in Blender
	mesh_name = "{}_mesh".format(file_tag)
	object_name = "{}_model".format(file_tag)
	point_cloud_mesh = bpy.data.meshes.new(mesh_name)
	point_cloud_mesh.from_pydata(mesh_vertices, [], mesh_faces)
	point_cloud_mesh.update()

	print("Created mesh from faces")

	print("Making object, linking to collection, selecting")
	point_cloud_object = bpy.data.objects.new(object_name, point_cloud_mesh)
	bpy.context.collection.objects.link(point_cloud_object)
	bpy.context.view_layer.objects.active = point_cloud_object
	point_cloud_object.select_set(state=True, view_layer=None)
	point_cloud_mesh = bpy.context.object.data
	bm = bmesh.new()
	bm.from_mesh(point_cloud_mesh)

	print("Made object, now making colors")

	color_layer = bm.loops.layers.color.new("color")

	if point_cloud_mesh.vertex_colors.active is None:
		point_cloud_mesh.vertex_colors.new()

	# faster coloring from batfinger
	# context = bpy.context
	# name = "Xxxx"
	# r, g, b, a = (1, 0, 0, 1) # red
	# ob = context.object
	# me = ob.data
	# color_layer = (me.vertex_colors.get(name)
	#                or me.vertex_colors.new(name=name)
	#                )
	# ones = np.ones(len(color_layer.data))
	# color_layer.data.foreach_set(
	#         "color",
	#         np.array((r * ones, g * ones, b * ones, a * ones)).T.ravel(),
	#         )
	# me.update()

	# assign colors of faces to mesh object
	face_index = 0
	for face in bm.faces:
		color_index = 0
		for loop in face.loops:
			loop[color_layer] = mesh_colors[face_index][color_index]
			color_index += 1
		face_index += 1

	point_cloud_object.data.update()

	bm.to_mesh(point_cloud_mesh)  
	bm.free()

	print("Added colors to faces")

	try:
		bpy.context.scene.update() 
	except:
		bpy.context.view_layer.update()

	bpy.ops.export_mesh.ply(filepath="{}_with_faces.ply".format(file_tag), check_existing=False)
	print("Exported mesh with faces")
	#bpy.ops.wm.quit_blender()



	#bpy.context.object.data.update()
	#point_cloud_object.data.update()


	# ob = context.object
	# me = ob.data
	# me.update()

	# all_materials = []
	# for face_index, (face, color) in enumerate(zip(mesh_faces,mesh_colors)):
	# 	r,g,b,a = color[0]
	# 	material_for_face = bpy.data.materials.new(name="face_{}_material".format(face_index))
	# 	material_for_face.diffuse_color = (r, g, b, a)
	# 	point_cloud_object.data.materials.append(material_for_face)
	# 	print("Added material {}".format(face_index))

	# point_cloud_object.data.update()

	# for i, face in enumerate(bm.faces): # Iterate over all of the object's faces
	# 	face.material_index = i # random.randint(0, len(total_materials) - 1)  # Assing random material to face
 
	# point_cloud_object.data.update() 
	# bpy.ops.object.mode_set(mode = 'OBJECT')

	# m1 = bpy.data.materials.new(name="x1")
	# m1.diffuse_color = (0.75, 0.0, 0.0, 1.0)

	# m2 = bpy.data.materials.new(name="x1")
	# m2.diffuse_color = (0.0, 0.75, 0.0, 1.0)

	# total_materials = [m1,m2]

	# for material in total_materials:
	# 	#if point_cloud_object.data.materials:
	# 	#    # assign to 1st material slot
	# 	#    point_cloud_object.data.materials[0] = material
	# 	#else:
	# 	#    # no slots
	# 	point_cloud_object.data.materials.append(material)

	# bpy.ops.object.mode_set(mode = 'EDIT')  # Go to edit mode to create bmesh
	# ob = bpy.context.object                 # Reference to selected object
	 
	# bm = bmesh.from_edit_mesh(ob.data)      # Create bmesh object from object mesh
	 
	# bm.select_mode = {'FACE'}               # Go to face selection mode
	# bm.faces[0].select_set(True)            # Select   face 0
	# bm.faces[1].select_set(False)           # Deselect face 1
	# ob.data.update()                        # Update these changes to the original mesh


	# bpy.ops.object.mode_set(mode = 'EDIT')
	# #bm = bmesh.from_edit_mesh(point_cloud_object.data)      # Create bmesh object from object mesh 
	# bm.select_mode = {'FACE'}               # Go to face selection mode
	# #bm.faces.select_all(action='SELECT')
	# #bpy.ops.mesh.select_face_by_sides(number=3)
	# for face_index in range(len(mesh_faces)):
	# 	bm.faces[face_index].select_set(True)




	# 
	#context = bpy.context
	#ob = context.object
	#me = ob.data
	#color_layer = (me.vertex_colors.get("color") or me.vertex_colors.new(name="color"))

	#color_layer = bm.loops.layers.color.new("color")
	# if point_cloud_mesh.vertex_colors.active is None:
	# 	point_cloud_mesh.vertex_colors.new()

	# fake_colors = np.full(fill_value=0.0, shape=(len(mesh_faces), 3, 4), dtype=np.float32)
	# fake_colors[:,:,0] = 0.75
	# fake_colors[:,:,1] = 0.0
	# fake_colors[:,:,2] = 0.0
	# fake_colors[:,:,3] = 1.0

	#r, g, b, a = (0.75, 0, 0, 1) # red
	#ones = np.ones(len(color_layer.data))

	#print("Length of color_layer.data: {}".format(len(color_layer.data)))

	#colors = np.array((r * ones, g * ones, b * ones, a * ones)).T.ravel()

	#color_layer.data.foreach_set("color",colors)

	#me.update()

	#fake_colors = fake_colors.flatten()

	# vertex_colors = point_cloud_mesh.vertex_colors.active.data
	# vertex_colors.foreach_set("color", fake_colors) #  mesh_colors.flatten()
	# point_cloud_mesh.update()




def new_face(vertices, color):
	global faces_made
	#print("Given vertices with local non-NaN indices of {} and color of {}".format(vertices, color))
	these_indices_in_mesh_vertices = [] 
	for i, vertex in enumerate(vertices):
		this_xyz = nonnan_vertices[vertex]
		x = this_xyz[0]
		y = this_xyz[1]
		z = this_xyz[2]
		#print("({}) x,y,z = ({},{},{})".format(i,x,y,z))
		vertex_exists = local_nonnan_index_to_existing_mesh_vertices.get(vertex, False)
		if not vertex_exists:
			local_nonnan_index_to_existing_mesh_vertices[vertex] = True
			current_index_in_mesh_vertices = len(mesh_vertices)
			local_nonnan_index_to_mesh_vertices_index[vertex] = current_index_in_mesh_vertices
			mesh_vertices.append((x,y,z))
			#print("Creating new vertex in mesh_vertices with current index {}".format(current_index_in_mesh_vertices))
		else:
			current_index_in_mesh_vertices = local_nonnan_index_to_mesh_vertices_index[vertex]
			#print("Found existing vertex in mesh_vertices with current index {}".format(current_index_in_mesh_vertices))
		these_indices_in_mesh_vertices.append(current_index_in_mesh_vertices)
	face = (these_indices_in_mesh_vertices[0], these_indices_in_mesh_vertices[1], these_indices_in_mesh_vertices[2])
	mesh_faces.append(face)

	blender_color = [color[0]/255.0, color[1]/255.0, color[2]/255.0, 1.0]
	mesh_colors.append([blender_color,blender_color,blender_color])
	faces_made += 1


pixels_traversed = 0
n_pixels_to_traverse = 1000
mesh_vertices_filename = "{}_mesh_vertices.npy".format(file_tag)
mesh_faces_filename = "{}_mesh_faces.npy".format(file_tag)
mesh_colors_filename = "{}_mesh_colors.npy".format(file_tag)

if recompute_edges_and_faces:
	start_face_making_time = time.time()
	valid_vertices = {}
	valid_edges = {}
	for exp_row in range(1,total_expanded_rows-1):
		if exp_row % 100 == 0:
			print("ROW {}: {} faces made".format(exp_row, len(mesh_faces)))
		if faces_made == n_faces_to_make:
			break
		for exp_col in range(1,total_expanded_cols-1):
			if faces_made == n_faces_to_make:
				break

			# determine if row and column are even or odd; this tells whether the pixel is based on a real value or derived
			if exp_row % 2 == 0:
				even_row = True
			else:
				even_row = False
			if exp_col % 2 == 0:
				even_col = True
			else:
				even_col = False

			# if not even_row or not even_col:
			# 	this_global_index = global_expanded_indices[exp_row,exp_col]
			# 	this_local_index = global_vertices.get(this_global_index,-1)
			# 	if this_local_index != -1:
			# 		pixels_traversed += 1
			# 		x,y,z = exp_xyz[exp_row,exp_col,:]
			# 		#print("Adding superpoint! We actually have a non-NaN virtual vertex")
			# 		print("({}): ({},{},{})".format(pixels_traversed,x,y,z))
			# 		cloud.add_superpoint(x, y, z, 255, 0, 0, sphere_radius=0.5, superpoint_samples=50, pixel_row=-1, pixel_column=-1, scan_index=-1)
			
			# if pixels_traversed == n_pixels_to_traverse:
			# 	cloud.save_as_ply(filename="{}_superpointed_virtual_vertices.ply".format(file_tag), from_tensor=False, communicate=True)
			# 	sys.exit(0)

			if even_row and even_col:

				row = int(exp_row / 2)
				col = int(exp_col / 2)
				color = rgb[row, col, :] # [red, green, blue]

				v_0_row = exp_row - 1
				v_0_col = exp_col - 1
				v_0_global_index = global_expanded_indices[v_0_row,v_0_col]
				v_0_local_index = global_vertices.get(v_0_global_index,-1)

				v_1_row = exp_row - 1
				v_1_col = exp_col
				v_1_global_index = global_expanded_indices[v_1_row,v_1_col]
				v_1_local_index = global_vertices.get(v_1_global_index,-1)

				v_2_row = exp_row - 1 
				v_2_col = exp_col + 1
				v_2_global_index = global_expanded_indices[v_2_row,v_2_col]
				v_2_local_index = global_vertices.get(v_2_global_index,-1)

				v_3_row = exp_row
				v_3_col = exp_col - 1
				v_3_global_index = global_expanded_indices[v_3_row,v_3_col]
				v_3_local_index = global_vertices.get(v_3_global_index,-1)

				v_4_row = exp_row
				v_4_col = exp_col
				v_4_global_index = global_expanded_indices[v_4_row,v_4_col]
				v_4_local_index = global_vertices.get(v_4_global_index,-1)

				v_5_row = exp_row
				v_5_col = exp_col + 1
				v_5_global_index = global_expanded_indices[v_5_row,v_5_col]
				v_5_local_index = global_vertices.get(v_5_global_index,-1)

				v_6_row = exp_row + 1
				v_6_col = exp_col - 1
				v_6_global_index = global_expanded_indices[v_6_row,v_6_col]
				v_6_local_index = global_vertices.get(v_6_global_index,-1)

				v_7_row = exp_row + 1
				v_7_col = exp_col
				v_7_global_index = global_expanded_indices[v_7_row,v_7_col]
				v_7_local_index = global_vertices.get(v_7_global_index,-1)

				v_8_row = exp_row + 1 
				v_8_col = exp_col + 1
				v_8_global_index = global_expanded_indices[v_8_row,v_8_col]
				v_8_local_index = global_vertices.get(v_8_global_index,-1)

				global_indices = [v_0_global_index, v_1_global_index, v_2_global_index, v_3_global_index, v_4_global_index, v_5_global_index, v_6_global_index, v_7_global_index, v_8_global_index]
				local_indices = [v_0_local_index, v_1_local_index, v_2_local_index, v_3_local_index, v_4_local_index, v_5_local_index, v_6_local_index, v_7_local_index, v_8_local_index]
				these_rows = [v_0_row, v_1_row, v_2_row, v_3_row, v_4_row, v_5_row, v_6_row, v_7_row, v_8_row]
				these_cols = [v_0_col, v_1_col, v_2_col, v_3_col, v_4_col, v_5_col, v_6_col, v_7_col, v_8_col]

				# indices over vertex IDs
				edge_indices = [[0,1], [1,2], [0,3], [0,4], [1,4], [2,4], [2,5], [3,4], [4,5], [3,6], [4,6], [4,7], [4,8], [5,8], [6,7], [7,8]]

				valid_edges_to_vertices = {}

				for relative_edge_index, (a, b) in enumerate(edge_indices):
					#print("Relative vertex: ({},{})".format(a,b))
					a_local_index = local_indices[a]
					b_local_index = local_indices[b]
					if a_local_index != -1 and b_local_index != -1:
						#print("*Indices that are not -1*")
						a_row = these_rows[a]
						b_row = these_rows[b]
						a_col = these_cols[a]
						b_col = these_cols[b]

						xyz_a = exp_xyz[a_row,a_col,:]
						xyz_b = exp_xyz[b_row,b_col,:]

						#xyz_a = nonnan_vertices[a_local_index]
						#xyz_b = nonnan_vertices[b_local_index]

						# compute distance
						distance = np.linalg.norm(xyz_a - xyz_b)
						# print("Distance {}".format(distance))

						if distance < max_distance_between_edges:
							# print("((row,col)=({},{}) Adding relative edge {} with distance {} into valid edges".format(row,col,relative_edge_index,distance))
							# #print("Vertex (row,col) positions, where v_4 is the real vertex in question, and the positions is in an (2x,2x) expanded coordinate system")

							# print("global expanded indices: {}".format(global_indices))
							# print("local indices: {}".format(local_indices))

							# print("v_0: ({},{}), v_1: ({},{}), v_2: ({},{})".format(v_0_row,v_0_col,v_1_row,v_1_col,v_2_row,v_2_col))
							# print("v_3: ({},{}), v_4: ({},{}), v_5: ({},{})".format(v_3_row,v_3_col,v_4_row,v_4_col,v_5_row,v_5_col))
							# print("v_6: ({},{}), v_7: ({},{}), v_8: ({},{})".format(v_6_row,v_6_col,v_7_row,v_7_col,v_8_row,v_8_col))
							# print()
							# print("v_0: ({}), v_1: ({}), v_2: ({})".format(exp_xyz[v_0_row,v_0_col,:], exp_xyz[v_1_row,v_1_col,:], exp_xyz[v_2_row,v_2_col,:]))
							# print("v_3: ({}), v_4: ({}), v_5: ({})".format(exp_xyz[v_3_row,v_3_col,:], exp_xyz[v_4_row,v_4_col,:], exp_xyz[v_5_row,v_5_col,:]))
							# print("v_6: ({}), v_7: ({}), v_8: ({})".format(exp_xyz[v_6_row,v_6_col,:], exp_xyz[v_7_row,v_7_col,:], exp_xyz[v_8_row,v_8_col,:]))
							# print()
							valid_edges_to_vertices[relative_edge_index] = np.array([a_local_index, b_local_index])

				# indices over edge IDs
				face_indices = [[0,3,4], [1,4,5], [3,2,7], [5,8,6], [7,9,10], [8,12,13], [10,14,11], [12,11,15]]
				for relative_face_index, (e1, e2, e3) in enumerate(face_indices):

					e1_vertices = valid_edges_to_vertices.get(e1,None)
					e2_vertices = valid_edges_to_vertices.get(e2,None)
					e3_vertices = valid_edges_to_vertices.get(e3,None)

					if type(e1_vertices) != type(None) and type(e2_vertices) != type(None) and type(e3_vertices) != type(None):
						vertices_for_face = np.unique(np.array([e1_vertices, e2_vertices, e3_vertices]))
						new_face(vertices=vertices_for_face, color=color)

	with open(mesh_vertices_filename, 'wb') as output_file:
		np.save(output_file, mesh_vertices)

	with open(mesh_faces_filename, 'wb') as output_file:
		np.save(output_file, mesh_faces)

	with open(mesh_colors_filename, 'wb') as output_file:
		np.save(output_file, mesh_colors)

	end_face_making_time = time.time()
	print("{} seconds to make {} faces".format(end_face_making_time - start_face_making_time, len(mesh_faces)))

else:
	with open(mesh_vertices_filename, 'rb') as input_file:
		mesh_vertices = np.load(input_file)
		mesh_vertices = list(zip(mesh_vertices[:,0], mesh_vertices[:,1], mesh_vertices[:,2]))
	with open(mesh_faces_filename, 'rb') as input_file:
		mesh_faces = np.load(input_file)
		mesh_faces = list(zip(mesh_faces[:,0], mesh_faces[:,1], mesh_faces[:,2]))
	with open(mesh_colors_filename, 'rb') as input_file:
		mesh_colors = np.load(input_file)
		mesh_colors = np.asarray(mesh_colors)

# min_face_index = 4000000
# max_face_index = 4300000
# mesh_faces = mesh_faces[min_face_index:max_face_index]
# mesh_colors = mesh_colors[min_face_index:max_face_index]

print("{} mesh vertices, {} mesh faces, {} mesh colors".format(len(mesh_vertices), len(mesh_faces), len(mesh_colors)))
print("First 3 vertices: {}".format(mesh_vertices[:3]))
print("First 3 faces: {}".format(mesh_faces[:3]))
print("First 3 colors: {}".format(mesh_colors[:3]))

make_faces_in_blender(mesh_vertices, mesh_faces, mesh_colors)

# 	bm.verts.new((vertex[0], vertex[1], vertex[2]))
# 	bm.edges.new( [vertex_a, vertex_b] )
#	bm.faces.new( [vertex_a, vertex_b, vertex_c])
# 	vertex_a_red = point_cloud['red'][vertex_index_a] / 255.0
# 	color_a = [vertex_a_red, vertex_a_green, vertex_a_blue, 1.0]
# 	colors_of_faces.append([color_a, color_b, color_c])
#	point_cloud_mesh.update()