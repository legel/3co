import numpy as np
import bpy, bmesh
from bpy import context
import sys

ply_filename_to_convert_to_csv = "/Users/x/Downloads/block_x_0.05_y_0.0_z_0.2_theta_0.0_phi_-45.0.ply"
filename_minus_type = ply_filename_to_convert_to_csv.replace(".ply", "").split("/")[-1]

bpy.ops.import_mesh.ply(filepath=ply_filename_to_convert_to_csv)
obj = bpy.data.objects[filename_minus_type] 
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode='EDIT')

# load vertices as numpy array
total_vertices = len(obj.data.vertices)
rows = 2280
columns = 1824 

print(total_vertices)

#points = np.zeros((pixel_rows, pixel_columns, 3))
invalid = 0
valid = 0

class Point():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def highlight_hitpoint(location, diffuse_color=(1.0, 0.0, 0.0, 1.0)):
    x = location.x
    y = location.y
    z = location.z
    mesh = bpy.data.meshes.new('hitpoint_({},{},{})'.format(x,y,z))
    sphere = bpy.data.objects.new('hitpoint_({},{},{})_object'.format(x,y,z), mesh)
    sphere.location = (location.x,location.y,location.z)
    bpy.context.collection.objects.link(sphere)
    bpy.context.view_layer.objects.active = sphere
    sphere.select_set( state = True, view_layer = None)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, diameter=0.1)
    bm.to_mesh(mesh)
    bpy.ops.object.modifier_add(type='SUBSURF')
    bpy.ops.object.shade_smooth()
    bm.free()
    material = bpy.data.materials.new(name="material_for_({},{},{})".format(x,y,z))
    material.use_nodes = True
    nodes = material.node_tree.nodes
    for node in nodes:
      nodes.remove(node)
    node_output = nodes.new("ShaderNodeOutputMaterial")
    node_output.location = (100, 450)
    principled_node = nodes.new("ShaderNodeBsdfPrincipled")
    principled_node.location = (100, 250)
    links = material.node_tree.links
    links.new(principled_node.outputs[0], node_output.inputs[0])
    principled_node.inputs['Specular'].default_value = 0.015
    principled_node.inputs['Base Color'].default_value = diffuse_color
    sphere.data.materials.append(material)

point_1 = Point(44.08039,7.73763,315.13249)
point_2 = Point(-12.69831,8.30327,365.94881)
point_3 = Point(72.22837,-20.33122,346.76271)
point_4 = Point(53.49774,28.91689,325.52058)
point_5 = Point(-3.28097,29.48253,376.33690)
point_6 = Point(15.44966,-19.76558,397.57903)
point_7 = Point(24.86701,1.41368,407.96712)
point_8 = Point(81.64571,0.84804,357.15080)

all_block_points = [point_1, point_2, point_3, point_4, point_5, point_6, point_7, point_8]

for point in all_block_points:
    highlight_hitpoint(location=point)


def create_plane_face(points, plane_id, diffuse_color):
    mesh = bpy.data.meshes.new("{}".format(plane_id))
    mesh_object = bpy.data.objects.new("{}_mesh".format(plane_id), mesh)
    bpy.context.collection.objects.link(mesh_object)
    bpy.context.view_layer.objects.active = mesh_object
    mesh_object.select_set( state = True, view_layer = None)
    mesh = bpy.context.object.data
    bm = bmesh.new()

    points[0].x

    top_left = bm.verts.new((points[0].x, points[0].y, points[0].z))
    top_right = bm.verts.new((points[1].x, points[1].y, points[1].z))
    bottom_left = bm.verts.new((points[2].x, points[2].y, points[2].z))
    bottom_right = bm.verts.new((points[3].x, points[3].y, points[3].z))
    center_x = (points[0].x + points[1].x + points[2].x + points[3].x) / 4.0
    center_y = (points[0].y + points[1].y + points[2].y + points[3].y) / 4.0
    center_z = (points[0].z + points[1].z + points[2].z + points[3].z) / 4.0
    center = bm.verts.new((center_x, center_y, center_z))

    # self.x_rotation_angle = np.random.normal(loc=0.0, scale=math.radians(5.0))
    # self.y_rotation_angle = np.random.normal(loc=0.0, scale=math.radians(5.0))
    # self.z_rotation_angle = np.random.normal(loc=0.0, scale=math.radians(5.0))
    # self.mesh.rotation_euler = [self.x_rotation_angle, self.y_rotation_angle, self.z_rotation_angle] # angular rotations about x,y,z axis

    bm.edges.new( [top_left, top_right] )
    bm.faces.new( [top_left, top_right, center]) 
    bm.edges.new( [top_left, bottom_left] )
    bm.faces.new( [top_left, bottom_left, center]) 
    bm.edges.new( [top_right, bottom_right] )
    bm.faces.new( [top_right, bottom_right, center]) 
    bm.edges.new( [bottom_left, bottom_right] )
    bm.faces.new( [bottom_left, bottom_right, center]) 
    bm.to_mesh(mesh)  
    bm.free()

    material = bpy.data.materials.new(name="material_for_{}".format(plane_id))
    material.use_nodes = True
    nodes = material.node_tree.nodes
    for node in nodes:
      nodes.remove(node)
    node_output = nodes.new("ShaderNodeOutputMaterial")
    node_output.location = (100, 450)
    principled_node = nodes.new("ShaderNodeBsdfPrincipled")
    principled_node.location = (100, 250)
    links = material.node_tree.links
    links.new(principled_node.outputs[0], node_output.inputs[0])
    principled_node.inputs['Specular'].default_value = 0.015
    principled_node.inputs['Base Color'].default_value = diffuse_color
    principled_node.inputs['Alpha'].default_value = 0.8
    mesh_object.data.materials.append(material)

    mesh_object.active_material.blend_method = 'BLEND'

    try:
      bpy.context.scene.update() 
    except:
      bpy.context.view_layer.update()

plane_23 = [point_2, point_1, point_6, point_3]
plane_23_o = [point_4, point_5, point_8, point_7]
plane_13 = [point_1, point_2, point_4, point_5]
plane_13_o = [point_6, point_3, point_7, point_8]
plane_12 = [point_1, point_4, point_3, point_8]
plane_12_o = [point_5, point_2, point_7, point_6]

all_planes_points = [plane_23, plane_23_o, plane_13, plane_13_o, plane_12, plane_12_o]
ids = ["plane_23", "plane_23_o", "plane_13", "plane_13_o", "plane_12", "plane_12_o"]
colors = [(37/255.0, 133/255.0, 225/255.0, 0.6), (140/255.0, 48/255.0, 15/255.0, 0.6), (219/255.0, 199/255.0, 143/255.0, 0.6), (88/255.0, 201/255.0, 70/255.0, 0.6), (206/255.0, 134/255.0, 34/255.0, 0.6), (83/255.0, 144/255.0, 89/255.0, 0.6)]
for plane_points, plane_id, color in zip(all_planes_points, ids, colors):
    create_plane_face(plane_points, plane_id, color)
