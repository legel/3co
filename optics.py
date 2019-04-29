import bpy
import math
import random
import time
import numpy as np
from pprint import pprint
from math import cos, sin
import bmesh
from PIL import Image
from mathutils import Vector

bpy.context.scene.render.engine = 'CYCLES'

class Point():
  def __init__(self, x=0, y=0, z=0):
    self.x = x
    self.y = y
    self.z = z

  def distance(self, point):
    a = np.array((self.x, self.y, self.z))
    b = np.array((point.x, point.y, point.z))
    return np.linalg.norm(a - b)

  def xyz(self):
    return (self.x, self.y, self.z)

class Pixel():
  def __init__(self, h, v):
    self.h = h
    self.v = v

  def calculate_unit_vectors_through_focal_point(self, focal_point):
    self.distance_to_focal_point = focal_point.distance(self.center)
    self.unit_x = (focal_point.x - self.center.x) / self.distance_to_focal_point
    self.unit_y = (focal_point.y - self.center.y) / self.distance_to_focal_point
    self.unit_z = (focal_point.z - self.center.z) / self.distance_to_focal_point

  def raycast(self, distance_from_pixel):
    x_projected = self.center.x + self.unit_x * distance_from_pixel
    y_projected = self.center.y + self.unit_y * distance_from_pixel
    z_projected = self.center.z + self.unit_z * distance_from_pixel
    return (self.unit_x, self.unit_y, self.unit_z)

  def manufacture_mesh(self, diffuse_color):
    mesh = bpy.data.meshes.new("pixel_{}_{}".format(self.h, self.v))
    obj = bpy.data.objects.new("object_of_pixel_{}_{}".format(self.h, self.v), mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set( state = True, view_layer = None)
    mesh = bpy.context.object.data
    bm = bmesh.new()
    top_left = bm.verts.new(self.top_left_corner.xyz())
    top_right = bm.verts.new(self.top_right_corner.xyz())
    bottom_left = bm.verts.new(self.bottom_left_corner.xyz())
    bottom_right = bm.verts.new(self.bottom_right_corner.xyz())
    center = bm.verts.new(self.center.xyz())
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
    emission_material = bpy.data.materials.new(name="emission_for_{}x{}_pixels")
    emission_material.use_nodes = True
    nodes = emission_material.node_tree.nodes
    for node in nodes:
      nodes.remove(node)
    node_emission = nodes.new(type='ShaderNodeEmission')
    node_emission.inputs[0].default_value = diffuse_color
    node_emission.inputs[1].default_value = 1.0 # strength
    node_emission.location = (0,0)
    node_output = nodes.new(type='ShaderNodeOutputMaterial')   
    node_output.location = (400,0)
    links = emission_material.node_tree.links
    link = links.new(node_emission.outputs[0], node_output.inputs[0])
    obj.data.materials.append(emission_material)

class OpticalSystem():
  def __init__(self, photonics, focal_length, pixel_size, vertical_pixels, horizontal_pixels):
    self.image = Image.open('entropy.png')
    self.focal_length = focal_length
    self.vertical_pixels = vertical_pixels
    self.horizontal_pixels = horizontal_pixels
    self.pixel_size = pixel_size
    self.vertical_size = vertical_pixels * pixel_size
    self.horizontal_size = horizontal_pixels * pixel_size
    self.pixels = [[Pixel(h,v) for v in range(vertical_pixels)] for h in range(horizontal_pixels)]
    self.image_center = Point()
    self.relative_horizontal_half_pixel_size = 0.5 * self.pixel_size / float(self.horizontal_size)
    self.relative_vertical_half_pixel_size = 0.5 * self.pixel_size / float(self.vertical_size)
    self.photonics = photonics
    if photonics == "projectors":
      self.initialize_projectors()
    elif photonics == "sensors":
      self.initialize_sensors()

  def initialize_sensors(self):
    self.sensor_data = bpy.data.cameras.new("sensor_data")
    self.sensors = bpy.data.objects.new("sensor_object", self.sensor_data)
    # parametersize camera
    bpy.context.scene.collection.objects.link(self.sensors)

  def initialize_projectors(self):
    self.projector_data = bpy.data.lights.new(name="projector_data", type='SPOT')
    self.projector_data.shadow_soft_size = 0
    self.projector_data.spot_size = 3.14159
    self.projector_data.cycles.max_bounces = 0
    self.projector_data.use_nodes = True  
    self.projector_data.node_tree.nodes["Emission"].inputs[1].default_value = 1000

    # warp mapping of light
    mapping = self.projector_data.node_tree.nodes.new(type='ShaderNodeMapping')
    mapping.location = 300,0
    mapping.rotation[2] = 3.14159
    mapping.scale[1] = 1.779 # derived from projected image dimensions (1366w / 768h)
    mapping.scale[2] = 0.7272404614 # this controls size of the projection

    # separate xyz
    separate_xyz = self.projector_data.node_tree.nodes.new(type='ShaderNodeSeparateXYZ')
    separate_xyz.location = 900,0
    self.projector_data.node_tree.links.new(mapping.outputs['Vector'], separate_xyz.inputs[0])

    # divide x
    divide_x = self.projector_data.node_tree.nodes.new(type='ShaderNodeMath')
    divide_x.operation = 'DIVIDE'
    divide_x.location = 1200,300
    self.projector_data.node_tree.links.new(separate_xyz.outputs['X'], divide_x.inputs[0])
    self.projector_data.node_tree.links.new(separate_xyz.outputs['Z'], divide_x.inputs[1])

    # add x
    add_x = self.projector_data.node_tree.nodes.new(type='ShaderNodeMath')
    add_x.operation = 'ADD'
    add_x.location = 1500,300
    self.projector_data.node_tree.links.new(divide_x.outputs[0], add_x.inputs[0])

    # divide y
    divide_y = self.projector_data.node_tree.nodes.new(type='ShaderNodeMath')
    divide_y.operation = 'DIVIDE'
    divide_y.location = 1200,0
    self.projector_data.node_tree.links.new(separate_xyz.outputs['Y'], divide_y.inputs[0])
    self.projector_data.node_tree.links.new(separate_xyz.outputs['Z'], divide_y.inputs[1])

    # add y
    add_y = self.projector_data.node_tree.nodes.new(type='ShaderNodeMath')
    add_y.operation = 'ADD'
    add_x.location = 1500,0
    self.projector_data.node_tree.links.new(divide_y.outputs[0], add_y.inputs[0])
    
    # combine xyz
    combine_xyz = self.projector_data.node_tree.nodes.new(type='ShaderNodeCombineXYZ')
    combine_xyz.location = 1800,0
    self.projector_data.node_tree.links.new(add_x.outputs['Value'], combine_xyz.inputs[0])
    self.projector_data.node_tree.links.new(add_y.outputs['Value'], combine_xyz.inputs[1])
    self.projector_data.node_tree.links.new(separate_xyz.outputs['Z'], combine_xyz.inputs[2])

    # texture coordinate
    texture_coordinate = self.projector_data.node_tree.nodes.new(type='ShaderNodeTexCoord')
    texture_coordinate.location = 0,0
    self.projector_data.node_tree.links.new(texture_coordinate.outputs['Normal'], mapping.inputs[0])

    # image texture
    image_texture = self.projector_data.node_tree.nodes.new(type='ShaderNodeTexImage')
    image_texture.image = bpy.data.images.load('entropy.png')

    image_texture.extension = 'CLIP'
    image_texture.location = 2100,0
    self.projector_data.node_tree.links.new(image_texture.outputs['Color'], self.projector_data.node_tree.nodes["Emission"].inputs[0])

    # connect combine with image
    self.projector_data.node_tree.links.new(combine_xyz.outputs['Vector'], image_texture.inputs[0])

    image_texture = self.projector_data.node_tree.nodes.new(type='ShaderNodeMixRGB')

    self.projectors = bpy.data.objects.new(name="projector_object", object_data=self.projector_data)
    bpy.context.scene.collection.objects.link(self.projectors)

  def perceive(self, focal_point, target):
    # focal_point  :: focal point in (x,y,z) at which the optical system is positioned
    # target :: target point in (x,y,z) toward which the optical system is oriented
    self.focal_point = focal_point
    self.target = target
    self.reorient()

  def reorient(self):
    if type(self.focal_point) == type(Point()) and type(self.target) == type(Point()):
      self.compute_image_center()
      self.compute_euler_angles()
      self.compute_pitch_yaw_roll()
      self.compute_xyz_of_boundary_pixels()
      self.orient_xyz_and_unit_vectors_for_all_pixels()
      if self.photonics == "projectors":
        self.projectors.location = (self.focal_point.x, self.focal_point.y, self.focal_point.z)
        self.projectors.rotation_euler = (self.rotation_euler_x, self.rotation_euler_y, self.rotation_euler_z)
        self.projectors.delta_rotation_euler = (0.0, math.radians(-90.0), 0.0)
        self.projectors.delta_scale = (-1.0, 1.0, 1.0) # flips image that is projected horizontally, to match inverted raycasting
      elif self.photonics == "sensors":
        self.sensors.location = (self.focal_point.x, self.focal_point.y, self.focal_point.z)
        self.sensors.rotation_euler = (self.rotation_euler_x, self.rotation_euler_y, self.rotation_euler_z)
        self.sensors.delta_rotation_euler = (0.0, math.radians(-90.0), 0.0)

  def compute_image_center(self):
    focal_ratio = self.focal_length / (self.focal_length + self.focal_point.distance(self.target))
    if focal_ratio == 1.0:
      raise("Distance between focal point and target point cannot be zero; make the optical system point somewhere else than its position.")    
    self.image_center.x = (self.focal_point.x - self.target.x * focal_ratio) / (1 - focal_ratio)
    self.image_center.y = (self.focal_point.y - self.target.y * focal_ratio) / (1 - focal_ratio)
    self.image_center.z = (self.focal_point.z - self.target.z * focal_ratio) / (1 - focal_ratio)

  def compute_pitch_yaw_roll(self):
    # initialize
    self.pitch = 0.0
    self.yaw = 0.0
    self.roll = 0.0 # always 0 degrees, because we never rotate optical system about its lens, radially

    # compute these terms once for use in derivations below
    x_orientation = (self.focal_point.x - self.image_center.x)**2
    y_orientation = (self.focal_point.y - self.image_center.y)**2
    z_orientation = (self.focal_point.z - self.image_center.z)**2

    if x_orientation + y_orientation != 0.0:
      self.yaw = np.arcsin((self.focal_point.x - self.image_center.x) / math.sqrt(x_orientation + y_orientation))

    if x_orientation + y_orientation + z_orientation != 0.0:
      self.pitch = np.arcsin((self.focal_point.z - self.image_center.z) / math.sqrt(x_orientation + y_orientation + z_orientation))

  def compute_euler_angles(self):
    dx = self.target.x - self.image_center.x
    dy = self.target.y - self.image_center.y
    dz = self.target.z - self.image_center.z
    distance_to_target = math.sqrt(dx**2 + dy**2 + dz**2)
    self.rotation_euler_x = 0.0
    self.rotation_euler_y = -1.0 * math.acos(dz / distance_to_target)
    self.rotation_euler_z = math.atan2(dy, dx)

  def compute_xyz_of_boundary_pixels(self):
    horizontal_boundary_x = -1 * cos(self.yaw) * 0.5 * self.horizontal_size + self.image_center.x                # x of rightmost vertically-centered point on sensor
    horizontal_boundary_y = sin(self.yaw) * 0.5 * self.horizontal_size + self.image_center.y                     # y of rightmost vertically-centered point on sensor
    horizontal_boundary_z = self.image_center.z                                                                  # z of rightmost vertically-centered point on sensor
    self.image_horizontal_edge = Point(horizontal_boundary_x, horizontal_boundary_y, horizontal_boundary_z)

    vertical_boundary_x = -1 * sin(self.pitch) * sin(self.yaw) * 0.5 * self.vertical_size + self.image_center.x  # x of topmost horizontally-centered point on sensor
    vertical_boundary_y = -1 * sin(self.pitch) * cos(self.yaw) * 0.5 * self.vertical_size + self.image_center.y  # y of topmost horizontally-centered point on sensor
    vertical_boundary_z = cos(self.pitch) * 0.5 * self.vertical_size + self.image_center.z                            # z of topmost horizontally-centered on sensor
    self.image_vertical_edge = Point(vertical_boundary_x, vertical_boundary_y, vertical_boundary_z)

  def orient_xyz_and_unit_vectors_for_all_pixels(self): 
    relative_horizontal_coordinates = [2 * h / float(self.horizontal_pixels) - 1 for h in range(self.horizontal_pixels + 1)] # indices: 0 = left edge, 1 = left edge + one pixel right, 2 = left edge + two pixels right, ...
    relative_vertical_coordinates = [2 * v / float(self.vertical_pixels) - 1 for v in range(self.vertical_pixels + 1)] # indices: 0 = bottom edge, 1 = bottom edge + one pixel up, 2 = bottom edge + two pixels up, ...
    x_of_horizontal_vectors = [(1 - relative_h) * self.image_center.x + relative_h * self.image_horizontal_edge.x for relative_h in relative_horizontal_coordinates]
    y_of_horizontal_vectors = [(1 - relative_h) * self.image_center.y + relative_h * self.image_horizontal_edge.y for relative_h in relative_horizontal_coordinates]
    z_of_horizontal_vectors = [(1 - relative_h) * self.image_center.z + relative_h * self.image_horizontal_edge.z for relative_h in relative_horizontal_coordinates]
    x_of_vertical_vectors = [(1 - relative_v) * self.image_center.x + relative_v * self.image_vertical_edge.x for relative_v in relative_vertical_coordinates]
    y_of_vertical_vectors = [(1 - relative_v) * self.image_center.y + relative_v * self.image_vertical_edge.y for relative_v in relative_vertical_coordinates]
    z_of_vertical_vectors = [(1 - relative_v) * self.image_center.z + relative_v * self.image_vertical_edge.z for relative_v in relative_vertical_coordinates]   

    for h in range(self.horizontal_pixels):
      left_x_of_horizontal_vector = x_of_horizontal_vectors[h]
      left_y_of_horizontal_vector = y_of_horizontal_vectors[h]
      left_z_of_horizontal_vector = z_of_horizontal_vectors[h]
      right_x_of_horizontal_vector = x_of_horizontal_vectors[h+1]
      right_y_of_horizontal_vector = y_of_horizontal_vectors[h+1]
      right_z_of_horizontal_vector = z_of_horizontal_vectors[h+1]      
      for v in range(self.vertical_pixels):
        bottom_x_of_vertical_vector = x_of_vertical_vectors[v]
        bottom_y_of_vertical_vector = y_of_vertical_vectors[v]
        bottom_z_of_vertical_vector = z_of_vertical_vectors[v]
        top_x_of_vertical_vector = x_of_vertical_vectors[v+1]
        top_y_of_vertical_vector = y_of_vertical_vectors[v+1]
        top_z_of_vertical_vector = z_of_vertical_vectors[v+1]

        bottom_left_x = bottom_x_of_vertical_vector + left_x_of_horizontal_vector - self.image_center.x
        bottom_left_y = bottom_y_of_vertical_vector + left_y_of_horizontal_vector - self.image_center.y
        bottom_left_z = bottom_z_of_vertical_vector + left_z_of_horizontal_vector - self.image_center.z
        self.pixels[h][v].bottom_left_corner = Point(bottom_left_x, bottom_left_y, bottom_left_z)

        bottom_right_x = bottom_x_of_vertical_vector + right_x_of_horizontal_vector - self.image_center.x
        bottom_right_y = bottom_y_of_vertical_vector + right_y_of_horizontal_vector - self.image_center.y
        bottom_right_z = bottom_z_of_vertical_vector + right_z_of_horizontal_vector - self.image_center.z
        self.pixels[h][v].bottom_right_corner = Point(bottom_right_x, bottom_right_y, bottom_right_z)

        top_left_x = top_x_of_vertical_vector + left_x_of_horizontal_vector - self.image_center.x
        top_left_y = top_y_of_vertical_vector + left_y_of_horizontal_vector - self.image_center.y
        top_left_z = top_z_of_vertical_vector + left_z_of_horizontal_vector - self.image_center.z
        self.pixels[h][v].top_left_corner = Point(top_left_x, top_left_y, top_left_z)

        top_right_x = top_x_of_vertical_vector + right_x_of_horizontal_vector - self.image_center.x
        top_right_y = top_y_of_vertical_vector + right_y_of_horizontal_vector - self.image_center.y
        top_right_z = top_z_of_vertical_vector + right_z_of_horizontal_vector - self.image_center.z
        self.pixels[h][v].top_right_corner = Point(top_right_x, top_right_y, top_right_z)

        center_x = round((bottom_left_x + bottom_right_x + top_left_x + top_right_x) / 4.0, 6)
        center_y = round((bottom_left_y + bottom_right_y + top_left_y + top_right_y) / 4.0, 6)
        center_z = round((bottom_left_z + bottom_right_z + top_left_z + top_right_z) / 4.0, 6)
        self.pixels[h][v].center = Point(center_x, center_y, center_z)

        self.pixels[h][v].calculate_unit_vectors_through_focal_point(self.focal_point)

      print("{}, {} : {}, {}, {}".format(h, v, center_x, center_y, center_z))
  
  def highlight_hitpoint(self, location, diffuse_color):
    x = location.x
    y = location.y
    z = location.z 
    print("hitpoint_({},{},{})".format(x,y,z))
    mesh = bpy.data.meshes.new('hitpoint_({},{},{})'.format(x,y,z))
    sphere = bpy.data.objects.new('hitpoint_({},{},{})_object'.format(x,y,z), mesh)
    sphere.location = location
    bpy.context.collection.objects.link(sphere)
    bpy.context.view_layer.objects.active = sphere
    sphere.select_set( state = True, view_layer = None)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, diameter=0.01)
    bm.to_mesh(mesh)
    bm.free()
    bpy.ops.object.modifier_add(type='SUBSURF')
    bpy.ops.object.shade_smooth()

    emission_material = bpy.data.materials.new(name="emission_for_({},{},{})".format(x,y,z))
    emission_material.use_nodes = True
    nodes = emission_material.node_tree.nodes
    for node in nodes:
      nodes.remove(node)
    node_emission = nodes.new(type='ShaderNodeEmission')
    node_emission.inputs[0].default_value = diffuse_color
    node_emission.inputs[1].default_value = 100.0 # strength
    node_emission.location = (0,0)
    node_output = nodes.new(type='ShaderNodeOutputMaterial')   
    node_output.location = (400,0)
    links = emission_material.node_tree.links
    link = links.new(node_emission.outputs[0], node_output.inputs[0])
    sphere.data.materials.append(emission_material)


  def raycasts_from_pixels(self):
    img = Image.open('entropy.png')
    hitpoint_xyz_coordinate = []
    hitpoint_pixel = []
    hitpoint_color = []
    for h in range(self.horizontal_pixels):   
      for v in range(self.vertical_pixels):
        origin = Vector((self.pixels[h][v].center.x, self.pixels[h][v].center.y, self.pixels[h][v].center.z))
        direction = Vector((self.pixels[h][v].unit_x, self.pixels[h][v].unit_y, self.pixels[h][v].unit_z))

        hit, location, normal, face_index, obj, matrix_world = bpy.context.scene.ray_cast(view_layer=bpy.context.view_layer, origin=origin, direction=direction)      
        if hit:
          pixel = img.getpixel((h, v))
          diffuse_color = (pixel[0]/float(255), pixel[1]/float(255), pixel[2]/float(255), 1)
          hitpoint_xyz_coordinate.append(location)
          hitpoint_color.append(diffuse_color)
          hitpoint_pixel.append((h,v))

    return hitpoint_xyz_coordinate, hitpoint_color, hitpoint_pixel

class ObjectModel():
  def __init__(self, filepath=None):
    if filepath:
      self.import_object_to_scan(filepath=filepath)
      self.normalize_position_and_size()

  def import_object_to_scan(self, filepath):
    bpy.ops.wm.collada_import(filepath=filepath)
    for object_in_scene in bpy.context.scene.objects:
      if object_in_scene.type == 'MESH':
        bpy.context.view_layer.objects.active = object_in_scene
        object_in_scene.select_set(state=True)
    bpy.ops.object.join()

  def normalize_position_and_size(self):
    bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='BOUNDS')
    obj = bpy.context.object
    obj.dimensions = (obj.dimensions / max(obj.dimensions))
    obj.location.x = 0.0
    obj.location.y = 0.0
    obj.location.z = 0.0
    self.x_rotation_angle = random.uniform(0, 2*math.pi)
    self.y_rotation_angle = random.uniform(0, 2*math.pi)
    self.z_rotation_angle = random.uniform(0, 2*math.pi)
    obj.rotation_euler = [self.x_rotation_angle, self.y_rotation_angle, self.z_rotation_angle] # random angular rotations about x,y,z axis
    bpy.context.scene.update() 

class Environment():
  def __init__(self):
    self.create_mesh()
    self.create_materials()

  def create_mesh(self):
    self.mesh = bpy.data.meshes.new("vinyl_backdrop")
    self.obj = bpy.data.objects.new("vinyl_backdrop_object", self.mesh)
    bpy.context.collection.objects.link(self.obj)
    bpy.context.view_layer.objects.active = self.obj
    self.obj.select_set( state = True, view_layer = None)
    mesh = bpy.context.object.data
    bm = bmesh.new()
    top_left = bm.verts.new((-100,-4,100))
    top_right = bm.verts.new((100,-4,100))
    bottom_left = bm.verts.new((-100,-4,-100))
    bottom_right = bm.verts.new((100,-4,-100))
    center = bm.verts.new((0,-4, 0))
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

  def create_materials(self):
    self.vinyl_material = bpy.data.materials.new(name="vinyl_backdrop_material")
    self.vinyl_material.use_nodes = True
    nodes = self.vinyl_material.node_tree.nodes
    for node in nodes:
      nodes.remove(node)
    self.vinyl = nodes.new(type='ShaderNodeBsdfPrincipled')
    self.vinyl.inputs['Sheen Tint'].default_value = 0.2
    self.vinyl.inputs['Roughness'].default_value = 0.2
    self.vinyl.inputs['Base Color'].default_value = (1,1,1,1)
    self.vinyl.location = (0,0)
    node_output = nodes.new(type='ShaderNodeOutputMaterial')   
    node_output.location = (400,0)
    links = self.vinyl_material.node_tree.links
    link = links.new(self.vinyl_material.outputs[0], node_output.inputs[0])
    self.obj.data.materials.append(self.vinyl_material)


environment = Environment()
model = ObjectModel(filepath="phone.dae")

lasers = OpticalSystem(photonics="projectors", focal_length=0.01127, pixel_size=0.000006, vertical_pixels=768, horizontal_pixels=1366) # 64 x 114 / 768 x 1366 -> distance / width = 0.7272404614
camera = OpticalSystem(photonics="sensors", focal_length=0.02400, pixel_size=0.00000429, vertical_pixels=3456, horizontal_pixels=5184) # 3456 x 5184

lasers.perceive(focal_point=Point(x=0.0, y=2.0, z=0.0), target=Point(x=0.0, y=0.0, z=0.0))
camera.perceive(focal_point=Point(x=0.0, y=2.0, z=0.1), target=Point(x=0.0, y=0.0, z=0.0))


hitpoint_locations, hitpoint_colors, hitpoint_image_origins = lasers.raycasts_from_pixels()