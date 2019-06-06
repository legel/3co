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
import pickle
from os import listdir, path

#bpy.ops.wm.open_mainfile(filepath="empty.blend")

bpy.context.scene.render.engine = 'CYCLES'

try:
  bpy.context.preferences.addons['cycles'].preferences.get_devices()
  print(bpy.context.preferences.addons['cycles'].preferences.get_devices())
  bpy.context.scene.cycles.device = 'GPU'
  bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
  bpy.context.preferences.addons['cycles'].preferences.devices[0].use = True
except TypeError:
  pass

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

class Triangle():
  def __init__(self, a, b, c):
    # each argument is a Point(x,y,z)
    self.a = a
    self.b = b
    self.c = c
    self.solve_distances()
    self.solve_angles_with_law_of_cosines()

  def solve_distances(self):
    self.distance_c = self.a.distance(self.b)
    self.distance_a = self.b.distance(self.c)
    self.distance_b = self.c.distance(self.a)

  def solve_angles_with_law_of_cosines(self):
    if self.distance_b != 0.0 and self.distance_c != 0.0:
      raw = (self.distance_b**2 + self.distance_c**2 - self.distance_a**2) / (2 * self.distance_b * self.distance_c)
      self.angle_a = math.acos(self.normalize_floating_point(raw))
    else:
      self.angle_a = 0.0

    if self.distance_c != 0.0 and self.distance_a != 0.0:
      raw = (self.distance_c**2 + self.distance_a**2 - self.distance_b**2) / (2 * self.distance_c * self.distance_a)
      self.angle_b = math.acos(self.normalize_floating_point(raw))
    else:
      self.angle_b = 0.0

    if self.distance_a != 0.0 and self.distance_b != 0.0:
      raw = (self.distance_a**2 + self.distance_b**2 - self.distance_c**2) / (2 * self.distance_a * self.distance_b)
      self.angle_c = math.acos(self.normalize_floating_point(raw))
    else:
      self.angle_c = 0.0

  def normalize_floating_point(self, i):
    if i < -1.0:
      return -1.0
    elif i > 1.0:
      return 1.0
    else:
      return i

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


class Photonics():
  def __init__(self, projectors_or_sensors, focal_point, focal_length, pixel_size, vertical_pixels, horizontal_pixels, hardcode_field_of_view=False, image=None, target=None):
    # projectors_or_sensors  :: value is a string "projectors" or "sensors" to describe if pixels should emit or sense photons
    # focal_point  :: focal point in (x,y,z) at which the optical system is positioned
    # focal_length  :: focal length in meters
    # pixel_size  :: size of one side of pixel in real space in meters, assuming pixel is a square
    # vertical_pixels  :: number of vertical pixels
    # horionztal_pixels :: number of horizontal pixels  
    # hardcode_field_of_view :: for testing purposes, fix FOV regardless of provided number of pixels
    # image :: for projectors, .png image to project 
    time_start = time.time()
    self.focal_point = focal_point
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
    self.projectors_or_sensors = projectors_or_sensors
    self.hardcode_field_of_view = hardcode_field_of_view
    self.highlighted_hitpoints = []
    self.sampled_hitpoint_pixels = []
    self.target = target 
    if projectors_or_sensors == "projectors":
      self.image = image
      self.initialize_projectors()
    elif projectors_or_sensors == "sensors":
      self.initialize_sensors()
    time_end = time.time()
    print("Launched {} in {} seconds".format(self.projectors_or_sensors, round(time_end - time_start, 4)))

  def initialize_sensors(self):
    self.sensor_data = bpy.data.cameras.new("sensor_data")
    self.sensors = bpy.data.objects.new("Sensor", self.sensor_data)
    bpy.context.scene.collection.objects.link(self.sensors)
    bpy.context.scene.camera = self.sensors
    #bpy.data.cameras["sensor_data"].clip_start = 0.01 # meters
    bpy.data.cameras["sensor_data"].lens = self.focal_length * 1000 # millimeters
    if self.hardcode_field_of_view:
      bpy.data.cameras["sensor_data"].sensor_width = 0.00000429 * 5184 * 1000 # millimeters of pixel size x horizontal pixels on Canon 1300D
      self.horizontal_size = 0.00000429 * 5184
      self.vertical_size = 0.00000429 * 3456
    else:
      bpy.data.cameras["sensor_data"].sensor_width = self.horizontal_size * 1000 # millimeters
    bpy.data.scenes["Scene"].render.resolution_x = self.horizontal_pixels
    bpy.data.scenes["Scene"].render.resolution_y = self.vertical_pixels


  def initialize_projectors(self):
    self.projector_data = bpy.data.lights.new(name="projector_data", type='SPOT')
    self.projector_data.shadow_soft_size = 0
    self.projector_data.spot_size = 3.14159
    self.projector_data.cycles.max_bounces = 0
    self.projector_data.use_nodes = True  

    lighting_strength = min(max(np.random.normal(loc=1000, scale=1500), 200), 5000)
    self.projector_data.node_tree.nodes["Emission"].inputs[1].default_value = lighting_strength # W/m^2

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

    image_texture.image = bpy.data.images.load(self.image)
    self.image_to_project = image_texture.image

    image_texture.extension = 'CLIP'
    image_texture.location = 2100,0
    self.projector_data.node_tree.links.new(image_texture.outputs['Color'], self.projector_data.node_tree.nodes["Emission"].inputs[0])

    # connect combine with image
    self.projector_data.node_tree.links.new(combine_xyz.outputs['Vector'], image_texture.inputs[0])

    image_texture = self.projector_data.node_tree.nodes.new(type='ShaderNodeMixRGB')

    self.projectors = bpy.data.objects.new(name="Projector", object_data=self.projector_data)
    bpy.context.scene.collection.objects.link(self.projectors)

  def project(self, filepath):
    self.filepath_of_image_to_project = filepath
    self.image_to_project = bpy.data.images.load(filepath)

  def expand_plane_of_sensor(self, expansion=1.0): # expansion is a multiplier of the size of the sensor plane
    min_h = 0
    min_v = 0
    max_h = self.horizontal_pixels - 1
    max_v = self.vertical_pixels - 1
    min_h_min_v_corner = self.pixels[min_h][min_v].center
    min_h_max_v_corner = self.pixels[min_h][max_v].center
    max_h_min_v_corner = self.pixels[max_h][min_v].center
    max_h_max_v_corner = self.pixels[max_h][max_v].center
    print("4 corners as computed from raw calculations:")
    print(min_h_min_v_corner.xyz())
    print(min_h_max_v_corner.xyz())
    print(max_h_min_v_corner.xyz())
    print(max_h_max_v_corner.xyz())
    self.highlight_hitpoint(min_h_min_v_corner.xyz(), (1,0,0,1))
    self.highlight_hitpoint(min_h_max_v_corner.xyz(), (0,1,0,1))
    self.highlight_hitpoint(max_h_min_v_corner.xyz(), (0,0,1,1))
    self.highlight_hitpoint(max_h_max_v_corner.xyz(), (1,1,1,1))

    expanded_corners = []
    for corner in [min_h_min_v_corner, min_h_max_v_corner, max_h_min_v_corner, max_h_max_v_corner]:
      expanded_x = (1 - expansion) * self.image_center.x + expansion * corner.x
      expanded_y = (1 - expansion) * self.image_center.y + expansion * corner.y
      expanded_z = (1 - expansion) * self.image_center.z + expansion * corner.z
      expanded_corners.append(Point(expanded_x, expanded_y, expanded_z))
    self.create_mesh_from_corners(expanded_corners)

  def create_mesh_from_corners(self, corners):
    # input is a list of 4 corners, each of which is a Point(x,y,z)
    # output is simulation of a rectangular mesh with faces for triangles connecting 4 corners 
    corner_1, corner_2, corner_3, corner_4 = corners
    plane = bpy.data.meshes.new("sensor_plane")
    obj = bpy.data.objects.new("sensor_plane_object", plane)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(state = True, view_layer = None)
    mesh = bpy.context.object.data
    bm = bmesh.new()
    top_left = bm.verts.new((corner_1.x, corner_1.y, corner_1.z))
    top_right = bm.verts.new((corner_2.x, corner_2.y, corner_2.z))
    bottom_left = bm.verts.new((corner_3.x, corner_3.y, corner_3.z))
    bottom_right = bm.verts.new((corner_4.x, corner_4.y, corner_4.z))
    center = bm.verts.new((self.image_center.x, self.image_center.y, self.image_center.z))
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

  def reorient(self, orientation_index=0):
    time_start = time.time()
    if type(self.focal_point) == type(Point()) and type(self.target) == type(Point()):
      self.compute_image_center()
      self.compute_euler_angles()
      self.compute_xyz_of_boundary_pixels()
      self.orient_xyz_and_unit_vectors_for_all_pixels()
      adjusted_euler_z = self.rotation_euler_z * -1.0 # to correct for a notational difference between rendering engine and notes
      if self.projectors_or_sensors == "projectors":
        self.projectors.location = (self.focal_point.x, self.focal_point.y, self.focal_point.z)
        self.projectors.rotation_euler = (self.rotation_euler_x, self.rotation_euler_y, adjusted_euler_z)
        self.projectors.delta_scale = (-1.0, 1.0, 1.0) # flips image that is projected horizontally, to match inverted raycasting
      elif self.projectors_or_sensors == "sensors":
        self.sensors.location = (self.focal_point.x, self.focal_point.y, self.focal_point.z)
        self.sensors.rotation_euler = (self.rotation_euler_x, self.rotation_euler_y, adjusted_euler_z)
        self.expand_plane_of_sensor()
    time_end = time.time()
    print("Orientations of {} computed in {} seconds".format(self.projectors_or_sensors, round(time_end - time_start, 4)))
    self.save_metadata(orientation_index)

  def save_metadata(self, orientation_index):
    with open("xyz_of_{}_{}_pixels".format(self.projectors_or_sensors, orientation_index), "w") as metadata:
      metadata.write("(h,v):(x,y,z)\n")
      for h in [0, self.horizontal_pixels - 1]:    
        for v in [0, self.vertical_pixels - 1]: 
          x,y,z = self.pixels[h][v].center.xyz()
          metadata.write("{},{}:{},{},{}\n".format(h,v,x,y,z))

  def compute_image_center(self):
    focal_ratio = self.focal_length / (self.focal_length + self.focal_point.distance(self.target))
    if focal_ratio == 1.0:
      raise("Distance between focal point and target point cannot be zero; make the optical system point somewhere else than its position.")    
    self.image_center.x = (self.focal_point.x - self.target.x * focal_ratio) / (1 - focal_ratio)
    self.image_center.y = (self.focal_point.y - self.target.y * focal_ratio) / (1 - focal_ratio)
    self.image_center.z = (self.focal_point.z - self.target.z * focal_ratio) / (1 - focal_ratio)

  def compute_euler_angles(self):
    # compute euler angles from default angle of image top pointing to +y, image right pointing to +x, image view pointing to -z
    o = self.image_center
    t = self.target
    b = Point(o.x, o.y, o.z - 2 * o.distance(t))

    euler_x_triangle = Triangle(a = o, b = t, c = b)
    self.rotation_euler_x = euler_x_triangle.angle_a
    if self.rotation_euler_x == 0.0:
      if o.z < t.z:
        self.rotation_euler_x == math.radians(180.0)

    target_point = Point(t.x, t.y, 0)
    origin_point = Point(o.x, o.y, 0)
    base_point = Point(t.x, t.y - 1.0, 0)

    euler_z_triangle = Triangle(a = target_point, b = origin_point, c = base_point)
    self.rotation_euler_z = euler_z_triangle.angle_a   
    if o.x > t.x:
      self.rotation_euler_z = self.rotation_euler_z * -1.0 
    if self.rotation_euler_z == 0.0:
      if o.y > t.y:
        self.rotation_euler_z == math.radians(180.0)

    self.rotation_euler_y = 0.0 # definition

  def compute_xyz_of_boundary_pixels(self):
    pitch = self.rotation_euler_x 
    yaw =  self.rotation_euler_z

    horizontal_boundary_x = self.image_center.x - cos(yaw) * 0.5 * self.horizontal_size # x of rightmost vertically-centered point on sensor
    horizontal_boundary_y = self.image_center.y + sin(yaw) * 0.5 * self.horizontal_size # y of rightmost vertically-centered point on sensor
    horizontal_boundary_z = self.image_center.z  # z of rightmost vertically-centered point on sensor
    self.image_horizontal_edge = Point(horizontal_boundary_x, horizontal_boundary_y, horizontal_boundary_z)

    vertical_boundary_x = self.image_center.x - cos(pitch) * sin(yaw) * 0.5 * self.vertical_size # x of topmost horizontally-centered point on sensor
    vertical_boundary_y = self.image_center.y - cos(yaw) * cos(pitch) * 0.5 * self.vertical_size # y of topmost horizontally-centered point on sensor
    vertical_boundary_z = self.image_center.z - sin(pitch) * 0.5 * self.vertical_size # z of topmost horizontally-centered on sensor
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

#    for h in range(self.horizontal_pixels):
    for h in [0, self.horizontal_pixels - 1]:

      left_x_of_horizontal_vector = x_of_horizontal_vectors[h]
      left_y_of_horizontal_vector = y_of_horizontal_vectors[h]
      left_z_of_horizontal_vector = z_of_horizontal_vectors[h]
      right_x_of_horizontal_vector = x_of_horizontal_vectors[h+1]
      right_y_of_horizontal_vector = y_of_horizontal_vectors[h+1]
      right_z_of_horizontal_vector = z_of_horizontal_vectors[h+1]      
      for v in [0, self.vertical_pixels - 1]: 
#      for v in range(self.vertical_pixels):

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

        center_x = (bottom_left_x + bottom_right_x + top_left_x + top_right_x) / 4.0
        center_y = (bottom_left_y + bottom_right_y + top_left_y + top_right_y) / 4.0
        center_z = (bottom_left_z + bottom_right_z + top_left_z + top_right_z) / 4.0
        self.pixels[h][v].center = Point(center_x, center_y, center_z)

        self.pixels[h][v].calculate_unit_vectors_through_focal_point(self.focal_point)
      #print("{}, {} : {}, {}, {}".format(h, v, center_x, center_y, center_z))
  
  def highlight_hitpoint(self, location, diffuse_color):
    x = location[0]
    y = location[1]
    z = location[2] 
    mesh = bpy.data.meshes.new('hitpoint_({},{},{})'.format(x,y,z))
    sphere = bpy.data.objects.new('hitpoint_({},{},{})_object'.format(x,y,z), mesh)
    sphere.location = location
    bpy.context.collection.objects.link(sphere)
    bpy.context.view_layer.objects.active = sphere
    sphere.select_set( state = True, view_layer = None)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, diameter=0.0025)
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

    #nodes["Output"] = node_output

    principled_node = nodes.new("ShaderNodeBsdfPrincipled")
    principled_node.location = (100, 250)
    #nodes["Principled"] = principled_node

    links = material.node_tree.links

    links.new(principled_node.outputs[0], node_output.inputs[0])

    principled_node.inputs['Specular'].default_value = 0.015
    principled_node.inputs['Base Color'].default_value = diffuse_color


    sphere.data.materials.append(material)
    if self.projectors_or_sensors == "projectors":
      sphere.hide_viewport = True
      self.highlighted_hitpoints.append(sphere)

  def measure_raycasts_from_pixels(self):
    time_start = time.time()

    min_h = 0
    min_v = 0
    max_h = self.horizontal_pixels - 1
    max_v = self.vertical_pixels - 1

    for h in [0, self.horizontal_pixels - 1]: #range(self.horizontal_pixels):   
      for v in [0, self.vertical_pixels - 1]: #range(self.vertical_pixels):

        origin = Vector((self.pixels[h][v].center.x, self.pixels[h][v].center.y, self.pixels[h][v].center.z))
        direction = Vector((self.pixels[h][v].unit_x, self.pixels[h][v].unit_y, self.pixels[h][v].unit_z))
        hit, location, normal, face_index, obj, matrix_world = bpy.context.scene.ray_cast(view_layer=bpy.context.view_layer, origin=origin, direction=direction)
        if not hit:
          print("No hitpoint for raycast from pixel ({},{})".format(h, v))
        self.pixels[h][v].hitpoint = Point(location[0], location[1], location[2])
      #print("{} pixel ({},{}) with hitpoint {}".format(self.projectors_or_sensors, h, v, self.pixels[h][v].hitpoint.xyz()))

    time_end = time.time()
    print("Raycasts of {} computed in {} seconds".format(self.projectors_or_sensors, round(time_end - time_start, 4)))

class Model():
  def __init__(self, filepath=None):
    if filepath:
      self.import_object_to_scan(filepath=filepath)
      self.dimensions = bpy.context.object.dimensions
      self.resample_parameters()

  def import_object_to_scan(self, filepath):
    bpy.ops.wm.collada_import(filepath=filepath)
    for object_in_scene in bpy.context.scene.objects:
      if object_in_scene.type == 'MESH':
        bpy.context.view_layer.objects.active = object_in_scene
        self.object_name = object_in_scene.name
        object_in_scene.select_set(state=True)
    bpy.ops.object.join()
    bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='BOUNDS')
    self.model_object = bpy.context.object
    print("IMPORTED OBJECT STUFF: 1,2")
    bpy.context.object.name = "Model"
    print(bpy.context.object)
    print(bpy.context.object.location)


  def resample_parameters(self):
      self.resample_orientation()
      self.resample_size()
      self.resample_position()
      self.resample_materials()    

  def resample_orientation(self):
    obj = bpy.context.object
    self.x_rotation_angle = random.uniform(0, 2*math.pi)
    self.y_rotation_angle = random.uniform(0, 2*math.pi)
    self.z_rotation_angle = random.uniform(0, 2*math.pi)
    obj.rotation_euler = [self.x_rotation_angle, self.y_rotation_angle, self.z_rotation_angle] # random angular rotations about x,y,z axis
    bpy.context.scene.update() 

  def resample_size(self):
    scale_factor = max(np.random.normal(loc=1.25, scale=0.5), 0.25)
    bpy.context.object.dimensions = (self.dimensions * scale_factor / max(self.dimensions))
    bpy.context.scene.update() 

  def resample_position(self):
    obj = bpy.context.object
    obj.location.x = max(min(np.random.normal(loc=0.0, scale=0.05), 0.15), -0.15)
    obj.location.y = max(min(np.random.normal(loc=0.0, scale=0.05), 0.15), -0.15)
    obj.location.z = max(min(np.random.normal(loc=0.0, scale=0.05), 0.15), -0.15)
    bpy.context.scene.update() 

  def resample_materials(self):
    obj = bpy.context.object
    for material_slot in obj.material_slots:
      m = material_slot.material
      # reset nodes
      m.blend_method = 'BLEND'
      m.use_nodes = True
      nodes = m.node_tree.nodes
      for node in nodes:
        nodes.remove(node)
      shader = nodes.new(type='ShaderNodeBsdfPrincipled')
      shader.location = (0,0)

      # parameterization by *The Principled Shader* (insert Blender guru accent)
      red = random.uniform(0, 1)
      green = random.uniform(0, 1)
      blue = random.uniform(0, 1)
      alpha = min(np.random.normal(loc=1.0, scale=0.15), 1.0)
      shader.inputs['Base Color'].default_value = (red, green, blue, alpha)

      shader.inputs['Metallic'].default_value = np.random.choice(a=[0.0, 1.0], p=[0.80, 0.20]) # left is choice, right is probabilty

      # weighted mixture of gaussians
      ior_guassian_a = np.random.normal(loc=1.45, scale=0.15)
      ior_gaussian_b = np.random.normal(loc=1.45, scale=0.5)
      ior = min(np.random.choice(a=[ior_guassian_a, ior_gaussian_b], p=[0.80, 0.20]), 1.0)
      shader.inputs['IOR'].default_value = ior

      shader.inputs['Specular'].default_value = ((ior-1.0)/(ior+1.0))**2 / 0.08 # from "special case of Fresnel formula" as in https://docs.blender.org/manual/en/dev/render/cycles/nodes/types/shaders/principled.html

      # weighted mixture of guassians
      transmission_guassian_a = np.random.normal(loc=0.1, scale=0.3)
      transmission_gaussian_b = np.random.normal(loc=0.9, scale=0.3)
      transmission = max(min(np.random.choice(a=[transmission_guassian_a, transmission_gaussian_b], p=[0.75, 0.25]), 1.0), 0.0)
      shader.inputs['Transmission'].default_value = transmission

      shader.inputs['Transmission Roughness'].default_value = random.uniform(0, 1)
      shader.inputs['Roughness'].default_value = random.uniform(0, 1)

      shader.inputs['Anisotropic'].default_value = max(min(np.random.normal(loc=0.1, scale=0.3), 1.0), 0.0)
      shader.inputs['Anisotropic Rotation'].default_value = random.uniform(0,1)

      shader.inputs['Sheen'].default_value = max(min(np.random.choice(a=[0.0, np.random.normal(loc=0.5, scale=0.25)], p=[0.9, 0.1]), 1.0), 0.0)

      # infrastructure
      node_output = nodes.new(type='ShaderNodeOutputMaterial')   
      node_output.location = (400,0)
      links = m.node_tree.links
      link = links.new(shader.outputs[0], node_output.inputs[0])

    bpy.context.scene.update() 


class Environment():
  def __init__(self, model="phone.dae"):
    self.resample_environment(model)

  def resample_environment(self, model):
    self.add_model(model_filepath=model)
    self.ambient_lighting()
    self.create_mesh()
    self.create_materials()
    self.index_materials_of_faces()

  def index_materials_of_faces(self):
    objects = {}
    for i, obj in enumerate(bpy.data.objects):
      obj.select_set( state = False, view_layer = None)
      print("({}) {}".format(i,obj.name))
      if obj.name == "Environment":
        objects["Environment"] = obj
      elif obj.name == "Model":
        objects["Model"] = obj 

    objects["Model"].select_set( state = True, view_layer = None)
    self.model_materials = {}
    #active_object = bpy.context.active_object

    for face in objects["Model"].data.polygons:  # iterate over faces
      material = objects["Model"].material_slots[face.material_index].material
      self.model_materials[face.index] = material
      # print("Model...")
      # print(material.name)
      # r = material.diffuse_color[0]
      # g = material.diffuse_color[1]
      # b = material.diffuse_color[2]
      # a = material.diffuse_color[3]
      # print("({},{},{},{})".format(r,g,b,a))

    objects["Model"].select_set( state = False, view_layer = None)
    objects["Environment"].select_set( state = True, view_layer = None)

    self.environment_materials = {}
    for face in objects["Environment"].data.polygons:  # iterate over faces
      material = objects["Environment"].material_slots[face.material_index].material
      self.environment_materials[face.index] = material
      # print("Environment...")
      # print(material.name)
      # r = material.diffuse_color[0]
      # g = material.diffuse_color[1]
      # b = material.diffuse_color[2]
      # a = material.diffuse_color[3]
      # print("({},{},{},{})".format(r,g,b,a))

  def delete_environment(self):

    objects = {}
    for i, obj in enumerate(bpy.data.objects):
      obj.select_set( state = False, view_layer = None)
      print("({}) {}".format(i,obj.name))
      if obj.name == "Environment":
        objects["Environment"] = obj
      elif obj.name == "Model":
        objects["Model"] = obj 
      elif obj.name == "Ambient Light":
        objects["Ambient Light"] = obj


    objects["Environment"].select_set(True)
    bpy.ops.object.delete()

    objects["Model"].select_set(True)
    bpy.ops.object.delete()

    objects["Ambient Light"].select_set(True)
    bpy.ops.object.delete() 


  def update(self, model):
    self.delete_environment()
    self.resample_environment(model)

  def setup_preferences(self):
    bpy.ops.wm.read_factory_settings(use_empty=True) # initialize empty world, removing default objects

  def add_model(self,model_filepath):
    self.model = Model(model_filepath)

  def ambient_lighting(self):
    # add light
    light = bpy.data.lights.new(name="sun", type='SUN')
    light.use_nodes = True  

    light.node_tree.nodes["Emission"].inputs[1].default_value = min(max(np.random.normal(loc=0.1, scale=0.2), 0.01), 0.4)
    self.light = bpy.data.objects.new(name="Ambient Light", object_data=light)

    x = np.random.normal(loc=0.0, scale=1.0)
    y = np.random.normal(loc=0.0, scale=1.0)
    z = np.random.normal(loc=10.0, scale=2.5)

    self.light.location = (x, y, z)
    bpy.context.scene.collection.objects.link(self.light)

  def create_mesh(self):
    mesh = bpy.data.meshes.new("vinyl_backdrop")
    self.mesh = bpy.data.objects.new("Environment", mesh)
    bpy.context.collection.objects.link(self.mesh)
    bpy.context.view_layer.objects.active = self.mesh
    self.mesh.select_set( state = True, view_layer = None)

    mesh = bpy.context.object.data
    bm = bmesh.new()

    self.distance_from_origin = max(np.random.normal(loc=-1.0, scale=0.25), -0.25)

    top_left = bm.verts.new((-100, 100, self.distance_from_origin))
    top_right = bm.verts.new((100, 100, self.distance_from_origin))
    bottom_left = bm.verts.new((-100,-100, self.distance_from_origin))
    bottom_right = bm.verts.new((100,-100, self.distance_from_origin))
    center = bm.verts.new((0, 0, self.distance_from_origin))

    self.x_rotation_angle = np.random.normal(loc=0.0, scale=math.radians(5.0))
    self.y_rotation_angle = np.random.normal(loc=0.0, scale=math.radians(5.0))
    self.z_rotation_angle = np.random.normal(loc=0.0, scale=math.radians(5.0))
    self.mesh.rotation_euler = [self.x_rotation_angle, self.y_rotation_angle, self.z_rotation_angle] # angular rotations about x,y,z axis

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

    bpy.context.scene.update() 

  def create_materials(self):
    self.vinyl_material = bpy.data.materials.new(name="vinyl_backdrop_material")
    self.vinyl_material.use_nodes = True
    nodes = self.vinyl_material.node_tree.nodes
    for node in nodes:
      nodes.remove(node)
    self.vinyl = nodes.new(type='ShaderNodeBsdfPrincipled')
    self.vinyl.inputs['Sheen'].default_value = 1.0
    self.vinyl.inputs['Sheen Tint'].default_value = 0.8
    self.vinyl.inputs['Roughness'].default_value = 0.2
    self.vinyl.inputs['Base Color'].default_value = (1,1,1,1)

    # parameterization by *The Principled Shader* (insert Blender guru accent)
    red = min(np.random.normal(loc=0.995, scale=0.03), 1.0)
    green = min(np.random.normal(loc=0.995, scale=0.03), 1.0)
    blue = min(np.random.normal(loc=0.995, scale=0.03), 1.0)
    alpha = min(np.random.normal(loc=0.995, scale=0.03), 1.0)
    self.vinyl_material.diffuse_color = (red,green,blue,alpha)
    self.vinyl.inputs['Base Color'].default_value = (red, green, blue, alpha)

    self.vinyl.inputs['Metallic'].default_value = 0.0 

    # weighted mixture of gaussians
    ior = np.random.normal(loc=1.45, scale=0.02)
    self.vinyl.inputs['IOR'].default_value = ior
    self.vinyl.inputs['Specular'].default_value = ((ior-1.0)/(ior+1.0))**2 / 0.08 # from "special case of Fresnel formula" as in https://docs.blender.org/manual/en/dev/render/cycles/nodes/types/shaders/principled.html

    # weighted mixture of guassians
    self.vinyl.inputs['Transmission'].default_value = max(np.random.normal(loc=0.1, scale=0.1), 0.0)

    self.vinyl.inputs['Transmission Roughness'].default_value = random.uniform(0, 1)
    self.vinyl.inputs['Roughness'].default_value = min(np.random.normal(loc=0.8, scale=0.1),1.0)

    self.vinyl.inputs['Sheen'].default_value = max(min(np.random.normal(loc=0.6, scale=0.2), 1.0), 0.0)


    self.vinyl.location = (0,0)
    node_output = nodes.new(type='ShaderNodeOutputMaterial')   
    node_output.location = (400,0)
    links = self.vinyl_material.node_tree.links
    link = links.new(self.vinyl.outputs[0], node_output.inputs[0])
    self.mesh.data.materials.append(self.vinyl_material)


class Scanner():
  def __init__(self, sensors, projectors=None, environment=None):
    self.environment = environment
    self.sensors = sensors
    self.projectors = projectors

  def scan(self, location=Point(0.0, 0.0, 0.0), counter=0, precomputed=False):
    # if projectors and/or sensors have a new target, reorient
    if self.projectors.target != location:
      self.projectors.target = location
      self.projectors.reorient(orientation_index=counter)
    if self.sensors.target != location:
      self.sensors.target = location
      self.sensors.reorient(orientation_index=counter)

    if self.projectors:
      self.localizations = []
      self.projectors.measure_raycasts_from_pixels()

    self.render("goddess.png")

    if self.projectors: 
      self.localize_projections_in_sensor_plane()

  def render(self, filename):
    print("Rendering...")
    time_start = time.time()
    bpy.data.scenes["Scene"].render.filepath = filename
    bpy.ops.render.render( write_still=True )
    time_end = time.time()
    print("Rendered image in {} seconds".format(round(time_end - time_start, 4)))


  def save_data(self):
    print("Pickling variable data...")

    with open("sensors.txt", 'wb') as f:
      pickle.dump(self.sensors, f)

    with open("projectors.txt", 'wb') as f:
      pickle.dump(self.projectors, f)
    

  def localize_projections_in_sensor_plane(self):
    time_start = time.time()

    object_name = self.environment.model.object_name
    bpy.data.objects[object_name].hide_viewport = True
    self.environment.mesh = hide_viewport = True

    img = Image.open(self.projectors.image)

#    for i in range(100):
#      h = int(random.uniform(0, self.projectors.horizontal_pixels))
#      v = int(random.uniform(0, self.projectors.vertical_pixels))
#      self.projectors.sampled_hitpoint_pixels.append((h,v))

    # for h in range(self.projectors.horizontal_pixels):   
    #   for v in range(self.projectors.vertical_pixels):

    for h in [0, self.projectors.horizontal_pixels - 1]: #range(self.horizontal_pixels):   
      for v in [0, self.projectors.vertical_pixels - 1]: 


        origin = self.projectors.pixels[h][v].hitpoint
        destination = self.sensors.focal_point
        distance = origin.distance(destination)
        unit_x = (destination.x - origin.x) / distance
        unit_y = (destination.y - origin.y) / distance
        unit_z = (destination.z - origin.z) / distance
        direction = Vector((unit_x, unit_y, unit_z))

        # move a millimeter closer to sensor, to escape object originally hit
        origin.x = origin.x + unit_x * 0.001
        origin.y = origin.y + unit_y * 0.001
        origin.z = origin.z + unit_z * 0.001

        hit, location, normal, face_index, obj, matrix_world = bpy.context.scene.ray_cast(view_layer=bpy.context.view_layer, origin=Vector((origin.x, origin.y, origin.z)), direction=direction)

        if not hit:
          print("No secondary hitpoint on sensor plane for raycast from hitpoint of projected pixel ({},{})".format(h, v))
          print("Try expanding the size of the sensor plane".format(h, v))

        if obj == self.environment.mesh:
          print("Hit the backdrop...")
          material = self.environment.environment_materials[face_index] # gather information about textures... 
          print("Color of material there: {}".format(material.diffuse_color))
        elif obj == self.environment.model:
          print("Hit the model...")
          material = self.environment.model_materials[face_index]
          print("Color of material there: {}".format(material.diffuse_color))

        self.projectors.pixels[h][v].hitpoint_in_sensor_plane = Point(location[0], location[1], location[2])
        #print("pixel ({},{}) hitpoint {} on sensor at {}".format(h, v, self.projectors.pixels[h][v].hitpoint.xyz(), self.projectors.pixels[h][v].hitpoint_in_sensor_plane.xyz()))

    self.localization_in_sensor_coordinates()
    for hitpoint in self.projectors.highlighted_hitpoints:
      hitpoint.hide_viewport = False
    bpy.data.objects[object_name].hide_viewport = False
    self.environment.mesh = hide_viewport = False

    time_end = time.time()
    print("Computed localizations in {} seconds".format(round(time_end - time_start, 4)))


  def localization_in_sensor_coordinates(self):
    min_h = 0
    min_v = 0
    max_h = self.sensors.horizontal_pixels - 1
    max_v = self.sensors.vertical_pixels - 1

    origin = self.sensors.pixels[min_h][min_v].center
    h_edge = self.sensors.pixels[max_h][min_v].center
    v_edge = self.sensors.pixels[min_h][max_v].center

    distance_v_o = origin.distance(v_edge)
    distance_h_o = origin.distance(h_edge)

    unit_x_v = (v_edge.x - origin.x) / distance_v_o
    unit_y_v = (v_edge.y - origin.y) / distance_v_o
    unit_z_v = (v_edge.z - origin.z) / distance_v_o

    unit_x_h = (h_edge.x - origin.x) / distance_h_o
    unit_y_h = (h_edge.y - origin.y) / distance_h_o
    unit_z_h = (h_edge.z - origin.z) / distance_h_o

    normalizing_h_denominator = distance_h_o * unit_y_h
    normalizing_v_denominator = distance_v_o * (unit_x_v - (unit_x_h / unit_y_h) * unit_y_v )
    y_v = unit_y_v * distance_v_o
    unit_h_xy = unit_x_h / unit_y_h 

    img = Image.open(self.projectors.image)
  
#    for h in range(self.projectors.horizontal_pixels):   
#      for v in range(self.projectors.vertical_pixels):

    for h in [0, self.projectors.horizontal_pixels - 1]:   
      for v in [0, self.projectors.vertical_pixels - 1]: 

        hitpoint = self.projectors.pixels[h][v].hitpoint_in_sensor_plane

        numerator_relative_v = ((hitpoint.x - origin.x) - (hitpoint.y - origin.y) * unit_h_xy)
        relative_v = numerator_relative_v / normalizing_v_denominator

        numerator_relative_h = ( (hitpoint.y - origin.y) - relative_v * y_v)
        relative_h = numerator_relative_h / normalizing_h_denominator

        relative_projected_h = h / float(self.projectors.horizontal_pixels)
        relative_projected_v = v / float(self.projectors.vertical_pixels)

        pixel = img.getpixel((h,v))
        diffuse_color = "RED: {}, GREEN: {}, BLUE: {}".format(pixel[0], pixel[1], pixel[2])

        print("")
        print(diffuse_color)
        diffuse_color_blender = (pixel[0]/float(255), pixel[1]/float(255), pixel[2]/float(255), 1)
        self.projectors.highlight_hitpoint(hitpoint.xyz(), diffuse_color_blender)

        print("PROJECTED V. SENSED horizontal position of pixel: {} (and {}) v. {}".format(round(relative_projected_h,6), round(1.0 - relative_projected_h,6), round(relative_h, 6) ))
        print("PROJECTED V. SENSED vertical position of pixel: {} (and {}) v. {}".format(round(relative_projected_v,6), round(1.0 - relative_projected_v,6), round(relative_v, 6) ))
        print("LOCALIZATION: pixel ({},{}) at ({}) with ({},{})".format(h, v, hitpoint.xyz(), relative_h, relative_v))


class Simulator():
  def __init__(self, scanner):
    self.scanner = scanner
    self.environment = scanner.environment
    self.metadata = self.get_metadata()
    self.samples = 0
    self.number_of_models = len(self.metadata)

  def on(self):
    while True:
      model_index = self.samples % self.number_of_models
      number_of_samples_for_model = max(int(np.random.normal(loc=100, scale=50)), 20)
      for i in range(number_of_samples_for_model):
        model = self.metadata[model_index]
        self.environment.update(model["filepath"])
        self.scanner.scan(counter=self.samples)
        self.samples += 1
      break

  def get_metadata(self, model_directory="/home/ubuntu/COLLADA"):
    models = [f for f in listdir(model_directory) if path.isfile(path.join(model_directory, f)) and ".dae" in f]
    total_models = len(models) - 1
    metadata = {}
    for i, model in enumerate(models):
      metadata[i] = {"filename": model, "filepath": path.join(model_directory, model), "samples": []}
    return metadata


if __name__ == "__main__":
  # v.01 : constant camera and laser focal points, focal lengths, pixel sizes, numbers of pixels
  camera = Photonics(projectors_or_sensors="sensors", focal_point=Point(0.1, 0.1, 2.0), focal_length=0.024, pixel_size=0.00000429, vertical_pixels=3456, horizontal_pixels=5184) # 100 x 150 / 3456 x 5184 with focal = 0.024
  lasers = Photonics(projectors_or_sensors="projectors", focal_point=Point(-0.1, -0.1, 2.0), focal_length=0.01127, pixel_size=0.000006, vertical_pixels=768, horizontal_pixels=1366, image="entropy.png") # 64 x 114 / 768 x 1366 -> distance / width = 0.7272404614
  environment = Environment()
  scanner = Scanner(sensors=camera, projectors=lasers, environment=environment)
  simulator = Simulator(scanner=scanner)
  simulator.on()
