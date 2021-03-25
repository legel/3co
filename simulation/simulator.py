import bpy
import math
import random
import time
import numpy as np
from math import cos, sin
import bmesh
from PIL import Image
from mathutils import Vector, Color
import json
from os import listdir, path, getcwd
from pprint import pprint
import sys
cwd = getcwd()
sys.path.append(cwd)
import os
import cv2 as cv
import imageio

# example way to run via command line locally:
# blender --python simulator.py -- cpu
#
# ...and remotely in the cloud:
# blender -noaudio -b --python simulator.py  -- gpu

# -b means run in the background with no GUI (useful for cloud)
# -noaudio disables audio (also useful for cloud)
# -- cpu means use CPUs
# -- gpu means use GPU

print("---------------------------------")
print("Initializing Blender")
print("---------------------------------\n")

home_directory = cwd[:-9]

# idiosyncratic handling of arguments for Python Blender
argv = sys.argv
argv = argv[argv.index("--") + 1:]

output_directory = "outputs"

# delete initial objects loaded in blender, e.g. light source
for o in bpy.context.scene.objects:
  o.select_set(True)
bpy.ops.object.delete()

# set rendering engine to be *Blender Cycles*, which is a physically-based raytracer (see e.g. https://www.cycles-renderer.org)
bpy.context.scene.render.engine = 'CYCLES'

gpu_or_cpu = argv[0]
if gpu_or_cpu == "gpu":
  try:
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.preferences.addons['cycles'].preferences.devices[0].use = True
  except TypeError:
    print("No GPU. Fuck it, do it live!")
elif gpu_or_cpu == "cpu":
    devices = bpy.context.preferences.addons['cycles'].preferences.get_devices()
    print(devices)
    bpy.context.scene.cycles.device = 'CPU'
    for device in devices:
      device[0].use = True

class Point(): # wrapper for (x,y,z) coordinates with helpers; can refactor to https://github.com/tensorflow/graphics
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

  def xyz_dictionary(self):
    return {"x": self.x, "y": self.y, "z": self.z}


class Triangle():
  def __init__(self, a, b, c):
    # a,b,c are each 3D points of class Point(x,y,z)
    self.a = a
    self.b = b
    self.c = c
    self.compute_sides_of_triangle()
    self.solve_angles_with_law_of_cosines()

  def compute_sides_of_triangle(self):
    self.distance_c = self.a.distance(self.b)
    self.distance_a = self.b.distance(self.c)
    self.distance_b = self.c.distance(self.a)

  def solve_angles_with_law_of_cosines(self): # see e.g. http://hyperphysics.phy-astr.gsu.edu/hbase/lcos.html
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


class Pixel(): # "... a discrete physically-addressable region of a photosensitive device, used for photonic projection (e.g. projector) or sensing (e.g. camera)." - https://3co.ai/static/assets/DifferentiablePhotonicGeneratorLocalizer.pdf
  def __init__(self, h, v):
    self.h = h # h := horizontal coordinate  
    self.v = v # v := vertical coordinate  

  def calculate_unit_vectors_through_focal_point(self, focal_point): # for a review of unit vectors computed from one 3D coordinate to another, see e.g. https://mathinsight.org/vectors_cartesian_coordinates_2d_3d , https://www.intmath.com/vectors/7-vectors-in-3d-space.php 
    self.distance_to_focal_point = focal_point.distance(self.center)
    self.unit_x = (focal_point.x - self.center.x) / self.distance_to_focal_point
    self.unit_y = (focal_point.y - self.center.y) / self.distance_to_focal_point
    self.unit_z = (focal_point.z - self.center.z) / self.distance_to_focal_point


class Photonics():
  def __init__(self, application, focal_point=None, target_point=None, focal_length=None, pixel_size=None, vertical_pixels=None, horizontal_pixels=None, image=None, resolution=None, simulation_method="render_layer_access"):
    # application  := string, either "sensors" or "projectors", to describe if the photonic system should *measure* or *project* photons
    # focal_point  := focal point as Point(x,y,z) at which the optical system is positioned
    # target_point := target point as Point(x,y,z) that the optics are oriented toward; location in 3D space of what is "scanned"
    # focal_length  := focal length in meters
    # pixel_size  := size of one side of pixel in real space in meters, assuming pixel is a square
    # vertical_pixels  := number of vertical pixels
    # horionztal_pixels := number of horizontal pixels  
    # image := for projectors, .png image to project
    # simulation_method := "render_layer_access" if using Blender's render layer for output data; "raycasting" if brute force raycast based
    self.simulation_method = simulation_method
    self.resolution = resolution
    self.time_start = time.time()
    self.application = application
    self.focal_length = focal_length
    self.vertical_pixels = int(vertical_pixels)
    self.horizontal_pixels = int(horizontal_pixels)
    self.pixel_size = pixel_size
    self.vertical_size = self.vertical_pixels * self.pixel_size
    self.horizontal_size = self.horizontal_pixels * self.pixel_size
    self.horizontal_fov = math.degrees(2.0 * math.atan(0.5 * self.horizontal_size / self.focal_length))
    self.vertical_fov = math.degrees(2.0 * math.atan(0.5 * self.vertical_size / self.focal_length))
    # self.horizontal_fov = math.degrees(2.0 * math.atan(0.5 * self.horizontal_size / self.focal_length))
    # self.vertical_fov = math.degrees(2.0 * math.atan(0.5 * self.vertical_size / self.focal_length))
    print("Photonic {} system with {} by {} degrees field of view".format(application, self.horizontal_fov, self.vertical_fov))
    self.focal_point = focal_point
    self.target_point = target_point
    if self.simulation_method == "raycasting":
      self.pixels = [[Pixel(h,v) for v in range(self.vertical_pixels)] for h in range(self.horizontal_pixels)]
    self.image_center = Point()
    self.exposure_time = ""
    self.projectors_watts_per_meters_squared = ""
    if type(image) != type(None):
      self.image = image
    else:
      self.image = ""
    if application == "projectors":
      self.initialize_projectors()
    elif application == "sensors":
      self.initialize_sensors()
    self.time_end = time.time()
    if self.simulation_method == "raycasting":
      self.reorient()
    print("Launched {} in {} seconds".format(self.application, round(self.time_end - self.time_start, 4)))

  def get_point_cloud(self):
    self.export_point_cloud("pc_export")
    pc = "pc_export.csv"
    return pc

  def export_point_cloud(self, f_name):
    if self.simulation_method == "raycasting":
      vertical_pixels = len(self.get_pixel_indices("vertical"))
      horizontal_pixels = len(self.get_pixel_indices("horizontal"))
      # project color from render onto point cloud
      render_filename = "{}/{}_render.png".format(output_directory, f_name)
      render_image = Image.open(render_filename).convert('RGB')
      for h in self.get_pixel_indices("horizontal"):
        for v in self.get_pixel_indices("vertical"):
          r, g, b = render_image.getpixel((h,vertical_pixels-v-1))
          self.pixels[h][v].rendered_red = r
          self.pixels[h][v].rendered_green = g
          self.pixels[h][v].rendered_blue = b
      with open("{}/{}.csv".format(output_directory,f_name), "w") as point_cloud_file:
        for h in self.get_pixel_indices("horizontal"):
          for v in self.get_pixel_indices("vertical"):
            if self.pixels[h][v].hitpoint_object == "model":
              r = self.pixels[h][v].rendered_red
              g = self.pixels[h][v].rendered_green
              b = self.pixels[h][v].rendered_blue
              point = self.pixels[h][v].hitpoint
              x = round(point.x,6)
              y = round(point.y,6)
              z = round(point.z,6)
              point_cloud_file.write("{},{},{},{},{},{},{},{}\n".format(h,v,x,y,z,r,g,b))
      # convert .csv data to .ply output
      self.csv2ply("{}/{}.csv".format(output_directory,f_name), "{}/{}.ply".format(output_directory,f_name))
    else:
      print("Point export failed: change simulation_method from \"render_layer_access\" to \"raycasting\"")
    
  def csv2ply(self, f_in, f_out):
    # write header
    n_vertices = sum(1 for line in open(f_in)) - 1
    ply_file = open(f_out, "w")
    ply_file.write('ply\n')
    ply_file.write('format ascii 1.0\n')
    ply_file.write('element vertex ' + str(n_vertices) + '\n')
    ply_file.write('property float x\n')
    ply_file.write('property float y\n')
    ply_file.write('property float z\n')
    ply_file.write('property uchar red\n')
    ply_file.write('property uchar green\n')
    ply_file.write('property uchar blue\n')
    ply_file.write('end_header\n')

    # read vertices and write each one
    with open(f_in, "r") as csv_file:
      # skip header
      #line = csv_file.readline()
      # read vertices
      for line in csv_file:
        l = line.split(",")
        x = l[2]
        y = l[3]
        z = l[4]
        r = l[5]
        g = l[6]
        b = l[7]
        vertex = '{} {} {} {} {} {}'.format(x,y,z,r,g,b)
        ply_file.write(vertex)

    ply_file.close()

  def get_pixel_indices(self, v_or_h):
    # v_or_h i.e. vertical or horizontal, is a string "vertical" or "horizontal", which returns the vertical or horizontal pixel indices
    if v_or_h == "horizontal":
      return range(self.horizontal_pixels)
    elif v_or_h == "vertical":
      return  range(self.vertical_pixels)      

  def initialize_sensors(self):
    self.sensor_data = bpy.data.cameras.new("sensor_data")
    self.sensors = bpy.data.objects.new("Sensor", self.sensor_data)
    bpy.context.scene.collection.objects.link(self.sensors)
    bpy.context.scene.camera = self.sensors
    bpy.data.cameras["sensor_data"].lens = self.focal_length * 1000 # millimeters
    bpy.data.cameras["sensor_data"].sensor_width = self.horizontal_size * 1000 # millimeters
    bpy.data.cameras["sensor_data"].display_size = 0.2      # remove this if you want old iris back
    self.sensors.location = (self.focal_point.x, self.focal_point.y, self.focal_point.z)
    bpy.data.scenes["Scene"].render.resolution_x = self.horizontal_pixels
    bpy.data.scenes["Scene"].render.resolution_y = self.vertical_pixels
    if gpu_or_cpu == "gpu":
      bpy.data.scenes["Scene"].render.tile_x = 512
      bpy.data.scenes["Scene"].render.tile_y = 512
    elif gpu_or_cpu == "cpu":
      bpy.data.scenes["Scene"].render.tile_x = 128
      bpy.data.scenes["Scene"].render.tile_y = 128
    self.set_exposure_time()

  def set_exposure_time(self, exposure_time=0.025):    
    bpy.data.scenes["Scene"].cycles.film_exposure = exposure_time
    
  def initialize_projectors(self):
    self.projector_data = bpy.data.lights.new(name="projector_data", type='SPOT')
    self.projector_data.shadow_soft_size = 0
    self.projector_data.spot_size = 3.14159
    self.projector_data.cycles.max_bounces = 0
    self.projector_data.use_nodes = True  
    

    # 802.2 derived from 700 lumens per LED light in real scanner
    # For details on how Blender models physical parameters of light, see:
    # https://devtalk.blender.org/t/why-watt-as-light-value/5658/14  
    lighting_strength = 802.2

    # (46.66 / 7.77) is ratio between Tungsten incandescent lamp and LED lamp
    # https://www.rapidtables.com/calc/light/lumen-to-watt-calculator.html

    self.projector_data.node_tree.nodes["Emission"].inputs[1].default_value = lighting_strength # W/m^2
    self.projectors_watts_per_meters_squared = lighting_strength

    # warp mapping of light
    mapping = self.projector_data.node_tree.nodes.new(type='ShaderNodeMapping')
    mapping.location = 300,0
    mapping.inputs['Rotation'].default_value = (0,0,3.14159)

    scale_x = 1.0
    scale_y = self.horizontal_pixels / float(self.vertical_pixels)
    scale_z = self.horizontal_size / self.focal_length
    mapping.inputs['Scale'].default_value = (scale_x, scale_y, scale_z)

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

    white_pixels_image = "{}/metadata/white.png".format(cwd)
    image_texture.image = bpy.data.images.load(white_pixels_image)
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

  def reorient(self, orientation_index=0):
    time_start = time.time()
    if self.simulation_method == "raycasting": # this is an expensive operation, so only do this if you're actually raycasting from each pixel
      self.compute_image_center()
      self.compute_euler_angles()
      self.compute_xyz_of_boundary_pixels()
      self.orient_xyz_and_unit_vectors_for_all_pixels()
      adjusted_euler_z = self.rotation_euler_z * -1.0 # to correct for a notational difference between rendering engine and notes
      if self.application == "projectors":
        self.projectors.location = (self.focal_point.x, self.focal_point.y, self.focal_point.z)
        self.projectors.rotation_euler = (self.rotation_euler_x, self.rotation_euler_y, adjusted_euler_z)
        self.projectors.delta_scale = (-1.0, 1.0, 1.0) # flips image that is projected horizontally, to match inverted raycasting
      elif self.application == "sensors":
        self.sensors.location = (self.focal_point.x, self.focal_point.y, self.focal_point.z)
        self.sensors.rotation_euler = (self.rotation_euler_x, self.rotation_euler_y, adjusted_euler_z)
    else:
      if self.application == "projectors":
        self.projectors.location = (self.focal_point.x, self.focal_point.y, self.focal_point.z)
        self.projectors.rotation_euler = (self.rotation_euler_x, self.rotation_euler_y, self.rotation_euler_z)
      elif self.application == "sensors":
        self.sensors.location = (self.focal_point.x, self.focal_point.y, self.focal_point.z)
        self.sensors.rotation_euler = (self.rotation_euler_x, self.rotation_euler_y, self.rotation_euler_z)
    time_end = time.time()
    print("--> Orientations of {} computed in {} seconds".format(self.application, round(time_end - time_start, 4)))

  def compute_image_center(self):
    focal_ratio = self.focal_length / (self.focal_length + self.focal_point.distance(self.target_point))
    if focal_ratio == 1.0:
      raise("Distance between focal point and target point cannot be zero; make the optical system point somewhere else than its position.")    
    self.image_center.x = (self.focal_point.x - self.target_point.x * focal_ratio) / (1 - focal_ratio)
    self.image_center.y = (self.focal_point.y - self.target_point.y * focal_ratio) / (1 - focal_ratio)
    self.image_center.z = (self.focal_point.z - self.target_point.z * focal_ratio) / (1 - focal_ratio)

  def compute_euler_angles(self):
    # compute euler angles from default angle of image top pointing to +y, image right pointing to +x, image view pointing to -z
    o = self.image_center
    t = self.target_point
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
    self.rotation_euler_y = 0.0

  def compute_xyz_of_boundary_pixels(self):
    if self.simulation_method == "raycasting":
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
    if self.simulation_method == "raycasting":
      relative_horizontal_coordinates = [2 * h / float(self.horizontal_pixels) - 1 for h in range(self.horizontal_pixels + 1)] # indices: 0 = left edge, 1 = left edge + one pixel right, 2 = left edge + two pixels right, ...
      relative_vertical_coordinates = [2 * v / float(self.vertical_pixels) - 1 for v in range(self.vertical_pixels + 1)] # indices: 0 = bottom edge, 1 = bottom edge + one pixel up, 2 = bottom edge + two pixels up, ...
      x_of_horizontal_vectors = [(1 - relative_h) * self.image_center.x + relative_h * self.image_horizontal_edge.x for relative_h in relative_horizontal_coordinates]
      y_of_horizontal_vectors = [(1 - relative_h) * self.image_center.y + relative_h * self.image_horizontal_edge.y for relative_h in relative_horizontal_coordinates]
      z_of_horizontal_vectors = [(1 - relative_h) * self.image_center.z + relative_h * self.image_horizontal_edge.z for relative_h in relative_horizontal_coordinates]
      x_of_vertical_vectors = [(1 - relative_v) * self.image_center.x + relative_v * self.image_vertical_edge.x for relative_v in relative_vertical_coordinates]
      y_of_vertical_vectors = [(1 - relative_v) * self.image_center.y + relative_v * self.image_vertical_edge.y for relative_v in relative_vertical_coordinates]
      z_of_vertical_vectors = [(1 - relative_v) * self.image_center.z + relative_v * self.image_vertical_edge.z for relative_v in relative_vertical_coordinates]   
      for h in self.get_pixel_indices("horizontal"):
        left_x_of_horizontal_vector = x_of_horizontal_vectors[h]
        left_y_of_horizontal_vector = y_of_horizontal_vectors[h]
        left_z_of_horizontal_vector = z_of_horizontal_vectors[h]
        right_x_of_horizontal_vector = x_of_horizontal_vectors[h+1]
        right_y_of_horizontal_vector = y_of_horizontal_vectors[h+1]
        right_z_of_horizontal_vector = z_of_horizontal_vectors[h+1]      
        for v in self.get_pixel_indices("vertical"): 
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
  
  def measure_raycasts_from_pixels(self, model):
    if self.simulation_method == "raycasting":
      self.model_object = model.model_object
      time_start = time.time()

      min_h = 0
      min_v = 0
      max_h = self.horizontal_pixels - 1
      max_v = self.vertical_pixels - 1

      model_hits = 0
      background_hits = 0
      other_hits = 0

      for h in self.get_pixel_indices("horizontal"):
        for v in self.get_pixel_indices("vertical"):
          origin = Vector((self.pixels[h][v].center.x, self.pixels[h][v].center.y, self.pixels[h][v].center.z))
          direction = Vector((self.pixels[h][v].unit_x, self.pixels[h][v].unit_y, self.pixels[h][v].unit_z))
          hit, location, normal, face_index, obj, matrix_world = bpy.context.scene.ray_cast(view_layer=bpy.context.view_layer, origin=origin, direction=direction)
          if not hit:
            #print("No hitpoint for raycast from pixel ({},{})".format(h, v))
            self.pixels[h][v].hitpoint = Point("None", "None", "None")
          else:
            self.pixels[h][v].hitpoint = Point(location[0], location[1], location[2])

          if obj == model_object:
            self.pixels[h][v].hitpoint_object = "model"
            self.pixels[h][v].hitpoint_face_index = face_index
            self.pixels[h][v].hitpoint_normal = Point(normal[0], normal[1], normal[2])
            model_hits += 1

          else:
            self.pixels[h][v].hitpoint_object = "None"
            self.pixels[h][v].hitpoint_face_index = "None"
            self.pixels[h][v].hitpoint_normal =  Point("None", "None", "None")
            other_hits += 1         
          
      time_end = time.time()
      print("--> Raycasts of {} computed in {} seconds".format(self.application, round(time_end - time_start, 4)))


class Model():
  def __init__(self, filename=None):
    if filename:
      self.filepath = "{}/models/{}".format(cwd,filename)
      print(self.filepath)
      self.import_object_to_scan()
      self.dimensions = bpy.context.object.dimensions
      print('object dimensions: {}'.format(self.dimensions))

  def import_object_to_scan(self):
    print("---------------------------------")
    print("Loading model at filepath {}".format(self.filepath))
    print("---------------------------------\n")
    obs = []
    self.description = self.filepath.split(".")[0].split("/")[-1]
    ext = self.filepath.split(".")[-1]
    if ext == "dae":
      bpy.ops.wm.collada_import(filepath=self.filepath)
    elif ext == "obj":
      bpy.ops.import_scene.obj(filepath=self.filepath, filter_glob="*.obj;*.mtl")
    elif ext == ".glb" or ".gltf":
      bpy.ops.import_scene.gltf(filepath=self.filepath)
    else:
      print("Unrecognized object extension: {}\nFeel free to implement in function \"import_object_to_scan()\"".format(ext))
      quit()
    for object_in_scene in bpy.context.scene.objects:
      if object_in_scene.type == 'MESH':
       obs.append(object_in_scene)
       bpy.context.view_layer.objects.active = object_in_scene
       self.object_name = object_in_scene.name
       object_in_scene.select_set(state=True)
    c = {} # override, see: https://blender.stackexchange.com/a/133024/72320
    c["object"] = c["active_object"] = bpy.context.object
    c["selected_objects"] = c["selected_editable_objects"] = obs
    bpy.ops.object.join(c)
    self.model_object = bpy.context.object
    bpy.context.object.name = "Model"
    bpy.data.objects["Model"].scale = (0.3, 0.3, 0.3)   ## size that makes sense for the scanner


class Iris():
  def __init__(self, model, resolution, simulation_method = "render_layer_access"):
    # to-do: add 3x3x3 meter enclosure with white paneling, equivalent to Iris 3D scanning space

    self.sensors =     Photonics(   application="sensors",
                                    focal_point=Point(x=1.0, y=0.0, z=1.0),
                                    focal_length=0.012, 
                                    vertical_pixels=2048 * resolution, 
                                    horizontal_pixels=2048 * resolution, 
                                    pixel_size=0.00000587 / resolution,
                                    target_point=Point(0.0,0.0,0.25),
                                    resolution = resolution,
                                    simulation_method = simulation_method)

    self.projectors =  Photonics(   application="projectors",
                                    focal_point=Point(x=1.0, y=0.0, z=1.0),
                                    focal_length=0.012, 
                                    vertical_pixels=2048 * resolution,  
                                    horizontal_pixels=2048 * resolution, 
                                    pixel_size=0.00000587 / resolution,  
                                    target_point=Point(0.0,0.0,0.25),
                                    simulation_method = simulation_method)

    self.model = Model(model)
    self.model.scale = (0.1, 0.1, 0.1)
    self.simulation_method = simulation_method
    

  def initialize_blender_render_layers(self, scan_name, resolution):

    directory_for_scan = '{}/outputs/{}'.format(cwd, scan_name)

    view_layer = bpy.data.scenes["Scene"].view_layers["View Layer"]
    view_layer.use = True
    view_layer.use_pass_combined = True
    view_layer.use_pass_z = True
    view_layer.use_pass_normal = True
    view_layer.use_pass_diffuse_color = True
    view_layer.use_pass_diffuse_direct = True
    view_layer.use_pass_glossy_direct = True
    view_layer.use_pass_glossy_color = True

    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True

    tree = bpy.context.scene.node_tree
    nodes = tree.nodes
    links = tree.links

    for node in tree.nodes:
      tree.nodes.remove(node)
    render_layer = tree.nodes.new('CompositorNodeRLayers') 

    viewer = tree.nodes.new('CompositorNodeViewer')   
    viewer.use_alpha = True

    ###############
    #### DEPTH ####
    ###############

    ## OPENEXR: UNNORMALIZED DEPTH VALUES #
    depth_file_output_exr = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output_exr.name = 'depth_output_gt_not_normalized' 
    depth_file_output_exr.format.file_format = "OPEN_EXR"
    normalize_depth = tree.nodes.new(type="CompositorNodeNormalize")
    links.new(render_layer.outputs['Depth'], depth_file_output_exr.inputs[0])
    depth_file_output_exr.base_path = '{}/{}_{}'.format(directory_for_scan, scan_name, depth_file_output_exr.name)
    ## OPENEXR: UNNORMALIZED DEPTH VALUES #

    ###############
    #### DEPTH ####
    ###############

    ###############
    ### NORMALS ###
    ###############
    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.name = 'normal_output'
    links.new(render_layer.outputs['Normal'], normal_file_output.inputs[0])
    normal_file_output.format.file_format = "PNG"
    normal_file_output.format.color_mode = "RGBA"
    normal_file_output.base_path = '{}/{}_{}'.format(directory_for_scan, scan_name, normal_file_output.name)
    ###############
    ### NORMALS ###
    ###############

    ###############
    ### DIFFUSE ###
    ###############
    diffuse_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    diffuse_file_output.name = 'diffuse_output'
    links.new(render_layer.outputs['DiffCol'], diffuse_file_output.inputs[0])
    diffuse_file_output.format.file_format = "PNG"
    diffuse_file_output.format.color_mode = "RGBA"
    diffuse_file_output.base_path = '{}/{}_{}'.format(directory_for_scan, scan_name, diffuse_file_output.name)
    ###############
    ### DIFFUSE ###
    ###############

    ### BAKE MATERIAL PROPERTIES ###
    resolution_bake = 2048 * resolution
    bake_types = [('roughness_bake', 'ROUGHNESS')] #, ('diffuse_bake', 'DIFFUSE')] # ['baseColor', 'metallic','subsurface','specular','roughness','specularTint','anisotropic','sheen','sheenTint','clearcoat','clearcoatGloss']
    for bake_type in bake_types:
      (name, baketype) = bake_type
      bake_file = bpy.data.images.new(name, resolution_bake, resolution_bake)
      for m in bpy.context.object.material_slots:
        nodes_bake = m.material.node_tree.nodes
        image_texture_node = nodes_bake.new(type="ShaderNodeTexImage")
        image_texture_node.image = bake_file
        image_texture_node.label = name
        image_texture_node.name = name
        m.material.node_tree.nodes.active = image_texture_node
      print('start baking of {}...'.format(bake_file))
      bpy.ops.object.bake(type=baketype, width=resolution_bake, height=resolution_bake)
      print('bake ready :)')
    ### BAKE MATERIAL PROPERTIES ###


  def scan(self, exposure_time=0.3, scan_id="", resolution=1.0):
    self.sensors.set_exposure_time(exposure_time=exposure_time)
    if self.simulation_method == "raycasting":
      self.sensors.measure_raycasts_from_pixels(model=self.model)
    if scan_id != "":
      scan_name = '{}_{}'.format(self.model.description, scan_id)
    else:
      current_time = int(time.time())
      scan_name = "{}_{}".format(self.model.description, current_time)

    if not os.path.exists('{}/outputs/{}'.format(cwd, scan_name)):
      os.mkdir('{}/outputs/{}'.format(cwd, scan_name))

    data = {'x_pos':     self.sensors.focal_point.x,
            'y_pos': self.sensors.focal_point.y,
            'z_pos': self.sensors.focal_point.z,
            'pitch' : self.sensors.rotation_euler_x,
            'roll' : self.sensors.rotation_euler_y,
            'yaw' : self.sensors.rotation_euler_z}
    with open('{}/outputs/{}/data.json'.format(cwd, scan_name), 'w') as outfile:
      json.dump(data, outfile)

    self.initialize_blender_render_layers(scan_name, resolution)
    self.render(scan_name)
    if self.simulation_method == "raycasting":
      self.sensors.export_point_cloud(scan_name)

  def render(self, scan_name):
    tree = bpy.context.scene.node_tree    # compositor
    nodes = tree.nodes

    directory_for_scan = '{}/outputs/{}'.format(cwd, scan_name)

    # RENDER IMAGE OF BAKE #
    bake_types = ['roughness_bake'] #, 'diffuse_bake']
    for bake_type in bake_types:
      for m in bpy.context.object.material_slots:
        links = m.material.node_tree.links
        nodes = m.material.node_tree.nodes
        shader_node = nodes.new("ShaderNodeEmission")
        shader_node.label = 'Shader Node'
        shader_node.inputs[1].default_value = 60.0
        links.new(nodes[bake_type].outputs[0], shader_node.inputs[0])
        links.new(shader_node.outputs[0], nodes['Material Output'].inputs[0])
        render_filepath = "{}/{}".format(directory_for_scan, scan_name)
      print("Rendering image to /outputs{}.png".format(render_filepath))
      time_start = time.time()
      bpy.data.scenes["Scene"].render.filepath = "{}_{}".format(render_filepath, bake_type)
      bpy.ops.render.render(animation=False, write_still=True)
      time_end = time.time()
      print("--> Rendered scan image in {} seconds".format(round(time_end - time_start, 4)))
    # RENDER IMAGE OF BAKE #

    # RECONNECT BSDF SHADER AS MATERIAL OUTPUT AND RENDER #
    for m in bpy.context.object.material_slots:   # reconnect Principled BSDF to 'Material Ooutput
      links = m.material.node_tree.links          # shader tree
      nodes = m.material.node_tree.nodes
      links.new(nodes['Principled BSDF'].outputs[0], nodes['Material Output'].inputs[0])
    render_filepath = "{}/{}".format(directory_for_scan, scan_name)
    print("Rendering image to /outputs/{}.png".format(scan_name))
    time_start = time.time()
    bpy.data.scenes["Scene"].render.filepath = "{}_render".format(render_filepath)
    bpy.ops.render.render(animation=False, write_still=True)
    time_end = time.time()
    print("--> Rendered scan image in {} seconds".format(round(time_end - time_start, 4)))
    # RECONNECT BSDF SHADER AS MATERIAL OUTPUT AND RENDER #

    # RENAMING FILES BECAUSE BLENDER WON'T DO IT #
    for output in ['diffuse_output', 'normal_output', 'depth_output_gt_not_normalized']: # the renaming and cleaning up can only be done after the render has taken place
      output_path = '{}_{}'.format(scan_name, output)
      current_file = '{}/{}'.format(directory_for_scan, output_path)
      output_file_current = os.listdir(current_file)[0]
      _, file_extension = os.path.splitext(output_file_current)
      output_file_new = '{}/{}{}'.format(directory_for_scan, output_path, file_extension)
      output_file_old = '{}/{}/{}'.format(directory_for_scan, output_path,output_file_current)
      os.rename(output_file_old, output_file_new)
      os.rmdir(current_file)
    # RENAMING FILES BECAUSE BLENDER WON'T DO IT #

    def add_noise(distance):
      '''
      Noise as a function of distance:

      Error values are an estimation of the error typically found at scan time.

      in meters:
      Distance   | Noise          Distance    | Noise
      --------------------        ----------------------- a = 0.2, b = 0.000025
      0.2        | 0.000025       1.0 * a     | 4^0 * b
      0.3        | 0.000100       1.5 * a     | 4^1 * b
      0.4        | 0.000400       2.0 * a     | 4^2 * b
      0.5        | 0.001600       2.5 * a     | 4^3 * b

      Used function fitting algorithm:
      b = 0.0625*e^2.77a 

      '''
      if(distance < 0.2):
        return 100000.0
      if(distance > 0.5):
        return 100000.0
      
      standard_noise_b = 0.000025                 
      standard_distance_a = 10 * (distance / 2)

      noise_in_meters_b =  0.0625 * (math.e ** (2.77 * standard_distance_a))
      noise_in_meters = noise_in_meters_b * standard_noise_b

      mean = noise_in_meters
      standard_deviation = noise_in_meters / 10

      noise_gaussian = np.round(np.random.normal(mean, standard_deviation), 8)
      
      return noise_gaussian
    
    
    blender_exr_depth_output = cv.imread('{}/{}_depth_output_gt_not_normalized.exr'.format(directory_for_scan, scan_name), cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    image = np.asarray(blender_exr_depth_output, dtype=np.float32)

    # READ GROUND TRUTH EXR VALUES AND NORMALIZE #
    min_unnormalized_image = np.min(image)
    max_unnormalized_image = sorted(list(set(image.flatten().tolist())))[-2] # max is 1000000.0
    gt_normalized = (image - min_unnormalized_image) / (max_unnormalized_image - min_unnormalized_image)
    gt_normalized[gt_normalized > 1] = 1.0
    output_gt_normalized = '{}/{}_depth_output_gt_normalized.exr'.format(directory_for_scan, scan_name)
    imageio.imwrite(output_gt_normalized, gt_normalized)
    # READ GROUND TRUTH EXR VALUES AND NORMALIZE #

    # ADD NOISE TO DEPTH VALUES #
    noise_output_slice = image[:,:,2]
    shape_noise = noise_output_slice.shape
    noise_output_slice_flat = noise_output_slice.flatten()
    apply_noise_map = [add_noise(x) for x in noise_output_slice_flat]
    noise_applied = np.reshape(apply_noise_map, shape_noise)
    noise_applied_image = np.asarray(np.dstack((noise_applied, noise_applied, noise_applied)), dtype=np.float32)
    output_noise_not_normalized = '{}/{}_depth_output_noise_not_normalized.exr'.format(directory_for_scan, scan_name)
    imageio.imwrite(output_noise_not_normalized, noise_applied_image)
    # ADD NOISE TO DEPTH VALUES #

    # ADD NOISE TO DEPTH VALUES AND NORMALIZE #
    min_unnormalized_noise = np.min(noise_applied_image)
    max_unnormalized_noise = sorted(list(set(noise_applied_image.flatten().tolist())))[-2] # max is 1000000.0
    noise_normalized = (noise_applied_image - min_unnormalized_noise) / (max_unnormalized_noise - min_unnormalized_noise)
    noise_normalized[noise_normalized > 1] = 1.0
    output_noise_normalized = '{}/{}_depth_output_noise_normalized.exr'.format(directory_for_scan,  scan_name)
    imageio.imwrite(output_noise_normalized, noise_normalized)
    # ADD NOISE TO DEPTH VALUES NORMALIZE #



  def non_zero_degeneracy_radians(self, pitch,yaw,roll):
    # epsilon non-zero value to prevent degeneracy
    if pitch == 0 or pitch == 0.0:
      pitch = 0.000001 

    if yaw == 0 or yaw == 0.0:
      yaw = 0.000001

    if roll == 0 or roll == 0.0:
      roll = 0.000001

    # convert to radians
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)
    roll = math.radians(roll)

    return pitch, yaw, roll

  def view(self, x, y, z, rotation_x=None, rotation_y=None, rotation_z=None, pitch=None, yaw=None, roll=None):
    # API supports either "rotation_x, rotation_y, rotation_z" (closer to Blender) or "pitch, yaw, roll" (closer to Iris)
    if type(pitch) == type(None) and type(yaw) == type(None) and type(roll) == type(None):
      pitch = rotation_x
      yaw = rotation_z
      roll = rotation_y

    print("Moving Iris to see a 6D view (x,y,z,pitch,yaw,roll):")
    print("({:.3f}, {:.3f}, {:.3f}, {:.3f}°, {:.3f}°, {:.3f}°)".format(x,y,z,pitch,yaw,roll))

    pitch, yaw, roll = self.non_zero_degeneracy_radians(pitch, yaw, roll)
    self.sensors.focal_point.x = x
    self.sensors.focal_point.y = y
    self.sensors.focal_point.z = z
    self.sensors.rotation_euler_x = pitch
    self.sensors.rotation_euler_y = roll
    self.sensors.rotation_euler_z = yaw

    if self.simulation_method == "raycasting":
      x_target = self.sensors.focal_point.x - math.sin(yaw) * math.sin(pitch)
      y_target = self.sensors.focal_point.y - math.cos(yaw) * math.sin(pitch)
      z_target = self.sensors.focal_point.z - math.cos(pitch) 
      self.sensors.target_point = Point(x_target, y_target, z_target)
      self.projectors.target_point = Point(x_target, y_target, z_target)

    self.sensors.reorient()

    # currently, projectors are in exactly the same position as the sensors
    self.projectors.focal_point.x = self.sensors.focal_point.x
    self.projectors.focal_point.y = self.sensors.focal_point.y
    self.projectors.focal_point.z = self.sensors.focal_point.z 
    self.projectors.rotation_euler_x = self.sensors.rotation_euler_x
    self.projectors.rotation_euler_y = self.sensors.rotation_euler_y
    self.projectors.rotation_euler_z = self.sensors.rotation_euler_z

    self.projectors.reorient()


if __name__ == "__main__": 
  # iris = Iris(model="pillow/pillow.glb", resolution=0.3)
  # iris.view(x=0, y=1.15, z=1.2, rotation_x=45, rotation_y=0.0, rotation_z=180)
  # iris.scan(exposure_time=0.025, scan_id=1) 

  iris = Iris(model="pillow/pillow.glb", resolution=0.1)
  iris.view(x=0, y=0.35, z=0.35, rotation_x=45, rotation_y=0.0, rotation_z=180)
  iris.scan(exposure_time=0.025, scan_id=1) 

  # iris = Iris(model="/tire/tire.glb", resolution=0.1)
  # iris.view(x=0, y=1.15, z=1.2, rotation_x=45, rotation_y=0.0, rotation_z=180)
  # iris.scan(exposure_time=0.1, scan_id=0)


  # iris = Iris(model="/chair/chair.glb", resolution=0.1)
  # iris.view(x=0.25, y=-1.5, z=0.75, rotation_x=75, rotation_y=0.5, rotation_z=10) 
  # iris.scan(exposure_time=0.4, scan_id=0)
