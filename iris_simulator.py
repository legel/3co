"""
To-Do:
-- Is rendering png image in blender really the best way to retrieve color info?
-- What is measure_raycasts_from_pixels doing, and is it needed? 
"""

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
import csv2ply
import path_planning
import iris_agent
import open3d as o3d
import evaluation
import socket

# for import errors, install new libraries like so:
# /Applications/Blender.app/Contents/Resources/2.81/python/bin/python3.7m pip install Pillow

print("---------------------------------")
print("Initializing Blender")
print("---------------------------------\n")

home_directory = cwd[:-9]

# idiosyncratic handling of arguments for Python Blender
argv = sys.argv
argv = argv[argv.index("--") + 1:]

if len(argv) != 2:
  print("GPU number (ex:0) and sim_mode (online or offline) must be specified")
  quit()

gpu_number = int(argv[0])
sim_mode = argv[1] # "online" or "offline"



if gpu_number == None:
  gpu_number = 0

output_directory = "simulated_scanner_outputs"

# cleanup
for o in bpy.context.scene.objects:
    if o.type == 'MESH':
        o.select_set(True)
    else:
        o.select_set(False)

# Call the operator only once
bpy.ops.object.delete()

# set rendering engine to be *Blender Cycles*, which is a physically-based raytracer (see e.g. https://www.cycles-renderer.org)
bpy.context.scene.render.engine = 'CYCLES'

try:
  # activate GPU
  bpy.context.preferences.addons['cycles'].preferences.get_devices()
  print(bpy.context.preferences.addons['cycles'].preferences.get_devices())
  bpy.context.scene.cycles.device = 'GPU'
  bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
  bpy.context.preferences.addons['cycles'].preferences.devices[gpu_number].use = True
except TypeError:
  print("No GPU. Fuck it, do it live!")


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

class Pixel(): # "... a discrete physically-addressable region of a photosensitive device, used for photonic projection (e.g. laser) or sensing (e.g. camera)." - https://3co.ai/static/assets/DifferentiablePhotonicGeneratorLocalizer.pdf
  def __init__(self, h, v):
    self.h = h # h := horizontal coordinate  
    self.v = v # v := vertical coordinate  

  def calculate_unit_vectors_through_focal_point(self, focal_point): # for a review of unit vectors computed from one 3D coordinate to another, see e.g. https://mathinsight.org/vectors_cartesian_coordinates_2d_3d , https://www.intmath.com/vectors/7-vectors-in-3d-space.php 
    self.distance_to_focal_point = focal_point.distance(self.center)
    self.unit_x = (focal_point.x - self.center.x) / self.distance_to_focal_point
    self.unit_y = (focal_point.y - self.center.y) / self.distance_to_focal_point
    self.unit_z = (focal_point.z - self.center.z) / self.distance_to_focal_point


class Optics():
  def __init__(self, photonics, environment=None, focal_point=None, target_point=None, focal_length=None, pixel_size=None, vertical_pixels=None, horizontal_pixels=None, image=None, resolution=None):
    # photonics  := string, either "sensors" or "lasers", to describe if the photonic system should *measure* or *project* photons
    # focal_point  := focal point as Point(x,y,z) at which the optical system is positioned
    # target_point := target point as Point(x,y,z) that the optics are oriented toward; location in 3D space of what is "scanned"
    # focal_length  := focal length in meters
    # pixel_size  := size of one side of pixel in real space in meters, assuming pixel is a square
    # vertical_pixels  := number of vertical pixels
    # horionztal_pixels := number of horizontal pixels  
    # image := for lasers, .png image to project
    self.resolution = resolution
    self.time_start = time.time()
    self.photonics = photonics
    self.environment = environment
    self.focal_length = focal_length
    self.vertical_pixels = int(vertical_pixels)
    self.horizontal_pixels = int(horizontal_pixels)
    self.pixel_size = pixel_size
    self.vertical_size = self.vertical_pixels * self.pixel_size
    self.horizontal_size = self.horizontal_pixels * self.pixel_size
    self.horizontal_fov = math.degrees(2.0 * math.atan(0.5 * self.horizontal_size / self.focal_length))
    self.vertical_fov = math.degrees(2.0 * math.atan(0.5 * self.vertical_size / self.focal_length))
    print("Optical system with {} by {} degrees field of view".format(self.horizontal_fov, self.vertical_fov))
    self.limiting_fov = min(self.horizontal_fov, self.vertical_fov)
    self.focal_point = focal_point
    self.target_point = target_point
    self.pixels = [[Pixel(h,v) for v in range(self.vertical_pixels)] for h in range(self.horizontal_pixels)]
    self.image_center = Point()
    self.highlighted_hitpoints = []
    self.sampled_hitpoint_pixels = []
    self.shutterspeed = ""
    self.lasers_watts_per_meters_squared = ""
    if type(image) != type(None):
      self.image = image
    else:
      self.image = ""
    if photonics == "lasers":
      self.initialize_lasers()
    elif photonics == "sensors":
      self.initialize_sensors()
    self.time_end = time.time()
    if type(target_point) == type(Point()):
      self.reorient()
    print("Launched {} in {} seconds".format(self.photonics, round(self.time_end - self.time_start, 4)))


  def get_point_cloud(self):

    self.export_point_cloud("pc_export")
    pc = "pc_export.csv"
    return pc


  def export_point_cloud(self, f_name):
    vertical_pixels = len(self.get_pixel_indices("vertical"))
    horizontal_pixels = len(self.get_pixel_indices("horizontal"))
    # get color data from render and project those onto point cloud
    """
    render_filename = "{}/{}_render.png".format(output_directory, f_name)
    render_image = Image.open(render_filename).convert('RGB')
    for h in self.get_pixel_indices("horizontal"):
      for v in self.get_pixel_indices("vertical"):
        r, g, b = render_image.getpixel((h,vertical_pixels-v-1))
        self.pixels[h][v].rendered_red = r
        self.pixels[h][v].rendered_green = g
        self.pixels[h][v].rendered_blue = b
    """

    with open("{}/{}.csv".format(output_directory,f_name), "w") as point_cloud_file:

      for h in self.get_pixel_indices("horizontal"):
        for v in self.get_pixel_indices("vertical"):
          if self.pixels[h][v].hitpoint_object == "model":
            #r = self.pixels[h][v].rendered_red
            #g = self.pixels[h][v].rendered_green
            #b = self.pixels[h][v].rendered_blue
            point = self.pixels[h][v].hitpoint
            x = round(point.x,6)
            y = round(point.y,6)
            z = round(point.z,6)
            #point_cloud_file.write("{},{},{},{},{},{},{},{}\n".format(h,v,x,y,z,r,g,b))
            point_cloud_file.write("{},{},{},{},{},{},{},{}\n".format(h,v,x,y,z,0,0,0))
    
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
    self.sensors.location = (self.focal_point.x, self.focal_point.y, self.focal_point.z)
    bpy.data.scenes["Scene"].render.resolution_x = self.horizontal_pixels
    bpy.data.scenes["Scene"].render.resolution_y = self.vertical_pixels
    bpy.data.scenes["Scene"].render.tile_x = 512
    bpy.data.scenes["Scene"].render.tile_y = 512

    # hack
    self.shutterspeed = 0.03 
    print("Shutterspeed of {} seconds".format(self.shutterspeed))
    bpy.data.scenes["Scene"].cycles.film_exposure = self.shutterspeed # seconds of exposure / shutterspeed!
    
  def initialize_lasers(self):
    self.laser_data = bpy.data.lights.new(name="laser_data", type='SPOT')
    self.laser_data.shadow_soft_size = 0
    self.laser_data.spot_size = 3.14159
    self.laser_data.cycles.max_bounces = 0
    self.laser_data.use_nodes = True  

    lighting_strength = 802.2 #* (46.66 / 7.77) 

    # 802.2 derived from 700 lumens per LED light in real scanner
    # For details on how Blender models physical parameters of light, see:
    # https://devtalk.blender.org/t/why-watt-as-light-value/5658/14   
    # (46.66 / 7.77) derived as ratio between Tungsten incandescent lamp and LED lamp
    # https://www.rapidtables.com/calc/light/lumen-to-watt-calculator.html

    self.laser_data.node_tree.nodes["Emission"].inputs[1].default_value = lighting_strength # W/m^2
    self.lasers_watts_per_meters_squared = lighting_strength

    # warp mapping of light
    mapping = self.laser_data.node_tree.nodes.new(type='ShaderNodeMapping')
    mapping.location = 300,0
    mapping.inputs['Rotation'].default_value = (0,0,3.14159)

    # for < Blender 2.81
    # mapping.rotation[2] = 3.14159 # broken in Blender 2.82
    #mapping.scale[1] = self.horizontal_pixels / float(self.vertical_pixels) # e.g. 1.779 for Laser Beam Pro
    #mapping.scale[2] = self.horizontal_size / self.focal_length # e.g. 0.7272404614 for Laser Beam Pro
      
    scale_x = 1.0
    scale_y = self.horizontal_pixels / float(self.vertical_pixels)
    scale_z = self.horizontal_size / self.focal_length
    mapping.inputs['Scale'].default_value = (scale_x, scale_y, scale_z)

    # separate xyz
    separate_xyz = self.laser_data.node_tree.nodes.new(type='ShaderNodeSeparateXYZ')
    separate_xyz.location = 900,0
    self.laser_data.node_tree.links.new(mapping.outputs['Vector'], separate_xyz.inputs[0])

    # divide x
    divide_x = self.laser_data.node_tree.nodes.new(type='ShaderNodeMath')
    divide_x.operation = 'DIVIDE'
    divide_x.location = 1200,300
    self.laser_data.node_tree.links.new(separate_xyz.outputs['X'], divide_x.inputs[0])
    self.laser_data.node_tree.links.new(separate_xyz.outputs['Z'], divide_x.inputs[1])

    # add x
    add_x = self.laser_data.node_tree.nodes.new(type='ShaderNodeMath')
    add_x.operation = 'ADD'
    add_x.location = 1500,300
    self.laser_data.node_tree.links.new(divide_x.outputs[0], add_x.inputs[0])

    # divide y
    divide_y = self.laser_data.node_tree.nodes.new(type='ShaderNodeMath')
    divide_y.operation = 'DIVIDE'
    divide_y.location = 1200,0
    self.laser_data.node_tree.links.new(separate_xyz.outputs['Y'], divide_y.inputs[0])
    self.laser_data.node_tree.links.new(separate_xyz.outputs['Z'], divide_y.inputs[1])

    # add y
    add_y = self.laser_data.node_tree.nodes.new(type='ShaderNodeMath')
    add_y.operation = 'ADD'
    add_x.location = 1500,0
    self.laser_data.node_tree.links.new(divide_y.outputs[0], add_y.inputs[0])
    
    # combine xyz
    combine_xyz = self.laser_data.node_tree.nodes.new(type='ShaderNodeCombineXYZ')
    combine_xyz.location = 1800,0
    self.laser_data.node_tree.links.new(add_x.outputs['Value'], combine_xyz.inputs[0])
    self.laser_data.node_tree.links.new(add_y.outputs['Value'], combine_xyz.inputs[1])
    self.laser_data.node_tree.links.new(separate_xyz.outputs['Z'], combine_xyz.inputs[2])

    # texture coordinate
    texture_coordinate = self.laser_data.node_tree.nodes.new(type='ShaderNodeTexCoord')
    texture_coordinate.location = 0,0
    self.laser_data.node_tree.links.new(texture_coordinate.outputs['Normal'], mapping.inputs[0])

    # image texture
    image_texture = self.laser_data.node_tree.nodes.new(type='ShaderNodeTexImage')

    image_texture.image = bpy.data.images.load(self.image)
    self.image_to_project = image_texture.image

    image_texture.extension = 'CLIP'
    image_texture.location = 2100,0
    self.laser_data.node_tree.links.new(image_texture.outputs['Color'], self.laser_data.node_tree.nodes["Emission"].inputs[0])

    # connect combine with image
    self.laser_data.node_tree.links.new(combine_xyz.outputs['Vector'], image_texture.inputs[0])

    image_texture = self.laser_data.node_tree.nodes.new(type='ShaderNodeMixRGB')

    self.lasers = bpy.data.objects.new(name="Laser", object_data=self.laser_data)
    bpy.context.scene.collection.objects.link(self.lasers)


  def project(self, filepath):
    self.filepath_of_image_to_project = filepath
    self.image_to_project = bpy.data.images.load(filepath)


  def reorient(self, orientation_index=0, compute_global_coordinates_of_all_pixels=True):
    time_start = time.time()
    if type(self.focal_point) == type(Point()) and type(self.target_point) == type(Point()):
      self.compute_image_center()
      self.compute_euler_angles()
      self.compute_xyz_of_boundary_pixels()
      if compute_global_coordinates_of_all_pixels: # this is an expensive operation, so only do this if you're actually raycasting from each pixel
        self.orient_xyz_and_unit_vectors_for_all_pixels()
      adjusted_euler_z = self.rotation_euler_z * -1.0 # to correct for a notational difference between rendering engine and notes
      if self.photonics == "lasers":
        #print("Lasers to be set with location ({:.3f},{:.3f},{:.3f}) and rotation ({:.3f},{:.3f},{:.3f})".format(self.focal_point.x, self.focal_point.y, self.focal_point.z, self.rotation_euler_x, self.rotation_euler_y, adjusted_euler_z))
        #self.lasers.inputs['Location'].default_value = (self.focal_point.x, self.focal_point.y, self.focal_point.z)
        #self.lasers.inputs['Rotation'].default_value = (self.rotation_euler_x, self.rotation_euler_y, adjusted_euler_z)
        # pre-Blender 2.81
        self.lasers.location = (self.focal_point.x, self.focal_point.y, self.focal_point.z)
        self.lasers.rotation_euler = (self.rotation_euler_x, self.rotation_euler_y, adjusted_euler_z)
        self.lasers.delta_scale = (-1.0, 1.0, 1.0) # flips image that is projected horizontally, to match inverted raycasting
      elif self.photonics == "sensors":
        #print("Sensors to be set with location ({:.3f},{:.3f},{:.3f}) and rotation ({:.3f},{:.3f},{:.3f})".format(self.focal_point.x, self.focal_point.y, self.focal_point.z, self.rotation_euler_x, self.rotation_euler_y, adjusted_euler_z))
        self.sensors.location = (self.focal_point.x, self.focal_point.y, self.focal_point.z)
        self.sensors.rotation_euler = (self.rotation_euler_x, self.rotation_euler_y, adjusted_euler_z)

    time_end = time.time()
    print("--> Orientations of {} computed in {} seconds".format(self.photonics, round(time_end - time_start, 4)))

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

    self.rotation_euler_y = 0.0 # definition

    #print("Euler rotations of (x={},y={},z={})".format(math.degrees(self.rotation_euler_x), math.degrees(self.rotation_euler_y), math.degrees(self.rotation_euler_z)))

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
  

  def measure_raycasts_from_pixels(self, environment):
    self.model_object = environment.model_object
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

        if obj == environment.model_object:
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
    print("--> Raycasts of {} computed in {} seconds".format(self.photonics, round(time_end - time_start, 4)))
    #print("{} hits on the model, {} hits on the background, {} hits elsewhere".format(model_hits, background_hits, other_hits))


class Model():
  def __init__(self, filepath=None):
    if filepath:
      self.filepath = filepath
      self.import_object_to_scan(filepath=filepath)
      self.dimensions = bpy.context.object.dimensions

  def import_object_to_scan(self, filepath):
    obs = []
    bpy.ops.wm.collada_import(filepath=filepath)
    for object_in_scene in bpy.context.scene.objects:
      if object_in_scene.type == 'MESH':
       obs.append(object_in_scene)
       bpy.context.view_layer.objects.active = object_in_scene
       self.object_name = object_in_scene.name
       object_in_scene.select_set(state=True)
       # print("INDIVIDUAL OBJECT FROM MODEL: {}".format(object_in_scene.name))

    c = {} # override, see: https://blender.stackexchange.com/a/133024/72320
    c["object"] = c["active_object"] = bpy.context.object
    c["selected_objects"] = c["selected_editable_objects"] = obs
    bpy.ops.object.join(c)

    self.model_object = bpy.context.object
    bpy.context.object.name = "Model"


class Environment():

  def add_model(self,model_filepath):
    self.model = Model(model_filepath)
    self.model_object = self.model.model_object

  def update(self, iris, action):
    iris.scanner.move(x=action[0], y=action[1], z=action[2], yaw=action[3], pitch=action[4])


class Scanner():
  def __init__(self, sensors, lasers=None, environment=None):
    self.environment = environment
    self.sensors = sensors
    self.lasers = lasers
    self.resolution = sensors.resolution

  def scan(self, f_out_name=None, render_png=True):
    self.sensors.measure_raycasts_from_pixels(environment=self.environment)
    if f_out_name == None:
      return self.sensors.get_point_cloud()
    else:
      if render_png == True:
        self.render("{}/{}_render.png".format(output_directory, f_out_name))
      self.sensors.export_point_cloud(f_out_name)



  def move(self, x=None, y=None, z=None, yaw=None, pitch=None):
    if x != None:
      self.sensors.focal_point.x = x
    if y != None:
      self.sensors.focal_point.y = y
    if z != None:
      self.sensors.focal_point.z = z
    
    if pitch == None:
      pitch = self.sensors.rotation_euler_x
    else:
      if pitch == 0 or pitch == 0.0:
        pitch = 0.000001 # epsilon non-zero value to prevent degeneracy
      pitch = math.radians(pitch)

    if yaw == None:
      yaw = math.radians(180 + math.degrees(self.sensors.rotation_euler_z)) # if using Blender's yaw, we need to convert e.g. -90 degrees to 90 degrees, -180 degrees to 0 degrees, for our calculations
    else:
      if yaw == 0 or yaw == 0.0:
        yaw = 0.000001 # epsilon non-zero value to prevent degeneracy
      yaw = math.radians(yaw) 

    # geometry based on coordinate system with origin at (0,0,0); see https://docs.google.com/document/d/1FsgnzzdmZE0qz_1uw7lePc5e3lh1HGlXNSBlKcXP4hU/edit?usp=sharing
    x_target = self.sensors.focal_point.x - math.sin(yaw) * math.sin(pitch)
    y_target = self.sensors.focal_point.y - math.cos(yaw) * math.sin(pitch)
    z_target = self.sensors.focal_point.z - math.cos(pitch) 

    self.sensors.target_point = Point(x_target, y_target, z_target)
    self.sensors.reorient()

    # now, do the same for lasers (i.e. lights) if they exist
    if type(self.lasers) != type(None):
      self.lasers.focal_point.x = self.sensors.focal_point.x
      self.lasers.focal_point.y = self.sensors.focal_point.y
      self.lasers.focal_point.z = self.sensors.focal_point.z 
      self.lasers.rotation_euler_x = self.sensors.rotation_euler_x
      self.lasers.rotation_euler_y = self.sensors.rotation_euler_y
      self.lasers.rotation_euler_z = self.sensors.rotation_euler_z
      self.lasers.target_point = Point(x_target, y_target, z_target)
      self.lasers.reorient()


  def render(self, filename):

    time_start = time.time()
    bpy.data.scenes["Scene"].render.filepath = filename
    bpy.ops.render.render( write_still=True )
    time_end = time.time()

    print("--> Rendered scan image in {} seconds".format(round(time_end - time_start, 4)))



class Iris():

  def __init__(self, scanner):
    self.scanner = scanner
    self.resolution = scanner.resolution
    self.workspace = "iris_workspace"
    if os.path.isdir(self.workspace):
      command = "rm -r {}".format(self.workspace)
      os.system(command)

    command = "mkdir {}".format(self.workspace)
    os.system(command)    


  def scan (self, t):
    self.scanner.render("{}/scan_{}.png".format(self.workspace,t))
    return self.scanner.scan(render_png=False)


def startOfflineSimulation(iris, environment, path, dataset):


  print("---------------------------------")
  print("Begin offline scan simulation")
  print("---------------------------------\n")

  i = 0
  scaling = 1.0
  for p in path:
    x = p[0] / scaling
    y = p[1] / scaling
    z = p[2] / scaling
    yaw = p[3]
    pitch = p[4]
    f_out_name="{}/{}_{}".format(dataset,dataset,i)
    print("Moving to new state: [({}, {}, {}), ({}, {})]".format(round(x,2),round(y,2),round(z,2),round(yaw,2),round(pitch,2)))
    action = [x, y, z, yaw, pitch]
    environment.update(iris, action) 
    #scanner.move(x=x, y=y, z=z, pitch=pitch, yaw=yaw)
    print("Scanning")
    iris.scanner.scan(f_out_name=f_out_name, render_png=True)
    csv2ply.csv2ply("simulated_scanner_outputs/{}.csv".format(f_out_name), "simulated_scanner_outputs/{}.ply".format(f_out_name))
    i = i + 1


"""
the following might be useful later, but needs to be updated

# Experiment runs main control loop given a constructed iris agent and environment
def experiment(iris, environment):

  print("---------------------------------")
  print("Begin scan simulation")
  print("---------------------------------\n")

  results = []
  results.append(open("experiment_results/avg_scan_error.txt","w"))
  results.append(open("experiment_results/avg_recon_error.txt","w"))
  results.append(open("experiment_results/avg_total_error.txt","w"))
  results.append(open("experiment_results/haus_scan_error.txt","w"))
  results.append(open("experiment_results/haus_recon_error.txt","w"))
  results.append(open("experiment_results/haus_total_error.txt","w"))


  # get the point cloud from ground truth mesh
  ground_truth_mesh_fn = "simulated_scanner_outputs/chalice_0.1/chalice_centered.ply"
  ground_truth_pc = o3d.io.read_point_cloud(ground_truth_mesh_fn)

  t = 0

  while iris.active():
    obs = iris.scan()
    iris.learn(obs)
    action = iris.act()
    environment.update(iris, action)

    
    if t > 0:
      # get the merged point cloud from all scans so far
      scan_pc_fn = "iris_workspace/scan_0.1_merged.ply"
      scan_pc = o3d.io.read_point_cloud(scan_pc_fn)

      # get the point cloud from the current reconstruction
      current_mesh_fn = "iris_workspace/scan_0.1_reconstructed_vcg.ply"
      current_pc = o3d.io.read_point_cloud(current_mesh_fn)

      scan_d_haus = evaluation.hausdorffDistance(scan_pc, ground_truth_pc)
      scan_d_max_avg = evaluation.maxAvgPointCloudDistance(scan_pc, ground_truth_pc)
      recon_d_haus = evaluation.hausdorffDistance(scan_pc, current_pc)
      recon_d_max_avg = evaluation.maxAvgPointCloudDistance(scan_pc, current_pc)
      final_d_haus = evaluation.hausdorffDistance(current_pc, ground_truth_pc)
      final_d_max_avg = evaluation.maxAvgPointCloudDistance(current_pc, ground_truth_pc)

      results[0].write("{}\n".format(scan_d_max_avg))
      results[1].write("{}\n".format(recon_d_max_avg))
      results[2].write("{}\n".format(final_d_max_avg))
      results[3].write("{}\n".format(scan_d_haus))
      results[4].write("{}\n".format(recon_d_haus))
      results[5].write("{}\n".format(final_d_haus))

    t = t + 1


  for f in results:
    f.close()



  print("---------------------------------")
  print("End scan simulation")
  print("---------------------------------\n")

"""


def parseAction(action):
  a = action.split(",")
  return [float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4])]

def startOnlineSimulation(iris, environment, start_pos):

  print("---------------------------------")
  print("Begin online scan simulation")
  print("---------------------------------\n")

  host = socket.gethostname()
  port = 8080

  
  sock = socket.socket()
  sock.bind((host, port))


  sock.listen(1)
  #print("Starting agent controller...")
  #os.system(controller_start_cmd)
  print("Waiting for agent controller to connect...")
  client_sock, address = sock.accept()
  print("Agent controller connected.\n")
  print("Sending start position to agent.")

  msg = "start_pos : {},{},{},{},{}".format(start_pos[0],start_pos[1],start_pos[2],start_pos[3],start_pos[4])
  client_sock.send(msg.encode("utf-8"))

  # wait for ack from agent
  data = client_sock.recv(1024).decode("utf-8")

  if data.strip() == "ack":
    print("Agent ack received.")
  else:
    print("Agent ack failed! Aborting simulation")
    client_sock.close()
    sock.close()
    quit()

  print("------ Begin simulation ------")
  

  t = 0

  status = "active"
  while status == "active":

    obs = iris.scan(t) # filename of obs

    # send obs to agent
    data = obs
    msg = "obs : {}".format(data)
    print("sending obs to agent")
    client_sock.send(msg.encode("utf-8"))

    # receive next action from agent
    data = client_sock.recv(1024).decode("utf-8")
    if not data:
      print("null data")
      client_sock.close()
      sock.close()
      break
    action_data = data.split(":")
    if action_data[0].strip() != "act":
      print("Error: received invalid action")
      client_sock.close()
      sock.close()
      quit()
    print("received action from agent")
    print(action_data)
    action = parseAction(action_data[1].strip())


    # execute action on environment
    environment.update(iris, action)

    t = t + 1
    if t > 12:
      status = "inactive"

  msg = "end_session"
  client_sock.send(msg.encode("utf-8"))
  client_sock.close()
  sock.close()
  print("------ Simulation concluded ------")



if __name__ == "__main__":  

  ####################################################################
  # Some models that can be used for ooi:
  ####################################################################
  #
  # chalice: reconstructables/data/9d506eb0e13514e167816b64852d28f.dae -> chalice_centered.dae
  # chair: reconstructables/data/1a2a5a06ce083786581bb5a25b17bed6.dae
  # ant: reconstructables/data/f16f37317eac2e37b21d2748b9ce78f4.dae -> ant_centered.dae
  # beer: reconstructables/data/f452c1053f88cd2fc21f7907838a35d1.dae -> beer_centered.dae
  # bplant: simulated_scanner_outputs/banana_plant/banana_plant/banana_plant.dae
  # brownchair: simulated_scanner_outputs/brownchair/brownchair/Zara_armchair_1.dae
  #
  ###################################################################

  print("---------------------------------")
  print("Initializing scan environment")
  print("---------------------------------\n")

  environment = Environment()
  ooi = "reconstructables/data/chalice_centered.dae"
  environment.add_model(ooi)
  sensor_resolution = 0.05
  sensors = Optics( photonics="sensors", 
                    environment=environment, 
                    focal_point=Point(x=-1.0, y=-1.0, z=-1.0), # dummy value; not used
                    focal_length=0.012, 
                    vertical_pixels=2048 * sensor_resolution, 
                    horizontal_pixels=2048 * sensor_resolution, 
                    pixel_size=0.00000587 / sensor_resolution,
                    target_point=Point(0.0,0.0,0.0),
                    resolution = sensor_resolution)

  """ Some version of below needs to be used for proper lighting!
  lasers =  Optics( photonics="lasers", # here, a "laser" is technically a pixel of projected light 
                    image="white.png", # white.png is just an image of all white pixels
                    environment=environment, 
                    focal_point=Point(x=-1.0, y=-1.0, z=-1.0), # dummy value; not used
                    focal_length=0.012, # technically for real scanner, projector and camera have different optics
                    vertical_pixels=2048 * sensor_resolution,  
                    horizontal_pixels=2048 * sensor_resolution, 
                    pixel_size=0.00000587 / sensor_resolution,  
                    target_point=Point(0.0,0.0,0.0))
  scanner = Scanner(sensors=sensors, environment=environment, lasers=lasers)
  """
  
  scanner = Scanner(sensors=sensors, environment=environment)
  iris = Iris(scanner)


  # in online mode, actions are supplied by agent client, but start position
  # must be specified below
  # data goes to iris_workspace/
  if sim_mode == "online":
    x,y,z,theta,phi = 4.8, 0.0, 1.7, 90.0, 90.0
    start_pos = [x,y,z,theta,phi]
    environment.update(iris,start_pos)
    startOnlineSimulation(iris, environment, start_pos)
  # in offline mode, a path and name of dataset must be specified below
  # data goes to simulated_scanner_outputs/{dataset}/
  elif sim_mode == "offline":
    path = path_planning.get_chalice_path()
    dataset = "chalice_{}".format(sensor_resolution)
    output_path = "simulated_scanner_outputs/{}".format(dataset)
    if not os.path.isdir(output_path):
      command = "mkdir {}".format(output_path)
      os.system(command)

    startOfflineSimulation(iris, environment, path, dataset)
  else:
    print("Error: invalid sim mode")
    quit()

