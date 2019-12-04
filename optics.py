import bpy
import math
import random
import time
import numpy as np
from math import cos, sin
import bmesh
from PIL import Image
from mathutils import Vector
import json
from os import listdir, path
from pprint import pprint
import sys

# idiosyncratic handling of arguments for Python Blender
argv = sys.argv
argv = argv[argv.index("--") + 1:]
gpu_number = int(argv[0])
output_directory = argv[1]

if gpu_number == None:
  gpu_number = 0

if output_directory == None:
  output_directory = ""

simulation_mode = "ALL" # "TEST" (raycasts for only 4 pixels) or "ALL" (all raycasts, default)
compute_localizations = False 

# cleanup, please
for o in bpy.context.scene.objects:
    if o.type == 'MESH':
        o.select_set(True)
    else:
        o.select_set(False)

# Call the operator only once
bpy.ops.object.delete()

# set rendering engine to be *Blender Cycles*, which is a physically-based raytracer (see e.g. https://www.cycles-renderer.org)
bpy.context.scene.render.engine = 'CYCLES'

# Save and re-open the file to clean up the data blocks
#bpy.ops.wm.save_as_mainfile(filepath="active_file.blend")
#bpy.ops.wm.open_mainfile(filepath="active_file.blend")

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

  def get_metadata(self, photonics):
    metadata = [] # structured list, for smaller memory footprint
    if photonics == "lasers":
      # horizontal_pixel_number = self.h
      # vertical_pixel_number = self.v
      # pixel_x_in_global_space = self.center.x  
      # pixel_y_in_global_space = self.center.y
      # pixel_z_in_global_space = self.center.z
      # hitpoint_in_environment.x = self.hitpoint.x
      # hitpoint_in_environment.y = self.hitpoint.y
      # hitpoint_in_environment.z = self.hitpoint.z
      # self.hitpoint_normal.x
      # self.hitpoint_normal.y
      # self.hitpoint_normal.z
      # self.hitpoint_in_sensor_plane.x
      # self.hitpoint_in_sensor_plane.y
      # self.hitpoint_in_sensor_plane.z
      # relative_horizontal_position_in_sensor_plane = self.relative_h
      # metadata[15] = relative_vertical_position_in_sensor_plane
      # metadata[16] = hitpoint_object_id_in_environment
      # metadata[17] = hitpoint_face_index_on_object
      ## NOT IN USE : self.hitpoint_in_sensor_plane_object
      ## NOT IN USE : self.projected_pixel_color

      self.hitpoint_object_code = 0
      if self.hitpoint_object == "background":
        self.hitpoint_object_code = 0
      elif self.hitpoint_object == "None":
        self.hitpoint_object_code = 1
      else:
        self.hitpoint_object_code = 2 # if not hitting the background or none, it's hitting the object 
      if self.relative_h != "OCCLUDED" and compute_localizations:
        self.relative_h = round(self.relative_h, 7)
      if self.relative_v != "OCCLUDED" and compute_localizations:
        self.relative_v = round(self.relative_v, 7)
      if compute_localizations:
        metadata.append(self.relative_h)
        metadata.append(self.relative_v)
      metadata.append(self.hitpoint_object_code)
      metadata.append(self.hitpoint_face_index)

    return metadata


class Optics():
  def __init__(self, photonics, environment=None, focal_point=None, target_point=None, focal_length=None, pixel_size=None, vertical_pixels=None, horizontal_pixels=None, image=None, position_anchor=None):
    # photonics  := string, either "sensors" or "lasers", to describe if the photonic system should *measure* or *project* photons
    # focal_point  := focal point as Point(x,y,z) at which the optical system is positioned
    # target_point := target point as Point(x,y,z) that the optics are oriented toward; location in 3D space of what is "scanned"
    # focal_length  := focal length in meters
    # pixel_size  := size of one side of pixel in real space in meters, assuming pixel is a square
    # vertical_pixels  := number of vertical pixels
    # horionztal_pixels := number of horizontal pixels  
    # image := for lasers, .png image to project
    # position_anchor := position to anchor the position sampling, i.e. base sampling off of another object's existing position
    self.time_start = time.time()
    self.photonics = photonics
    self.environment = environment
    self.focal_length = self.sample_focal_length(focal_length)
    vertical_pixels, horizontal_pixels = self.sample_pixel_resolution(vertical_pixels, horizontal_pixels)
    self.vertical_pixels = vertical_pixels
    self.horizontal_pixels = horizontal_pixels
    self.pixel_size = self.sample_pixel_size(pixel_size)
    self.vertical_size = self.vertical_pixels * self.pixel_size
    self.horizontal_size = self.horizontal_pixels * self.pixel_size
    self.horizontal_fov = math.degrees(2.0 * math.atan(0.5 * self.horizontal_size / self.focal_length))
    self.vertical_fov = math.degrees(2.0 * math.atan(0.5 * self.vertical_size / self.focal_length))
    print("Optical system with {} by {} degrees field of view".format(self.horizontal_fov, self.vertical_fov))
    self.limiting_fov = min(self.horizontal_fov, self.vertical_fov)
    self.focal_point = self.sample_focal_point(focal_point=focal_point, position_anchor=position_anchor)
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
    print("Launched {} in {} seconds".format(self.photonics, round(self.time_end - self.time_start, 4)))

  def extract_optical_metadata(self, launch_time):
    if compute_localizations:
      pixel_metadata = self.extract_pixel_metadata() # structured list for memory optimization, see "get_metadata()" in Pixel class
    else:
      pixel_metadata = None

    if self.photonics == "lasers":
      self.export_point_cloud(launch_time)

    if self.photonics == "sensors":
      if self.sensor_as_scanner:
        self.export_point_cloud(launch_time)

    optical_metadata = {"photonics": self.photonics,
                        "focal_point": self.focal_point.xyz_dictionary(),
                        "target_point": self.target_point.xyz_dictionary(),
                        "principal_point": self.image_center.xyz_dictionary(),
                        "focal_length": self.focal_length, 
                        "pixel_size": self.pixel_size,
                        "vertical_pixels": self.vertical_pixels,
                        "horizontal_pixels": self.horizontal_pixels,
                        "image": self.image,
                        "vertical_size": self.vertical_size,
                        "horizontal_size": self.horizontal_size,
                        "shutterspeed": self.shutterspeed,
                        "watts_per_meters_squared": self.lasers_watts_per_meters_squared,
                        "pixel_metadata": pixel_metadata,
                        }
    return optical_metadata

  def export_point_cloud(self, launch_time):
    point_cloud_mesh = bpy.data.meshes.new("point_cloud_mesh")
    point_cloud_object = bpy.data.objects.new("point_cloud_object", point_cloud_mesh)
    bpy.context.collection.objects.link(point_cloud_object)
    bpy.context.view_layer.objects.active = point_cloud_object
    point_cloud_object.select_set( state = True, view_layer = None)
    point_cloud_mesh = bpy.context.object.data
    bm = bmesh.new()

    # make x,y,z points
    horizontal_pixels = len(self.get_pixel_indices("horizontal"))
    vertical_pixels = len(self.get_pixel_indices("vertical"))
    vertices = {}
    points = {}
    
    with open("{}.csv".format(launch_time), "w") as point_cloud_file:
      point_cloud_file.write("h,v,x,y,z,face_index\n")

      for h in self.get_pixel_indices("horizontal"):
        points[h] = {}
        vertices[h] = {}
        for v in self.get_pixel_indices("vertical"):
          if self.pixels[h][v].hitpoint_object == "model":
            face_index = self.pixels[h][v].hitpoint_face_index
            #material_index = self.model_object.data.polygons[face_index].material_index
            #diffuse_color = self.model_object.material_slots[material_index].material.diffuse_color
            #r = int(diffuse_color[0]*255)
            #g = int(diffuse_color[1]*255)
            #b = int(diffuse_color[2]*255)
            point = self.pixels[h][v].hitpoint
            x = round(point.x,6)
            y = round(point.y,6)
            z = round(point.z,6)
            point_cloud_file.write("{},{},{},{},{},{}\n".format(h,v,x,y,z,face_index))
            #print(("HIT: ({},{},{})".format(x, y, z)))
            vertex = bm.verts.new((point.x, point.y, point.z))
            vertices[h][v] = vertex
            points[h][v] = point
          else:
            point_cloud_file.write("{},{},N,N,N,N\n".format(h,v))
            vertices[h][v] = None
            points[h][v] = None

    for h in self.get_pixel_indices("horizontal"):
      for v in self.get_pixel_indices("vertical"):
        if h+1 <= horizontal_pixels - 1 and v+1 <= vertical_pixels - 1:
          a = vertices[h][v]
          b = vertices[h+1][v]
          c = vertices[h][v+1]
          a_p = points[h][v]
          b_p = points[h+1][v]
          c_p = points[h][v+1]
          if a != None and b != None and c != None:
            if a_p.distance(b_p) < 0.01 and a_p.distance(c_p) < 0.01 and b_p.distance(c_p) < 0.01:
              bm.edges.new( [a, b] )
              bm.edges.new( [a, c] )
              bm.edges.new( [b, c] )
              bm.faces.new( [a, b, c])

    for h in self.get_pixel_indices("horizontal"):
      for v in self.get_pixel_indices("vertical"):
        if h+1 <= horizontal_pixels - 1 and v+1 <= vertical_pixels - 1:
          #a = vertices[h][v]
          b = vertices[h+1][v]
          c = vertices[h][v+1]
          d = vertices[h+1][v+1]
          b_p = points[h+1][v]
          c_p = points[h][v+1]
          d_p = points[h+1][v+1]          
          if b != None and c != None and d != None:
            if b_p.distance(c_p) < 0.01 and c_p.distance(d_p) < 0.01 and b_p.distance(d_p) < 0.01:
              bm.faces.new( [b, c, d])

    # left = bm.verts.new((-1, -1, 0))
    # middle = bm.verts.new((0, 1, 0))
    # right = bm.verts.new((1, -1, 0))
    # bm.edges.new( [left, middle] )
    # bm.edges.new( [middle, right] )
    # bm.edges.new( [left, right] )
    # bm.faces.new( [left, middle, right]) 

    bm.to_mesh(point_cloud_mesh)  
    bm.free()
    bpy.context.scene.update() 
    bpy.ops.export_mesh.ply(filepath="{}/{}.ply".format(output_directory, launch_time), check_existing=False)



  def extract_pixel_metadata(self):
    # pixel metadata is a dictionary wrapper in the form pixel_metadata[h][v] = metadata at horizontal pixel position h, vertical pixel position v
    #self.pixel_metadata = {}
    self.metadata_wrapper = []
    for h in self.get_pixel_indices("horizontal"):
      h = int(h)
      #self.pixel_metadata[h] = {}
      for v in self.get_pixel_indices("vertical"):
        v = int(v)
        self.metadata_wrapper.extend(self.pixels[h][v].get_metadata(self.photonics))
        #self.pixel_metadata[h][v] = self.pixels[h][v].get_metadata(self.photonics)
    return self.metadata_wrapper

  def get_pixel_indices(self, v_or_h):
    # v_or_h i.e. vertical or horizontal, is a string "vertical" or "horizontal", which returns the vertical or horizontal pixel indices
    if simulation_mode == "TEST":
      if v_or_h == "horizontal":
        return [0, self.horizontal_pixels - 1]
      elif v_or_h == "vertical":
        return [0, self.vertical_pixels - 1]
    elif simulation_mode == "ALL":
      if v_or_h == "horizontal":
        return range(self.horizontal_pixels)
      elif v_or_h == "vertical":
        return  range(self.vertical_pixels)      

  def sample_focal_point(self, focal_point, position_anchor):
    if type(self.environment) != type(None):
      if type(position_anchor) == type(None):
        # z_enclosure_visible_distance = 0.5 * self.environment.limiting_edge / math.tan(0.5 * math.radians(self.limiting_fov))
        # mean_z = self.environment.model.max_z #+ z_enclosure_visible_distance * 0.15 # +15% of distance needed to see entire object

        x = np.random.normal(loc=0.0, scale=0.001)
        y = np.random.normal(loc=0.06, scale=0.0033)
        z = np.random.normal(loc=1.50, scale=0.25)
        # hack
        x = 2.0
        y = 0.0
        z = 0.0

      else:  # if we're making a rigid connection to another optical system 
        x = np.random.normal(loc=0.0, scale=0.001) 
        y = np.random.normal(loc=-0.06, scale=0.0033)
        z = position_anchor.focal_point.z + np.random.normal(loc=0.0, scale=0.001)    

      print("{} focal point of ({},{},{}) sampled".format(self.photonics, x, y, z))
      return Point(x, y, z)

    elif type(focal_point) == type(Point()):
      print("Using supplied focal point ({},{},{}) for {}".format(focal_point.x, focal_point.y, focal_point.z, self.photonics))
      return focal_point
    else:
      raise("Either supply an environment or a focal point of type Point(x,y,z), declared like focal_point=Point(0.0, 0.0, 0.0)")

  def sample_pixel_resolution(self, vertical_pixels, horizontal_pixels):
    if type(vertical_pixels) == type(None) and type(horizontal_pixels) == type(None):
      vertical_to_horizontal_ratio_statistical_family = (5184 / 3456) * np.random.normal(loc=1.0, scale=0.1) # working from current DSLR in use, + - 10%
      # vertical_to_horizontal_ratio_statistical_family = 1.0 # square image
      self.horizontal_to_vertical_pixel_ratio = np.random.choice(a=[vertical_to_horizontal_ratio_statistical_family, 1.0], p=[1.00, 0.00])

      vertical_statistical_family_a = 3456.0 #random.uniform(3200.0, 3456.0) 
      vertical_statistical_family_b = np.random.normal(loc=1250.0, scale=1000.0)

      vertical_pixels = max(np.random.choice(a=[vertical_statistical_family_a, vertical_statistical_family_b], p=[1.00, 0.00]), 256.0)
      horizontal_pixels = vertical_pixels * self.horizontal_to_vertical_pixel_ratio

      # hack to speed up testing
      vertical_pixels = 500 #1824
      horizontal_pixels = 500  #2280

    return [int(vertical_pixels), int(horizontal_pixels)]


  def sample_focal_length(self, focal_length):
    if type(focal_length) == type(None):
      if self.photonics == "sensors":
        statistical_family_a = random.uniform(20.0, 30.0) # millimeters
        statistical_family_b = np.random.normal(loc=24.0, scale=3.0)
        focal_length = max(min(np.random.choice(a=[statistical_family_a, statistical_family_b], p=[0.25, 0.75]), 100.0), 10.0)
        # hack
        focal_length = 7.0
        return focal_length / 1000 # meters
      elif self.photonics == "lasers":
        statistical_family_a = random.uniform(7.0, 13.0)
        statistical_family_b = np.random.normal(loc=11.27, scale=1.0)
        focal_length = max(min(np.random.choice(a=[statistical_family_a, statistical_family_b], p=[0.25, 0.75]), 50.0), 5.0)       
        # hack 
        focal_length = 20
        return focal_length / 1000 # meters
      else: 
        raise("The photonics argument must be a string, 'lasers' or 'sensors'...")
    elif type(focal_length) == type(0.0):
      print("Using supplied focal length of {} meters for {}".format(focal_length, self.photonics))
      return focal_length
    else:
      raise("The focal length argument should either be left None, to be randomly sampled, or of type 0.0 (floating point number) in meters, declared like focal_length=0.010, which would be a 10mm focal length")

  def sample_pixel_size(self, pixel_size):
    if type(pixel_size) == type(None):
      if self.photonics == "sensors":
        pixel_size = np.random.normal(loc=4.29, scale=0.25) 
        # hack
        pixel_size = 10
      elif self.photonics == "lasers":
        pixel_size = np.random.normal(loc=6.0, scale=0.25)    
        # hack 
        pixel_size = 40 
      return pixel_size / 1000000 # meters
    elif type(pixel_size) == type(0.0):
      print("Using supplied pixel size of {} meters for {}".format(pixel_size, self.photonics))
      return pixel_size
    else:
      raise("The pixel size argument should either be left None, to be randomly sampled, or of type 0.0 (floating point number) in meters, declared like pixel_size=0.000006, which would be a 6 micrometer pixel stride")

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

    statistical_family_a = random.uniform(0.01, 1.0) 
    statistical_family_b = np.random.normal(loc=0.066, scale=0.033)
    # hack
    self.shutterspeed = 0.05 # max(np.random.choice(a=[statistical_family_a, statistical_family_b], p=[0.1, 0.9]), 0.01)

    print("Shutterspeed of {} seconds".format(self.shutterspeed))
    bpy.data.scenes["Scene"].cycles.film_exposure = self.shutterspeed # seconds of exposure / shutterspeed!
    
  def initialize_lasers(self):
    self.laser_data = bpy.data.lights.new(name="laser_data", type='SPOT')
    self.laser_data.shadow_soft_size = 0
    self.laser_data.spot_size = 3.14159
    self.laser_data.cycles.max_bounces = 0
    self.laser_data.use_nodes = True  

    lighting_strength = min(max(np.random.normal(loc=4525, scale=500), 4000), 5000)
    self.laser_data.node_tree.nodes["Emission"].inputs[1].default_value = lighting_strength # W/m^2
    self.lasers_watts_per_meters_squared = lighting_strength

    # warp mapping of light
    mapping = self.laser_data.node_tree.nodes.new(type='ShaderNodeMapping')
    mapping.location = 300,0
    mapping.rotation[2] = 3.14159

    mapping.scale[1] = self.horizontal_pixels / float(self.vertical_pixels) # e.g. 1.779 for Laser Beam Pro
    mapping.scale[2] = self.horizontal_size / self.focal_length # e.g. 0.7272404614 for Laser Beam Pro
  
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

  def expand_plane_of_sensor(self, expansion=10.0): # expansion is a multiplier of the size of the sensor plane
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
    # self.highlight_hitpoint(min_h_min_v_corner.xyz(), (1,0,0,1))
    # self.highlight_hitpoint(min_h_max_v_corner.xyz(), (0,1,0,1))
    # self.highlight_hitpoint(max_h_min_v_corner.xyz(), (0,0,1,1))
    # self.highlight_hitpoint(max_h_max_v_corner.xyz(), (1,1,1,1))

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
    self.photonic_plane = obj
    bm.free()

  def reorient(self, orientation_index=0):
    print("Beginning to reorient perspective...")
    time_start = time.time()
    if type(self.focal_point) == type(Point()) and type(self.target_point) == type(Point()):
      self.compute_image_center()
      self.compute_euler_angles()
      self.compute_xyz_of_boundary_pixels()
      self.orient_xyz_and_unit_vectors_for_all_pixels()
      adjusted_euler_z = self.rotation_euler_z * -1.0 # to correct for a notational difference between rendering engine and notes
      if self.photonics == "lasers":
        self.lasers.location = (self.focal_point.x, self.focal_point.y, self.focal_point.z)
        self.lasers.rotation_euler = (self.rotation_euler_x, self.rotation_euler_y, adjusted_euler_z)
        self.lasers.delta_scale = (-1.0, 1.0, 1.0) # flips image that is projected horizontally, to match inverted raycasting
      elif self.photonics == "sensors":
        self.sensors.location = (self.focal_point.x, self.focal_point.y, self.focal_point.z)
        self.sensors.rotation_euler = (self.rotation_euler_x, self.rotation_euler_y, adjusted_euler_z)
        self.expand_plane_of_sensor()
    time_end = time.time()
    print("Orientations of {} computed in {} seconds".format(self.photonics, round(time_end - time_start, 4)))

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

    print("Euler rotations of (x={},y={},z={})".format(math.degrees(self.rotation_euler_x), math.degrees(self.rotation_euler_y), math.degrees(self.rotation_euler_z)))

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
    if self.photonics == "lasers":
      sphere.hide_viewport = True
      self.highlighted_hitpoints.append(sphere)

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

        if obj == environment.mesh:
          self.pixels[h][v].hitpoint_object = "background"
          self.pixels[h][v].hitpoint_face_index = face_index
          self.pixels[h][v].hitpoint_normal = Point(normal[0], normal[1], normal[2])
          background_hits += 1
        elif obj == environment.model_object:
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
    print("Raycasts of {} computed in {} seconds".format(self.photonics, round(time_end - time_start, 4)))
    print("{} hits on the model, {} hits on the background, {} hits elsewhere".format(model_hits, background_hits, other_hits))

class Model():
  def __init__(self, filepath=None):
    if filepath:
      self.filepath = filepath
      self.import_object_to_scan(filepath=filepath)
      self.dimensions = bpy.context.object.dimensions
      self.resample_parameters()

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

    bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='BOUNDS')
    self.model_object = bpy.context.object
    # print("IMPORTED OBJECT STUFF: 1,2")
    bpy.context.object.name = "Model"
    # print(bpy.context.object)
    # print(bpy.context.object.location)

  def resample_parameters(self):
      self.resample_orientation()
      self.resample_size()
      self.resample_position()
      # self.resample_materials()   # no probabilistic materials

  def extract_model_metadata(self):
    metadata = {"filepath": self.filepath,
                "position":         { "x": self.x,
                                      "y": self.y,
                                      "z": self.z
                                    },
                "rotation_euler":   { "x": math.degrees(self.x_rotation_angle), 
                                      "y": math.degrees(self.y_rotation_angle),
                                      "z": math.degrees(self.z_rotation_angle) 
                                    },
                "dimensions_size":  { "x": self.dimensions[0],
                                      "y": self.dimensions[1],
                                      "z": self.dimensions[2]
                                    },
                "materials": self.materials_metadata
                }

    return metadata

  def resample_orientation(self, x_rotation_angle=0.0, y_rotation_angle=0.0, z_rotation_angle=0.0):
    obj = bpy.context.object
    self.x_rotation_angle = math.radians(x_rotation_angle) # random.uniform(0, 2*math.pi)
    self.y_rotation_angle = math.radians(y_rotation_angle) # random.uniform(0, 2*math.pi)
    self.z_rotation_angle = math.radians(z_rotation_angle) # random.uniform(0, 2*math.pi)
    obj.rotation_euler = [self.x_rotation_angle, self.y_rotation_angle, self.z_rotation_angle] # random angular rotations about x,y,z axis
    bpy.context.scene.update() 

  def get_global_vertices(self, local_vertices, world_matrix):
    local_vertices.shape = (-1, 3)
    local_vertices = np.c_[local_vertices, np.ones(local_vertices.shape[0])]
    global_vertices = np.dot(world_matrix, local_vertices.T)[0:3].T.reshape((-1))
    return global_vertices

  def resample_size(self):
    #self.scale_factor = max(np.random.normal(loc=3.0, scale=2.0), 0.75)
    start = time.time()
    print("Starting to measure size of object at {}".format(start))
    min_x = 0.0
    max_x = 0.0
    min_y = 0.0
    max_y = 0.0
    min_z = 0.0

    local_vertices = np.zeros(len(bpy.context.object.data.vertices) * 3) 
    bpy.context.object.data.vertices.foreach_get('co', local_vertices)
    global_vertices = self.get_global_vertices(local_vertices, bpy.context.object.matrix_world)
    x_coordinates = global_vertices[0::3]
    y_coordinates = global_vertices[1::3]
    z_coordinates = global_vertices[2::3]
    print("Total of {} object points ({} coordinates of (x,y,z))".format(len(global_vertices), len(x_coordinates)))
    self.min_x = min(x_coordinates)
    self.max_x = max(x_coordinates)
    self.min_y = min(y_coordinates)
    self.max_y = max(y_coordinates)
    self.min_z = min(z_coordinates)
    self.max_z = max(z_coordinates)
    print("MINIMA: (x,y,z) = ({},{},{})".format(self.min_x, self.min_y, self.min_z))
    print("MAXIMA: (x,y,z) = ({},{},{})".format(self.max_x, self.max_y, self.max_z))
    largest_axis_edge = max(self.max_x-self.min_x, self.max_y-self.min_y)
    naive_estimation_for_field_of_view = 40.0
    self.scale_factor = 1.0 # naive_estimation_for_field_of_view / largest_axis_edge
    print("Scale factor is {}".format(self.scale_factor))

    self.min_x = self.min_x * self.scale_factor
    self.max_x = self.max_x * self.scale_factor
    self.min_y = self.min_y * self.scale_factor
    self.max_y = self.max_y * self.scale_factor
    self.min_z = self.min_z * self.scale_factor
    self.max_z = self.max_z * self.scale_factor

    # for vertex in global_vertices:
    #   print(vertex)
      # vertex = self.get_global_vertex(vertex, bpy.context.object.matrix_world)
      # print(vertex)
      # if vertex.co.x < min_x:
      #   min_x = vertex.co.x 
      # elif vertex.co.x > max_x:
      #   max_x = vertex.co.x
      # if vertex.co.y < min_y:
      #   min_y = vertex.co.y
      # elif vertex.co.y > max_y:
      #   max_y = vertex.co.y
      # if vertex.co.z < min_z:
      #   min_z = vertex.co.z
      # elif vertex.co.z > max_z:
      #   max_z = vertex.co.z

    bpy.context.object.dimensions = (self.dimensions * self.scale_factor / max(self.dimensions))
    bpy.context.scene.update() 
    end = time.time()
    print("Resized object in {} seconds".format(end-start))

  def resample_position(self):
    obj = bpy.context.object
    self.x = max(min(np.random.normal(loc=0.0, scale=0.05), 0.15), -0.15)
    self.y = max(min(np.random.normal(loc=0.0, scale=0.05), 0.15), -0.15)
    self.z = max(min(np.random.normal(loc=0.0, scale=0.05), 0.15), -0.15)
    obj.location.x = self.x
    obj.location.y = self.y
    obj.location.z = self.z
    bpy.context.scene.update() 

  def resample_materials(self):
    obj = bpy.context.object
    self.materials_metadata = {}

    material_slot_index = 0
    for material_slot in obj.material_slots:
      material_name = material_slot.name
      metadata = {"index": material_slot_index, "name": material_name}

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

      metadata["red"] = red
      metadata["green"] = green
      metadata["blue"] = blue
      metadata["alpha"] = alpha

      metallic = np.random.choice(a=[0.0, 1.0], p=[0.80, 0.20]) # left is choice, right is probabilty
      shader.inputs['Metallic'].default_value = metallic
      metadata["metallic"] = metallic

      # weighted mixture of gaussians
      ior_guassian_a = np.random.normal(loc=1.45, scale=0.15)
      ior_gaussian_b = np.random.normal(loc=1.45, scale=0.5)
      ior = max(np.random.choice(a=[ior_guassian_a, ior_gaussian_b], p=[0.80, 0.20]), 1.0)
      shader.inputs['IOR'].default_value = ior
      metadata["index_of_refraction"] = ior

      specular = ((ior-1.0)/(ior+1.0))**2 / 0.08 # from "special case of Fresnel formula" as in https://docs.blender.org/manual/en/dev/render/cycles/nodes/types/shaders/principled.html
      shader.inputs['Specular'].default_value = specular
      metadata["specular"] = specular

      # weighted mixture of guassians
      transmission_guassian_a = np.random.normal(loc=0.1, scale=0.3)
      transmission_gaussian_b = np.random.normal(loc=0.9, scale=0.3)
      transmission = max(min(np.random.choice(a=[transmission_guassian_a, transmission_gaussian_b], p=[0.75, 0.25]), 1.0), 0.0)
      shader.inputs['Transmission'].default_value = transmission
      metadata["transmission"] = transmission

      transmission_roughness = random.uniform(0, 1)
      shader.inputs['Transmission Roughness'].default_value
      metadata["transmission_roughness"] = transmission_roughness

      roughness = random.uniform(0, 1)
      shader.inputs['Roughness'].default_value = roughness
      metadata["roughness"] = roughness

      anisotropic = max(min(np.random.normal(loc=0.1, scale=0.3), 1.0), 0.0)
      shader.inputs['Anisotropic'].default_value = anisotropic
      metadata["anisotropic"] = anisotropic

      anisotropic_rotation = random.uniform(0,1)
      shader.inputs['Anisotropic Rotation'].default_value = anisotropic_rotation
      metadata["anisotropic_rotation"] = anisotropic_rotation

      sheen = max(min(np.random.choice(a=[0.0, np.random.normal(loc=0.5, scale=0.25)], p=[0.9, 0.1]), 1.0), 0.0)
      shader.inputs['Sheen'].default_value = sheen
      metadata["sheen"] = sheen

      self.materials_metadata[int(material_slot_index)] = metadata

      # infrastructure
      node_output = nodes.new(type='ShaderNodeOutputMaterial')   
      node_output.location = (400,0)
      links = m.node_tree.links
      link = links.new(shader.outputs[0], node_output.inputs[0])

      material_slot_index += 1

    bpy.context.scene.update() 

class Environment():
  def __init__(self, cloud_compute=True):
    if cloud_compute:
      model_directory="/home/ubuntu/COLLADA"
      models = [f for f in listdir(model_directory) if path.isfile(path.join(model_directory, f)) and ".dae" in f]
      #sampled_model_index = int(random.uniform(0, len(models) - 1))
      #model = models[sampled_model_index]
      model = random.sample(population=models, k=1)[0]
      print("Randomly sampled {}".format(model))
      filepath = path.join(model_directory, model)
      self.model_name = model
      self.resample_environment(filepath)
    else:
      self.resample_environment("/Users/3co/research/phone.dae") # check for the almighty smartphone in your pocket

  def resample_environment(self, model):
    self.add_model(model_filepath=model)
    self.ambient_lighting()
    #self.create_mesh()
    #self.create_materials()
    #self.index_materials_of_faces()
    #self.analyze_perspective()

  def extract_environment_metadata(self):
    model_metadata = self.model.extract_model_metadata()
    model_materials_to_faces = self.invert_faces_to_materials(self.model_face_index_to_material_index)
    environment_materials_to_faces = self.invert_faces_to_materials(self.environment_face_index_to_material_index)
    model_metadata["material_index_to_face_indices"] = model_materials_to_faces 
    
    metadata = {"model": model_metadata,
                "ambient_lighting": 
                    { "watts_per_meters_squared": self.ambient_light_strength,
                      "position":
                      { "x": self.ambient_light_position.x,
                        "y": self.ambient_light_position.y,
                        "z": self.ambient_light_position.z
                      } 
                    },
                "background": 
                    { "position": 
                      { "x": 0.0,
                        "y": 0.0,
                        "z": self.distance_from_origin
                      },
                      "rotation_euler":
                      { "x": math.degrees(self.x_rotation_angle),
                        "y": math.degrees(self.y_rotation_angle),
                        "z": math.degrees(self.z_rotation_angle)
                      },
                      "material": self.background_material_metadata ,
                      "material_index_to_face_indices": environment_materials_to_faces
                    },
                "analysis":
                    {
                      "extrema": {
                        "maxima": {
                          "x": self.model.max_x,
                          "y": self.model.max_y,
                          "z": self.model.max_z
                        },
                        "minima": {
                          "x": self.model.min_x,
                          "y": self.model.min_y,
                          "z": self.model.min_z
                        }
                      },
                      "midpoints": { 
                          "x": self.x_midpoint,
                          "y": self.y_midpoint,
                          "z": self.z_midpoint
                        },
                      "object_enclosure_dimensions": {
                          "x_edge": self.x_edge_size,
                          "y_edge": self.y_edge_size,
                          "z_edge": self.z_edge_size
                      }
                    }
                }

    self.metadata = metadata
    return metadata

  def invert_faces_to_materials(self, faces_to_materials):
    materials_to_faces_using_a_list = {}
    for face, material in faces_to_materials.items():
      list_of_faces = materials_to_faces_using_a_list.get(material, [])
      list_of_faces.append(int(face))
      materials_to_faces_using_a_list[material] = list_of_faces

    materials_to_faces = {}
    for material, faces in materials_to_faces_using_a_list.items():
      materials_to_faces[material] = ",".join([str(face) for face in faces])

    return materials_to_faces

  def index_materials_of_faces(self):
    objects = {}
    for i, obj in enumerate(bpy.data.objects):
      obj.select_set( state = False, view_layer = None)
      if obj.name == "Environment":
        objects["Environment"] = obj
      elif obj.name == "Model":
        objects["Model"] = obj 

    objects["Model"].select_set( state = True, view_layer = None)
    self.model_materials = {}
    self.model_face_index_to_material_index = {}
    #active_object = bpy.context.active_object

    for face in objects["Model"].data.polygons:  # iterate over faces
      material = objects["Model"].material_slots[face.material_index].material
      self.model_face_index_to_material_index[face.index] = face.material_index
      self.model_materials[face.index] = material

    objects["Model"].select_set( state = False, view_layer = None)
    objects["Environment"].select_set( state = True, view_layer = None)

    self.environment_materials = {}
    self.environment_face_index_to_material_index = {}
    for face in objects["Environment"].data.polygons:  # iterate over faces
      material = objects["Environment"].material_slots[face.material_index].material
      self.environment_face_index_to_material_index[face.index] = face.material_index
      self.environment_materials[face.index] = material

  def analyze_perspective(self):
    self.x_midpoint = (self.model.min_x + self.model.max_x) / 2.0
    self.y_midpoint = (self.model.min_y + self.model.max_y) / 2.0
    self.z_midpoint = (self.model.min_z + self.model.max_z) / 2.0
    self.x_edge_size = self.model.max_x - self.model.min_x 
    self.y_edge_size = self.model.max_y - self.model.min_y 
    self.z_edge_size = self.model.max_z - self.model.min_z 
    self.limiting_edge = max(self.x_edge_size, self.y_edge_size)
    self.scale_factor = self.model.scale_factor

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

    env = objects.get("Environment", None)
    if env:
      env.select_set(True)
      bpy.ops.object.delete()

    objects["Model"].select_set(True)
    bpy.ops.object.delete()

    objects["Ambient Light"].select_set(True)
    bpy.ops.object.delete() 

  def setup_preferences(self):
    bpy.ops.wm.read_factory_settings(use_empty=True) # initialize empty world, removing default objects

  def add_model(self,model_filepath):
    self.model = Model(model_filepath)
    self.model_object = self.model.model_object

  def ambient_lighting(self):
    # add light
    light = bpy.data.lights.new(name="sun", type='SUN')
    light.use_nodes = True  

    #light_power = min(max(np.random.normal(loc=0.05, scale=0.05), 0.001), 0.2)
    light_power = 75.0
    light.node_tree.nodes["Emission"].inputs[1].default_value = light_power
    self.ambient_light_strength = light_power

    self.light = bpy.data.objects.new(name="Ambient Light", object_data=light)

    #x = np.random.normal(loc=0.0, scale=1.0)
    #y = np.random.normal(loc=0.0, scale=1.0)
    #z = np.random.normal(loc=10.0, scale=2.5)

    x = 10.0
    y = 0.0
    z = 0.0 

    self.ambient_light_position = Point(x,y,z)

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

    self.distance_from_origin = max(np.random.normal(loc=-2.0, scale=0.25), -0.25)

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
    self.background_material_metadata = {}
    self.vinyl_material = bpy.data.materials.new(name="vinyl_backdrop_material")
    self.vinyl_material.use_nodes = True
    nodes = self.vinyl_material.node_tree.nodes
    for node in nodes:
      nodes.remove(node)
    self.vinyl = nodes.new(type='ShaderNodeBsdfPrincipled')

    # parameterization by *The Principled Shader* (insert Blender guru accent)
    red = min(np.random.normal(loc=0.995, scale=0.03), 1.0)
    green = min(np.random.normal(loc=0.995, scale=0.03), 1.0)
    blue = min(np.random.normal(loc=0.995, scale=0.03), 1.0)
    alpha = min(np.random.normal(loc=0.995, scale=0.03), 1.0)
    #self.vinyl_material.diffuse_color = (red,green,blue)
    self.vinyl.inputs['Base Color'].default_value = (red, green, blue, alpha)
    self.background_material_metadata["red"] = red
    self.background_material_metadata["green"] = green
    self.background_material_metadata["blue"] = blue
    self.background_material_metadata["alpha"] = alpha

    metallic = 0.0
    self.vinyl.inputs['Metallic'].default_value = metallic
    self.background_material_metadata["metallic"] = metallic

    # weighted mixture of gaussians
    ior = np.random.normal(loc=1.45, scale=0.02)
    self.vinyl.inputs['IOR'].default_value = ior
    self.background_material_metadata["index_of_refraction"] = ior

    specular = ((ior-1.0)/(ior+1.0))**2 / 0.08 # from "special case of Fresnel formula" as in https://docs.blender.org/manual/en/dev/render/cycles/nodes/types/shaders/principled.html
    self.vinyl.inputs['Specular'].default_value = specular
    self.background_material_metadata["specular"] = specular

    # weighted mixture of guassians
    transmission = max(np.random.normal(loc=0.1, scale=0.1), 0.0)
    self.vinyl.inputs['Transmission'].default_value = transmission
    self.background_material_metadata["transmission"] = transmission

    transmission_roughness = random.uniform(0, 1)
    self.vinyl.inputs['Transmission Roughness'].default_value = transmission_roughness
    self.background_material_metadata["transmission_roughness"] = transmission_roughness


    roughness = min(np.random.normal(loc=0.8, scale=0.1),1.0)
    self.vinyl.inputs['Roughness'].default_value = roughness
    self.background_material_metadata["roughness"] = roughness


    sheen = max(min(np.random.normal(loc=0.6, scale=0.2), 1.0), 0.0)
    self.vinyl.inputs['Sheen'].default_value = sheen
    self.background_material_metadata["sheen"] = sheen

    self.vinyl.location = (0,0)
    node_output = nodes.new(type='ShaderNodeOutputMaterial')   
    node_output.location = (400,0)
    links = self.vinyl_material.node_tree.links
    link = links.new(self.vinyl.outputs[0], node_output.inputs[0])
    self.mesh.data.materials.append(self.vinyl_material)


class Scanner():
  def __init__(self, sensors, lasers=None, environment=None):
    self.environment = environment
    self.sensors = sensors
    self.lasers = lasers

  def scan(self, target_point=None, counter=0, precomputed=False, sensor_as_scanner=False):
    self.sensors.sensor_as_scanner = sensor_as_scanner 
    launch_time = time.time()
    print("LAUNCH TIME: {}".format(launch_time))
    if type(target_point) == type(None): # derive scanning location(s) based on topology of object(s)
      x = 0.0 # np.random.normal(loc=self.environment.x_midpoint, scale=0.10 * self.environment.x_edge_size)
      y = 0.0 # np.random.normal(loc=self.environment.y_midpoint, scale=0.10 * self.environment.y_edge_size)
      z = 0.0 # np.random.normal(loc=self.environment.z_midpoint, scale=0.10 * self.environment.z_edge_size)
      target_point = Point(x, y, z)
      print("Scanner targeting point ({}, {}, {})".format(x, y, z))

    # if lasers and/or sensors have a new target point, reorient
    if self.lasers:
      if self.lasers.target_point != target_point:
        self.lasers.target_point = target_point
        self.lasers.reorient(orientation_index=counter)

    if self.sensors.target_point != target_point:
      self.sensors.target_point = target_point
      self.sensors.reorient(orientation_index=counter)

    if self.lasers:
      self.localizations = []
      self.lasers.measure_raycasts_from_pixels(environment=self.environment)

    if sensor_as_scanner:
      self.sensors.measure_raycasts_from_pixels(environment=self.environment)

    # localization to nearest pixel of sensor via localization: source of error // subpixel considerations

    self.render("{}/{}.png".format(output_directory, int(launch_time)))

    if self.lasers and compute_localizations: 
      self.localize_projections_in_sensor_plane()

    self.save_metadata(int(launch_time))

    if self.lasers and compute_localizations:
      self.visualize_ground_truth_pixel_overlap(int(launch_time))

  def move(self, x=None, y=None, z=None, pitch=None, yaw=None, turntable=None):
    if x != None:
      self.sensors.focal_point.x = x
    if y != None:
      self.sensors.focal_point.y = y
    if z != None:
      self.sensors.focal_point.z = z
    
    if pitch == None:
      pitch = self.sensors.rotation_euler_x
    else:
      pitch = math.radians(pitch)

    if yaw == None:
      yaw = math.radians(180 + math.degrees(self.sensors.rotation_euler_z)) # if using Blender's yaw, we need to convert e.g. -90 degrees to 90 degrees, -180 degrees to 0 degrees, for our calculations
    else:
      yaw = math.radians(yaw) 

    if turntable != None:
      self.environment.model.resample_orientation(z_rotation_angle=z_rotation_angle)

    # geometry based on coordinate system with turntable origin at (0,0,0); see https://docs.google.com/document/d/1FsgnzzdmZE0qz_1uw7lePc5e3lh1HGlXNSBlKcXP4hU/edit?usp=sharing
    x_target = self.sensors.focal_point.x - math.sin(yaw) * math.sin(pitch)
    y_target = self.sensors.focal_point.y - math.cos(yaw) * math.sin(pitch)
    z_target = self.sensors.focal_point.z - math.cos(pitch) 

    self.sensors.target_point = Point(x_target, y_target, z_target)

    if x != None or y != None or z != None or pitch != None or yaw != None:
      self.sensors.reorient()


  def render(self, filename):
    print("Rendering {}".format(filename))
    time_start = time.time()
    bpy.data.scenes["Scene"].render.filepath = filename
    bpy.ops.render.render( write_still=True )
    time_end = time.time()
    print("Rendered image in {} seconds".format(round(time_end - time_start, 4)))


  def save_metadata(self, launch_time):
    environment_metadata = self.environment.extract_environment_metadata() 
    sensors_metadata = self.sensors.extract_optical_metadata(launch_time=launch_time)
    if self.lasers:
      lasers_metadata = self.lasers.extract_optical_metadata(launch_time=launch_time)
    else:
      lasers_metadata = ""

    unix_time = int(time.time())
    human_time = time.ctime(unix_time)
    human_launch_time = time.ctime(launch_time)

    metadata = {"time": 
                  { "launch_unix_time": launch_time,
                    "launch_human_time": human_launch_time,
                    "save_unix_time": unix_time,
                    "save_human_time": human_time 
                  },
                "environment": environment_metadata, 
                "sensors": sensors_metadata,
                "lasers": lasers_metadata
                }

    #pprint(metadata)
    filename = "{}/{}_metadata.json".format(output_directory, launch_time)
    with open(filename, "w") as json_file:
      json.dump(metadata, json_file)
      print("Created {} file with metadata on render".format(filename))


  def visualize_ground_truth_pixel_overlap(self, launch_time):
    render_filename = "{}/{}.png".format(output_directory, launch_time)

    rgb_projection = Image.open(self.lasers.image)
    vertical_pixels = self.lasers.vertical_pixels
    horizontal_pixels = self.lasers.horizontal_pixels

    render = Image.open("{}/{}.png".format(output_directory, launch_time))
    rendered_horizontal_pixels, rendered_vertical_pixels = render.size
    render_pixels = render.load()

    localization = Image.new('RGB', (rendered_horizontal_pixels, rendered_vertical_pixels), color = 'white')
    localization_pixels = localization.load()

    with open("{}/{}_metadata.json".format(output_directory, launch_time)) as json_file:
      data = json.load(json_file)
      pixels = data['lasers']['pixel_metadata']
      for h in self.lasers.get_pixel_indices("horizontal"):    
        for v in self.lasers.get_pixel_indices("vertical"): 
          if simulation_mode == "TEST": # no visualizations since there were no raycasts
            continue

          relative_h = pixels[4*(h*vertical_pixels + v)]
          relative_v = pixels[4*(h*vertical_pixels + v) + 1]    
                  
          if relative_h == "OCCLUDED" and relative_v == "OCCLUDED":
            continue

          h_of_render = int(relative_h * rendered_horizontal_pixels) # axis flipped, see https://3co.s3.amazonaws.com/cdl/ground_truth_localization.png
          v_of_render = rendered_vertical_pixels - int(relative_v * rendered_vertical_pixels) 

          if h_of_render >= rendered_horizontal_pixels or h_of_render < 0 or v_of_render >= rendered_vertical_pixels or v_of_render < 0:
            continue

          pixel = rgb_projection.getpixel((horizontal_pixels-h-1,vertical_pixels-v-1))
          localization_pixels[h_of_render, v_of_render] = pixel
          render_pixels[h_of_render, v_of_render] = pixel

    localization.save('{}/{}_localization_only.png'.format(output_directory, launch_time))
    render.save('{}/{}_overlapping.png'.format(output_directory, launch_time))


  def localize_projections_in_sensor_plane(self):
    time_start = time.time()
    self.environment.model.model_object.hide_viewport = False # True False = OCCLUSION FROM OBJECT
    self.environment.mesh.hide_viewport = True
    img = Image.open(self.lasers.image)

    for h in self.lasers.get_pixel_indices("horizontal"):    
      for v in self.lasers.get_pixel_indices("vertical"): 
        origin = self.lasers.pixels[h][v].hitpoint
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

        #if not hit:
        #  print("No secondary hitpoint on sensor plane for raycast from hitpoint of projected pixel ({},{})".format(h, v))
        #  print("Try expanding the size of the sensor plane".format(h, v))


        if not hit:
          #print("No secondary hitpoint on sensor plane for raycast from hitpoint of projected pixel ({},{})".format(h, v))
          self.lasers.pixels[h][v].hitpoint_in_sensor_plane = Point("None", "None", "None")
        else:
          self.lasers.pixels[h][v].hitpoint_in_sensor_plane = Point(location[0], location[1], location[2])

        if obj == self.environment.mesh:
          self.lasers.pixels[h][v].hitpoint_in_sensor_plane_object = "background"
        elif obj == self.environment.model_object:
          self.lasers.pixels[h][v].hitpoint_in_sensor_plane_object = "model"
        elif obj == self.sensors.photonic_plane:
          self.lasers.pixels[h][v].hitpoint_in_sensor_plane_object = "sensor_plane" 
        else:
          self.lasers.pixels[h][v].hitpoint_in_sensor_plane_object = "None" 

    self.localization_in_sensor_coordinates()
    # for hitpoint in self.lasers.highlighted_hitpoints:
    #   hitpoint.hide_viewport = False

    self.environment.model.model_object.hide_viewport = False
    self.environment.mesh.hide_viewport = False

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

    img = Image.open(self.lasers.image)
    occlusions = 0
    out_of_image = 0
    for h in self.lasers.get_pixel_indices("horizontal"):   
      for v in self.lasers.get_pixel_indices("vertical"): 
        pixel = img.getpixel((h,v))
        if self.lasers.pixels[h][v].hitpoint_in_sensor_plane_object == "model":
          self.lasers.pixels[h][v].projected_pixel_color = {"red": pixel[0]/float(255), "green": pixel[1]/float(255), "blue": pixel[2]/float(255)}
          self.lasers.pixels[h][v].relative_h = "OCCLUDED"
          self.lasers.pixels[h][v].relative_v = "OCCLUDED"
          occlusions += 1
        else:
          hitpoint = self.lasers.pixels[h][v].hitpoint_in_sensor_plane
          numerator_relative_v = ((hitpoint.x - origin.x) - (hitpoint.y - origin.y) * unit_h_xy)
          relative_v = numerator_relative_v / normalizing_v_denominator
          numerator_relative_h = ( (hitpoint.y - origin.y) - relative_v * y_v)
          relative_h = numerator_relative_h / normalizing_h_denominator
          relative_projected_h = h / float(self.lasers.horizontal_pixels)
          relative_projected_v = v / float(self.lasers.vertical_pixels)
          self.lasers.pixels[h][v].projected_pixel_color = {"red": pixel[0]/float(255), "green": pixel[1]/float(255), "blue": pixel[2]/float(255)}
          self.lasers.pixels[h][v].relative_h = relative_h
          self.lasers.pixels[h][v].relative_v = relative_v
          if relative_h <= 0.0 or relative_h >= 1.0 or relative_v <= 0.0 or relative_v >= 1.0:
            #print("Out of sensor plane: ({},{})".format(relative_h, relative_v))
            out_of_image += 1

    total_pixels = self.lasers.horizontal_pixels * self.lasers.vertical_pixels 
    print("{} of {} laser pixels were occluded by object".format(occlusions, total_pixels))
    print("{} of {} laser pixels are outside of image field of view".format(out_of_image, total_pixels))
    print("{} of {} laser pixels have been localized in image".format(total_pixels - occlusions - out_of_image, total_pixels))

if __name__ == "__main__":
  begin_time = time.time()
  print("\n\nSimulation beginning at UNIX TIME {}".format(int(begin_time)))
  
  # read files to render
  models = []
  with open("/home/ubuntu/reconstructables/reconstructables.txt", "r") as reconstructables:
    for reconstructable in reconstructables:
      model = reconstructable.rstrip("\n")
      filepath = "/home/ubuntu/reconstructables/data/{}".format(model)
      models.append(filepath)

  # initialize environment and scanner
  environment = Environment()
  sensors = Optics(photonics="sensors", environment=environment, target_point=Point(0.0,0.0,0.0))
  sensors.reorient()
  #lasers = Optics(photonics="lasers", environment=environment, vertical_pixels=100, horizontal_pixels=160, image="160x100rgb.png", position_anchor=sensors) 
  scanner = Scanner(sensors=sensors, lasers=None, environment=environment)

  for i, model in enumerate(models):
    if i == 1:
      print("Moving pitch up to 110 degrees...")
      scanner.move(pitch=110)
    if i == 2:
      print("Moving pitch back to 90 degrees, yaw to 110 degrees (facing right of view)...")
      scanner.move(pitch=90, yaw=110)
    if i == 3:
      print("Moving yaw to 70 degrees (facing left of view)...")
      scanner.move(yaw=70)
    if i == 4:
      print("Moving pitch to 0 degrees...")
      scanner.move(pitch=0)
    if i == 5:
      print("Moving yaw to 0 degrees...")
      scanner.move(yaw=0)
    if i == 6:
      print("Moving pitch and yaw to 90 degrees...")
      scanner.move(pitch=90, yaw=90)

    for z_rotation_angle in [0]:  #,90, 120, 150, 180, 210, 240, 270, 300, 330]:
      environment.delete_environment()
      environment.resample_environment(model=model)
      environment.model.resample_orientation(z_rotation_angle=z_rotation_angle)
      x = scanner.sensors.focal_point.x
      y = scanner.sensors.focal_point.y
      z = scanner.sensors.focal_point.z
      pitch = scanner.sensors.rotation_euler_x
      yaw = scanner.sensors.rotation_euler_z
      print("(x,y,z,pitch,yaw)=({},{},{},{},{})".format(x,y,z,math.degrees(pitch),math.degrees(yaw)))
      print("(x,y,z) of target point: ({},{},{})".format(scanner.sensors.target_point.x, scanner.sensors.target_point.y, scanner.sensors.target_point.z))
   
      scanner.render("{}/experimental_model_{}_file_{}_table_{}.png".format(output_directory, i, model.rstrip(".dae").split("/")[-1], z_rotation_angle))

    if i == 6:
      break

    #scanner.scan(sensor_as_scanner=True) 

  end_time = time.time()
  print("\n\nSimulation finished in {} seconds".format(end_time - begin_time))


# # API
# scanner.move(x,y,z,pitch,yaw,turntable)
# scanner.scan()