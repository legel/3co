import numpy as np
import random
import time
import logging
import sys
from scipy.optimize import minimize
import multiprocessing
from PIL import Image
import math
from os import listdir, path, getcwd, remove, devnull
import pickle
from plyfile import PlyData, PlyElement
import warnings
warnings.filterwarnings(action='ignore', message='Mean of empty slice')

camera_roi_width = 256
camera_roi_height = 32
total_pixels = camera_roi_width * camera_roi_width 

# launch parallel processing engine
cpus = multiprocessing.cpu_count() 
np.seterr(divide='ignore', invalid='ignore')
sys.stdout.flush()

transforms = {}
changes = ["increasing", "decreasing"]
axes = ["x","y","z","pitch","yaw"]
types_of_transform = ["translation", "rotation"]
for axis in axes:
    transforms[axis] = {}
    for change in changes:
        transforms[axis][change] = {}
        for type_of_transform in types_of_transform:
            with open('/home/sense/3cobot/{}_{}_{}_matrix.npy'.format(axis, change, type_of_transform), 'rb') as input_file:
                transforms[axis][change][type_of_transform] = np.load(input_file)



class PointCloud():
    # to do: automatic reindex_into_tensor() and reindex_tensor_back_into_core() after every relevant change
    def __init__(self, scanner_output=None, filename=None, distance_center_sample_size=4, project_name="x", scan_index=0, relative_scan_index=0, reload_after_saving=False, focus_on_center_pixels=False, row_crop=400, column_crop=400, ignore_dimensions=False, reindex_into_tensor=True, clip_outer_n_pixels=0, colors_from_photo=None, export_to_npy=False, camera_coordinate_system="camera_coordinate_system"):
        #print("Point cloud from filename {}".format(filename))
  
        if camera_coordinate_system == "projector_coordinate_system":
            self.camera_image_width = 1824
            self.camera_image_height = 2280
        elif camera_coordinate_system == "camera_coordinate_system":
            self.camera_image_width = 2048
            self.camera_image_height = 2048  
 
        self.superpoints = []
        self.scan_index = scan_index # temporary scalar placeholder, converted to numpy array later in load( ), used for indexing multiple scans that may be embedded into single point cloud
        self.relative_scan_index = relative_scan_index
        self.image_width = self.camera_image_width
        self.image_height = self.camera_image_height
        if scanner_output:              
            self.scanner_output = scanner_output
            self.filename = "{}-{}-{}.ply".format(project_name, scan_index, relative_scan_index)
            self.save_scanner_output()
            self.load(reindex_into_tensor=True, clip_outer_n_pixels=0) # clip outer_n_pixels is currently broken
            if type(colors_from_photo) != type(None):
                self.set_colors_by_photo(colors_from_photo)
            # self.filter_nan_values()
            self.save_as_ply(filename=self.filename, from_tensor=True)
            if export_to_npy:
                self.export_to_npy(project_name=project_name, from_tensor=True)

        elif filename:
            self.scanner_output = None
            if ".ply" in filename:
                self.timestamp = filename.split(".ply")[0]
            elif ".npy" in filename:
                self.timestamp = filename.split(".npy")[0]
            self.filename = filename
            self.load(ignore_dimensions=True, reindex_into_tensor=reindex_into_tensor, clip_outer_n_pixels=0, assert_scan_index=self.scan_index, camera_coordinate_system=camera_coordinate_system)


    def euler_angles_to_rotation_matrix(self, euler_x, euler_y, euler_z):
        R_x = np.array([[1,         0,                  0                ],
                        [0,         math.cos(euler_x), -math.sin(euler_x)],
                        [0,         math.sin(euler_x), math.cos(euler_x) ]])
        
        R_y = np.array([[math.cos(euler_y),    0,      math.sin(euler_y) ],
                        [0,                    1,      0                 ],
                        [-math.sin(euler_y),   0,      math.cos(euler_y) ]])
                
        R_z = np.array([[math.cos(euler_z),    -math.sin(euler_z),      0],
                        [math.sin(euler_z),    math.cos(euler_z),       0],
                        [0,                    0,                       1]])
                       
        R = np.dot(R_z, np.dot(R_y, R_x))

        return R


    def rotation_matrix_to_euler_angles(self, rotation_matrix):
        def is_rotation_matrix(rotation_matrix):
            transposed_rotation_matrix = np.transpose(rotation_matrix)
            should_be_identity = np.dot(transposed_rotation_matrix, rotation_matrix)
            identity = np.identity(3, dtype = rotation_matrix.dtype)
            distance = np.linalg.norm(identity - should_be_identity)
            return distance - 1e-6

        assert(is_rotation_matrix(rotation_matrix))

        s_y = math.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + rotation_matrix[1,0] * rotation_matrix[1,0])
        singular = s_y < 1e-6
        if not singular:
            euler_x = math.atan2(rotation_matrix[2,1], rotation_matrix[2,2])
            euler_y = math.atan2(-1 * rotation_matrix[2,0], s_y)
            euler_z = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0])
        else:
            euler_x = math.atan2(-1 * rotation_matrix[1,2], rotation_matrix[1,1])
            euler_y = math.atan2(-1 * rotation_matrix[2,0], s_y)
            z = 0
        return np.array([euler_x, euler_y, euler_z])


    def scale_rotation_matrix(self, unit_rotation_matrix, scale_of_transform):
        unit_euler_x, unit_euler_y, unit_euler_z = self.rotation_matrix_to_euler_angles(unit_rotation_matrix)
        scaled_euler_x = unit_euler_x * abs(scale_of_transform)
        scaled_euler_y = unit_euler_y * abs(scale_of_transform)
        scaled_euler_z = unit_euler_z * abs(scale_of_transform)
        scaled_rotation_matrix = self.euler_angles_to_rotation_matrix(scaled_euler_x, scaled_euler_y, scaled_euler_z)
        return scaled_rotation_matrix

    def transform_axis(self, axis, end_position, start_position):
        if axis in ["x","y","z"]:
            axis_transform_scale = 0.01 # centimeters
        elif axis in ["pitch", "yaw"]:
            axis_transform_scale = 1.00 # degrees
        difference = end_position - start_position
        scale_of_transform = difference / axis_transform_scale
        if difference > 0:
            unit_rotation_matrix = transforms[axis]["increasing"]["rotation"]
            unit_translation_matrix = transforms[axis]["increasing"]["translation"]
        elif difference < 0:
            unit_rotation_matrix = transforms[axis]["decreasing"]["rotation"]
            unit_translation_matrix = transforms[axis]["decreasing"]["translation"]
        if difference != 0.0:
            scaled_rotation_matrix = self.scale_rotation_matrix(unit_rotation_matrix, scale_of_transform)
            scaled_translation_matrix = unit_translation_matrix * scale_of_transform 
            print("Transforming {} from {} to {} with translation {} and rotation {}".format(axis, start_position, end_position, scaled_translation_matrix, scaled_rotation_matrix))
            self.transform(scaled_rotation_matrix, scaled_translation_matrix)

    def transform_by_robot_commands(self,start_command, end_command):
        canonical_pitch = -90.0
        canonical_yaw = 90.0

        # begin by transforming start_command to canonical (pitch, yaw) orientation, where unit vector transformations are derived
        self.transform_axis(axis="pitch", end_position=canonical_pitch, start_position=start_command["pitch"])
        self.transform_axis(axis="yaw", end_position=canonical_yaw, start_position=start_command["yaw"])

        # then transform the (x,y,z) from start to finish position
        self.transform_axis(axis="x", end_position=end_command["x"], start_position=start_command["x"])
        self.transform_axis(axis="y", end_position=end_command["y"], start_position=start_command["y"])
        self.transform_axis(axis="z", end_position=end_command["z"], start_position=start_command["z"])

        # then transform from canonical yaw position to final yaw position, and then lastly transform pitch out of its canonical position
        self.transform_axis(axis="yaw", end_position=end_command["yaw"], start_position=canonical_yaw)        
        self.transform_axis(axis="pitch", end_position=end_command["pitch"], start_position=canonical_pitch)


    def focus_on_center_pixels(self, row_crop=200, column_crop=300):
        self.delete_by_pixel_position(dimension="column", start=self.camera_image_height-column_crop, end=self.camera_image_height)
        self.delete_by_pixel_position(dimension="column", start=0, end=column_crop)
        self.delete_by_pixel_position(dimension="row", start=self.camera_image_width-row_crop, end=self.camera_image_width)
        self.delete_by_pixel_position(dimension="row", start=0, end=row_crop)       


    def process_depth_map(self, distance_center_sample_size=4):
        if len(self.scanner_output.DepthMapRef().imageData) > 0:
            image = self.scanner_output.DepthMapRef()
            self.depth_map = np.zeros(shape=(image.height, image.width, 1), dtype=np.float32)
            image.WriteToMemory32(self.depth_map)
            self.print_sample_of_points_in_center(distance_center_sample_size=distance_center_sample_size)

    # def save_png_from_3d(self, photo_label="", missing_data_as="alpha"):
    #     if missing_data_as == "alpha":
    #         image_from_3d_data = Image.new('RGBA', (self.image_height, self.image_width), (0, 0, 0, 0))
    #     elif missing_data_as == "black":
    #         image_from_3d_data = Image.new('RGBA', (self.image_height, self.image_width), (0, 0, 0, 255))
    #     elif missing_data_as == "blue":
    #         image_from_3d_data = Image.new('RGBA', (self.image_height, self.image_width), (0, 0, 255, 255))

    #     output_pixels = image_from_3d_data.load()
    #     for r,g,b,row,column in zip(self.red, self.green, self.blue, self.pixel_row, self.pixel_column):
    #         output_pixels[int(row),int(column)] = (r,g,b, 255)

    #     image_from_3d_data = image_from_3d_data.transpose(Image.FLIP_LEFT_RIGHT)
    #     image_from_3d_data = image_from_3d_data.transpose(Image.ROTATE_270)

    #     if photo_label != "":
    #         image_from_3d_data.save("{}.png".format(photo_label))
    #     else:
    #         image_from_3d_data.save("{}.png".format(self.timestamp))

    def set_all_points_to_color(self, red, green, blue):
        number_of_points = len(self.x)
        all_reds = np.full(shape=number_of_points, fill_value=red, dtype=np.int8)
        all_greens = np.full(shape=number_of_points, fill_value=green, dtype=np.int8)
        all_blues = np.full(shape=number_of_points, fill_value=blue, dtype=np.int8)
        self.red = all_reds
        self.green = all_greens
        self.blue = all_blues


    def print_sample_of_points_in_center(self, distance_center_sample_size=4):
        got_z_value = False
        while not got_z_value:
            center_z_value_points = []
            for vertical_pixel in range(int(self.camera_image_height/2.0) - distance_center_sample_size,  int(self.camera_image_height/2.0) + distance_center_sample_size):
                for horizontal_pixel in range(int(self.camera_image_width/2.0) - distance_center_sample_size, int(self.camera_image_width/2.0) + distance_center_sample_size):
                    z_value = self.depth_map[vertical_pixel, horizontal_pixel]
                    if z_value != 0.0:
                        center_z_value_points.append(z_value)
            if len(center_z_value_points) > 0: 
                average_center_z = sum(center_z_value_points) / float(len(center_z_value_points))
                self.average_center_z = average_center_z[0]
                got_z_value = True
                print("\nZ-distance in center of field of view: {} millimeters\n".format(average_center_z[0]))
            else:
                print("No points in FOV of center {}x{} pixels, expanding by 2x".format(distance_center_sample_size*2, distance_center_sample_size*2))
                distance_center_sample_size = distance_center_sample_size * 2
                if distance_center_sample_size * 2 >= self.camera_image_width:
                    print("No average center point extracted from image")
                    self.average_center_z = "N/A"
                    got_z_value = True          

    def filter(self, min_z, max_z):
        indices_to_delete = np.where(np.logical_or(self.z < min_z, self.z > max_z))[0]
        print("{:,} points outside of {}mm and {}mm to delete".format(len(indices_to_delete), int(min_z), int(max_z)))
        self.delete_points(indices_to_delete)

    def save_scanner_output(self, filename=None):
        if not filename:
            filename = self.filename
        
        self.scanner_output.SavePointCloud(filename)

    def assert_all_indices_aligned(self):
        # ensure that we are working with data from the Ajile scanner
        total_pixels_in_this_scan = self.camera_image_width * self.camera_image_height

        assert(total_pixels_in_this_scan == len(self.x))
        assert(len(self.x) == len(self.y))
        assert(len(self.y) == len(self.z))
        assert(len(self.z) == len(self.red))
        assert(len(self.red) == len(self.green))
        assert(len(self.green) == len(self.blue))
        assert(len(self.blue) == len(self.pixel_row))
        assert(len(self.pixel_row) == len(self.pixel_column))
        assert(len(self.pixel_row) == len(self.scan_index))

    def delete_points(self, indices):
        self.x = np.delete(self.x, indices)
        self.y = np.delete(self.y, indices)
        self.z = np.delete(self.z, indices)
        self.red = np.delete(self.red, indices)
        self.blue = np.delete(self.blue, indices)
        self.green = np.delete(self.green, indices)
        self.pixel_row = np.delete(self.pixel_row, indices)
        self.pixel_column = np.delete(self.pixel_column, indices)
        self.scan_index = np.delete(self.scan_index, indices)

    def add_points(self, x, y, z, red, green, blue, pixel_row=-1, pixel_column=-1, scan_index=-1):
        # x,y,z could be scalars for a single point, or lists for multiple points
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)
        self.z = np.append(self.z, z)
        self.red = np.append(self.red, red)
        self.green = np.append(self.green, green)
        self.blue = np.append(self.blue, blue)
        self.pixel_row = np.append(self.pixel_row, pixel_row)
        self.pixel_column = np.append(self.pixel_column, pixel_column)
        self.scan_index = np.append(self.scan_index, scan_index)

    def add_superpoint(self, x, y, z, red, green, blue, sphere_radius=5, superpoint_samples=1000, pixel_row=-1, pixel_column=-1, scan_index=-1):
        xs, ys, zs = self.fibonacci_sphere(samples=superpoint_samples, sphere_radius = sphere_radius, x_origin = x, y_origin = y, z_origin= z)
        reds = np.full(shape=superpoint_samples, fill_value=red, dtype=self.red.dtype)
        greens = np.full(shape=superpoint_samples, fill_value=green, dtype=self.green.dtype)
        blues = np.full(shape=superpoint_samples, fill_value=blue, dtype=self.blue.dtype)
        pixel_rows = np.full(shape=superpoint_samples, fill_value=-1, dtype=self.pixel_row.dtype)
        pixel_columns = np.full(shape=superpoint_samples, fill_value=-1, dtype=self.pixel_column.dtype)
        scan_indexes = np.full(shape=superpoint_samples, fill_value=-1, dtype=self.scan_index.dtype)

        self.x = np.append(self.x, xs)
        self.y = np.append(self.y, ys)
        self.z = np.append(self.z, zs)
        self.red = np.append(self.red, reds)
        self.green = np.append(self.green, greens)
        self.blue = np.append(self.blue, blues)
        self.pixel_row = np.append(self.pixel_row, pixel_rows)
        self.pixel_column = np.append(self.pixel_column, pixel_columns)
        self.scan_index = np.append(self.scan_index, scan_indexes)

    def fibonacci_sphere(self,samples=1000, sphere_radius = 10, x_origin = 0, y_origin = 0, z_origin= 0):
        xs = np.zeros(samples)
        ys = np.zeros(samples)
        zs = np.zeros(samples)
        phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            theta = phi * i  # golden angle increment
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            x = x_origin + x * sphere_radius
            y = y_origin + y * sphere_radius
            z = z_origin + z * sphere_radius
            xs[i] = x
            ys[i] = y
            zs[i] = z
        return xs, ys, zs

    def downsample(self, reduce_by_1_divided_by_n):
        print("Downsampling {} by {}".format(self.timestamp, reduce_by_1_divided_by_n))
        self.x = self.x[0:-1:reduce_by_1_divided_by_n]
        self.y = self.y[0:-1:reduce_by_1_divided_by_n]
        self.z = self.z[0:-1:reduce_by_1_divided_by_n]
        self.red = self.red[0:-1:reduce_by_1_divided_by_n]
        self.blue = self.blue[0:-1:reduce_by_1_divided_by_n]
        self.green = self.green[0:-1:reduce_by_1_divided_by_n]
        self.pixel_row = self.pixel_row[0:-1:reduce_by_1_divided_by_n]
        self.pixel_column = self.pixel_column[0:-1:reduce_by_1_divided_by_n]
        self.scan_index = self.scan_index[0:-1:reduce_by_1_divided_by_n]

    def filter_nan_values(self):
        nan_indices = np.argwhere(np.isnan(self.x))
        # print("{:,} points with NaN value removed".format(len(nan_indices)))
        self.delete_points(nan_indices)
        print("{:,} points".format(len(self.x)))

    def delete_by_pixel_position(self, dimension="column", start=0, end=10):
        if dimension == "column":
            points_to_delete = np.argwhere(np.logical_and(self.pixel_column >= start, self.pixel_column <= end ))
            print("Deleting {} points with columns between {} and {}".format(len(points_to_delete), start, end))
            self.delete_points(points_to_delete)            
        elif dimension == "row":
            points_to_delete = np.argwhere(np.logical_and(self.pixel_row >= start, self.pixel_row <=  end ))
            print("Deleting {} points with rows between {} and {}".format(len(points_to_delete), start, end))
            self.delete_points(points_to_delete)


    def load(self, ignore_dimensions=False, reindex_into_tensor=False, clip_outer_n_pixels=0, assert_scan_index=None, camera_coordinate_system="camera_coordinate_system"):
        #print("Loading {}".format(self.filename))
        if ".npy" in self.filename:
            with open('/home/sense/3cobot/{}'.format(self.filename), 'rb') as input_file:
                import_tensor = np.load(input_file)
                self.x = import_tensor[:,0]
                self.y = import_tensor[:,1]
                self.z = import_tensor[:,2]
                self.red = import_tensor[:,3]
                self.green = import_tensor[:,4]
                self.blue = import_tensor[:,5]
                self.hdr_red = import_tensor[:,6]
                self.hdr_green = import_tensor[:,7]
                self.hdr_blue = import_tensor[:,8]
                self.pixel_row = import_tensor[:,9]
                self.pixel_column = import_tensor[:,10]
                self.scan_index = import_tensor[:,11]
            print("Loaded {} points from {}".format(len(self.x), self.filename))

        elif ".ply" in self.filename:
            self.data = PlyData.read(self.filename)
            self.x              = self.data.elements[0].data['x']
            self.y              = self.data.elements[0].data['y']
            self.z              = self.data.elements[0].data['z']
            self.red            = self.data.elements[0].data['red']
            self.green          = self.data.elements[0].data['green']
            self.blue           = self.data.elements[0].data['blue']

            if camera_coordinate_system == "camera_coordinate_system":
                #print("Camera camera_coordinate_system")
                self.pixel_row      = pickle.load(open("point_index_to_pixel_row_for_camera_coordinate_system.pkl", "rb"))
                self.pixel_column   = pickle.load(open("point_index_to_pixel_column_for_camera_coordinate_system.pkl", "rb"))           
            else:
                #print("Else")
                self.pixel_row      = pickle.load(open("point_index_to_pixel_row.pkl", "rb"))
                self.pixel_column   = pickle.load(open("point_index_to_pixel_column.pkl", "rb"))
            if type(assert_scan_index) == type(None):
                if "-" in self.filename:
                    # "{}-{}-{}.ply".format(project_name, scan_index, relative_scan_index)
                    self.scan_index = int(self.filename.split("-")[2].split(".")[0])
                    #print("Scan index is {}".format(self.scan_index))
            else:
                self.scan_index = assert_scan_index

            self.scan_index     = np.asarray([self.scan_index] * len(self.x), dtype=np.uint16) # now scan_index exists for every point, in case of merged point clouds from different scans
            self.hdr_red        = np.asarray([-1.0] * len(self.x))
            self.hdr_green        = np.asarray([-1.0] * len(self.x))
            self.hdr_blue        = np.asarray([-1.0] * len(self.x))

        if reindex_into_tensor:
            self.reindex_into_tensor()

        if not ignore_dimensions:
            self.assert_all_indices_aligned()

        if clip_outer_n_pixels > 0:
            self.clip_outer_n_pixels_of_cloud_tensor(n_pixels=clip_outer_n_pixels)

        
        #self.delete_by_pixel_position(dimension="column", start=0, end=2000)
        #self.average_neighboring_point_distance()

    def save_as_ply(self, filename=None, from_tensor=False, communicate=False):
        if type(filename) == type(None):
            filename = self.filename

        if communicate:
            print("\nSaving {}\n".format(filename))

        if not from_tensor:
            n = len(self.x)
            vertices = np.empty(n, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
            vertices['x'] = self.x.astype('f4')
            vertices['y'] = self.y.astype('f4')
            vertices['z'] = self.z.astype('f4')
            vertices['red'] = self.red.astype('u1')
            vertices['green'] = self.green.astype('u1')
            vertices['blue'] = self.blue.astype('u1')
        else:
            number_of_scans = self.cloud_tensor.shape[0]
            width = self.cloud_tensor.shape[1]
            height = self.cloud_tensor.shape[2]
            n = number_of_scans * width * height
            vertices = np.empty(n, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
            xs = np.empty(n, dtype='f4')
            ys = np.empty(n, dtype='f4')
            zs = np.empty(n, dtype='f4')
            reds = np.empty(n, dtype='f4')
            greens = np.empty(n, dtype='f4')
            blues = np.empty(n, dtype='f4')
            for scan_index in range(number_of_scans):
                a = width*height*scan_index
                b = width*height*(scan_index+1)
                xs[a:b] = self.cloud_tensor[scan_index, :, :, 0].flatten()
                ys[a:b]= self.cloud_tensor[scan_index, :, :, 1].flatten()
                zs[a:b] = self.cloud_tensor[scan_index, :, :, 2].flatten()
                reds[a:b] = self.cloud_tensor[scan_index, :, :, 3].flatten()
                greens[a:b] = self.cloud_tensor[scan_index, :, :, 4].flatten()
                blues[a:b] = self.cloud_tensor[scan_index, :, :, 5].flatten()
            vertices['x'] = xs.astype('f4')
            vertices['y'] = ys.astype('f4')
            vertices['z'] = zs.astype('f4')
            vertices['red'] = reds.astype('u1')
            vertices['green'] = greens.astype('u1')
            vertices['blue'] = blues.astype('u1')

        ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=False)
        ply.write(filename)

    def get_points_by_pixel_index(self, index_dimension="pixel_row", index_value=0, pixel_row=None, pixel_column=None):
        if index_dimension == "pixel_row":
            filtered_indices = np.where(self.pixel_row == index_value)
        elif index_dimension == "pixel_column":
            filtered_indices = np.where(self.pixel_column == index_value)
        elif index_dimension == "pixel_row_x_pixel_column":
            filtered_indices = np.where(np.logical_and(self.pixel_row == pixel_row, self.pixel_column == pixel_column ))
        subset = PointCloud()
        subset.x = np.take(self.x, filtered_indices)
        subset.y = np.take(self.y, filtered_indices)
        subset.z = np.take(self.z, filtered_indices)
        subset.red = np.take(self.red, filtered_indices)
        subset.green = np.take(self.green, filtered_indices)
        subset.blue = np.take(self.blue, filtered_indices)
        subset.pixel_row = np.take(self.pixel_row, filtered_indices)
        subset.pixel_column = np.take(self.pixel_column, filtered_indices)
        subset.scan_index = np.take(self.scan_index, filtered_indices)
        return subset

    def get_nearest_neighbor_point(self, dimension="x", value=0.0):
        if dimension == "x":
            closest_index = (np.abs(self.x - value)).argmin()
        if dimension == "y":
            closest_index = (np.abs(self.y - value)).argmin()
        if dimension == "z":
            closest_index = (np.abs(self.z - value)).argmin()

        subset = PointCloud()
        subset.x = np.take(self.x, [closest_index])
        subset.y = np.take(self.y, [closest_index])
        subset.z = np.take(self.z, [closest_index])
        subset.red = np.take(self.red, [closest_index])
        subset.green = np.take(self.green, [closest_index])
        subset.blue = np.take(self.blue, [closest_index])
        subset.pixel_row = np.take(self.pixel_row, [closest_index])
        subset.pixel_column = np.take(self.pixel_column, [closest_index])
        subset.scan_index = np.take(self.scan_index, [closest_index])
        return subset

    def get_closest_point_by_subpixel_row_column(self, row, column):
        closest_index = (np.abs(self.pixel_row - row) + np.abs(self.pixel_column - column)).argmin()
        subset = PointCloud()
        subset.x = np.take(self.x, [closest_index])
        subset.y = np.take(self.y, [closest_index])
        subset.z = np.take(self.z, [closest_index])
        subset.red = np.take(self.red, [closest_index])
        subset.green = np.take(self.green, [closest_index])
        subset.blue = np.take(self.blue, [closest_index])
        subset.pixel_row = np.take(self.pixel_row, [closest_index])
        subset.pixel_column = np.take(self.pixel_column, [closest_index])
        subset.scan_index = np.take(self.scan_index, [closest_index])
        return subset


    def summarize(self, point_index=0):
        x = self.x[point_index]
        y = self.y[point_index]         
        z = self.z[point_index]
        r = self.red[point_index]
        g = self.green[point_index]
        b = self.blue[point_index]
        row = self.pixel_row[point_index]
        col = self.pixel_column[point_index]
        scan_index = self.scan_index[point_index]
        summary = "PIXEL (ROW {}, COL {}): (x,y,z) = ({:.3f},{:.3f},{:.3f}), (r,g,b) = ({},{},{}), scan {}".format(row,col,x,y,z,r,g,b,scan_index)
        return summary

    def preprocess_points_for_plane_finding(self):
        xs = self.x
        ys = self.y
        zs = self.z
        rows = self.pixel_row
        columns = self.pixel_column
        cwd = getcwd()
        current_timestamp = int(time.time() * 100)
        output_filepath = "{}/{}_{}_hvxyz.csv".format(cwd, current_timestamp, self.timestamp)
        with open(output_filepath, "w") as output_file: # {timestamp}_hvxyz.csv
            for x,y,z,row,column in zip(xs,ys,zs,rows,columns):
                output_file.write("{},{},{:.3f},{:.3f},{:.3f}\n".format(int(row),int(column),x,y,z))
        return output_filepath

    def export_to_csv(self, project_name, from_tensor=False):
        output_filepath = "{}.csv".format(project_name)
        #print("Exporting to {}".format(output_filepath))
        if not from_tensor:
            xs = self.x
            ys = self.y
            zs = self.z
            reds = self.red
            greens = self.green
            blues = self.blue
            rows = self.pixel_row
            columns = self.pixel_column
            scan_index = self.scan_index
            with open(output_filepath, "w") as output_file: # {timestamp}_hvxyz.csv
                for x,y,z,r,g,b,row,column,i in zip(xs,ys,zs,reds,greens,blues,rows,columns,scan_index):
                    output_file.write("{},{},{:.3f},{:.3f},{:.3f},{},{},{},{}\n".format(int(row),int(column),x,y,z,r,g,b,int(i)))
        return output_filepath

    def reindex_tensor_back_into_core(self):
        number_of_clouds = self.cloud_tensor.shape[0]
        number_of_rows = self.cloud_tensor.shape[1]
        number_of_columns = self.cloud_tensor.shape[2]

        self.x = []
        self.y = []
        self.z = []
        self.red = []
        self.green = []
        self.blue = []
        self.pixel_row = []
        self.pixel_column = []
        self.scan_index = []
        self.hdr_red = []
        self.hdr_green = []
        self.hdr_blue = []

        for cloud_index in range(number_of_clouds):
            cloud = self.cloud_tensor[cloud_index,:,:,:]
            clean_tensor = cloud[~np.isnan(cloud[:,:,0:3]).any(axis=2)] # clean out the NaNs
            next_x = clean_tensor[:,0]
            next_y = clean_tensor[:,1]                  
            next_z = clean_tensor[:,2]
            next_red = clean_tensor[:,3]
            next_green = clean_tensor[:,4]
            next_blue = clean_tensor[:,5]
            next_hdr_red = clean_tensor[:,6]
            next_hdr_green = clean_tensor[:,7]
            next_hdr_blue = clean_tensor[:,8]
            next_pixel_row = clean_tensor[:,9]
            next_pixel_column = clean_tensor[:,10]
            next_scan_index = clean_tensor[:,11]

            self.x.extend(next_x)
            self.y.extend(next_y)
            self.z.extend(next_z)               
            self.red.extend(next_red)
            self.green.extend(next_green)
            self.blue.extend(next_blue)
            self.hdr_red.extend(next_hdr_red)                   
            self.hdr_green.extend(next_hdr_green)
            self.hdr_blue.extend(next_hdr_blue)
            self.scan_index.extend(next_scan_index)
            self.pixel_row.extend(next_pixel_row)
            self.pixel_column.extend(next_pixel_column)

        self.x = np.asarray(self.x)
        self.y = np.asarray(self.y)
        self.z = np.asarray(self.z)
        self.red = np.asarray(self.red)
        self.green = np.asarray(self.green)
        self.blue = np.asarray(self.blue)
        self.pixel_row = np.asarray(self.pixel_row)
        self.pixel_column = np.asarray(self.pixel_column)
        self.scan_index = np.asarray(self.scan_index)
        self.hdr_red = np.asarray(self.hdr_red)
        self.hdr_green = np.asarray(self.hdr_green)
        self.hdr_blue = np.asarray(self.hdr_blue)        

    def export_to_npy(self, project_name, from_tensor=False):
        output_filepath = "{}.npy".format(project_name)
        #print("Exporting to {}".format(output_filepath))
        if from_tensor:
            number_of_clouds = self.cloud_tensor.shape[0]
            number_of_rows = self.cloud_tensor.shape[1]
            number_of_columns = self.cloud_tensor.shape[2]
            #print("Exporting {} clouds with {}x{} dimensions".format(number_of_clouds, number_of_rows, number_of_columns))

            #print("Prior to export, {} x size".format(len(self.x)))
            self.reindex_tensor_back_into_core()
            #print("Exporting {} x size".format(len(self.x)))

            export_tuple = (self.x, self.y, self.z, self.red, self.green, self.blue, self.hdr_red, self.hdr_green, self.hdr_blue, self.pixel_row, self.pixel_column, self.scan_index)
            export_tensor = np.stack(export_tuple, axis=1)
            export_file_name = "{}.npy".format(project_name)
            #print("Reindex tensor and now saving all data in {}".format(export_file_name))
            with open(export_file_name, 'wb') as output_file:
                np.save(output_file, export_tensor)

        return output_filepath

    def delete_ply_file(self):
        cwd = getcwd()
        output_filepath = "{}/{}.ply".format(cwd, self.filename)
        try:
            remove(output_filepath)
        except:
            pass

    def copy(self):
        new_point_cloud = PointCloud()
        new_point_cloud.x = self.x.copy()
        new_point_cloud.y = self.y.copy()
        new_point_cloud.z = self.z.copy()
        new_point_cloud.red = self.red.copy()
        new_point_cloud.green = self.green.copy()
        new_point_cloud.blue = self.blue.copy()
        new_point_cloud.pixel_row = self.pixel_row.copy()
        new_point_cloud.pixel_column = self.pixel_column.copy()
        new_point_cloud.scan_index = self.scan_index.copy()
        new_point_cloud.scanner_output = None
        new_point_cloud.timestamp = self.timestamp
        new_point_cloud.filename = self.filename
        return new_point_cloud

    def get_average_z_distance(self, downsample_ratio=50):
        sample = self.copy()
        sample.downsample(downsample_ratio)
        average_distance = np.average(sample.z)
        print("Computed average distance of {}mm for points in FOV from a {}x downsampling".format(average_distance, downsample_ratio))
        return average_distance

    def average_neighboring_point_distance(self, number_of_samples=1000):
        start_time = time.time()
        total_distances = np.zeros(1000)
        total_points_in_cloud = len(self.x)
        total_number_of_index_look_ups = 0
        for sample_number in range(number_of_samples):
            neighboring_points_found = False
            while not neighboring_points_found:
                total_number_of_index_look_ups += 1
                random_index = random.randint(1, total_points_in_cloud-1)
                sample_row = self.pixel_row[random_index]
                sample_column = self.pixel_column[random_index]
                sample_minus_1_row = self.pixel_row[random_index-1]
                sample_minus_1_column = self.pixel_column[random_index-1]
                sample_plus_1_row = self.pixel_row[random_index+1]
                sample_plus_1_column = self.pixel_column[random_index+1]
                if abs(sample_row - sample_minus_1_row) <= 1 and abs(sample_column - sample_minus_1_column) <= 1:
                    neighboring_points_found = True
                    distance = math.sqrt( (self.x[random_index] - self.x[random_index-1])**2 + (self.y[random_index] - self.y[random_index-1])**2  + (self.z[random_index] - self.z[random_index-1])**2) 
                    total_distances[sample_number] = distance

        min_point_to_point_distance = np.min(total_distances)
        max_point_to_point_distance = np.max(total_distances)
        avg_point_to_point_distance = np.average(total_distances)
        end_time = time.time()

        print("After {} look-ups in {} seconds, found {} examples of neighboring point distances: min = {}, max = {}, average = {}".format(total_number_of_index_look_ups, end_time - start_time, number_of_samples, min_point_to_point_distance, max_point_to_point_distance, avg_point_to_point_distance))

        self.min_point_to_point_distance = min_point_to_point_distance
        self.max_point_to_point_distance = max_point_to_point_distance
        self.avg_point_to_point_distance = avg_point_to_point_distance

        return min_point_to_point_distance, max_point_to_point_distance, avg_point_to_point_distance

    def filter_roi_pixels(self):
        min_pixel_row = int(self.camera_image_height/2) - int(camera_roi_height/2)
        max_pixel_row = int(self.camera_image_height/2) + int(camera_roi_height/2)
        min_pixel_column = int(self.camera_image_width/2) - int(camera_roi_width/2)
        max_pixel_column = int(self.camera_image_width/2) + int(camera_roi_width/2)

        # first delete rows
        indices_to_delete = np.where(np.logical_or(self.pixel_row < min_pixel_row, self.pixel_row > max_pixel_row))[0]
        print("{:,} points from pixel rows outside of ROI to delete".format(len(indices_to_delete)))
        self.delete_points(indices_to_delete)

        # then delete columns
        indices_to_delete = np.where(np.logical_or(self.pixel_column < min_pixel_column, self.pixel_column > max_pixel_column))[0]
        print("{:,} points from pixel columns outside of ROI to delete".format(len(indices_to_delete)))
        self.delete_points(indices_to_delete)

        remaining_points_in_roi = len(self.x)
        print("{:,} points remaining in ROI after filtering".format(remaining_points_in_roi))

        return remaining_points_in_roi

    def get_average_3d_distance_from(self, x=0.0, y=0.0, z=0.0):
        average_distance = np.average(np.sqrt((self.x - x) ** 2 + (self.y - y)**2 + (self.z - z)**2 ))
        print("Average distance from all points to ({:.3f},{:.3},{:.3f}) = {:.6f} mm".format(x,y,z,average_distance))
        return average_distance

    def add_point_cloud(self, other):
        self.x = np.concatenate((self.x, other.x), axis=0)
        self.y = np.concatenate((self.y, other.y), axis=0)
        self.z = np.concatenate((self.z, other.z), axis=0)
        self.red = np.concatenate((self.red, other.red), axis=0)
        self.green = np.concatenate((self.green, other.green), axis=0)
        self.blue = np.concatenate((self.blue, other.blue), axis=0)
        self.pixel_row = np.concatenate((self.pixel_row, other.pixel_row), axis=0)
        self.pixel_column = np.concatenate((self.pixel_column, other.pixel_column), axis=0)
        self.hdr_red = np.concatenate((self.hdr_red, other.hdr_red), axis=0)
        self.hdr_green = np.concatenate((self.hdr_green, other.hdr_green), axis=0)
        self.hdr_blue = np.concatenate((self.hdr_blue, other.hdr_blue), axis=0)
        self.scan_index = np.concatenate((self.scan_index, other.scan_index), axis=0)

    def reindex_into_tensor(self):
        #print("Reindexing the point cloud into a tensor")
        unique_cloud_indices = np.unique(self.scan_index) # check for largest scan index, and technically create empty values in case there are scans in between missing
        number_of_clouds = len(unique_cloud_indices)
        height = self.camera_image_width
        width = self.camera_image_height
        self.cloud_tensor = np.empty((int(number_of_clouds), int(width), int(height), 12)) * np.nan # 6 = len([x,y,z,r,g,b])

        self.scan_index = self.scan_index.astype(int)
        self.pixel_row = self.pixel_row.astype(int)
        self.pixel_column = self.pixel_column.astype(int)

        for scan_index in unique_cloud_indices:
            scan_index = int(scan_index)
            indices_for_this_scan = np.argwhere(self.scan_index == scan_index).flatten()
            #print("{} points for scan {}".format(len(indices_for_this_scan), scan_index))
            #print("e.g. {}, {}, {}, {}, {}".format(indices_for_this_scan[0], indices_for_this_scan[1], indices_for_this_scan[2], indices_for_this_scan[3], indices_for_this_scan[4]))

            x_values_for_this_scan = self.x[indices_for_this_scan]
            y_values_for_this_scan = self.y[indices_for_this_scan]
            z_values_for_this_scan = self.z[indices_for_this_scan]
            r_values_for_this_scan = self.red[indices_for_this_scan]
            g_values_for_this_scan = self.green[indices_for_this_scan]
            b_values_for_this_scan = self.blue[indices_for_this_scan]
            hdr_r_values_for_this_scan = self.hdr_red[indices_for_this_scan]
            hdr_g_values_for_this_scan = self.hdr_green[indices_for_this_scan]
            hdr_b_values_for_this_scan = self.hdr_blue[indices_for_this_scan]
            pixel_row_for_scan = self.pixel_row[indices_for_this_scan]
            pixel_column_for_scan = self.pixel_column[indices_for_this_scan]
            scan_indices_for_scan = self.scan_index[indices_for_this_scan]


            if int(number_of_clouds) == 1:
                scan_index_for_scan = 0
            else:
                scan_index_for_scan = scan_index

            #print("Relevant scan indexes: {} e.g. {}, {}, {}".format(len(scan_index_for_scan), scan_index_for_scan[0], scan_index_for_scan[1], scan_index_for_scan[2]))
            #print("Relevant pixel rows: {} e.g. {}, {}, {}".format(len(pixel_row_for_scan), pixel_row_for_scan[0], pixel_row_for_scan[1], pixel_row_for_scan[2]))
            #print("Relevant pixel columns: {} e.g. {}, {}, {}".format(len(pixel_column_for_scan), pixel_column_for_scan[0], pixel_column_for_scan[1], pixel_column_for_scan[2]))

            self.cloud_tensor[scan_index_for_scan, pixel_row_for_scan, pixel_column_for_scan, 0] = x_values_for_this_scan

            #print("{} X values being added, e.g. {}, {}, {}, {}".format(len(x_values_for_this_scan), x_values_for_this_scan[0], x_values_for_this_scan[1], x_values_for_this_scan[2], x_values_for_this_scan[3]))
            # print("From what's in the self.cloud_tensor: {}, {}, {}, {}".format(self.cloud_tensor[scan_index_for_scan, pixel_row_for_scan[0], pixel_column_for_scan[0], 0],
            #                                                                     self.cloud_tensor[scan_index_for_scan, pixel_row_for_scan[1], pixel_column_for_scan[1], 0],
            #                                                                     self.cloud_tensor[scan_index_for_scan, pixel_row_for_scan[2], pixel_column_for_scan[2], 0],
            #                                                                     self.cloud_tensor[scan_index_for_scan, pixel_row_for_scan[3], pixel_column_for_scan[3], 0]
            #     ))


            self.cloud_tensor[scan_index_for_scan, pixel_row_for_scan, pixel_column_for_scan, 1] = y_values_for_this_scan
            self.cloud_tensor[scan_index_for_scan, pixel_row_for_scan, pixel_column_for_scan, 2] = z_values_for_this_scan
            self.cloud_tensor[scan_index_for_scan, pixel_row_for_scan, pixel_column_for_scan, 3] = r_values_for_this_scan

            #print("{} Red values being added, e.g. {}, {}, {}, {}".format(len(self.red[indices_for_this_scan]), self.red[indices_for_this_scan][0], self.red[indices_for_this_scan][1], self.red[indices_for_this_scan][2], self.red[indices_for_this_scan][3]))


            self.cloud_tensor[scan_index_for_scan, pixel_row_for_scan, pixel_column_for_scan, 4] = g_values_for_this_scan
            self.cloud_tensor[scan_index_for_scan, pixel_row_for_scan, pixel_column_for_scan, 5] = b_values_for_this_scan
            self.cloud_tensor[scan_index_for_scan, pixel_row_for_scan, pixel_column_for_scan, 6] = hdr_r_values_for_this_scan
            self.cloud_tensor[scan_index_for_scan, pixel_row_for_scan, pixel_column_for_scan, 7] = hdr_g_values_for_this_scan
            self.cloud_tensor[scan_index_for_scan, pixel_row_for_scan, pixel_column_for_scan, 8] = hdr_b_values_for_this_scan
            self.cloud_tensor[scan_index_for_scan, pixel_row_for_scan, pixel_column_for_scan, 9] = pixel_row_for_scan
            self.cloud_tensor[scan_index_for_scan, pixel_row_for_scan, pixel_column_for_scan, 10] = pixel_column_for_scan
            self.cloud_tensor[scan_index_for_scan, pixel_row_for_scan, pixel_column_for_scan, 11] = scan_indices_for_scan

        #print("Reindexed {} into {} tensor".format(self.filename, self.cloud_tensor.shape))
        #indices_with_non_nan = np.argwhere(self.cloud_tensor != np.nan)
        #print("Non nan indices: {}".format(len(indices_with_non_nan)))


        n_sample_values_to_print = 10
        max_scan_index = int(self.cloud_tensor.shape[0]) - 1
        max_row = int(self.cloud_tensor.shape[1]) - 1
        max_column = int(self.cloud_tensor.shape[2]) - 1
        #print("Here are {} sample values from the tensor:".format(n_sample_values_to_print))
        for n in range(n_sample_values_to_print):
            random_scan = random.randint(0, max_scan_index)
            random_row = random.randint(0, max_row)
            random_column = random.randint(0, max_column)
            x = self.cloud_tensor[random_scan, random_row, random_column, 0]
            y = self.cloud_tensor[random_scan, random_row, random_column, 1]
            z = self.cloud_tensor[random_scan, random_row, random_column, 2]
            r = self.cloud_tensor[random_scan, random_row, random_column, 3]
            g = self.cloud_tensor[random_scan, random_row, random_column, 4]
            b = self.cloud_tensor[random_scan, random_row, random_column, 5]
            #print("({}) Scan {}, Row {}, Column {}, X={:.2f}, Y={:.2f}, Z={:.2f}, R={}, G={}, B={}".format(n, random_scan, random_row, random_column, x, y, z, r, g, b))

        total_number_of_points = len(self.x)
        #print("Here are {} samples from the original core:".format(n_sample_values_to_print))
        for n in range(n_sample_values_to_print):
            random_point = random.randint(0, total_number_of_points)
            x = self.x[random_point]
            y = self.y[random_point]
            z = self.z[random_point]
            r = self.red[random_point]
            g = self.green[random_point]
            b = self.blue[random_point]
            #print("({}) Point {}, X={:.2f}, Y={:.2f}, Z={:.2f}, R={}, G={}, B={}".format(n, random_point, x, y, z, r, g, b))


    def average_xyz_positions_with_outlier_removal(self, ignore_deviations_below=0.01, maximum_deviation=0.075, max_standard_deviations=2.0, new_scan_index=0):
        # TO DO: Make the geometric average probablistic based on (a) known focus distance, (b) focus masks, (c) surrounding pixels

        x = self.cloud_tensor[:, :, :, 0]
        y = self.cloud_tensor[:, :, :, 1]
        z = self.cloud_tensor[:, :, :, 2]

        #print("Shape of x: {}".format(x.shape))
        #print("Example x: {}".format(x[:,750,750]))

        # first get an average, including outliers
        avg_x = np.average(x, axis=0)
        avg_y = np.average(y, axis=0)
        avg_z = np.average(z, axis=0)

        #print("Shape of avg_x: {}".format(avg_x.shape))
        #print("Example of avg_x: {}".format(avg_x[750,750]))

        # measure the difference from the average for all points
        diff_x = np.abs(avg_x - x)
        diff_y = np.abs(avg_y - y)
        diff_z = np.abs(avg_z - z)

        #print("Shape of diff_x: {}".format(diff_x.shape))
        #print("Example of diff_x: {}".format(diff_x[:,750,750]))

        # also get the standard deviation
        std_dev_x = np.std(x, axis=0)
        std_dev_y = np.std(y, axis=0)
        std_dev_z = np.std(z, axis=0)

        #print("Shape of std_dev_x: {}".format(std_dev_x.shape))
        #print("Example of std_dev_x: {}".format(std_dev_x[750,750]))

        # remove points with a difference beyond max_standard_deviations
        x_std_dev_mask = np.ones(x.shape)
        y_std_dev_mask = np.ones(y.shape)
        z_std_dev_mask = np.ones(z.shape)

        #print("Shape of x_std_dev_mask: {}".format(x_std_dev_mask.shape))
        #print("Example of x_std_dev_mask: {}".format(x_std_dev_mask[:,750,750]))

        x_std_dev_mask[diff_x > std_dev_x * max_standard_deviations] = np.nan
        y_std_dev_mask[diff_y > std_dev_y * max_standard_deviations] = np.nan
        z_std_dev_mask[diff_z > std_dev_z * max_standard_deviations] = np.nan

        #print("Shape of x_std_dev_mask: {}".format(x_std_dev_mask.shape))
        #print("Example of x_std_dev_mask: {}".format(x_std_dev_mask[:,750,750]))      

        nan_from_standard_dev = np.count_nonzero(np.isnan(x_std_dev_mask))
        #print("NaN from standard deviation removal: {}".format(nan_from_standard_dev))

        # in case of very noisy data, also have a hard limit on maximum deviation, and remove those points
        x_max_deviation_mask = np.ones(x.shape)
        y_max_deviation_mask = np.ones(y.shape)
        z_max_deviation_mask = np.ones(z.shape)

        #print("Shape of x_max_deviation_mask: {}".format(x_max_deviation_mask.shape))
        #print("Example of x_max_deviation_mask: {}".format(x_max_deviation_mask[:,750,750]))      

        x_max_deviation_mask[diff_x > maximum_deviation] = np.nan
        y_max_deviation_mask[diff_y > maximum_deviation] = np.nan
        z_max_deviation_mask[diff_z > maximum_deviation] = np.nan

        #print("Shape of x_max_deviation_mask: {}".format(x_max_deviation_mask.shape))
        #print("Example of x_max_deviation_mask after change: {}".format(x_max_deviation_mask[:,750,750]))  

        nan_from_max_dev = np.count_nonzero(np.isnan(x_max_deviation_mask))
        #print("NaN from max deviation removal: {}".format(nan_from_max_dev))

        new_x = x * x_std_dev_mask
        new_y = y * y_std_dev_mask
        new_z = z * z_std_dev_mask

        #print("Shape of new_x after standard deviation removal: {}".format(new_x.shape))
        #print("Example of new_x after standard deviation removal: {}".format(new_x[:,750,750]))  

        new_x = x * x_max_deviation_mask
        new_y = y * y_max_deviation_mask
        new_z = z * z_max_deviation_mask

        #print("Shape of new_x after max deviation removal: {}".format(new_x.shape))
        #print("Example of new_x after max deviation removal: {}".format(new_x[:,750,750]))  

        new_avg_x = np.nanmean(new_x, axis=0)
        new_avg_y = np.nanmean(new_y, axis=0)
        new_avg_z = np.nanmean(new_z, axis=0)

        #print("Shape of new_avg_x: {}".format(new_avg_x.shape))
        #print("Example of new_avg_x: {}".format(new_avg_x[750,750]))  

        new_cloud_tensor = self.cloud_tensor[0, :, :, :]
        new_cloud_tensor[:,:,0] = new_avg_x
        new_cloud_tensor[:,:,1] = new_avg_y
        new_cloud_tensor[:,:,2] = new_avg_z

        new_cloud_tensor = np.expand_dims(new_cloud_tensor, axis=0)

        number_of_rows = self.cloud_tensor.shape[1]
        number_of_columns = self.cloud_tensor.shape[2]

        new_cloud_tensor[0,:,:,11] = np.full(shape=(number_of_rows,number_of_columns), fill_value=new_scan_index, dtype=np.uint16)

        #print("Shape of new_cloud_tensor: {}".format(new_cloud_tensor.shape))
        #print("Example of new_cloud_tensor: {}".format(new_cloud_tensor[0,750,750]))  

        self.cloud_tensor = new_cloud_tensor

        #print("Shape of cloud_tensor: {}".format(self.cloud_tensor.shape))
        #print("Example of cloud_tensor: {}".format(self.cloud_tensor[0,750,750])) 

        self.reindex_tensor_back_into_core()

    def clip_outer_n_pixels_of_cloud_tensor(self, n_pixels=10):
        #print("Clipping outer {} pixels".format(n_pixels))
        def remove_edge_non_nan(first_or_last="first", row_or_column="row", n_pixels=10):
            for scan_index in np.unique(self.cloud_tensor, axis=0):
                print("This scan index is {}".format(scan_index))

                x = self.cloud_tensor[scan_index, :, :, 0]
                where_x_is_non_nan = ~np.isnan(x)

                if row_or_column == "row":
                    axis = 0
                    axis_size = where_x_is_non_nan.shape[axis]
                elif row_or_column == "column":
                    axis = 1
                    axis_size = where_x_is_non_nan.shape[axis]

                if first_or_last == "first":
                    edge_index = where_x_is_non_nan.argmax(axis=axis)
                elif first_or_last == "last":
                    edge_index = x.shape[axis] - np.flip(where_x_is_non_nan, axis=axis).argmax(axis=axis) - 1

                non_nan_edge = np.where(where_x_is_non_nan.any(axis=axis), edge_index, np.nan)
                non_nan_edge = non_nan_edge.astype(np.int16)
                non_nan_edge_expanded = np.expand_dims(non_nan_edge, axis=0) # axis

                # if row_or_column == "row":
                target_shape = (n_pixels, axis_size)
                # elif row_or_column == "column":
                #     #target_shape = (axis_size, n_pixels)
                #     target_shape = (n_pixels, axis_size)
                
                broadcasted_non_nan_edge = np.broadcast_to(non_nan_edge_expanded, target_shape)
                other_dimension_index = np.arange(x.shape[0]) # axis
                other_dimension_index_expanded = np.expand_dims(other_dimension_index, axis=0) # axis

                if first_or_last == "first":
                    following_n_indices = np.arange(0,n_pixels,1).astype(np.int16)
                elif first_or_last == "last":
                    following_n_indices = np.arange(0,int(-1*n_pixels),-1).astype(np.int16)

                # if row_or_column == "row":
                expanded_following_indices = np.expand_dims(following_n_indices, axis=1)
                broadcasted_other_dimension = np.broadcast_to(other_dimension_index_expanded, (n_pixels, x.shape[1]))
                # elif row_or_column == "column":
                #     expanded_following_indices = np.expand_dims(following_n_indices, axis=0) #
                #     broadcasted_other_dimension = np.broadcast_to(other_dimension_index_expanded, (x.shape[0], n_pixels))

                edge_indexes = broadcasted_non_nan_edge + expanded_following_indices
                flattened_edge_indexes = edge_indexes.flatten()
                broadcasted_other_dimension = broadcasted_other_dimension.flatten()

                if row_or_column == "row":
                    self.cloud_tensor[scan_index, flattened_edge_indexes, broadcasted_other_dimension, :] = np.nan
                elif row_or_column == "column":
                    self.cloud_tensor[scan_index, broadcasted_other_dimension, flattened_edge_indexes, :] = np.nan
        
        remove_edge_non_nan(first_or_last="first", row_or_column="row", n_pixels=n_pixels)
        remove_edge_non_nan(first_or_last="first", row_or_column="column", n_pixels=n_pixels)
        remove_edge_non_nan(first_or_last="last", row_or_column="row", n_pixels=n_pixels)
        remove_edge_non_nan(first_or_last="last", row_or_column="column", n_pixels=n_pixels)

        self.reindex_tensor_back_into_core()


    def set_colors_by_photo(self, photo_filename):
        photo = Image.open(photo_filename)
        photo = photo.rotate(90)
        photo_array = np.asarray(photo)
        
        reds = photo_array[:,:,0]
        greens = photo_array[:,:,1]
        blues = photo_array[:,:,2]

        number_of_clouds = self.cloud_tensor.shape[0]
        for cloud_index in range(number_of_clouds):
            self.cloud_tensor[cloud_index,:,:,3] = reds
            self.cloud_tensor[cloud_index,:,:,4] = greens
            self.cloud_tensor[cloud_index,:,:,5] = blues

        self.reindex_tensor_back_into_core()

    def set_hdr_colors_by_exr_data(self, hdr_data):
        print("Setting HDR colors of tensor data")
        # following previous function
        hdr_data = np.rot90(hdr_data)

        hdr_reds = hdr_data[:,:,0]
        hdr_greens = hdr_data[:,:,1]
        hdr_blues = hdr_data[:,:,2]

        number_of_clouds = self.cloud_tensor.shape[0]
        number_of_rows = self.cloud_tensor.shape[1]
        number_of_columns = self.cloud_tensor.shape[2]

        dummy_image_tensor = np.ones((number_of_rows, number_of_columns))
        row_column_indices = np.argwhere(dummy_image_tensor)

        row = [i for i in range(self.camera_image_height)]
        column = [i for i in range(self.camera_image_width)]
        row_indices = np.array([row,]*self.camera_image_width).transpose() # np.repeat(a=range(0,self.camera_image_height),repeats=self.camera_image_width,axis=0)
        column_indices = np.array([column,]*self.camera_image_height).transpose() #np.repeat(a=range(0,self.camera_image_width),repeats=self.camera_image_height,axis=0)

        hdr_data_tensor = np.zeros((number_of_clouds, number_of_rows, number_of_columns, 6))

        for cloud_index in range(number_of_clouds):
            hdr_data_tensor[cloud_index,:,:,0] = hdr_reds
            hdr_data_tensor[cloud_index,:,:,1] = hdr_greens
            hdr_data_tensor[cloud_index,:,:,2] = hdr_blues
            hdr_data_tensor[cloud_index,:,:,3] = row_indices
            hdr_data_tensor[cloud_index,:,:,4] = column_indices
            hdr_data_tensor[cloud_index,:,:,5] = np.full(shape=(number_of_rows,number_of_columns), fill_value=cloud_index, dtype=np.uint8)

        self.cloud_tensor[:,:,:,6:12] = hdr_data_tensor

        self.reindex_tensor_back_into_core()
        #self.cloud_tensor = np.concatenate([self.cloud_tensor, hdr_data_tensor], axis=3)

    def transform(self, rotation_matrix, translation_matrix):
        points = np.matrix([self.x, self.y, self.z])
        points = np.transpose(points)

        print("{} points to transform...".format(points.shape))

        if type(rotation_matrix) != type(points) and type(translation_matrix) != type(points):
            rotation_matrix = np.asmatrix(rotation_matrix)
            translation_matrix = np.asmatrix(translation_matrix)

        scale_factor = 1.0000 
        transformed_points = scale_factor * np.dot(points, rotation_matrix) + translation_matrix
        #print("\n6D transformation:\nRotation: {}\nTranslation: {}\ne.g. from {} to {}".format(rotation_matrix, translation_matrix, points[0,:], transformed_points[0,:]))
        self.x = transformed_points[:,0]
        self.y = transformed_points[:,1]
        self.z = transformed_points[:,2]
        self.x = np.asarray(self.x).flatten()
        self.y = np.asarray(self.y).flatten()
        self.z = np.asarray(self.z).flatten()

def analyze(label, np_array):
    geo_shape = np_array.shape
    min_value = np.min(np_array)
    max_value = np.max(np_array)
    average_value = np.average(np_array)
    std_value = np.std(np_array)
    print("{} has shape {} with min={:.4f}, max={:.4f}, average={:.4f}, standard deviation={:.4f}".format(label, geo_shape, min_value, max_value, average_value, std_value))
    return geo_shape, min_value, max_value, average_value, std_value