import numpy as np
import random
import ray
import time
import logging
import sys
from scipy.optimize import minimize
import multiprocessing
from PIL import Image
import math
from itertools import zip_longest, combinations
from os import listdir, path, getcwd, remove
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from itertools import zip_longest
import pickle
from plyfile import PlyData, PlyElement
from color_sampler import sample_n_colors_uniformly_from_human_perceptual_space

class Point():
    def __init__(self, x, y, z, row=None, column=None):
        self.x = x
        self.y = y
        self.z = z
        self.xyz = [self.x, self.y, self.z]
        self.h = row # let h = row
        self.v = column # let v = column
        self.validate()
        
    def validate(self):
        if self.x == 0.0 and self.y == 0.0 and self.z == 0.0:
            self.valid = False
        else:
            self.valid = True
            
    def distance(self, point):
        return math.sqrt((self.x - point.x)**2 + (self.y - point.y)**2 + (self.z - point.z)**2)
    
class Triangle():
    def __init__(self, points):
        self.p1 = points[0]
        self.p2 = points[1]
        self.p3 = points[2]
        self.validate()
    
    def validate(self):
        if self.p1.valid and self.p2.valid and self.p3.valid:
            self.valid = True
        else:
            self.valid = False
        
class Plane():
    def __init__(self, points = [], a=None, b=None, c=None, d=None):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.points_h = []
        self.points_v = []
        self.points = np.zeros((len(points), 3))
        for i, point in enumerate(points):
            self.points[i][0] = point.x
            self.points[i][1] = point.y
            self.points[i][2] = point.z
            self.points_h.append(point.h)
            self.points_v.append(point.v)
        if len(self.points) > 2:
            self.estimate_initial_parameters_with_first_three_points()


    def point_exists(self, pixel_row, pixel_column):
        if np.logical_and(pixel_row in self.points_h, pixel_column in self.points_v):
            return True
        else:
            return False

    def normal_vector(self):
        normalized_magnitude = math.sqrt(self.a*self.a + self.b*self.b + self.c*self.c)
        return [self.a/normalized_magnitude, self.b/normalized_magnitude, self.c/normalized_magnitude]

    def find_closest_point_on_plane(self,x_1,y_1,z_1):
        # a = self.a
        # b = self.b
        # c = self.c
        # d = self.d

        # x_intercept = -self.d / self.a
        # y_intercept = -self.d / self.b
        # z_intercept = -self.d / self.c

        initial_guess = [x_1 + 10.0, y_1 + 10.0, z_1 + 10.0]
        
        def model(params): # abcd
            #a, b, c, d = abcd
            x_2, y_2, z_2 = params
            distance_from_original_point = math.sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2 + (z_2 - z_1)**2)
            deviation_from_plane = self.a * x_2 + self.b * y_2 + self.c * z_2 + self.d
            objective_error = distance_from_original_point + abs(deviation_from_plane)*100
            print("Error = ({:.3f}): {:.3f}mm from original point ({},{},{}), {:.3f} deviation from plane".format(objective_error, distance_from_original_point, x_1, y_1, z_1, deviation_from_plane))
            return objective_error

        # def point_is_on_plane(params):
        #     x_2, y_2, z_2 = params
        #     a,b,c,d = abcd
        #     return a*x_2 + b*y_2 + c*z_2 + d

        # constrain the vector perpendicular to the plane be of unit length
        #cons = ({'type': 'eq', 'fun': point_is_on_plane})
        sol = minimize(model, x0=initial_guess) # constraints=cons args=[self.a, self.b, self.c, self.d]
        closest_x = tuple(sol.x)[0] 
        closest_y = tuple(sol.x)[1]
        closest_z = tuple(sol.x)[2]
        print("FINAL ({},{},{}) from {}".format(closest_x, closest_y, closest_z, sol))
        return closest_x, closest_y, closest_z
    

    def estimate_initial_parameters_with_first_three_points(self):
        # a*x + b*y + c*z = d
        p1 = self.points[0]
        p2 = self.points[1]
        p3 = self.points[2]
        a1 = p2[0] - p1[0]
        b1 = p2[1] - p1[1]
        c1 = p2[2] - p1[2]
        a2 = p3[0] - p1[0]
        b2 = p3[1] - p1[1] 
        c2 = p3[2] - p1[2]
        self.a = b1 * c2 - b2 * c1 
        self.b = a2 * c1 - a1 * c2 
        self.c = a1 * b2 - b1 * a2 
        self.d = (- self.a * p1[0] - self.b * p1[1] - self.c * p1[2]) 

        if self.d < 0:
            self.a = self.a * -1
            self.b = self.b * -1
            self.c = self.c * -1
            self.d = self.d * -1
        
    def add_points_to_plane(self, new_points):
        new_xyz = [p.xyz for p in new_points]
        self.points = np.vstack((self.points, new_xyz))
        
        new_h = [p.h for p in new_points]
        self.points_h.extend(new_h)

        new_v = [p.v for p in new_points]
        self.points_v.extend(new_v)
        
    def remove_points_by_indices(self, indices_of_points_to_remove):
         self.points = np.delete(self.points, indices_of_points_to_remove, 0)
         self.points_h = np.delete(self.points_h, indices_of_points_to_remove, 0).tolist()
         self.points_v = np.delete(self.points_v, indices_of_points_to_remove, 0).tolist()

    def get_planarity(self, point):  
        planarity = abs( self.a * point[0] + self.b * point[1] + self.c * point[2] + self.d ) / (self.a**2 + self.b**2 + self.c**2)
        return planarity
    
    def optimize_plane_perpendicular_distance_to_points(self):
        initial_error = self.perpendicular_error()
        
        x, y, z = self.points[:, 0], self.points[:, 1], self.points[:, 2]
        initial_guess = [self.a, self.b, self.c, self.d] # use previous plane equation as initial guess
        
        def model(params, xyz):
            a, b, c, d = params
            x, y, z = xyz
            length_squared = a**2 + b**2 + c**2
            if length_squared > 0:
                return ((a * x + b * y + c * z + d) ** 2 / length_squared).sum() 
            else:
                return 1000.0

        def unit_length(params):
            a, b, c, d = params
            return a**2 + b**2 + c**2 - 1

        # constrain the vector perpendicular to the plane be of unit length
        cons = ({'type': 'eq', 'fun': unit_length})
        sol = minimize(model, initial_guess, args=[x, y, z], constraints=cons)
        self.a = tuple(sol.x)[0]
        self.b = tuple(sol.x)[1]
        self.c = tuple(sol.x)[2]
        self.d = tuple(sol.x)[3]
        return tuple(sol.x)
    

    def find_point_position_on_plane(self, x,y,z, camera_extrinsics_matrix=None):
        # we model the "error correction" for projecting the raw measured 3D point onto the statistically robust 3D plane equation,
        # on top of a camera pinhole model, where the light comes from a point on the plane, through the measured 3D point, and hits the focal point
        # in this way, we assume that there is no error in the x-y / u-v camera space, and the error is confined to the depth estimation
        # so what we are doing is "fixing" the depth, using the vector between the camera focal point and the point that is projected onto the plane through the measured 3D point

        x1 = 0.0
        y1 = 0.0
        z1 = 0.0
        x2 = x
        y2 = y
        z2 = z

        t_denominator = self.a * (x2 - x1) + self.b * (y2 - y1) + self.c * (z2 - z1)
        # if t_denominator == 0:
        #     return None
        t = -1 * (self.a * x1 + self.b * y1 + self.c * z1 + self.d) / t_denominator

        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        z = z1 + t * (z2 - z1)

        #y = y - 13.0

        return x,y,z


        # https://mathinsight.org/distance_point_plane
        # distance_to_plane = math.abs(self.a * point.x + self.b * point.y + self.c * point.z + self.d) / math.sqrt(self.a**2 + self.b**2 + self.c**2)


    def perpendicular_error(self):
        x, y, z = self.points[:, 0], self.points[:, 1], self.points[:, 2]
        length_squared = self.a**2 + self.b**2 + self.c**2
        if length_squared > 0:
            return ((self.a * x + self.b * y + self.c * z + self.d) ** 2 / length_squared).sum()
        else:
            return "N/A"
    
    def perpendicular_error_for_all_points(self):
        x, y, z = self.points[:, 0], self.points[:, 1], self.points[:, 2]
        length_squared = self.a**2 + self.b**2 + self.c**2
        if length_squared > 0:
            return ((self.a * x + self.b * y + self.c * z + self.d) ** 2 / length_squared)
        else:
            return "N/A"

    
    def perpendicular_error_for_point(self, x,y,z):
        length_squared = self.a**2 + self.b**2 + self.c**2
        if length_squared > 0:
            return ((self.a * x + self.b * y + self.c * z + self.d) ** 2 / length_squared)
        else:
            return "N/A"


    def unique_from(self, other_planes, minimum_total_planar_parameter_distance, minimum_parallel_planer_distance):
        # check if this plane has already been found with slight perturbation, e.g.
        # self        -0.5074x + -0.0269y + 0.8613z = -393.6713
        # other_plane  0.5074x + 0.0267y + -0.8613z = 393.6643
        # print("Total other planes: {}".format(len(other_planes)))
        # print("UNIQUENESS:")
        # print("a1={},b1={},c1={},d1={}".format(self.a, self.b, self.c, self.d))
        # print("a2={},b2={},c2={},d2={}".format(other_planes[0].a, other_planes[0].b, other_planes[0].c, other_planes[0].d))

        for other_plane in other_planes:
            a_distance = abs( self.a - other_plane.a )
            b_distance = abs( self.b - other_plane.b )
            c_distance = abs( self.c - other_plane.c )
            d_distance = abs( self.d - other_plane.d )
            if a_distance + b_distance + c_distance < minimum_total_planar_parameter_distance and d_distance < minimum_parallel_planer_distance:
                return False
        return True

    def combine_planes(self, other_plane):
        self.points = np.vstack((self.points, other_plane.points))
        self.points_h.extend(other_plane.points_h)
        self.points_v.extend(other_plane.points_v)
        #self.points_h = np.vstack((self.points_h, other_plane.points_h))
        #self.points_v = np.vstack((self.points_v, other_plane.points_v))
        self.optimize_plane_perpendicular_distance_to_points()

    def average_point(self):
        average_x, average_y, average_z = np.average(self.points, axis=0)
        return Point(x=average_x, y=average_y, z=average_z)


class PlaneFinder():
    def __init__(self, project_name, points = None, path_to_point_cloud_file = None, path_to_plane_file=None, maximum_planes_to_fit = 10, minimum_points_per_plane = 1000, min_distance = 0.0, max_distance = 10000.0, delete_ply_file=True):
        self.project_name = project_name 
        if type(path_to_plane_file) == type(None):
            if type(points) != type(None) and type(path_to_point_cloud_file) == type(None):
                path_to_point_cloud_file = self.preprocess_points_for_plane_finding(points = points, min_distance = min_distance, max_distance = max_distance)

            self.point_cloud_directory = "/".join(path_to_point_cloud_file.split("/")[0:-1])
            self.point_cloud_filename = path_to_point_cloud_file.split("/")[-1]
            self.point_cloud_identifier = self.point_cloud_filename.replace("_hvxyz.csv", "")

            self.read_point_cloud(self.point_cloud_filename)

            # set parameters 
            self.maximum_planes_to_fit = maximum_planes_to_fit
            self.minimum_points_per_plane = minimum_points_per_plane

            self.maximum_point_to_plane_error_in_microns = 75
            self.maximum_number_of_trials = 25
            self.point_to_plane_distance_threshold = 0.30 # maximum planarity error when adding new points, in millimeters
            self.outlier_to_remove_for_this_plane_threshold = self.point_to_plane_distance_threshold * 2.5
            self.minimum_total_planar_parameter_distance = 0.2
            self.minimum_parallel_planer_distance = 2.0
            self.new_points_per_plane_equation_update = 1
            self.plane_milestones = [10**exp for exp in range(2,7)] 
            self.best_planes = []
            self.total_start_time = time.time()

            self.find_planes()

            # delete .ply file to save memory
            if delete_ply_file:
                remove(self.point_cloud_filename)
        else:
            self.best_planes = []
            self.read_plane_data_from_file(path_to_plane_file)

    def read_plane_data_from_file(self, path_to_plane_file):
        total_planes = 0
        points_seen = []
        with open("{}".format(path_to_plane_file), "r") as lines:
            for line in lines:
                if "PLANE" in line:
                    if len(points_seen) > 0:
                        self.best_planes[total_planes-1].add_points_to_plane(points_seen)
                        points_seen = []

                    plane_index = total_planes #int(line.split("PLANE ")[1].split(":")[0])
                    a = float(line.split(": a=")[1].split(",b=")[0])
                    b = float(line.split("b=")[1].split(",c=")[0])
                    c = float(line.split("c=")[1].split(",d=")[0])
                    d = float(line.split("d=")[1].split("\n")[0])
                    print("Plane {}: {:.4f}x + {:.4f}y + {:.4f}z + {:.4f} = 0".format(total_planes, a, b ,c, d))
                    total_planes += 1
                    self.best_planes.append(Plane(a=a, b=b, c=c, d=d))
                elif "," in line:
                    h = int(line.rstrip("\n").split(",")[0])
                    v = int(line.rstrip("\n").split(",")[1])
                    x = float(line.rstrip("\n").split(",")[2])
                    y = float(line.rstrip("\n").split(",")[3])
                    z = float(line.rstrip("\n").split(",")[4])
                    p = Point(x=x,y=y,z=z,row=h,column=v)
                    points_seen.append(p)

            if len(points_seen) > 0:
                self.best_planes[total_planes-1].add_points_to_plane(points_seen)
                points_seen = []
        

    def find_new_planes(self, points):
        path_to_point_cloud_file = self.preprocess_points_for_plane_finding(points = points)
        self.point_cloud_directory = "/".join(path_to_point_cloud_file.split("/")[0:-1])
        self.point_cloud_filename = path_to_point_cloud_file.split("/")[-1]
        self.point_cloud_identifier = self.point_cloud_filename.replace("_hvxyz.csv", "")
        self.read_point_cloud(self.point_cloud_filename)
        self.best_planes = []
        self.total_start_time = time.time()
        self.find_planes()
        
    def grouper(self, n, iterable, padvalue=None):
        #"grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
        return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

    def read_point_cloud(self, point_cloud_filename):
        self.point_cloud = []
        sys.stdout.flush()
        invalid_points = 0
        print(point_cloud_filename)
        with open(point_cloud_filename, 'rb') as f:
            lines = [l.decode('utf8', 'ignore') for l in f.readlines()]
        #with open(point_cloud_filename, "r") as lines: # block_z_plus_3_cm_hvxyz.csv
            for i, line in enumerate(lines):
                row,column,x,y,z = line.rstrip("\n").split(",")
                point = Point(float(x), float(y), float(z), int(row), int(column))
                if point.valid:
                    self.point_cloud.append(point)
                else:
                    invalid_points += 1
        print("Removed {:,} invalid points (no data from scanner)".format(invalid_points))


    # 
    #     elif ".ply" in self.filename:
    #         self.data = PlyData.read(self.filename)
    #         self.x              = self.data.elements[0].data['x']
    #         self.y              = self.data.elements[0].data['y']
    #         self.z              = self.data.elements[0].data['z']
    #         self.red            = self.data.elements[0].data['red']
    #         self.green          = self.data.elements[0].data['green']
    #         self.blue           = self.data.elements[0].data['blue']

    #         if camera_coordinate_system == "camera_coordinate_system":
    #             #print("Camera camera_coordinate_system")
    #             self.pixel_row      = pickle.load(open("point_index_to_pixel_row_for_camera_coordinate_system.pkl", "rb"))
    #             self.pixel_column   = pickle.load(open("point_index_to_pixel_column_for_camera_coordinate_system.pkl", "rb"))           
    #         else:
    #             #print("Else")
    #             self.pixel_row      = pickle.load(open("point_index_to_pixel_row.pkl", "rb"))
    #             self.pixel_column   = pickle.load(open("point_index_to_pixel_column.pkl", "rb"))





    def sample_n_consecutive_points_without_replacement(self, n, set_of_points): 
        maximum_distance_between_sampled_points = 10.0 # mm
        new_set_of_points = set_of_points.copy() # ensure original set is unedited
        sampled_valid_point = False
        number_of_attempts = 0
        start_time = time.time()
        while not sampled_valid_point:
            number_of_attempts += 1
            sampled_point_indices = random.sample(range(len(new_set_of_points) - n + 1), 3)
            p1 = new_set_of_points[sampled_point_indices[0]]
            p2 = new_set_of_points[sampled_point_indices[1]]
            p3 = new_set_of_points[sampled_point_indices[2]]
            if not p1.valid or not p2.valid or not p3.valid:
                continue
            if not p1.distance(p2) < maximum_distance_between_sampled_points:
                continue
            if not p1.distance(p3) < maximum_distance_between_sampled_points:
                continue
            if not p2.distance(p3) < maximum_distance_between_sampled_points:
                continue
            sampled_valid_point = True
        end_time = time.time()
        duration = round(end_time - start_time,4)
        print("  Plane initialized after {:,} samples, {} seconds at p1=({},{},{}), p2=({},{},{}), p3=({},{},{})".format(number_of_attempts, duration, round(p1.x,1), round(p1.y,1), round(p1.z,1), round(p2.x,1), round(p2.y,1), round(p2.z,1), round(p3.x,1), round(p3.y,1), round(p3.z,1)))
        sys.stdout.flush()

        # now we remove the points we sampled
        sampled_point_indices.sort(reverse=True)
        sampled_points = [new_set_of_points[sampled_point_index] for sampled_point_index in sampled_point_indices]
        for sampled_point_index in sampled_point_indices:
           del(new_set_of_points[sampled_point_index])
        return sampled_points, new_set_of_points

    def average_point(self, plane_number=0):
        plane = self.best_planes[plane_number]
        average_x, average_y, average_z = np.average(plane.points, axis=0)
        return Point(x=average_x, y=average_y, z=average_z)

    def find_planes(self, rows = 2048, columns = 2048):
        ray.shutdown()
        ray.init(num_cpus=cpus, logging_level=logging.WARNING, ignore_reinit_error=True)

        @ray.remote
        def add_points_in_plane(sampled_points, potential_points_to_add, best_planes, point_to_plane_distance_threshold):
            new_points_per_plane_equation_update = 1
            plane_milestones = [10**exp for exp in range(2,7)]
            sub_plane = Plane(sampled_points)
            points_to_add_to_plane = []
            points_to_remove_from_set = []
            total_outliers_for_this_plane_removed = 0
            outlier_to_remove_for_this_plane_threshold = point_to_plane_distance_threshold * 2.0
            for i in range(len(potential_points_to_add) - 6, -1, -1):
                next_point = potential_points_to_add[i]
                if type(next_point) == type(None):
                    break
                planarity_of_next_point = sub_plane.get_planarity(next_point.xyz)
                if planarity_of_next_point < point_to_plane_distance_threshold:
                    points_to_add_to_plane.append(i)
                    sub_plane.add_points_to_plane([next_point])
                    points_to_remove_from_set.append(i) # save index relative to this batch
                    if i % new_points_per_plane_equation_update == 0: # potentially update plane calculation
                        sub_plane.optimize_plane_perpendicular_distance_to_points()
                    if len(sub_plane.points) in plane_milestones: # update frequency of plane calculation, depending on how many existing points
                        new_points_per_plane_equation_update = math.ceil( len(sub_plane.points) / 4.0 ) # int(math.sqrt( len(current_plane.points)*25 ))
                elif planarity_of_next_point < outlier_to_remove_for_this_plane_threshold:
                    total_outliers_for_this_plane_removed += 1
                    points_to_remove_from_set.append(i)
            return points_to_add_to_plane, points_to_remove_from_set



        #ray.register_class(PointCloud)
        print("Launching plane finding algorithm on {} CPUs".format(cpus), flush=True)

        remaining_points_across_all_planes_indices = random.sample(range(len(self.point_cloud)), len(self.point_cloud) - 1)
        remaining_points_across_all_planes_indices.sort()
        remaining_points_across_all_planes = [self.point_cloud[i] for i in remaining_points_across_all_planes_indices]
        random.shuffle(remaining_points_across_all_planes) # reshuffle sampling so distributed processing is less correlated to sample

        for trial_number in range(self.maximum_number_of_trials):
            if len(remaining_points_across_all_planes) < self.minimum_points_per_plane or len(self.best_planes) >= self.maximum_planes_to_fit:
                break
            print("\nTrial {} with {:,} points remaining, {} unique planes currently found".format(trial_number+1, len(remaining_points_across_all_planes), len(self.best_planes)))    
            sys.stdout.flush()

            sampled_points, remaining_points_for_this_trial  = self.sample_n_consecutive_points_without_replacement(n=3, set_of_points=remaining_points_across_all_planes)
            random.shuffle(remaining_points_for_this_trial) # keep shuffling each iteration, to continue to break correlations 

            current_plane = Plane(sampled_points)
            total_initial_points = len(remaining_points_for_this_trial)
            total_outliers_for_this_plane_removed = 0

            # loop through all other points and measure fit for this plane
            points_per_cpu = math.ceil(len(remaining_points_for_this_trial) / cpus)
            points_to_process_for_cpus = self.grouper(points_per_cpu, remaining_points_for_this_trial)
            parallel_cpu_computations = [add_points_in_plane.remote(sampled_points=sampled_points, potential_points_to_add=points_to_process_for_cpu, best_planes=self.best_planes, point_to_plane_distance_threshold=self.point_to_plane_distance_threshold) for points_to_process_for_cpu in points_to_process_for_cpus]
            parallel_cpu_results = ray.get(parallel_cpu_computations)
            current_index = 0
            all_points_to_remove = []
            for cpu, result in enumerate(parallel_cpu_results):
                points_to_add, points_to_remove = result
                start_index = cpu * points_per_cpu

                points_to_add = [remaining_points_for_this_trial[start_index + point_index] for point_index in points_to_add]
                if len(points_to_add) > 0:
                    current_plane.add_points_to_plane(points_to_add)
                    current_plane.optimize_plane_perpendicular_distance_to_points()

                for point_index_to_remove in points_to_remove:
                    all_points_to_remove.append(start_index + point_index_to_remove)

            remaining_points_for_this_trial = np.delete(remaining_points_for_this_trial, all_points_to_remove).tolist()

            indices_in_current_plane_to_remove = []
            for point_number, point in enumerate(current_plane.points):
                if current_plane.get_planarity(point) > self.point_to_plane_distance_threshold:
                    indices_in_current_plane_to_remove.append(point_number)

            print("  Combined subplanes into plane with {:,} points, removing {:,} outliers".format(len(current_plane.points), len(indices_in_current_plane_to_remove)))
            current_plane.remove_points_by_indices(indices_in_current_plane_to_remove)
            current_plane.optimize_plane_perpendicular_distance_to_points()

            final_error = current_plane.perpendicular_error()
            if final_error == "N/A":
                continue
            else:
                error_per_point = final_error / float(len(current_plane.points) )

            print("  From trial {} = {:,} points in plane with average error {} microns / point ({:,} points total to remove)".format(trial_number + 1, len(current_plane.points), round(1000 * error_per_point, 1), len(all_points_to_remove)))
            print("  {}x + {}y + {}z + {} = 0".format(round(current_plane.a,4), round(current_plane.b,4), round(current_plane.c,4), round(current_plane.d,4)))

            # save this plane if it's sufficiently large, and the largest seen for this set of trials
            if len(current_plane.points) > self.minimum_points_per_plane:   
                final_error = current_plane.perpendicular_error()
                error_per_point = final_error / float(len(current_plane.points))
                if error_per_point * 1000 < self.maximum_point_to_plane_error_in_microns: # ensure this plane is an excellent overall fit
                    is_unique = True
                    for plane_number, best_plane in enumerate(self.best_planes):
                        if not current_plane.unique_from([best_plane], self.minimum_total_planar_parameter_distance, self.minimum_parallel_planer_distance): # ensure this plane is sufficiently unique
                            print("  -> This plane is not unique, combining with one of our existing planes")
                            is_unique = False
                            self.best_planes[plane_number].combine_planes(current_plane) # allow another plane to absorb this one
                            break
                    if is_unique:
                        print("  -> This plane is unique, adding to list of best planes")
                        self.best_planes.append(current_plane) # otherwise create a new plane
                else:
                    print("  -> This plane is not precise enough for us to save, tossing it out")
            else:
                print("  -> Sufficiently large plane not found, moving on")

            # regardless, let's not reconsider the points from this plane ever again
            remaining_points_across_all_planes = remaining_points_for_this_trial

        # after all plane fitting has finished
        # remove points from smaller planes that could very well belong to the largest plane
        print("")
        for combo in combinations( range(len(self.best_planes)), 2):
            plane_i = self.best_planes[combo[0]]
            plane_ii = self.best_planes[combo[1]]

            # remove points in plane i that are too close in planarity to points in plane ii 
            indices_in_point_i_to_remove = []
            for point_number, point in enumerate(plane_i.points):
                if plane_ii.get_planarity(point) < self.outlier_to_remove_for_this_plane_threshold:
                    indices_in_point_i_to_remove.append(point_number)
            print("Removing {} points from plane {} that are too close in planarity to plane {}".format(len(indices_in_point_i_to_remove), combo[0], combo[1]))
            plane_i.remove_points_by_indices(indices_in_point_i_to_remove)
            plane_i.optimize_plane_perpendicular_distance_to_points()

            # and do so in reverse, just in case
            indices_in_point_ii_to_remove = []
            for point_number, point in enumerate(plane_ii.points):
                if plane_i.get_planarity(point) < self.outlier_to_remove_for_this_plane_threshold:
                    indices_in_point_ii_to_remove.append(point_number)
            print("Removing {} points from plane {} that are too close in planarity to plane {}".format(len(indices_in_point_ii_to_remove), combo[1], combo[0]))
            plane_ii.remove_points_by_indices(indices_in_point_ii_to_remove)
            plane_ii.optimize_plane_perpendicular_distance_to_points()

        print("\nNow removing outliers with planarity below 40 μm...")
        for plane_number, plane in enumerate(self.best_planes):
            indices_in_plane_to_remove = []
            for point_number, point in enumerate(plane.points):
                if plane.get_planarity(point) > 0.40:
                    indices_in_plane_to_remove.append(point_number)
            print("Removing {} points from plane {}".format(len(indices_in_plane_to_remove), plane_number))
            plane.remove_points_by_indices(indices_in_plane_to_remove)
            plane.optimize_plane_perpendicular_distance_to_points()

        # print("\nNow removing outliers with planarity below 0.01 mm...")
        # for plane_number, plane in enumerate(self.best_planes):
        #     indices_in_plane_to_remove = []
        #     for point_number, point in enumerate(plane.points):
        #         if plane.get_planarity(point) > 0.01:
        #             indices_in_plane_to_remove.append(point_number)
        #     print("Removing {} points from plane {}".format(len(indices_in_plane_to_remove), plane_number))
        #     plane.remove_points_by_indices(indices_in_plane_to_remove)
        #     plane.optimize_plane_perpendicular_distance_to_points()

        # print("\nNow removing outliers with planarity below 0.001 microns...")
        # for plane_number, plane in enumerate(self.best_planes):
        #     indices_in_plane_to_remove = []
        #     for point_number, point in enumerate(plane.points):
        #         if plane.get_planarity(point) > 0.001:
        #             indices_in_plane_to_remove.append(point_number)
        #     print("Removing {} points from plane {}".format(len(indices_in_plane_to_remove), plane_number))
        #     plane.remove_points_by_indices(indices_in_plane_to_remove)
        #     plane.optimize_plane_perpendicular_distance_to_points()



        reds,greens,blues = sample_n_colors_uniformly_from_human_perceptual_space(n_colors=len(self.best_planes))

        new_best_planes = []
        plane_colors = [[0,0,0]]
        print("\nBEST PLANES:")
        for plane_number, plane in enumerate(self.best_planes):
            red = reds[plane_number]
            green = greens[plane_number]
            blue = blues[plane_number]
            plane_colors.append([red, green, blue])
            if len(plane.points) > self.minimum_points_per_plane:
                new_best_planes.append(plane)
                print("\n({}): {:,} points with perpendicular error per point = {:.4f} μm".format(plane_number+1, len(plane.points), 1000 * plane.perpendicular_error() / float(len(plane.points)) ))
                print("{}x + {}y + {}z + {} = 0 with color ({},{},{})".format(round(plane.a,4), round(plane.b,4), round(plane.c,4), round(plane.d,4), red, green, blue ))
            
        self.best_planes = new_best_planes

        


        # visualize planes as colored pixels overlaid on original perspective
        img_abcd = Image.new('RGBA', (rows, columns), color = 'white')
        pixels_abcd = img_abcd.load()
        plane_fitted = np.full((rows, columns), 0, dtype='int')
        plane_error = np.full((rows, columns), 0, dtype=np.float32)


        for row in range(rows):
            for column in range(columns):
                pixels_abcd[row, column] = (255, 255, 255, 255)

        a = np.full((rows, columns), 0.0)
        b = np.full((rows, columns), 0.0)
        c = np.full((rows, columns), 0.0)
        d = np.full((rows, columns), 0.0)

        for plane_index, plane in enumerate(self.best_planes):

            perpendicular_error = plane.perpendicular_error_for_all_points()

            x,y,z = plane.find_point_position_on_plane(x=plane.points[:][0], y=plane.points[:][1], z=plane.points[:][2])

            # print(len(x))
            # print(len(plane.points[:][0]))
            # print("x,y,z points project on plane:")
            # print("x: {}".format(x[:10]))
            # print("y: {}".format(y[:10]))
            # print("x: {}".format(z[:10]))

            # print("x,y,z points original:")
            # print("x: {}".format(plane.points[:10][0]))
            # print("y: {}".format(plane.points[:10][1]))
            # print("x: {}".format(plane.points[:10][2]))

            x_distance = x - plane.points[:][0]
            y_distance = y - plane.points[:][1]
            z_distance = z - plane.points[:][2]
            three_d_distance = np.sqrt(x_distance**2 + y_distance**2 + z_distance**2)

            # analyze(label="x distance from plane", np_array=x_distance)
            # analyze(label="y distance from plane", np_array=y_distance)
            # analyze(label="z distance from plane", np_array=z_distance)
            # analyze(label="3D distance from plane", np_array=three_d_distance)

            min_error = np.min(perpendicular_error)
            max_error = np.max(perpendicular_error)
            average_error = np.average(perpendicular_error)
            std_error = np.std(perpendicular_error)

            print("For plane {}, error shape is {}, min_error = {}, max_error = {}, average_error = {}, standard deviation {}".format(plane_index, perpendicular_error.shape, min_error, max_error, average_error, std_error))

            for point_index, point in enumerate(plane.points):
                column = plane.points_v[point_index]
                row = plane.points_h[point_index]
                plane_fitted[row, column] = plane_index + 1
                plane_error[row, column] = 1.0 - ((perpendicular_error[point_index] - min_error) / (1.5 * (max_error - min_error)))

        print("\nSaving data visualization...")
        for row in range(rows):
            for column in range(columns):
                plane_index = plane_fitted[row, column]
                pixel_error = plane_error[row, column]
                red, green, blue = plane_colors[plane_index]
                pixels_abcd[row, columns - column - 1] = (int(red * pixel_error), int(green * pixel_error), int(blue * pixel_error), 255)

        plane_filename = "{}/{}_abcd.png".format(self.point_cloud_directory, self.point_cloud_identifier)
        img_abcd.save('{}'.format(plane_filename))
        print("Saved visualization to {}".format(plane_filename))

        print("Saving output plane point data to {}_plane_data.csv".format(self.project_name))
        with open("{}/{}_plane_data.csv".format(self.point_cloud_directory, self.project_name), "w") as output_file:
            for i, plane in enumerate(self.best_planes):
                error_per_point = round(1000 * plane.perpendicular_error() / float(len(plane.points)), 1 )
                #if error_per_point < 2.5: # microns, probably removes the turntable and any other planes
                a = plane.a
                b = plane.b
                c = plane.c
                d = plane.d
                if i == 0:
                    # save easy variable lookup for entire plane_finder
                    self.a = a
                    self.b = b
                    self.c = c
                    self.d = d
                output_file.write("PLANE {}: a={},b={},c={},d={}\n".format(i,a,b,c,d))
                for i, point in enumerate(plane.points):
                    h = plane.points_h[i]
                    v = plane.points_v[i]
                    x = point[0]
                    y = point[1]
                    z = point[2]
                    output_file.write("{},{},{},{},{}\n".format(h,v,x,y,z))

        total_end_time = time.time()
        total_duration = round(total_end_time - self.total_start_time, 2)
        print("\nTotal duration of plane finding algorithm on {} CPUs: {} seconds".format(cpus, total_duration))

        ray.shutdown()


def find_block_corners(point_cloud_filename, project_scan_name=None, rows = 2048, columns = 2048):
    def grouper(n, iterable, padvalue=None):
        "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
        return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

    cpus = multiprocessing.cpu_count() 
    print("\nLaunching 8 corner point recognition algorithm on {} CPUs".format(cpus), flush=True)
    ray.shutdown()
    ray.init(num_cpus=cpus, logging_level=logging.WARNING, ignore_reinit_error=True)

    mm_per_inch = 25.4

    point_cloud_directory = getcwd() 
    point_cloud_identifier = point_cloud_filename.replace("_hvxyz.csv", "")
    point_cloud_filename = "{}/{}_plane_data.csv".format(point_cloud_directory, point_cloud_identifier)

    class Vector():
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
            
    class BlockPlane():
        def __init__(self, a, b, c, d):
            self.a = a
            self.b = b 
            self.c = c
            self.d = d 
            self.points = []

        def perpendicular_error(self):
            length_squared = self.a**2 + self.b**2 + self.c**2
            if length_squared > 0:
                return ((self.a * self.xs + self.b * self.ys + self.c * self.zs + self.d) ** 2 / length_squared).sum() / len(self.xs)

        def get_planarity(self, point):  
            planarity = abs( self.a * point[0] + self.b * point[1] + self.c * point[2] + self.d ) / (self.a**2 + self.b**2 + self.c**2)
            return planarity

        def remove_points_by_indices(self, indices_of_points_to_remove):
             self.points = [points for i, points in enumerate(self.points) if i not in indices_of_points_to_remove]
             self.xs = np.delete(self.xs, indices_of_points_to_remove, 0)
             self.ys = np.delete(self.ys, indices_of_points_to_remove, 0)
             self.zs = np.delete(self.ys, indices_of_points_to_remove, 0)


    points_of_planes = [[] for i in range(4)]
    xs_of_planes = [[] for i in range(4)]
    ys_of_planes = [[] for i in range(4)]
    zs_of_planes = [[] for i in range(4)]

    print("\nReading plane data from {}".format(point_cloud_identifier))

    all_planes = []
    total_planes = 0
    with open("{}".format(point_cloud_filename), "r") as lines:
        for line in lines:
            if "PLANE" in line:
                plane_index = total_planes #int(line.split("PLANE ")[1].split(":")[0])
                a = float(line.split(": a=")[1].split(",b=")[0])
                b = float(line.split("b=")[1].split(",c=")[0])
                c = float(line.split("c=")[1].split(",d=")[0])
                d = float(line.split("d=")[1].split("\n")[0])
                print("PLANE {}: a={:.4f}, b={:.4f}, c={:.4f}, d={:.4f}".format(total_planes, a, b ,c, d))
                total_planes += 1
                all_planes.append(BlockPlane(a=a, b=b, c=c, d=d))
            elif "," in line:
                h = int(line.rstrip("\n").split(",")[0])
                v = int(line.rstrip("\n").split(",")[1])
                x = float(line.rstrip("\n").split(",")[2])
                y = float(line.rstrip("\n").split(",")[3])
                z = float(line.rstrip("\n").split(",")[4])
                p = Point(x=x, y=y, z=z, row=h, column=v)
                points_of_planes[plane_index].append(p)
                xs_of_planes[plane_index].append(x)
                ys_of_planes[plane_index].append(y)
                zs_of_planes[plane_index].append(z)


    for i, points_of_plane in enumerate(points_of_planes):
        try:
            all_planes[i].points = points_of_plane
            all_planes[i].xs = np.asarray(xs_of_planes[i])
            all_planes[i].ys = np.asarray(ys_of_planes[i])
            all_planes[i].zs = np.asarray(zs_of_planes[i])
        except IndexError:
            print("Unable to find a plane at index {}, which probably means that not all planes were fitted for this block; moving on, this scan invalid".format(i))
            return None, None, None

    @ray.remote
    def compute_average_distance(points):
        pairwise_distances = []
        total_points = float(len(points)*len(points))
        for point_1 in points:
            for point_2 in points:
                if type(point_1) != type(None) and type(point_2) != type(None):
                    pairwise_distances.append(point_1.distance(point_2) / mm_per_inch)
                else:
                    total_points -= 1
        average_distance_for_cpu = sum(pairwise_distances) / total_points
        return average_distance_for_cpu

    sample_size_as_percent_of_number_of_plane_points = 0.75
    all_distances_per_plane = [[] for i in range(3)]
    average_distances = {0: 0, 1: 0, 2:0}
    max_distances = {0: 0, 1: 0, 2:0}
    max_distance_points = []
    surfaces = {0: None, 1: None, 2: None}
    max_sample_size = 5000

    for plane_number in [0,1,2,3]:
        max_distance_points.append([])
        max_distance = 0
        sample_size = sample_size_as_percent_of_number_of_plane_points * len(points_of_planes[plane_number])
        if sample_size > max_sample_size:
            sample_size = max_sample_size
        sample_of_points = random.sample(points_of_planes[plane_number], int(sample_size))
        print("\nComputing pairwise distances for {:,} points in plane {} to find average distance, corresponding to a topological metric".format(len(sample_of_points), plane_number))
        points_per_cpu = math.ceil(len(sample_of_points) / cpus)
        points_to_process_for_cpus = grouper(points_per_cpu, sample_of_points)
        parallel_cpu_computations = [compute_average_distance.remote(points_to_process_for_cpu) for points_to_process_for_cpu in points_to_process_for_cpus]
        parallel_cpu_results = ray.get(parallel_cpu_computations)
        if len(parallel_cpu_results) > 0:
            average_distances[plane_number] = sum(parallel_cpu_results) / float(len(parallel_cpu_results))
            print("For plane {}, average distance of {:.4f}\"".format(plane_number, average_distances[plane_number])) # for points (h,v)=(x,y,z) at p1: ({},{})=({:.3f},{:.3f},{:.3f}) and p2: ({},{})=({:.3f},{:.3f},{:.3f})".format(plane_number, max_distance, point_a.h, point_a.v, point_a.x, point_a.y, point_a.z, point_b.h, point_b.v, point_b.x, point_b.y, point_b.z))
        else:
            print("For plane {}, there are no results returned from parallel computing; throwing out data, moving on\n".format(plane_number))    
            return None, None, None                 

    average_distances_sorted_keys = sorted(average_distances, key=average_distances.get, reverse=True)
    plane_surface_identification = {"Ground": None, "2\"x3\"": None, "1\"x3\"": None, "1\"x2\"": None}

    print("")
    for key, surface in zip(average_distances_sorted_keys, ["Ground", "2\"x3\"", "1\"x3\"", "1\"x2\""]):
        average_distance = average_distances[key]
        if surface == "Ground":
            print("PLANE {} is identified as the Ground turntable surface with average distance {:.2f}\" between points".format(key, average_distance))
            plane_surface_identification["Ground"] = all_planes[key]
        if surface == "2\"x3\"":
            if average_distance > math.sqrt(13) + 0.1:
                print("For surface 2\"x3\" the average distance {:.4f} inches is longer than SQRT(3^2 + 2^2) = 3.6055; data needs further filtering or plane recognition wrong".format(average_distance))
                print("Average distance too large for this surface; moving on\n")
                return None, None, None
            elif average_distance < 1.1:
                print("For surface 2\"x3\" the average distance {:.4f} inches is smaller than data history; data needs further filtering or plane recognition wrong".format(average_distance))
                print("Average distance too small for this surface; moving on\n")    
                return None, None, None            
            else:
                print("PLANE {} is identified as a 2\"x3\" surface with average distance {:.2f}\" between points".format(key, average_distance))
                plane_surface_identification["2\"x3\""] = all_planes[key]
        elif surface == "1\"x3\"":
            if average_distance > math.sqrt(10) + 0.1:
                print("For surface 1\"x3\" the average distance {:.4f} inches is longer than SQRT(3^2 + 1^2) = 3.1622; data needs further filtering or plane recognition wrong".format(average_distance))
                print("Average distance too large for this surface; moving on\n")
                return None, None, None
            elif average_distance < 0.9:
                print("For surface 1\"x3\" the average distance {:.4f} inches is smaller than data history; data needs further filtering or plane recognition wrong".format(average_distance))
                print("Average distance too small for this surface; moving on\n")    
                return None, None, None   
            else:
                print("PLANE {} is identified as a 1\"x3\" surface with average distance {:.2f}\" between points".format(key, average_distance))
                plane_surface_identification["1\"x3\""] = all_planes[key]
        elif surface == "1\"x2\"":
            if average_distance > math.sqrt(5) + 0.1:
                print("For surface 1\"x2\" the average distance {:.4f} is longer than SQRT(2^2 + 1^2) = 2.2360; data needs further filtering or plane recognition wrong".format(average_distance))
                print("Average distance too large for this surface; moving on\n")
                return None, None, None
            elif average_distance < 0.6:
                print("For surface 1\"x2\" the average distance {:.4f} inches is smaller than data history; data needs further filtering or plane recognition wrong".format(average_distance))
                print("Average distance too small for this surface; moving on\n")    
                return None, None, None   
            else:
                print("PLANE {} is identified as a 1\"x2\" surface with average distance {:.2f}\" between points".format(key, average_distance))
                plane_surface_identification["1\"x2\""] = all_planes[key]

    def angle_between(plane_1, plane_2):
        s = plane_1.a * plane_2.a + plane_1.b * plane_2.b + plane_1.c * plane_2.c
        t = math.sqrt(plane_1.a ** 2 + plane_1.b ** 2 + plane_1.c ** 2)
        u = math.sqrt(plane_2.a ** 2 + plane_2.b ** 2 + plane_2.c ** 2)
        s = s / (t * u)
        angle = (180.0 / math.pi) * (math.acos(s))
        angle = abs(90.0 - angle)
        return angle

    def fast_angle_between(a1,b1,c1,d1,a2,b2,c2,d2):
        s = a1 * a2 + b1 * b2 + c1 * c2
        t = math.sqrt(a1 ** 2 + b1 ** 2 + c1 ** 2)
        u = math.sqrt(a2 ** 2 + b2 ** 2 + c2 ** 2)
        s = s / (t * u)
        angle = (180.0 / math.pi) * (math.acos(s))
        angle = abs(90.0 - angle)
        return angle

    # bundle adjustment: given that we have 3 sets of points, and 3 plane equations, we want to adjust these plane equations such that the angle between each of them = 90.000 degrees AND the sum of perpendicular error with normalized constraint is = 0.0
    def optimize_plane_to_plane_angle_while_optimizing_point_to_plane_error(plane_1, plane_2, plane_3):
        x1 = plane_1.xs
        y1 = plane_1.ys
        z1 = plane_1.zs
        x2 = plane_2.xs
        y2 = plane_2.ys
        z2 = plane_2.zs
        x3 = plane_3.xs
        y3 = plane_3.ys
        z3 = plane_3.zs

        initial_guess = [plane_1.a, plane_1.b, plane_1.c, plane_1.d, plane_2.a, plane_2.b, plane_2.c, plane_2.d, plane_3.a, plane_3.b, plane_3.c, plane_3.d]

        def model(params, all_xyz):
            a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3 = params
            x1, y1, z1, x2, y2, z2, x3, y3, z3 = all_xyz

            length_squared_1 = a1**2 + b1**2 + c1**2
            length_squared_2 = a2**2 + b2**2 + c2**2
            length_squared_3 = a3**2 + b3**2 + c3**2

            if length_squared_1 > 0 and length_squared_2 > 0 and length_squared_3 > 0:
                error_from_plane_1_to_points = ((a1 * x1 + b1 * y1 + c1 * z1 + d1) ** 2 / length_squared_1).sum()
                error_from_plane_2_to_points = ((a2 * x2 + b2 * y2 + c2 * z2 + d2) ** 2 / length_squared_2).sum()
                error_from_plane_3_to_points = ((a3 * x3 + b3 * y3 + c3 * z3 + d3) ** 2 / length_squared_3).sum()

                sum_of_differences = error_from_plane_1_to_points + error_from_plane_2_to_points + error_from_plane_3_to_points

                return sum_of_differences #* angle_error

            else:
                return 1000.0

        def unit_length_plane_1(params):
            a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3 = params
            return a1**2 + b1**2 + c1**2 - 1

        def unit_length_plane_2(params):
            a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3 = params
            return a2**2 + b2**2 + c2**2 - 1

        def unit_length_plane_3(params):
            a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3 = params
            return a3**2 + b3**2 + c3**2 - 1

        def planar_angle_1(params):
            a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3 = params
            angle_error_1 = fast_angle_between(a1, b1, c1, d1, a2, b2, c2, d2)
            return angle_error_1

        def planar_angle_2(params):
            a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3 = params
            angle_error_2 = fast_angle_between(a1, b1, c1, d1, a3, b3, c3, d3)
            return angle_error_2

        def planar_angle_3(params):
            a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3 = params
            angle_error_3 = fast_angle_between(a2, b2, c2, d2, a3, b3, c3, d3)
            return angle_error_3

        constraint_1 = ({'type': 'eq', 'fun': unit_length_plane_1})
        constraint_2 = ({'type': 'eq', 'fun': unit_length_plane_2})
        constraint_3 = ({'type': 'eq', 'fun': unit_length_plane_3})
        constraint_4 = ({'type': 'eq', 'fun': planar_angle_1})
        constraint_5 = ({'type': 'eq', 'fun': planar_angle_2})
        constraint_6 = ({'type': 'eq', 'fun': planar_angle_3})

        solution = minimize(model, initial_guess, args=[x1, y1, z1, x2, y2, z2, x3, y3, z3], constraints=[constraint_1, constraint_2, constraint_3, constraint_4, constraint_5, constraint_6])

        a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3 = tuple(solution.x)[0:12]

        return [a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3]


    plane_i = plane_surface_identification["1\"x2\""]
    plane_ii = plane_surface_identification["1\"x3\""]
    plane_iii = plane_surface_identification["2\"x3\""]

    print("\nBefore bundle adjustment, perpendicular error per plane followed by plane angle relative errors:")
    for plane, identifier in zip([plane_i, plane_ii, plane_iii], ["1\"x2\"","1\"x3\"","2\"x3\""]):
        error_in_microns = plane.perpendicular_error() * 1000.0
        print("Plane {} has point-to-plane average error of {:.3f} μm".format(identifier, error_in_microns))

    for plane_a_surface, plane_b_surface in [["1\"x2\"", "1\"x3\""], ["1\"x2\"", "2\"x3\""], ["1\"x3\"", "2\"x3\""]]:
        plane_a = plane_surface_identification[plane_a_surface]
        plane_b = plane_surface_identification[plane_b_surface]
        angle = angle_between(plane_a, plane_b)
        print("{} and {} planes: angle error of {:.5f} degrees relative to each other".format(plane_a_surface, plane_b_surface, angle))

    a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3 = optimize_plane_to_plane_angle_while_optimizing_point_to_plane_error(plane_i, plane_ii, plane_iii)
    plane_i.a = a1
    plane_i.b = b1
    plane_i.c = c1
    plane_i.d = d1
    plane_ii.a = a2
    plane_ii.b = b2
    plane_ii.c = c2
    plane_ii.d = d2
    plane_iii.a = a3
    plane_iii.b = b3
    plane_iii.c = c3
    plane_iii.d = d3

    plane_surface_identification["1\"x2\""] = plane_i
    plane_surface_identification["1\"x3\""] = plane_ii
    plane_surface_identification["2\"x3\""] = plane_iii

    print("\nAfter bundle adjustment, perpendicular error per plane followed by plane angle relative errors:")
    for plane, identifier in zip([plane_i, plane_ii, plane_iii], ["1\"x2\"","1\"x3\"","2\"x3\""]):
        error_in_microns = plane.perpendicular_error() * 1000.0
        print("Plane {} has point-to-plane average error of {:.3f} μm".format(identifier, error_in_microns))

    for plane_a_surface, plane_b_surface in [["1\"x2\"", "1\"x3\""], ["1\"x2\"", "2\"x3\""], ["1\"x3\"", "2\"x3\""]]:
        plane_a = plane_surface_identification[plane_a_surface]
        plane_b = plane_surface_identification[plane_b_surface]
        angle = angle_between(plane_a, plane_b)
        print("{} and {} planes: angle error of {:.5f} degrees relative to each other".format(plane_a_surface, plane_b_surface, angle))

    max_point_to_plane_error = 15.0
    for plane, identifier in zip([plane_i, plane_ii, plane_iii], ["1\"x2\"","1\"x3\"","2\"x3\""]):
        error_in_microns = plane.perpendicular_error() * 1000.0
        if error_in_microns > max_point_to_plane_error:
            print("Plane {} error of {:.3f} μm is outside of max tolerance of {:.1f} μm; quitting and moving on\n".format(identifier, error_in_microns, max_point_to_plane_error))
            return None, None, None


    # # print("\nNow removing outliers with planarity below 10 μm...")
    # # for plane, identifier in zip([plane_i, plane_ii, plane_iii], ["1\"x2\"","1\"x3\"","2\"x3\""]):
    # #     indices_in_plane_to_remove = []
    # #     for point_number, point in enumerate(plane.points):
    # #         if plane.get_planarity([point.x, point.y, point.z]) > 0.30:
    # #             indices_in_plane_to_remove.append(point_number)
    # #     print("Removing {} points from plane {}".format(len(indices_in_plane_to_remove), identifier))
    # #     plane.remove_points_by_indices(indices_in_plane_to_remove)


    # print("Bundle adjustment, round 2")
    # a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3 = optimize_plane_to_plane_angle_while_optimizing_point_to_plane_error(plane_i, plane_ii, plane_iii)
    # plane_i.a = a1
    # plane_i.b = b1
    # plane_i.c = c1
    # plane_i.d = d1
    # plane_ii.a = a2
    # plane_ii.b = b2
    # plane_ii.c = c2
    # plane_ii.d = d2
    # plane_iii.a = a3
    # plane_iii.b = b3
    # plane_iii.c = c3
    # plane_iii.d = d3

    # plane_surface_identification["1\"x2\""] = plane_i
    # plane_surface_identification["1\"x3\""] = plane_ii
    # plane_surface_identification["2\"x3\""] = plane_iii

    # print("\nAfter bundle adjustment, perpendicular error per plane followed by plane angle relative errors:")
    # for plane, identifier in zip([plane_i, plane_ii, plane_iii], ["1\"x2\"","1\"x3\"","2\"x3\""]):
    #     error_in_microns = plane.perpendicular_error() * 1000.0
    #     print("Plane {} has point-to-plane average error of {:.3f} μm".format(identifier, error_in_microns))

    # for plane_a_surface, plane_b_surface in [["1\"x2\"", "1\"x3\""], ["1\"x2\"", "2\"x3\""], ["1\"x3\"", "2\"x3\""]]:
    #     plane_a = plane_surface_identification[plane_a_surface]
    #     plane_b = plane_surface_identification[plane_b_surface]
    #     angle = angle_between(plane_a, plane_b)
    #     print("{} and {} planes: angle error of {:.5f} degrees relative to each other".format(plane_a_surface, plane_b_surface, angle))

    # max_point_to_plane_error = 20.0
    # for plane, identifier in zip([plane_i, plane_ii, plane_iii], ["1\"x2\"","1\"x3\"","2\"x3\""]):
    #     error_in_microns = plane.perpendicular_error() * 1000.0
    #     if error_in_microns > max_point_to_plane_error:
    #         print("Plane {} error of {:.3f} μm is outside of max tolerance of {:.1f} μm; quitting and moving on\n".format(identifier, error_in_microns, max_point_to_plane_error))
    #         return None, None, None




    min_angles = {}
    max_angles = {}

    min_angles["1\"x2\" by 1\"x3\""] = 0.0
    max_angles["1\"x2\" by 1\"x3\""] = 0.1
    min_angles["1\"x2\" by 2\"x3\""] = 0.0
    max_angles["1\"x2\" by 2\"x3\""] = 0.1
    min_angles["1\"x3\" by 2\"x3\""] = 0.0
    max_angles["1\"x3\" by 2\"x3\""] = 0.1
    print("")

    for plane_a_surface, plane_b_surface in [["1\"x2\"", "1\"x3\""], ["1\"x2\"", "2\"x3\""], ["1\"x3\"", "2\"x3\""]]:
        min_angle = min_angles["{} by {}".format(plane_a_surface, plane_b_surface)]
        max_angle = max_angles["{} by {}".format(plane_a_surface, plane_b_surface)]
        plane_a = plane_surface_identification[plane_a_surface]
        plane_b = plane_surface_identification[plane_b_surface]
        angle = angle_between(plane_a, plane_b)
        #print("{} and {} planes: angle error of {:.5f} degrees relative to each other".format(plane_a_surface, plane_b_surface, angle))
        #print("\n{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(angle, plane_a.a, plane_b.a, plane_a.b, plane_b.b, plane_a.c, plane_b.c, plane_a.d, plane_b.d))
        if angle < min_angle or angle > max_angle:
            print("Angle {:.3f} degrees outside of (min,max) bounds ({:.2f},{:.2f}); throwing out data, moving on\n".format(angle, min_angle, max_angle))
            return None, None, None

    p1 = plane_surface_identification["1\"x2\""] 
    p2 = plane_surface_identification["2\"x3\""]
    p3 = plane_surface_identification["1\"x3\""]

    abstract_1 = -1 * p3.d - (p3.c / (p1.c - p2.c)) * (p2.d - p1.d)
    abstract_2 = -1 * p3.a - p3.c * (p2.a - p1.a) / (p1.c - p2.c)
    abstract_3 = p3.b + p3.c * (p2.b - p1.b) / (p1.c - p2.c)
    abstract_4 = p1.c / (p1.c - p2.c)
    abstract_5 = abstract_4 * (p2.a - p1.a)
    abstract_6 = abstract_4 * (p2.b - p1.b)
    abstract_7 = abstract_4 * (p2.d - p1.d)
    abstract_8 = abstract_6 * abstract_1 / abstract_3
    abstract_9 = abstract_6 * abstract_2 / abstract_3
    abstract_10 = p1.b * abstract_1 / abstract_3
    abstract_11 = p1.b * abstract_2 / abstract_3

    x_s = (-1 * abstract_10 - abstract_8 - abstract_7 - p1.d) / (p1.a + abstract_11 + abstract_5 + abstract_9)
    y_s = (abstract_1 + abstract_2 * x_s) / abstract_3
    z_s = (x_s * (p2.a - p1.a) + y_s * (p2.b - p1.b) + (p2.d - p1.d)) / (p1.c - p2.c)

    print("\nIntersection of 3 planes calculated at (x,y,z)=({:.4f},{:.4f},{:.4f})".format(round(x_s,5), round(y_s,5), round(z_s,5)))

    def point_at_distance_from_origin_along_plane_intersection(plane_1_surface, plane_2_surface, origin, distance):
        p1 = plane_surface_identification[plane_1_surface]
        p2 = plane_surface_identification[plane_2_surface]

        abstract_1 = (p2.a / p1.a) * (-1 * p1.b) + p2.b
        abstract_2 = (p2.a * p1.d / p1.a) - p2.d
        abstract_3 = (p2.a / p1.a) * p1.c - p2.c
        abstract_4 = abstract_2 / abstract_1
        abstract_5 = abstract_3 / abstract_1
        abstract_6 = -1 * (p1.d / p1.a) - (p1.b / p1.a) * abstract_4
        abstract_7 = -1 * (p1.c / p1.a) - (p1.b / p1.a) * abstract_5
        abstract_8 = abstract_6 - origin.x
        abstract_9 = abstract_4 - origin.y
        abstract_10 = abstract_8 ** 2 + abstract_9 ** 2 + origin.z ** 2
        abstract_11 = 2 * (abstract_8 * abstract_7 + abstract_9 * abstract_5 - origin.z)
        abstract_12 = abstract_7 ** 2 + abstract_5 ** 2 + 1
        abstract_13 = abstract_10 - distance ** 2

        t_possibility_1 = (-1 * abstract_11 + math.sqrt(abstract_11**2 - 4 * abstract_12 * abstract_13)) / (2 * abstract_12)
        t_possibility_2 = (-1 * abstract_11 - math.sqrt(abstract_11**2 - 4 * abstract_12 * abstract_13)) / (2 * abstract_12)

        y_possibility_1 = (p1.d * (p2.a / p1.a) + t_possibility_1 * ((p2.a / p1.a) * p1.c - p2.c) - p2.d) /  ((p2.a / p1.a) * -1 * p1.b + p2.b)
        y_possibility_2 = (p1.d * (p2.a / p1.a) + t_possibility_2 * ((p2.a / p1.a) * p1.c - p2.c) - p2.d) /  ((p2.a / p1.a) * -1 * p1.b + p2.b)

        x_possibility_1 = (-1 * p1.d - p1.c * t_possibility_1 - p1.b * y_possibility_1) / p1.a
        x_possibility_2 = (-1 * p1.d - p1.c * t_possibility_2 - p1.b * y_possibility_1) / p1.a

        z_possibility_1 = t_possibility_1
        z_possibility_2 = t_possibility_2

        # get a few sample points to reference position of points in plane relative to origin
        sample_point_from_p1 = random.sample(p1.points, 1)[0]
        sample_point_from_p2 = random.sample(p1.points, 1)[0]
        sampled_x = (sample_point_from_p1.x + sample_point_from_p2.x) / 2.0
        sampled_y = (sample_point_from_p1.y + sample_point_from_p2.y) / 2.0
        sampled_z = (sample_point_from_p1.z + sample_point_from_p2.z) / 2.0
        sampled_point = Point(sampled_x, sampled_y, sampled_z)

        possible_point_1 = Point(x_possibility_1, y_possibility_1, z_possibility_1)
        possible_point_2 = Point(x_possibility_2, y_possibility_2, z_possibility_2)

        sampled_distance_from_possibility_1 = possible_point_1.distance(sampled_point)
        sampled_distance_from_possibility_2 = possible_point_2.distance(sampled_point)

        # print("\nEquation for line at the intersection of {} and {} planes:".format(plane_1_surface, plane_2_surface))
        # print("{:.2f}(x-{:.2f}) + {:.2f}(y-{:.2f}) + {:.2f}(z-{:.2f}) = 0".format(a, origin.x, b, origin.y, c, origin.z))

        if sampled_distance_from_possibility_1 < sampled_distance_from_possibility_2:
            x = possible_point_1.x
            y = possible_point_1.y
            z = possible_point_1.z
            distance_from_origin = round(possible_point_1.distance(origin) / 25.4, 1)
        else:
            x = possible_point_2.x
            y = possible_point_2.y
            z = possible_point_2.z
            distance_from_origin = round(possible_point_2.distance(origin) / 25.4, 1)

        print("Point ({:.2f},{:.2f},{:.2f}) is {:.4f} inches from ({:.4f},{:.4f},{:.2f}) along the intersection of the {} and {} surfaces".format(x,y,z,distance_from_origin,origin.x, origin.y, origin.z, plane_1_surface, plane_2_surface))

        return Point(x,y,z)


    def parallel_plane_from_point_to_point(existing_surface, point_on_existing_plane, point_on_new_plane):
        existing_plane = plane_surface_identification[existing_surface]
        new_d = -1 * (existing_plane.a * point_on_new_plane.x + existing_plane.b * point_on_new_plane.y + existing_plane.c * point_on_new_plane.z)
        parallel_plane = BlockPlane(a=existing_plane.a, b=existing_plane.b, c=existing_plane.c, d=new_d)
        translation = Vector(point_on_new_plane.x - point_on_existing_plane.x, point_on_new_plane.y - point_on_existing_plane.y, point_on_new_plane.z - point_on_existing_plane.z)
        parallel_plane.points = [Point(p.x + translation.x, p.y + translation.y , p.z + translation.z) for p in existing_plane.points]
        print("\nFrom {} plane with original equation {:.2f}x + {:.2f}y + {:.2f}z + {:.2f} = 0...".format(existing_surface, existing_plane.a, existing_plane.b, existing_plane.c, existing_plane.d))
        print("Given that ({:.2f},{:.2f},{:.2f}) exists in a parallel plane, equation for plane {} opposite derived: {:.2f}x + {:.2f}y + {:.2f}z + {:.2f} = 0 (i.e. same slope, different intercept)".format(point_on_new_plane.x, point_on_new_plane.y, point_on_new_plane.z, surface, parallel_plane.a, parallel_plane.b, parallel_plane.c, parallel_plane.d))
        return parallel_plane

    point_0 = Point(x_s, y_s, z_s)
    point_1 = point_at_distance_from_origin_along_plane_intersection(plane_1_surface="1\"x3\"", plane_2_surface="1\"x2\"", origin=point_0, distance=1 * 25.4)
    point_2 = point_at_distance_from_origin_along_plane_intersection(plane_1_surface="2\"x3\"", plane_2_surface="1\"x2\"", origin=point_0, distance=2 * 25.4)
    point_3 = point_at_distance_from_origin_along_plane_intersection(plane_1_surface="2\"x3\"", plane_2_surface="1\"x3\"", origin=point_0, distance=3 * 25.4)

    plane_1x2_opposite = parallel_plane_from_point_to_point(existing_surface="1\"x2\"", point_on_existing_plane=point_0, point_on_new_plane=point_3)
    plane_1x3_opposite = parallel_plane_from_point_to_point(existing_surface="1\"x3\"", point_on_existing_plane=point_0, point_on_new_plane=point_2)
    plane_2x3_opposite = parallel_plane_from_point_to_point(existing_surface="2\"x3\"", point_on_existing_plane=point_0, point_on_new_plane=point_1)

    plane_surface_identification["1\"x2\" opposite"] = plane_1x2_opposite
    plane_surface_identification["1\"x3\" opposite"] = plane_1x3_opposite
    plane_surface_identification["2\"x3\" opposite"] = plane_2x3_opposite

    point_4 = point_at_distance_from_origin_along_plane_intersection(plane_1_surface="1\"x2\" opposite", plane_2_surface="1\"x3\"", origin=point_3, distance=1 * 25.4)
    point_5 = point_at_distance_from_origin_along_plane_intersection(plane_1_surface="1\"x2\" opposite", plane_2_surface="2\"x3\"", origin=point_3, distance=2 * 25.4)
    point_6 = point_at_distance_from_origin_along_plane_intersection(plane_1_surface="1\"x2\" opposite", plane_2_surface="1\"x3\" opposite", origin=point_5, distance=1 * 25.4)
    point_7 = point_at_distance_from_origin_along_plane_intersection(plane_1_surface="1\"x2\"", plane_2_surface="1\"x3\" opposite", origin=point_2, distance=1 * 25.4)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    names = ["point_0", "point_1", "point_2", "point_3", "point_4", "point_5", "point_6", "point_7"]

    all_block_points = [point_0, point_1, point_2, point_3, point_4, point_5, point_6, point_7]

    plane_colors = ['r','g','b']
    xs = []
    ys = []
    zs = []
    colors = []
    markers = []

    block_point_colors = ['yellow' for i in range(len(all_block_points))]
    for block_point_index, block_point in enumerate(all_block_points):
        xs.append(block_point.x)
        ys.append(block_point.y)
        zs.append(block_point.z)
        colors.append(block_point_colors[block_point_index])
        #markers.append('o')

    for plane_number, plane in enumerate([p1,p2,p3]):
        for point in plane.points[:1000]:
            xs.append(point.x)
            ys.append(point.y)
            zs.append(point.z)
            colors.append(plane_colors[plane_number])
            #markers.append('o')

    ax.scatter(xs, ys, zs, c=colors, marker='o', ) 

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    for i, txt in enumerate(names):
        ax.text(xs[i], ys[i], zs[i], txt, None)

    print("")

    with open("{}/{}_corner_points.txt".format(point_cloud_directory, project_scan_name), "w") as output_file:
        for point, name in zip(all_block_points, names):
            print("{} = Point({:.5f},{:.5f},{:.5f})".format(name, point.x, point.y, point.z))
            output_file.write("{:.5f},{:.5f},{:.5f}\n".format(point.x, point.y, point.z))

    print("\nSaving visualization of 8 corners points to {}/{}_corner_point_visualization.png".format(point_cloud_directory, project_scan_name))

    fig = plt.gcf()
    fig.set_size_inches(40.0, 40.0)
    plt.savefig("{}/{}_corner_point_visualization.png".format(point_cloud_directory, project_scan_name), dpi=100)

    ray.shutdown()

    return xs, ys, zs


def find_circle_center(x, y):
    x_m = np.mean(x)
    y_m = np.mean(y)
    global counts
    counts = 0

    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    def f_2(c):
        global counts
        counts += 1 
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = x_m, y_m
    center_2, ier = optimize.leastsq(f_2, center_estimate)

    xc_2, yc_2 = center_2
    Ri_2       = calc_R(xc_2, yc_2)
    R_2        = Ri_2.mean()
    average_error_in_point_distance_from_center = sum(abs(Ri_2 - R_2)) / len(x)
    # ncalls_2   = f_2.ncalls

    print("({},{}) = center of circle (after {} calls)".format(xc_2, yc_2, counts))
    print("Average radius error for yaw rotation around center: {}mm".format(average_error_in_point_distance_from_center))

    return [xc_2, yc_2, R_2]

def roll_correction(roll, x, y, z):
    # roll an angle in degrees
    # x, y, z are each numpy arrays of float values
    roll = math.radians(roll)
    roll_origin = np.arctan2(x, y)
    radius = np.sqrt(x**2 + y**2)
    x = radius * np.sin(roll + roll_origin)
    y = radius * np.cos(roll + roll_origin)
    z = z
    return x,y,z

def pitch_correction(pitch, x, y, z):
    # pitch an angle in degrees
    # x, y, z are each numpy arrays of float values
    pitch = math.radians(pitch)
    pitch_origin = np.arctan2(x, z)
    radius = np.sqrt(x**2 + z**2)
    x = radius * np.sin(pitch + pitch_origin)
    y = y
    z = radius * np.cos(pitch + pitch_origin)
    return x,y,z
    
def yaw_correction(yaw, x, y, z):
    # yaw an angle in degrees
    # x, y, z are each numpy arrays of float values
    yaw = math.radians(yaw)
    yaw_origin = np.arctan2(z, y)
    radius = np.sqrt(y**2 + z**2)
    x = x
    y = radius * np.cos(yaw + yaw_origin)
    z = radius * np.sin(yaw + yaw_origin)
    return x,y,z

