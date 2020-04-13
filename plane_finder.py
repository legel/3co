import numpy as np
import random
import ray
import time
import logging
import sys
import scipy.optimize as optimize
import multiprocessing
from PIL import Image
import math
from itertools import zip_longest, combinations

point_cloud_directory = sys.argv[1].split("/")[0:-1][0]
point_cloud_filename = sys.argv[1].split("/")[-1]
point_cloud_identifier = point_cloud_filename.replace("_hvxyz.csv", "")

# set parameters 
maximum_planes_to_fit = 4
maximum_point_to_plane_error_in_microns = 50
maximum_number_of_trials = 1000
minimum_points_per_plane = 10000
point_to_plane_distance_threshold = 0.25 # maximum planarity error when adding new points, in millimeters
outlier_to_remove_for_this_plane_threshold = point_to_plane_distance_threshold * 10
minimum_total_planar_parameter_distance = 0.50 * 4
new_points_per_plane_equation_update = 1
plane_milestones = [10**exp for exp in range(2,7)] 
best_planes = []

# launch parallel processing engine
cpus = multiprocessing.cpu_count() 
print("Launching plane finding algorithm on {} CPUs".format(cpus))
ray.init(num_cpus=cpus, logging_level=logging.WARNING)
np.seterr(divide='ignore', invalid='ignore')
total_start_time = time.time()
sys.stdout.flush()

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
    def __init__(self, points = []):
        self.points_h = []
        self.points_v = []
        self.points = np.zeros((len(points), 3))
        for i, point in enumerate(points):
            self.points[i][0] = point.x
            self.points[i][1] = point.y
            self.points[i][2] = point.z
            self.points_h.append(point.h)
            self.points_v.append(point.v)
        self.estimate_initial_parameters_with_first_three_points()
        
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
        planarity = abs( self.a * point[0] + self.b * point[1] + self.c * point[2] + self.d )
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
        sol = optimize.minimize(model, initial_guess, args=[x, y, z], constraints=cons)
        self.a = tuple(sol.x)[0]
        self.b = tuple(sol.x)[1]
        self.c = tuple(sol.x)[2]
        self.d = tuple(sol.x)[3]
        return tuple(sol.x)
    
    def perpendicular_error(self):
        x, y, z = self.points[:, 0], self.points[:, 1], self.points[:, 2]
        length_squared = self.a**2 + self.b**2 + self.c**2
        if length_squared > 0:
            return ((self.a * x + self.b * y + self.c * z + self.d) ** 2 / length_squared).sum()
        else:
            return "N/A"
    
    def unique_from(self, other_planes, minimum_total_planar_parameter_distance):
        # check if this plane has already been found with slight perturbation, e.g.
        # self        -0.5074x + -0.0269y + 0.8613z = -393.6713
        # other_plane  0.5074x + 0.0267y + -0.8613z = 393.6643
        # print("Total other planes: {}".format(len(other_planes)))
        # print("UNIQUENESS:")
        # print("a1={},b1={},c1={},d1={}".format(self.a, self.b, self.c, self.d))
        # print("a2={},b2={},c2={},d2={}".format(other_planes[0].a, other_planes[0].b, other_planes[0].c, other_planes[0].d))

        for other_plane in other_planes:
            a_distance = abs( abs(self.a) - abs(other_plane.a) )
            b_distance = abs( abs(self.b) - abs(other_plane.b) )
            c_distance = abs( abs(self.c) - abs(other_plane.c) )
            d_distance = abs( abs(self.d) - abs(other_plane.d) )
            if a_distance + b_distance + c_distance + d_distance < minimum_total_planar_parameter_distance:
                return False
        return True

    def combine_planes(self, other_plane):
        self.points = np.vstack((self.points, other_plane.points))
        self.points_h.extend(other_plane.points_h)
        self.points_v.extend(other_plane.points_v)
        #self.points_h = np.vstack((self.points_h, other_plane.points_h))
        #self.points_v = np.vstack((self.points_v, other_plane.points_v))
        self.optimize_plane_perpendicular_distance_to_points()

def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def read_point_cloud(point_cloud_filename):
    def file_len(fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    point_cloud = []
    total_points = file_len(point_cloud_filename)
    print("Loading {:,} points from {}".format(total_points, point_cloud_filename))
    sys.stdout.flush()
    invalid_points = 0
    with open(point_cloud_filename, "r") as lines: # block_z_plus_3_cm_hvxyz.csv
        for i, line in enumerate(lines):
            row,column,x,y,z = line.rstrip("\n").split(",")
            point = Point(float(x), float(y), float(z), int(row), int(column))
            if point.valid:
                point_cloud.append(point)
            else:
                invalid_points += 1
    print("Removed {:,} invalid points (no data from scanner)".format(invalid_points))
    return point_cloud

def sample_n_consecutive_points_without_replacement(n, set_of_points): 
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

@ray.remote
def add_points_in_plane(sampled_points, potential_points_to_add, best_planes):
    new_points_per_plane_equation_update = 1
    plane_milestones = [10**exp for exp in range(2,7)]
    sub_plane = Plane(sampled_points) 
    points_to_add_to_plane = []
    points_to_remove_from_set = []
    total_outliers_for_this_plane_removed = 0
    outlier_to_remove_for_this_plane_threshold = point_to_plane_distance_threshold * 10
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

point_cloud = read_point_cloud(point_cloud_filename)
remaining_points_across_all_planes_indices = random.sample(range(len(point_cloud)), len(point_cloud) - 1)
remaining_points_across_all_planes_indices.sort()
remaining_points_across_all_planes = [point_cloud[i] for i in remaining_points_across_all_planes_indices]
random.shuffle(remaining_points_across_all_planes) # reshuffle sampling so distributed processing is less correlated to sample

for trial_number in range(maximum_number_of_trials):
    if len(remaining_points_across_all_planes) < minimum_points_per_plane or len(best_planes) >= maximum_planes_to_fit:
        break
    print("\nTrial {} with {:,} points remaining, {} unique planes currently found".format(trial_number+1, len(remaining_points_across_all_planes), len(best_planes)))    
    sys.stdout.flush()

    sampled_points, remaining_points_for_this_trial  = sample_n_consecutive_points_without_replacement(n=3, set_of_points=remaining_points_across_all_planes)
    random.shuffle(remaining_points_for_this_trial) # keep shuffling each iteration, to continue to break correlations 

    current_plane = Plane(sampled_points)
    total_initial_points = len(remaining_points_for_this_trial)
    total_outliers_for_this_plane_removed = 0

    # loop through all other points and measure fit for this plane
    points_per_cpu = math.ceil(len(remaining_points_for_this_trial) / cpus)
    points_to_process_for_cpus = grouper(points_per_cpu, remaining_points_for_this_trial)
    parallel_cpu_computations = [add_points_in_plane.remote(sampled_points, points_to_process_for_cpu, best_planes) for points_to_process_for_cpu in points_to_process_for_cpus]
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
        if current_plane.get_planarity(point) > point_to_plane_distance_threshold:
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
    if len(current_plane.points) > minimum_points_per_plane:   
        final_error = current_plane.perpendicular_error()
        error_per_point = final_error / float(len(current_plane.points))
        if error_per_point * 1000 < maximum_point_to_plane_error_in_microns: # ensure this plane is an excellent overall fit
            is_unique = True
            for best_plane in best_planes:
                if not current_plane.unique_from([best_plane], minimum_total_planar_parameter_distance): # ensure this plane is sufficiently unique
                    print("  -> This plane is not unique, combining with one of our existing planes")
                    is_unique = False
                    best_plane.combine_planes(current_plane) # allow another plane to absorb this one
                    break
            if is_unique:
                print("  -> This plane is unique, adding to list of best planes")
                best_planes.append(current_plane) # otherwise create a new plane
        else:
            print("  -> This plane is not precise enough for us to save, tossing it out")
    else:
        print("  -> Sufficiently large plane not found, moving on")

    # regardless, let's not reconsider the points from this plane ever again
    remaining_points_across_all_planes = remaining_points_for_this_trial

# after all plane fitting has finished
# remove points from smaller planes that could very well belong to the largest plane
print("")
for combo in combinations( range(len(best_planes)), 2):
    plane_i = best_planes[combo[0]]
    plane_ii = best_planes[combo[1]]

    # remove points in plane i that are too close in planarity to points in plane ii 
    indices_in_point_i_to_remove = []
    for point_number, point in enumerate(plane_i.points):
        if plane_ii.get_planarity(point) < point_to_plane_distance_threshold * 7.5:
            indices_in_point_i_to_remove.append(point_number)
    print("Removing {} points from plane {} that are too close in planarity to plane {}".format(len(indices_in_point_i_to_remove), combo[0], combo[1]))
    plane_i.remove_points_by_indices(indices_in_point_i_to_remove)
    plane_i.optimize_plane_perpendicular_distance_to_points()

    # and do so in reverse, just in case
    indices_in_point_ii_to_remove = []
    for point_number, point in enumerate(plane_ii.points):
        if plane_i.get_planarity(point) < point_to_plane_distance_threshold * 7.5:
            indices_in_point_ii_to_remove.append(point_number)
    print("Removing {} points from plane {} that are too close in planarity to plane {}".format(len(indices_in_point_ii_to_remove), combo[1], combo[0]))
    plane_ii.remove_points_by_indices(indices_in_point_ii_to_remove)
    plane_ii.optimize_plane_perpendicular_distance_to_points()

plane_colors = [[0,0,0]]
print("\nBEST PLANES:")
for plane_number, plane in enumerate(best_planes):
    red = int(random.uniform(0,1) * 255)
    green = int(random.uniform(0,1) * 255)
    blue = int(random.uniform(0,1) * 255)
    plane_colors.append([red, green, blue])
    print("\n({}): {:,} points with perpendicular error per point = {} microns".format(plane_number+1, len(plane.points), round(1000 * plane.perpendicular_error() / float(len(plane.points)), 1 ) ))
    print("{}x + {}y + {}z + {} = 0 with color ({},{},{})".format(round(plane.a,4), round(plane.b,4), round(plane.c,4), round(plane.d,4), red, green, blue ))
    
rows = 2280 # vertical pixels from scanner
columns = 1824 # horizontal pixels from scanner

# visualize planes as colored pixels overlaid on original perspective
img_abcd = Image.new('RGBA', (rows, columns), color = 'white')
pixels_abcd = img_abcd.load()
plane_fitted = np.full((rows, columns), 0, dtype='int')

for row in range(rows):
    for column in range(columns):
        pixels_abcd[row, column] = (255, 255, 255, 255)

a = np.full((rows, columns), 0.0)
b = np.full((rows, columns), 0.0)
c = np.full((rows, columns), 0.0)
d = np.full((rows, columns), 0.0)

for plane_index, plane in enumerate(best_planes):

    for point_index, point in enumerate(plane.points):
        column = plane.points_v[point_index]
        row = plane.points_h[point_index]
        plane_fitted[row, column] = plane_index + 1

print("\nSaving data visualization...")
for row in range(rows):
    for column in range(columns):
        plane_index = plane_fitted[row, column]
        red, green, blue = plane_colors[plane_index]
        pixels_abcd[row, columns - column - 1] = (red, green, blue, 255)

end_time = int(time.time())
plane_filename = "{}/{}_abcd.png".format(point_cloud_directory, point_cloud_identifier)
img_abcd.save('{}'.format(plane_filename))
print("Saved visualization to {}".format(plane_filename))

print("Saving output plane point data to {}_plane_data.csv".format(point_cloud_identifier))
with open("{}/{}_plane_data.csv".format(point_cloud_directory, point_cloud_identifier), "w") as output_file:
    for i, plane in enumerate(best_planes):
        error_per_point = round(1000 * plane.perpendicular_error() / float(len(plane.points)), 1 )
        if error_per_point < 4.0: # microns, probably removes the turntable and any other planes
            a = plane.a
            b = plane.b
            c = plane.c
            d = plane.d
            output_file.write("PLANE {}: a={},b={},c={},d={}\n".format(i,a,b,c,d))
            for i, point in enumerate(plane.points):
                h = plane.points_h[i]
                v = plane.points_v[i]
                x = point[0]
                y = point[1]
                z = point[2]
                output_file.write("{},{},{},{},{}\n".format(h,v,x,y,z))

total_end_time = time.time()
total_duration = round(total_end_time - total_start_time, 2)
print("\nTotal duration of plane finding algorithm on {} CPUs: {} seconds".format(cpus, total_duration))