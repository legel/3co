import numpy as np
from matplotlib import pyplot as plt
import optimizer.pyqt_fit.nonparam_regression as smooth
from optimizer.pyqt_fit import npr_methods

#print("Importing and fitting camera optics curves...")
polarization_lenses = True

# values to fit: e.g. camera_focuses (motor control value), as a function of distances
# distances =  			[195.0, 		341.945831,	361.776886,	381.738800,	421.888458,	502.716736,	665.006653,	985.424744,	1189.634888]
# camera_focuses = 		[72.0,			149.38,		150.98,		161.69,		172.73,		188.87,		210.04,		224.38,		228.64]
# projector_focuses = 	[14.0,			23.56,		24.21,		24.89,		26.40,		29.18,		31.20,		34.05,		35.33]
# exposure_times = 		[10.0,			14.0,		18.5,		20.5,		24.5,		34.5,		55.0,		117.5,		179.13]

# NEW
distances =  			[190.0,			195.0,		242.2,		292.6,		393.33,		421.888458,	502.716736,	665.006653,	985.424744]#341.945831,	361.776886,	381.738800,	421.888458,	502.716736,	665.006653,	985.424744,	1189.634888]
camera_focuses = 		[68.68,			72.0,		102.35,		119.26,		163,		172.73,		188.87,		210.04,		224.38]#149.38,		150.98,		161.69,		172.73,		188.87,		210.04,		224.38,		228.64]
projector_focuses = 	[13.4,			14.0,		19.86,		23.97,		28.45,		26.40,		29.18,		31.20,		34.05]#23.56,		24.21,		24.89,		26.40,		29.18,		31.20,		34.05,		35.33]
exposure_times = 		[35.0,			50.6,		66.4,		95.2,		161.3,		200.0,		250.0,		300.0,		350.0]#14.0,		18.5,		20.5,		24.5,		34.5,		55.0,		117.5,		179.13]

projector_f_stop = 2.8 # current guess, contact jeremy@ajile.ca to confirm exact value
projector_circle_of_confusion = 0.0350 # mm
projector_focal_length = 12.00 # mm

projector_x = -1.333130
projector_y = 121.736145
projector_z = -1.133850

class ScannerOptics():
	def __init__(self):
		self.fit_camera_focus_to_distance()
		self.fit_projector_focus_to_distance()
		self.fit_exposure_time_to_distance()

	def fit_camera_focus_to_distance(self):
		self.camera_focus_kernel = smooth.NonParamRegression(distances, camera_focuses, method=npr_methods.LocalPolynomialKernel(q=2))
		self.camera_focus_kernel.fit()

	def fit_projector_focus_to_distance(self):
		self.projector_focus_kernel = smooth.NonParamRegression(distances, projector_focuses, method=npr_methods.LocalPolynomialKernel(q=2))
		self.projector_focus_kernel.fit()

	def fit_exposure_time_to_distance(self):
		self.exposure_time_kernel = smooth.NonParamRegression(distances, exposure_times, method=npr_methods.LocalPolynomialKernel(q=2))
		self.exposure_time_kernel.fit()

	def plot_camera_focus_to_distance(self):
		grid = np.r_[min(distances):max(distances):1731.0j]
		plt.clf()
		plt.title("target distance vs. camera focus")
		plt.plot(distances, camera_focuses, 'o', alpha=0.5)
		plt.plot(grid, self.camera_focus_kernel(grid), 'y', linewidth=2)
		plt.ylabel("camera focus ring rotation (degrees)")
		plt.xlabel("distance (millimeters)")
		plt.savefig("{}_vs_{}.png".format("camera_focuses", "distances"))
		print("Camera angle (degrees) evaluated at 300mm, 400mm, 500mm: {}".format(self.camera_focus_kernel([300.0, 400.0, 500.0])))

	def plot_projector_focus_to_distance(self):
		grid = np.r_[min(distances):max(distances):1731.0j]
		plt.clf()
		plt.title("target distance vs. projector focus")
		plt.plot(distances, projector_focuses, 'o', alpha=0.5)
		plt.plot(grid, self.projector_focus_kernel(grid), 'y', linewidth=2)
		plt.ylabel("projector focus ring rotation (degrees)")
		plt.xlabel("distance (millimeters)")
		plt.savefig("{}_vs_{}.png".format("projector_focuses", "distances"))
		print("Projector angle (degrees) evaluated at 300mm, 400mm, 500mm: {}".format(self.projector_focus_kernel([300.0, 400.0, 500.0])))

	def plot_exposure_time_to_distance(self):
		grid = np.r_[min(distances):max(distances):1731.0j]
		plt.clf()
		plt.title("target distance vs. exposure times")
		plt.plot(distances, exposure_times, 'o', alpha=0.5)
		plt.plot(grid, self.exposure_time_kernel(grid), 'y', linewidth=2)
		plt.ylabel("exposure time (milliseconds)")
		plt.xlabel("distance (millimeters)")
		plt.savefig("{}_vs_{}.png".format("exposure_times", "distances"))
		print("Exposure time (ms) evaluated at 300mm, 400mm, 500mm: {}".format(self.exposure_time_kernel([300.0, 400.0, 500.0])))

	def camera_focus(self, distance):
		return self.camera_focus_kernel([distance])[0]

	def projector_focus(self, distance):
		return self.projector_focus_kernel([distance])[0]

	def exposure_time(self, distance):
		return self.exposure_time_kernel([distance])[0]

	def focus(self, distance):
		camera_focus = self.camera_focus(distance = distance)
		projector_focus = self.projector_focus(distance = distance)
		exposure_time = self.exposure_time(distance = distance)
		return [camera_focus, projector_focus, exposure_time]

	def depth_of_field(self, distance):
		if distance:
			minimum_in_focus_depth = distance * projector_focal_length**2 / (projector_focal_length**2 + projector_f_stop * projector_circle_of_confusion * distance)
			if distance > 1000.0:
				maximum_in_focus_depth = 3000.0
			else:
				maximum_in_focus_depth = distance * projector_focal_length**2 / (projector_focal_length**2 - projector_f_stop * projector_circle_of_confusion * distance)
		else:
			minimum_in_focus_depth = 0.0
			maximum_in_focus_depth = 10000.0
		return minimum_in_focus_depth, maximum_in_focus_depth


#   TO DO: REIMPLEMENT OPTICS TUNING
#  
# 	def find_optimal_focus_values_and_exposure_dynamic_range_for_given_apertures_and_distances(self, apertures=[0.0], distances=[0.0, 0.05, 0.10, 0.20, 0.40], initial_x=0.0, initial_y=0.0, initial_z=0.31, initial_pitch=-90, initial_yaw=0):

# 		for aperture in apertures:
# 			print("Setting camera aperture to {} degrees (where 90 degrees is maximum sized aperture with smallest depth of field, shortest exposure need; 0 degrees is minimum aperture, widest depth of field)".format(aperture))
# 			self.camera_aperture_position = aperture
# 			calibrate('camera_aperture')
# 			move({'camera_aperture': self.camera_aperture_position})

# 			#all_distances = []
# 			all_camera_distances = []
# 			all_projector_distances = []
# 			all_camera_focus_positions = []
# 			all_projector_focus_positions = []
# 			all_min_exposure_times = []

# 			for distance in distances:
# 				move({'x': initial_x, 'y': initial_y, 'z': initial_z + distance, 'pitch': initial_pitch, 'yaw': initial_yaw})
# 				calibrate('camera_focus')
# 				calibrate('projector_focus')

# 				time.sleep(10.0)

# 				print("Focusing optics to initial local optimum using pre-existing mapping for distance {:.3f} meters".format(initial_z + distance))

# 				self.focus_optics(distance = 1000 * (initial_z + distance + 0.15))
				
# 				self.scanner.StartSetup()
# 				min_exposure_time = self.find_min_exposure_time()
# 				optimum_projector_focus = self.find_local_optimum_for_projector_focus(coarse_search=False)
# 				optimum_camera_focus = self.find_local_optimum_for_camera_focus(coarse_search=False)
# 				min_exposure_time = self.find_min_exposure_time(coarse_search=False)
# 				# calibrate('camera_focus')
# 				# calibrate('projector_focus')
# 				# optimum_projector_focus = self.find_local_optimum_for_projector_focus()
# 				# optimum_camera_focus = self.find_local_optimum_for_camera_focus()
# 				# min_exposure_time = self.find_min_exposure_time()

# 				#max_exposure_time = self.find_max_exposure_time()
# 				#dual_exposure_multiplier = max_exposure_time / min_exposure_time

# 				point_cloud = self.scan(exposure_time=min_exposure_time, project_name="aperture{}_distance{}_exposure{}".format(aperture, distance + initial_z, min_exposure_time), distance_center_sample_size=100)
# 				remaining_points_in_roi = point_cloud.filter_roi_pixels()

# 				if remaining_points_in_roi > 0:
# 					camera_distance = point_cloud.get_average_3d_distance_from(x=0.0, y=0.0, z=0.0)
# 					projector_distance = point_cloud.get_average_3d_distance_from(x=projector_x, y=projector_y, z=projector_z)
# 					all_camera_distances.append(str(camera_distance))
# 					all_projector_distances.append(str(projector_distance))
# 					all_camera_focus_positions.append(str(optimum_camera_focus))
# 					all_projector_focus_positions.append(str(optimum_projector_focus))
# 					all_min_exposure_times.append(str(min_exposure_time))
# 				else:
# 					print("The scan for this path has no valid points remaining in ROI, so moving on to next...\n")

# 				self.scanner.StopSetup()

# 			with open("aperture_{}_degrees_v4.tsv".format(int(aperture)), "w") as o:
# 				o.write("camera_distances = [{}]\n".format(",".join(all_camera_distances)))
# 				o.write("camera_focuses = [{}]\n".format(",".join(all_camera_focus_positions)))
# 				o.write("projector_distances = [{}]\n".format(",".join(all_projector_distances)))
# 				o.write("projector_focuses = [{}]\n".format(",".join(all_projector_focus_positions)))
# 				o.write("exposure_times = [{}]\n".format(",".join(all_min_exposure_times)))

# 			self.die()
# 			sys.exit(0)


# 	def update_validation_metrics(self):
# 		self.saturation = self.get_saturation()
# 		self.validation = self.get_validation()
# 		print("Validated Pixels (In-Focus) = {:.2f} percent; Saturated Pixels = {:.2f} percent; ".format(self.validation, self.saturation))




	# def get_saturation(self, initial_sleep=0.25, samples=40, sleep_per_sample=0.0125):
	# 	time.sleep(initial_sleep)
	# 	all_saturated_pixels_count = 0
	# 	for sample in range(samples):
	# 		saturated_pixels = self.scanner.GetROISaturatedPixelCount()
	# 		all_saturated_pixels_count += saturated_pixels
	# 		time.sleep(sleep_per_sample)
	# 	average_saturated_pixels = all_saturated_pixels_count / float(samples)
	# 	percent_saturated = average_saturated_pixels / total_pixels * 100
	# 	self.update_view()
	# 	return percent_saturated

	# def get_validation(self, initial_sleep=0.25, samples=40, sleep_per_sample=0.0125):
	# 	time.sleep(initial_sleep)
	# 	all_valid_pixels_count = 0
	# 	for sample in range(samples):
	# 		valid_pixels = self.scanner.GetROIValidPointCount()
	# 		all_valid_pixels_count += valid_pixels
	# 		time.sleep(sleep_per_sample)
	# 	average_valid_pixels = all_valid_pixels_count / float(samples)
	# 	percent_validated = average_valid_pixels / total_pixels * 100
	# 	self.update_view()
	# 	return percent_validated	

	# def check_if_found_local_optimum(self, trends, minimum_trend_size = 6, positive_trend_threshold = 0.5):
	# 	found_local_optimum = False
	# 	if len(trends) >= minimum_trend_size+1:
	# 		average_trend = sum([trends[-i] - trends[-i-1] for i in range(1,minimum_trend_size+1)]) / minimum_trend_size
	# 		if average_trend > positive_trend_threshold:
	# 			found_local_optimum = True
	# 	return found_local_optimum

	# def check_if_leaving_local_optimum(self, trends, minimum_trend_size = 3, negative_trend_threshold = -0.5):
	# 	leaving_local_optimum = False
	# 	if len(trends) >= minimum_trend_size+1:
	# 		average_trend = sum([trends[-i] - trends[-i-1] for i in range(1,minimum_trend_size+1)]) / minimum_trend_size
	# 		if average_trend < negative_trend_threshold:
	# 			leaving_local_optimum = True
	# 	return leaving_local_optimum

	# def optimize_exposure_time(self, distance = None, step_size = 0.01, target_percent_saturated = 2.5):
	# 	print("Optimizing exposure time for current aperture...")
	# 	percent_saturated = self.get_saturation()
	# 	if (percent_saturated > target_percent_saturated):
	# 		while(percent_saturated > target_percent_saturated):
	# 			self.exposure_time = self.exposure_time - step_size
	# 			self.scanner.StopSetup()
	# 			self.scanner.SetExposureTimeMSec(self.exposure_time)
	# 			self.scanner.StartSetup()
	# 			percent_saturated = self.get_saturation()
	# 			print("{:.2f}% saturated after decreasing exposure to {:.2f} milliseconds".format(percent_saturated, self.exposure_time))
	# 	else:
	# 		while(percent_saturated < target_percent_saturated):
	# 			percent_saturated = self.get_saturation()
	# 			self.exposure_time = self.exposure_time + step_size
	# 			self.scanner.StopSetup()
	# 			self.scanner.SetExposureTimeMSec(self.exposure_time)
	# 			self.scanner.StartSetup()
	# 			percent_saturated = self.get_saturation()
	# 			print("{:.2f}% saturated after increasing exposure to {:.2f} milliseconds".format(percent_saturated, self.exposure_time))

	# def optimize_camera_aperture(self, distance = None, step_size = 0.1, target_percent_saturated = 0.01):
	# 	print("Optimizing camera aperture...")
	# 	percent_saturated = self.get_saturation()
	# 	if (percent_saturated > target_percent_saturated):
	# 		while(percent_saturated > target_percent_saturated):
	# 			percent_saturated = self.get_saturation()
	# 			self.camera_aperture_position = self.camera_aperture_position - step_size
	# 			move({'camera_aperture': self.camera_aperture_position})
	# 			print("{:.2f}% saturated for camera aperture position at {:.2f} degrees".format(percent_saturated, self.camera_aperture_position))
	# 	else:
	# 		while(percent_saturated < target_percent_saturated):
	# 			percent_saturated = self.get_saturation()
	# 			self.camera_aperture_position = self.camera_aperture_position + step_size
	# 			move({'camera_aperture': self.camera_aperture_position})
	# 			print("{:.2f}% saturated for camera aperture position at {:.2f} degrees".format(percent_saturated, self.camera_aperture_position))

	# def optimize_projector_focus(self, distance = None, step_size = 0.1):
	# 	found_local_optimum = False
	# 	leaving_local_optimum = False
	# 	print("Optimizing projector focus...")
	# 	validated = self.get_validation()
	# 	self.max_validated = validated
	# 	trends = []
	# 	while True: #not leaving_local_optimum:
	# 		validated = self.get_validation()
	# 		trends.append(validated)
	# 		if found_local_optimum:
	# 			leaving_local_optimum = self.check_if_leaving_local_optimum(trends)
	# 		else:
	# 			found_local_optimum = self.check_if_found_local_optimum(trends)
	# 		self.projector_focus_position = self.projector_focus_position + step_size
	# 		move({'projector_focus': self.projector_focus_position})
	# 		print("{:.2f}% validated for projector focus position at {:.2f} degrees".format(validated, self.projector_focus_position))

	# def find_min_exposure_time(self, coarse_search = True, coarse_step = 1.0, coarse_minimum = 0.01, coarse_maximum = 400.0, initial_reduction_multiplier = 0.8, gap_multiplier_growth_rate = 0.02, initial_multiplier=1, target_minimum_saturation = 0.1):
	# 	# Exposure time is initialized at 0.0, and then iteratively stepped up to find the minimum ideal white exposure

	# 	#self.exposure_time * initial_multiplier
	# 	self.exposure_time = self.exposure_time * 0.75
	# 	print("Searching for minimum exposure time (white colors) by starting from {}x baseline value ({:.2f} ms) and then stopping when percent saturation is at target value".format(initial_multiplier, self.exposure_time))
	# 	#self.exposure_time = 0.01

	# 	self.scanner.StopSetup()
	# 	self.scanner.SetExposureTimeMSec(self.exposure_time)
	# 	self.scanner.StartSetup()
	# 	percent_saturated = 0.0 #self.get_saturation()
	# 	#validated = #self.get_validation()
	# 	#reduction_multiplier = initial_reduction_multiplier
	# 	while percent_saturated < target_minimum_saturation: #not leaving_local_optimum:
	# 		#if self.exposure_time < 1.0:
	# 		self.exposure_time = self.exposure_time + 0.5 #/ reduction_multiplier
	# 		# if new_exposure_time - self.exposure_time < 0.05:
	# 		# 	self.exposure_time = self.exposure_time + 0.05
	# 		# else:
	# 		# 	self.exposure_time = new_exposure_time
	# 		# #else:
	# 		#	self.exposure_time = self.exposure_time - step_size
	# 		self.scanner.StopSetup()
	# 		self.scanner.SetExposureTimeMSec(self.exposure_time)
	# 		self.scanner.StartSetup()
	# 		percent_saturated = self.get_saturation()
	# 		validated = self.get_validation()
	# 		print("{:.2f}% validated, {:.2f}% saturated for exposure time at {:.3f} ms".format(validated, percent_saturated, self.exposure_time))
	# 		# gap = 1.0 - reduction_multiplier
	# 		# reduction_multiplier = reduction_multiplier + gap * gap_multiplier_growth_rate
	# 		#reduction_multiplier = reduction_multiplier * reduction_multiplier_growth_rate

	# 	print("Found {}ms as minimum exposure time, breaking".format(self.exposure_time))
	# 	return self.exposure_time

	# def find_max_exposure_time(self, min_step_increase = 0.25, step_multiplier = 1.25, lower_multiplier=2.0, upper_multiplier=4.0, target_multiplier=2.5):

	# 	print("Searching for maximum exposure time, starting from {} baseline value and increasing to {}x baseline, which is {}".format(lower_multiplier, upper_multiplier, self.exposure_time))
	# 	initial_exposure_time = self.exposure_time

	# 	second_half = []
	# 	current_exposure = self.exposure_time * target_multiplier
	# 	current_step_size = min_step_increase
	# 	upper_limit = self.exposure_time * float(upper_multiplier)
	# 	if upper_limit > 400.0:
	# 		upper_limit = 400
	# 	while current_exposure < upper_limit:
	# 		second_half.append(current_step_size)
	# 		current_step_size = current_step_size * step_multiplier
	# 		current_exposure = current_exposure + current_step_size

	# 	first_half = []
	# 	current_exposure = self.exposure_time * target_multiplier
	# 	current_step_size = min_step_increase
	# 	while current_exposure > self.exposure_time * float(lower_multiplier):
	# 		first_half.append(current_step_size)
	# 		current_step_size = current_step_size * step_multiplier
	# 		current_exposure = current_exposure - current_step_size

	# 	first_half.reverse()
	# 	first_half.extend(second_half)
	# 	step_schedule = first_half

	# 	all_exposure_times = []
	# 	all_validations = []
	# 	all_saturations = []

	# 	self.exposure_time = self.exposure_time * lower_multiplier

	# 	self.scanner.StopSetup()
	# 	self.scanner.SetExposureTimeMSec(self.exposure_time)
	# 	self.scanner.StartSetup()
	# 	percent_saturated = self.get_saturation()
	# 	validated = self.get_validation()

	# 	for step_increase in step_schedule:
	# 		all_exposure_times.append(self.exposure_time)
	# 		percent_saturated = self.get_saturation()
	# 		validated = self.get_validation()
	# 		print("{:.2f}% validated, {:.2f}% saturated for exposure time at {:.3f} ms".format(validated, percent_saturated, self.exposure_time))
	# 		all_validations.append(validated)
	# 		if percent_saturated == 0.0:
	# 			all_saturations.append(100.0) # don't believe the hype
	# 		else:
	# 			all_saturations.append(percent_saturated)
	# 		self.exposure_time = self.exposure_time + step_increase
	# 		self.scanner.StopSetup()
	# 		self.scanner.SetExposureTimeMSec(self.exposure_time)
	# 		self.scanner.StartSetup()

	# 	weighted_saturation_and_validation_scores = [validation / saturation**(0.5) for saturation, validation in zip(all_saturations, all_validations)] # experimentally derived
	# 	max_weighted_saturation_and_validation_score = max(weighted_saturation_and_validation_scores)
	# 	max_weighted_saturation_and_validation_score_index = weighted_saturation_and_validation_scores.index(max_weighted_saturation_and_validation_score)
	# 	optimum_max_exposure_time = all_exposure_times[max_weighted_saturation_and_validation_score_index]
	# 	best_score_validation = all_validations[max_weighted_saturation_and_validation_score_index]
	# 	best_score_saturation = all_saturations[max_weighted_saturation_and_validation_score_index]

	# 	print("Best exposure validation of {:.2f}%, saturation of {:.2f}% for upper exposure time of {:.2f}; moving projector focus to that position, breaking".format(best_score_validation, best_score_saturation, optimum_max_exposure_time))
	# 	print("Resetting exposure to initial minimum white optimal value...")

	# 	self.scanner.StopSetup()
	# 	self.scanner.SetExposureTimeMSec(initial_exposure_time)
	# 	self.scanner.StartSetup()

	# 	self.exposure_time = initial_exposure_time

	# 	return optimum_max_exposure_time


	# def find_local_optimum_for_projector_focus(self, coarse_search = True, coarse_step = 1.0, coarse_minimum = 0, coarse_maximum = 40, min_step_increase = 0.25, step_multiplier=1.1, initial_multiplier_upper=5.0, initial_multiplier_lower=0.5):
	# 	# Projector focus is initialized to a much closer value than is likely optimum, and then iteratively stepped up to a maximum much further than likely optimum, maximum value is selected after
	# 	print("Recalibrating projector focus prior to searching for optimum value...")
	# 	calibrate('projector_focus')

	# 	if coarse_search:
	# 		step_schedule = []
	# 		for i in range(coarse_minimum, coarse_maximum):
	# 			step_schedule.append(coarse_step)
	# 	else:
	# 		print("Searching for optimum projector focus, starting from {} baseline value and increasing to {}x baseline, which is {}".format(initial_multiplier_lower, initial_multiplier_upper, self.projector_focus_position))
	# 		second_half = []
	# 		current_focus = self.projector_focus_position
	# 		current_step_size = min_step_increase
	# 		while current_focus < self.projector_focus_position * float(initial_multiplier_upper):
	# 			second_half.append(current_step_size)
	# 			current_step_size = current_step_size * step_multiplier
	# 			current_focus = current_focus + current_step_size

	# 		first_half = []
	# 		current_focus = self.projector_focus_position
	# 		current_step_size = min_step_increase
	# 		while current_focus > self.projector_focus_position * float(initial_multiplier_lower):
	# 			first_half.append(current_step_size)
	# 			current_step_size = current_step_size * step_multiplier
	# 			current_focus = current_focus - current_step_size

	# 		first_half.reverse()
	# 		first_half.extend(second_half)
	# 		step_schedule = first_half

	# 		print("Planning to move projector along following step schedule, favoring closer inspection of points near higher likelihood:\n{}\n".format(step_schedule))

	# 	start_projector_focus = self.projector_focus_position * float(initial_multiplier_lower)
	# 	all_projector_focus_values = []
	# 	all_validations = []
	# 	self.projector_focus_position = start_projector_focus
	# 	move({'projector_focus': self.projector_focus_position})

	# 	for step_increase in step_schedule:
	# 		all_projector_focus_values.append(self.projector_focus_position)
	# 		percent_saturated = self.get_saturation()
	# 		validated = self.get_validation()
	# 		print("{:.2f}% validated, {:.2f}% saturated for projector focus at {:.2f} degrees".format(validated, percent_saturated, self.projector_focus_position))
	# 		all_validations.append(validated)
	# 		self.projector_focus_position = self.projector_focus_position + step_increase
	# 		move({'projector_focus': self.projector_focus_position})

	# 	max_validation = max(all_validations)
	# 	max_validation_index = all_validations.index(max_validation)
	# 	optimum_projector_focus = all_projector_focus_values[max_validation_index]
	# 	self.projector_focus_position = optimum_projector_focus

	# 	print("Recalibrating projector focus prior to setting to optimum value...")
	# 	calibrate('projector_focus')
	# 	print("Found max validation of {:.2f}% for projector focus of {:.2f}; moving projector focus to that position, breaking".format(max_validation, optimum_projector_focus))
	# 	move({'projector_focus': self.projector_focus_position})
	# 	self.update_view()
	# 	self.update_validation_metrics()
	# 	return self.projector_focus_position


	# def find_local_optimum_for_camera_focus(self, coarse_search = True, coarse_step = 1.0, coarse_minimum = 50, coarse_maximum = 205, min_step_increase = 0.05, step_multiplier=1.05, upper_multiplier=10.0, lower_multiplier=0.9):
	# 	# Camera focus is initialized to a much closer value than is likely optimum, and then iteratively stepped up to a maximum much further than likely optimum, with more sampling of points near higher likelihood
	# 	print("Recalibrating camera focus prior to searching for optimum value...")
	# 	calibrate('camera_focus')

	# 	if coarse_search:
	# 		step_schedule = []
	# 		for i in range(coarse_minimum, coarse_maximum):
	# 			step_schedule.append(coarse_step)
	# 	else:
	# 		print("Searching for optimum camera focus, starting from {} baseline value and increasing to {}x baseline, which is {} degrees".format(lower_multiplier, upper_multiplier, self.camera_focus_position))
	# 		second_half = []
	# 		current_focus = self.camera_focus_position
	# 		current_step_size = min_step_increase
	# 		while current_focus < self.camera_focus_position * float(upper_multiplier):
	# 			second_half.append(current_step_size)
	# 			current_step_size = current_step_size * step_multiplier
	# 			current_focus = current_focus + current_step_size

	# 		first_half = []
	# 		current_focus = self.camera_focus_position
	# 		current_step_size = min_step_increase
	# 		while current_focus > self.camera_focus_position * float(lower_multiplier):
	# 			first_half.append(current_step_size)
	# 			current_step_size = current_step_size * step_multiplier
	# 			current_focus = current_focus - current_step_size

	# 		first_half.reverse()
	# 		first_half.extend(second_half)
	# 		step_schedule = first_half

	# 	print("Planning to move camera focus along following step schedule, favoring closer inspection of points near higher likelihood:\n{}\n".format(step_schedule))


	# 	start_camera_focus = self.camera_focus_position * float(lower_multiplier)
	# 	end_camera_focus = self.camera_focus_position * upper_multiplier
	# 	all_camera_focus_values = []
	# 	all_validations = []
	# 	self.camera_focus_position = start_camera_focus
	# 	move({'camera_focus': self.camera_focus_position})
	# 	max_camera_focus_position_to_break_at = 300.0

	# 	for step_increase in step_schedule:
	# 		all_camera_focus_values.append(self.camera_focus_position)
	# 		percent_saturated = self.get_saturation()
	# 		validated = self.get_validation()
	# 		print("{:.2f}% validated, {:.2f}% saturated for camera focus at {:.2f} degrees".format(validated, percent_saturated, self.camera_focus_position))
	# 		all_validations.append(validated)
	# 		self.camera_focus_position = self.camera_focus_position + step_increase
	# 		if self.camera_focus_position >= max_camera_focus_position_to_break_at:
	# 			break
	# 		move({'camera_focus': self.camera_focus_position})

	# 	max_validation = max(all_validations)
	# 	max_validation_index = all_validations.index(max_validation)
	# 	optimum_camera_focus = all_camera_focus_values[max_validation_index]
	# 	self.camera_focus_position = optimum_camera_focus
	# 	print("Recalibrating camera focus prior to setting to optimum value...")
	# 	calibrate('camera_focus')

	# 	print("Found max validation of {:.2f}% for camera focus of {:.2f}; moving camera focus to that position, breaking".format(max_validation, optimum_camera_focus))
	# 	move({'camera_focus': self.camera_focus_position})
	# 	self.update_view()
	# 	self.update_validation_metrics()
	# 	return self.camera_focus_position


	# def search_max_exposure_times(self, step_size = 0.5):
	# 	found_local_optimum = False
	# 	leaving_local_optimum = False
	# 	print("Searching for maximum exposure time needed (black colors) by starting from current exposure and growing until user Ctrl-C interruption...")
	# 	validated = self.get_validation()
	# 	self.max_validated = validated
	# 	trends = []
	# 	while True: #not leaving_local_optimum:
	# 		self.exposure_time = self.exposure_time + step_size
	# 		self.scanner.StopSetup()
	# 		self.scanner.SetExposureTimeMSec(self.exposure_time)
	# 		self.scanner.StartSetup()
	# 		percent_saturated = self.get_saturation()
	# 		validated = self.get_validation()
	# 		print("{:.2f}% validated, {:.2f}% saturated for exposure time at {:.2f} seconds".format(validated, percent_saturated))

	# def optimize_camera_focus(self, distance = None, step_size = 0.1):
	# 	found_local_optimum = False
	# 	leaving_local_optimum = False
	# 	print("Optimizing camera focus...")
	# 	validated = self.get_validation()
	# 	self.max_validated = validated
	# 	trends = []
	# 	while True: #not leaving_local_optimum:
	# 		validated = self.get_validation()
	# 		trends.append(validated)
	# 		if found_local_optimum:
	# 			leaving_local_optimum = self.check_if_leaving_local_optimum(trends)
	# 		else:
	# 			found_local_optimum = self.check_if_found_local_optimum(trends)	
	# 		self.camera_focus_position = self.camera_focus_position + step_size
	# 		move({'camera_focus': self.camera_focus_position})	
	# 		print("{:.2f}% validated for camera focus position at {:.2f} degrees".format(validated, self.camera_focus_position))

	# def update_view(self):
	# 	img = self.scanner.GetLatestSetupImage()
	# 	if img.width > 0 and img.height > 0 and len(img.imageData) > 0:
	# 		np_image = np.zeros(shape=(img.height, img.width, 1), dtype=np.uint8)
	# 		img.WriteToMemory(np_image)
	# 		left_top = (int(img.height/2-camera_roi_height/2), int(img.width/2-camera_roi_width/2))
	# 		right_bottom = (int(img.height/2+camera_roi_height/2), int(img.width/2+camera_roi_width/2))
	# 		image_cropped = np_image[left_top[0] : right_bottom[0], left_top[1] : right_bottom[1]]
	# 		#np_image = cv2.rectangle(np_image, left_top, right_bottom, (0,255,0), 3)
	# 		#image_scaled = cv2.resize(np_image, (1024, 1024))
	# 		scale_percent = 500
	# 		scale_width = int(image_cropped.shape[1] * scale_percent / 100)
	# 		scale_height = int(image_cropped.shape[0] * scale_percent / 100)
	# 		cropped_scaled = cv2.resize(image_cropped, (scale_width, scale_height))
	# 		cropped_scaled = cv2.equalizeHist(cropped_scaled)

	# 		#image_scaled = cv2.equalizeHist(np_image)
	# 		#cv2.imshow("Ajile3DImager Example", image_scaled)
	# 		cv2.imshow("Ajile3DImager ROI", cropped_scaled)
	# 		cv2.waitKey(5)
	# 	else:
	# 		pass

	# def die(self, wait=True, rebirth=True, reset_usb=True):
	# 	self.scanner.StopSystem()
	# 	time.sleep(3.0)
	# 	del self.scanner
	# 	import board
	# 	import digitalio
	# 	self.power = digitalio.DigitalInOut(board.C0) # connect via USB to power switch relay, which is connected to the C0 GPIO pin of the FT232H board
	# 	self.power.direction = digitalio.Direction.OUTPUT # set relay pin as output
	# 	self.power.value = False # Relay is active low, so when it's low, relay is open and scanner is powered OFF
	# 	print('Powered OFF for Scanner, waiting 10 seconds for all power to leave')
	# 	time.sleep(10) # wait for 10 seconds for the scanner to fully power OFF
	# 	self.power.value = True # Relay inactive, contact is closed and the scanner is powered ON
	# 	if wait:
	# 		print('Powered ON for Scanner, waiting 44 seconds for FPGA to boot up and come back online')
	# 		time.sleep(44) # wait for 10 seconds for the scanner to fully power OFF
	# 	else:
	# 		print('Powered ON for Scanner, moving forward')
	# 	if rebirth:
	# 		self.initialize_scanner()




if __name__ == "__main__":
	optics = ScannerOptics()
	for distance in range(200, 1500, 50):
		minimum_in_focus_depth, maximum_in_focus_depth = optics.depth_of_field(distance=distance)
		camera_focus = optics.camera_focus(distance = distance)
		projector_focus = optics.projector_focus(distance = distance)
		exposure_time = optics.exposure_time(distance = distance)
		print("Targeting {:.1f}mm away, minimum in-focus depth = {:.1f}mm, maximum in-focus depth = {:.1f}mm, camera focus at {} degrees, projector focus at {} degrees, exposure time at {} seconds".format(distance, minimum_in_focus_depth, maximum_in_focus_depth, camera_focus, projector_focus, exposure_time))
	optics.plot_exposure_time_to_distance()
	optics.plot_projector_focus_to_distance()
	optics.plot_camera_focus_to_distance()