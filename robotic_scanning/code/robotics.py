from ops import print_with_time
import time
import commander
from distributions import *
from scanner_optics import ScannerOptics
from scipy.stats import norm
import depthscan
from os import listdir, path, getcwd, remove
import numpy as np
from os import listdir
from hdr import *
from geometry import *
import subprocess
from color_correction import normalize_colors
from utils import quietly_execute

# old_stdout = sys.stdout # backup current stdout
# sys.stdout = open(os.devnull, "w")
# my_nasty_function()
# sys.stdout = old_stdout # reset old stdout

camera_coordinate_system = "camera_coordinate_system" 
if camera_coordinate_system == "projector_coordinate_system":
	camera_image_width = 1824
	camera_image_height = 2280
elif camera_coordinate_system == "camera_coordinate_system":
	camera_image_width = 2048
	camera_image_height = 2048	

camera_roi_width = 256
camera_roi_height = 32
total_pixels = camera_roi_width * camera_roi_width 


class Iris():
	def __init__(self, x=None, y=None, z=None, pitch=None, yaw=None, roll=None, reinitialize_focus=True, camera_aperture_position=10, camera_focus_position=0.0, projector_focus_position=0.0, move_positions=True, distance=None, auto_focus=False):
		#self.optics = ScannerOptics()
		self.focus_distance_min = 200
		self.focus_distance_max = 400
		self.focus_distance_mean = 270
		self.focus_distance_variance = 25
		self.focus_distance_samples = 25
		self.min_focus_information_gain = 0.99

		self.exposure_time_min = 10.0
		self.exposure_time_max = 375.0
		self.exposure_time_mean = 75.0
		self.exposure_time_variance = 50.0
		self.exposure_time_samples = 25
		self.min_exposure_information_gain = 0.99

		# Intelligence: Focus v. Exposure variational sampling
		# Topological surface model, penalizing deviation
		# Mesh every scan to find outlier points, and remove them; implement Rob's script finally


		self.focus_stack_distance = []
		self.focus_stack_masks = {}
		self.sets_of_focus_ranges = []


		self.sets_of_exposures = []
		self.sets_of_exposures_masks = []

		if x:
			self.x = x
		if y:	
			self.y = y
		if z:	
			self.z = z
		if pitch:
			self.pitch = pitch
		if yaw:
			self.yaw = yaw
		if roll:
			self.roll = roll

		self.initialized = False
		self.reinitialize_focus = reinitialize_focus

		self.distance = distance
		self.camera_focus_position = camera_focus_position
		self.projector_focus_position = projector_focus_position
		self.camera_aperture_position = camera_aperture_position
		self.move_positions = move_positions
		self.project_name = "x"
		if auto_focus:
			focus_distances, focus_distances_masks, focus_ranges = self.focus_sweep()
			print("Focusing on most informative distance: {}mm".format(focus_distances[0]))
			self.focus_optics(distance=focus_distances[0])

	def initialize_robot(self):
		self.optics = ScannerOptics()
		if self.reinitialize_focus:
			self.initialize_focus()

		if self.camera_focus_position != 0.0 and self.move_positions:
			commander.move({'camera_focus': self.camera_focus_position})
		
		if self.projector_focus_position != 0.0 and self.move_positions:
			commander.move({'projector_focus': self.projector_focus_position})

		if self.camera_aperture_position != 0.0 and self.move_positions:
			commander.move({'camera_aperture': self.camera_aperture_position})

		#self.scanner = Scanner()

		if type(self.distance) != type(None):
			self.focus_optics(distance)

		self.initialized = True
		self.calibrated = True

	def initialize_focus(self, ignore_aperture=False):
		#print("Initializing camera focus, projector focus to closest distance; aperture initializing to widest depth of field",flush=True)
		commander.calibrate('projector_focus')
		commander.calibrate('camera_focus')
		if not ignore_aperture:
			commander.calibrate('camera_aperture')


	def focus_optics(self, distance=None, aperture=None, camera_focus=None, projector_focus=None):
		if type(distance) != type(None):
			camera_focus, projector_focus, exposure_time = self.optics.focus(distance)

		camera_focus = max(camera_focus, 0.01)
		projector_focus = max(projector_focus, 0.01)
		# exposure_time = max(exposure_time, 0.01)

		# exposure_time = exposure_time * exposure_time_multiplier
		# self.scanner.StartSetup()

		# if type(user_defined_exposure_time) != type(None):
		# 	self.scanner.SetExposureTimeMSec(user_defined_exposure_time)
		# 	self.exposure_time = user_defined_exposure_time
		# elif exposure_time != self.exposure_time:
		# 	self.scanner.SetExposureTimeMSec(self.exposure_time)
		# 	self.exposure_time = exposure_time

		if aperture != None:
			if aperture != self.camera_aperture_position:
				start_calibration_time = time.time()
				commander.calibrate('camera_aperture')
				end_calibration_time = time.time()
				#print("Recalibrated camera aperture in {:.1f} seconds".format(end_calibration_time - start_calibration_time))
				commander.move({'camera_aperture': aperture})
				self.camera_aperture_position = aperture

		if projector_focus != self.projector_focus_position:
			if projector_focus < self.projector_focus_position:
				start_calibration_time = time.time()
				commander.calibrate('projector_focus')
				end_calibration_time = time.time()
				#print("Recalibrated projector focus in {:.1f} seconds, because requested focus is less than current focus, and backlash not yet measured".format(end_calibration_time - start_calibration_time))
			commander.move({'projector_focus': projector_focus})
			self.projector_focus_position = projector_focus

		if camera_focus != self.camera_focus_position:
			if camera_focus < self.camera_focus_position:
				start_calibration_time = time.time()
				commander.calibrate('camera_focus')
				end_calibration_time = time.time()
				#print("Recalibrated camera focus in {:.1f} seconds, because requested focus is less than current focus, and backlash not yet measured".format(end_calibration_time - start_calibration_time))
			commander.move({'camera_focus': camera_focus})
			self.camera_focus_position = camera_focus

		self.calibrated = False

		self.current_focus_distance = distance

		#print("Camera Aperture = {:.2f} degrees; Camera Focus = {:.2f} degrees; Projector Focus = {:.2f} degrees".format(self.camera_aperture_position, self.camera_focus_position, self.projector_focus_position))

		# self.scanner.StopSetup()

	def calibrate_axis(self, axis):
		print("Calibrating {}".format(axis))
		commander.calibrate(axis)
		if axis == 'x':
			self.x = 0.0
		if axis == 'y':
			self.y = 0.0
		if axis == 'z':
			self.z = 0.0
		if axis == 'pitch':
			self.pitch = 0.0
		if axis == 'yaw':
			self.yaw = 0.0

	def calibrate(self):
		start_time = time.time()
		self.move(z = 1.58)
		self.calibrate_axis('x')
		self.calibrate_axis('y')
		self.calibrate_axis('z')
		self.calibrate_axis('pitch')
		self.calibrate_axis('yaw')
		end_time = time.time()
		print("(x,y,z,pitch,yaw) calibrated in {:.1f} minutes".format((end_time-start_time)/60.0))

	def move(self, x=None, y=None, z=None, pitch=None, yaw=None, perspective=None, sleep_for_dampening_vibration=0.0):
		start_move_time = time.time()
		move_command = {}

		if type(x) != type(None):
			move_command['x'] = x
			self.x = x

		if type(y) != type(None):
			move_command['y'] = y
			self.y = y

		if type(z) != type(None):
			move_command['z'] = z
			self.z = z

		if type(pitch) != type(None):
			move_command['pitch'] = pitch
			self.pitch = pitch

		if type(yaw) != type(None):
			move_command['yaw'] = yaw
			self.yaw = yaw

		print("\nRobot move: {}\n".format(move_command))
		commander.move(move_command)

		if sleep_for_dampening_vibration > 0.0:
			print("Sleeping {:.1f} seconds to dampen vibration".format(sleep_for_dampening_vibration))
			time.sleep(sleep_for_dampening_vibration) # sleep off that vibration
		end_move_time = time.time()
		print("Finished moving in {:.3f} seconds".format(end_move_time-start_move_time))
		return 1

	def scan(self, auto_focus=True, auto_exposure=True, distance=None, exposure_time=None, exposure_time_multiplier=1.0, dual_exposure_multiplier=2.0, dual_exposure_mode=False, hdr_exposure_times=[], project_name="", filter_outliers=False, outliers_std_dev=0.0, distance_center_sample_size=4, process_point_cloud=True, scan_index=0, export_to_csv=False, export_to_npy=True, save_to_png=True, missing_data_as="black", red_led_current=1.0, green_led_current=1.0, blue_led_current=1.0, save_exr=False, photos_to_project="None", save_16_bit_image=False):
		self.project_name = project_name

		if not self.initialized:
			self.initialize_robot()

		if auto_focus and auto_exposure:
			focus_distances, focus_distances_masks, focus_ranges = self.focus_sweep()
			sets_of_exposures, sets_of_exposures_masks = self.exposure_sweep(focus_distances=focus_distances)
		else:
			focus_distances, focus_distances_masks, focus_ranges = self.focus_stack_distance, self.focus_stack_masks, self.sets_of_focus_ranges
			sets_of_exposures, sets_of_exposures_masks = self.sets_of_exposures, self.sets_of_exposures_masks
			if len(sets_of_exposures) == 0:
				sets_of_exposures = [[self.exposure_time_mean]]
			if len(focus_distances) == 0:
				focus_distances = [self.focus_distance_mean] 
			if len(focus_ranges) == 0:
				focus_ranges = [[self.focus_distance_min, self.focus_distance_max]] 

		if hdr_exposure_times:
			sets_of_exposures = [hdr_exposure_times]

		point_clouds_with_different_focuses = []
		for scan_index, (focus_distance, set_of_exposures, focus_range) in enumerate(zip(focus_distances, sets_of_exposures, focus_ranges)):
			self.focus_optics(distance=focus_distance)

			args = "dual_exposure_mode={}".format(dual_exposure_mode) + "&"
			args += "dual_exposure_multiplier={:.3f}".format(dual_exposure_multiplier) + "&"
			args += "focus_distance={:.3f}".format(focus_distance) + "&"
			args += "save_to_png={}".format(save_to_png) + "&"
			args += "missing_data_as={}".format(missing_data_as) + "&"
			args += "export_to_csv={}".format(export_to_csv) + "&"
			args += "outliers_std_dev={}".format(outliers_std_dev) + "&"
			args += "filter_outliers={}".format(filter_outliers) + "&"
			args += "distance_center_sample_size={}".format(distance_center_sample_size) + "&"
			args += "process_point_cloud={}".format(process_point_cloud) + "&"
			args += "hard_filter_min={}".format(focus_range[0]) + "&"
			args += "hard_filter_max={}".format(focus_range[1]) + "&"
			args += "scan_index={}".format(scan_index) + "&"
			args += "project_name={}".format(project_name) + "&"
			args += "red_led_current={}".format(red_led_current) + "&"
			args += "green_led_current={}".format(green_led_current) + "&"
			args += "blue_led_current={}".format(blue_led_current) + "&"
			args += "photos_to_project={}".format(photos_to_project) + "&"
			args += "save_16_bit_image={}".format(save_16_bit_image) + "&"
			
			scan_tag = "{}-{}".format(project_name, scan_index)

			point_clouds = []
			hdr_input_photos = []
			valid_exposure_times = []

			if len(set_of_exposures) > 1:
				hdr_scan_export_to_npy = False
			else:
				hdr_scan_export_to_npy = export_to_npy

			for hdr_scan_number, exposure_time in enumerate(set_of_exposures):
				scan_args = args + "relative_scan_index={}".format(hdr_scan_number) + "&"
				scan_args += "export_to_npy={}".format(hdr_scan_export_to_npy) + "&"
				scan_args += "exposure_time={:.3f}".format(exposure_time)
				sub_scan_tag = "{}-{}-{}".format(project_name,scan_index,hdr_scan_number)
				subprocess.call(["python3", "depthscan.py", scan_args])
				if process_point_cloud:
					if self.scan_was_successful(sub_scan_tag):
						point_clouds.append("{}.ply".format(sub_scan_tag))
						hdr_input_photos.append("{}.png".format(sub_scan_tag))
						valid_exposure_times.append(exposure_time)
					else:
						print("Scan Failed")
						continue
						#raise Exception("SCAN FAILED")
				else:
					hdr_input_photos.append("{}.png".format(sub_scan_tag))
					valid_exposure_times.append(exposure_time)
			try:					
				tonemapped_png, radiance_data = pngs_to_exr_to_png(project_name=scan_tag, photo_filenames=hdr_input_photos, exposure_times=valid_exposure_times, save_exr=save_exr)
				normalize_colors(tonemapped_png)
			except cv2.error as e:
				print("Error: {}\nContinuing.".format(e))
			
			if process_point_cloud:
				if len(point_clouds) == 1:
					cloud = PointCloud(filename=point_clouds[0], scan_index=scan_index)
				else:
					cloud = self.combine_point_clouds(point_clouds)
					cloud.set_colors_by_photo(tonemapped_png)
					cloud.set_hdr_colors_by_exr_data(radiance_data)
					cloud.average_xyz_positions_with_outlier_removal(new_scan_index=scan_index)
				cloud.export_to_npy(project_name=scan_tag, from_tensor=True)
				cloud.save_as_ply(filename="{}.ply".format(scan_tag), from_tensor=True)
				point_clouds_with_different_focuses.append("{}.ply".format(scan_tag))

		if process_point_cloud:
			focus_stacked_cloud = self.combine_point_clouds(point_clouds_with_different_focuses, from_object=False)
			focus_stacked_cloud.average_xyz_positions_with_outlier_removal(new_scan_index=0)
			if export_to_npy:
				focus_stacked_cloud.export_to_npy(project_name="{}_focus_stacked".format(project_name), from_tensor=True)
			focus_stacked_cloud.save_as_ply(filename="{}_focus_stacked.ply".format(project_name), from_tensor=True)
		else:
			focus_stacked_cloud = None

		# if distance == None:
		# 	if exposure_time:
		# 		self.exposure_time = exposure_time
		# 	minimum_in_focus_depth = self.focus_distance_min
		# 	maximum_in_focus_depth = self.focus_distance_max
		# else:
		# 	self.focus_optics(distance=distance)
		# 	minimum_in_focus_depth, maximum_in_focus_depth = self.optics.depth_of_field(distance=distance)

		self.clean_up()
		#point_cloud = self.scanner.scan(focus_distances=self.focus_stack_distance, focus_masks=self.focus_stack_masks, exposure_time=exposure_time, dual_exposure_mode=dual_exposure_mode, dual_exposure_multiplier=dual_exposure_multiplier, hdr_exposure_times=hdr_exposure_times, project_name=project_name, filter_outliers=filter_outliers, outliers_std_dev=outliers_std_dev, distance_center_sample_size=distance_center_sample_size, process_point_cloud=process_point_cloud, hard_filter_min=minimum_in_focus_depth, hard_filter_max=maximum_in_focus_depth, scan_index=scan_index, export_to_csv=export_to_csv, export_to_npy=export_to_npy)
		return focus_stacked_cloud

	def clean_up(self):
		try:
			os.remove("mask.png")
			os.remove("decodedRows.png")
			os.remove("decodedCols.png")
		except:
			pass

	def scan_was_successful(self, scan_tag, scan_directory="/home/sense/3cobot"):
		files_in_scan = [f for f in listdir(scan_directory) if isfile(join(scan_directory, f)) and scan_tag in f]
		rgb_image_saved = False
		ply_point_cloud_saved = False
		for f in files_in_scan:
			if ".png" in f:
				rgb_image_saved = True
			if ".ply" in f:
				ply_point_cloud_saved = True
		if rgb_image_saved and ply_point_cloud_saved:
			return True
		else:
			return False

	def combine_point_clouds(self, point_clouds, from_object=False):
		if not from_object:
			point_clouds = [PointCloud(filename=point_cloud, scan_index=i) for i,point_cloud in enumerate(point_clouds)]
		base_point_cloud = point_clouds[0]
		for other_point_cloud in point_clouds[1:]:
			base_point_cloud.add_point_cloud(other_point_cloud)
		base_point_cloud.reindex_into_tensor()
		return base_point_cloud

	def focus_sweep(self):
		start_time = time.time()

		scanner, camera_image_width, camera_image_height, camera_roi_width, camera_roi_height = depthscan.initialize_scanner()

		#scanner, camera_image_width, camera_image_height, camera_roi_width, camera_roi_height = depthscan.initialize_scanner()



		focus_distances_to_probe =	n_samples_from_distribution(n_samples=self.focus_distance_samples, 
																sample_lower_bound=self.focus_distance_min, 
																sample_upper_bound=self.focus_distance_max, 
																mean=self.focus_distance_mean, 
																variance=self.focus_distance_variance, 
																percentile_lower_bound=0.01, 
																percentile_upper_bound=0.99, 
																min_sample_distance_as_percent_of_mean=0.01)

		masks = np.zeros((camera_image_width, camera_image_height, len(focus_distances_to_probe)))

		scanner.SetExposureTimeMSec(self.exposure_time_mean)

		if self.exposure_time_mean <= 25.0:
			sleep_time_after_exposure =  0.1
		elif self.exposure_time_mean <= 50.0:
			sleep_time_after_exposure =  0.1
		elif self.exposure_time_mean <= 75.0:
			sleep_time_after_exposure =  0.33
		elif self.exposure_time_mean < 150.0:
			sleep_time_after_exposure =  0.50
		elif self.exposure_time_mean < 315.0:
			sleep_time_after_exposure =  1.10
		elif self.exposure_time_mean <= 400.0:
			sleep_time_after_exposure =  1.30

		measured_camera_view = False

		if not self.initialized:
			self.initialize_robot()

		for i, focus_distance in enumerate(focus_distances_to_probe): 
			#print("Focus distance: {}mm".format(focus_distance))
			self.focus_optics(distance=focus_distance)
			scanner.StartSetup()
			time.sleep(sleep_time_after_exposure) # wait until exposure is taken + metadata is recorded
			mask = scanner.GetLatestValidPointMask()
			# write mask to array
			np_mask = np.zeros(shape=(mask.height, mask.width, 1), dtype=np.uint8)
			mask.WriteToMemory(np_mask)
			np_mask = np.where(np_mask == 255.0, 1, 0)
			masks[:,:,i] = np_mask[:,:,0]
			# save mask as image
			#mask.WriteToFile("{}_{}mm_focus.png".format(self.project_name, int(focus_distance)))

			if abs(focus_distance - self.focus_distance_mean) < 50.0 and not measured_camera_view:
				scanner.DetectCameraView()
				first_row = scanner.GetCameraViewFirstRow()
				last_row = scanner.GetCameraViewLastRow()
				number_of_rows = scanner.GetCameraViewNumRows()
				#print("Camera view 1st row = {}, Last row = {}, Number of Rows = {}".format(first_row, last_row, number_of_rows))
				measured_camera_view = True

			scanner.StopSetup()

		end_time = time.time()
		if not measured_camera_view:
			first_row = 0
			last_row = 2047
			number_of_rows = 2048
		self.select_most_informative_masks(masks=masks, all_mask_labels=focus_distances_to_probe, first_row=first_row, last_row=last_row, operate_on="focus", relative_index=0)
		#print("Final focus distances (mm): {}".format(self.focus_stack_distance))
		#print("\nProbed {} focus values in {} seconds".format(len(focus_distances_to_probe), end_time - start_time))

		self.sets_of_focus_ranges = []
		number_of_focuses = len(self.focus_stack_distance)
		for i in range(number_of_focuses):
			if i == 0:
				min_focus = self.focus_distance_min
				if number_of_focuses > 1:
					max_focus = self.focus_stack_distance[i+1] + 150
				else:
					max_focus = self.focus_stack_distance[i] + 150
			elif i+1 == number_of_focuses:
				min_focus = self.focus_stack_distance[i-1] - 150
				max_focus = self.focus_distance_max
			else:
				min_focus = self.focus_stack_distance[i-1] - 150
				max_focus = self.focus_stack_distance[i+1] + 150
			self.sets_of_focus_ranges.append([min_focus, max_focus])

		return self.focus_stack_distance, self.focus_stack_masks, self.sets_of_focus_ranges

	def get_saturation(self, scanner, initial_sleep=0.01, samples=40, sleep_per_sample=0.0125):
		all_saturated_pixels_count = 0
		for sample in range(samples):
			saturated_pixels = scanner.GetROISaturatedPixelCount()
			all_saturated_pixels_count += saturated_pixels
			time.sleep(sleep_per_sample)
		average_saturated_pixels = all_saturated_pixels_count / float(samples)
		percent_saturated = average_saturated_pixels / total_pixels * 100
		return percent_saturated

	def exposure_sweep(self, focus_distances=[]):
		if len(focus_distances) == 0:
			focus_distances = [self.focus_distance_mean]

		start_time = time.time()
		scanner, camera_image_width, camera_image_height, camera_roi_width, camera_roi_height = depthscan.initialize_scanner()

		measured_camera_view = False

		if not self.initialized:
			self.initialize_robot()

		self.sets_of_exposures = []
		self.sets_of_exposures_masks = []
		self.all_masks = []
		for focus_index, focus_distance in enumerate(focus_distances):

			self.focus_optics(distance=focus_distance)
			exposure_times_to_probe =	n_samples_from_distribution(n_samples=self.exposure_time_samples, 
																	sample_lower_bound=self.exposure_time_min, 
																	sample_upper_bound=self.exposure_time_max, 
																	mean=self.exposure_time_mean, 
																	variance=self.exposure_time_variance, 
																	percentile_lower_bound=0.01, 
																	percentile_upper_bound=0.99, 
																	min_sample_distance_as_percent_of_mean=0.05)

			masks = np.zeros((camera_image_width, camera_image_height, len(exposure_times_to_probe)))

			set_of_exposures = []

			for exposure_time_index, exposure_time in enumerate(exposure_times_to_probe):
				#print("Exposure time: {}ms".format(exposure_time))
				scanner.SetExposureTimeMSec(exposure_time)

				if exposure_time <= 75.0:
					sleep_time_after_exposure =  0.33
				elif exposure_time < 150.0:
					sleep_time_after_exposure =  0.75
				elif exposure_time < 315.0:
					sleep_time_after_exposure = 1.5
				elif exposure_time <= 400.0:
					sleep_time_after_exposure = 2.0

				scanner.StartSetup()
				time.sleep(sleep_time_after_exposure) # wait until exposure is taken + metadata is recorded

				mask = scanner.GetLatestValidPointMask()

				# percent_saturation = self.get_saturation(scanner)
				# SetCameraFocusROI
				# GetROISaturatedPixelCount()

				# write mask to array
				np_mask = np.zeros(shape=(mask.height, mask.width, 1), dtype=np.uint8)
				mask.WriteToMemory(np_mask)
				np_mask = np.where(np_mask == 255.0, 1, 0)
				masks[:,:,exposure_time_index] = np_mask[:,:,0]
				# save mask as image
				#mask.WriteToFile("{}_{}mm_focus_{}_exposure.png".format(self.project_name, int(focus_distance), int(exposure_time)))

				if not measured_camera_view:
					scanner.DetectCameraView()
					first_row = scanner.GetCameraViewFirstRow()
					last_row = scanner.GetCameraViewLastRow()
					number_of_rows = scanner.GetCameraViewNumRows()
					#print("Camera view 1st row = {}, Last row = {}, Number of Rows = {}".format(first_row, last_row, number_of_rows))
					measured_camera_view = True

				scanner.StopSetup()

			if not measured_camera_view:
				first_row = 0
				last_row = 2047
				number_of_rows = 2048
			final_mask_for_hdr_focus = self.select_most_informative_masks(masks=masks, all_mask_labels=exposure_times_to_probe, first_row=first_row, last_row=last_row, operate_on="exposure", relative_index=focus_index)
			self.all_masks.append(final_mask_for_hdr_focus)
			end_time = time.time()

			#print("For {}mm focus distance, HDR exposure times: {}".format(focus_distance, self.sets_of_exposures[focus_index]))


		all_masks_combined = self.all_masks[0]
		for additional_mask in self.all_masks[1:]:
			all_masks_combined = self.combine_masks(all_masks_combined, additional_mask)

		all_masks_combined_image = all_masks_combined * 255
		#print(current_mask[1000:1010,1000:1010])
		im = Image.fromarray(np.uint8(all_masks_combined_image), 'L')
		#im.save("all_masks_combined.png")


		return self.sets_of_exposures, self.sets_of_exposures_masks
			#print("\nProbed {} exposure times in {} seconds".format(len(exposure_times_to_probe), end_time - start_time))




	def get_next_most_informative_mask(self, current_mask, current_mask_labels, all_masks, all_mask_labels, min_information_gain, operate_on, relative_index=0):
		# get mask with least overlapping valid points
		valid_differences = all_masks - current_mask[..., np.newaxis]
		valid_differences = np.clip(valid_differences, a_min=0, a_max=1)
		#print("Valid differences: {}".format(valid_differences.shape))
		valid_differences_copy = valid_differences.copy()
		#print(valid_differences[1000:1010,1000:1010,:])
		most_different_masks = valid_differences.sum(axis=(0, 1))
		#print("Most different masks from current mask:\n{}".format(most_different_masks))
		most_different_index = np.argmax(most_different_masks)
		most_different_mask = all_masks[:,:,most_different_index]

		additional_pixels_added = valid_differences_copy[:,:,most_different_index]
		#print("Additional new pixels added for most different mask: {}".format(additional_pixels_added.shape))

		sum_of_different_information = most_different_masks[most_different_index]
		current_information = current_mask.sum(axis=(0, 1))
		information_gain = sum_of_different_information / current_information
		#print("Measure of difference: {}% information gain".format(information_gain))
		most_different_mask_label = all_mask_labels[most_different_index] #exposure_time_multipliers[most_different_index] * original_exposure_time
		#print("Next most different mask: {}".format(most_different_mask_label))
		if information_gain > min_information_gain:
			final_mask = self.combine_masks(most_different_mask, current_mask)
			current_mask_labels.append(most_different_mask_label)
			additional_pixels_added_image = additional_pixels_added * 255
			im = Image.fromarray(np.uint8(additional_pixels_added_image), 'L')
			if operate_on == "focus":
				self.focus_stack_distance.append(float(most_different_mask_label))
				self.focus_stack_masks["{:.3f}".format(most_different_mask_label)] = additional_pixels_added
				im.save("{}_additional_pixels_in_focus.png".format(int(most_different_mask_label)))
			if operate_on == "exposure":
				self.sets_of_exposures[-1].append(float(most_different_mask_label))
				self.sets_of_exposures_masks[-1]["{:.3f}".format(most_different_mask_label)] = additional_pixels_added
				im.save("focus_{}_exposure_{}ms_additional_pixels_in_focus.png".format(relative_index, int(most_different_mask_label)))
			sufficiently_more_information_to_gain = True
		else:
			final_mask = current_mask
			sufficiently_more_information_to_gain = False
		return final_mask, current_mask_labels, sufficiently_more_information_to_gain

	def combine_masks(self, mask_a, mask_b):
		# make union of two masks
		mask_union = mask_a + mask_b
		#print(mask_union[1000:1010,1000:1010])

		mask_union = np.clip(mask_union, a_min=0, a_max=1)
		#print(mask_union[1000:1010,1000:1010])

		#mask_union = np.where(mask_union == 2.0, 1, 0)
		#print("Sum of both masks = {}".format(mask_union.sum()))
		return mask_union


	def select_most_informative_masks(self, masks, all_mask_labels, first_row=0, last_row=2047, operate_on="focus", relative_index=0):
		# minimum information gain is defined as a minimum percent (e.g. 10%) of pixels added by a mask that are new
		current_mask_labels = [] 
		
		# mask rows estimated to be non-projector
		masks[0:first_row,:,:] = 0.0
		masks[last_row:-1,:,:] = 0.0

		# get total valid points per mask
		sum_of_valid_points = masks.sum(axis=(0, 1))
		#print("Valid point sum:\n{}".format(sum_of_valid_points))

		# get mask with most valid points
		max_valid_index = np.argmax(sum_of_valid_points)
		best_coverage_mask = all_mask_labels[max_valid_index] #exposure_time_multipliers[max_valid_index] * original_exposure_time
		#print("Best coverage mask: {}".format(best_coverage_mask))
		current_mask = masks[:,:,max_valid_index]
		current_mask_labels.append(best_coverage_mask)

		if operate_on == "focus":
			self.focus_stack_distance = []
			self.focus_stack_masks = {}
			self.focus_stack_distance.append(float(best_coverage_mask))
			self.focus_stack_masks["{:.3f}".format(best_coverage_mask)] = current_mask
			min_information_gain = self.min_focus_information_gain
		elif operate_on == "exposure":
			set_of_exposures = [float(best_coverage_mask)]
			set_of_exposures_masks = {"{:.3f}".format(best_coverage_mask): current_mask}
			self.sets_of_exposures.append(set_of_exposures)
			self.sets_of_exposures_masks.append(set_of_exposures_masks)
			min_information_gain = self.min_exposure_information_gain

		# guarantee at least one mask is selected above, and then go into a loop to add more masks as deemed useful by minimum information gain
		sufficiently_more_information_to_gain = True
		while sufficiently_more_information_to_gain:
			current_mask, current_mask_labels, sufficiently_more_information_to_gain = self.get_next_most_informative_mask(current_mask=current_mask, current_mask_labels=current_mask_labels, all_masks=masks, all_mask_labels=all_mask_labels, min_information_gain=min_information_gain, operate_on=operate_on, relative_index=relative_index)
 
		current_mask_image = current_mask * 255
		#print(current_mask[1000:1010,1000:1010])
		im = Image.fromarray(np.uint8(current_mask_image), 'L')

		if operate_on == "focus":
			#print("Final focus distances (mm): {}".format(current_mask_labels))
			#im.save("focus_stacked_mask_at_mean_exposure.png")
			self.focus_stack_distance.sort()
		elif operate_on == "exposure":
			pass
			#print("Final exposure times (mm): {}".format(current_mask_labels))
			#im.save("focus_{}_hdr_mask.png".format(int(self.current_focus_distance)))

		return current_mask

	def catch_segmentation_faults(self):
		signal.signal(signal.SIGSEGV, if_segmentation_fault)
		def if_segmentation_fault(self, signum, frame):
			raise Exception("\nA segmentation fault has been caught, with information:\nSignum: {}\nFrame: {}".format(signum, frame))
