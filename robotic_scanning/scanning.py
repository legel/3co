import numpy as np
from commander import *
from os import listdir
from hdr import *
from geometry import *
import subprocess
from color_correction import normalize_colors

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

class Scanner():
	def __init__(self, exposure_time = 1.0, save_calibration_file=False, fast_imaging=False):
		self.exposure_time = exposure_time
		self.save_calibration_file = save_calibration_file
		self.fast_imaging = fast_imaging
		
	def scan(self, focus_distances=[], focus_masks={}, sets_of_exposures=[], dual_exposure_mode=False, dual_exposure_multiplier=4.0, project_name="x", save_to_png=True, missing_data_as="black", export_to_csv=False, export_to_npy=True, outliers_std_dev=0.1, filter_outliers=True, distance_center_sample_size=4, process_point_cloud=True, hard_filter_min=200, hard_filter_max=600, hdr_3d_projection=True, scan_index=0, relative_scan_index=0):
		#exposure_time = exposure_time if exposure_time != None else self.exposure_time
		args = "dual_exposure_mode={}".format(dual_exposure_mode) + "&"
		args += "dual_exposure_multiplier={:.3f}".format(dual_exposure_multiplier) + "&"
		args += "save_to_png={}".format(save_to_png) + "&"
		args += "missing_data_as={}".format(missing_data_as) + "&"
		args += "export_to_csv={}".format(export_to_csv) + "&"
		args += "outliers_std_dev={}".format(outliers_std_dev) + "&"
		args += "filter_outliers={}".format(filter_outliers) + "&"
		args += "distance_center_sample_size={}".format(distance_center_sample_size) + "&"
		args += "process_point_cloud={}".format(process_point_cloud) + "&"
		args += "hard_filter_min={}".format(hard_filter_min) + "&"
		args += "hard_filter_max={}".format(hard_filter_max) + "&"
		args += "scan_index={}".format(scan_index) + "&"

		# if len(hdr_exposure_times) > 0:

		# system first finds optimal focus distances, then for each focus distance, finds best set of HDR exposure times
		point_clouds_with_different_focuses = []
		focus_index = 0
		for focus_distance, set_of_exposures in zip(focus_distances, sets_of_exposures):
			focus_index += 1
			project_name = "{}-f{}".format(project_name, focus_index)
			focus_scan_args = args + "project_name={}".format(project_name) + "&"

			scan_tag = "{}-{}".format(project_name, scan_index)

			point_clouds = []
			hdr_input_photos = []
			valid_exposure_times = []

			if len(set_of_exposures) > 1:
				hdr_scan_export_to_npy = False
			else:
				hdr_scan_export_to_npy = export_to_npy

			for hdr_scan_number, exposure_time in enumerate(set_of_exposures):
				scan_args = focus_scan_args + "relative_scan_index={}".format(hdr_scan_number) + "&"
				scan_args += "export_to_npy={}".format(hdr_scan_export_to_npy) + "&"
				scan_args += "exposure_time={:.3f}".format(exposure_time)
				sub_scan_tag = "{}-{}-{}".format(project_name,current_focus_scan_index,hdr_scan_number)
				subprocess.call(["python3", "depthscan.py", scan_args])
				if self.scan_was_successful(sub_scan_tag):
					point_clouds.append("{}.ply".format(sub_scan_tag))
					hdr_input_photos.append("{}.png".format(sub_scan_tag))
					valid_exposure_times.append(exposure_time)
				else:
					print("Scan Failed")
					return None
					#raise Exception("SCAN FAILED")
			tonemapped_png, radiance_data = pngs_to_exr_to_png(project_name=scan_tag, photo_filenames=hdr_input_photos, exposure_times=valid_exposure_times)
			normalize_colors(tonemapped_png)
			cloud = self.combine_point_clouds(point_clouds)
			cloud.reindex_into_tensor()
			cloud.set_colors_by_photo(tonemapped_png)
			cloud.set_hdr_colors_by_exr_data(radiance_data)
			cloud.average_xyz_positions_with_outlier_removal()

			if export_to_csv:
				cloud.export_to_csv(project_name=scan_tag, from_tensor=True)
			if export_to_npy:
				cloud.export_to_npy(project_name=scan_tag, from_tensor=True)
			cloud.save_as_ply(filename="{}.ply".format(scan_tag), from_tensor=False)
			point_clouds_with_different_focuses.append(cloud)

		focus_stacked_cloud = self.combine_point_clouds(point_clouds_with_different_focuses)
		focus_stacked_cloud.reindex_into_tensor()
		focus_stacked_cloud.average_xyz_positions_with_outlier_removal()
		if export_to_npy:
			focus_stacked_cloud.export_to_npy(project_name="{}_focused".format(project_name), from_tensor=True)
		focus_stacked_cloud.save_as_ply(filename="{}_focused.ply".format(project_name), from_tensor=False)

		
		return cloud


		# else:
		# 	scan_args = args + "relative_scan_index={}&".format(0) 
		# 	scan_args += "exposure_time={:.3f}".format(exposure_time)
		# 	sub_scan_tag = "{}-{}-{}".format(project_name,scan_index,relative_scan_index)
		# 	subprocess.call(["python3", "depthscan.py", scan_args])
		# 	if self.scan_was_successful(sub_scan_tag):
		# 		point_cloud_filename = "{}.ply".format(sub_scan_tag)
		# 		hdr_input_photo = "{}.png".format(sub_scan_tag)
		# 		valid_exposure_time = exposure_time
		# 	else:
		# 		raise Exception("SCAN FAILED")
		# 	#tonemapped_png, radiance_data = pngs_to_exr_to_png(project_name=scan_tag, photo_filenames=[hdr_input_photo], exposure_times=[valid_exposure_time])
		# 	cloud = PointCloud(filename=point_cloud_filename)
		# 	#cloud.reindex_into_tensor()
		# 	#cloud.set_colors_by_photo(tonemapped_png)
		# 	#cloud.set_hdr_colors_by_exr_data(radiance_data)
		# 	if export_to_csv:
		# 		cloud.export_to_csv(project_name=scan_tag, from_tensor=True)
		# 	if export_to_npy:
		# 		cloud.export_to_npy(project_name=scan_tag, from_tensor=True)
		# 	cloud.save_as_ply(filename="{}.ply".format(scan_tag), from_tensor=False)
		# 	return cloud

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

	def combine_point_clouds(self, point_clouds):
		point_clouds = [PointCloud(filename=point_cloud, scan_index=i) for i,point_cloud in enumerate(point_clouds)]
		base_point_cloud = point_clouds[0]
		for other_point_cloud in point_clouds[1:]:
			base_point_cloud.add_point_cloud(other_point_cloud)
		return base_point_cloud



# REIMPLEMENT EXPOSURE TIME PROBING

# if auto_hdr:
# 	successful_exposure_time_indices = 0
# 	possible_exposure_times = []
# 	if not hdr_exposure_times:
# 		 	hdr_exposure_times.append(self.exposure_time)
# 		 	for exposure_stop in range(1, auto_positive_negative_exposure_x4 + 1):
# 		 		increased_exposure_time = self.exposure_time * 4**(exposure_stop)
# 		 		decreased_exposure_time = self.exposure_time * (1 / 4**(exposure_stop))
# 		 		hdr_exposure_times.append(increased_exposure_time)
# 		 		hdr_exposure_times.append(decreased_exposure_time)

# def probe_best_exposure_time():
# 	target_number_of_scans_per_view = 3
# 	max_exposure_time = 400.0
# 	exposure_time_multipliers = [0.4, 1.0, 1.6]
# 	masks = np.zeros((camera_image_width, camera_image_height, len(exposure_time_multipliers)))

# 	start_time = time.time()
# 	for i, multiplier in enumerate(exposure_time_multipliers): # first 3 for removing noise
# 		self.exposure_time = exposure_time * multiplier
# 		if self.exposure_time > max_exposure_time:
# 			self.exposure_time = max_exposure_time

# 		possible_exposure_times.append(self.exposure_time)

# 		self.scanner.SetExposureTimeMSec(self.exposure_time)
# 		self.scanner.StartSetup()


# 		if self.exposure_time < 25.0:
# 			time.sleep(0.1)
# 		if self.exposure_time < 75.0:
# 			time.sleep(0.2)
# 		if self.exposure_time < 150.0:
# 			time.sleep(0.5)
# 		elif self.exposure_time < 315.0:
# 			time.sleep(1.1)
# 		elif self.exposure_time <= 400.0:
# 			time.sleep(1.3)

# 		mask = self.scanner.GetLatestValidPointMask()
# 		np_mask = np.zeros(shape=(mask.height, mask.width, 1), dtype=np.uint8)
# 		mask.WriteToMemory(np_mask)

# 		np_mask = np.where(np_mask == 255.0, 1, 0)
# 		masks[:,:,i] = np_mask[:,:,0]
# 		#print(masks[1000:1010,:,i])
# 		#print(np_mask.shape)

# 		#mask.WriteToFile("{}_valid_points_exposure_{}x_{:.1f}ms.png".format(project_name, multiplier,self.exposure_time))

# 		if multiplier == 1.0:
# 			self.scanner.DetectCameraView()
# 			first_row = self.scanner.GetCameraViewFirstRow()
# 			last_row = self.scanner.GetCameraViewLastRow()
# 			number_of_rows = self.scanner.GetCameraViewNumRows()
# 			print("Camera view 1st row = {}, Last row = {}, Number of Rows = {}".format(first_row, last_row, number_of_rows))
# 		self.scanner.StopSetup()

# 		next_time = time.time()
# 		print("\nExposure time {}x set to {} after {} seconds".format(multiplier, self.exposure_time, next_time - start_time))

# 		if self.exposure_time == max_exposure_time:
# 			break

# 		successful_exposure_time_indices += 1

# 	masks = masks[:,:,:successful_exposure_time_indices+1]
# 	# The union of masks for an HDR set which first maximizes overall valid points, then additional new valid points x2 is finished
	
# 	exposure_times_to_scan = []
# 	# mask rows estimated to be non-projector
# 	masks[0:first_row,:,:] = 0.0
# 	masks[last_row:-1,:,:] = 0.0

# 	# get total valid points per mask
# 	sum_of_valid_points = masks.sum(axis=(0, 1))
# 	print("Valid point sum across all exposure times:\n{}".format(sum_of_valid_points))

# 	# get mask with most valid points
# 	max_valid_index = np.argmax(sum_of_valid_points)
# 	best_coverage_exposure_time = possible_exposure_times[max_valid_index] #exposure_time_multipliers[max_valid_index] * original_exposure_time
# 	print("Best coverage exposure time: {:.2f}ms".format(best_coverage_exposure_time))
# 	max_valid_mask = masks[:,:,max_valid_index]
# 	exposure_times_to_scan.append(best_coverage_exposure_time)

# 	# get mask with least overlapping valid points
# 	valid_differences = np.absolute(masks - max_valid_mask[..., np.newaxis])
# 	print(valid_differences[1000:1010,1000:1010,:])
# 	most_different_masks = valid_differences.sum(axis=(0, 1))
# 	print("Most different masks from best coverage:\n{}".format(most_different_masks))
# 	most_different_index = np.argmax(most_different_masks)
# 	most_different_mask = masks[:,:,most_different_index]
# 	most_different_exposure_time = possible_exposure_times[most_different_index] #exposure_time_multipliers[most_different_index] * original_exposure_time
# 	print("Next most different exposure time: {:.2f}ms".format(most_different_exposure_time))
# 	exposure_times_to_scan.append(most_different_exposure_time)

# 	# make union of first two selected masks
# 	mask_union = max_valid_mask + most_different_mask
# 	print(mask_union[1000:1010,1000:1010])

# 	mask_union = np.clip(mask_union, a_min=0, a_max=1)
# 	print(mask_union[1000:1010,1000:1010])

# 	#mask_union = np.where(mask_union == 2.0, 1, 0)
# 	print("Sum of both masks = {}".format(mask_union.sum()))

# 	# get final mask with least overlapping valid points
# 	print("Shape of mask union: {}".format(mask_union.shape))
# 	remaining_valid = masks - mask_union[..., np.newaxis]
# 	remaining_valid = np.where(remaining_valid >= 1, 1, 0)
# 	print("Remaining valid:\n{}".format(remaining_valid[1000:1010,1000:1010,:]))

# 	#valid_differences = np.absolute(masks - mask_union[..., np.newaxis])
# 	#print(valid_differences[1000:1010,1000:1010,:])
# 	most_different_masks = remaining_valid.sum(axis=(0, 1))
# 	print("Most different masks from union:\n{}".format(most_different_masks))
# 	most_different_index = np.argmax(most_different_masks)
# 	most_different_mask = masks[:,:,most_different_index]
# 	most_different_exposure_time = possible_exposure_times[most_different_index] #exposure_time_multipliers[most_different_index] * original_exposure_time
# 	print("Final most different exposure time: {:.2f}ms".format(most_different_exposure_time))
# 	exposure_times_to_scan.append(most_different_exposure_time)

# 	# final expected mask union of three selected masks
# 	new_mask_union = mask_union + most_different_mask
# 	new_mask_union = np.clip(new_mask_union, a_min=0, a_max=1)
# 	print(new_mask_union.shape)
# 	print(new_mask_union[1000:1010,1000:1010])
# 	print("Final of 3 masks = {}".format(new_mask_union.sum()))			
# 	exposure_times_to_scan = [t if t < max_exposure_time else max_exposure_time for t in exposure_times_to_scan]
# 	print("Exposure times to scan: {}".format(exposure_times_to_scan))

# 	new_mask_union = new_mask_union * 255
# 	print(new_mask_union[1000:1010,1000:1010])
# 	im = Image.fromarray(np.uint8(new_mask_union), 'L')
# 	im.save("hdr_mask.png")

# 	hdr_exposure_times = exposure_times_to_scan