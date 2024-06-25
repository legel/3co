import ajile3d
from util import parse
from geometry import PointCloud
import logging
from registration import combine_images
from color_correction import normalize_colors
import color
import time
import sys
from pprint import pprint

camera_coordinate_system = ajile3d.CAMERA_PIXEL_COORDINATE_SYSTEM 
if camera_coordinate_system == ajile3d.PROJECTOR_PIXEL_COORDINATE_SYSTEM:
	camera_image_width = 1824
	camera_image_height = 2280
elif camera_coordinate_system == ajile3d.CAMERA_PIXEL_COORDINATE_SYSTEM:
	camera_image_width = 2048
	camera_image_height = 2048

camera_roi_width = ajile3d.C_DEFAULT_CAMERA_FOCUS_ROI_WIDTH 
camera_roi_height = ajile3d.C_ROI_NUM_ROWS_MULTIPLE
 
#print("Launching scanner...")

def initialize_scanner(filter_outliers=False,outliers_std_dev=0.1):
	scanner = ajile3d.Ajile3DImager()
	scanner.IsCameraFlipped = False
	scanner.SetDebugLevel(1)
	scanner.SetPatternMode(ajile3d.DOUBLE_LINESCAN_PATTERN_MODE)
	system_result = scanner.StartSystem()
	if system_result != 0:
		raise ConnectionError("\nCannot connect to DepthScan. Please make sure no other connections are persistent (e.g. GUI). Reset power if needed.\n")
	scanner.SetPixelCoordinateSystem(camera_coordinate_system)
	if filter_outliers:
		scanner.SetFilterOutliers(enabled=True, standardDev=outliers_std_dev)
	scanner.SetKeepRawImages(True)
	return scanner, camera_image_width, camera_image_height, camera_roi_width, camera_roi_height

def initialize_custom_scanner():
	scanner = ajile3d.Ajile3DImager()
	scanner.IsCameraFlipped = False
	scanner.SetDebugLevel(0)
	scanner.SetPatternMode(ajile3d.DOUBLE_LINESCAN_PATTERN_MODE)
	system_result = scanner.StartSystem()
	if system_result != 0:
		raise ConnectionError("\nCannot connect to DepthScan. Please make sure no other connections are persistent (e.g. GUI). Reset power if needed.\n")
	scanner.SetPixelCoordinateSystem(camera_coordinate_system)
	# if filter_outliers:
	# 	scanner.SetFilterOutliers(enabled=True, standardDev=outliers_std_dev)
	scanner.SetKeepRawImages(True)
	return scanner, camera_image_width, camera_image_height, camera_roi_width, camera_roi_height

def set_scanner_roi():
	scanner.SetExposureTimeMSec(exposure_time)
	scanner.SetCameraFocusROI( int(camera_image_width/2) - int(camera_roi_width/2), int(camera_image_height/2) - int(camera_roi_height/2))

def set_fast_imaging():
	scanner.SetTriggerMode(ajile3d.INTERNAL_SW_TRIGGER) # using software trigger for speed testing
	scanner.SetDepthImagesToCapture(0) 					# zero means forever, until we call stop

def clear_usb_channel_for_scanner(self, board="FT232H"):
	import reset_usb
	reset_usb.reset_device_by_substring(substring="FT232H")

def process_scanner_output(scanner_output, project_name="x", scan_index=0, relative_scan_index=0, dual_exposure_mode=False, distance_center_sample_size=100, export_to_npy=False, process_point_cloud=True, save_16_bit_image=False):
	for i in range(len(scanner_output.StructuredLightImages())):
		filename = "/home/sense/3cobot/{}-{}-{}-{}.png".format(project_name,scan_index,relative_scan_index,i)
		scanner_output.StructuredLightImages()[i].WriteToFile(filename, 16)
	combined_image = combine_images(project_name="{}-{}-{}".format(project_name,scan_index,relative_scan_index), dual_exposure_mode=dual_exposure_mode, save_16_bit_image=save_16_bit_image)
	if save_16_bit_image:
		input_as_npy = True
	else:
		input_as_npy = False
	corrected_image = color.apply_color_correction_matrix(combined_image, input_as_npy=input_as_npy)
	if process_point_cloud:
		point_cloud = PointCloud(project_name=project_name, scanner_output=scanner_output, distance_center_sample_size=distance_center_sample_size, scan_index=scan_index, relative_scan_index=relative_scan_index, colors_from_photo=corrected_image, export_to_npy=export_to_npy)
	#point_cloud.set_colors_by_photo(combined_image)
	del scanner_output

def downsample(point_cloud):	
	downsample_ratio = 50
	sample = point_cloud.copy()
	sample.downsample(downsample_ratio)
	average_distance = np.average(sample.z)
	print("Computed average distance of {}mm for points in FOV from a {}x downsampling".format(average_distance, downsample_ratio))

def scan(scanner, project_name="x", scan_index=0, relative_scan_index=0, exposure_time=75.0, dual_exposure_mode=False, dual_exposure_multiplier=1.0, distance_center_sample_size=100, export_to_npy=False, red_led_current=1.0, green_led_current=1.0, blue_led_current=1.0, process_point_cloud=True, save_16_bit_image=False):
	#print("Scanning {}ms exposure".format(exposure_time))
	scanner.SetExposureTimeMSec(exposure_time)
	time.sleep(0.1)
	if dual_exposure_mode:
		self.scanner.SetDualExposure(exposure_time * dual_exposure_multiplier)
		self.scanner.SetDepthImagesToCapture(2)
	scanner.SetColorExposureGain(ledChannel=0,exposureGain=red_led_current)
	scanner.SetColorExposureGain(ledChannel=1,exposureGain=green_led_current)
	scanner.SetColorExposureGain(ledChannel=2,exposureGain=blue_led_current)
	scanner.StartCapture()
	scanner.WaitForCaptureResult(60000) # milliseconds, -1 = wait forever
	if not scanner.IsCaptureQueueEmpty():
		scanner_output = scanner.RetrieveNextCaptured()
		scanner.StopCapture()
		scanner.ClearCaptureQueue()
		process_scanner_output(scanner_output=scanner_output, project_name=project_name, scan_index=scan_index, relative_scan_index=relative_scan_index, dual_exposure_mode=dual_exposure_mode, distance_center_sample_size=distance_center_sample_size, export_to_npy=export_to_npy, process_point_cloud=process_point_cloud, save_16_bit_image=save_16_bit_image)

def custom_sequence_scan(scanner, project_name="x", scan_index=0, relative_scan_index=0, exposure_time=75.0, dual_exposure_mode=False, dual_exposure_multiplier=1.0, distance_center_sample_size=100, export_to_npy=False, red_led_current=1.0, green_led_current=1.0, blue_led_current=1.0, process_point_cloud=True, photos_to_project="None"):
	photos_to_project = photos_to_project.split(",")

	image_queue = ajile3d.ImageQueue()
	list_of_image_data = ajile3d.ImageDataList()
	for image_number, photo_to_project in enumerate(photos_to_project):
		image_data = ajile3d.ImageData()
		image_data.ReadFromFile(photo_to_project)
		list_of_image_data.push_back(image_data)

	scanner.SetExposureTimeMSec(exposure_time) # SetExposureTimeSec
	time.sleep(0.1)

	scanner.SetColorExposureGain(ledChannel=0,exposureGain=red_led_current)
	scanner.SetColorExposureGain(ledChannel=1,exposureGain=green_led_current)
	scanner.SetColorExposureGain(ledChannel=2,exposureGain=blue_led_current)

	scanner.StartCustomCapture(list_of_image_data, image_queue)

	# #scanner.StartCapture()
	# scanner.WaitForCaptureResult(60000) # milliseconds, -1 = wait forever
	# if not scanner.IsCaptureQueueEmpty():
	for image_number, photo_to_project in enumerate(photos_to_project):
		image_queue.WaitForImages(60000)
		#scanner_output = scanner.RetrieveNextCaptured()
		#camera_image_que = image_queue.Front().first
		image_name = "{}_{}.png".format(project_name, image_number)
		filename = "/home/sense/3cobot/{}-{}-{}-{}.png".format(project_name,scan_index,relative_scan_index,image_number)
		#camera_image.WriteToFile(image_name, 16)
		#image_queue.Pop()
		image_queue.Front().first.WriteToFile(filename, 16)
		image_queue.Pop()

	scanner.StopCapture()
	scanner.ClearCaptureQueue()
	combined_image = combine_images(project_name="{}-{}-{}".format(project_name,scan_index,relative_scan_index), dual_exposure_mode=dual_exposure_mode, ambient_image_index = 3)
	# #for i in range(len(scanner_output.StructuredLightImages())):
	# #	filename = "/home/sense/3cobot/{}-{}-{}-{}.png".format(project_name,scan_index,relative_scan_index,i)
	# 	scanner_output.StructuredLightImages()[i].WriteToFile(filename, 16)
	# combined_image = combine_images(project_name="{}-{}-{}".format(project_name,scan_index,relative_scan_index), dual_exposure_mode=dual_exposure_mode)


		#scanner_output = scanner.RetrieveNextCaptured()

	
	# process_scanner_output(scanner_output=scanner_output, project_name=project_name, scan_index=scan_index, relative_scan_index=relative_scan_index, dual_exposure_mode=dual_exposure_mode, distance_center_sample_size=distance_center_sample_size, export_to_npy=export_to_npy, process_point_cloud=process_point_cloud)





	# Print("Starting Ajile Logo Pattern Capture Test!", Ajile3DTester::PrintLogType::Info);
    
 #    NUM_DMD_IMAGES = 3;
 #    customPatternList.clear();
    
 #    for (int i=0; i<NUM_DMD_IMAGES; i++) {
 #        aj3d::ImageData image;
 #        sprintf(filename, "Images/ajileLogo_RGB.png");
 #        image.ReadFromFile(filename);
 #        customPatternList.push_back(image);
 #    }
    
 #    aj3D_.SetExposureTime(100);

 #    if (!aj3D_.StartCustomCapture(customPatternList, cameraImageQueue)) {
 #        Print("Error capturing.", Ajile3DTester::PrintLogType::Error);
 #        return false;
 #    }
    
 #    // save the raw structured light images
 #    for (int i=0; i<NUM_DMD_IMAGES; i++) {
 #        cameraImageQueue.WaitForImages(-1); // -1 means wait forever
 #        aj3d::ImageData& cameraImage = cameraImageQueue.Front().first;
 #        // location to output images
 #        sprintf(filename, "cameraAjileLogoRGBImage_%d.png", i);
 #        cameraImage.WriteToFile(filename, 16);
 #        cameraImageQueue.Pop();
 #    }
    
    





def stop():
	scanner.ClearCaptureQueue()
	scanner.StopCapture()

def save_calibration_file():
	scanner.SaveCalibrationFile("/home/sense/3cobot/scanner_opencv_calibration_v2.xml")

if __name__ == "__main__":
	external_api = True
	if external_api:
		args = parse(args=sys.argv, defaults={"exposure_time": "10.0"})
		exposure_time = float(args["exposure_time"])
		dual_exposure_mode = True if args["dual_exposure_mode"] == "True" else False
		dual_exposure_multiplier = float(args["dual_exposure_multiplier"])
		project_name = args["project_name"]
		save_to_png = True if args["save_to_png"] == "True" else False
		missing_data_as = args["missing_data_as"]
		export_to_csv = True if args["export_to_csv"] == "True" else False
		export_to_npy = True if args["export_to_npy"] == "True" else False
		outliers_std_dev = float(args["outliers_std_dev"])
		filter_outliers = True if args["filter_outliers"] == "True" else False
		distance_center_sample_size = int(args["distance_center_sample_size"])
		process_point_cloud = True if args["process_point_cloud"] == "True" else False
		hard_filter_min = float(args["hard_filter_min"])
		hard_filter_max = float(args["hard_filter_max"])
		scan_index = int(args["scan_index"])
		relative_scan_index = int(args["relative_scan_index"])
		red_led_current = float(args["red_led_current"])
		green_led_current = float(args["green_led_current"])
		blue_led_current = float(args["blue_led_current"])	
		focus_distance = float(args["focus_distance"])	
		photos_to_project = args["photos_to_project"]	
		save_16_bit_image = True if args["save_16_bit_image"] == "True" else False
	else:
		args = {}
		exposure_time = 10.0
		dual_exposure_mode = False
		dual_exposure_multiplier = 1.0
		project_name = "test"
		save_to_png = False
		missing_data_as = "black"
		export_to_csv = False
		export_to_npy = False
		outliers_std_dev = 0.0
		filter_outliers = False
		distance_center_sample_size = 4
		process_point_cloud = True
		hard_filter_min = 200
		hard_filter_max = 600
		scan_index = 0
		relative_scan_index = 0

	print("")
	pprint(args)
	print("")
	try:

		if photos_to_project != "None":
			scanner, camera_image_width, camera_image_height, camera_roi_width, camera_roi_height = initialize_custom_scanner()

			custom_sequence_scan(	scanner=scanner, 
									project_name=project_name, 
									scan_index=scan_index, 
									relative_scan_index=relative_scan_index, 
									exposure_time=exposure_time,
									dual_exposure_mode=dual_exposure_mode, 
									dual_exposure_multiplier=dual_exposure_multiplier,
									distance_center_sample_size=distance_center_sample_size, 
									export_to_npy=export_to_npy,
									red_led_current=red_led_current,
									green_led_current=green_led_current,
									blue_led_current=blue_led_current,
									process_point_cloud=process_point_cloud,
									photos_to_project=photos_to_project)

		else:
			scanner, camera_image_width, camera_image_height, camera_roi_width, camera_roi_height = initialize_scanner(filter_outliers, outliers_std_dev)

			scan(	scanner=scanner, 
					project_name=project_name, 
					scan_index=scan_index, 
					relative_scan_index=relative_scan_index, 
					exposure_time=exposure_time,
					dual_exposure_mode=dual_exposure_mode, 
					dual_exposure_multiplier=dual_exposure_multiplier,
					distance_center_sample_size=distance_center_sample_size, 
					export_to_npy=export_to_npy,
					red_led_current=red_led_current,
					green_led_current=green_led_current,
					blue_led_current=blue_led_current,
					process_point_cloud=process_point_cloud,
					save_16_bit_image=save_16_bit_image)
	except Exception as e:
		print("System failure: {}".format(e))