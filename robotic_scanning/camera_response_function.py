import numpy as np
import cv2
import time

def return_camera_response_function():
	exposure_times_ms = [5.0, 10.0]#, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0]
	exposure_times_s = [time/1000.0 for time in exposure_times_ms]
	exposure_times = np.array(exposure_times_s, dtype=np.float32)
	filenames = [	"max_aperture_5ms.png",
					"max_aperture_10ms.png"] 
					# "1589398762625_camera_response_function_45ms.png",
					# "1589398774353_camera_response_function_50ms.png",
					# "1589398786088_camera_response_function_55ms.png",
					# "1589398798396_camera_response_function_60ms.png",
					# "1589398810647_camera_response_function_65ms.png",
					# "1589398823316_camera_response_function_70ms.png",
					# "1589398836286_camera_response_function_75ms.png"]
	images = []
	for filename in filenames:
		im = cv2.imread(filename)
		images.append(im)

	calibrateDebevec = cv2.createCalibrateDebevec()
	responseDebevec = calibrateDebevec.process(images, exposure_times)

	mergeDebevec = cv2.createMergeDebevec()
	hdrDebevec = mergeDebevec.process(images, exposure_times, responseDebevec)

	cv2.imwrite("hdrDebevec.hdr", hdrDebevec)

	tonemap = cv2.createTonemapDrago(gamma=1.0, saturation=0.0, bias=1.0)
	ldr = tonemap.process(hdrDebevec)
	ldr = 3 * ldr
	cv2.imwrite("ldr.png", ldr * 255)

start_time = time.time()
return_camera_response_function()
end_time = time.time()
print("{} seconds".format(end_time-start_time))