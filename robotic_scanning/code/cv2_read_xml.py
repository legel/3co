import cv2


calibrations = cv2.FileStorage("/home/sense/3cobot/scanner_opencv_calibration.xml", cv2.FILE_STORAGE_READ)
i=0
while True:
	rotations = calibrations.getNode("C_r").at(i).mat() ## 
	print("({}): {}".format(i,rotations))
	i+=1 
