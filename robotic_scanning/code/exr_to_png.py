import numpy as np
import cv2
import sys

np.set_printoptions(threshold=sys.maxsize, suppress=True, linewidth=2000)

def convert_linear_to_srgb(color, defog_for_color, im):

	def knee(x, f):
		return np.log(x * f + 1) / f

	def find_knee_f(x, y):
	    f0 = 0
	    f1 = 1
	    while knee(x, f1) > y:
	        f0 = f1
	        f1 = f1 * 2
	    for i in range(0,30):
	        f2 = (f0 + f1) / 2.0
	        y2 = knee(x, f2)
	        if y2 < y:
	        	f1 = f2
	        else:
	        	f0 = f2
	    return (f0 + f1) / 2.0

	exposure = 0.0
	exposure_factor = 2.47393
	knee_low = 0
	knee_high = 16.0

	d = 0.001 * defog_for_color
	print("Defog for {}: {} ({} after scaling)".format(color, defog_for_color, d))

	g = 2.2
	k1 = np.power(2, knee_low)
	x = np.power(2, knee_high) - k1
	y = np.power(2, 3.5) - k1
	m = np.power(2, exposure + exposure_factor)

	im = im - d # defog
	stuff = im < 0
	im[im < 0] = 0 # clip negative values to zero
	im = im * m # exposure
	f =  find_knee_f( np.power(2, knee_high) - k1, np.power(2, 3.5) - k1)
	true_change_value = k1 + knee(im - k1, f)

	im = np.where(im > k1, true_change_value, im)
	im = np.power(im, g)

	scaling_factor = np.power(2, -3.5 * g)
	print("Scaling factor: {}".format(scaling_factor))
	im = im * 255.0 * scaling_factor
	im = np.clip(im, a_min = 0, a_max = 255)

	return im

def exr_to_png(project_name):
	exr_image_to_read = "{}.exr".format(project_name)

	im = cv2.imread(exr_image_to_read,-1) 

	# where any one color channel is under-saturated (value of 1.0), recognize the aggregate color will also be partially saturated, so clamp the other channels to be 1.0, and thus a saturated black color (easy to filter)
	im[:,:,0][im[:,:,1] == 1.0] = 1.0
	im[:,:,0][im[:,:,2] == 1.0] = 1.0
	im[:,:,1][im[:,:,0] == 1.0] = 1.0
	im[:,:,1][im[:,:,2] == 1.0] = 1.0
	im[:,:,2][im[:,:,0] == 1.0] = 1.0
	im[:,:,2][im[:,:,1] == 1.0] = 1.0

	multiplier = 2*12
	im = im * multiplier

	image_width = im.shape[0]
	image_height = im.shape[1]
	image_pixels = image_width * image_height
	print("Reading {} image {} x {} pixels".format(exr_image_to_read, image_width, image_height))

	r_fog_color =  np.sum(im[:,:,0]) / image_pixels 
	g_fog_color =  np.sum(im[:,:,1]) / image_pixels
	b_fog_color =  np.sum(im[:,:,2]) / image_pixels

	im[:,:,0] = convert_linear_to_srgb("red", r_fog_color, im[:,:,0])
	im[:,:,1] = convert_linear_to_srgb("green", g_fog_color, im[:,:,1])
	im[:,:,2] = convert_linear_to_srgb("blue", b_fog_color, im[:,:,2])

	png_output_filename = "{}_from_exr.png".format(project_name)
	cv2.imwrite(png_output_filename,im)

	return png_output_filename