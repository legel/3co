import numpy as np
from PIL import Image
import time

color_correction_matrix = np.matrix([[2.251206, -0.023585, -0.048801],
                                     [-0.081512, 1.365058, -0.004268],
                                     [0.040179, -0.055866, 1.313286]])

def apply_color_correction_matrix(image_filepath, input_gamma = 1.0, output_gamma = 2.2, input_image_max_size=255, input_as_npy=True):
	if not input_as_npy:
		# read image and convert to numpy array
		raw_image = Image.open(image_filepath)
		pixel_values = np.asarray(raw_image)

		# normalize values between 0.0 - 1.0, based on maximum possible value size (e.g. if image is 8-bit, i.e. values will be divided by (2^8)-1, i.e. 255)
		normalized_pixel_values = pixel_values / input_image_max_size 

	else:
		with open('/home/sense/3cobot/{}'.format(image_filepath), 'rb') as input_file:
			normalized_pixel_values = np.load(input_file)

	# fix gamma if not already linear
	if input_gamma != 1.0:
		inverse_input_gamma = 1 / input_gamma
		normalized_pixel_values = np.power(normalized_pixel_values, inverse_input_gamma)

	# reshape input image to flat x 3 color channels
	width, height, colors = normalized_pixel_values.shape
	flattened_normalized_pixels = normalized_pixel_values.flatten().reshape(width*height,colors)

	# apply color correction matrix
	matrix_pixel_values = np.matrix(flattened_normalized_pixels)
	corrected_pixel_values = matrix_pixel_values * color_correction_matrix
	corrected_pixel_values = np.asarray(corrected_pixel_values)

	# clip values beyond saturation threshold
	corrected_pixel_values[corrected_pixel_values > 1.0] = 1.0
	corrected_pixel_values[corrected_pixel_values < 0.0] = 0.0

	# apply output gamma (e.g. 2.2 for Adobe sRGB)
	inverse_output_gamma = 1 / output_gamma
	corrected_pixel_values = np.power(corrected_pixel_values, inverse_output_gamma)

	# handle saturation in original image by ensuring that those points remain saturated
	corrected_pixel_values[flattened_normalized_pixels == 1.0] = 1.0

	# reshape back into 2D image with 3 color channels
	corrected_pixel_values = corrected_pixel_values.reshape(width,height,colors)

	# convert back into 8-bit image
	corrected_pixel_values = np.round(corrected_pixel_values * 255)
	corrected_pixel_values = corrected_pixel_values.astype(np.uint8)

	output_image = Image.fromarray(corrected_pixel_values)
	image_tag = image_filepath.split(".png")[0]
	output_image_name = "{}_corrected.png".format(image_tag)
	output_image.save(output_image_name)
	return output_image_name

#apply_color_correction_matrix("/home/sense/3cobot/raw_image.png")

