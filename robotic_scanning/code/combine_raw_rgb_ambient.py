import cv2
import numpy as np
import sys
from PIL import Image
import os

# R,G,B are the first 3 patterns captured, respectively
# Read the dark (ambient lighting) image. Note: this is the 5th image in the sequence, not the 4th

def combine_images(project_name = "caligon_view540_yaw-175_pitch45"):
	red_image_path = "/home/sense/3cobot/{}_0.png".format(project_name)
	green_image_path = "/home/sense/3cobot/{}_1.png".format(project_name)
	blue_image_path = "/home/sense/3cobot/{}_2.png".format(project_name)
	ambient_image_path = "/home/sense/3cobot/{}_4.png".format(project_name)

	red_image = cv2.imread(red_image_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
	green_image = cv2.imread(green_image_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
	blue_image = cv2.imread(blue_image_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
	ambient_image = cv2.imread(ambient_image_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)

	width = red_image.shape[0]
	height = red_image.shape[1]

	# subtract the ambient lighting image from the color images
	color_channels = np.zeros((width,height,3), np.float32)

	# vis2 = cv2.CreateMat(3, 1, cv2.CV_32FC3)
	# color_channels = cv2.fromarray(raw_color_channels)

	# convert the color channels to 8-bit to create a 24-bit RGB image. You could possibly make a 16-bit/channel color image, or have 10-bit/channel, but you will have an issue finding viewers to display it!
	color_channels[:,:,0] = cv2.absdiff(red_image, ambient_image) 
	color_channels[:,:,1] = cv2.absdiff(green_image, ambient_image) 
	color_channels[:,:,2] = cv2.absdiff(blue_image, ambient_image) 

	relative_colors = color_channels / 1024
	relative_colors = relative_colors * 2**8

	color_channels = relative_colors.astype('uint8')
	pil_image = Image.fromarray(color_channels) #, 'I;8'

	output_image_name = "{}.png".format(project_name)
	pil_image.save(output_image_name) #,bits=8

def combine_all_images_with_tag(project_directory = "/home/sense/3cobot", tag = "focus_at_1300mm_", total_views = 9):
	all_files_in_directory = os.listdir(project_directory)
	files_matching_tag = [f for f in all_files_in_directory if tag in f]
	print(files_matching_tag)
	view_indices = [int(f.split(tag)[1].split("_")[0].split("view")[1]) for f in files_matching_tag]

	view_index_to_perspective = {}
	for view_index, file_name in zip(view_indices, files_matching_tag):
		if not view_index_to_perspective.get(view_index, False):
			perspective_header = "_".join(file_name.split(".png")[0].split("_")[:-1])
			print(perspective_header)
			if "pitch" in perspective_header and "yaw" in perspective_header:
				view_index_to_perspective[view_index] = perspective_header

	ordered_project_names = []
	for view_index in range(1, total_views+1):
		ordered_project_names.append(view_index_to_perspective[view_index])

	for project_name in ordered_project_names:
		print(project_name)
		combine_images(project_name=project_name)

if __name__ == "__main__":
	combine_images("1300mm_on_origin")
	#combine_all_images_with_tag()