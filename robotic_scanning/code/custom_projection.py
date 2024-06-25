from robotics import Iris
from PIL import Image
import numpy as np

image_to_project = "/home/sense/Downloads/3co.png"

# 912x1140
# 1.67719298246
projector_width = 1140  
projector_height = 912
color_channels = 3

image = Image.open(image_to_project)
#data = image.getdata()
image_data = np.array(image, dtype=np.uint8)

r_channel = image_data[:,:,0]
g_channel = image_data[:,:,1]
b_channel = image_data[:,:,2]


reds = np.zeros(shape=(projector_width, projector_height, color_channels), dtype=np.uint8)
reds[:,:,0] = r_channel

greens = np.zeros(shape=(projector_width, projector_height, color_channels), dtype=np.uint8)
greens[:,:,1] = g_channel

blues = np.zeros(shape=(projector_width, projector_height, color_channels), dtype=np.uint8)
blues[:,:,2] = b_channel

ambient = np.zeros(shape=(projector_width, projector_height, color_channels), dtype=np.uint8)

red_image = Image.fromarray(reds)
green_image = Image.fromarray(greens)
blue_image = Image.fromarray(blues)
ambient_image = Image.fromarray(ambient)


red_image.save("/home/sense/3cobot/3co_red.png")
green_image.save("/home/sense/3cobot/3co_green.png")
blue_image.save("/home/sense/3cobot/3co_blue.png")
ambient_image.save("/home/sense/3cobot/ambient.png")

pure_reds = np.zeros(shape=(projector_width, projector_height, color_channels), dtype=np.uint8)
pure_reds[:,:,0] = 255

pure_greens = np.zeros(shape=(projector_width, projector_height, color_channels), dtype=np.uint8)
pure_greens[:,:,1] = 255

pure_blues = np.zeros(shape=(projector_width, projector_height, color_channels), dtype=np.uint8)
pure_blues[:,:,2] = 255

pure_red_image = Image.fromarray(pure_reds)
pure_green_image = Image.fromarray(pure_greens)
pure_blue_image = Image.fromarray(pure_blues)

pure_red_image.save("/home/sense/3cobot/r.png")
pure_green_image.save("/home/sense/3cobot/g.png")
pure_blue_image.save("/home/sense/3cobot/b.png")

robot = Iris()
robot.scan(project_name="custom_v2", photos_to_project="r.png,g.png,b.png,ambient.png", export_to_npy=False, process_point_cloud=False, auto_focus=False, auto_exposure=False, hdr_exposure_times=[150.0])