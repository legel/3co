import time
import numpy as np
import pickle

start_time = time.time()

import ajile3d

coordinate_system = "camera" # or "projector"

if coordinate_system == "camera":
	pixel_width = 2048
	pixel_height = 2048
elif coordinate_system == "projector":
	pixel_width = 1824
	pixel_height = 2280

point_index_to_pixel_row = np.zeros(pixel_width*pixel_height)
point_index_to_pixel_column = np.zeros(pixel_width*pixel_height)

for pixel_row in range(pixel_height):
	for pixel_column in range(pixel_width):
		point_cloud_index = pixel_row * pixel_width + pixel_column
		point_index_to_pixel_row[point_cloud_index] = pixel_row
		point_index_to_pixel_column[point_cloud_index] = pixel_column

pickle.dump(point_index_to_pixel_row, open("point_index_to_pixel_row_for_camera_coordinate_system.pkl", "wb"))
pickle.dump(point_index_to_pixel_column, open("point_index_to_pixel_column_for_camera_coordinate_system.pkl", "wb"))

end_time = time.time()

print("Indexing row and column time: {:.3f} seconds".format(end_time-start_time))