import pickle
import numpy as np

pixel_rows      = pickle.load(open("point_index_to_pixel_row_for_camera_coordinate_system.pkl", "rb")).astype(np.uint16)
pixel_columns   = pickle.load(open("point_index_to_pixel_column_for_camera_coordinate_system.pkl", "rb")).astype(np.uint16) 

for ply_point_index, (row, column) in enumerate(zip(pixel_rows, pixel_columns)):
	print("POINT INDEX {} = (ROW {}, COL {})".format(ply_point_index, row, column))
	if ply_point_index == 100000:
		break

print("...")