import os
import re

def get_nearest_neighbors(image_paths, N):
    output_list = []
    for i, base_img in enumerate(image_paths):
        for j in range(max(0, i - N), min(len(image_paths), i + N + 1)):
            if i != j:
                output_list.append(f"{base_img} {image_paths[j]}")
    return output_list

def define_a_list_of_images_to_extract_features(directory_of_images, min_image_index=200, max_image_index=210, plus_minus_n_temporally_local_images=3):
	image_paths = []

	# Iterate through all files in directory
	for filename in os.listdir(directory_of_images):
	    match = re.match(r'(\d{6})\.jpg', filename)
	    if match:
	        img_num = int(match.group(1))
	        if min_image_index <= img_num <= max_image_index:
	            full_path = os.path.join(directory_of_images, filename)
	            image_paths.append(full_path)

	# Sort the list
	image_paths.sort()

	output_pairs = get_nearest_neighbors(image_paths, plus_minus_n_temporally_local_images)

	with open("image_pairs_to_extract_features.txt", "w") as output_file:
		for pair in output_pairs:
			output_file.write("{}\n".format(pair))

if __name__ == "__main__":
	define_a_list_of_images_to_extract_features(directory_of_images="/home/photon/sense/3cology/plantvine/scans/aglaonema_silver_bay/color")

