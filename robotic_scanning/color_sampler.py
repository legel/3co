import numpy as np
import seaborn as sns
import matplotlib

def sample_n_colors_uniformly_from_human_perceptual_space(n_colors, prepare_for_hdbscan=False):
	colors = np.zeros((n_colors,3), dtype=np.int16)

	# https://seaborn.pydata.org/tutorial/color_palettes.html
	# "Using husl means that the extreme values, and the resulting ramps to the midpoint, while not perfectly perceptually uniform, will be well-balanced:"
	if prepare_for_hdbscan:
		husl_colors = sns.husl_palette(n_colors=n_colors - 1)
	else:
		husl_colors = sns.husl_palette(n_colors=n_colors)

	for i, husl_color in enumerate(husl_colors):
	    colors[i,0] = int(husl_color[0] * 255) 
	    colors[i,1] = int(husl_color[1] * 255) 
	    colors[i,2] = int(husl_color[2] * 255)

	# hard code initialize the first color as R,G,B = (15,15,15), i.e. a dark grey, which can correspond to HDBSCAN cluster indexing of [-1, 0, 1, ..., N-1]
	# this way, we guarantee the overall aesthetic of non-clustered nodes does not dominate
	if prepare_for_hdbscan:
		reds = [15]
		greens = [15]
		blues = [15]
	else:
		reds = []
		greens = []
		blues = []		

	reds.extend(colors[:,0].tolist())
	greens.extend(colors[:,1].tolist())
	blues.extend(colors[:,2].tolist())

	return reds, greens, blues

if __name__ == "__main__":
	reds,greens,blues = sample_n_colors_uniformly_from_human_perceptual_space(n_colors=1000)
	for r,g,b in zip(reds,greens,blues):
		print("(R,G,B) = ({},{},{})".format(r,g,b))