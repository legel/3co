import numpy as np
from PIL import Image, ImageDraw
import random

image_size = 2048

img = Image.open('normed_hierarchical_perlin_noise.png')
pixels = img.load()

def draw_line(x1, y1, x2, y2, r, g, b, alpha):
	draw = ImageDraw.Draw(img, 'RGBA') 
	draw.line((x1,y1, x2, y2), fill=(r,g,b, alpha ))

total_lines = 0
for skip_size in (256, 128, 64, 32, 16, 8, 4, 2, 1):
	section_size = int(image_size / skip_size)
	for x_section in range(skip_size):
		x_min = x_section * section_size
		x_max = (x_section+1) * section_size
		for y_section in range(skip_size):
			y_min = y_section * section_size
			y_max = (y_section+1) * section_size
			for line in range(20):
				x1 = int(random.uniform(x_min, x_max))
				y1 = int(random.uniform(y_min, y_max))
				x2 = int(random.uniform(x_min, x_max))
				y2 = int(random.uniform(y_min, y_max))
				r = int(random.uniform(0, 255))
				g = int(random.uniform(0, 255))
				b = int(random.uniform(0, 255))
				alpha = int(random.uniform(0, 255))
				draw_line(x1, y1, x2, y2, r, g, b, alpha)
			total_lines += 20
			print(total_lines)

img.save('entopy.png')