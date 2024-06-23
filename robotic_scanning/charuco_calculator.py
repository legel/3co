height = 1500
width = 1160

horizontal_squares_per_paper = 27
vertical_squares_per_paper = 35

width_per_square = 40
height_per_square = 40

page_border_x = ((width - horizontal_squares_per_paper * width_per_square) / 2.0) / 1000.0
page_border_y = ((height - vertical_squares_per_paper * height_per_square) / 2.0) / 1000.0

print("x border: {}m".format(page_border_x))
print("y border: {}m".format(page_border_y))

# 18 x 0.5 meters system
# py MarkerPrinter.py --charuco --file "./charuco.pdf" --dictionary DICT_5X5_1000 --size_x 126 --size_y 7 --square_length 0.065 --marker_length 0.050 --border_bits 1 --sub_size_x 7 --sub_size_y 7 --page_border_x 0.0225 --page_border_y 0.0225


# py MarkerPrinter.py --charuco --file "./big_charuco.pdf" --dictionary DICT_5X5_1000 --size_x 27 --size_y 35 --square_length 0.040 --marker_length 0.033 --border_bits 1 --sub_size_x 0 --sub_size_y 0 --page_border_x 0.040 --page_border_y 0.050