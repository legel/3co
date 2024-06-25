from PIL import Image

list_of_pixels_to_get = [	(35,294), 
							(38,408), 
							(70,520), 
							(77,638), 
							(68,755), 
							(66,909), 
							(57,1019), 
							(66,1147), 
							(79,1257), 
							(87,1393), 
							(118,1516), 
							(105,1632), 
							(85,1754), 
							(145,249), 
							(398,321), 
							(420,380), 
							(422,508), 
							(419,644), 
							(418,753), 
							(447,884), 
							(402,1001), 
							(429,1137), 
							(434,1261), 
							(430,1371), 
							(434,1513), 
							(437,1618), 
							(437,1748), 
							(493,274), 
							(537,287), 
							(611,400), 
							(616,501), 
							(607,632), 
							(621,764), 
							(639,871), 
							(646,1043), 
							(606,1129), 
							(608,1253), 
							(621,1381), 
							(615,1510), 
							(601,1625), 
							(644,1751), 
							(657,103), 
							(653,153), 
							(838,47), 
							(841,240), 
							(846,333), 
							(859,484), 
							(851,624), 
							(870,752), 
							(880,871), 
							(850,1013), 
							(872,1150), 
							(865,1266), 
							(879,1396), 
							(878,1534), 
							(895,1671), 
							(876,1804), 
							(959,151), 
							(968,366), 
							(990,687), 
							(1004,114), 
							(1123,223), 
							(1151,351), 
							(1156,474), 
							(1171,604), 
							(1177,748), 
							(1158,919), 
							(1189,1014), 
							(1199,1134), 
							(1198,1265), 
							(1212,1398), 
							(1194,1528), 
							(1205,1664), 
							(1206,1783), 
							(1091,99), 
							(1087,4), 
							(975,3), 
							(744,84), 
							(614,122), 
							(639,89), 
							(1506,253), 
							(1505,381), 
							(1518,511), 
							(1532,632), 
							(1529,753), 
							(1508,869), 
							(1520,998), 
							(1508,1115), 
							(1514,1241), 
							(1499,1373), 
							(1521,1499), 
							(1501,1638), 
							(1506,1765), 
							(1865,205), 
							(2032,209), 
							(1908,240), 
							(1883,366), 
							(1904,489), 
							(1920,619), 
							(1926,754),
							(1767,851), 
							(1891,991), 
							(1921,1104), 
							(1922,1228), 
							(2040,1244), 
							(1910,1360), 
							(1943,1487), 
							(1973,1594), 
							(1974,1724), 
							(2126,253), 
							(2144,388), 
							(2147,495), 
							(2155,604), 
							(2170,738), 
							(2095,848), 
							(2147,978), 
							(2188,109), 
							(2138,1019), 
							(2139,1358), 
							(2145,1464), 
							(2129,1603), 
							(2098,1713)			
							]

unit_width = 16
unit_height = 1

number_of_samples = len(list_of_pixels_to_get) * unit_width

exposure_times = [6,9,12,15,18,21,24,27,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,110,120,130,140,150,160,170,180,190,200,225,250,275,300,350,400]

number_of_exposure_times = len(exposure_times)

output_image = Image.new('RGB', (number_of_samples, number_of_exposure_times * unit_height), (255,255,255))
output_pixels = output_image.load()

for h, exposure_time in enumerate(exposure_times):
	input_image = "/home/sense/3cobot/color_calibrations/color_calibration_v4_{}ms.png".format(exposure_time)
	input_image_data = Image.open(input_image)

	for i, pixel_to_get in enumerate(list_of_pixels_to_get):
		x,y = pixel_to_get
		r1,g1,b1,a1 = input_image_data.getpixel((x,y))
		r2,g2,b2,a2 = input_image_data.getpixel((x+1,y))
		r3,g3,b3,a3 = input_image_data.getpixel((x,y+1))
		r4,g4,b4,a4 = input_image_data.getpixel((x+1,y+1))
		r5,g5,b5,a5 = input_image_data.getpixel((x,y+2))
		r6,g6,b6,a6 = input_image_data.getpixel((x+2,y))
		r7,g7,b7,a7 = input_image_data.getpixel((x+1,y+2))
		r8,g8,b8,a8 = input_image_data.getpixel((x+2,y+1))
		r9,g9,b9,a9 = input_image_data.getpixel((x,y))
		r10,g10,b10,a10 = input_image_data.getpixel((x-1,y))
		r11,g11,b11,a11 = input_image_data.getpixel((x,y-1))
		r12,g12,b12,a12 = input_image_data.getpixel((x-1,y-1))
		r13,g13,b13,a13 = input_image_data.getpixel((x,y-2))
		r14,g14,b14,a14 = input_image_data.getpixel((x-2,y))
		r15,g15,b15,a15 = input_image_data.getpixel((x-1,y-2))
		r16,g16,b16,a16 = input_image_data.getpixel((x-2,y-1))


		for j in range(unit_height):
			if b1 != 255:
				output_pixels[i * unit_width + 0, h*unit_height + j] = (r1,b1,g1) # switch g and b because of previous error
			if b2 != 255:
				output_pixels[i * unit_width + 1, h*unit_height + j] = (r2,b2,g2)	
			if b3 != 255:	
				output_pixels[i * unit_width + 2, h*unit_height + j] = (r3,b3,g3)
			if b4 != 255:
				output_pixels[i * unit_width + 3, h*unit_height + j] = (r4,b4,g4)
			if b5 != 255:
				output_pixels[i * unit_width + 4, h*unit_height + j] = (r5,b5,g5)
			if b6 != 255:		
				output_pixels[i * unit_width + 5, h*unit_height + j] = (r6,b6,g6)
			if b7 != 255:
				output_pixels[i * unit_width + 6, h*unit_height + j] = (r7,b7,g7)
			if b8 != 255:
				output_pixels[i * unit_width + 7, h*unit_height + j] = (r8,b8,g8)
			if b9 != 255:
				output_pixels[i * unit_width + 8, h*unit_height + j] = (r9,b9,g9)
			if b10 != 255:
				output_pixels[i * unit_width + 9, h*unit_height + j] = (r10,b10,g10)
			if b11 != 255:		
				output_pixels[i * unit_width + 10, h*unit_height + j] = (r11,b11,g11)
			if b12 != 255:
				output_pixels[i * unit_width + 11, h*unit_height + j] = (r12,b12,g12)
			if b13 != 255:
				output_pixels[i * unit_width + 12, h*unit_height + j] = (r13,b13,g13)
			if b14 != 255:
				output_pixels[i * unit_width + 13, h*unit_height + j] = (r14,b14,g14)		
			if b15 != 255:
				output_pixels[i * unit_width + 14, h*unit_height + j] = (r15,b15,g15)
			if b16 != 255:
				output_pixels[i * unit_width + 15, h*unit_height + j] = (r16,b16,g16)


output_image.save("calib_image.png")

for h, exposure_time in enumerate(exposure_times):
	input_image_base = "/home/sense/3cobot/color_calibrations/color_calibration_v4"
	input_image = "{}_{}ms.png".format(input_image_base, exposure_time)
	input_image_data = Image.open(input_image)

	output_image = Image.new('RGB', (number_of_samples, unit_height), (255,255,255))
	output_pixels = output_image.load()

	for i, pixel_to_get in enumerate(list_of_pixels_to_get):
		x,y = pixel_to_get
		r1,g1,b1,a1 = input_image_data.getpixel((x,y))
		r2,g2,b2,a2 = input_image_data.getpixel((x+1,y))
		r3,g3,b3,a3 = input_image_data.getpixel((x,y+1))
		r4,g4,b4,a4 = input_image_data.getpixel((x+1,y+1))
		r5,g5,b5,a5 = input_image_data.getpixel((x,y+2))
		r6,g6,b6,a6 = input_image_data.getpixel((x+2,y))
		r7,g7,b7,a7 = input_image_data.getpixel((x+1,y+2))
		r8,g8,b8,a8 = input_image_data.getpixel((x+2,y+1))
		r9,g9,b9,a9 = input_image_data.getpixel((x,y))
		r10,g10,b10,a10 = input_image_data.getpixel((x-1,y))
		r11,g11,b11,a11 = input_image_data.getpixel((x,y-1))
		r12,g12,b12,a12 = input_image_data.getpixel((x-1,y-1))
		r13,g13,b13,a13 = input_image_data.getpixel((x,y-2))
		r14,g14,b14,a14 = input_image_data.getpixel((x-2,y))
		r15,g15,b15,a15 = input_image_data.getpixel((x-1,y-2))
		r16,g16,b16,a16 = input_image_data.getpixel((x-2,y-1))

		for j in range(unit_height):
			if b1 != 255:
				output_pixels[i * unit_width + 0, j] = (r1,b1,g1) # switch g and b because of previous error
			if b2 != 255:
				output_pixels[i * unit_width + 1, j] = (r2,b2,g2)	
			if b3 != 255:	
				output_pixels[i * unit_width + 2, j] = (r3,b3,g3)
			if b4 != 255:
				output_pixels[i * unit_width + 3, j] = (r4,b4,g4)
			if b5 != 255:
				output_pixels[i * unit_width + 4, j] = (r5,b5,g5)
			if b6 != 255:		
				output_pixels[i * unit_width + 5, j] = (r6,b6,g6)
			if b7 != 255:
				output_pixels[i * unit_width + 6, j] = (r7,b7,g7)
			if b8 != 255:
				output_pixels[i * unit_width + 7, j] = (r8,b8,g8)
			if b9 != 255:
				output_pixels[i * unit_width + 8, j] = (r9,b9,g9)
			if b10 != 255:
				output_pixels[i * unit_width + 9, j] = (r10,b10,g10)
			if b11 != 255:		
				output_pixels[i * unit_width + 10, j] = (r11,b11,g11)
			if b12 != 255:
				output_pixels[i * unit_width + 11, j] = (r12,b12,g12)
			if b13 != 255:
				output_pixels[i * unit_width + 12, j] = (r13,b13,g13)
			if b14 != 255:
				output_pixels[i * unit_width + 13, j] = (r14,b14,g14)		
			if b15 != 255:
				output_pixels[i * unit_width + 14, j] = (r15,b15,g15)
			if b16 != 255:
				output_pixels[i * unit_width + 15, j] = (r16,b16,g16)


	output_image.save("{}_{}ms_sampled_fix.png".format(input_image_base, exposure_time))

print(number_of_samples)