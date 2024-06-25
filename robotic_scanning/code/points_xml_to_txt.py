import cv2

point_cloud = cv2.FileStorage("/home/sense/cloud_points.xml", cv2.FILE_STORAGE_READ)
i = 0
all_xs = []
all_ys = []
all_zs = []
try:
	while True:
		point = point_cloud.getNode("cloud_points").at(i).mat()
		all_xs.append(point[0][0])
		all_ys.append(point[1][0])
		all_zs.append(point[2][0])
		i+=1
except:
	pass

with open("cloud_points.txt", "w") as output_file:
	for x,y,z in zip(all_xs,all_ys,all_zs):
		output_file.write("{} {} {}\n".format(x,y,z))