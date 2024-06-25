import numpy as np
import cv2, PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

aruco_dict = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h10)

fig = plt.figure()
#nx = 4
#ny = 3
#for i in range(1, nx*ny+1):
#  ax = fig.add_subplot(ny,nx, i)
for i in [0,1000,2000,2319, 2315]: # 2320
	img = aruco.drawMarker(aruco_dict, i, 700)
	plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
	#ax.axis("off")

	#plt.savefig("markers.pdf")
	plt.show()


