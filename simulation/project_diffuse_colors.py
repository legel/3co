import math
import numpy as np
import open3d as o3d
import disney_brdf
import path_planning
import os
import json
import cv2

from sklearn.cluster import KMeans

def main():

  path = "models/toucan_0.5"
  fname = "{}/toucan_0.5_0_diffuse_colors.png".format(path)
  img = cv2.imread(fname)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  height = len(img)
  width = len(img[0])
  colors = []

  for i in range(height):
    for j in range(width):

      x = img[i,j,:]
      if (x == np.array([70,70,70])).all() or (x == np.array([71,71,71])).all() or (x == np.array([72,72,72])).all() :
        img[i,j,:] = [255,255,255]

  for i in range(height):
    for j in range(width):
      x = img[i,j,:]
      if x[0] == 255 and x[1] == 255 and x[2] == 255:
        pass
      else:
        colors.append(x)


  # flamingo
  #diffuse_colors = [
  #  [206, 153, 163], # pink
  #  [1, 1, 1], # black
  #  [206, 205, 205], # light gray 1
  #  [192, 178, 55], # yellow                
  #  [124, 122, 122], # light gray 2
  #  [63, 60, 49], # gray    
  #  [190, 180, 173] # light gray 3
  #]

#  thresh = 10.0
#  for color in colors:
#    if ( np.linalg.norm(np.asarray(diffuse_colors[0]) - np.asarray(color)) > thresh ):
#      print(color)


  kmeans = KMeans(n_clusters=10, init='k-means++', max_iter=300, n_init=10)
  kmeans.fit(colors)

  #print(kmeans.cluster_centers_)

  for i in range(height):
    for j in range(width):
      x = img[i,j,:]
      if x[0] == 255 and x[1] == 255 and x[2] == 255:
        img[i,j,:] = [70,70,70]
      else:
        #distances = []
        #for k in range(len(diffuse_colors)):
        #  distances.append(np.linalg.norm(np.asarray(diffuse_colors[k]) - np.asarray(x)))
        #img[i,j,:] = diffuse_colors[distances.index(min(distances))]
        
        label = kmeans.predict([x])[0]
        color = kmeans.cluster_centers_[label]
        img[i,j,:] = color

  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)      

  cv2.imwrite("{}/toucan_0.5_0_diffuse_colors_projected.png".format(path), img)
  

  
  

    


  
  

if __name__ == "__main__":
  main()