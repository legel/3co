import os
import numpy as np
import cv2
import brdf_fit
import disney_brdf



def main():
  model = "toucan_0.5"
  experiment = 9
  n_colors = 10
  n_steps = 22
  combined = []
  height = 0
  width = 0


  for c in range(n_colors):

    loss_c = []
    fname = "outputs/{}/experiment_{}/diffuse_loss_color{}.txt".format(model,experiment,c)
    with open(fname, "r") as fin:
      for line in fin:
        loss_c.append(float(line))

    for t in range(n_steps):
      tt = min(t, len(loss_c)-1)
      fname = "outputs/{}/experiment_{}/diffuse_fit_color{}_{}.png".format(model,experiment,c,tt)
      img = cv2.imread(fname, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

      if len(combined) < t+1:
        height = len(img)
        width = len(img[0])        
        combined.append(np.zeros((height,width,3), np.float32))        

                
      for i in range(height):
        for j in range(width):
          if not (img[i,j,:] == [70,70,70]).all():
            combined[t][i,j,:] = img[i,j,:]      
      
      for i in range(height):
        for j in range(width):
          if (combined[t][i,j,:] == [0,0,0]).all():
            combined[t][i,j,:] = [70,70,70]


  diffuse_loss = []
  ground_truth = cv2.imread("models/{}/{}_0_render.png".format(model,model))
  ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB)

  for t in range(len(combined)):

    fname = "outputs/{}/experiment_{}/diffuse_fit_{}.png".format(model,experiment,t)  
    loss = brdf_fit.photometric_error(ground_truth, combined[t])
    diffuse_loss.append(loss)

    combined[t] = cv2.cvtColor(combined[t], cv2.COLOR_RGB2BGR)  
    cv2.imwrite(fname, combined[t])


  for c in range(n_colors):

    loss_c = []
    fname = "outputs/{}/experiment_{}/reflectance_loss_color{}.txt".format(model,experiment,c)
    with open(fname, "r") as fin:
      for line in fin:
        loss_c.append(float(line))

    for t in range(n_steps):
      tt = min(t, len(loss_c)-1)
      fname = "outputs/{}/experiment_{}/reflectance_fit_color{}_{}.png".format(model,experiment,c,tt)
      img = cv2.imread(fname, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

      if len(combined) < t+1:
        height = len(img)
        width = len(img[0])        
        combined.append(np.zeros((height,width,3), np.float32))        

                
      for i in range(height):
        for j in range(width):
          if not (img[i,j,:] == [70,70,70]).all():
            combined[t][i,j,:] = img[i,j,:]      
      
      for i in range(height):
        for j in range(width):
          if (combined[t][i,j,:] == [0,0,0]).all():
            combined[t][i,j,:] = [70,70,70]


  reflectance_loss = []
  for t in range(len(combined)):
            
    fname = "outputs/{}/experiment_{}/reflectance_fit_{}.png".format(model,experiment,t)           
    loss = brdf_fit.photometric_error(ground_truth, combined[t])
    reflectance_loss.append(loss)

    combined[t] = cv2.cvtColor(combined[t], cv2.COLOR_RGB2BGR)  
    cv2.imwrite(fname, combined[t])

  
  fname = "outputs/{}/experiment_{}/total_diffuse_loss.txt".format(model, experiment)
  fout = open(fname, "w")
  for t in range(len(diffuse_loss)):
    fout.write("{}\n".format(diffuse_loss[t]))
  fout.close()
    
  fname = "outputs/{}/experiment_{}/total_reflectance_loss.txt".format(model, experiment)
  fout = open(fname, "w")
  for t in range(len(reflectance_loss)):
    fout.write("{}\n".format(reflectance_loss[t]))
  fout.close()




if __name__ == "__main__":
  main()