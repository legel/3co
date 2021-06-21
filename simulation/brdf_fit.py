import numpy as np
from numpy.lib.function_base import extract
import scipy
import disney_brdf
import open3d as o3d
import json
import os
import copy
import math
import reconstruction
from scipy.optimize import minimize
import matplotlib 
import matplotlib.pyplot as plt 
from operator import itemgetter
import cv2

def range2rgb(img):
  return np.ubyte (np.floor(img * 255.0))

def rgb2range(img):
  return img / 255.0

#######################################################
############# Diffuse Color Optimization ##############
#######################################################

def luminosity(x):
  p = x[1]
  return (0.3*p[0] + 0.6*p[1] + 0.1*p[2]) # same approx used in disney brdf code

# output: [ [x,y,z], [r,g,b], [cam_x,cam_y,cam_z] ]
def get_diffuse_optimization_data(geometry_imgs, normal_imgs, rgb_imgs, camera_positions):
  data = []
  height = len(geometry_imgs[0])
  width = len(geometry_imgs[0][0])
  for geometry_img, normal_img, rgb_img, camera_pos in zip(geometry_imgs, normal_imgs, rgb_imgs, camera_positions):                  
    for i in range(height):
      for j in range(width):
        vertex = geometry_img[i,j,:]
        color = rgb_img[i,j,:]
        N = normal_img[i,j,:]
        data.append([vertex, color, N, camera_pos])  

  data.sort(key = luminosity) # sort by luminosity
  
  data[: int(len(data)/4)]  # take first quartile
  return data

def diffuse_loss(x, *args):
  data = args[0]
  use_disney_diffuse_brdf_approx = args[1]
  total_loss = 0.0
  for i in range(len(data)):        
    L = data[i][3]
    N = data[i][2]
    camera_pos = data[i][3]
    p = data[i][0]
    if use_disney_diffuse_brdf_approx == True:
      predicted_color = disney_brdf.render_diffuse_disney_brdf_on_point(L, N, camera_pos, p, x)
    else:
      predicted_color = disney_brdf.render_diffuse_brdf_on_point(L, N, camera_pos, p, x)
    
    loss = np.linalg.norm(predicted_color - data[i][1]) ** 2.0
    total_loss = total_loss + loss      
  
  return total_loss / len(data)

def diffuse_step(x):
  global timestep
  data = diffuse_data
  print_params(x)  
  loss = diffuse_loss(x, data, use_disney_diffuse_brdf_approx)
  print("diffuse loss: {}".format(loss))  
  print("")
  diffuse_loss_plot.append(loss)

  fig, ax = plt.subplots(1,1)
  plt.title("Diffuse Loss")
  ax.plot(np.asarray(range(timestep + 1)), diffuse_loss_plot)
  plt.savefig("outputs/{}/experiment_{}/diffuse_loss_color{}.png".format(model, experiment_num, diffuse_color_index))
  plt.close()

  fname = "outputs/{}/experiment_{}/diffuse_loss_color{}.txt".format(model, experiment_num, diffuse_color_index)
  with open (fname, "a") as file:
    file.write("{}\n".format(loss))

  params = extract_brdf_params(np.append(x, np.zeros(10)))
  render = disney_brdf.render_disney_brdf_image(x, geometry_imgs[0], normal_imgs[0], camera_positions[0], params, True)
  outfname = "outputs/{}/experiment_{}/diffuse_fit_color{}_{}.png".format(model, experiment_num, diffuse_color_index, timestep)
  render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)   
  render = np.round(render * 255.0)
  cv2.imwrite(outfname, render)      
  timestep = timestep + 1    

def diffuse_color_optimization(geometry_imgs, normal_imgs, rgb_imgs, camera_positions, data):  
  initial_guess = np.random.rand(3)
  diffuse_step(initial_guess)
  factr = 1e3
  pgtol = 1e-4
  eps = 1e-4 # step size used by finite difference, since we don't supply custom gradient
  soln = scipy.optimize.fmin_l_bfgs_b(func=diffuse_loss, x0=initial_guess, args=(data,use_disney_diffuse_brdf_approx), fprime=None, approx_grad=True, bounds=[(0.0,1.0),(0.0,1.0),(0.0,1.0)], factr=factr, pgtol=pgtol, epsilon=eps, maxiter=100, callback=diffuse_step)  
  return soln


#######################################################
######### Reflectance Parameter Optimization ##########
#######################################################

def photometric_error(image_1, image_2, normals_img):
  
  photo_loss = 0.0
  height = len(image_1)
  width = len(image_1[0])
  n = 0
  if (image_1.shape != image_2.shape):
    raise Exception("Error: images must have the same dimensions")
  for i in range(height):
    for j in range(width):
      N = normals_img[i,j,:]
      # skip pixels with no data
      if (N == [0,0,0]).all():
        continue      
      v1 = image_1[i,j,:]
      v2 = image_2[i,j,:]            
      photo_loss = photo_loss + np.linalg.norm(v1 - v2) ** 2
      n = n + 1

  return photo_loss / n

def extract_brdf_params(x):
  brdf_params = {}
  for i, param in enumerate(disney_brdf.get_brdf_param_names()):
    brdf_params[param] = x[i]    
  return brdf_params

def brdf_params_to_array(brdf_params):
  brdf_params_arr = []
  for param in disney_brdf.get_brdf_param_names():
    brdf_params_arr.append(brdf_params[param])
  return brdf_params_arr

def print_params(x):
  for name, value in zip(disney_brdf.get_brdf_param_names(), x):
    print("{0} : {1:0.3f}".format(name, value))
  print("")

# args: [geometry_img], [normals_img], [rgb_img], [camera_position], diffuse_colors
def reflectance_loss(x, *args):
  geometry_imgs = args[0] # sequence of 2d arrays of x, y, z data
  normals_imgs = args[1] # sequence of 2d arrays of nx, ny, nz data
  rgb_imgs = args[2]  # sequence of 2d arrays of RGB data
  camera_positions = args[3] # sequence of x,y,z data
  diffuse_colors = args[4]  # assumed fixed (solved for in diffuse optimization)
    
  height = len(geometry_imgs[0])
  width = len(geometry_imgs[0][0])
  
  reflectance_params = extract_brdf_params(x)  
  loss = 0.0  
  n = 0

  for geometry_img, normals_img, rgb_img, camera_pos in zip(geometry_imgs, normals_imgs, rgb_imgs, camera_positions):
    predicted_img = disney_brdf.render_disney_brdf_image(diffuse_colors, geometry_img, normals_img, camera_pos, reflectance_params, False)
    photo_error = photometric_error(rgb_img, predicted_img, normals_img)    
    loss = loss + photo_error
    n = n + height * width
  
  return 0.5 * loss


# See Equation 5 in
# https://docs.google.com/document/d/1_4bEW0cvtAsq6qlvX3CmGH8zlCPL3HGEjaDz0rlP6wA/edit
def reflectance_loss_gradient(x, *args):
  geometry_imgs = args[0] # x, y, z
  normals_imgs = args[1] # nx, ny, nz
  rgb_imgs = args[2]  # RGB
  camera_positions = args[3]      
  height = len(geometry_imgs[0])
  width = len(geometry_imgs[0][0])
  brdf_params = extract_brdf_params(x)
  n = 0
  loss_gradient = np.zeros( (13, 1) ) 

  for geometry_img, normals_img, rgb_img, camera_pos in zip(geometry_imgs, normals_imgs, rgb_imgs, camera_positions):
      
    avg_loss_term = 0.0

    for i in range(height):
      for j in range(width):

        N = normals_img[i,j,:]

        # skip pixels with no data
        if (N == [0,0,0]).all():
          continue

        p = geometry_img[i,j,:]
        light_pos = camera_pos[:3]

        # x: surface tangent    
        U = disney_brdf.normalize(np.random.rand(3))          
        X = disney_brdf.normalize(np.cross(N, U))  

        # y: surface bitangent
        Y = disney_brdf.normalize(np.cross(N, X))

        #  V: view direction  
        V = disney_brdf.normalize(np.asarray(camera_pos[:3]) - np.asarray(p))

        L = V

        # Irradiance
        irradiance_i = disney_brdf.compute_irradiance(light_pos, N, camera_pos[:3], p)            

        brdf = disney_brdf.BRDF(L, V, N, X, Y, [brdf_params['red'], brdf_params['green'], brdf_params['blue']], brdf_params['metallic'], brdf_params['subsurface'], brdf_params['specular'], brdf_params['roughness'], brdf_params['specularTint'], brdf_params['anisotropic'], brdf_params['sheen'], brdf_params['sheenTint'], brdf_params['clearcoat'], brdf_params['clearcoatGloss'])
        saturate_check = brdf * irradiance_i
        predicted_rgb = brdf * irradiance_i
      
        loss_gradient_i = brdf * irradiance_i

        predicted_rgb = np.minimum(predicted_rgb, 1.0)    
        predicted_rgb = np.power(predicted_rgb, 1.0 / 2.2)

        truth_rgb = rgb_img[i,j,:]

        # note: the derivative of the projection to [0,255] and back 
        # done by BRDF should be captured for loss_gradient_i term above as well,
        # probably just minor effect though
        for k in range(3):
          predicted_rgb[k] = round(predicted_rgb[k] * 255.0) / 255.0
        
        loss_i = np.sum(predicted_rgb - truth_rgb)        

        loss_gradient_i = np.power(loss_gradient_i, -1.2 / 2.2)

        loss_gradient_i = loss_gradient_i / 2.2      
        loss_gradient_i = loss_gradient_i * irradiance_i
        
        loss_gradient_i = np.asarray(loss_gradient_i)
        loss_gradient_i = loss_gradient_i.reshape(3,1)
          
        # account for saturation: derivative is 0 if saturated      
        # the following should work, but seems to mess up the results
        # what is the right way to do this?
        #for j in range(3):
        #  if saturate_check[j] > 1:  
        #    loss_gradient_i[j] = 0.0      


        # brdf gradient
        brdf_gradient_i = np.matrix(disney_brdf.brdf_gradient(L, V, N, X, Y, [brdf_params['red'], brdf_params['green'], brdf_params['blue']], brdf_params['metallic'], brdf_params['subsurface'], brdf_params['specular'], brdf_params['roughness'], brdf_params['specularTint'], brdf_params['anisotropic'], brdf_params['sheen'], brdf_params['sheenTint'], brdf_params['clearcoat'], brdf_params['clearcoatGloss']))      

        # set all partials except roughness, clearcoat, clearcoatGloss to zero
        brdf_gradient_i[3,:] = 0.0 # metallic
        brdf_gradient_i[4,:] = 0.0 # subsurface
        brdf_gradient_i[5,:] = 0.0 # specular
        #brdf_gradient_i[6,:] = 0.0 # roughness
        brdf_gradient_i[7,:] = 0.0 # specularTint
        brdf_gradient_i[8,:] = 0.0 # anisotropic
        brdf_gradient_i[9,:] = 0.0 # sheen
        brdf_gradient_i[10,:] = 0.0 # sheenTint
        #brdf_gradient_i[11,:] = 0.0 # clearcoat
        #brdf_gradient_i[12,:] = 0.0 # clearcoatGloss

        loss_gradient_i = np.dot(brdf_gradient_i, loss_gradient_i)
        loss_gradient_i = loss_gradient_i * loss_i
        loss_gradient_i = loss_gradient_i.reshape(13,1)

        loss_gradient = loss_gradient + loss_gradient_i
        n = n + 1
  
  loss_gradient = loss_gradient / n
  loss_gradient = np.asarray(loss_gradient).reshape(-1)  

  return loss_gradient

def brdf_numerical_gradient_wrapper(x, *args):
  L = args[0]
  V = args[1]
  N = args[2]
  X = args[3]
  Y = args[4]
  brdf_params = extract_brdf_params(x)
  return disney_brdf.BRDF_wrapper(L,V,N,X,Y,brdf_params)[0]

def reflectance_step(x):
  global timestep
  print_params(x)
  brdf_params = extract_brdf_params(x)
  
  diffuse_colors = [ brdf_params['red'], brdf_params['green'], brdf_params['blue'] ]

  loss = reflectance_loss(x, geometry_imgs, normal_imgs, rgb_imgs, camera_positions, diffuse_colors)
  print("reflectance loss: {}".format(loss))
  print("")
  loss_plot.append(loss)
  grad = reflectance_loss_gradient(x, geometry_imgs, normal_imgs, rgb_imgs, camera_positions)
  print("reflectance loss gradient: {}".format(grad))

  fig, ax = plt.subplots(1,1)
  plt.title("Reflectance Loss")
  ax.plot(np.asarray(range(timestep + 1)), loss_plot)
  plt.savefig("outputs/{}/experiment_{}/reflectance_loss_color{}.png".format(model, experiment_num, diffuse_color_index))
  plt.close()

  fname = "outputs/{}/experiment_{}/reflectance_loss_color{}.txt".format(model, experiment_num, diffuse_color_index)
  with open (fname, "a") as file:
    file.write("{}\n".format(loss))

  render = disney_brdf.render_disney_brdf_image(diffuse_colors, geometry_imgs[0], normal_imgs[0], camera_positions[0], extract_brdf_params(x), False)
  outfname = "outputs/{}/experiment_{}/reflectance_fit_color{}_{}.png".format(model, experiment_num, diffuse_color_index, timestep)
  render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)   
  render = np.round(render * 255.0)
  cv2.imwrite(outfname, render)

  timestep = timestep + 1

  
def reflectance_parameter_optimization(geometry_imgs, normal_imgs, rgb_imgs, camera_positions, diffuse_colors):
  param_bounds = disney_brdf.get_brdf_param_bounds()
  initial_guess = np.append(np.asarray(diffuse_colors), np.asarray(np.random.rand(10)))  

  # initialize to reasonable values
  initial_guess[3] = 0.0 # metallic
  initial_guess[4] = 0.0 # subsurface
  initial_guess[5] = 0.5 # specular
  initial_guess[6] = 0.5 # roughness
  initial_guess[7] = 0.0 # specularTint
  initial_guess[8] = 0.0 # aniso
  initial_guess[9] = 0.0 # sheen
  initial_guess[10] = 0.0 # sheenTint
  initial_guess[11] = 0.0 # clearcoat
  initial_guess[12] = 0.0 # clearcoatGloss

  reflectance_step(initial_guess)
  factr = 1e3
  pgtol = 1e-8
  sol = scipy.optimize.fmin_l_bfgs_b(func=reflectance_loss, x0=initial_guess, args=(geometry_imgs, normal_imgs, rgb_imgs, camera_positions, diffuse_colors), fprime=reflectance_loss_gradient, approx_grad=False, bounds=param_bounds, factr=factr, pgtol=pgtol, maxiter=100, callback=reflectance_step)    
  return sol


#######################################################
############### Gradient Visualizations ###############
#######################################################

def plot_brdf_gradient(meshes, camera_pos, x_init):

  param_names = disney_brdf.get_brdf_param_names()
  vertex_indices = range(0, len(meshes[0].vertices))
  
  for index in range(3,13):  
    name = param_names[index]  
    print("creating graph for {}".format(name))    
    raw_brdfs = np.zeros((36,3))
    raw_gradients = np.zeros((36,3))
    numerical_gradients = np.zeros((36,3))
        
    x = copy.deepcopy(x_init)
    
    for vertex_index in vertex_indices:
      mesh = meshes[0]
      camera = camera_pos[0]
      N = disney_brdf.normalize(mesh.vertex_normals[vertex_index])
      p = [mesh.vertices[vertex_index][0], mesh.vertices[vertex_index][1], mesh.vertices[vertex_index][2]]
      light_pos = camera[:3]
      L = disney_brdf.normalize(light_pos)
      U = disney_brdf.normalize(np.random.rand(3))    
      X = disney_brdf.normalize(np.cross(N, U))  
      Y = disney_brdf.normalize(np.cross(N, X))
      vertex = p
      V = disney_brdf.normalize(np.asarray(camera[:3]) - np.asarray(vertex))
      
      for i in range(4,40):
        x[index] = i/40.0
        brdf_params = extract_brdf_params(x)
        raw_brdf_output = disney_brdf.BRDF_wrapper(L, V, N, X, Y, brdf_params)
        for j in range(3):
          raw_brdfs[i-4][j] = raw_brdfs[i-4][j] + round(raw_brdf_output[j],9)

      for i in range(4,40):
        x[index] = i/40.0
        brdf_params = extract_brdf_params(x)
        raw_gradient = disney_brdf.brdf_gradient_wrapper(L,V,N,X,Y,brdf_params)[index]
        for j in range(3):
          raw_gradients[i-4][j] = raw_gradients[i-4][j] + round(raw_gradient[j],9)

      for i in range(4,40):
        x[index] = i/40.0
        brdf_params = extract_brdf_params(x)
        epsilon = 1e-4
        numerical_gradient = scipy.optimize.approx_fprime(x, brdf_numerical_gradient_wrapper, epsilon, L,V,N,X,Y)[index]
        numerical_gradients[i-4] = numerical_gradients[i-4] + round(numerical_gradient,9)
      
    raw_brdfs = raw_brdfs / len(vertex_indices)
    raw_gradients = raw_gradients / len(vertex_indices)
    numerical_gradients = numerical_gradients / len(vertex_indices)

    for j in range(3):
      fig, (ax1, ax2, ax3) = plt.subplots(3,1)
      plt.title(name)
      ax1.plot(np.asarray(range(4,40))/40.0, raw_brdfs[:,j])
      ax1.set_ylabel("raw brdf")

      ax2.plot(np.asarray(range(4,40))/40.0, raw_gradients[:,j])
      ax2.set_ylabel("raw gradient")

      ax3.plot(np.asarray(range(4,40))/40.0, numerical_gradients)
      ax3.set_ylabel("numerical gradient")

      plt.savefig("outputs/{}_{}.png".format(name, j))


def plot_reflectance_loss_gradient(geometry_img, normal_img, rgb_img, camera_pos, x_init):

  param_names = disney_brdf.get_brdf_param_names()

  for index in range(6,7):
    name = param_names[index]  
    print("creating graph for {}".format(name))
    gradients = []    
    raw_brdf = []
    raw_gradients = []
    numerical_gradients = []
    x = copy.deepcopy(x_init)

    for i in range(4,40):
      x[index] = i/40.0
      grad = reflectance_loss_gradient(x, [geometry_img], [normal_img], [rgb_img], [camera_pos])
      print(grad[index])
      gradients.append(grad[index])

    losses = []
    for i in range(4,40):
      x[index] = i/40.0
      loss = reflectance_loss(x, [geometry_img], [normal_img], [rgb_img], [camera_pos], x[:3])        
      losses.append(loss)
  
    fig, (ax1, ax2) = plt.subplots(2,1)
    plt.title(name)
    ax1.plot(np.asarray(range(4,40))/40.0, gradients)
    ax1.set_ylabel("gradient")

    ax2.plot(np.asarray(range(4,40))/40.0, losses)
    ax2.set_ylabel("loss")

    plt.savefig("outputs/{}.png".format(name))
  
#####################################################
#################### Optimization ###################
#####################################################
 
def bird_demo():

  # note: there is some unnecessary generality here for dealing with
  # cases with multiple scans that is leftover from previous version,
  # but the diffuse color extraction is hacked to only work for single scan case.
  # I'm leaving it like this to make it easier to extend later on


  # note: we have to make some of the following variables global so that the
  # step() functions can access them
  n_params = 13
  global model
  model = "toucan_0.5"

  global experiment_num
  experiment_num = 10

  n_scans = 1
  n_scans_to_process = 1

  global geometry_imgs
  geometry_imgs = []

  global normal_imgs
  normal_imgs = []

  global rgb_imgs
  rgb_imgs = []

  global camera_positions
  camera_positions = [None] * n_scans  
  
  global timestep
  timestep = 0

  global loss_plot
  loss_plot = []

  global diffuse_loss_plot
  diffuse_loss_plot = []

  print("_____________________\nInitializing run:\n\nmodel: {}\nexperiment_num: {}\nnumber of scans: {}\nscans_to_process: {}".format(model, experiment_num, n_scans, n_scans_to_process))
  print("\nResults will be output to:\n{}".format("outputs/{}/experiment_{}".format(model, experiment_num)))
  print("_____________________\n")

  if not os.path.isdir("outputs/{}".format(model)):
    os.system("mkdir outputs/{}".format(model))
  if not os.path.isdir("outputs/{}/experiment_{}".format(model, experiment_num)):
    os.system("mkdir outputs/{}/experiment_{}".format(model, experiment_num))

  #####################################################
  ################## Load dataset #####################
  #####################################################


  # toucan_0.1
  diffuse_colors_partition = np.floor(np.asarray([
    [201.87623318,  67.87982063, 116.97713004],
    [ 12.37968305,  19.16081209, 181.90856642],
    [204.69582909, 204.77280434, 204.94438793],
    [202.75614251, 162.71990172,  28.74692875],
    [205.84931507, 204.45493872,   2.9408796 ],
    [ 97.23448276, 100.4137931,  170.79310345],
    [159.20449438, 186.37977528, 197.50337079],
    [ 21.48648649,  27.2972973,   29.27027027],
    [193.97761194, 142.94776119,  85.90298507],
    [127.58267717,  36.81102362, 167.1496063 ]
  ]))

  # toucan 0.5
  #diffuse_colors_partition = np.floor(np.asarray([
  #  [202.62710373,  67.94683473, 116.35374269],
  #  [ 12.06595837,  19.02312268, 181.99192569],
  #  [204.98807344, 205.00969201, 205.05510984],
  #  [205.94693889, 204.90470314,   2.22590235],
  #  [202.98925538, 162.10725262,  28.17731046],
  #  [  3.6136725,    4.56756757,   4.91494436],
  #  [161.2948296,  192.66323389, 200.34833846],
  #  [119.36476427, 126.97890819, 167.10545906],
  #  [ 97.81245012,  46.40542698, 168.16201117],
  #  [195.40391677, 145.60954712,  88.92411261]
  #]))

  global diffuse_color_index
  
  for c in range(len(diffuse_colors_partition)):
    diffuse_color_index = c
    diffuse_colors = diffuse_colors_partition[c]
    timestep = 0
    geometry_imgs = []
    normal_imgs = []
    rgb_imgs = []
    camera_positions = [None] * n_scans  
    loss_plot = []
    diffuse_loss_plot = []

    # read rgb, geometry, and normals
    for t in range(n_scans):      

      fname = "models/{}/{}_{}_render.png".format(model,model,t)
      rgb_img = cv2.imread(fname)      
      rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)    
      rgb_img = rgb2range(rgb_img)      
      rgb_imgs.append(rgb_img)
            
      fname = "models/{}/{}_{}_geometry.exr".format(model,model,t)
      geometry_img = cv2.imread(fname, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)            
      geometry_img = cv2.cvtColor(geometry_img, cv2.COLOR_BGR2RGB)      
      geometry_imgs.append(geometry_img)   

      fname = "models/{}/{}_{}_normals.exr".format(model,model,t)
      normal_img = cv2.imread(fname, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
      normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
      normal_imgs.append(normal_img)    

      fname = "models/{}/{}_{}_diffuse_colors_projected.png".format(model,model,t)
      diffuse_img = cv2.imread(fname, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
      diffuse_img = cv2.cvtColor(diffuse_img, cv2.COLOR_BGR2RGB)      

      # remove pixels from render that have diffuse color != c
      partition_indices = []  
      height = len(diffuse_img)
      width = len(diffuse_img[0])
      for i in range (height):
        for j in range (width):          
          if not (diffuse_img[i,j,:] == diffuse_colors).all():
            rgb_imgs[0][i,j,:] = [70/255.0, 70/255.0, 70/255.0]          
            normal_imgs[0][i,j,:] = [0,0,0]


    # read camera positions
    #fname = "models/{}/camera_pos.json".format(model,dataset_num)
    #with open(fname) as f:
    #  camera_positions_json = json.load(f)  
    
    #for pos in camera_positions_json:
    #  camera_positions[int(pos)] = camera_positions_json[pos][:3]


    # toucan camera position
    camera_positions[0] = [1.7, 0.11, 0.7]

    rgb_imgs = rgb_imgs[:n_scans_to_process]
    geometry_imgs = geometry_imgs[:n_scans_to_process]
    normal_imgs = normal_imgs[:n_scans_to_process]  
    camera_positions = camera_positions[:n_scans_to_process]

    ######################################################
    ############# Diffuse optimization ###################
    ######################################################
    
    global diffuse_data
    diffuse_data = get_diffuse_optimization_data(geometry_imgs, normal_imgs, rgb_imgs, camera_positions)  
    global use_disney_diffuse_brdf_approx
    use_disney_diffuse_brdf_approx = True  

    soln = diffuse_color_optimization(geometry_imgs, normal_imgs, rgb_imgs, camera_positions, diffuse_data)  
    diffuse_colors = soln[0]  
    print("diffuse solution: {}".format(diffuse_colors))


    ######################################################
    ############# Reflectance optimization ###############
    ######################################################

    timestep = 0
    soln = reflectance_parameter_optimization(geometry_imgs, normal_imgs, rgb_imgs, camera_positions, diffuse_colors)
    print("final solution: ")
    print_params(soln[0])  
    

    ######################################################
    ################## Render solution ###################
    ######################################################  

    print("Converged. Rendering solution")
    #for i in range(len(meshes)):
    #  mesh = disney_brdf.render_disney_brdf_on_mesh(meshes[i],camera_positions[0][:3],extract_brdf_params(soln[0]))
    #  outfname = "outputs/{}/dataset_{}/experiment_{}/final_fit_{}.ply".format(model, dataset_num, experiment_num, i)
    #  o3d.io.write_triangle_mesh(outfname, mesh)

    render = disney_brdf.render_disney_brdf_image(diffuse_colors, geometry_imgs[0], normal_imgs[0], camera_positions[0][:3], extract_brdf_params(soln[0]), False)
    render = np.round(render * 255.0)
    outfname = "outputs/{}/experiment_{}/final_fit_color{}.png".format(model, experiment_num, c)
    render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)   
    cv2.imwrite(outfname, render)

def main():
  bird_demo()

if __name__ == "__main__":
  main()

