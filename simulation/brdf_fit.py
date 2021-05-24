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


#######################################################
######### Reflectance Parameter Optimization ##########
#######################################################

def photometric_error(mesh_1, mesh_2):
  
  photo_loss = 0.0
  n_vertices = len(mesh_1.vertices)
  if (n_vertices != len(mesh_2.vertices)):
    raise Exception("Error: mesh_1 and mesh_2 must have the same geometry")
  for i in range(len(mesh_1.vertices)):
    v1 = mesh_1.vertex_colors[i]
    v2 = mesh_2.vertex_colors[i]
    photo_loss = photo_loss + np.linalg.norm(np.asarray(v1) - np.asarray(v2)) ** 2
  return photo_loss

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

def disney_brdf_loss(x, *args):
  scanned_meshes = args[0]
  camera_positions = args[1]
  brdf_params = extract_brdf_params(x)
  loss = 0.0  
  n = 0
  for mesh, camera_pos in zip(scanned_meshes, camera_positions):        
    predicted_mesh = copy.deepcopy(mesh)    
    predicted_mesh = disney_brdf.render_disney_brdf_on_mesh(predicted_mesh, camera_pos[:3], brdf_params)       
    photo_error = photometric_error(mesh, predicted_mesh)
    loss = loss + photo_error
    n = n + len(mesh.vertices)

  return 0.5 * loss / n  


# See Equation 5 in
# https://docs.google.com/document/d/1_4bEW0cvtAsq6qlvX3CmGH8zlCPL3HGEjaDz0rlP6wA/edit
def reflectance_loss_gradient(x, *args):

  scanned_meshes = args[0]
  camera_positions = args[1]  

  brdf_params = extract_brdf_params(x)
  n = 0
  loss_gradient = np.zeros( (13, 1) )

  for mesh, camera_pos in zip(scanned_meshes, camera_positions):        
      
    avg_loss_term = 0.0
    for i in range(len(mesh.vertices)):
      # N: surface normal
      N = disney_brdf.normalize(mesh.vertex_normals[i])
      p = [mesh.vertices[i][0], mesh.vertices[i][1], mesh.vertices[i][2]]
      # L: light direction (same as view direction)
      light_pos = camera_pos[:3]
      L = disney_brdf.normalize(light_pos)

      # x: surface tangent    
      U = disney_brdf.normalize(np.random.rand(3))          
      X = disney_brdf.normalize(np.cross(N, U))  

      # y: surface bitangent
      Y = disney_brdf.normalize(np.cross(N, X))

      #  V: view direction  
      V = disney_brdf.normalize(np.asarray(camera_pos[:3]) - np.asarray(p))

      # Irradiance
      irradiance_i = disney_brdf.compute_irradiance(light_pos, N, camera_pos[:3], p)      

      brdf = disney_brdf.BRDF(L, V, N, X, Y, [brdf_params['red'], brdf_params['green'], brdf_params['blue']], brdf_params['metallic'], brdf_params['subsurface'], brdf_params['specular'], brdf_params['roughness'], brdf_params['specularTint'], brdf_params['anisotropic'], brdf_params['sheen'], brdf_params['sheenTint'], brdf_params['clearcoat'], brdf_params['clearcoatGloss'])
      saturate_check = brdf * irradiance_i
      predicted_rgb = brdf * irradiance_i
    
      loss_gradient_i = brdf * irradiance_i

      predicted_rgb = np.minimum(predicted_rgb, 1.0)    
      predicted_rgb = np.power(predicted_rgb, 1.0 / 2.2)

      truth_rgb = np.asarray(mesh.vertex_colors[i])

      # note: the derivative of the projection to [0,255] and back 
      # done by BRDF needs to be captured correctly for loss_gradient_i term above as well
      # probably minor effect though
      for j in range(3):
        predicted_rgb[j] = round(predicted_rgb[j] * 255.0) / 255.0

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
      
      # set anisotropic and specularTint partials to zero
      brdf_gradient_i[7,:] = 0.0
      brdf_gradient_i[8,:] = 0.0

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
  loss = disney_brdf_loss(x, meshes, camera_positions)
  print("reflectance loss: {}".format(loss))
  print("")
  loss_plot.append(loss)
  grad = reflectance_loss_gradient(x, meshes, camera_positions)
  print("reflectance loss gradient: {}".format(grad))

  fig, ax = plt.subplots(1,1)
  plt.title("Reflectance Loss")
  ax.plot(np.asarray(range(timestep + 1)), loss_plot)
  plt.savefig("outputs/{}/dataset_{}/experiment_{}/reflectance_loss.png".format(model, dataset_num, experiment_num))

  fname = "outputs/{}/dataset_{}/experiment_{}/reflectance_loss.txt".format(model, dataset_num, experiment_num)
  with open (fname, "a") as file:
    file.write("{}\n".format(loss))

  mesh = copy.deepcopy(meshes[0]) 
  disney_brdf.render_disney_brdf_on_mesh(mesh,camera_positions[0][:3],extract_brdf_params(x))
  outfname = "outputs/{}/dataset_{}/experiment_{}/reflectance_fit_{}.ply".format(model, dataset_num, experiment_num, timestep)
  o3d.io.write_triangle_mesh(outfname, mesh)

  timestep = timestep + 1

  
def reflectance_parameter_optimization(meshes, camera_positions, diffuse_colors):
  param_bounds = disney_brdf.get_brdf_param_bounds()
  initial_guess = np.append(np.asarray(diffuse_colors), np.asarray(np.random.rand(10)))  
  initial_guess[8] = 0.0 # aniso
  initial_guess[7] = 0.0 # specularTint
  reflectance_step(initial_guess)
  factr = 1e3
  pgtol = 1e-8
  sol = scipy.optimize.fmin_l_bfgs_b(func=disney_brdf_loss, x0=initial_guess, args=(meshes, camera_positions), fprime=reflectance_loss_gradient, approx_grad=False, bounds=param_bounds, factr=factr, pgtol=pgtol, maxiter=100, callback=reflectance_step)    
  return sol


#######################################################
############# Diffuse Color Optimization ##############
#######################################################

def luminosity(x):
  p = x[1]
  return (0.3*p[0] + 0.6*p[1] + 0.1*p[2]) # same approx used in disney brdf code

# output: [ [x,y,z], [r,g,b], NdotL, [cam_x,cam_y,cam_z] ]
def get_diffuse_optimization_data(meshes, camera_positions):

  data = []
  for mesh, camera_pos in zip(meshes, camera_positions):                  
    for i in range(len(mesh.vertices)):
      vertex = np.asarray(mesh.vertices[i])
      color = np.asarray(mesh.vertex_colors[i])
      N = np.asarray(mesh.vertex_normals[i])
      NdotL = max(0, np.dot(N, np.asarray(camera_pos[:3])))
      data.append([vertex, color, N, camera_pos])  

  data.sort(key = luminosity) # sort by luminosity
  data[:len(data)/4]  # take first quartile
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
  plt.savefig("outputs/{}/dataset_{}/experiment_{}/diffuse_loss.png".format(model, dataset_num, experiment_num))

  fname = "outputs/{}/dataset_{}/experiment_{}/diffuse_loss.txt".format(model, dataset_num, experiment_num)
  with open (fname, "a") as file:
    file.write("{}\n".format(loss))

  mesh = copy.deepcopy(meshes[0])   
  disney_brdf.render_diffuse_disney_brdf_on_mesh(mesh,camera_positions[0][:3],x)
  outfname = "outputs/{}/dataset_{}/experiment_{}/diffuse_fit_{}.ply".format(model, dataset_num, experiment_num, timestep)
  o3d.io.write_triangle_mesh(outfname, mesh)  
  timestep = timestep + 1    

def diffuse_color_optimization(meshes, camera_positions, data):  
  initial_guess = np.random.rand(3)
  diffuse_step(initial_guess)
  factr = 1e3
  pgtol = 1e-4
  eps = 1e-4 # step size used by finite difference, since we don't supply custom gradient
  soln = scipy.optimize.fmin_l_bfgs_b(func=diffuse_loss, x0=initial_guess, args=(data,use_disney_diffuse_brdf_approx), fprime=None, approx_grad=True, bounds=[(0.0,1.0),(0.0,1.0),(0.0,1.0)], factr=factr, pgtol=pgtol, epsilon=eps, maxiter=100, callback=diffuse_step)  
  return soln


#######################################################
################## Gradient Tests #####################
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


def plot_reflectance_loss_gradient(meshes, camera_pos, x_init):

  param_names = disney_brdf.get_brdf_param_names()

  for index in range(3,13):
    name = param_names[index]  
    print("creating graph for {}".format(name))
    gradients = []    
    raw_brdf = []
    raw_gradients = []
    numerical_gradients = []
    x = copy.deepcopy(x_init)

    for i in range(4,40):
      x[index] = i/40.0
      grad = reflectance_loss_gradient(x, meshes, camera_positions)
      print(grad[index])
      gradients.append(grad[index])

    losses = []
    for i in range(4,40):
      x[index] = i/40.0
      loss = disney_brdf_loss(x, meshes, camera_positions)        
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
 
def main():

  # note: we have to make some of the following variables global so that the
  # step() functions can access them
  n_params = 13
  global model
  model = "balustervase_0.2"
  global dataset_num
  dataset_num = 13
  global experiment_num
  experiment_num = 1
  n_scans = 13
  n_scans_to_process = 1
  global meshes
  meshes = []
  global timestep
  timestep = 0
  global loss_plot
  loss_plot = []
  global diffuse_loss_plot
  diffuse_loss_plot = []

  print("_____________________\nInitializing run:\n\nmodel: {}\ndataset_num: {}\nexperiment_num: {}\nnumber of scans: {}\nscans_to_process: {}".format(model, dataset_num, experiment_num, n_scans, n_scans_to_process))
  print("\nResults will be output to:\n{}".format("outputs/{}/dataset_{}/experiment_{}".format(model, dataset_num, experiment_num)))
  print("_____________________\n")

  if not os.path.isdir("outputs/{}".format(model)):
    os.system("mkdir outputs/{}".format(model))
  if not os.path.isdir("outputs/{}/dataset_{}".format(model, dataset_num)):
    os.system("mkdir outputs/{}/dataset_{}".format(model, dataset_num))
  if not os.path.isdir("outputs/{}/dataset_{}/experiment_{}".format(model, dataset_num, experiment_num)):
    os.system("mkdir outputs/{}/dataset_{}/experiment_{}".format(model, dataset_num, experiment_num))

  #####################################################
  ################## Load dataset #####################
  #####################################################

  # read meshes  
  for t in range(n_scans):
    fname = "models/{}/dataset_{}/{}_{}_mesh_rendered.ply".format(model,dataset_num,model,t)
    mesh = o3d.io.read_triangle_mesh(fname)
    mesh.compute_vertex_normals()    
    meshes.append(mesh)

  # read camera positions
  fname = "models/{}/dataset_{}/camera_pos.json".format(model,dataset_num)
  with open(fname) as f:
    camera_positions_json = json.load(f)  

  
  global camera_positions
  camera_positions = [None] * n_scans  
  for pos in camera_positions_json:
    camera_positions[int(pos)] = camera_positions_json[pos][:3]

  meshes = meshes[:n_scans_to_process]
  camera_positions = camera_positions[:n_scans_to_process]
  
  ######################################################
  ############# Diffuse optimization ###################
  ######################################################
  
  global diffuse_data
  diffuse_data = get_diffuse_optimization_data(meshes, camera_positions)  
  global use_disney_diffuse_brdf_approx
  use_disney_diffuse_brdf_approx = True  

  soln = diffuse_color_optimization(meshes, camera_positions, diffuse_data)  
  diffuse_colors = soln[0]  
  print("diffuse solution: {}".format(diffuse_colors))


  ######################################################
  ############# Reflectance optimization ###############
  ######################################################

  timestep = 0
  soln = reflectance_parameter_optimization(meshes, camera_positions, diffuse_colors)
  print("final solution: ")
  print_params(soln[0])  
  

  ######################################################
  ################## Render solution ###################
  ######################################################  

  print("Converged. Writing meshes")
  for i in range(len(meshes)):
    mesh = disney_brdf.render_disney_brdf_on_mesh(meshes[i],camera_positions[0][:3],extract_brdf_params(soln[0]))
    outfname = "outputs/{}/dataset_{}/experiment_{}/final_fit_{}.ply".format(model, dataset_num, experiment_num, i)
    o3d.io.write_triangle_mesh(outfname, mesh)

if __name__ == "__main__":
  main()

