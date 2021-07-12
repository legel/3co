import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import numpy as np
import cv2
from math import pi
from pathlib import Path
import os, sys
import copy
import time

@tf.function(jit_compile=True)
def sqr(x):
  return tf.math.square(x)

@tf.function(jit_compile=True)
def clamp(x, a, b):
  absolute_min = tf.constant(a, dtype=tf.float64)
  absolute_max = tf.constant(b, dtype=tf.float64)
  x = tf.math.minimum(x, absolute_max)
  x = tf.math.maximum(absolute_min, x)
  return x

@tf.function(jit_compile=True)
def normalize(x):
  norm = tf.linalg.norm(x, axis=1)
  ones = tf.ones(x.shape[0], dtype=tf.float64)
  norm = tf.where(norm == 0.0, ones, norm)
  norm = tf.broadcast_to(tf.expand_dims(norm, axis=1), [x.shape[0], 3])
  result = x / norm
  result = tf.cast(result, dtype=tf.float64)
  return result

@tf.function(jit_compile=True)
def mix(x, y, a):
  return x * (1 - a) + y * a

@tf.function(jit_compile=True)
def SchlickFresnel(u):
  m = clamp(1-u, 0, 1)
  return tf.pow(m, 5)

@tf.function(jit_compile=True)
def GTR1(NdotH, a):
  number_of_pixels = NdotH.shape[0]
  if (a >= 1): 
    # value = tf.cast(1/pi, dtype=tf.float64)
    return tf.fill(dims=[number_of_pixels], value=tf.constant(1/pi, tf.float64))

  a2 = tf.cast(a*a, dtype=tf.float64)
  t = tf.cast(1 + (a2-1)*NdotH*NdotH, dtype=tf.float64)
  return (a2-1) / (pi*tf.math.log(a2)*t)

@tf.function(jit_compile=True)
def GTR2_aniso(NdotH, HdotX, HdotY, ax, ay):
  shape = tf.shape(NdotH)
  ones = tf.ones(shape, dtype=tf.float64)
  return ones / ( pi * ax * ay * sqr( sqr(HdotX/ax) + sqr(HdotY/ay) + sqr(NdotH)))

@tf.function(jit_compile=True)
def smithG_GGX(Ndotv, alphaG):
  a = tf.cast(sqr(alphaG), dtype=tf.float64)
  b = tf.cast(sqr(Ndotv), dtype=tf.float64)
  sqr_root = tf.math.sqrt(a + b - a * b)
  noemer = tf.math.add(Ndotv, sqr_root)
  teller = tf.constant(1, dtype=tf.float64)
  return teller / noemer

@tf.function(jit_compile=True)
def d_GGX_aG(NdotA, aG):
  k = tf.math.sqrt( sqr(aG) + sqr(NdotA) - sqr(aG) * sqr(NdotA) )
  return aG * (sqr(NdotA) - 1.0) / (k * sqr((NdotA + k)))

@tf.function(jit_compile=True)
def smithG_GGX_aniso(NdotV, VdotX, VdotY, ax, ay):
  return 1 / (NdotV + tf.math.sqrt( sqr(VdotX*ax) + sqr(VdotY*ay) + sqr(NdotV) ))

@tf.function(jit_compile=True)
def mon2lin(x):
  x = tf.math.pow(x, 2.2)
  return x

def mask_data(data, condition, replace_to="zeros"):  
  original_data_shape = data.shape
  if len(original_data_shape) == 0:
    return data
  if len(original_data_shape) == 2:
    data = tf.broadcast_to(tf.expand_dims(data, axis=2), [1024, 1024,3])
  if replace_to == "zeros":
    if len(original_data_shape) > 0:
      replacement_data = tf.zeros(data.shape, dtype=tf.float64)
  masked_data = tf.where(condition, replacement_data, data)
  if len(original_data_shape) == 2:
    masked_data = masked_data[:,:,0]
  return masked_data

@tf.function(jit_compile=True)
def brdf_gradient( L, V, N, X, Y, diffuse, metallic = 0, subsurface = 0, specular = 0.5,
roughness = 0.5, specularTint = 0, anisotropic = 0, sheen = 0, sheenTint = 0.5, clearcoat = 0, clearcoatGloss = 1.0 ):

  L, V, N, X, Y, NdotL, NdotV, NdotH, LdotH, C_d, Cdlum, C_tint, C_spec0,\
     C_sheen, F_L, F_V, F_d90, F_d, F_ss90, F_ss, ss, anisotropic, aspect, ax, ay,\
       D_s, F_H, F_s, aG, G_s, F_sheen, D_r, F_r, G_r, brdf = BRDF(L, V, N, X, Y, diffuse, metallic, subsurface, specular,
	roughness, specularTint, anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss)  

  # L, V, N, X, Y, NdotL, NdotV, NdotH, LdotH, C_d, Cdlum, C_tint, C_spec0,\
  #    C_sheen, F_L, F_V, F_d90, F_d, F_ss90, F_ss, ss, anisotropic, aspect, ax, ay,\
  #      D_s, F_H, F_s, aG, G_s, F_sheen, D_r, F_r, G_r, brdf = BRDF(L, V, N, X, Y, diffuse, baseColor, metallic, subsurface, specular,
	# roughness, specularTint, anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss)

  H = normalize(L+V)
  HdotX = tf.reduce_sum(tf.math.multiply(H, X), axis=2)
  HdotY = tf.reduce_sum(tf.math.multiply(H, Y), axis=2)
  
  ## metallic ## 
  right_d_Fs_metallic = tf.expand_dims((1.0 - F_H), axis=2)
  right_d_Fs_metallic = tf.broadcast_to(right_d_Fs_metallic, [1024, 1024, 3])
  d_Fs_metallic = C_d - 0.08 * specular * mix(tf.ones((1024, 1024, 3), dtype=tf.float64), C_tint, specularTint) * right_d_Fs_metallic

  #print(d_Fs_metallic)

  NdotL3d = tf.broadcast_to(tf.expand_dims(NdotL, axis=2), [1024, 1024, 3])
  mix_f_d_ss_subsurface = tf.broadcast_to(tf.expand_dims(mix(F_d, ss, subsurface), axis=2), [1024, 1024, 3])
  G_s_D_s = tf.broadcast_to(tf.expand_dims(G_s * D_s, axis=2), [1024, 1024, 3])

  d_f_metallic = NdotL3d * ((-1.0 / pi) * mix_f_d_ss_subsurface * C_d + F_sheen + G_s_D_s * d_Fs_metallic)
  ## metallic ## 
  
  ## subsurface ## 
  left = tf.broadcast_to(tf.expand_dims(NdotL * (1.0 / pi) * (1.0 - metallic) * (ss - F_d), axis=2), [1024, 1024, 3])
  d_f_subsurface = left * C_d
  ## subsurface ##

  ## specular ##  
  left = tf.broadcast_to(tf.expand_dims(NdotL * G_s * D_s * (1.0 - F_H) * (1.0 - metallic) * 0.08, axis=2), [1024, 1024, 3])
  d_f_specular = left * mix(tf.ones(3, dtype=tf.float64), C_tint, specularTint)
  ## specular ##  

  ## roughness ##  
  d_ss_roughness = 1.25 * LdotH * LdotH * (F_V - 2.0*F_L*F_V + F_L + 2.0 * LdotH * LdotH * F_L * F_V * roughness )

  d_Fd_roughness = 2.0 * LdotH ** 2 * (F_V + F_L + 2.0 * F_L * F_V * (F_d90 - 1.0))
  
  d_Gs_roughness = 0.5 * (roughness + 1.0) * (d_GGX_aG(NdotL, aG) * smithG_GGX(NdotV, aG) + d_GGX_aG(NdotV, aG) * smithG_GGX(NdotL, aG) )    

  roughness = tf.cond(roughness <= 0, lambda: 0.001, lambda: roughness)
  
  D_s_expand = tf.broadcast_to(tf.expand_dims(D_s, axis=2), [1024, 1024, 3])
  G_s_expand = tf.broadcast_to(tf.expand_dims(G_s, axis=2), [1024, 1024, 3])
  c = tf.convert_to_tensor(sqr(HdotX) / sqr(ax) + sqr(HdotY) / sqr(ay) + sqr(NdotH), dtype=tf.float64)

  d_Ds_roughness = 4.0 * ( (2.0 *  (HdotX**2 * aspect**4 + HdotY ** 2) / (aspect**2 * roughness)) - c * roughness**3) / (pi * ax**2 * ay**2 * c**3)
  left = tf.broadcast_to(tf.expand_dims(NdotL * (1.0 - metallic) * (1.0 / pi) * mix(d_Fd_roughness, d_ss_roughness, subsurface), axis=2), [1024, 1024, 3])
  right = tf.broadcast_to(tf.expand_dims((d_Gs_roughness * D_s + d_Ds_roughness * G_s), axis=2), [1024, 1024, 3])
  d_f_roughness = left * C_d + F_s * right
  ## roughness ##  

  ## specularTint ##  
  middle = tf.broadcast_to(tf.expand_dims((1.0 - F_H) * specular * 0.08 * (1.0 - metallic), axis=2), [1024, 1024, 3])
  d_f_specularTint = NdotL3d * G_s_expand * D_s_expand * middle * (C_tint - 1.0)
  ## specularTint ## 

  ## anisotropic ## 
  d_GTR2aniso_aspect = 4.0 * (sqr(HdotY) - sqr(HdotX) * tf.math.pow(aspect,4)) / (pi * sqr(ax) * sqr(ay) * tf.math.pow(c,3) * tf.math.pow(aspect,3))
  d_Ds_anisotropic = (-0.45 / aspect) * (d_GTR2aniso_aspect)
  aniso_left = tf.broadcast_to(tf.expand_dims(NdotL * G_s * d_Ds_anisotropic, axis=2), [1024, 1024, 3])
  d_f_anisotropic = aniso_left * F_s
  ## anisotropic ## 

  ## sheen ## 
  F_H_expand = tf.broadcast_to(tf.expand_dims(F_H, axis=2), [1024, 1024, 3])
  d_f_sheen = NdotL3d * (1.0 - metallic) * F_H_expand * C_sheen
  ## sheen ## 

  ## sheenTint ## 
  d_f_sheenTint = NdotL3d * (1.0 - metallic) * F_H_expand * sheen * (C_tint - 1.0)
  ## sheenTint ##   

  ## clearcoat ##   
  d_f_clearcoat = NdotL * 0.25 * G_r * F_r * D_r * 1.0
  d_f_clearcoat = tf.broadcast_to(tf.expand_dims(d_f_clearcoat, axis=2), [1024, 1024, 3])
  ## clearcoat ##   

  ## clearcoatGloss ##   
  a = mix(0.1,.001,clearcoatGloss)
  t = 1.0 + (sqr(a) - 1.0) * sqr(NdotH)
  d_GTR1_a = 2.0 * a * ( tf.math.log(sqr(a)) * t - (sqr(a) - 1.0) * (t/(sqr(a)) + tf.math.log(sqr(a)) * sqr(NdotH))) / (pi * sqr((tf.math.log(sqr(a)) * t))  )  
  d_f_clearcoatGloss = NdotL * 0.25 * clearcoat * -0.099 * G_r * F_r * d_GTR1_a
  d_f_clearcoatGloss = tf.broadcast_to(tf.expand_dims(d_f_clearcoatGloss, axis=2), [1024, 1024, 3])
  d_f_clearcoat= tf.ones((1024, 1024, 3), dtype=tf.float64) * d_f_clearcoat
  d_f_clearcoatGloss = tf.ones((1024, 1024, 3), dtype=tf.float64) * d_f_clearcoatGloss
  ## clearcoatGloss ##   

  a, b, c = tf.zeros((1024, 1024, 3), dtype=tf.float64), tf.zeros((1024, 1024, 3), dtype=tf.float64), tf.zeros((1024, 1024, 3), dtype=tf.float64)
  
  # names = ['metallic_loss','subsurface_loss','specular_loss','roughness_loss','specularTint_loss','anisotropic_loss','sheen_loss','sheenTint_loss','clearcoat_loss','clearcoatGloss_loss']
  # for i,(name,thing) in enumerate(zip(names, [d_f_metallic, d_f_subsurface, d_f_specular,d_f_roughness, d_f_specularTint, d_f_anisotropic, d_f_sheen, d_f_sheenTint,d_f_clearcoat, d_f_clearcoatGloss])):
  #   loss = np.array(thing * 255.0, dtype=np.float32)
  #   loss = cv2.cvtColor(loss, cv2.COLOR_RGB2BGR)
  #   cv2.imwrite(f'models/toucan_0.5/def_brdf_gradient/{name}.png', loss)

  return [d_f_metallic, d_f_subsurface, d_f_specular,d_f_roughness, d_f_specularTint, d_f_anisotropic, d_f_sheen, d_f_sheenTint,d_f_clearcoat, d_f_clearcoatGloss]

def brdf_gradient_wrapper(L,V,N,X,Y, diffuse, brdf_params):
  # baseColor = np.asarray([brdf_params['red'], brdf_params['green'], brdf_params['blue']])
  metallic = brdf_params['metallic']
  subsurface = brdf_params['subsurface'] 
  specular = brdf_params['specular'] 
  roughness = brdf_params['roughness'] 
  specularTint = brdf_params['specularTint'] 
  anisotropic = brdf_params['anisotropic'] 
  sheen = brdf_params['sheen'] 
  sheenTint = brdf_params['sheenTint']
  clearcoat = brdf_params['clearcoat'] 
  clearcoatGloss = brdf_params['clearcoatGloss']  
  # return brdf_gradient(L, V, N, X, Y, diffuse, baseColor, metallic, subsurface, specular, roughness, specularTint, anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss)  
  return brdf_gradient(L, V, N, X, Y, diffuse, metallic, subsurface, specular, roughness, specularTint, anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss)  

@tf.function(jit_compile=True)
def photometric_error(ground_truth, hypothesis):
  error = (hypothesis - ground_truth)

  # visualize_image_condition(data=error, label="loss_function_error", condition="is_nan", is_true="white", is_false="black")
  # visualize_image_condition(data=error, label="loss_function_error", condition="is_inf", is_true="white", is_false="black")

  return error # tf.abs(ground_truth - hypothesis)


def visualize_image_condition(data, label, condition="is_nan", is_true="white", is_false="black"):
  global iteration_number

  if len(data.shape) == 2:
    data = tf.broadcast_to(tf.expand_dims(data, axis=2), [1024, 1024,3])

  elif len(data.shape) == 0:
    return

  if is_true == "white":
    is_true_color = tf.fill(dims=[1024, 1024, 3], value=tf.constant(255.0, tf.float64))

  if is_false == "black":
    is_false_color = tf.zeros([1024, 1024, 3], dtype=tf.float64)

  if condition == "is_nan":
    conditional_data = tf.where(tf.math.is_nan(data), is_true_color, is_false_color)
  elif condition == "is_inf":
    conditional_data = tf.where(tf.math.is_inf(data), is_true_color, is_false_color)

  conditional_data = np.array(conditional_data, dtype=np.float32)
  conditional_data = cv2.cvtColor(conditional_data, cv2.COLOR_RGB2BGR)
  cv2.imwrite('models/toucan_0.5/def_brdf_gradient/{}_{}_{}.png'.format(iteration_number,label,condition), conditional_data)


# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer#example_2  
@tf.function(jit_compile=True)
def lossgradient_hypothesis(scene):

  [width, height, _] = tf.shape(diffuse_colors)
  total = tf.cast(width * height, dtype=tf.float64)
  brdf_params = brdf_hypothesis
  
  scene = render(scene)

  # gradient loss attributable to gamma encoding 
  loss_radiance = tf.math.divide(tf.math.pow(scene["brdf"] * scene["irradiance"], -1.2 / 2.2), 2.2) * scene["irradiance"]
  scene_data["loss_radiance"] = tf.where(background_mask, surface_diffuse_colors, loss_radiance)                        


  grey = tf.constant([70/255,70/255,70/255], tf.float64)

  # # pixelwise difference across rgb channels                         
  loss = tf.reduce_sum(photometric_error(ground_truth, hypothesis), axis = 2)

  brdf_gradients = brdf_gradient_wrapper(L=L, V=V, N=N, X=X, Y=Y, diffuse=diffuse_colors, brdf_params=brdf_params)

  loss_radiance = tf.reduce_sum(tf.math.multiply(brdf_gradients, loss_radiance), axis=3) #* loss

  red_color = tf.fill(dims=[1024, 1024, 3], value=tf.constant(255.0, tf.float64))
  zeros=tf.zeros([width, height, 3], dtype=tf.float64)

  # # average gradient over all pixels
  metallic_loss = tf.broadcast_to(tf.expand_dims(loss_radiance[0], axis=2), [1024, 1024,3])
  metallic_loss = tf.where(ground_truth==grey, zeros, metallic_loss)
  # visualize_image_condition(data=metallic_loss, label="metallic_loss", condition="is_nan", is_true="white", is_false="black")
  metallic_loss = (tf.reduce_sum(metallic_loss) / total) / 3.0

  subsurface_loss = tf.broadcast_to(tf.expand_dims(loss_radiance[1], axis=2), [1024, 1024,3])
  subsurface_loss = tf.where(ground_truth==grey, zeros, subsurface_loss)
  # visualize_image_condition(data=subsurface_loss, label="subsurface_loss", condition="is_nan", is_true="white", is_false="black")
  subsurface_loss = (tf.reduce_sum(subsurface_loss) / total) / 3.0

  specular_loss = tf.broadcast_to(tf.expand_dims(loss_radiance[2], axis=2), [1024, 1024,3])
  specular_loss = tf.where(ground_truth==grey, zeros, specular_loss)
  # visualize_image_condition(data=specular_loss, label="specular_loss", condition="is_nan", is_true="white", is_false="black")
  specular_loss = (tf.reduce_sum(specular_loss) / total) / 3.0

  roughness_loss = tf.broadcast_to(tf.expand_dims(loss_radiance[3], axis=2), [1024, 1024,3])
  roughness_loss = tf.where(ground_truth==grey, zeros, roughness_loss)
  # visualize_image_condition(data=roughness_loss, label="roughness_loss", condition="is_nan", is_true="white", is_false="black")
  roughness_loss = (tf.reduce_sum(roughness_loss) / total) / 3.0

  specularTint_loss = tf.broadcast_to(tf.expand_dims(loss_radiance[4], axis=2), [1024, 1024,3])
  specularTint_loss = tf.where(ground_truth==grey, zeros, specularTint_loss)
  # visualize_image_condition(data=specularTint_loss, label="specularTint_loss", condition="is_nan", is_true="white", is_false="black")
  specularTint_loss = (tf.reduce_sum(specularTint_loss) / total) / 3.0

  anisotropic_loss = tf.broadcast_to(tf.expand_dims(loss_radiance[5], axis=2), [1024, 1024,3])
  anisotropic_loss = tf.where(ground_truth==grey, zeros, anisotropic_loss)
  # visualize_image_condition(data=anisotropic_loss, label="anisotropic_loss", condition="is_nan", is_true="white", is_false="black")
  anisotropic_loss = (tf.reduce_sum(anisotropic_loss) / total) / 3.0

  sheen_loss = tf.broadcast_to(tf.expand_dims(loss_radiance[6], axis=2), [1024, 1024,3])
  sheen_loss = tf.where(ground_truth==grey, zeros, sheen_loss)
  # visualize_image_condition(data=sheen_loss, label="sheen_loss", condition="is_nan", is_true="white", is_false="black")
  sheen_loss = (tf.reduce_sum(sheen_loss) / total) / 3.0

  sheenTint_loss = tf.broadcast_to(tf.expand_dims(loss_radiance[7], axis=2), [1024, 1024,3])
  sheenTint_loss = tf.where(ground_truth==grey, zeros, sheenTint_loss)
  # visualize_image_condition(data=sheenTint_loss, label="sheenTint_loss", condition="is_nan", is_true="white", is_false="black")
  sheenTint_loss = (tf.reduce_sum(sheenTint_loss) / total) / 3.0

  clearcoat_loss = tf.broadcast_to(tf.expand_dims(loss_radiance[8], axis=2), [1024, 1024,3])
  clearcoat_loss = tf.where(ground_truth==grey, zeros, clearcoat_loss)
  # visualize_image_condition(data=clearcoat_loss, label="clearcoat_loss", condition="is_nan", is_true="white", is_false="black")
  clearcoat_loss = (tf.reduce_sum(clearcoat_loss) / total) / 3.0

  clearcoatGloss_loss = tf.broadcast_to(tf.expand_dims(loss_radiance[9], axis=2), [1024, 1024,3])
  clearcoatGloss_loss = tf.where(ground_truth==grey, zeros, clearcoatGloss_loss)
  # visualize_image_condition(data=clearcoatGloss_loss, label="clearcoatGloss_loss", condition="is_nan", is_true="white", is_false="black")
  clearcoatGloss_loss = (tf.reduce_sum(clearcoatGloss_loss) / total) / 3.0

  loss_gradients = [metallic_loss,\
    subsurface_loss,specular_loss,\
    roughness_loss,specularTint_loss,\
    anisotropic_loss,sheen_loss,sheenTint_loss,\
    clearcoat_loss,clearcoatGloss_loss]

  return loss_gradients, hypothesis

@tf.function(jit_compile=True)
def inverse_render_optimization(render_parameters):
  
  # initialize output directories
  Path("{}/inverse_render_hypotheses".format(folder)).mkdir(parents=True, exist_ok=True)
  Path("{}/debugging_visualizations".format(folder)).mkdir(parents=True, exist_ok=True)

  number_of_iterations = 1000
  exponential_decay_learning = [1.0 * (1.0**i) for i in range(number_of_iterations)]
  brdf_hypothesis = {}
  brdf_hypothesis['metallic'] = 0.00
  brdf_hypothesis['subsurface'] = 0.00
  brdf_hypothesis['specular'] = 0.5
  brdf_hypothesis['roughness'] = 0.9
  brdf_hypothesis['specularTint'] = 0.0
  brdf_hypothesis['anisotropic'] = 0.0
  brdf_hypothesis['sheen'] = 0.0
  brdf_hypothesis['sheenTint'] = 0.0
  brdf_hypothesis['clearcoat'] = 0.0
  brdf_hypothesis['clearcoatGloss'] = 0.0
  
  print("\nInitial hypothesis:")
  for parameter in brdf_hypothesis:
    print("{}: {:.6f}".format(parameter, brdf_hypothesis[parameter]))
    # print(f'{parameter}: {brdf_hypothesis[parameter]} (Loss: {loss_gradient})')


  for iteration in range(number_of_iterations):
    global iteration_number
    iteration_number = iteration

    print('-----------------------------------')
    print(f'         Iteration {iteration}     ')
    loss_gradients, hypothesis = lossgradient_hypothesis(diffuse_colors, xyz_coordinates, normals, camera_pos, brdf_hypothesis, ground_truth)



    #render = np.array(hypothesis * 255.0, dtype=np.float32)
    #render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
    #cv2.imwrite(f'outputs/optimization/optimization_iter_{iteration}.png', render)


    [metallic_loss,subsurface_loss,specular_loss,roughness_loss,\
      specularTint_loss,anisotropic_loss,sheen_loss,sheenTint_loss,\
        clearcoat_loss,clearcoatGloss_loss] = loss_gradients

    brdf_hypothesis['metallic'] = clamp(brdf_hypothesis['metallic'] - exponential_decay_learning[iteration] * metallic_loss, 0, 1) ## clamp to keep brdf params in [0,1]
    brdf_hypothesis['subsurface'] = clamp(brdf_hypothesis['subsurface'] - exponential_decay_learning[iteration] *  subsurface_loss, 0, 1)
    brdf_hypothesis['specular'] = clamp(brdf_hypothesis['specular'] - exponential_decay_learning[iteration] * specular_loss, 0, 1)
    brdf_hypothesis['roughness'] = clamp(brdf_hypothesis['roughness'] - exponential_decay_learning[iteration] * roughness_loss, 0, 1)
    brdf_hypothesis['specularTint'] = clamp(brdf_hypothesis['specularTint'] - exponential_decay_learning[iteration] * specularTint_loss, 0, 1)
    brdf_hypothesis['anisotropic'] = clamp(brdf_hypothesis['anisotropic'] - exponential_decay_learning[iteration] * anisotropic_loss, 0, 1)
    brdf_hypothesis['sheen'] = clamp(brdf_hypothesis['sheen'] - exponential_decay_learning[iteration] * sheen_loss, 0, 1)
    brdf_hypothesis['sheenTint'] = clamp(brdf_hypothesis['sheenTint'] - exponential_decay_learning[iteration] * sheenTint_loss, 0, 1)
    brdf_hypothesis['clearcoat'] = clamp(brdf_hypothesis['clearcoat'] - exponential_decay_learning[iteration] * clearcoat_loss, 0, 1)
    brdf_hypothesis['clearcoatGloss'] = clamp(brdf_hypothesis['clearcoatGloss'] - exponential_decay_learning[iteration] * clearcoatGloss_loss, 0, 1)

    print("\nLearning rate multiplier: {}\n".format(exponential_decay_learning[iteration]))
    for parameter, loss_gradient in zip(brdf_hypothesis, loss_gradients):
      print("{}: {:.6f} (Loss Gradient: {:.6f})".format(parameter, brdf_hypothesis[parameter], -1*loss_gradient))
      # print(f'{parameter}: {brdf_hypothesis[parameter]} (Loss: {loss_gradient})')
    print('-----------------------------------')

@tf.function(jit_compile=True)
def BRDF( diffuse_colors,
          surface_xyz,
          normals, 
          surface_tangents,
          surface_bitangents,
          light_angles,
          view_angles,
          metallic,
          subsurface,
          specular, 
          roughness,
          specularTint,
          anisotropic,
          sheen,
          sheenTint,
          clearcoat,
          clearcoatGloss):

  # Moving to mathematical notation used by Innmann et al. 2020: https://openaccess.thecvf.com/content_WACV_2020/papers/Innmann_BRDF-Reconstruction_in_Photogrammetry_Studio_Setups_WACV_2020_paper.pdf
  L = normalize(light_angles)
  V = normalize(view_angles)
  N = normalize(normals)
  X = normalize(surface_tangents)
  Y = normalize(surface_bitangents)

  number_of_pixels = diffuse_colors.shape[0]

  # compute dot products between surface normals and lighting, as well as surface normals and viewing angles
  NdotL = tf.reduce_sum(tf.math.multiply(N, L), axis=1)
  NdotV = tf.reduce_sum(tf.math.multiply(N, V), axis=1)                 

  # integrate viewing and lighting angle with respect to normals 
  H = normalize(tf.math.add(L, V)) 
  NdotH = tf.reduce_sum(tf.math.multiply(N, H), axis=1)
  LdotH = tf.reduce_sum(tf.math.multiply(L, H), axis=1)

  # aproximate luminance
  Cdlin = mon2lin(diffuse_colors)
  Cdlum =  0.3*Cdlin[:,0] + 0.6*Cdlin[:,1]  + 0.1*Cdlin[:,2]
  Cdlum_exp = tf.broadcast_to(tf.expand_dims(Cdlum, axis=1), [number_of_pixels, 3])
  Ctint = tf.where(Cdlum_exp > 0, Cdlin/Cdlum_exp, tf.ones((number_of_pixels, 3), dtype=tf.float64))
  Cspec0 = mix(specular * .08 * mix(tf.ones((number_of_pixels, 3), dtype=tf.float64), Ctint, specularTint), Cdlin, metallic)
  Csheen = mix(tf.ones((number_of_pixels, 3), dtype=tf.float64), Ctint, sheenTint) 

  # Diffuse Ffresnel - go from 1 at normal incidence to 0.5 at grazing, and mix in diffuse retro-reflection based on roughness
  FL = SchlickFresnel(NdotL)
  FV = SchlickFresnel(NdotV)
  Fd90 = 0.5 + 2 * sqr(LdotH) * roughness 
  Fd = mix(1, Fd90, FL) * mix(1, Fd90, FV)

  # Based on Hanrahan-Krueger BRDF approximation of isotropic BSSRDF; 1.25 scale is used to (roughly) preserve albedo; Fss90 used to "flatten" retroreflection based on roughness
  Fss90 = LdotH*LdotH*roughness
  Fss = mix(1, Fss90, FL) * mix(1, Fss90, FV)
  ss = 1.25 * (Fss * (1 / (NdotL + NdotV) - .5) + .5)

  # Specular
  anisotropic = tf.math.minimum(tf.constant(0.99, dtype=tf.float64), anisotropic) # added this to prevent division by zero
  aspect = tf.math.sqrt(1-anisotropic*.9)
  ax = tf.math.maximum(tf.constant(.001, dtype=tf.float64), sqr(roughness)/aspect)
  ay = tf.math.maximum(tf.constant(.001, dtype=tf.float64), sqr(roughness)*aspect)

  HdotX = tf.reduce_sum(H * X, axis=1)
  HdotY = tf.reduce_sum(H * Y, axis=1)
  Ds = GTR2_aniso(NdotH, HdotX, HdotY, ax, ay)
  Ds_exp = tf.broadcast_to(tf.expand_dims(Ds, axis=1),[number_of_pixels, 3])
  FH = SchlickFresnel(LdotH)
  FH_exp = tf.broadcast_to(tf.expand_dims(FH, axis=1),[number_of_pixels, 3])
  Fs = mix(Cspec0, tf.ones((number_of_pixels, 3), dtype=tf.float64), FH_exp)

  aG = tf.cast(sqr((0.5 * (roughness + 1))), dtype=tf.float64)
  Gs = smithG_GGX(NdotL, aG) * smithG_GGX(NdotV, aG)
  Gs_exp = tf.broadcast_to(tf.expand_dims(Gs, axis=1),[number_of_pixels, 3])

  # Sheen
  Fsheen = FH_exp * sheen * Csheen

  # Clearcoat (IOR = 1.5 -> F0 = 0.04)
  Dr = GTR1(NdotH, mix(.1,.001,clearcoatGloss))
  Fr = mix(.04, 1, FH)
  Gr = smithG_GGX(NdotL, .25) * smithG_GGX(NdotV, .25)

  mix_fd_ss_subs = mix(Fd, ss, subsurface)
  mix_fd_ss_subs = tf.broadcast_to(tf.expand_dims(mix_fd_ss_subs,axis=1),[number_of_pixels, 3])
  Cdlin_mix_fd = Cdlin * mix_fd_ss_subs

  clearcoat_gr_fr_dr = .25 * clearcoat*Gr*Fr*Dr 
  clearcoat_gr_fr_dr = tf.broadcast_to(tf.expand_dims(clearcoat_gr_fr_dr,axis=1),[number_of_pixels, 3])

  brdf = ((1/pi) * Cdlin_mix_fd + Fsheen) * (1-metallic) + clearcoat_gr_fr_dr + Gs_exp*Fs*Ds_exp

  # outputs = { "NdotL": NdotL, 
  #             "NdotV": NdotV, 
  #             "NdotH": NdotH, 
  #             "LdotH": LdotH, 
  #             "Cdlin": Cdlin, 
  #             "Cdlum": Cdlum, 
  #             "Ctint": Ctint, 
  #             "Cspec0": Cspec0, 
  #             "Csheen": Csheen, 
  #             "FL": FL, 
  #             "FV": FV, 
  #             "Fd90": Fd90, 
  #             "Fd": Fd, 
  #             "Fss90": Fss90, 
  #             "Fss": Fss, 
  #             "ss": ss, 
  #             "anisotropic": anisotropic, 
  #             "aspect": aspect, 
  #             "ax": ax, 
  #             "ay": ay,
  #             "Ds": Ds, 
  #             "FH": FH, 
  #             "Fs": Fs, 
  #             "aG": aG, 
  #             "Gs": Gs, 
  #             "Fsheen": Fsheen, 
  #             "Dr": Dr, 
  #             "Fr": Fr, 
  #             "Gr": Gr, 
  #             "brdf": brdf }

  # for output_name, output_value in outputs.items():
  #   scene[output_name] = output_value
  #   #output_value = mask_data(data=output_value, condition=diffuse==background, replace_to="zeros")
  #   # visualize_image_condition(data=output_value, condition="is_inf", label=output_name)
  #   # visualize_image_condition(data=output_value, condition="is_nan", label=output_name)
  #   # scene[output_name] = output_value

  return brdf

@tf.function(jit_compile=True)
def compute_radiance(surface_xyz, normals, light_angles, light_xyz, light_color, brdf):
  number_of_pixels = surface_xyz.shape[0]
  # approximate light intensity requirements
  light_distance_metric = light_xyz - tf.constant([1, 1, 0], dtype=tf.float64)
  light_intensity_scale = tf.reduce_sum(tf.multiply(light_distance_metric, light_distance_metric))
  light_intensity = light_color * light_intensity_scale
  # compute angles between surface normal geometry and lighting incidence
  cosine_term = tf.reduce_sum(tf.math.multiply(normals, light_angles), axis=1)
  cosine_term = tf.math.maximum(tf.cast(0.5, dtype=tf.float64), tf.cast(cosine_term, dtype=tf.float64))  # hack!
  # compute orientations from lighting to surface positions
  vector_light_to_surface = light_xyz - surface_xyz
  light_to_surface_distance_squared = tf.reduce_sum(tf.math.multiply(vector_light_to_surface, vector_light_to_surface), axis=1)
  light_intensity = tf.fill(dims=[number_of_pixels], value=tf.cast(light_intensity, dtype=tf.float64))
  irradiance = light_intensity / light_to_surface_distance_squared * cosine_term
  # compute irradiance for all points on surface
  irradiance =  tf.broadcast_to(tf.expand_dims(irradiance, axis=1), [number_of_pixels, 3])
  # "apply the rendering equation
  radiance = brdf * irradiance  
  # saturate radiance at 1 for rendering purposes
  radiance = tf.math.minimum(radiance, 1.0)
  # gamma correction
  radiance = tf.math.pow(radiance, 1.0 / 2.2)
  # discretization in 0-255 pixel space
  radiance = tf.math.round(radiance * 255.0) / 255.0   
  return irradiance, radiance

# @tf.function(jit_compile=True)
def render( diffuse_colors, 
            surface_xyz,
            normals, 
            camera_xyz, 
            light_xyz, 
            light_color,
            background_color,
            image_shape,
            metallic,
            subsurface,
            specular, 
            roughness,
            specularTint,
            anisotropic,
            sheen,
            sheenTint,
            clearcoat,
            clearcoatGloss,
            is_not_background,
            pixel_indices_to_render,
            file_path=None):
  '''
  diffuse_colors :: diffuse (base) albedo colors, in the form of a 64 bit float tensor of dimensions (number_of_pixels, 3) 
  surface_xyz :: 3D (x,y,z) points, in the form of a 64 bit float tensor of dimensions (number_of_pixels, 3) 
  normals :: geometric orientations of the surface normal vector in 3D space for every point, in the form of a 64 bit float tensor of dimensions (number_of_pixels, 3) 
  camera_xyz :: 3D (x,y,z) point representing the camera focal point, in the form of a 64 bit float tensor of dimensions (3)
  light_xyz :: 3D (x,y,z) point representing the light source position, in the form of a 64 bit float tensor of dimensions (3)
  light_color :: RGB color value for the light projected into the scene, in the form of a 64 bit float tensor of dimensions (3)
  background_color :: RGB color value for the background (pixels where no information was available), in the form of a 64 bit float tensor of dimensions (3)
  image_shape :: width, height, and color channels for the image to be rendered, in the form of a .shape output from TensorFlow on the original image
  metallic :: BRDF parameter, in the form of a 64 bit float tensor of dimension (1), defined in 2012 by Disney: https://www.disneyanimation.com/publications/physically-based-shading-at-disney/
  subsurface :: (same)
  roughness :: (same)
  specularTint :: (same)
  anisotropic :: (same)
  sheen :: (same)
  sheenTint :: (same)
  clearcoat :: (same)
  clearcoatGloss :: (same)
  is_not_background :: a tensor of conditional values for whether a pixel is active or not, in the shape of [image_width, image_height, 3], where every pixel has its indices in the original saved here 
  pixel_indices_to_render :: a tensor of the indices in (pixel row, pixel column) of dimensions [number_of_pixels, 2]
  file_path :: optionally, user may specify a global file_path for an output .png of the render.
  '''
  
  # save image dimensions
  number_of_pixels, color_channels = diffuse_colors.shape
  
  # compute orthogonal vectors in surface tangent plane; any two orthogonal vectors on the plane will do; choose the first one arbitrarily
  random_normals = normalize(tf.random.uniform(shape=(number_of_pixels, 3), dtype=tf.float64))
  surface_tangents = normalize(tf.linalg.cross(normals, random_normals))
  surface_bitangents = normalize(tf.linalg.cross(normals, surface_tangents))

  # compute view orientiations as vectors in (x,y,z) space, between the camera and every surface point
  view_angles = normalize(camera_xyz - surface_xyz)

  # compute light orientiations as vectors in (x,y,z) space, between the camera and every surface point
  light_angles = normalize(light_xyz - surface_xyz)

  # compute the bidirectional reflectance distribution function (https://en.wikipedia.org/wiki/Bidirectional_reflectance_distribution_function)
  brdf = BRDF(  diffuse_colors=diffuse_colors,
                surface_xyz=surface_xyz,
                normals=normals, 
                surface_tangents=surface_tangents,
                surface_bitangents=surface_bitangents,
                light_angles=light_angles,
                view_angles=view_angles,
                metallic=metallic,
                subsurface=subsurface,
                specular=specular, 
                roughness=roughness,
                specularTint=specularTint,
                anisotropic=anisotropic,
                sheen=sheen,
                sheenTint=sheenTint,
                clearcoat=clearcoat,
                clearcoatGloss=clearcoatGloss)

  # compute irradiance (incident light on surface)
  irradiance, radiance = compute_radiance(surface_xyz, normals, light_angles, light_xyz, light_color, brdf)
  
  # save image if desired
  if file_path:
    print("Rendered {}".format(file_path))
    # save_image(image_data=radiance, background_color=background_color, image_shape=image_shape, is_not_background=is_not_background, pixel_indices_to_render=pixel_indices_to_render, file_path=file_path)

  return radiance


def load_data(filepath):
  # get file type
  file_type = filepath.split(".")[-1] 
  # handle different image file types that we have encoded data for
  if file_type == "png":
    image_data = cv2.imread(filepath)
  elif file_type == "exr":
    image_data = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  # switch color channels to RGB
  image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
  # normalize between 0-1.0
  if file_type == "png":
    image_data = image_data / 255.0
  # convert to tensor
  image_tensor = tf.constant(image_data, dtype=tf.float64)
  return image_tensor


def save_image(image_data, background_color, image_shape, is_not_background, pixel_indices_to_render, file_path):
  image_width, image_height, color_channels = image_shape

  background = tf.constant(value=[[[70,70,70]]], dtype=tf.uint8) # hacked
  background_image = tf.broadcast_to(input=background, shape=image_shape)

  # separate computations by color to isolate dimensions more easily (API should probably support this with less code, but doesn't appear to)
  red_pixel_values = image_data[:,0]
  green_pixel_values = image_data[:,1]
  blue_pixel_values = image_data[:,2]

  red_image_pixels = tf.scatter_nd( indices=pixel_indices_to_render,
                                    updates=red_pixel_values,
                                    shape=(image_shape[0], image_shape[1]))

  green_image_pixels = tf.scatter_nd( indices=pixel_indices_to_render,
                                      updates=green_pixel_values,
                                      shape=(image_shape[0], image_shape[1]))

  blue_image_pixels = tf.scatter_nd(  indices=pixel_indices_to_render,
                                      updates=blue_pixel_values,
                                      shape=(image_shape[0], image_shape[1]))

  image_pixels = tf.stack([red_image_pixels, green_image_pixels, blue_image_pixels], axis=2)
  discretized_image_pixels = tf.cast(image_pixels * 255, dtype=tf.uint8) 

  is_not_background_for_all_colors = tf.broadcast_to(tf.expand_dims(is_not_background, axis=2), [image_width, image_height, color_channels])
  final_image = tf.where(is_not_background_for_all_colors, discretized_image_pixels, background_image)

  encoded_png_image_data = tf.io.encode_png(final_image)
  tf.config.run_functions_eagerly(True)

  file_write_operation = tf.io.write_file(file_path, encoded_png_image_data)


def get_pixel_indices_to_render(diffuse_color, background_color):
  # get indices for non-background pixels (i.e. only pixels with geometry and texture information)
  red_background_color = background_color[0]
  green_background_color = background_color[1]
  blue_background_color = background_color[2]

  red_diffuse_color = diffuse_color[:,:,0]
  green_diffuse_color = diffuse_color[:,:,1]
  blue_diffuse_color = diffuse_color[:,:,2]

  red_not_same = red_background_color != red_diffuse_color
  green_not_same = green_background_color != green_diffuse_color
  blue_not_same = blue_background_color != blue_diffuse_color

  not_background = tf.math.logical_and(tf.math.logical_and(red_not_same, green_not_same), blue_not_same)

  return tf.where(not_background), not_background


def load_scene(folder):
  # load data on surface textures and geometry of object 
  diffuse = load_data(filepath="{}/diffuse.png".format(folder))
  normals = load_data(filepath="{}/normals.exr".format(folder))
  xyz = load_data(filepath="{}/xyz.exr".format(folder))

  # save image shape, which will be used when reformatting computations back into an image
  image_shape = tf.constant([diffuse.shape[0], diffuse.shape[1], diffuse.shape[2]], dtype=tf.int64) 

  # set background color to ignore
  light_color = tf.constant(1.0, tf.float64)
  background_color = tf.constant([70/255,70/255,70/255], tf.float64)

  # get a mask for selecting only pixels that are not background values (eventually, this could be saved in production as .png with alpha channel = 0.0)
  pixel_indices_to_render, is_not_background = get_pixel_indices_to_render(diffuse_color=diffuse, background_color=background_color)

  # convert data structures for textures and geometry from an image-based tensor of (width, height, colors) to a pixel-based tensor (total_active_pixels, colors)
  diffuse = tf.gather_nd(params=diffuse, indices=pixel_indices_to_render)
  normals = tf.gather_nd(params=normals, indices=pixel_indices_to_render)
  xyz = tf.gather_nd(params=xyz, indices=pixel_indices_to_render)

  # set camera and light (x,y,z) positions
  camera_xyz = tf.constant([1.7, 0.11, 0.7], dtype=tf.float64)
  light_xyz = tf.constant([1.7, 0.11, 0.7], dtype=tf.float64)

  # set brdf ground truth parameters as tensor constants
  metallic = tf.constant(0.0, dtype=tf.float64)
  subsurface = tf.constant(0.0, dtype=tf.float64)
  specular = tf.constant(0.5, dtype=tf.float64)
  roughness = tf.constant(0.3, dtype=tf.float64)
  specularTint = tf.constant(0.0, dtype=tf.float64)
  anisotropic = tf.constant(0.0, dtype=tf.float64)
  sheen = tf.constant(0.0, dtype=tf.float64)
  sheenTint = tf.constant(0.0, dtype=tf.float64)
  clearcoat = tf.constant(0.0, dtype=tf.float64)
  clearcoatGloss = tf.constant(0.0, dtype=tf.float64)

  start_time = time.time()
  number_of_renders = 100
  for i in range(number_of_renders):
    file_path = tf.constant("{}/ground_truth_{}.png".format(folder, i))
    scene = render( diffuse_colors=diffuse, 
                    surface_xyz=xyz,
                    normals=normals, 
                    camera_xyz=camera_xyz, 
                    light_xyz=light_xyz, 
                    light_color=light_color,
                    background_color=background_color,
                    image_shape=image_shape,
                    metallic=metallic,
                    subsurface=subsurface,
                    specular=specular, 
                    roughness=roughness,
                    specularTint=specularTint,
                    anisotropic=anisotropic,
                    sheen=sheen,
                    sheenTint=sheenTint,
                    clearcoat=clearcoat,
                    clearcoatGloss=clearcoatGloss,
                    pixel_indices_to_render=pixel_indices_to_render,
                    is_not_background=is_not_background,
                    file_path=file_path)


  end_time = time.time()
  print("Produced {} renders in {} seconds ({} seconds per render)".format(number_of_renders, end_time - start_time, (end_time - start_time) / number_of_renders))

if __name__ == "__main__":
  project_directory = "{}/inverse_renders/toucan".format(os.getcwd())
  load_scene(folder=project_directory)
  # inverse_render_optimization(scene)