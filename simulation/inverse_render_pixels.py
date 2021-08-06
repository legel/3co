from numpy.lib.function_base import diff
import tensorflow as tf
import numpy as np
import cv2
from math import pi
from pathlib import Path
import os, sys
import copy
import time
import os

# initialize global variables for optimization
use_gpu = True
iteration = 1
explorer = 1

if use_gpu:
  print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
else:
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"]=""


@tf.function(jit_compile=True)
def avg(x):
  return tf.math.reduce_mean(x)


@tf.function(jit_compile=True)
def sqr(x):
  return tf.math.square(x)


@tf.function(jit_compile=True)
def clamp(x, a, b):
  absolute_min = tf.cast(a, dtype=tf.float32) #tf.constant(a, dtype=tf.float32)
  absolute_max = tf.cast(b, dtype=tf.float32) #tf.constant(b, dtype=tf.float32)
  x = tf.math.minimum(x, absolute_max)
  x = tf.math.maximum(absolute_min, x)
  return x


@tf.function(jit_compile=True)
def normalize(x):
  norm = tf.linalg.norm(x, axis=1)
  ones_shape = tf.shape(x)[0]
  ones = tf.ones(ones_shape, dtype=tf.float32)
  norm = tf.where(norm == 0.0, ones, norm)
  norm = tf.broadcast_to(tf.expand_dims(norm, axis=1), [ones_shape, 3])
  result = x / norm
  result = tf.cast(result, dtype=tf.float32)
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
  number_of_pixels = tf.shape(NdotH)[0]
  greater_than_value = tf.fill(dims=[number_of_pixels, 3], value=tf.constant(1/pi, tf.float32))
  a2 = tf.cast(a*a, dtype=tf.float32)
  t = tf.cast(1 + (a2-1)*NdotH*NdotH, dtype=tf.float32)
  less_than_value = (a2-1) / (pi*tf.math.log(a2)*t)
  return tf.where(a >= 1, greater_than_value, less_than_value)


@tf.function(jit_compile=True)
def GTR2_aniso(NdotH, HdotX, HdotY, ax, ay):
  shape = tf.shape(NdotH)
  ones = tf.ones(shape, dtype=tf.float32)
  return ones / ( pi * ax * ay * sqr( sqr(HdotX/ax) + sqr(HdotY/ay) + sqr(NdotH)))


@tf.function(jit_compile=True)
def smithG_GGX(Ndotv, alphaG):
  a = tf.cast(sqr(alphaG), dtype=tf.float32)
  b = tf.cast(sqr(Ndotv), dtype=tf.float32)
  sqr_root = tf.math.sqrt(a + b - a * b)
  noemer = tf.math.add(Ndotv, sqr_root)
  teller = tf.constant(1, dtype=tf.float32)
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


@tf.function(jit_compile=True)
def compute_gradients(brdf_metadata, brdf_parameters, brdf_parameters_to_hold_constant_in_optimization):
  number_of_pixels = tf.shape(brdf_metadata)[0]

  NdotL = brdf_metadata[:,:,0]
  NdotV = brdf_metadata[:,:,1]
  NdotH = brdf_metadata[:,:,2]
  LdotH = brdf_metadata[:,:,3]
  Cdlin = brdf_metadata[:,:,4]
  Ctint = brdf_metadata[:,:,5]
  Csheen = brdf_metadata[:,:,6]
  FL = brdf_metadata[:,:,7]
  FV = brdf_metadata[:,:,8]
  Fd90 = brdf_metadata[:,:,9]
  Fd = brdf_metadata[:,:,10]
  ss = brdf_metadata[:,:,11]
  Ds = brdf_metadata[:,:,12]
  FH = brdf_metadata[:,:,13]
  Fs = brdf_metadata[:,:,14]
  Gs = brdf_metadata[:,:,15]
  Fsheen = brdf_metadata[:,:,16]
  Dr = brdf_metadata[:,:,17]
  Fr = brdf_metadata[:,:,18]
  Gr = brdf_metadata[:,:,19]
  aspect = brdf_metadata[:,:,20]
  ax = brdf_metadata[:,:,21]
  ay = brdf_metadata[:,:,22]
  aG = brdf_metadata[:,:,23]
  L = brdf_metadata[:,:,24]
  V = brdf_metadata[:,:,25]
  N = brdf_metadata[:,:,26]
  X = brdf_metadata[:,:,27]
  Y = brdf_metadata[:,:,28]
  diffuse_colors = brdf_metadata[:,:,29]

  broadcaster = lambda x: tf.broadcast_to(tf.expand_dims(x, axis=1), shape=(number_of_pixels,3))

  # unpack BRDF parameters
  metallic = broadcaster(brdf_parameters[:,0])
  subsurface = broadcaster(brdf_parameters[:,1])
  specular = broadcaster(brdf_parameters[:,2])
  roughness = broadcaster(brdf_parameters[:,3])
  specularTint = broadcaster(brdf_parameters[:,4])
  anisotropic = broadcaster(brdf_parameters[:,5])
  sheen = broadcaster(brdf_parameters[:,6])
  sheenTint = broadcaster(brdf_parameters[:,7])
  clearcoat = broadcaster(brdf_parameters[:,8])
  clearcoatGloss = broadcaster(brdf_parameters[:,9])

  # Note the following, with respect to potential to pre-cache as many inverse rendering optimization computations as possible:
  # - values which are unchanged with respect to everything but BRDF parameters are marked with a *
  # - values which change with respect to one or more BRDF parameters are marked with a f( ) and those parameters inside, e.g. f(parameter_1, parameter_2, ...)

  # halfway vector
  H = normalize(L+V) # *
  HdotX = broadcaster(tf.reduce_sum(tf.math.multiply(H, X), axis=1)) # *
  HdotY = broadcaster(tf.reduce_sum(tf.math.multiply(H, Y), axis=1)) # *
  
  # metallic gradient
  if "metallic" not in brdf_parameters_to_hold_constant_in_optimization:
    right_d_Fs_metallic = tf.expand_dims((1.0 - FH), axis=1) # *
    right_d_Fs_metallic = tf.broadcast_to(right_d_Fs_metallic, [number_of_pixels, 3]) # *
    d_Fs_metallic = Cdlin - 0.08 * specular * mix(tf.ones((number_of_pixels, 3), dtype=tf.float32), Ctint, specularTint) * right_d_Fs_metallic # f(specular, specularTint)
    NdotL3d = tf.broadcast_to(tf.expand_dims(NdotL, axis=1), [number_of_pixels, 3]) # *
    mix_Fd_ss_subsurface = tf.broadcast_to(tf.expand_dims(mix(Fd, ss, subsurface), axis=1), [number_of_pixels, 3]) # f(subsurface)
    Gs_Ds = tf.broadcast_to(tf.expand_dims(Gs * Ds, axis=1), [number_of_pixels, 3]) # *
    df_metallic = NdotL3d * ((-1.0 / pi) * mix_Fd_ss_subsurface * Cdlin + Fsheen + Gs_Ds * d_Fs_metallic) # f(specular, specularTint, subsurface)
  else:
    df_metallic = tf.constant(0.0, dtype=tf.float32)
    
  # subsurface gradient 
  if "subsurface" not in brdf_parameters_to_hold_constant_in_optimization:
    left = tf.broadcast_to(tf.expand_dims(NdotL * (1.0 / pi) * (1.0 - metallic) * (ss - Fd), axis=1), [number_of_pixels, 3]) # f(metallic)
    df_subsurface = left * Cdlin # f(metallic)
  else:
    df_subsurface = tf.constant(0.0, dtype=tf.float32)

  # specular gradient
  if "specular" not in brdf_parameters_to_hold_constant_in_optimization:
    left = tf.broadcast_to(tf.expand_dims(NdotL * Gs * Ds * (1.0 - FH) * (1.0 - metallic) * 0.08, axis=1), [number_of_pixels, 3]) # f(metallic)
    df_specular = left * mix(tf.ones(3, dtype=tf.float32), Ctint, specularTint) # f(metallic, specularTint)
  else:
    df_specular = tf.constant(0.0, dtype=tf.float32)

  # roughness gradient
  if "roughness" not in brdf_parameters_to_hold_constant_in_optimization:
    d_ss_roughness = 1.25 * LdotH * LdotH * (FV - 2.0 * FL * FV + FL + 2.0 * LdotH * LdotH * FL * FV * roughness) # f(roughness)
    d_Fd_roughness = 2.0 * LdotH ** 2 * (FV + FL + 2.0 * FL * FV * (Fd90 - 1.0)) # * 
    d_Gs_roughness = 0.5 * (roughness + 1.0) * (d_GGX_aG(NdotL, aG) * smithG_GGX(NdotV, aG) + d_GGX_aG(NdotV, aG) * smithG_GGX(NdotL, aG)) # f(roughness)
    roughness = tf.where(roughness <= 0, broadcaster(tf.constant([0.001], dtype=tf.float32)), roughness) # f(roughness)
    Ds_expand = Ds #tf.broadcast_to(tf.expand_dims(Ds, axis=1), [number_of_pixels, 3]) # *
    Gs_expand = Gs #tf.broadcast_to(tf.expand_dims(Gs, axis=1), [number_of_pixels, 3]) # *

    c = tf.convert_to_tensor(sqr(HdotX) / sqr(ax) + sqr(HdotY) / sqr(ay) + sqr(NdotH), dtype=tf.float32) # *
    d_Ds_roughness = 4.0 * ((2.0 *  (HdotX**2 * aspect**4 + HdotY ** 2) / (aspect**2 * roughness)) - c * roughness**3) / (pi * ax**2 * ay**2 * c**3) # f(roughness)
    left = NdotL * (1.0 - metallic) * (1.0 / pi) * mix(d_Fd_roughness, d_ss_roughness, subsurface) # f(metallic, roughness, subsurface)
    right = d_Gs_roughness * Ds + d_Ds_roughness * Gs # f(roughness)
    df_roughness = left * Cdlin + Fs * right #  f(metallic, roughness, subsurface)
  else:
    df_roughness = tf.constant([0.0], dtype=tf.float32)

  # specularTint gradient 
  if "specularTint" not in brdf_parameters_to_hold_constant_in_optimization:
    middle = tf.broadcast_to(tf.expand_dims((1.0 - FH) * specular * 0.08 * (1.0 - metallic), axis=1), [number_of_pixels, 3]) # f(specular, metallic)
    df_specularTint = NdotL3d * Gs_expand * Ds_expand * middle * (Ctint - 1.0) # f(specular, metallic)
  else:
    df_specularTint = tf.constant(0.0, dtype=tf.float32)

  # anisotropic gradient
  if "anisotropic" not in brdf_parameters_to_hold_constant_in_optimization:
    d_GTR2aniso_aspect = 4.0 * (sqr(HdotY) - sqr(HdotX) * tf.math.pow(aspect,4)) / (pi * sqr(ax) * sqr(ay) * tf.math.pow(c,3) * tf.math.pow(aspect,3)) # *
    d_Ds_anisotropic = (-0.45 / aspect) * (d_GTR2aniso_aspect) # *
    aniso_left = tf.broadcast_to(tf.expand_dims(NdotL * Gs * d_Ds_anisotropic, axis=1), [number_of_pixels, 3]) # *
    df_anisotropic = aniso_left * Fs # *
  else:
    df_anisotropic = tf.constant(0.0, dtype=tf.float32)

  # sheen gradient
  if "sheen" not in brdf_parameters_to_hold_constant_in_optimization:
    FH_expand = tf.broadcast_to(tf.expand_dims(FH, axis=1), [number_of_pixels, 3]) # *
    if "metallic" in brdf_parameters_to_hold_constant_in_optimization:
      NdotL3d = tf.broadcast_to(tf.expand_dims(NdotL, axis=1), [number_of_pixels, 3]) # *
    df_sheen = NdotL3d * (1.0 - metallic) * FH_expand * Csheen # f(metallic)
  else:
    df_sheen = tf.constant(0.0, dtype=tf.float32)

  # sheenTint gradient
  if "sheenTint" not in brdf_parameters_to_hold_constant_in_optimization:
    df_sheenTint = NdotL3d * (1.0 - metallic) * FH_expand * sheen * (Ctint - 1.0) # f(metallic, sheen)
  else:
    df_sheenTint = tf.constant(0.0, dtype=tf.float32)

  # clearcoat gradient
  if "clearcoat" not in brdf_parameters_to_hold_constant_in_optimization:    
    df_clearcoat = NdotL * 0.25 * Gr * Fr * Dr * 1.0 # *
    df_clearcoat = tf.broadcast_to(tf.expand_dims(df_clearcoat, axis=1), [number_of_pixels, 3]) # *
    df_clearcoat= tf.ones((number_of_pixels, 3), dtype=tf.float32) * df_clearcoat
  else:
    df_clearcoat = tf.constant(0.0, dtype=tf.float32)

  # clearcoatGloss gradient
  if "clearcoatGloss" not in brdf_parameters_to_hold_constant_in_optimization:    
    a = mix(0.1, 0.001, clearcoatGloss) # f(clearcoatGloss)
    t = 1.0 + (sqr(a) - 1.0) * sqr(NdotH) # *
    d_GTR1_a = 2.0 * a * ( tf.math.log(sqr(a)) * t - (sqr(a) - 1.0) * (t/(sqr(a)) + tf.math.log(sqr(a)) * sqr(NdotH))) / (pi * sqr((tf.math.log(sqr(a)) * t))) # f(clearcoatGloss)  
    df_clearcoatGloss = NdotL * 0.25 * clearcoat * -0.099 * Gr * Fr * d_GTR1_a # f(clearcoat, clearcoatGloss)
    df_clearcoatGloss = tf.broadcast_to(tf.expand_dims(df_clearcoatGloss, axis=1), [number_of_pixels, 3])
    df_clearcoatGloss = tf.ones((number_of_pixels, 3), dtype=tf.float32) * df_clearcoatGloss
  else:
    df_clearcoatGloss = tf.constant(0.0, dtype=tf.float32)
  

  # # diffuse colors (C_d) gradient 
  # #if "diffuseColors" not in brdf_parameters_to_hold_constant_in_optimization:

  # t = mix(Fd, ss, subsurface) / pi
  # xi = 0.3 * diffuse_colors[0] + 0.6 * diffuse_colors[1] + 0.1 * diffuse_colors[2]
  # dCtint_dCd = [
  #   (xi - diffuse_colors[0]) / (xi*xi),
  #   (xi - diffuse_colors[1]) / (xi*xi),
  #   (xi - diffuse_colors[2]) / (xi*xi)
  # ]
  
  # df_diffuse_colors = NdotL * (  (t + FH * sheen * sheenTint) * dCtint_dCd * (1.0 - metallic) + Gs * Ds * (1.0 - FH) * metallic )

  gradients = [df_metallic, df_subsurface, df_specular, df_roughness, df_specularTint, df_anisotropic, df_sheen, df_sheenTint, df_clearcoat, df_clearcoatGloss]
  #gradients = [tf.expand_dims(gradient, axis=2) for gradient in gradients]
  #brdf_gradients = tf.concat(gradients, axis=2)

  return gradients


@tf.function(jit_compile=True)
def photometric_error(ground_truth, hypothesis):
  return hypothesis - ground_truth


@tf.function(jit_compile=True)
def apply_gradients_from_inverse_rendering_loss(optimizer,
                                                ground_truth_radiance, 
                                                hypothesis_radiance,
                                                hypothesis_irradiance,
                                                hypothesis_brdf,
                                                hypothesis_brdf_parameters,
                                                hypothesis_brdf_metadata,
                                                ground_truth_brdf_parameters,
                                                brdf_parameters_to_hold_constant_in_optimization,
                                                report_results_on_this_iteration):

  number_of_pixels = tf.shape(ground_truth_radiance)[0]
  number_of_colors = tf.shape(ground_truth_radiance)[1]

  # gradients of the BRDF equation for each parameter, with respect to the loss function
  brdf_gradients = compute_gradients( brdf_metadata=hypothesis_brdf_metadata, 
                                      brdf_parameters=hypothesis_brdf_parameters,
                                      brdf_parameters_to_hold_constant_in_optimization=brdf_parameters_to_hold_constant_in_optimization)

  df_metallic, df_subsurface, df_specular, df_roughness, df_specularTint, df_anisotropic, df_sheen, df_sheenTint, df_clearcoat, df_clearcoatGloss = brdf_gradients

  # pixelwise difference across RGB channels                         
  photometric_loss = photometric_error(ground_truth_radiance, hypothesis_radiance)

  # gradient of gamma encoding
  df_gamma_encoding = tf.math.divide(tf.math.pow(hypothesis_brdf * hypothesis_irradiance, -1.2 / 2.2), 2.2) * hypothesis_irradiance

  # (hack) hard-code BRDF parameters not being fitted to 0.0
  delta_metallic = tf.constant(0.0, dtype=tf.float32) #tf.reduce_sum(df_metallic * df_gamma_encoding * photometric_loss) / (number_of_pixels * number_of_colors)
  delta_subsurface = tf.constant(0.0, dtype=tf.float32) #tf.reduce_sum(df_subsurface * df_gamma_encoding * photometric_loss) / (number_of_pixels * number_of_colors)
  delta_specular = tf.constant(0.0, dtype=tf.float32) #tf.reduce_sum(df_specular * df_gamma_encoding * photometric_loss) / (number_of_pixels * number_of_colors)
  delta_specularTint = tf.constant(0.0, dtype=tf.float32) #tf.reduce_sum(df_specularTint * df_gamma_encoding * photometric_loss) / (number_of_pixels * number_of_colors)
  delta_anisotropic = tf.constant(0.0, dtype=tf.float32) #tf.reduce_sum(df_anisotropic * df_gamma_encoding * photometric_loss) / (number_of_pixels * number_of_colors)
  delta_sheen = tf.constant(0.0, dtype=tf.float32) #tf.reduce_sum(df_sheen * df_gamma_encoding * photometric_loss) / (number_of_pixels * number_of_colors)
  delta_sheenTint = tf.constant(0.0, dtype=tf.float32) #tf.reduce_sum(df_sheenTint * df_gamma_encoding * photometric_loss) / (number_of_pixels * number_of_colors) 
  delta_clearcoat = tf.constant(0.0, dtype=tf.float32) #tf.reduce_sum(df_clearcoat * df_gamma_encoding * photometric_loss) / (number_of_pixels * number_of_colors)
  delta_clearcoatGloss = tf.constant(0.0, dtype=tf.float32) #tf.reduce_sum(df_clearcoatGloss * df_gamma_encoding * photometric_loss) / (number_of_pixels * number_of_colors)

  # sum the loss of the color channels, only, on a per-pixel basis
  delta_roughness = tf.reduce_sum(df_roughness * df_gamma_encoding * photometric_loss, axis = 1)

  # get the previous BRDF parameters
  metallic = tf.Variable(hypothesis_brdf_parameters[:,0])
  subsurface = tf.Variable(hypothesis_brdf_parameters[:,1])
  specular = tf.Variable(hypothesis_brdf_parameters[:,2])
  roughness = tf.Variable(hypothesis_brdf_parameters[:,3])
  specularTint = tf.Variable(hypothesis_brdf_parameters[:,4])
  anisotropic = tf.Variable(hypothesis_brdf_parameters[:,5])
  sheen = tf.Variable(hypothesis_brdf_parameters[:,6])
  sheenTint = tf.Variable(hypothesis_brdf_parameters[:,7])
  clearcoat = tf.Variable(hypothesis_brdf_parameters[:,8])
  clearcoatGloss = tf.Variable(hypothesis_brdf_parameters[:,9])

  # get the ground truth BRDF parameters for showing to human
  true_metallic = ground_truth_brdf_parameters[:,0]
  true_subsurface = ground_truth_brdf_parameters[:,1]
  true_specular = ground_truth_brdf_parameters[:,2]
  true_roughness = ground_truth_brdf_parameters[:,3]
  true_specularTint = ground_truth_brdf_parameters[:,4]
  true_anisotropic = ground_truth_brdf_parameters[:,5]
  true_sheen = ground_truth_brdf_parameters[:,6]
  true_sheenTint = ground_truth_brdf_parameters[:,7]
  true_clearcoat = ground_truth_brdf_parameters[:,8]
  true_clearcoatGloss = ground_truth_brdf_parameters[:,9]

  # set up hard limits
  lower_bound = tf.broadcast_to([tf.constant(0.0, dtype=tf.float32)], shape=metallic.shape)
  upper_bound = tf.broadcast_to([tf.constant(1.0, dtype=tf.float32)], shape=metallic.shape)

  # compute the new BRDF parameter with clamping in the update
  new_metallic = clamp(metallic - delta_metallic, lower_bound, upper_bound)
  new_subsurface = clamp(subsurface - delta_subsurface, lower_bound, upper_bound)
  new_specular = clamp(specular - delta_specular, lower_bound, upper_bound)
  new_roughness = clamp(roughness - delta_roughness, lower_bound, upper_bound)
  new_specularTint = clamp(specularTint - delta_specularTint, lower_bound, upper_bound)
  new_anisotropic = clamp(anisotropic - delta_anisotropic, lower_bound, upper_bound)
  new_sheen = clamp(sheen - delta_sheen, lower_bound, upper_bound)
  new_sheenTint = clamp(sheenTint - delta_sheenTint, lower_bound, upper_bound)
  new_clearcoat = clamp(clearcoat - delta_clearcoat, lower_bound, upper_bound)
  new_clearcoatGloss = clamp(clearcoatGloss - delta_clearcoatGloss, lower_bound, upper_bound)

  # compute clipped gradient for optimizer
  metallic_grad = metallic - new_metallic
  subsurface_grad = subsurface -  new_subsurface
  specular_grad = specular - new_specular 
  roughness_grad = roughness - new_roughness 
  specularTint_grad = specularTint - new_specularTint
  anisotropic_grad = anisotropic - new_anisotropic 
  sheen_grad = sheen - new_sheen
  sheenTint_grad = sheenTint - new_sheenTint
  clearcoat_grad = clearcoat - new_clearcoat
  clearcoatGloss_grad = clearcoatGloss - new_clearcoatGloss

  # compute average values for report
  if report_results_on_this_iteration:
    print(":::::::::::: BRDF Truth vs. Hypothesis (Δ Update) ::::::::::::")
    print("Metallic:        {:.5f} vs. {:.5f} (Δ {:+5f})".format(avg(true_metallic), avg(new_metallic), tf.constant(-1.0, dtype=tf.float32) * avg(delta_metallic)))
    print("Subsurface:      {:.5f} vs. {:.5f} (Δ {:+5f})".format(avg(true_subsurface), avg(new_subsurface), tf.constant(-1.0, dtype=tf.float32) * avg(delta_subsurface)))
    print("Specular:        {:.5f} vs. {:.5f} (Δ {:+5f})".format(avg(true_specular), avg(new_specular), tf.constant(-1.0, dtype=tf.float32) * avg(delta_specular)))
    print("Roughness:       {:.5f} vs. {:.5f} (Δ {:+5f})".format(avg(true_roughness), avg(new_roughness), tf.constant(-1.0, dtype=tf.float32) * avg(delta_roughness)))
    print("Specular Tint:   {:.5f} vs. {:.5f} (Δ {:+5f})".format(avg(true_specularTint), avg(new_specularTint), tf.constant(-1.0, dtype=tf.float32) * avg(delta_specularTint)))
    print("Anisotropic:     {:.5f} vs. {:.5f} (Δ {:+5f})".format(avg(true_anisotropic), avg(new_anisotropic), tf.constant(-1.0, dtype=tf.float32) * avg(delta_anisotropic)))
    print("Sheen:           {:.5f} vs. {:.5f} (Δ {:+5f})".format(avg(true_sheen), avg(new_sheen), tf.constant(-1.0, dtype=tf.float32) * avg(delta_sheen)))
    print("Sheen Tint:      {:.5f} vs. {:.5f} (Δ {:+5f})".format(avg(true_sheenTint), avg(new_sheenTint), tf.constant(-1.0, dtype=tf.float32) * avg(delta_sheenTint)))
    print("Clearcoat:       {:.5f} vs. {:.5f} (Δ {:+5f})".format(avg(true_clearcoat), avg(new_clearcoat), tf.constant(-1.0, dtype=tf.float32) * avg(delta_clearcoat)))
    print("Clearcoat Gloss: {:.5f} vs. {:.5f} (Δ {:+5f})".format(avg(true_clearcoatGloss), avg(new_clearcoatGloss), tf.constant(-1.0, dtype=tf.float32) * avg(delta_clearcoatGloss)))

  parameters_to_update = [metallic, subsurface, specular, roughness, specularTint, anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss]
  clipped_gradients = [metallic_grad, subsurface_grad, specular_grad, roughness_grad, specularTint_grad, anisotropic_grad, sheen_grad, sheenTint_grad, clearcoat_grad, clearcoatGloss_grad]
  gradients_as_tensor = tf.stack(clipped_gradients)
  total_photometric_loss = tf.reduce_sum(photometric_loss) / (tf.cast(number_of_pixels, dtype=tf.float32) * tf.cast(number_of_colors, dtype=tf.float32))

  # apply gradients with the power of a TensorFlow optimizer to tune learning rate automatically
  optimizer.apply_gradients(zip(clipped_gradients, parameters_to_update))

  # variables were updated by above operation, now we wrap it up and send it back for rendering...
  new_hypothesis_brdf_parameters = tf.concat([  tf.expand_dims(metallic, axis=1), 
                                                tf.expand_dims(subsurface, axis=1),  
                                                tf.expand_dims(specular, axis=1),  
                                                tf.expand_dims(roughness, axis=1),  
                                                tf.expand_dims(specularTint, axis=1),  
                                                tf.expand_dims(anisotropic, axis=1),  
                                                tf.expand_dims(sheen, axis=1),  
                                                tf.expand_dims(sheenTint, axis=1),  
                                                tf.expand_dims(clearcoat, axis=1),  
                                                tf.expand_dims(clearcoatGloss, axis=1)], axis=1)

  return new_hypothesis_brdf_parameters, gradients_as_tensor, total_photometric_loss, photometric_loss


def initialize_random_brdf_parameters(brdf_parameters_to_hold_constant_in_optimization, number_of_active_pixels):
  if "metallic" not in brdf_parameters_to_hold_constant_in_optimization:
    metallic = tf.random.uniform(shape=[number_of_active_pixels,1], dtype=tf.float32)
  else:
    metallic = tf.broadcast_to(tf.constant([[0.0]], dtype=tf.float32), shape=(number_of_active_pixels,1))

  if "subsurface" not in brdf_parameters_to_hold_constant_in_optimization:
    subsurface = tf.random.uniform(shape=[number_of_active_pixels,1], dtype=tf.float32)
  else:
    subsurface = tf.broadcast_to(tf.constant([[0.0]], dtype=tf.float32), shape=(number_of_active_pixels,1))

  if "specular" not in brdf_parameters_to_hold_constant_in_optimization:
    specular = tf.random.uniform(shape=[number_of_active_pixels,1], dtype=tf.float32)
  else:
    specular = tf.broadcast_to(tf.constant([[0.5]], dtype=tf.float32), shape=(number_of_active_pixels,1))

  if "roughness" not in brdf_parameters_to_hold_constant_in_optimization:
    roughness = tf.broadcast_to(tf.constant([[0.5]], dtype=tf.float32), shape=(number_of_active_pixels,1))
    #roughness = tf.random.uniform(shape=[number_of_active_pixels,1], dtype=tf.float32)
  else:
    roughness = tf.broadcast_to(tf.constant([[0.5]], dtype=tf.float32), shape=(number_of_active_pixels,1))

  if "specularTint" not in brdf_parameters_to_hold_constant_in_optimization:
    specularTint = tf.random.uniform(shape=[number_of_active_pixels,1], dtype=tf.float32)
  else:
    specularTint = tf.broadcast_to(tf.constant([[0.0]], dtype=tf.float32), shape=(number_of_active_pixels,1))

  if "anisotropic" not in brdf_parameters_to_hold_constant_in_optimization:
    anisotropic = tf.random.uniform(shape=[number_of_active_pixels,1], dtype=tf.float32)
  else:
    anisotropic = tf.broadcast_to(tf.constant([[0.0]], dtype=tf.float32), shape=(number_of_active_pixels,1))

  if "sheen" not in brdf_parameters_to_hold_constant_in_optimization:
    sheen = tf.random.uniform(shape=[number_of_active_pixels,1], dtype=tf.float32)
  else:
    sheen = tf.broadcast_to(tf.constant([[0.0]], dtype=tf.float32), shape=(number_of_active_pixels,1))

  if "sheenTint" not in brdf_parameters_to_hold_constant_in_optimization:
    sheenTint = tf.random.uniform(shape=[number_of_active_pixels,1], dtype=tf.float32)
  else:
    sheenTint = tf.broadcast_to(tf.constant([[0.0]], dtype=tf.float32), shape=(number_of_active_pixels,1))

  if "clearcoat" not in brdf_parameters_to_hold_constant_in_optimization:
    clearcoat = tf.random.uniform(shape=[number_of_active_pixels,1], dtype=tf.float32)
  else:
    clearcoat = tf.broadcast_to(tf.constant([[0.0]], dtype=tf.float32), shape=(number_of_active_pixels,1))

  if "clearcoatGloss" not in brdf_parameters_to_hold_constant_in_optimization:
    clearcoatGloss = tf.random.uniform(shape=[number_of_active_pixels,1], dtype=tf.float32)
  else:
    clearcoatGloss = tf.broadcast_to(tf.constant([[0.0]], dtype=tf.float32), shape=(number_of_active_pixels,1))

  brdf_parameters = tf.concat([metallic, subsurface, specular, roughness, specularTint, anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss], axis=1)
  return brdf_parameters


def compute_inverse_rendering_loss_and_gradients( hypothesis_brdf_parameters, 
                                                  optimizer,
                                                  diffuse_colors, 
                                                  surface_xyz,
                                                  normals, 
                                                  camera_xyz, 
                                                  light_xyz, 
                                                  light_color,
                                                  background_color,
                                                  image_shape,
                                                  pixel_indices_to_render,
                                                  is_not_background,
                                                  folder,
                                                  ground_truth_radiance,
                                                  brdf_parameters_to_hold_constant_in_optimization,
                                                  ground_truth_brdf_parameters,
                                                  frequency_of_human_output):
  global iteration
  global explorer

  report_results_on_this_iteration = iteration % frequency_of_human_output == 0
  render_file_path = tf.constant("{}/inverse_render_hypotheses/inverse_render_hypothesis_{}.png".format(folder, iteration))

  if report_results_on_this_iteration:
    print("\n\n                EXPLORER {}, ITERATION {}               ".format(explorer, iteration))
    save_file = tf.constant(True)
  else:
    save_file = tf.constant(False)      

  hypothesis_radiance, hypothesis_irradiance, hypothesis_brdf, hypothesis_brdf_metadata = render( diffuse_colors=diffuse_colors, 
                                                                                                  surface_xyz=surface_xyz,
                                                                                                  normals=normals, 
                                                                                                  camera_xyz=camera_xyz, 
                                                                                                  light_xyz=light_xyz, 
                                                                                                  light_color=light_color,
                                                                                                  background_color=background_color,
                                                                                                  image_shape=image_shape,
                                                                                                  brdf_parameters=hypothesis_brdf_parameters,
                                                                                                  pixel_indices_to_render=pixel_indices_to_render,
                                                                                                  is_not_background=is_not_background,
                                                                                                  file_path=render_file_path,
                                                                                                  save_file=save_file)

  hypothesis_brdf_parameters, gradients, inverse_rendering_loss, pixelwise_loss = apply_gradients_from_inverse_rendering_loss(optimizer=optimizer,
                                                                                                                              ground_truth_radiance=ground_truth_radiance, 
                                                                                                                              hypothesis_radiance=hypothesis_radiance,
                                                                                                                              hypothesis_irradiance=hypothesis_irradiance,
                                                                                                                              hypothesis_brdf=hypothesis_brdf,
                                                                                                                              hypothesis_brdf_parameters=hypothesis_brdf_parameters,
                                                                                                                              hypothesis_brdf_metadata=hypothesis_brdf_metadata,
                                                                                                                              brdf_parameters_to_hold_constant_in_optimization=brdf_parameters_to_hold_constant_in_optimization,
                                                                                                                              ground_truth_brdf_parameters=ground_truth_brdf_parameters, # for human eyes only
                                                                                                                              report_results_on_this_iteration=report_results_on_this_iteration
                                                                                                                              )

  if report_results_on_this_iteration:
    print("::::::::::::::::: AVERAGE PIXEL ERROR: {:.4f} :::::::::::::::::".format(tf.math.abs(inverse_rendering_loss*tf.constant(255, dtype=tf.float32))))
    new_file_path = tf.constant("{}/inverse_render_hypotheses/photometric_loss_{}.png".format(folder, iteration))
    save_image(image_data=pixelwise_loss*10, background_color=background_color, image_shape=image_shape, is_not_background=is_not_background, pixel_indices_to_render=pixel_indices_to_render, file_path=new_file_path)

  iteration += 1
  return inverse_rendering_loss, gradients, hypothesis_brdf_parameters


# @tf.function(jit_compile=True)
def inverse_render_optimization(folder, random_hypothesis_brdf_parameters=True, number_of_iterations = 270, frequency_of_human_output = 10, maximum_parallel_explorers = 10, pixel_error_termination_threshold = 0.01):
  # compute ground truth scene parameters (namely, the radiance values from the render, used in the photometric loss function)
  ground_truth_render_parameters, ground_truth_radiance, ground_truth_irradiance, ground_truth_brdf, ground_truth_brdf_metadata, brdf_parameters_to_hold_constant_in_optimization = load_scene(folder=project_directory)

  # unwrap render parameters
  diffuse_colors, surface_xyz, normals, camera_xyz, light_xyz, light_color, background_color, image_shape, ground_truth_brdf_parameters, pixel_indices_to_render, is_not_background = ground_truth_render_parameters

  # initialize output directories
  Path("{}/inverse_render_hypotheses".format(folder)).mkdir(parents=True, exist_ok=True)
  Path("{}/debugging_visualizations".format(folder)).mkdir(parents=True, exist_ok=True)

  final_inverse_rendering_losses = []
  final_hypothesis_brdf_parameters = []
  initial_hypothesis_brdf_parameters = []
  for parallel_explorer in range(1, maximum_parallel_explorers+1):
    global iteration
    global explorer

    explorer = parallel_explorer
    iteration = 1

    #learning_rate_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=0.35, decay_steps=number_of_iterations, end_learning_rate=0.0005, power=3.0, cycle=False)
    #optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_schedule, epsilon = 1e-9, beta_1 = 0.9, beta_2 = 0.999, amsgrad = True)
    learning_rate_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=0.6, decay_steps=360, end_learning_rate=0.01, power=3.0, cycle=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_schedule, epsilon = 1e-8, beta_1 = 0.99, beta_2 = 0.999, amsgrad = True)


    hypothesis_brdf_parameters = initialize_random_brdf_parameters(brdf_parameters_to_hold_constant_in_optimization=brdf_parameters_to_hold_constant_in_optimization,
                                                                   number_of_active_pixels=tf.shape(diffuse_colors)[0])

    initial_hypothesis_brdf_parameters.append(hypothesis_brdf_parameters)
    state_iterations = 0
    for i in range(number_of_iterations):
      inverse_rendering_loss, gradients, hypothesis_brdf_parameters = compute_inverse_rendering_loss_and_gradients( hypothesis_brdf_parameters=hypothesis_brdf_parameters, 
                                                                                                                    optimizer=optimizer,
                                                                                                                    diffuse_colors=diffuse_colors, 
                                                                                                                    surface_xyz=surface_xyz,
                                                                                                                    normals=normals, 
                                                                                                                    camera_xyz=camera_xyz, 
                                                                                                                    light_xyz=light_xyz, 
                                                                                                                    light_color=light_color,
                                                                                                                    background_color=background_color,
                                                                                                                    image_shape=image_shape,
                                                                                                                    pixel_indices_to_render=pixel_indices_to_render,
                                                                                                                    is_not_background=is_not_background,
                                                                                                                    folder=folder,
                                                                                                                    ground_truth_radiance=ground_truth_radiance,
                                                                                                                    brdf_parameters_to_hold_constant_in_optimization=brdf_parameters_to_hold_constant_in_optimization,
                                                                                                                    ground_truth_brdf_parameters=ground_truth_brdf_parameters,
                                                                                                                    frequency_of_human_output=frequency_of_human_output)


      # Below, memory issue tracked down to Keras state; resetting that solves the issue.
      # TODO: Automate Below
      if state_iterations % 90 == 0:
        print("Resetting the GPU state to prevent memory overload by Keras")
        tf.keras.backend.clear_session()
        learning_rate_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=0.3, decay_steps=90, end_learning_rate=0.1, power=3.0, cycle=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_schedule, epsilon = 1e-8, beta_1 = 0.9, beta_2 = 0.999, amsgrad = True)

      if state_iterations % 180 == 0:
        print("Resetting the GPU state to prevent memory overload by Keras")
        tf.keras.backend.clear_session()
        learning_rate_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=0.1, decay_steps=90, end_learning_rate=0.05, power=3.0, cycle=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_schedule, epsilon = 1e-8, beta_1 = 0.9, beta_2 = 0.999, amsgrad = True)

      if state_iterations % 270 == 0:
        print("Resetting the GPU state to prevent memory overload by Keras")
        tf.keras.backend.clear_session()
        learning_rate_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=0.05, decay_steps=90, end_learning_rate=0.01, power=3.0, cycle=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_schedule, epsilon = 1e-8, beta_1 = 0.9, beta_2 = 0.999, amsgrad = True)
        state_iterations = 0

      state_iterations += 1

    final_inverse_rendering_losses.append(inverse_rendering_loss*tf.constant(255, dtype=tf.float32))
    final_hypothesis_brdf_parameters.append(hypothesis_brdf_parameters)

  return final_inverse_rendering_losses, final_hypothesis_brdf_parameters, initial_hypothesis_brdf_parameters, ground_truth_brdf_parameters


@tf.function(jit_compile=True)
def BRDF( diffuse_colors,
          surface_xyz,
          normals, 
          surface_tangents,
          surface_bitangents,
          light_angles,
          view_angles,
          brdf_parameters):

  # Moving to mathematical notation used by Innmann et al. 2020: https://openaccess.thecvf.com/content_WACV_2020/papers/Innmann_BRDF-Reconstruction_in_Photogrammetry_Studio_Setups_WACV_2020_paper.pdf
  L = normalize(light_angles)
  V = normalize(view_angles)
  N = normalize(normals)
  X = normalize(surface_tangents)
  Y = normalize(surface_bitangents)

  number_of_pixels = tf.shape(diffuse_colors)[0]

  broadcaster = lambda x: tf.broadcast_to(tf.expand_dims(x, axis=1), shape=(number_of_pixels,3))

  # unpack BRDF parameters
  metallic = broadcaster(brdf_parameters[:,0])
  subsurface = broadcaster(brdf_parameters[:,1])
  specular = broadcaster(brdf_parameters[:,2])
  roughness = broadcaster(brdf_parameters[:,3])
  specularTint = broadcaster(brdf_parameters[:,4])
  anisotropic = broadcaster(brdf_parameters[:,5])
  sheen = broadcaster(brdf_parameters[:,6])
  sheenTint = broadcaster(brdf_parameters[:,7])
  clearcoat = broadcaster(brdf_parameters[:,8])
  clearcoatGloss = broadcaster(brdf_parameters[:,9])

  # compute dot products between surface normals and lighting, as well as surface normals and viewing angles
  NdotL = broadcaster(tf.reduce_sum(tf.math.multiply(N, L), axis=1))
  NdotV = broadcaster(tf.reduce_sum(tf.math.multiply(N, V), axis=1))                 

  # integrate viewing and lighting angle with respect to normals 
  H = normalize(tf.math.add(L, V))
  NdotH = broadcaster(tf.reduce_sum(tf.math.multiply(N, H), axis=1))
  LdotH = broadcaster(tf.reduce_sum(tf.math.multiply(L, H), axis=1))

  # aproximate luminance
  Cdlin = mon2lin(diffuse_colors)
  Cdlum =  0.3*Cdlin[:,0] + 0.6*Cdlin[:,1]  + 0.1*Cdlin[:,2]
  Cdlum_exp = broadcaster(Cdlum)
  Ctint = tf.where(Cdlum_exp > 0, Cdlin/Cdlum_exp, tf.ones((number_of_pixels, 3), dtype=tf.float32))

  Cspec0 = mix(specular * .08 * mix(tf.ones((number_of_pixels, 3), dtype=tf.float32), Ctint, specularTint), Cdlin, metallic)
  Csheen = mix(tf.ones((number_of_pixels, 3), dtype=tf.float32), Ctint, sheenTint) 

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
  anisotropic = tf.math.minimum(tf.constant(0.99, dtype=tf.float32), anisotropic) # added this to prevent division by zero
  aspect = tf.math.sqrt(1-anisotropic*.9)
  ax = tf.math.maximum(tf.constant(.001, dtype=tf.float32), sqr(roughness)/aspect)
  ay = tf.math.maximum(tf.constant(.001, dtype=tf.float32), sqr(roughness)*aspect)

  HdotX = broadcaster(tf.reduce_sum(H * X, axis=1))
  HdotY = broadcaster(tf.reduce_sum(H * Y, axis=1))
  Ds = GTR2_aniso(NdotH, HdotX, HdotY, ax, ay)
  Ds_exp = Ds
  FH = SchlickFresnel(LdotH)
  FH_exp = FH
  Fs = mix(Cspec0, tf.ones((number_of_pixels, 3), dtype=tf.float32), FH_exp)

  aG = tf.cast(sqr((0.5 * (roughness + 1))), dtype=tf.float32)
  Gs = smithG_GGX(NdotL, aG) * smithG_GGX(NdotV, aG)
  Gs_exp = Gs

  # Sheen
  Fsheen = FH_exp * sheen * Csheen

  # Clearcoat (IOR = 1.5 -> F0 = 0.04)
  Dr = GTR1(NdotH, mix(.1,.001,clearcoatGloss))
  Fr = mix(.04, 1, FH)
  Gr = smithG_GGX(NdotL, .25) * smithG_GGX(NdotV, .25)

  mix_fd_ss_subs = mix(Fd, ss, subsurface)
  # mix_fd_ss_subs = broadcaster(mix_fd_ss_subs)
  Cdlin_mix_fd = Cdlin * mix_fd_ss_subs

  clearcoat_gr_fr_dr = .25 * clearcoat*Gr*Fr*Dr 
  # clearcoat_gr_fr_dr = tf.broadcast_to(tf.expand_dims(clearcoat_gr_fr_dr,axis=1),[number_of_pixels, 3])

  brdf = ((1/pi) * Cdlin_mix_fd + Fsheen) * (1-metallic) + clearcoat_gr_fr_dr + Gs_exp*Fs*Ds_exp

  # saturated minimum BRDF to 0.0
  brdf = tf.math.maximum(brdf, 0.001)

  # pack up per-pixel BRDF representations for future computational access
  brdf_metadata = tf.concat([ tf.expand_dims(NdotL, axis=2), 
                              tf.expand_dims(NdotV, axis=2),
                              tf.expand_dims(NdotH, axis=2),
                              tf.expand_dims(LdotH, axis=2),
                              tf.expand_dims(Cdlin, axis=2),
                              tf.expand_dims(Ctint, axis=2),
                              tf.expand_dims(Csheen, axis=2),
                              tf.expand_dims(FL, axis=2),
                              tf.expand_dims(FV, axis=2),
                              tf.expand_dims(Fd90, axis=2),
                              tf.expand_dims(Fd, axis=2),
                              tf.expand_dims(ss, axis=2),
                              tf.expand_dims(Ds, axis=2),
                              tf.expand_dims(FH, axis=2),
                              tf.expand_dims(Fs, axis=2),
                              tf.expand_dims(Gs, axis=2),
                              tf.expand_dims(Fsheen, axis=2),
                              tf.expand_dims(Dr, axis=2),
                              tf.expand_dims(Fr, axis=2),
                              tf.expand_dims(Gr, axis=2),
                              tf.expand_dims(aspect, axis=2),
                              tf.expand_dims(ax, axis=2),
                              tf.expand_dims(ay, axis=2),
                              tf.expand_dims(aG, axis=2),
                              tf.expand_dims(L, axis=2),
                              tf.expand_dims(V, axis=2),
                              tf.expand_dims(N, axis=2),
                              tf.expand_dims(X, axis=2),
                              tf.expand_dims(Y, axis=2),
                              tf.expand_dims(diffuse_colors, axis=2)
                             ], axis=2)

  return brdf, brdf_metadata


@tf.function(jit_compile=True)
def filmic( RGB,
            shoulder_strength=0.22,
            linear_strength=0.3,
            linear_angle=0.1,
            toe_strength=0.2,
            toe_numerator=0.01,
            toe_denominator=0.3,
            exposure_bias=2,
            linear_whitepoint=11.2):

    A = shoulder_strength
    B = linear_strength
    C = linear_angle
    D = toe_strength
    E = toe_numerator
    F = toe_denominator

    X = RGB * exposure_bias
    RGB = ((X * (A * X + C * B) + D * E) / (X * (A * X + B) + D * F)) - E / F

    X = linear_whitepoint
    DENOM = ((X * (A * X + C * B) + D * E) / (X * (A * X + B) + D * F)) - E / F

    RGB = RGB * (1 / DENOM)

    return RGB


@tf.function(jit_compile=True)
def compute_radiance(surface_xyz, normals, light_angles, light_xyz, light_color, brdf):
  number_of_pixels = tf.shape(surface_xyz)[0]
  # approximate light intensity requirements
  light_distance_metric = light_xyz - surface_xyz

  # get the distance (square root of sum of squares across (x,y,z) dimensions) between the light source and every point
  light_intensity_scale = tf.reduce_max(tf.reduce_sum(light_distance_metric * light_distance_metric, axis=1))
  light_intensity = light_color * light_intensity_scale

  # compute angles between surface normal geometry and lighting incidence
  cosine_term = tf.reduce_sum(tf.math.multiply(normals, light_angles), axis=1)

  cosine_term = tf.math.maximum(tf.cast(0.5, dtype=tf.float32), tf.cast(cosine_term, dtype=tf.float32))  # hack!

  # compute orientations from lighting to surface positions
  vector_light_to_surface = light_xyz - surface_xyz
  light_to_surface_distance_squared = tf.reduce_sum(tf.math.multiply(vector_light_to_surface, vector_light_to_surface), axis=1)
  light_intensity = tf.fill(dims=[number_of_pixels], value=tf.cast(light_intensity, dtype=tf.float32))

  irradiance = light_intensity / light_to_surface_distance_squared * cosine_term
  # compute irradiance for all points on surface
  irradiance =  tf.broadcast_to(tf.expand_dims(irradiance, axis=1), [number_of_pixels, 3])

  # apply the rendering equation
  radiance = brdf * irradiance  

  # HDR tonemapping technique, which Blender uses under the hood
  # radiance = filmic(radiance)

  # saturate radiance at 1 for rendering purposes
  radiance = tf.math.minimum(radiance, 1.0)

  # saturated minimum radiance to 0.001
  radiance = tf.math.maximum(radiance, 0.001)

  # gamma correction
  radiance = tf.math.pow(radiance, 1.0 / 2.2)

  # discretization in 0-255 pixel space
  radiance = tf.math.round(radiance * 255.0) / 255.0

  return irradiance, radiance


def render( diffuse_colors, 
            surface_xyz,
            normals, 
            camera_xyz, 
            light_xyz, 
            light_color,
            background_color,
            image_shape,
            brdf_parameters,
            is_not_background,
            pixel_indices_to_render,
            file_path,
            save_file):
  '''
  diffuse_colors :: diffuse (base) albedo colors, in the form of a 64 bit float tensor of dimensions (number_of_pixels, 3) 
  surface_xyz :: 3D (x,y,z) points, in the form of a 64 bit float tensor of dimensions (number_of_pixels, 3) 
  normals :: geometric orientations of the surface normal vector in 3D space for every point, in the form of a 64 bit float tensor of dimensions (number_of_pixels, 3) 
  camera_xyz :: 3D (x,y,z) point representing the camera focal point, in the form of a 64 bit float tensor of dimensions (3)
  light_xyz :: 3D (x,y,z) point representing the light source position, in the form of a 64 bit float tensor of dimensions (3)
  light_color :: RGB color value for the light projected into the scene, in the form of a 64 bit float tensor of dimensions (3)
  background_color :: RGBA color value for the background (pixels where no information was available), in the form of a 64 bit float tensor of dimensions (4)
  image_shape :: width, height, and color channels for the image to be rendered, in the form of a .shape output from TensorFlow on the original image
  brdf_parameters :: 10 BRDF parameters, in the form of a 64 bit float tensor of dimension (number_of_pixels, 10), defined in 2012 by Disney: https://www.disneyanimation.com/publications/physically-based-shading-at-disney/
  is_not_background :: a tensor of conditional values for whether a pixel is active or not, in the shape of [image_width, image_height, 3], where every pixel has its indices in the original saved here 
  pixel_indices_to_render :: a tensor of the indices in (pixel row, pixel column) of dimensions [number_of_pixels, 2]
  file_path :: optionally, user may specify a global file_path for an output .png of the render.
  '''

  # save image dimensions
  number_of_pixels = tf.shape(diffuse_colors)[0]
  color_channels = tf.shape(diffuse_colors)[1]

  # compute orthogonal vectors in surface tangent plane; any two orthogonal vectors on the plane will do; choose the first one arbitrarily
  random_normals = normalize(tf.random.uniform(shape=(number_of_pixels, 3), dtype=tf.float32))
  surface_tangents = normalize(tf.linalg.cross(normals, random_normals))
  surface_bitangents = normalize(tf.linalg.cross(normals, surface_tangents))

  # compute view orientiations as vectors in (x,y,z) space, between the camera and every surface point
  view_angles = normalize(camera_xyz - surface_xyz)

  # compute light orientiations as vectors in (x,y,z) space, between the camera and every surface point
  light_angles = normalize(light_xyz - surface_xyz)

  # compute the bidirectional reflectance distribution function (https://en.wikipedia.org/wiki/Bidirectional_reflectance_distribution_function)
  brdf, brdf_metadata = BRDF(   diffuse_colors=diffuse_colors,
                                surface_xyz=surface_xyz,
                                normals=normals, 
                                surface_tangents=surface_tangents,
                                surface_bitangents=surface_bitangents,
                                light_angles=light_angles,
                                view_angles=view_angles,
                                brdf_parameters=brdf_parameters)

  # compute irradiance (incident light on surface)
  irradiance, radiance = compute_radiance(surface_xyz, normals, light_angles, light_xyz, light_color, brdf)

  # save image if desired
  if save_file:
    # save render
    save_image(image_data=radiance, background_color=background_color, image_shape=image_shape, is_not_background=is_not_background, pixel_indices_to_render=pixel_indices_to_render, file_path=file_path)

    # save roughness map
    roughness = tf.broadcast_to(tf.expand_dims(brdf_parameters[:,3], axis=1), shape=tf.shape(diffuse_colors))
    new_file_path = tf.strings.regex_replace(input=file_path, pattern=".png", rewrite="_roughness.png")
    save_image(image_data=roughness, background_color=background_color, image_shape=image_shape, is_not_background=is_not_background, pixel_indices_to_render=pixel_indices_to_render, file_path=new_file_path)

  return radiance, irradiance, brdf, brdf_metadata


def load_data(filepath):
  # get file type
  file_type = filepath.split(".")[-1] 
  # handle different image file types that we have encoded data for
  if file_type == "png":
    image_data = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
  elif file_type == "exr":
    image_data = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  # switch color channels to RGB
  # image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
  # normalize between 0-1.0
  if file_type == "png":
    image_data = image_data / 255.0
  # convert to tensor
  image_tensor = tf.zeros(shape=image_data.shape)
  channel_0 = tf.expand_dims(tf.constant(image_data[:,:,2], dtype=tf.float32), axis=2)
  channel_1 = tf.expand_dims(tf.constant(image_data[:,:,1], dtype=tf.float32), axis=2)
  channel_2 = tf.expand_dims(tf.constant(image_data[:,:,0], dtype=tf.float32), axis=2)
  if image_data.shape[2] == 4:
    channel_3 = tf.expand_dims(tf.constant(image_data[:,:,3], dtype=tf.float32), axis=2)
    image_tensor = tf.concat([channel_0, channel_1, channel_2, channel_3], axis=2)
  else:
    image_tensor = tf.concat([channel_0, channel_1, channel_2], axis=2)

  # print(image_tensor.shape)
  # print(image_tensor)
  return image_tensor


def save_image(image_data, background_color, image_shape, is_not_background, pixel_indices_to_render, file_path):
  color_channels = 4 # alpha
  image_width = image_shape[0]
  image_height = image_shape[1]

  background = tf.constant(value=[[[0,0,0,0]]], dtype=tf.uint8)
  background_image = tf.broadcast_to(input=background, shape=(image_width, image_height, color_channels)) # use alpha channel for background

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

  alpha_foreground = tf.constant(value=1.0, dtype=tf.float32)
  alpha_image_pixels = tf.broadcast_to(input=alpha_foreground, shape=(image_width, image_height)) # use alpha channel for background

  image_pixels = tf.stack([red_image_pixels, green_image_pixels, blue_image_pixels, alpha_image_pixels], axis=2)
  discretized_image_pixels = tf.cast(image_pixels * 255, dtype=tf.uint8) 

  is_not_background_for_all_colors = tf.broadcast_to(tf.expand_dims(is_not_background, axis=2), [image_width, image_height, color_channels])
  final_image = tf.where(is_not_background_for_all_colors, discretized_image_pixels, background_image)

  encoded_png_image_data = tf.io.encode_png(final_image)
  tf.config.run_functions_eagerly(True)

  file_write_operation = tf.io.write_file(file_path, encoded_png_image_data)


def get_pixel_indices_to_render(diffuse_color, background_color):
  number_of_color_channels = diffuse_color.shape[2]
  if number_of_color_channels == 3:
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

  elif number_of_color_channels == 4:
    diffuse_alpha_channel = diffuse_color[:,:,3]
    background_alpha_channel = background_color[3]

    alpha_not_same = background_alpha_channel != diffuse_alpha_channel

    not_background = alpha_not_same

  return tf.where(not_background), not_background


# @tf.function(jit_compile=False)
def load_scene( folder, 
                random_ground_truth_brdf_parameters = False,
                light_color = tf.constant(1.0, tf.float32),
                background_color = tf.constant([0/255,0/255,0/255,0/255], tf.float32),
                camera_xyz = tf.constant([1.7, 0.11, 1.5], dtype=tf.float32),
                light_xyz = tf.constant([1.7, 0.11, 1.5], dtype=tf.float32)):

  # load data on surface textures and geometry of object 
  diffuse = load_data(filepath="{}/diffuse.png".format(folder))
  normals = load_data(filepath="{}/normals.exr".format(folder))
  xyz = load_data(filepath="{}/xyz.exr".format(folder))
  roughness = load_data(filepath="{}/roughness.png".format(folder))

  # get a mask for selecting only pixels that are not background values (eventually, this could be saved in production as .png with alpha channel = 0.0)
  pixel_indices_to_render, is_not_background = get_pixel_indices_to_render(diffuse_color=diffuse, background_color=background_color)

  # clip alpha channels
  diffuse = diffuse[:,:,0:3]

  # get single value from greyscale
  roughness = roughness[:,:,0]

  # save image shape, which will be used when reformatting computations back into an image
  image_shape = tf.constant([diffuse.shape[0], diffuse.shape[1], diffuse.shape[2]], dtype=tf.int64) 

  # convert data structures for textures and geometry from an image-based tensor of (width, height, colors) to a pixel-based tensor (total_active_pixels, colors)
  diffuse = tf.gather_nd(params=diffuse, indices=pixel_indices_to_render)
  normals = tf.gather_nd(params=normals, indices=pixel_indices_to_render)
  xyz = tf.gather_nd(params=xyz, indices=pixel_indices_to_render)
  roughness = tf.gather_nd(params=roughness, indices=pixel_indices_to_render)

  # experimentally, and algebraically, the following parameters are found to destabilize the optimization
  brdf_parameters_to_hold_constant_in_optimization = ["metallic", "subsurface", "specular", "specularTint", "anisotropic", "sheen", "sheenTint", "clearcoat", "clearcoatGloss"]

  number_of_active_pixels = tf.shape(diffuse)[0]

  metallic = tf.broadcast_to(tf.cast([[0.0]], dtype=tf.float32), shape=(number_of_active_pixels,1))
  subsurface = tf.broadcast_to(tf.constant([[0.0]], dtype=tf.float32), shape=(number_of_active_pixels,1))
  specular = tf.broadcast_to(tf.constant([[0.5]], dtype=tf.float32), shape=(number_of_active_pixels,1))
  roughness = tf.expand_dims(roughness, axis=1)
  #roughness = tf.broadcast_to(tf.constant([[0.0]], dtype=tf.float32), shape=(number_of_active_pixels,1))
  specularTint = tf.broadcast_to(tf.constant([[0.0]], dtype=tf.float32), shape=(number_of_active_pixels,1))
  anisotropic = tf.broadcast_to(tf.constant([[0.0]], dtype=tf.float32), shape=(number_of_active_pixels,1))
  sheen = tf.broadcast_to(tf.constant([[0.0]], dtype=tf.float32), shape=(number_of_active_pixels,1))
  sheenTint = tf.broadcast_to(tf.constant([[0.0]], dtype=tf.float32), shape=(number_of_active_pixels,1))
  clearcoat = tf.broadcast_to(tf.constant([[0.0]], dtype=tf.float32), shape=(number_of_active_pixels,1))
  clearcoatGloss = tf.broadcast_to(tf.constant([[0.0]], dtype=tf.float32), shape=(number_of_active_pixels,1))


  ground_truth_brdf_parameters = tf.concat([metallic, subsurface, specular, roughness, specularTint, anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss], axis=1)

  file_path = tf.constant("{}/ground_truth.png".format(folder))
  radiance, irradiance, brdf, brdf_metadata = render( diffuse_colors=diffuse, 
                                                      surface_xyz=xyz,
                                                      normals=normals, 
                                                      camera_xyz=camera_xyz, 
                                                      light_xyz=light_xyz, 
                                                      light_color=light_color,
                                                      background_color=background_color,
                                                      image_shape=image_shape,
                                                      brdf_parameters=ground_truth_brdf_parameters,
                                                      pixel_indices_to_render=pixel_indices_to_render,
                                                      is_not_background=is_not_background,
                                                      file_path=file_path,
                                                      save_file=tf.constant(True))

  # wrap up scene parameters
  render_parameters = [diffuse, xyz, normals, camera_xyz, light_xyz, light_color, background_color, image_shape, ground_truth_brdf_parameters, pixel_indices_to_render, is_not_background]
  scene = [render_parameters, radiance, irradiance, brdf, brdf_metadata, brdf_parameters_to_hold_constant_in_optimization]

  return scene


if __name__ == "__main__":
  project_directory = "{}/inverse_renders/pillow".format(os.getcwd())

  # project_directory = "{}/inverse_renders/toucan".format(os.getcwd())
  #project_directory = "{}/inverse_renders/flamingo".format(os.getcwd())

  total_experiments = 1

  minimum_pixel_errors = np.zeros(shape=total_experiments, dtype=np.float32)
  number_of_trials_per_experiment = np.zeros(shape=total_experiments, dtype=np.float32)

  for experiment in range(total_experiments):
    final_inverse_rendering_losses, final_hypothesis_brdf_parameters, initial_hypothesis_brdf_parameters, true_brdf_parameters = inverse_render_optimization(folder=project_directory)
    number_of_trials_per_experiment[experiment] = len(final_inverse_rendering_losses)
    minimum_pixel_errors[experiment] = min(final_inverse_rendering_losses)

    print("\n\n***************************************** EXPERIMENT {} ******************************************".format(experiment+1))
    print("                  PER_PIXEL_ERROR      METALLIC          SUBSURFACE        SPECULAR          ROUGHNESS ")
    print("GROUND TRUTH:    {:6.3f}                {:.3f}             {:.3f}             {:.3f}             {:.3f}    ".format(0.0, avg(true_brdf_parameters[0]), avg(true_brdf_parameters[1]), avg(true_brdf_parameters[2]), avg(true_brdf_parameters[3])))

    for optimization_number, (loss, final_brdf, initial_brdf) in enumerate(zip(final_inverse_rendering_losses, final_hypothesis_brdf_parameters, initial_hypothesis_brdf_parameters)):
      print("OPTIMIZATION {}:  {:6.3f}                {:.3f} to {:.3f}    {:.3f} to {:.3f}    {:.3f} to {:.3f}    {:.3f} to {:.3f}".format(optimization_number, tf.math.abs(loss), avg(initial_brdf[0]), avg(final_brdf[0]), avg(initial_brdf[1]), avg(final_brdf[1]), avg(initial_brdf[2]), avg(final_brdf[2]), avg(initial_brdf[3]), avg(final_brdf[3]) ))

  print("\n\nFINAL EXPERIMENT RESULTS:")
  print("    Average # of Trials Per Experiment:      {:.3f}".format(np.average(number_of_trials_per_experiment)))
  print("    Average Pixel Error for Best Experiment: {:.3f}".format(np.average(np.abs(minimum_pixel_errors))))
