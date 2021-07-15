import tensorflow as tf
from tensorflow_graphics.math.optimizer import levenberg_marquardt
import tensorflow_probability as tfp
import numpy as np
import cv2
from math import pi
from pathlib import Path
import os, sys
import copy
import time
import os

dont_use_gpu = True

if dont_use_gpu:
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"]=""

iteration = 0

@tf.function(experimental_compile=True)
def sqr(x):
  return tf.math.square(x)


@tf.function(experimental_compile=True)
def clamp(x, a, b):
  absolute_min = tf.constant(a, dtype=tf.float64)
  absolute_max = tf.constant(b, dtype=tf.float64)
  x = tf.math.minimum(x, absolute_max)
  x = tf.math.maximum(absolute_min, x)
  return x


@tf.function(experimental_compile=True)
def normalize(x):
  norm = tf.linalg.norm(x, axis=1)
  ones = tf.ones(x.shape[0], dtype=tf.float64)
  norm = tf.where(norm == 0.0, ones, norm)
  norm = tf.broadcast_to(tf.expand_dims(norm, axis=1), [x.shape[0], 3])
  result = x / norm
  result = tf.cast(result, dtype=tf.float64)
  return result


@tf.function(experimental_compile=True)
def mix(x, y, a):
  return x * (1 - a) + y * a


@tf.function(experimental_compile=True)
def SchlickFresnel(u):
  m = clamp(1-u, 0, 1)
  return tf.pow(m, 5)


@tf.function(experimental_compile=True)
def GTR1(NdotH, a):
  number_of_pixels = NdotH.shape[0]
  if (a >= 1): 
    # value = tf.cast(1/pi, dtype=tf.float64)
    return tf.fill(dims=[number_of_pixels], value=tf.constant(1/pi, tf.float64))

  a2 = tf.cast(a*a, dtype=tf.float64)
  t = tf.cast(1 + (a2-1)*NdotH*NdotH, dtype=tf.float64)
  return (a2-1) / (pi*tf.math.log(a2)*t)


@tf.function(experimental_compile=True)
def GTR2_aniso(NdotH, HdotX, HdotY, ax, ay):
  shape = tf.shape(NdotH)
  ones = tf.ones(shape, dtype=tf.float64)
  return ones / ( pi * ax * ay * sqr( sqr(HdotX/ax) + sqr(HdotY/ay) + sqr(NdotH)))


@tf.function(experimental_compile=True)
def smithG_GGX(Ndotv, alphaG):
  a = tf.cast(sqr(alphaG), dtype=tf.float64)
  b = tf.cast(sqr(Ndotv), dtype=tf.float64)
  sqr_root = tf.math.sqrt(a + b - a * b)
  noemer = tf.math.add(Ndotv, sqr_root)
  teller = tf.constant(1, dtype=tf.float64)
  return teller / noemer


@tf.function(experimental_compile=True)
def d_GGX_aG(NdotA, aG):
  k = tf.math.sqrt( sqr(aG) + sqr(NdotA) - sqr(aG) * sqr(NdotA) )
  return aG * (sqr(NdotA) - 1.0) / (k * sqr((NdotA + k)))


@tf.function(experimental_compile=True)
def smithG_GGX_aniso(NdotV, VdotX, VdotY, ax, ay):
  return 1 / (NdotV + tf.math.sqrt( sqr(VdotX*ax) + sqr(VdotY*ay) + sqr(NdotV) ))


@tf.function(experimental_compile=True)
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


@tf.function(experimental_compile=True)
def compute_gradients(brdf_metadata, brdf_parameters, brdf_parameters_to_hold_constant_in_optimization):
  number_of_pixels = brdf_metadata.shape[0]

  # unwrap previously computed BRDF representations
  NdotL = brdf_metadata[:,0]
  NdotV = brdf_metadata[:,1]
  NdotH = brdf_metadata[:,2]
  LdotH = brdf_metadata[:,3]
  Cdlin = brdf_metadata[:,4:7]
  Ctint = brdf_metadata[:,7:10]
  Csheen = brdf_metadata[:,10:13]
  FL = brdf_metadata[:,13]
  FV = brdf_metadata[:,14]
  Fd90 = brdf_metadata[:,15]
  Fd = brdf_metadata[:,16]
  ss = brdf_metadata[:,17]
  Ds = brdf_metadata[:,18]
  FH = brdf_metadata[:,19]
  Fs = brdf_metadata[:,20:23]
  Gs = brdf_metadata[:,23]
  Fsheen = brdf_metadata[:,24:27]
  Dr = brdf_metadata[:,27]
  Fr = brdf_metadata[:,28]
  Gr = brdf_metadata[:,29]
  aspect = brdf_metadata[:,30]
  ax = brdf_metadata[:,31]
  ay = brdf_metadata[:,32]
  aG = brdf_metadata[:,33]
  L = brdf_metadata[:,34:37]
  V = brdf_metadata[:,37:40] 
  N = brdf_metadata[:,40:43]
  X = brdf_metadata[:,43:46]
  Y = brdf_metadata[:,46:49]

  # unwrap current hypothesis BRDF parameters
  metallic = brdf_parameters[0]
  subsurface = brdf_parameters[1] 
  specular = brdf_parameters[2]
  roughness = brdf_parameters[3]
  specularTint = brdf_parameters[4]
  anisotropic = brdf_parameters[5]
  sheen = brdf_parameters[6]
  sheenTint = brdf_parameters[7]
  clearcoat = brdf_parameters[8]
  clearcoatGloss = brdf_parameters[9] 

  # Note the following, with respect to potential to pre-cache as many inverse rendering optimization computations as possible:
  # - values which are unchanged with respect to everything but BRDF parameters are marked with a *
  # - values which change with respect to one or more BRDF parameters are marked with a f( ) and those parameters inside, e.g. f(parameter_1, parameter_2, ...)

  # halfway vector
  H = normalize(L+V) # *
  HdotX = tf.reduce_sum(tf.math.multiply(H, X), axis=1) # *
  HdotY = tf.reduce_sum(tf.math.multiply(H, Y), axis=1) # *
  
  # metallic gradient
  if "metallic" not in brdf_parameters_to_hold_constant_in_optimization:
    right_d_Fs_metallic = tf.expand_dims((1.0 - FH), axis=1) # *
    right_d_Fs_metallic = tf.broadcast_to(right_d_Fs_metallic, [number_of_pixels, 3]) # *
    d_Fs_metallic = Cdlin - 0.08 * specular * mix(tf.ones((number_of_pixels, 3), dtype=tf.float64), Ctint, specularTint) * right_d_Fs_metallic # f(specular, specularTint)
    NdotL3d = tf.broadcast_to(tf.expand_dims(NdotL, axis=1), [number_of_pixels, 3]) # *
    mix_Fd_ss_subsurface = tf.broadcast_to(tf.expand_dims(mix(Fd, ss, subsurface), axis=1), [number_of_pixels, 3]) # f(subsurface)
    Gs_Ds = tf.broadcast_to(tf.expand_dims(Gs * Ds, axis=1), [number_of_pixels, 3]) # *
    df_metallic = NdotL3d * ((-1.0 / pi) * mix_Fd_ss_subsurface * Cdlin + Fsheen + Gs_Ds * d_Fs_metallic) # f(specular, specularTint, subsurface)
  else:
    df_metallic = tf.constant(0.0, dtype=tf.float64)
    
  # subsurface gradient 
  if "subsurface" not in brdf_parameters_to_hold_constant_in_optimization:
    left = tf.broadcast_to(tf.expand_dims(NdotL * (1.0 / pi) * (1.0 - metallic) * (ss - Fd), axis=1), [number_of_pixels, 3]) # f(metallic)
    df_subsurface = left * Cdlin # f(metallic)
  else:
    df_subsurface = tf.constant(0.0, dtype=tf.float64)

  # specular gradient
  if "specular" not in brdf_parameters_to_hold_constant_in_optimization:
    left = tf.broadcast_to(tf.expand_dims(NdotL * Gs * Ds * (1.0 - FH) * (1.0 - metallic) * 0.08, axis=1), [number_of_pixels, 3]) # f(metallic)
    df_specular = left * mix(tf.ones(3, dtype=tf.float64), Ctint, specularTint) # f(metallic, specularTint)
  else:
    df_specular = tf.constant(0.0, dtype=tf.float64)

  # roughness gradient
  if "roughness" not in brdf_parameters_to_hold_constant_in_optimization:
    d_ss_roughness = 1.25 * LdotH * LdotH * (FV - 2.0 * FL * FV + FL + 2.0 * LdotH * LdotH * FL * FV * roughness) # f(roughness)
    d_Fd_roughness = 2.0 * LdotH ** 2 * (FV + FL + 2.0 * FL * FV * (Fd90 - 1.0)) # * 
    d_Gs_roughness = 0.5 * (roughness + 1.0) * (d_GGX_aG(NdotL, aG) * smithG_GGX(NdotV, aG) + d_GGX_aG(NdotV, aG) * smithG_GGX(NdotL, aG)) # f(roughness)
    roughness = tf.cond(roughness <= 0, lambda: 0.001, lambda: roughness) # f(roughness)
    Ds_expand = tf.broadcast_to(tf.expand_dims(Ds, axis=1), [number_of_pixels, 3]) # *
    Gs_expand = tf.broadcast_to(tf.expand_dims(Gs, axis=1), [number_of_pixels, 3]) # *
    c = tf.convert_to_tensor(sqr(HdotX) / sqr(ax) + sqr(HdotY) / sqr(ay) + sqr(NdotH), dtype=tf.float64) # *
    d_Ds_roughness = 4.0 * ((2.0 *  (HdotX**2 * aspect**4 + HdotY ** 2) / (aspect**2 * roughness)) - c * roughness**3) / (pi * ax**2 * ay**2 * c**3) # f(roughness)
    left = tf.broadcast_to(tf.expand_dims(NdotL * (1.0 - metallic) * (1.0 / pi) * mix(d_Fd_roughness, d_ss_roughness, subsurface), axis=1), [number_of_pixels, 3]) # f(metallic, roughness, subsurface)
    right = tf.broadcast_to(tf.expand_dims((d_Gs_roughness * Ds + d_Ds_roughness * Gs), axis=1), [number_of_pixels, 3]) # f(roughness)
    df_roughness = left * Cdlin + Fs * right #  f(metallic, roughness, subsurface)
  else:
    df_roughness = tf.constant(0.0, dtype=tf.float64)

  # specularTint gradient 
  if "specularTint" not in brdf_parameters_to_hold_constant_in_optimization:
    middle = tf.broadcast_to(tf.expand_dims((1.0 - FH) * specular * 0.08 * (1.0 - metallic), axis=1), [number_of_pixels, 3]) # f(specular, metallic)
    df_specularTint = NdotL3d * Gs_expand * Ds_expand * middle * (Ctint - 1.0) # f(specular, metallic)
  else:
    df_specularTint = tf.constant(0.0, dtype=tf.float64)

  # anisotropic gradient
  if "anisotropic" not in brdf_parameters_to_hold_constant_in_optimization:
    d_GTR2aniso_aspect = 4.0 * (sqr(HdotY) - sqr(HdotX) * tf.math.pow(aspect,4)) / (pi * sqr(ax) * sqr(ay) * tf.math.pow(c,3) * tf.math.pow(aspect,3)) # *
    d_Ds_anisotropic = (-0.45 / aspect) * (d_GTR2aniso_aspect) # *
    aniso_left = tf.broadcast_to(tf.expand_dims(NdotL * Gs * d_Ds_anisotropic, axis=1), [number_of_pixels, 3]) # *
    df_anisotropic = aniso_left * Fs # *
  else:
    df_anisotropic = tf.constant(0.0, dtype=tf.float64)

  # sheen gradient
  if "sheen" not in brdf_parameters_to_hold_constant_in_optimization:
    FH_expand = tf.broadcast_to(tf.expand_dims(FH, axis=1), [number_of_pixels, 3]) # *
    df_sheen = NdotL3d * (1.0 - metallic) * FH_expand * Csheen # f(metallic)
  else:
    df_sheen = tf.constant(0.0, dtype=tf.float64)

  # sheenTint gradient
  if "sheenTint" not in brdf_parameters_to_hold_constant_in_optimization:
    df_sheenTint = NdotL3d * (1.0 - metallic) * FH_expand * sheen * (Ctint - 1.0) # f(metallic, sheen)
  else:
    df_sheenTint = tf.constant(0.0, dtype=tf.float64)

  # clearcoat gradient
  if "clearcoat" not in brdf_parameters_to_hold_constant_in_optimization:    
    df_clearcoat = NdotL * 0.25 * Gr * Fr * Dr * 1.0 # *
    df_clearcoat = tf.broadcast_to(tf.expand_dims(df_clearcoat, axis=1), [number_of_pixels, 3]) # *
    df_clearcoat= tf.ones((number_of_pixels, 3), dtype=tf.float64) * df_clearcoat
  else:
    df_clearcoat = tf.constant(0.0, dtype=tf.float64)

  # clearcoatGloss gradient
  if "clearcoatGloss" not in brdf_parameters_to_hold_constant_in_optimization:    
    a = mix(0.1, 0.001, clearcoatGloss) # f(clearcoatGloss)
    t = 1.0 + (sqr(a) - 1.0) * sqr(NdotH) # *
    d_GTR1_a = 2.0 * a * ( tf.math.log(sqr(a)) * t - (sqr(a) - 1.0) * (t/(sqr(a)) + tf.math.log(sqr(a)) * sqr(NdotH))) / (pi * sqr((tf.math.log(sqr(a)) * t))) # f(clearcoatGloss)  
    df_clearcoatGloss = NdotL * 0.25 * clearcoat * -0.099 * Gr * Fr * d_GTR1_a # f(clearcoat, clearcoatGloss)
    df_clearcoatGloss = tf.broadcast_to(tf.expand_dims(df_clearcoatGloss, axis=1), [number_of_pixels, 3])
    df_clearcoatGloss = tf.ones((number_of_pixels, 3), dtype=tf.float64) * df_clearcoatGloss
  else:
    df_clearcoatGloss = tf.constant(0.0, dtype=tf.float64)
  
  gradients = [df_metallic, df_subsurface, df_specular, df_roughness, df_specularTint, df_anisotropic, df_sheen, df_sheenTint, df_clearcoat, df_clearcoatGloss]
  #gradients = [tf.expand_dims(gradient, axis=2) for gradient in gradients]
  #brdf_gradients = tf.concat(gradients, axis=2)

  return gradients


@tf.function(experimental_compile=True)
def photometric_error(ground_truth, hypothesis):
  # # number_of_pixels = ground_truth.shape[0]
  # pixelwise_squared_error = hypothesis - ground_truth
  # # average_pixel_error = tf.reduce_sum(pixelwise_squared_error) / number_of_pixels
  return hypothesis - ground_truth


def visualize_image_condition(data, label, condition="is_nan", is_true="white", is_false="black", iteration_number=0):
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


@tf.function(experimental_compile=True)
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

  number_of_pixels = ground_truth_radiance.shape[0]
  number_of_colors = ground_truth_radiance.shape[1]

  # gradients of the BRDF equation for each parameter, with respect to the loss function
  brdf_gradients = compute_gradients( brdf_metadata=hypothesis_brdf_metadata, 
                                      brdf_parameters=hypothesis_brdf_parameters,
                                      brdf_parameters_to_hold_constant_in_optimization=brdf_parameters_to_hold_constant_in_optimization)

  df_metallic, df_subsurface, df_specular, df_roughness, df_specularTint, df_anisotropic, df_sheen, df_sheenTint, df_clearcoat, df_clearcoatGloss = brdf_gradients

  # pixelwise difference across RGB channels                         
  photometric_loss = photometric_error(ground_truth_radiance, hypothesis_radiance)

  # gradient loss attributable to gamma encoding
  gamma_encoding_loss = tf.math.divide(tf.math.pow(hypothesis_brdf * hypothesis_irradiance, -1.2 / 2.2), 2.2) * hypothesis_irradiance

  # compute update to each BRDF parameter
  delta_metallic = tf.reduce_sum(df_metallic * gamma_encoding_loss * photometric_loss) / (number_of_pixels * number_of_colors)
  delta_subsurface = tf.reduce_sum(df_subsurface * gamma_encoding_loss * photometric_loss) / (number_of_pixels * number_of_colors)
  delta_specular = tf.reduce_sum(df_specular * gamma_encoding_loss * photometric_loss) / (number_of_pixels * number_of_colors)
  delta_roughness = tf.reduce_sum(df_roughness * gamma_encoding_loss * photometric_loss) / (number_of_pixels * number_of_colors)
  delta_specularTint = tf.reduce_sum(df_specularTint * gamma_encoding_loss * photometric_loss) / (number_of_pixels * number_of_colors)
  delta_anisotropic = tf.reduce_sum(df_anisotropic * gamma_encoding_loss * photometric_loss) / (number_of_pixels * number_of_colors)
  delta_sheen = tf.reduce_sum(df_sheen * gamma_encoding_loss * photometric_loss) / (number_of_pixels * number_of_colors)
  delta_sheenTint = tf.reduce_sum(df_sheenTint * gamma_encoding_loss * photometric_loss) / (number_of_pixels * number_of_colors) 
  delta_clearcoat = tf.reduce_sum(df_clearcoat * gamma_encoding_loss * photometric_loss) / (number_of_pixels * number_of_colors)
  delta_clearcoatGloss = tf.reduce_sum(df_clearcoatGloss * gamma_encoding_loss * photometric_loss) / (number_of_pixels * number_of_colors)

  # get the previous BRDF parameters
  metallic = tf.Variable(hypothesis_brdf_parameters[0])
  subsurface = tf.Variable(hypothesis_brdf_parameters[1])
  specular = tf.Variable(hypothesis_brdf_parameters[2])
  roughness = tf.Variable(hypothesis_brdf_parameters[3])
  specularTint = tf.Variable(hypothesis_brdf_parameters[4])
  anisotropic = tf.Variable(hypothesis_brdf_parameters[5])
  sheen = tf.Variable(hypothesis_brdf_parameters[6])
  sheenTint = tf.Variable(hypothesis_brdf_parameters[7])
  clearcoat = tf.Variable(hypothesis_brdf_parameters[8])
  clearcoatGloss = tf.Variable(hypothesis_brdf_parameters[9])

  # get the ground truth BRDF parameters for showing to human
  true_metallic = ground_truth_brdf_parameters[0]
  true_subsurface = ground_truth_brdf_parameters[1]
  true_specular = ground_truth_brdf_parameters[2]
  true_roughness = ground_truth_brdf_parameters[3]
  true_specularTint = ground_truth_brdf_parameters[4]
  true_anisotropic = ground_truth_brdf_parameters[5]
  true_sheen = ground_truth_brdf_parameters[6]
  true_sheenTint = ground_truth_brdf_parameters[7]
  true_clearcoat = ground_truth_brdf_parameters[8]
  true_clearcoatGloss = ground_truth_brdf_parameters[9]

  # compute the new BRDF parameter with clamping in the update
  new_metallic = clamp(metallic - delta_metallic, 0, 1)
  new_subsurface = clamp(subsurface - delta_subsurface, 0, 1)
  new_specular = clamp(specular - delta_specular, 0, 1)
  new_roughness = clamp(roughness - delta_roughness, 0, 1)
  new_specularTint = clamp(specularTint - delta_specularTint, 0, 1)
  new_anisotropic = clamp(anisotropic - delta_anisotropic, 0, 1)
  new_sheen = clamp(sheen - delta_sheen, 0, 1)
  new_sheenTint = clamp(sheenTint - delta_sheenTint, 0, 1)
  new_clearcoat = clamp(clearcoat - delta_clearcoat, 0, 1)
  new_clearcoatGloss = clamp(clearcoatGloss - delta_clearcoatGloss, 0, 1)

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

  if report_results_on_this_iteration:
    print(":::::::::::: BRDF Truth vs. Hypothesis (Δ Update) ::::::::::::")
    print("Metallic:        {:.5f} vs. {:.5f} (Δ {:+5f})".format(true_metallic, new_metallic, -1 * delta_metallic))
    print("Subsurface:      {:.5f} vs. {:.5f} (Δ {:+5f})".format(true_subsurface, new_subsurface, -1 * delta_subsurface))
    print("Specular:        {:.5f} vs. {:.5f} (Δ {:+5f})".format(true_specular, new_specular, -1 * delta_specular))
    print("Roughness:       {:.5f} vs. {:.5f} (Δ {:+5f})".format(true_roughness, new_roughness, -1 * delta_roughness))
    print("Specular Tint:   {:.5f} vs. {:.5f} (Δ {:+5f})".format(true_specularTint, new_specularTint, -1 * delta_specularTint))
    print("Anisotropic:     {:.5f} vs. {:.5f} (Δ {:+5f})".format(true_anisotropic, new_anisotropic, -1 * delta_anisotropic))
    print("Sheen:           {:.5f} vs. {:.5f} (Δ {:+5f})".format(true_sheen, new_sheen, -1 * delta_sheen))
    print("Sheen Tint:      {:.5f} vs. {:.5f} (Δ {:+5f})".format(true_sheenTint, new_sheenTint, -1 * delta_sheenTint))
    print("Clearcoat:       {:.5f} vs. {:.5f} (Δ {:+5f})".format(true_clearcoat, new_clearcoat, -1 * delta_clearcoat))
    print("Clearcoat Gloss: {:.5f} vs. {:.5f} (Δ {:+5f})\n".format(true_clearcoatGloss, new_clearcoatGloss, -1 * delta_clearcoatGloss))

  if optimizer != "L-BFGS":
    clipped_gradients = [metallic_grad, subsurface_grad, specular_grad, roughness_grad, specularTint_grad, anisotropic_grad, sheen_grad, sheenTint_grad, clearcoat_grad, clearcoatGloss_grad]
    parameters_to_update = [metallic, subsurface, specular, roughness, specularTint, anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss]


    # apply gradients with the power of a TensorFlow optimizer to tune learning rate automatically
    optimizer.apply_gradients(zip(clipped_gradients, parameters_to_update))

    # variables were updated by above operation, now we wrap it up and send it back for rendering...
    new_hypothesis_brdf_parameters = tf.concat([  metallic, 
                                                  subsurface, 
                                                  specular, 
                                                  roughness, 
                                                  specularTint, 
                                                  anisotropic, 
                                                  sheen, 
                                                  sheenTint, 
                                                  clearcoat, 
                                                  clearcoatGloss], axis=0)

    return parameters_to_update, clipped_gradients

  else:
    clipped_gradients = [metallic_grad, subsurface_grad, specular_grad, roughness_grad, specularTint_grad, anisotropic_grad, sheen_grad, sheenTint_grad, clearcoat_grad, clearcoatGloss_grad]

    gradients_as_tensor = tf.stack(clipped_gradients)
    photometric_loss = tf.reduce_sum(photometric_loss) / (number_of_pixels * number_of_colors)

    print(photometric_loss)
    print(gradients_as_tensor)

    return photometric_loss, gradients_as_tensor


def initialize_random_brdf_parameters(brdf_parameters_to_hold_constant_in_optimization):
  if "metallic" not in brdf_parameters_to_hold_constant_in_optimization:
    metallic = tf.random.uniform(shape=[1], dtype=tf.float64)
  else:
    metallic = tf.constant([0.0], dtype=tf.float64)
  if "subsurface" not in brdf_parameters_to_hold_constant_in_optimization:
    subsurface = tf.random.uniform(shape=[1], dtype=tf.float64)
  else:
    subsurface = tf.constant([0.0], dtype=tf.float64)
  if "specular" not in brdf_parameters_to_hold_constant_in_optimization:
    specular = tf.random.uniform(shape=[1], dtype=tf.float64)
  else:
    specular = tf.constant([0.5], dtype=tf.float64)
  if "roughness" not in brdf_parameters_to_hold_constant_in_optimization:
    roughness = tf.random.uniform(shape=[1], dtype=tf.float64)
  else:
    roughness = tf.constant([0.5], dtype=tf.float64)
  if "specularTint" not in brdf_parameters_to_hold_constant_in_optimization:
    specularTint = tf.random.uniform(shape=[1], dtype=tf.float64)
  else:
    specularTint = tf.constant([0.0], dtype=tf.float64)
  if "anisotropic" not in brdf_parameters_to_hold_constant_in_optimization:
    anisotropic = tf.random.uniform(shape=[1], dtype=tf.float64)
  else:
    anisotropic = tf.constant([0.0], dtype=tf.float64)
  if "sheen" not in brdf_parameters_to_hold_constant_in_optimization:
    sheen = tf.random.uniform(shape=[1], dtype=tf.float64)
  else:
    sheen = tf.constant([0.0], dtype=tf.float64)
  if "sheenTint" not in brdf_parameters_to_hold_constant_in_optimization:
    sheenTint = tf.random.uniform(shape=[1], dtype=tf.float64)
  else:
    sheenTint = tf.constant([0.0], dtype=tf.float64)
  if "clearcoat" not in brdf_parameters_to_hold_constant_in_optimization:
    clearcoat = tf.random.uniform(shape=[1], dtype=tf.float64)
  else:
    clearcoat = tf.constant([0.0], dtype=tf.float64)
  if "clearcoatGloss" not in brdf_parameters_to_hold_constant_in_optimization:
    clearcoatGloss = tf.random.uniform(shape=[1], dtype=tf.float64)
  else:
    clearcoatGloss = tf.constant([0.0], dtype=tf.float64)

  brdf_parameters = tf.concat([metallic, subsurface, specular, roughness, specularTint, anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss], axis=0)
  return brdf_parameters


# @tf.function(experimental_compile=True)
def inverse_render_optimization(folder, random_hypothesis_brdf_parameters=True, number_of_iterations = 750, frequency_of_human_output = 1):
  # compute ground truth scene parameters (namely, the radiance values from the render, used in the photometric loss function)
  ground_truth_render_parameters, ground_truth_radiance, ground_truth_irradiance, ground_truth_brdf, ground_truth_brdf_metadata, brdf_parameters_to_hold_constant_in_optimization = load_scene(folder=project_directory)

  # unwrap render parameters
  diffuse, xyz, normals, camera_xyz, light_xyz, light_color, background_color, image_shape, ground_truth_brdf_parameters, pixel_indices_to_render, is_not_background = ground_truth_render_parameters

  # initialize hypothesis for BRDF parameters
  if random_hypothesis_brdf_parameters:
    # get random hypothesis values with some parameters that are held fixed at reasonable defaults
    hypothesis_brdf_parameters = initialize_random_brdf_parameters(brdf_parameters_to_hold_constant_in_optimization=brdf_parameters_to_hold_constant_in_optimization)
  else:
    # manually set brdf ground truth parameters as tensor constants
    metallic = tf.constant(0.5, dtype=tf.float64)
    subsurface = tf.constant(0.5, dtype=tf.float64)
    specular = tf.constant(0.5, dtype=tf.float64)
    roughness = tf.constant(0.7, dtype=tf.float64)
    specularTint = tf.constant(0.0, dtype=tf.float64)
    anisotropic = tf.constant(0.0, dtype=tf.float64)
    sheen = tf.constant(0.0, dtype=tf.float64)
    sheenTint = tf.constant(0.0, dtype=tf.float64)
    clearcoat = tf.constant(0.0, dtype=tf.float64)
    clearcoatGloss = tf.constant(0.0, dtype=tf.float64)
    hypothesis_brdf_parameters = tf.concat([metallic, subsurface, specular, roughness, specularTint, anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss], axis=0)

  # initialize output directories
  Path("{}/inverse_render_hypotheses".format(folder)).mkdir(parents=True, exist_ok=True)
  Path("{}/debugging_visualizations".format(folder)).mkdir(parents=True, exist_ok=True)


  #learning_rate_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=1.0, decay_steps=500, end_learning_rate=0.01, power=2.0, cycle=False)

  # TESTED, NOT HIGHEST PERFORMERS:
  # optimizer = tf.keras.optimizers.Adadelta(learning_rate = 0.1, rho = 0.95, epsilon=1e-05)
  # optimizer = tf.keras.optimizers.Adagrad(learning_rate = learning_rate_schedule, initial_accumulator_value=0.1, epsilon=1e-05)
  # optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate_schedule, momentum = 0.99, nesterov=True)
  # optimizer = tf.keras.optimizers.Nadam(learning_rate = 0.001, epsilon = 1e-3, beta_1 = 0.9, beta_2 = 0.999)
  # optimizer = tf.keras.optimizers.Adamax(learning_rate = 0.01, epsilon = 1e-5, beta_1 = 0.9, beta_2 = 0.999)
  # optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.9, epsilon=1e-07, centered=False)
  # optimizer = tf.keras.optimizers.SGD(learning_rate = 1.0, momentum = 0.0, nesterov=False)

  # MOST RELIABLE THUS FAR IS ADAM, FASTEST SOMETIMES IS L-BFGS BUT IT IS NOT RELIABLE
  optimizer = tf.keras.optimizers.Adam(learning_rate = 0.025, epsilon =  1e-9, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
  # optimizer = "L-BFGS"
  

  # NOT YET IMPLEMENTED:  
  # optimizer = tf.math.optimizer.levenberg_marquardt() # TO DO: Levenberg Marquardt, requires manually constructing jacobian from partial gradients, editing source code in TFG API


  def compute_inverse_rendering_loss_and_gradients(hypothesis_brdf_parameters):
    global iteration

    report_results_on_this_iteration = iteration % frequency_of_human_output == 0
    if report_results_on_this_iteration:
      print("\n\nITERATION {}:".format(iteration))
      render_file_path = tf.constant("{}/inverse_render_hypotheses/inverse_render_hypothesis_{}.png".format(folder, iteration))
    else:
      render_file_path = None

    hypothesis_radiance, hypothesis_irradiance, hypothesis_brdf, hypothesis_brdf_metadata = render( diffuse_colors=diffuse, 
                                                                                                    surface_xyz=xyz,
                                                                                                    normals=normals, 
                                                                                                    camera_xyz=camera_xyz, 
                                                                                                    light_xyz=light_xyz, 
                                                                                                    light_color=light_color,
                                                                                                    background_color=background_color,
                                                                                                    image_shape=image_shape,
                                                                                                    brdf_parameters=hypothesis_brdf_parameters,
                                                                                                    pixel_indices_to_render=pixel_indices_to_render,
                                                                                                    is_not_background=is_not_background,
                                                                                                    file_path=render_file_path)

    hypothesis_brdf_parameters, gradients = apply_gradients_from_inverse_rendering_loss(optimizer=optimizer,
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

    iteration += 1
    return hypothesis_brdf_parameters, gradients


  if optimizer != "L-BFGS":
    for i in range(number_of_iterations):
      hypothesis_brdf_parameters, _ = compute_inverse_rendering_loss_and_gradients(hypothesis_brdf_parameters)

  else:

    tfp.optimizer.lbfgs_minimize(
      value_and_gradients_function=compute_inverse_rendering_loss_and_gradients,
      initial_position=hypothesis_brdf_parameters,
      # previous_optimizer_results=None,
      # num_correction_pairs=100, 
      # tolerance=1e-07, 
      # x_tolerance=0, 
      # f_relative_tolerance=0,
      # initial_inverse_hessian_estimate=None, 
      # max_iterations=50, 
      # parallel_iterations=1,
      # stopping_condition=None, 
      # max_line_search_iterations=100, 
      # name=None
    )


@tf.function(experimental_compile=True)
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

  number_of_pixels = diffuse_colors.shape[0]

  # unpack BRDF parameters
  metallic = brdf_parameters[0]
  subsurface = brdf_parameters[1]
  specular = brdf_parameters[2]
  roughness = brdf_parameters[3]
  specularTint = brdf_parameters[4]
  anisotropic = brdf_parameters[5]
  sheen = brdf_parameters[6]
  sheenTint = brdf_parameters[7]
  clearcoat = brdf_parameters[8]
  clearcoatGloss = brdf_parameters[9]


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

  # saturated minimum BRDF to 0.0
  brdf = tf.math.maximum(brdf, 0.001)

  # pack up per-pixel BRDF representations for future computational access
  brdf_metadata = tf.concat([ tf.expand_dims(NdotL, axis=1), 
                              tf.expand_dims(NdotV, axis=1),
                              tf.expand_dims(NdotH, axis=1),
                              tf.expand_dims(LdotH, axis=1),
                              Cdlin,
                              Ctint,
                              Csheen,
                              tf.expand_dims(FL, axis=1),
                              tf.expand_dims(FV, axis=1),
                              tf.expand_dims(Fd90, axis=1),
                              tf.expand_dims(Fd, axis=1),
                              tf.expand_dims(ss, axis=1),
                              tf.expand_dims(Ds, axis=1),
                              tf.expand_dims(FH, axis=1),
                              Fs,
                              tf.expand_dims(Gs, axis=1),
                              Fsheen,
                              tf.expand_dims(Dr, axis=1),
                              tf.expand_dims(Fr, axis=1),
                              tf.expand_dims(Gr, axis=1),
                              tf.expand_dims(tf.broadcast_to(aspect, [number_of_pixels]), axis=1),
                              tf.expand_dims(tf.broadcast_to(ax, [number_of_pixels]), axis=1),
                              tf.expand_dims(tf.broadcast_to(ay, [number_of_pixels]), axis=1),
                              tf.expand_dims(tf.broadcast_to(aG, [number_of_pixels]), axis=1),
                              L,
                              V,
                              N,
                              X,
                              Y
                             ], axis=1)

  return brdf, brdf_metadata


@tf.function(experimental_compile=True)
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

  # apply the rendering equation
  radiance = brdf * irradiance  

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
  brdf_parameters :: 10 BRDF parameter, in the form of a 64 bit float tensor of dimension (10), defined in 2012 by Disney: https://www.disneyanimation.com/publications/physically-based-shading-at-disney/
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
  if file_path:
    print("Rendered {}".format(file_path))
    save_image(image_data=radiance, background_color=background_color, image_shape=image_shape, is_not_background=is_not_background, pixel_indices_to_render=pixel_indices_to_render, file_path=file_path)

  return radiance, irradiance, brdf, brdf_metadata


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


def load_scene( folder, 
                random_ground_truth_brdf_parameters = True,
                light_color = tf.constant(1.0, tf.float64),
                background_color = tf.constant([70/255,70/255,70/255], tf.float64),
                camera_xyz = tf.constant([1.7, 0.11, 0.7], dtype=tf.float64),
                light_xyz = tf.constant([1.7, 0.11, 0.7], dtype=tf.float64)):

  # load data on surface textures and geometry of object 
  diffuse = load_data(filepath="{}/diffuse.png".format(folder))
  normals = load_data(filepath="{}/normals.exr".format(folder))
  xyz = load_data(filepath="{}/xyz.exr".format(folder))

  # save image shape, which will be used when reformatting computations back into an image
  image_shape = tf.constant([diffuse.shape[0], diffuse.shape[1], diffuse.shape[2]], dtype=tf.int64) 

  # get a mask for selecting only pixels that are not background values (eventually, this could be saved in production as .png with alpha channel = 0.0)
  pixel_indices_to_render, is_not_background = get_pixel_indices_to_render(diffuse_color=diffuse, background_color=background_color)

  # convert data structures for textures and geometry from an image-based tensor of (width, height, colors) to a pixel-based tensor (total_active_pixels, colors)
  diffuse = tf.gather_nd(params=diffuse, indices=pixel_indices_to_render)
  normals = tf.gather_nd(params=normals, indices=pixel_indices_to_render)
  xyz = tf.gather_nd(params=xyz, indices=pixel_indices_to_render)

  # experimentally, and algebraically, the following parameters are found to destabilize the optimization
  brdf_parameters_to_hold_constant_in_optimization = ["specularTint", "anisotropic", "sheen", "sheenTint", "clearcoat", "clearcoatGloss"]

  if random_ground_truth_brdf_parameters:
    ground_truth_brdf_parameters = initialize_random_brdf_parameters(brdf_parameters_to_hold_constant_in_optimization=brdf_parameters_to_hold_constant_in_optimization)
  else:
    # manually set brdf ground truth parameters as tensor constants
    metallic = tf.constant(0.5, dtype=tf.float64)
    subsurface = tf.constant(0.5, dtype=tf.float64)
    specular = tf.constant(0.5, dtype=tf.float64)
    roughness = tf.constant(0.7, dtype=tf.float64)
    specularTint = tf.constant(0.0, dtype=tf.float64)
    anisotropic = tf.constant(0.0, dtype=tf.float64)
    sheen = tf.constant(0.0, dtype=tf.float64)
    sheenTint = tf.constant(0.0, dtype=tf.float64)
    clearcoat = tf.constant(0.0, dtype=tf.float64)
    clearcoatGloss = tf.constant(0.0, dtype=tf.float64)
    ground_truth_brdf_parameters = tf.concat([metallic, subsurface, specular, roughness, specularTint, anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss], axis=0)

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
                                                      file_path=file_path)

  # wrap up scene parameters
  render_parameters = [diffuse, xyz, normals, camera_xyz, light_xyz, light_color, background_color, image_shape, ground_truth_brdf_parameters, pixel_indices_to_render, is_not_background]
  scene = [render_parameters, radiance, irradiance, brdf, brdf_metadata, brdf_parameters_to_hold_constant_in_optimization]

  return scene


if __name__ == "__main__":
  project_directory = "{}/inverse_renders/toucan".format(os.getcwd())
  inverse_render_optimization(folder=project_directory)




