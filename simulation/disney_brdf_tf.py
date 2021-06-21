import math
import numpy as np
# import open3d as o3d
import copy
import cv2
from numpy.linalg import inv
import tensorflow as tf
import time


######################################################################  ########
################################### BRDF #####################################
##############################################################################

PI = tf.constant(3.14159265358979323846, dtype=tf.float64)
  # preserving this silliness for the sake of posterity

def sqr(x):
  x = tf.constant(x, dtype=tf.float64)
  return tf.math.square(x)

def clamp(x, a, b):
  absolute_min = tf.constant(a, dtype=tf.float64)
  absolute_max = tf.constant(b, dtype=tf.float64)
  x = tf.math.minimum(x, absolute_max)
  x = tf.math.maximum(absolute_min, x)
  return x

def normalize(x):
  norm = np.array(tf.linalg.norm(x, axis=2))
  ones = tf.ones([1024, 1024], dtype=tf.float64)
  norm[norm == 0.0] = tf.constant(1.0, dtype=tf.float64)
  norm = tf.broadcast_to(tf.expand_dims(norm, axis=2), [1024, 1024, 3])
  result = x / norm
  return result

def mix(x, y, a):
  return x * (1 - a) + y * a

def SchlickFresnel(u):
  m = clamp(1-u, 0, 1)
  return tf.pow(m, 5) # pow(m,5)

def GTR1(NdotH, a):
  if (a >= 1): 
    return 1/PI
  a2 = a*a
  t = 1 + (a2-1)*NdotH*NdotH
  return (a2-1) / (PI*math.log(a2)*t)

def GTR2_aniso(NdotH, HdotX, HdotY, ax, ay):
  shape = tf.shape(NdotH)
  ones = tf.ones(shape, dtype=tf.float64)
  return ones / ( PI * ax*ay * sqr( sqr(HdotX/ax) + sqr(HdotY/ay) + sqr(NdotH)))

def smithG_GGX(Ndotv, alphaG):
  a = sqr(alphaG)
  b = sqr(Ndotv)
  sqr_root = tf.math.sqrt(a + b - a * b)
  noemer = tf.math.add(Ndotv, sqr_root)
  teller = tf.constant(1, dtype=tf.float64)
  return teller/noemer

# innmann version of this partial derivative seems to have an extra aG which is incorrect?
#def d_GGX_aG_innmann(NdotA, aG):
#  k = math.sqrt( aG**2 + NdotA**2 - aG**2 * NdotA**2 )
#  return aG*aG * (NdotA**2 - 1.0) / (k * (NdotA + k)**2)

def d_GGX_aG(NdotA, aG):
  k = tf.math.sqrt( sqr(aG) + sqr(NdotA) - sqr(aG) * sqr(NdotA) )
  return aG * (sqr(NdotA) - 1.0) / (k * sqr((NdotA + k)))

def smithG_GGX_aniso(NdotV, VdotX, VdotY, ax, ay):
  return 1 / (NdotV + math.sqrt( sqr(VdotX*ax) + sqr(VdotY*ay) + sqr(NdotV) ))


def mon2lin(x):
  x = tf.math.pow(x, 2.2)
  return x

def BRDF_wrapper(L, V, N, X, Y, diffuse,brdf_params):
  baseColor = np.asarray([brdf_params['red'], brdf_params['green'], brdf_params['blue']])
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

  return BRDF(L, V, N, X, Y, diffuse, baseColor, metallic, subsurface, specular, roughness, specularTint, anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss) 

def BRDF( L, V, N, X, Y, diffuse, baseColor = np.asarray([.82, .67, .16]), metallic = 0, subsurface = 0, specular = 0.5,
	roughness = 0.5, specularTint = 0, anisotropic = 0, sheen = 0, sheenTint = 0.5, clearcoat = 0, clearcoatGloss = 1.0 ):

  L = normalize(L)
  V = normalize(V)
  N = normalize(N)
  X = normalize(X)
  Y = normalize(Y)

  NdotL = tf.reduce_sum(tf.math.multiply(N, L), axis=2)
  NdotV = tf.reduce_sum(tf.math.multiply(N, V), axis=2)

  H = normalize(tf.math.add(L, V)) 

  NdotH = tf.reduce_sum(tf.math.multiply(N, H), axis=2)
  LdotH = tf.reduce_sum(tf.math.multiply(L, H), axis=2)
  print(tf.shape(diffuse))
  Cdlin = mon2lin(diffuse)
  print(tf.shape(Cdlin))
  Cdlum =  0.3*Cdlin[:,:,0] + 0.6*Cdlin[:,:,1]  + 0.1*Cdlin[:,:,2] # luminance approx.
  Cdlum_exp = tf.broadcast_to(tf.expand_dims(Cdlum, axis=2), [1024, 1024, 3])
  Ctint = tf.where(Cdlum_exp > 0, Cdlin/Cdlum_exp, tf.ones((1024, 1024, 3), dtype=tf.float64))

  Cspec0 = mix(specular * .08 * mix(tf.ones((1024, 1024, 3), dtype=tf.float64), Ctint, specularTint), Cdlin, metallic)
  Csheen = mix(tf.ones((1024, 1024, 3), dtype=tf.float64), Ctint, sheenTint) 

  # Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
  # and mix in diffuse retro-reflection based on roughness
  FL = SchlickFresnel(NdotL)
  FV = SchlickFresnel(NdotV)
  Fd90 = 0.5 + 2 * sqr(LdotH) * roughness 
  Fd = mix(1, Fd90, FL) * mix(1, Fd90, FV)

  # Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
  # 1.25 scale is used to (roughly) preserve albedo
  # Fss90 used to "flatten" retroreflection based on roughness
  Fss90 = LdotH*LdotH*roughness
  Fss = mix(1, Fss90, FL) * mix(1, Fss90, FV)
  ss = 1.25 * (Fss * (1 / (NdotL + NdotV) - .5) + .5)

  # specular
  anisotropic = tf.math.minimum(tf.constant(0.99, dtype=tf.float64), anisotropic) # added this to prevent division by zero
  aspect = tf.constant(tf.math.sqrt(1-anisotropic*.9), dtype=tf.float64)
  
  ax = tf.math.maximum(tf.constant(.001, dtype=tf.float64), sqr(roughness)/aspect)
  ay = tf.math.maximum(tf.constant(.001, dtype=tf.float64), sqr(roughness)*aspect)
  HdotX = np.sum(H * X, axis=2)
  HdotY = np.sum(H * Y, axis=2)
  Ds = GTR2_aniso(NdotH, HdotX, HdotY, ax, ay)
  Ds_exp = tf.broadcast_to(tf.expand_dims(Ds, axis=2),[1024, 1024, 3])
  FH = SchlickFresnel(LdotH)
  FH_exp = tf.broadcast_to(tf.expand_dims(FH, axis=2),[1024, 1024, 3])
  Fs = mix(Cspec0, tf.ones((1024, 1024, 3), dtype=tf.float64), FH_exp)

  # Fs = mix(Cspec0, tf.ones((1024, 1024, 3), dtype=tf.float64), FN)

  # Gs = smithG_GGX_aniso(NdotL, np.dot(L, X), np.dot(L, Y), ax, ay)
  # Gs = Gs * smithG_GGX_aniso(NdotV, np.dot(V, X), np.dot(V, Y), ax, ay)
  aG = sqr((0.5 * (roughness + 1)))

  Gs = smithG_GGX(NdotL, aG) * smithG_GGX(NdotV, aG)
  Gs_exp = tf.broadcast_to(tf.expand_dims(Gs, axis=2),[1024, 1024, 3])
  # sheen
  Fsheen = FH_exp * sheen * Csheen

  # clearcoat (ior = 1.5 -> F0 = 0.04)
  Dr = GTR1(NdotH, mix(.1,.001,clearcoatGloss))
  Fr = mix(.04, 1, FH)
  Gr = smithG_GGX(NdotL, .25) * smithG_GGX(NdotV, .25)
  # return NdotL * (((1/PI) * mix(Fd, ss, subsurface)*Cdlin + Fsheen) * (1-metallic) + Gs*Fs*Ds + .25*clearcoat*Gr*Fr*Dr)

  # Innmann has the above leading NdotL, whereas original (below) does not?  
  mix_fd_ss_subs = mix(Fd, ss, subsurface)
  mix_fd_ss_subs = tf.broadcast_to(tf.expand_dims(mix_fd_ss_subs,axis=2),[1024, 1024, 3])
  Cdlin_mix_fd = Cdlin * mix_fd_ss_subs

  clearcoat_gr_fr_dr = .25 * clearcoat*Gr*Fr*Dr 
  clearcoat_gr_fr_dr = tf.broadcast_to(tf.expand_dims(clearcoat_gr_fr_dr,axis=2),[1024, 1024, 3])
  brdf = ((1/PI) * Cdlin_mix_fd + Fsheen) * (1-metallic) + clearcoat_gr_fr_dr + Gs_exp*Fs*Ds_exp

  return L, V, N, X, Y, NdotL, NdotV, NdotH, LdotH, Cdlin, Cdlum, Ctint, Cspec0, Csheen, FL, FV, Fd90, Fd, Fss90, Fss, ss, anisotropic, aspect, ax, ay,Ds, FH, Fs, aG, Gs, Fsheen, Dr, Fr, Gr, brdf

def brdf_gradient( L, V, N, X, Y, diffuse, baseColor = np.asarray([.82, .67, .16]), metallic = 0, subsurface = 0, specular = 0.5,
	roughness = 0.5, specularTint = 0, anisotropic = 0, sheen = 0, sheenTint = 0.5, clearcoat = 0, clearcoatGloss = 1.0 ):
  
  
  L, V, N, X, Y, NdotL, NdotV, NdotH, LdotH, C_d, Cdlum, C_tint, C_spec0,\
     C_sheen, F_L, F_V, F_d90, F_d, F_ss90, F_ss, ss, anisotropic, aspect, ax, ay,\
       D_s, F_H, F_s, aG, G_s, F_sheen, D_r, F_r, G_r, brdf = BRDF(L, V, N, X, Y, diffuse, baseColor, metallic, subsurface, specular,
	roughness, specularTint, anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss)
  H = normalize(L+V)
  HdotX = tf.reduce_sum(tf.math.multiply(H, X), axis=2)
  HdotY = tf.reduce_sum(tf.math.multiply(H, Y), axis=2)
  
  ## metallic ## 
  right_d_Fs_metallic = tf.expand_dims((1.0 - F_H), axis=2)
  right_d_Fs_metallic = tf.broadcast_to(right_d_Fs_metallic, [1024, 1024, 3])
  d_Fs_metallic = C_d - 0.08 * specular * mix(tf.ones((1024, 1024, 3), dtype=tf.float64), C_tint, specularTint) * right_d_Fs_metallic

  NdotL3d = tf.broadcast_to(tf.expand_dims(NdotL, axis=2), [1024, 1024, 3])
  mix_f_d_ss_subsurface = tf.broadcast_to(tf.expand_dims(mix(F_d, ss, subsurface), axis=2), [1024, 1024, 3])
  G_s_D_s = tf.broadcast_to(tf.expand_dims(G_s * D_s, axis=2), [1024, 1024, 3])
  d_f_metallic = NdotL3d * ((-1.0 / PI) * mix_f_d_ss_subsurface * C_d + F_sheen + G_s_D_s * d_Fs_metallic)
  ## metallic ## 
  
  ## subsurface ## 
  left = tf.broadcast_to(tf.expand_dims(NdotL * (1.0 / PI) * (1.0 - metallic) * (ss - F_d), axis=2), [1024, 1024, 3])
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

  d_Ds_roughness = 4.0 * ( (2.0 *  (HdotX**2 * aspect**4 + HdotY ** 2) / (aspect**2 * roughness)) - c * roughness**3) / (PI * ax**2 * ay**2 * c**3)
  left = tf.broadcast_to(tf.expand_dims(NdotL * (1.0 - metallic) * (1.0 / PI) * mix(d_Fd_roughness, d_ss_roughness, subsurface), axis=2), [1024, 1024, 3])
  right = tf.broadcast_to(tf.expand_dims((d_Gs_roughness * D_s + d_Ds_roughness * G_s), axis=2), [1024, 1024, 3])
  d_f_roughness = left * C_d + F_s * right
  ## roughness ##  

  ## specularTint ##  
  middle = tf.broadcast_to(tf.expand_dims((1.0 - F_H) * specular * 0.08 * (1.0 - metallic), axis=2), [1024, 1024, 3])
  d_f_specularTint = NdotL3d * G_s_expand * D_s_expand * middle * (C_tint - 1.0)
  ## specularTint ## 

  ## anisotropic ## 
  d_GTR2aniso_aspect = 4.0 * (sqr(HdotY) - sqr(HdotX) * tf.math.pow(aspect,4)) / (PI * sqr(ax) * sqr(ay) * tf.math.pow(c,3) * tf.math.pow(aspect,3))
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
  d_GTR1_a = 2.0 * a * ( tf.math.log(sqr(a)) * t - (sqr(a) - 1.0) * (t/(sqr(a)) + tf.math.log(sqr(a)) * sqr(NdotH))) / (PI * sqr((tf.math.log(sqr(a)) * t))  )  
  d_f_clearcoatGloss = NdotL * 0.25 * clearcoat * -0.099 * G_r * F_r * d_GTR1_a
  d_f_clearcoatGloss = tf.broadcast_to(tf.expand_dims(d_f_clearcoatGloss, axis=2), [1024, 1024, 3])
  d_f_clearcoat= tf.ones((1024, 1024, 3), dtype=tf.float64) * d_f_clearcoat
  d_f_clearcoatGloss = tf.ones((1024, 1024, 3), dtype=tf.float64) * d_f_clearcoatGloss
  ## clearcoatGloss ##   
 
  a, b, c = tf.zeros((1024, 1024, 3), dtype=tf.float64), tf.zeros((1024, 1024, 3), dtype=tf.float64), tf.zeros((1024, 1024, 3), dtype=tf.float64)
  return a,b,c, d_f_metallic, d_f_subsurface, d_f_specular, \
          d_f_roughness, d_f_specularTint, d_f_anisotropic, d_f_sheen, d_f_sheenTint, \
            d_f_clearcoat, d_f_clearcoatGloss


# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer#example_2  


# number_of_iterations = 1000
# exponential_decay_learning = range(0.01 * (0.99**i) for i in range(number_of_iterations))
#
# initial_values = ...
# hypothesis = initial_values
# for iteration in number_of_iterations:
#   observation 
#   inverse_render = f(hypothesis)
#   loss = observation - inverse_render
#   
#   gradients = brdf_gradient()
#   for parameter in parameters:
#     roughness = roughness - loss * exponential_decay_learning[iteration] * gradients["d_f_roughness"]
#     ...
# 
#   print(gradients)
#   print(loss)
#   good to visualize for every step

# 


def brdf_gradient_wrapper(L,V,N,X,Y, diffuse, brdf_params):
  baseColor = np.asarray([brdf_params['red'], brdf_params['green'], brdf_params['blue']])
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

  return brdf_gradient(L, V, N, X, Y, diffuse, baseColor, metallic, subsurface, specular, roughness, specularTint, anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss)  

def render_disney_brdf_on_point(width, height, light_pos, N, camera_pos, p, diffuse, brdf_params, diffuse_approximation=False):
  # compute orthogonal vectors in surface tangent plane
  # note: any two orthogonal vectors on the plane will do. 
  # choose the first one arbitrarily
  # baseColor_diffuse = np.asarray(diffuse)

  brdf_params['red'], brdf_params['green'], brdf_params['blue'] = diffuse[:,:,0], diffuse[:,:,1], diffuse[:,:,2]

  U = normalize(tf.experimental.numpy.random.rand(width, height, 3))

  # x: surface tangent    
  X = normalize(tf.linalg.cross(N, U))

  # y: surface bitangent
  Y = normalize(tf.linalg.cross(N, X))

  #  V: view direction  
  vertex = p
  V = normalize(np.asarray(camera_pos[:3]) - np.asarray(vertex))

  #  L: light direction (same as view direction)
  L = V

  L, V, N, X, Y, NdotL, NdotV, NdotH, LdotH, Cdlin, Cdlum, Ctint, Cspec0, Csheen, FL, FV, Fd90, Fd, Fss90, Fss, ss, anisotropic, aspect, ax, ay,Ds, FH, Fs, aG, Gs, Fsheen, Dr, Fr, Gr, brdf = tf.cond(diffuse_approximation == True,
  lambda:BRDF(L=L, V=V, N=N, X=X, Y=Y, baseColor=np.asarray(brdf_params['red'], brdf_params['green'], brdf_params['blue']), metallic=0, subsurface=0, specular=0, roughness=1.0, specularTint=0, anisotropic=0,sheen=0,sheenTint=0,clearcoat=0,clearcoatGloss=0),
  lambda:BRDF_wrapper(L=L, V=V, N=N, X=X, Y=Y, diffuse=diffuse, brdf_params=brdf_params))

  # # Irradiance
  irradiance = compute_irradiance(light_pos, N, camera_pos, vertex)

  print('calculating radiance...')
  # # Rendering equation 
  radiance = brdf * irradiance     

  # # Saturate radiance at 1 for rendering purposes
  radiance = tf.math.minimum(radiance, 1.0)

  # # Gamma correction
  radiance = tf.math.pow(radiance, 1.0 / 2.2)    

  # # Project to [0-255] and back for consistency with .ply format
  radiance = np.round(radiance * 255.0) / 255.0
  # # radiance = [NdotL, NdotL, NdotL]
  print('calculations finished.')
  return radiance

def compute_irradiance(light_pos, N, camera_pos, p):
  light_red = 1
  light_green = 1
  light_blue = 1
  light_pos = light_pos[:3] # in case additional camera info passed in
  # L = normalize(light_pos)
  L = normalize(tf.convert_to_tensor((np.asarray(camera_pos[:3]) - np.asarray(p)), dtype = tf.float64))

  # as long as the object is centered at the origin, 
  # the following should result in a reasonable intensity
  # light_intensity_scale = np.dot(light_pos - np.asarray([1,1,0]), light_pos - np.asarray([1,1,0])) 
  
  light_intensity_scale = tf.experimental.numpy.dot(light_pos - np.asarray([1,1,0]), light_pos - np.asarray([1,1,0])) 
  light_intensity = tf.constant([light_red, light_green, light_blue], dtype=tf.float64) * light_intensity_scale  

  # Irradiance
  cosine_term = tf.convert_to_tensor(np.sum(N * L, axis=2),dtype=tf.float64)
  cosine_term = tf.math.maximum(0.5, cosine_term)  # TEMP HACK TO DEAL WITH BIRD FLEAS

  vector_light_to_surface = tf.convert_to_tensor(np.asarray(light_pos[:3]) - np.asarray(p), dtype=tf.float64)
  light_to_surface_distance_squared = tf.convert_to_tensor(np.sum(vector_light_to_surface * vector_light_to_surface, axis=2), dtype=tf.float64)
  light_surf_sq_cosine = tf.broadcast_to(tf.expand_dims(light_to_surface_distance_squared * cosine_term, axis=2), [1024, 1024,3])
  light_intensity = tf.broadcast_to(light_intensity, [1024, 1024, 3])
  irradiance = light_intensity / light_surf_sq_cosine
  return irradiance


def render_disney_brdf_image(diffuse_colors, xyz_coordinates, normals, camera_pos, reflectance_params, diffuse_approximation=False):

  width, height, _ = diffuse_colors.shape

  render = tf.zeros((height,width,3), dtype=tf.float64)
  diffuse_colors = tf.constant(diffuse_colors, dtype=tf.float64)
  xyz_coordinates = tf.constant(xyz_coordinates, dtype=tf.float64)
  normals = tf.constant(normals, dtype=tf.float64)
  brdf_params = copy.deepcopy(reflectance_params)
  grey = tf.constant([70/255,70/255,70/255], tf.float64)
  print('calculating brdf...')

  brdf = render_disney_brdf_on_point(width, 
                                        height, 
                                        camera_pos,
                                        normals, 
                                        camera_pos, 
                                        xyz_coordinates, 
                                        diffuse_colors, 
                                        brdf_params)
  render = tf.where(diffuse_colors!=grey,  # where diffuse colors aren't grey... 
   brdf,                                   # calculate the brdf 
   diffuse_colors)                         # otherwise, make it grey!
  return render

def main():

  x_offset = 0#1.1
  y_offset = 0#1.1
  z_offset = 0#0.5
  
  path = "/Users/x/3cology/research/simulation/models/toucan_0.5"
  fname = "{}/toucan_0.5_0_diffuse_colors.png".format(path)  
  img = cv2.imread(fname)
  diffuse_colors = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
  diffuse_colors = diffuse_colors / 255.0

  fname = "{}/toucan_0.5_0_geometry.exr".format(path)  
  img = cv2.imread(fname, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  xyz_coordinates = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # xyz_coordinates = xyz_coordinates - [x_offset, y_offset, z_offset]
  xyz_coordinates = xyz_coordinates 


  # fname = "{}/toucan_1_normal_output_res_05.exr".format(path)  
  fname = '{}/toucan_0.5_0_normals.exr'.format(path)
  img = cv2.imread(fname, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  normals = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

  # flamingo
  # camera_pos = [1.62 - x_offset, 0.07 - y_offset, 0.62 - z_offset]

  # toucan
  camera_pos = [1.7 - x_offset, 0.11 - y_offset, 0.7 - z_offset]
  
  brdf_params = {}
  brdf_params['metallic'] = 0.4
  brdf_params['subsurface'] = 0.3  
  brdf_params['specular'] = 0.5
  brdf_params['roughness'] = 0.3
  brdf_params['specularTint'] = 0.1
  brdf_params['anisotropic'] = 0.3
  brdf_params['sheen'] = 0.4
  brdf_params['sheenTint'] = 0.1
  brdf_params['clearcoat'] = 1.0
  brdf_params['clearcoatGloss'] = 1.0

  print('----------------------------------------')
  print('starting render...')

  render = render_disney_brdf_image(diffuse_colors, xyz_coordinates, normals, camera_pos, brdf_params, False)
  render = np.array(render * 255.0,dtype=np.float32)
  render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
  cv2.imwrite("{}/toucan_0.5_0_render.png".format(path), render)

  
if __name__ == "__main__":
  start_time = time.time()
  main()
  end_time = time.time()
  print(f'render took {np.round(end_time - start_time, 2)} seconds.')
  print('----------------------------------------')
