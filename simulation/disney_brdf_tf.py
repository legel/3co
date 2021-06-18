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

def sqr_tf(x):
  x = tf.constant(x, dtype=tf.float64)
  return tf.math.square(x)

def clamp_tf(x, a, b):
  absolute_min = tf.constant(a, dtype=tf.float64)
  absolute_max = tf.constant(b, dtype=tf.float64)
  x = tf.math.minimum(x, absolute_max)
  x = tf.math.maximum(absolute_min, x)
  return x

def normalize_tf(x):
  norm = np.array(tf.linalg.norm(x, axis=2))
  ones = tf.ones([1024, 1024], dtype=tf.float64)
  # nnorm = tf.where(norm[norm==0], lambda: ones, lambda: norm)
  norm[norm == 0.0] = tf.constant(1.0, dtype=tf.float64)
  norm = tf.broadcast_to(tf.expand_dims(norm, axis=2), [1024, 1024, 3])
  result = x / norm
  # result = tf.where(norm[norm == tf.constant([0.0, 0.0, 0.0], dtype=tf.float64)], lambda: x, lambda: x / norm)
  return result

# def mix_tf(x, y, a):
#   x = tf.constant(x, dtype=tf.float64)
#   y = tf.constant(y, dtype=tf.float64)
#   a = tf.constant(a, dtype=tf.float64)

#   one = tf.constant(1.0, dtype=tf.float64)

#   one_minus_a = tf.math.subtract(one, a)
#   x_times_one_minus_a = tf.math.multiply(x, one_minus_a)
#   y_times_a = tf.math.multiply(y, a)
#   addition = tf.math.add(x_times_one_minus_a, y_times_a)
#   return addition

def mix_tf(x, y, a):
  return x * (1 - a) + y * a

def SchlickFresnel_tf(u):
  m = clamp_tf(1-u, 0, 1)
  return tf.pow(m, 5) # pow(m,5)

def GTR1_tf(NdotH, a):
  NdotH = tf.constant(NdotH, dtype=tf.float64)
  a = tf.constant(a, dtype=tf.float64)

  if (a >= 1):
    result = tf.constant(1/PI, dtype=tf.float64)
    return result
  power = tf.constant(2, dtype=tf.float64)
  a2 = tf.pow(a, power)
  NdotH2 = tf.pow(NdotH, power)
  t = 1 + (a2-1)*NdotH2
  return (a2-1) / (PI*tf.math.log(a2)*t)

def GTR2_aniso_tf(NdotH, HdotX, HdotY, ax, ay):
  PI = tf.constant(3.14159265358979323846, dtype=tf.float64)
  ax_times_ay = tf.math.multiply(ax, ay)
  PI_times_ax_times_ay = tf.math.multiply(PI, ax_times_ay)
  HdotX_divide_ax = tf.math.divide(HdotX, ax)
  HdotX_divide_ax_squared = sqr_tf(HdotX_divide_ax)
  HdotY_divide_ay = tf.math.divide(HdotY, ay)
  HdotY_divide_ay_squared = sqr_tf(HdotY_divide_ay)
  NdotH_squared = sqr_tf(NdotH)
  left_side_added = tf.math.add(HdotX_divide_ax_squared, HdotY_divide_ay_squared)
  left_side_more_added = tf.math.add(left_side_added, NdotH_squared)
  left_side_sqr = sqr_tf(left_side_more_added)
  noemer = tf.math.multiply(PI_times_ax_times_ay, left_side_sqr)
  teller = tf.constant(1, dtype=tf.float64)
  return tf.math.divide(1, noemer)

def smithG_GGX_tf(Ndotv, alphaG):
  a = sqr_tf(alphaG)
  b = sqr_tf(Ndotv)
  sqr_root = tf.math.sqrt(a + b - a * b)
  noemer = tf.math.add(Ndotv, sqr_root)
  teller = tf.constant(1, dtype=tf.float64)
  return teller/noemer

def d_GGX_aG_tf(NdotA, aG):
  NdotA_squared = sqr_tf(NdotA)
  aG_squared = sqr_tf(aG)
  sum = tf.math.add(NdotA_squared, aG_squared)
  multiplied = tf.math.multiply(NdotA_squared, aG_squared)
  subtract = tf.math.subtract(sum, multiplied)
  k = tf.math.sqrt(subtract)
  # left
  subtract_1_of_ndota = tf.math.subtract(NdotA_squared, tf.constant(1.0,dtype=tf.float64))
  aG_times_subtract = tf.math.multiply(aG, subtract_1_of_ndota)

  # right
  sum_ndota_k = tf.math.add(NdotA, k)
  sum_ndota_k_squared = sqr_tf(sum_ndota_k)
  k_times_sum_ndota_k_squared = tf.multiply(k, sum_ndota_k_squared)
  return tf.math.divide(aG_times_subtract, k_times_sum_ndota_k_squared)

def mon2lin_tf(x):
  x_tf = tf.math.pow(x, 2.2)
  return x_tf

def BRDF_wrapper_tf(L, V, N, X, Y, diffuse,brdf_params):
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

  return BRDF_tf(L, V, N, X, Y, diffuse, baseColor, metallic, subsurface, specular, roughness, specularTint, anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss) 

def BRDF_tf( L, V, N, X, Y, diffuse, baseColor = np.asarray([.82, .67, .16]), metallic = 0, subsurface = 0, specular = 0.5,
	roughness = 0.5, specularTint = 0, anisotropic = 0, sheen = 0, sheenTint = 0.5, clearcoat = 0, clearcoatGloss = 1.0 ):

  L = normalize_tf(L)
  V = normalize_tf(V)
  N = normalize_tf(N)
  X = normalize_tf(X)
  Y = normalize_tf(Y)

  NdotL = tf.reduce_sum(tf.math.multiply(N, L), axis=2)
  NdotV = tf.reduce_sum(tf.math.multiply(N, V), axis=2)

  H = normalize_tf(tf.math.add(L, V)) 

  NdotH = tf.reduce_sum(tf.math.multiply(N, H), axis=2)
  LdotH = tf.reduce_sum(tf.math.multiply(L, H), axis=2)
  print(tf.shape(diffuse))
  Cdlin = mon2lin_tf(diffuse)
  print(tf.shape(Cdlin))
  Cdlum =  0.3*Cdlin[:,:,0] + 0.6*Cdlin[:,:,1]  + 0.1*Cdlin[:,:,2] # luminance approx.
  Cdlum_exp = tf.broadcast_to(tf.expand_dims(Cdlum, axis=2), [1024, 1024, 3])
  Ctint = tf.where(Cdlum_exp > 0, Cdlin/Cdlum_exp, tf.ones((1024, 1024, 3), dtype=tf.float64))

  Cspec0 = mix_tf(specular * .08 * mix_tf(tf.ones((1024, 1024, 3), dtype=tf.float64), Ctint, specularTint), Cdlin, metallic)
  Csheen = mix_tf(tf.ones((1024, 1024, 3), dtype=tf.float64), Ctint, sheenTint) 

  # Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
  # and mix in diffuse retro-reflection based on roughness
  FL = SchlickFresnel_tf(NdotL)
  FV = SchlickFresnel_tf(NdotV)
  Fd90 = 0.5 + 2 * sqr_tf(LdotH) * roughness 
  Fd = mix_tf(1, Fd90, FL) * mix_tf(1, Fd90, FV)

  # Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
  # 1.25 scale is used to (roughly) preserve albedo
  # Fss90 used to "flatten" retroreflection based on roughness
  Fss90 = LdotH*LdotH*roughness
  Fss = mix_tf(1, Fss90, FL) * mix_tf(1, Fss90, FV)
  ss = 1.25 * (Fss * (1 / (NdotL + NdotV) - .5) + .5)

  # specular
  anisotropic = tf.math.minimum(tf.constant(0.99, dtype=tf.float64), anisotropic) # added this to prevent division by zero
  aspect = tf.constant(tf.math.sqrt(1-anisotropic*.9), dtype=tf.float64)
  
  ax = tf.math.maximum(tf.constant(.001, dtype=tf.float64), sqr_tf(roughness)/aspect)
  ay = tf.math.maximum(tf.constant(.001, dtype=tf.float64), sqr_tf(roughness)*aspect)
  HdotX = np.sum(H * X, axis=2)
  HdotY = np.sum(H * Y, axis=2)
  Ds = GTR2_aniso_tf(NdotH, HdotX, HdotY, ax, ay)
  Ds_exp = tf.broadcast_to(tf.expand_dims(Ds, axis=2),[1024, 1024, 3])
  FH = SchlickFresnel_tf(LdotH)
  FH_exp = tf.broadcast_to(tf.expand_dims(FH, axis=2),[1024, 1024, 3])
  Fs = mix_tf(Cspec0, tf.ones((1024, 1024, 3), dtype=tf.float64), FH_exp)
  # Fs = Fs[:,:,0]
  # Fs = mix(Cspec0, tf.ones((1024, 1024, 3), dtype=tf.float64), FN)

  # Gs = smithG_GGX_aniso(NdotL, np.dot(L, X), np.dot(L, Y), ax, ay)
  # Gs = Gs * smithG_GGX_aniso(NdotV, np.dot(V, X), np.dot(V, Y), ax, ay)
  aG = sqr_tf((0.5 * (roughness + 1)))

  Gs = smithG_GGX_tf(NdotL, aG) * smithG_GGX_tf(NdotV, aG)
  Gs_exp = tf.broadcast_to(tf.expand_dims(Gs, axis=2),[1024, 1024, 3])
  # sheen
  Fsheen = FH_exp * sheen * Csheen

  # clearcoat (ior = 1.5 -> F0 = 0.04)
  Dr = GTR1_tf(NdotH, mix_tf(.1,.001,clearcoatGloss))
  Fr = mix_tf(.04, 1, FH)
  Gr = smithG_GGX_tf(NdotL, .25) * smithG_GGX_tf(NdotV, .25)
  # return NdotL * (((1/PI) * mix(Fd, ss, subsurface)*Cdlin + Fsheen) * (1-metallic) + Gs*Fs*Ds + .25*clearcoat*Gr*Fr*Dr)

  # Innmann has the above leading NdotL, whereas original (below) does not?  
  mix_fd_ss_subs = mix_tf(Fd, ss, subsurface)
  mix_fd_ss_subs = tf.broadcast_to(tf.expand_dims(mix_fd_ss_subs,axis=2),[1024, 1024, 3])
  Cdlin_mix_fd = Cdlin * mix_fd_ss_subs

  clearcoat_gr_fr_dr = .25 * clearcoat*Gr*Fr*Dr 
  clearcoat_gr_fr_dr = tf.broadcast_to(tf.expand_dims(clearcoat_gr_fr_dr,axis=2),[1024, 1024, 3])
  brdf = ((1/PI) * Cdlin_mix_fd + Fsheen) * (1-metallic) + clearcoat_gr_fr_dr + Gs_exp*Fs*Ds_exp

  return L, V, N, X, Y, NdotL, NdotV, NdotH, LdotH, Cdlin, Cdlum, Ctint, Cspec0, Csheen, FL, FV, Fd90, Fd, Fss90, Fss, ss, anisotropic, aspect, ax, ay,Ds, FH, Fs, aG, Gs, Fsheen, Dr, Fr, Gr, brdf
  # return ((1/PI) * test + Fsheen) * (1-metallic) + Gs*Fs*Ds + .25*clearcoat*Gr*Fr*Dr

def brdf_gradient_tf( L, V, N, X, Y, diffuse, baseColor = np.asarray([.82, .67, .16]), metallic = 0, subsurface = 0, specular = 0.5,
	roughness = 0.5, specularTint = 0, anisotropic = 0, sheen = 0, sheenTint = 0.5, clearcoat = 0, clearcoatGloss = 1.0 ):
  
  
  L, V, N, X, Y, NdotL, NdotV, NdotH, LdotH, C_d, Cdlum, C_tint, C_spec0,\
     C_sheen, F_L, F_V, F_d90, F_d, F_ss90, F_ss, ss, anisotropic, aspect, ax, ay,\
       D_s, F_H, F_s, aG, G_s, F_sheen, D_r, F_r, G_r, brdf = BRDF_tf(L, V, N, X, Y, diffuse, baseColor, metallic, subsurface, specular,
	roughness, specularTint, anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss)
  H = normalize_tf(L+V)
  HdotX = tf.reduce_sum(tf.math.multiply(H, X), axis=2)
  HdotY = tf.reduce_sum(tf.math.multiply(H, Y), axis=2)
  
  ## metallic ## 
  right_d_Fs_metallic = tf.expand_dims((1.0 - F_H), axis=2)
  right_d_Fs_metallic = tf.broadcast_to(right_d_Fs_metallic, [1024, 1024, 3])
  d_Fs_metallic = C_d - 0.08 * specular * mix_tf(tf.ones((1024, 1024, 3), dtype=tf.float64), C_tint, specularTint) * right_d_Fs_metallic

  NdotL3d = tf.broadcast_to(tf.expand_dims(NdotL, axis=2), [1024, 1024, 3])
  mix_tf_f_d_ss_subsurface = tf.broadcast_to(tf.expand_dims(mix_tf(F_d, ss, subsurface), axis=2), [1024, 1024, 3])
  G_s_D_s = tf.broadcast_to(tf.expand_dims(G_s * D_s, axis=2), [1024, 1024, 3])
  d_f_metallic = NdotL3d * ((-1.0 / PI) * mix_tf_f_d_ss_subsurface * C_d + F_sheen + G_s_D_s * d_Fs_metallic)
  ## metallic ## 
  
  ## subsurface ## 
  left = tf.broadcast_to(tf.expand_dims(NdotL * (1.0 / PI) * (1.0 - metallic) * (ss - F_d), axis=2), [1024, 1024, 3])
  d_f_subsurface = left * C_d
  ## subsurface ##

  ## specular ##  
  left = tf.broadcast_to(tf.expand_dims(NdotL * G_s * D_s * (1.0 - F_H) * (1.0 - metallic) * 0.08, axis=2), [1024, 1024, 3])
  d_f_specular = left * mix_tf(tf.ones(3, dtype=tf.float64), C_tint, specularTint)
  ## specular ##  

  ## roughness ##  
  d_ss_roughness = 1.25 * LdotH * LdotH * (F_V - 2.0*F_L*F_V + F_L + 2.0 * LdotH * LdotH * F_L * F_V * roughness )

  d_Fd_roughness = 2.0 * LdotH ** 2 * (F_V + F_L + 2.0 * F_L * F_V * (F_d90 - 1.0))
  
  d_Gs_roughness = 0.5 * (roughness + 1.0) * (d_GGX_aG_tf(NdotL, aG) * smithG_GGX_tf(NdotV, aG) + d_GGX_aG_tf(NdotV, aG) * smithG_GGX_tf(NdotL, aG) )    

  roughness = tf.cond(roughness <= 0, lambda: 0.001, lambda: roughness)
  
  D_s_expand = tf.broadcast_to(tf.expand_dims(D_s, axis=2), [1024, 1024, 3])
  G_s_expand = tf.broadcast_to(tf.expand_dims(G_s, axis=2), [1024, 1024, 3])
  c = tf.convert_to_tensor(sqr_tf(HdotX) / sqr_tf(ax) + sqr_tf(HdotY) / sqr_tf(ay) + sqr_tf(NdotH), dtype=tf.float64)

  d_Ds_roughness = 4.0 * ( (2.0 *  (HdotX**2 * aspect**4 + HdotY ** 2) / (aspect**2 * roughness)) - c * roughness**3) / (PI * ax**2 * ay**2 * c**3)
  left = tf.broadcast_to(tf.expand_dims(NdotL * (1.0 - metallic) * (1.0 / PI) * mix_tf(d_Fd_roughness, d_ss_roughness, subsurface), axis=2), [1024, 1024, 3])
  right = tf.broadcast_to(tf.expand_dims((d_Gs_roughness * D_s + d_Ds_roughness * G_s), axis=2), [1024, 1024, 3])
  d_f_roughness = left * C_d + F_s * right
  ## roughness ##  

  ## specularTint ##  
  middle = tf.broadcast_to(tf.expand_dims((1.0 - F_H) * specular * 0.08 * (1.0 - metallic), axis=2), [1024, 1024, 3])
  d_f_specularTint = NdotL3d * G_s_expand * D_s_expand * middle * (C_tint - 1.0)
  ## specularTint ## 

  ## anisotropic ## 
  d_GTR2aniso_aspect = 4.0 * (sqr_tf(HdotY) - sqr_tf(HdotX) * tf.math.pow(aspect,4)) / (PI * sqr_tf(ax) * sqr_tf(ay) * tf.math.pow(c,3) * tf.math.pow(aspect,3))
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
  a = mix_tf(0.1,.001,clearcoatGloss)
  t = 1.0 + (sqr_tf(a) - 1.0) * sqr_tf(NdotH)
  d_GTR1_a = 2.0 * a * ( tf.math.log(sqr_tf(a)) * t - (sqr_tf(a) - 1.0) * (t/(sqr_tf(a)) + tf.math.log(sqr_tf(a)) * sqr_tf(NdotH))) / (PI * sqr_tf((tf.math.log(sqr_tf(a)) * t))  )  
  d_f_clearcoatGloss = NdotL * 0.25 * clearcoat * -0.099 * G_r * F_r * d_GTR1_a
  d_f_clearcoatGloss = tf.broadcast_to(tf.expand_dims(d_f_clearcoatGloss, axis=2), [1024, 1024, 3])
  d_f_clearcoat= tf.ones((1024, 1024, 3), dtype=tf.float64) * d_f_clearcoat
  d_f_clearcoatGloss = tf.ones((1024, 1024, 3), dtype=tf.float64) * d_f_clearcoatGloss
  ## clearcoatGloss ##   
 
  a, b, c = tf.zeros((1024, 1024, 3), dtype=tf.float64), tf.zeros((1024, 1024, 3), dtype=tf.float64), tf.zeros((1024, 1024, 3), dtype=tf.float64)
  return a,b,c, d_f_metallic, d_f_subsurface, d_f_specular, \
          d_f_roughness, d_f_specularTint, d_f_anisotropic, d_f_sheen, d_f_sheenTint, \
            d_f_clearcoat, d_f_clearcoatGloss


def brdf_gradient_wrapper_tf(L,V,N,X,Y, diffuse, brdf_params):
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

  return brdf_gradient_tf(L, V, N, X, Y, diffuse, baseColor, metallic, subsurface, specular, roughness, specularTint, anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss)  

def render_disney_brdf_on_point_tf(width, height, light_pos, N, camera_pos, p, diffuse, brdf_params, diffuse_approximation=False):
  # compute orthogonal vectors in surface tangent plane
  # note: any two orthogonal vectors on the plane will do. 
  # choose the first one arbitrarily
  # baseColor_diffuse = np.asarray(diffuse)

  brdf_params['red'], brdf_params['green'], brdf_params['blue'] = diffuse[:,:,0], diffuse[:,:,1], diffuse[:,:,2]

  U = normalize_tf(tf.experimental.numpy.random.rand(width, height, 3))

  # x: surface tangent    
  X = normalize_tf(tf.linalg.cross(N, U))

  # y: surface bitangent
  Y = normalize_tf(tf.linalg.cross(N, X))

  #  V: view direction  
  vertex = p
  V = normalize_tf(np.asarray(camera_pos[:3]) - np.asarray(vertex))

  #  L: light direction (same as view direction)
  L = V

  L, V, N, X, Y, NdotL, NdotV, NdotH, LdotH, Cdlin, Cdlum, Ctint, Cspec0, Csheen, FL, FV, Fd90, Fd, Fss90, Fss, ss, anisotropic, aspect, ax, ay,Ds, FH, Fs, aG, Gs, Fsheen, Dr, Fr, Gr, brdf = tf.cond(diffuse_approximation == True,
  lambda:BRDF_tf(L=L, V=V, N=N, X=X, Y=Y, baseColor=np.asarray(brdf_params['red'], brdf_params['green'], brdf_params['blue']), metallic=0, subsurface=0, specular=0, roughness=1.0, specularTint=0, anisotropic=0,sheen=0,sheenTint=0,clearcoat=0,clearcoatGloss=0),
  lambda:BRDF_wrapper_tf(L=L, V=V, N=N, X=X, Y=Y, diffuse=diffuse, brdf_params=brdf_params))

  # # Irradiance
  irradiance = compute_irradiance_tf(light_pos, N, camera_pos, vertex)

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

def compute_irradiance_tf(light_pos, N, camera_pos, p):
  light_red = 1
  light_green = 1
  light_blue = 1
  light_pos = light_pos[:3] # in case additional camera info passed in
  # L = normalize(light_pos)
  L = normalize_tf(tf.convert_to_tensor((np.asarray(camera_pos[:3]) - np.asarray(p)), dtype = tf.float64))

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


def render_disney_brdf_image_tf(diffuse_colors, xyz_coordinates, normals, camera_pos, reflectance_params, diffuse_approximation=False):

  width, height, _ = diffuse_colors.shape

  render = tf.zeros((height,width,3), dtype=tf.float64)
  diffuse_colors = tf.constant(diffuse_colors, dtype=tf.float64)
  xyz_coordinates = tf.constant(xyz_coordinates, dtype=tf.float64)
  normals = tf.constant(normals, dtype=tf.float64)
  brdf_params = copy.deepcopy(reflectance_params)
  grey = tf.constant([70/255,70/255,70/255], tf.float64)
  print('calculating brdf...')

  brdf = render_disney_brdf_on_point_tf(width, 
                                        height, 
                                        camera_pos,
                                        normals, 
                                        camera_pos, 
                                        xyz_coordinates, 
                                        diffuse_colors, 
                                        brdf_params)
  render = tf.where(diffuse_colors!=grey,
   brdf,
   diffuse_colors)
  return render

def main_tf():

  x_offset = 0#1.1
  y_offset = 0#1.1
  z_offset = 0#0.5
  
  path = "models/toucan_05"
  fname = "{}/toucan_0.5_0_diffuse_colors_projected.png".format(path)  
  img = cv2.imread(fname)
  diffuse_colors = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
  diffuse_colors = diffuse_colors / 255.0

  fname = "{}/toucan_0.5_0_geometry.exr".format(path)  
  img = cv2.imread(fname, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  xyz_coordinates = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # xyz_coordinates = xyz_coordinates - [x_offset, y_offset, z_offset]
  xyz_coordinates = xyz_coordinates 


  # fname = "{}/toucan_1_normal_output_res_05.exr".format(path)  
  fname = 'models/toucan_05/toucan_1_normal_output_res_05.exr'
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

  render = render_disney_brdf_image_tf(diffuse_colors, xyz_coordinates, normals, camera_pos, brdf_params, False)
  render = np.array(render * 255.0,dtype=np.float32)
  render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
  cv2.imwrite("{}/toucan_0.5_0_render.png".format(path), render)

  
if __name__ == "__main__":
  start_time = time.time()
  main_tf()
  end_time = time.time()
  print(f'render took {np.round(end_time - start_time, 2)} seconds.')
  print('----------------------------------------')
