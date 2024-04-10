import math
import numpy as np
# import open3d as o3d
import copy
import cv2
from numpy.linalg import inv

##############################################################################
################################### BRDF #####################################
##############################################################################

PI = 3.14159265358979323846 # preserving this silliness for the sake of posterity

<<<<<<< HEAD
####################
## test functions ##
####################

def tf_tests():
  _test_sqr_tf()
  _test_clamp_tf()
  # _test_normalize_tf()
  #_test_SchlickFresnel_tf()
  # _test_GTR1_tf()


####################
## test functions ##
####################

## sqr ## 
def sqr(x):
  return x*x

def sqr_tf(x):
  return tf.math.square(x)

def _test_sqr_tf():
  test_data = np.asarray([[2.0, 4.0, 8.0], [3.0, 5.0, 6.0]], dtype=np.float64)
  original_result = sqr(test_data)

  tf_test_data = tf.convert_to_tensor(test_data, dtype=tf.float64)
  tf_result = sqr_tf(tf_test_data)

  tf_result_numpy = tf_result.numpy()

  if np.array_equal(original_result, tf_result_numpy):
    print("\nsqr() test passed")
    print("{} is equal to {}".format(original_result, tf_result_numpy))
  else:
    print("\nsqr() test failed")
    print("{} is not equal to {}".format(original_result, tf_result_numpy))

def clamp(x, a, b):
  absolute_min = a
  absolute_max = b
  x = np.minimum(x, absolute_max)
  x = np.maximum(absolute_min, x)
  return x

# def clamp_tf(x, a, b):
#   absolute_min = tf.constant(a, dtype= tf.float64)
#   absolute_max = tf.constant(b, dtype= tf.float64)
#   x = tf.math.minimum(x, absolute_max)
#   x = tf.math.maximum(absolute_min, x)
#   return x
  
def clamp_tf(x, a, b):
  tf.constant(x, dtype= tf.float64)
  tf.constant(a, dtype= tf.float64)
  tf.constant(b, dtype= tf.float64)
  return tf.clip_by_value(x, a, b, name=None)

def _test_clamp_tf():
  a = np.float64(1.0)
  b = np.float64(10.0)           
  test_data = [0.1, 0.0, 0.6, 0.19, 1.4, 10, -1.0, 0.0, 5.0, 10.0, 11.0] 
  original_results = []
  tf_results = []

  for x in test_data:
    x = np.float64(x)

    original_result = clamp(x, a, b)
    original_results.append(original_result)

    tensor_x = tf.constant(x, dtype= tf.float64)
    tf_result = np.asarray(clamp_tf(tensor_x, a, b))
    tf_results.append(tf_result)

  original_results = np.asarray(original_results)
  tf_results = np.asarray(tf_results)

  if np.array_equal(original_results, tf_results):
    print("\nclamp() test passed")
    print("{} is equal to {}".format(original_results, tf_results))

  else:
    print("\nclamp() test failed")
    print("{} is not equal to {}".format(original_results, tf_results))
=======
def sqr(x):
  return x*x

def clamp(x, a, b):
  return max(a, min(x, b))
>>>>>>> 266567b33c9e1999683924136a0b2c61f9b3268c

def normalize(x):
	norm = np.linalg.norm(x)
	if norm == 0:
		return x
	return x / norm

def mix(x, y, a):
	return x * (1 - a) + y * a

def SchlickFresnel(u):
  m = clamp(1-u, 0, 1)
  m2 = m*m
  return m2*m2*m # pow(m,5)

def GTR1(NdotH, a):
  if (a >= 1): 
    return 1/PI
  a2 = a*a
  t = 1 + (a2-1)*NdotH*NdotH
  return (a2-1) / (PI*math.log(a2)*t)

def GTR2_aniso(NdotH, HdotX, HdotY, ax, ay):
  return 1 / ( PI * ax*ay * sqr( sqr(HdotX/ax) + sqr(HdotY/ay) + NdotH*NdotH ))

def smithG_GGX(Ndotv, alphaG):
  a = alphaG*alphaG
  b = Ndotv*Ndotv
  return 1/(Ndotv + math.sqrt(a + b - a*b))

# innmann version of this partial derivative seems to have an extra aG which is incorrect?
#def d_GGX_aG_innmann(NdotA, aG):
#  k = math.sqrt( aG**2 + NdotA**2 - aG**2 * NdotA**2 )
#  return aG*aG * (NdotA**2 - 1.0) / (k * (NdotA + k)**2)

def d_GGX_aG(NdotA, aG):
  k = math.sqrt( aG**2 + NdotA**2 - aG**2 * NdotA**2 )
  return aG* (NdotA**2 - 1.0) / (k * (NdotA + k)**2)

def smithG_GGX_aniso(NdotV, VdotX, VdotY, ax, ay):
  return 1 / (NdotV + math.sqrt( sqr(VdotX*ax) + sqr(VdotY*ay) + sqr(NdotV) ))

def mon2lin(x):
  return np.asarray( [pow(x[0], 2.2), pow(x[1], 2.2), pow(x[2], 2.2)] )


def get_brdf_param_names():
  return [
    'red', 'green', 'blue', 'metallic', 'subsurface', 'specular', 'roughness',
    'specularTint', 'anisotropic', 'sheen', 'sheenTint', 'clearcoat', 'clearcoatGloss'
  ]

def get_brdf_param_bounds():
  return [
    (0,1), (0,1), (0,1), 
    (0,1), 
    (0,1), 
    (0,1),
    (0,1),
    (0,1),
    (0,1),
    (0,1),
    (0,1),
    (0,1),
    (0,1)
  ]

def BRDF_wrapper(L, V, N, X, Y, brdf_params):
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

  return BRDF(L, V, N, X, Y, baseColor, metallic, subsurface, specular, roughness, specularTint, anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss)  


# Ported from https://github.com/wdas/brdf/blob/main/src/brdfs/disney.brdf
def BRDF( L, V, N, X, Y, baseColor = np.asarray([.82, .67, .16]), metallic = 0, subsurface = 0, specular = 0.5,
	roughness = 0.5, specularTint = 0, anisotropic = 0, sheen = 0, sheenTint = 0.5, clearcoat = 0, clearcoatGloss = 1.0 ):

  L = normalize(L)
  V = normalize(V)
  N = normalize(N)
  X = normalize(X)
  Y = normalize(Y)  

  NdotL = np.dot(N,L)
  NdotV = np.dot(N,V)
  if (NdotL < 0): 
    NdotL = 0.1 # TEMP HACK TO DEAL WITH BIRD FLEAS
  if (NdotV < 0):
    NdotV = 0.1 # TEMP HACK TO DEAL WITH BIRD FLEAS
  H = normalize(L+V) 
  
  NdotH = np.dot(N,H)
  LdotH = np.dot(L,H)
  Cdlin = mon2lin(baseColor)
  Cdlum = 0.3*Cdlin[0] + 0.6*Cdlin[1]  + 0.1*Cdlin[2] # luminance approx.
  Ctint = Cdlin/Cdlum if Cdlum > 0 else np.ones(3) # normalize lum. to isolate hue+sat
  Cspec0 = mix(specular*.08*mix(np.ones(3), Ctint, specularTint), Cdlin, metallic)
  Csheen = mix(np.ones(3), Ctint, sheenTint) 

  # Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
  # and mix in diffuse retro-reflection based on roughness
  FL = SchlickFresnel(NdotL)
  FV = SchlickFresnel(NdotV)
  Fd90 = 0.5 + 2 * LdotH*LdotH * roughness
  Fd = mix(1, Fd90, FL) * mix(1, Fd90, FV)

  # Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
  # 1.25 scale is used to (roughly) preserve albedo
  # Fss90 used to "flatten" retroreflection based on roughness
  Fss90 = LdotH*LdotH*roughness
  Fss = mix(1, Fss90, FL) * mix(1, Fss90, FV)
  ss = 1.25 * (Fss * (1 / (NdotL + NdotV) - .5) + .5)

  # specular
  anisotropic = min(0.99, anisotropic) # added this to prevent division by zero
  aspect = math.sqrt(1-anisotropic*.9)
  ax = max(.001, sqr(roughness)/aspect)
  ay = max(.001, sqr(roughness)*aspect)
  Ds = GTR2_aniso(NdotH, np.dot(H, X), np.dot(H, Y), ax, ay)  
  FH = SchlickFresnel(LdotH)  
  Fs = mix(Cspec0, np.ones(3), FH)
  #Gs = smithG_GGX_aniso(NdotL, np.dot(L, X), np.dot(L, Y), ax, ay)
  #Gs = Gs * smithG_GGX_aniso(NdotV, np.dot(V, X), np.dot(V, Y), ax, ay)
  aG = (0.5 * (roughness + 1))**2
  Gs = smithG_GGX(NdotL, aG) * smithG_GGX(NdotV, aG)

  # sheen
  Fsheen = FH * sheen * Csheen

  # clearcoat (ior = 1.5 -> F0 = 0.04)
  Dr = GTR1(NdotH, mix(.1,.001,clearcoatGloss))
  Fr = mix(.04, 1, FH)
  Gr = smithG_GGX(NdotL, .25) * smithG_GGX(NdotV, .25)
  
  #return NdotL * (((1/PI) * mix(Fd, ss, subsurface)*Cdlin + Fsheen) * (1-metallic) + Gs*Fs*Ds + .25*clearcoat*Gr*Fr*Dr)

  # Innmann has the above leading NdotL, whereas original (below) does not?
  return ((1/PI) * mix(Fd, ss, subsurface)*Cdlin + Fsheen) * (1-metallic) + Gs*Fs*Ds + .25*clearcoat*Gr*Fr*Dr


##############################################################################
############################## BRDF Gradient #################################
##############################################################################

def brdf_gradient_wrapper(L,V,N,X,Y, brdf_params):
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

  return brdf_gradient(L, V, N, X, Y, baseColor, metallic, subsurface, specular, roughness, specularTint, anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss)  


def brdf_gradient( L, V, N, X, Y, baseColor = np.asarray([.82, .67, .16]), metallic = 0, subsurface = 0, specular = 0.5,
	roughness = 0.5, specularTint = 0, anisotropic = 0, sheen = 0, sheenTint = 0.5, clearcoat = 0, clearcoatGloss = 1.0 ):

  L = normalize(L)
  V = normalize(V)
  N = normalize(N)
  X = normalize(X)
  Y = normalize(Y)

  NdotL = np.dot(N,L)
  NdotV = np.dot(N,V)
  H = normalize(L+V)
  
  NdotH = np.dot(N,H)
  LdotH = np.dot(L,H)

  HdotX = np.dot(H,X)
  HdotY = np.dot(H,Y)
  
  C_d = mon2lin(baseColor)

  C_tint = C_d / (C_d[0] * 0.3 + C_d[1] * 0.6 + C_d[2] * 0.1)
  F_V = (1 - NdotV) ** 5
  F_L = (1 - NdotL) ** 5
  F_H = (1 - LdotH) ** 5
  F_d90 = 0.5 + 2.0 * LdotH * LdotH * roughness
  F_d = mix(1, F_d90, F_L) * mix(1, F_d90, F_V)

  F_ss90 = LdotH**2 * roughness
  F_ss = mix(1, F_ss90, F_L) * mix(1, F_ss90, F_V)
  ss = 1.25 * (F_ss * (1.0 / (NdotL + NdotV) - 0.5 ) + 0.5)
    
  C_sheen = mix(np.ones(3), C_tint, sheenTint)    

  F_sheen = F_H * sheen * C_sheen

  C_spec0 = mix(specular*0.08 * mix(np.ones(3), C_tint, specularTint), C_d, metallic)
  anisotropic = min(0.99, anisotropic) # prevent division by zero
  aspect = math.sqrt(1-anisotropic*.9)
  ax = max(.001, sqr(roughness)/aspect) # note: this seems to cause slight inaccuracies
  ay = max(.001, sqr(roughness)*aspect)
  aG = (0.5 * (roughness + 1.0))**2
  D_s = GTR2_aniso(NdotH, HdotX, HdotY, ax, ay)
  F_s = mix(C_spec0, np.ones(3), F_H)    
  G_s = smithG_GGX(NdotL, aG) * smithG_GGX(NdotV, aG)

  a = mix(0.1, 0.001, clearcoatGloss)
  D_r = GTR1(NdotH, a)
  F_r = mix(0.04, 1.0, F_H)
  G_r = smithG_GGX(NdotL, 0.25) * smithG_GGX(NdotV, 0.25)

  d_Fs_metallic = (C_d - 0.08 * specular * mix(np.ones(3), C_tint, specularTint)) * (1.0 - F_H)
  
  d_f_metallic = NdotL * ((-1.0 / PI) * mix (F_d, ss, subsurface) * C_d + F_sheen + G_s * D_s * d_Fs_metallic)

  d_f_subsurface = NdotL * (1.0 / PI) * (1.0 - metallic) * (ss - F_d) * C_d
  
  d_f_specular = NdotL * G_s * D_s * (1.0 - F_H) * (1.0 - metallic) * 0.08 * mix(np.ones(3), C_tint, specularTint)
  
  d_ss_roughness = 1.25 * LdotH * LdotH * (F_V - 2.0*F_L*F_V + F_L + 2.0 * LdotH * LdotH * F_L * F_V * roughness )

  c = (HdotX**2) / (ax**2) + (HdotY**2) / (ay**2) + NdotH**2
  d_Fd_roughness = 2.0 * LdotH ** 2 * (F_V + F_L + 2.0 * F_L * F_V * (F_d90 - 1.0))
  
  d_Gs_roughness = 0.5 * (roughness + 1.0) * (d_GGX_aG(NdotL, aG) * smithG_GGX(NdotV, aG) + d_GGX_aG(NdotV, aG) * smithG_GGX(NdotL, aG) )    

  if roughness <= 0:
    roughness = 0.001

  d_Ds_roughness = 4.0 * ( (2.0 *  (HdotX**2 * aspect**4 + HdotY ** 2) / (aspect**2 * roughness)) - c* roughness**3) / (PI * ax**2 * ay**2 * c**3)

  d_f_roughness = NdotL * (1.0 - metallic) * (1.0 / PI) * mix(d_Fd_roughness, d_ss_roughness, subsurface) * C_d + F_s * (d_Gs_roughness * D_s + d_Ds_roughness * G_s)

  d_f_specularTint = NdotL * G_s * D_s * (1.0 - F_H) * specular * 0.08 * (1.0 - metallic) * (C_tint - 1.0)
  
  d_GTR2aniso_aspect = 4.0 * (HdotY**2 - HdotX**2 * aspect**4) / (PI * ax**2 * ay**2 * c**3 * aspect**3)
  d_Ds_anisotropic = (-0.45 / aspect) * (d_GTR2aniso_aspect)
  d_f_anisotropic = NdotL * G_s * d_Ds_anisotropic * F_s
  
  d_f_sheen = NdotL * (1.0 - metallic) * F_H * C_sheen

  d_f_sheenTint = NdotL * (1.0 - metallic) * F_H * sheen * (C_tint - 1.0)
    
  d_f_clearcoat = NdotL * 0.25 * G_r * F_r * D_r * 1.0

  t = 1.0 + (a**2 - 1.0) * (NdotH)**2
  d_GTR1_a = 2.0 * a * ( math.log(a**2) * t - (a**2 - 1.0) * (t/(a**2) + math.log(a**2) * NdotH**2)) / (PI * (math.log(a**2) * t)**2  )  
  d_f_clearcoatGloss = NdotL * 0.25 * clearcoat * -0.099 * G_r * F_r * d_GTR1_a

  return [[0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0],  d_f_metallic, d_f_subsurface, d_f_specular, d_f_roughness, d_f_specularTint, d_f_anisotropic, d_f_sheen, d_f_sheenTint, np.ones(3) * d_f_clearcoat, np.ones(3) * d_f_clearcoatGloss]


##############################################################################
############################### Rendering ####################################
##############################################################################


# Since innmann disney brdf includes NdotL in BRDF output, should it be
#   ommitted from below? Could it be that Innmann somehow wrapped irradiance into their
#   implementation of the BRDF...?
def render_disney_brdf_on_point(light_pos, N, camera_pos, p, brdf_params, diffuse_approximation=False):



  # compute orthogonal vectors in surface tangent plane
  # note: any two orthogonal vectors on the plane will do. 
  # choose the first one arbitrarily

  # x: surface tangent    
  U = normalize(np.random.rand(3))    
  X = normalize(np.cross(N, U))  

  # y: surface bitangent
  Y = normalize(np.cross(N, X))

  #  V: view direction  
  vertex = p
  V = normalize(np.asarray(camera_pos[:3]) - np.asarray(vertex))

  #  L: light direction (same as view direction)
  
  L = V

  if diffuse_approximation == True:
    baseColor = np.asarray([brdf_params['red'], brdf_params['green'], brdf_params['blue']])    
    brdf = BRDF(L=L, V=V, N=N, X=X, Y=Y, baseColor=baseColor, metallic=0, subsurface=0, specular=0, roughness=1.0, specularTint=0, anisotropic=0,sheen=0,sheenTint=0,clearcoat=0,clearcoatGloss=0)
  else:
    brdf = BRDF_wrapper(L=L, V=V, N=N, X=X, Y=Y, brdf_params=brdf_params)

  # Irradiance
  irradiance = compute_irradiance(light_pos, N, camera_pos, vertex)

  # Rendering equation 
  radiance = brdf * irradiance     

  # Saturate radiance at 1 for rendering purposes
  radiance = np.minimum(radiance, 1.0)

  # Gamma correction
  radiance = np.power(radiance, 1.0 / 2.2)    

  # Project to [0-255] and back for consistency with .ply format
  for i in range(3):
    radiance[i] = round(radiance[i] * 255.0) / 255.0


  #radiance = [NdotL, NdotL, NdotL]

  return radiance


# Adapted from Tensorflow Graphics example:
# https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/reflectance.ipynb
def compute_irradiance(light_pos, N, camera_pos, p):
  light_red = 1
  light_green = 1
  light_blue = 1
  light_pos = light_pos[:3] # in case additional camera info passed in
  #L = normalize(light_pos)
  L = normalize(np.asarray(camera_pos[:3]) - np.asarray(p))

  # as long as the object is centered at the origin, 
  # the following should result in a reasonable intensity
  #light_intensity_scale = np.dot(light_pos - np.asarray([1,1,0]), light_pos - np.asarray([1,1,0])) 
  light_intensity_scale = np.dot(light_pos - np.asarray([1,1,0]), light_pos - np.asarray([1,1,0])) 
  light_intensity = np.asarray([light_red, light_green, light_blue]) * light_intensity_scale  

  # Irradiance
  cosine_term = np.dot(N, L)
  cosine_term = max(0.5, cosine_term)  # TEMP HACK TO DEAL WITH BIRD FLEAS
  vector_light_to_surface = np.asarray(light_pos[:3]) - np.asarray(p)
  light_to_surface_distance_squared = np.dot(vector_light_to_surface, vector_light_to_surface)  
  irradiance = light_intensity / (light_to_surface_distance_squared) * cosine_term  
  return irradiance


def render_disney_brdf_on_mesh(mesh, camera_pos, brdf_params, diffuse_approximation=False):

  if not mesh.has_vertex_normals():
    raise Exception("Mesh must have vertex normals")
  for i in range(len(mesh.vertices)):
    # N: surface normal
    N = normalize(mesh.vertex_normals[i])
    p = [mesh.vertices[i][0], mesh.vertices[i][1], mesh.vertices[i][2]]
    # L: light direction (same as view direction)
    L = camera_pos
    mesh.vertex_colors[i] = render_disney_brdf_on_point(camera_pos, N, camera_pos, p, brdf_params, diffuse_approximation)
  return mesh

def render_diffuse_disney_brdf_on_point(light_pos, N, camera_pos, p, c):
  brdf_params = {}
  brdf_params['red'] = c[0]
  brdf_params['green'] = c[1]
  brdf_params['blue'] = c[2]  
  return render_disney_brdf_on_point(light_pos, N, camera_pos, p, brdf_params, True)  

def render_diffuse_disney_brdf_on_mesh(mesh, camera_pos, c):
  brdf_params = {}
  brdf_params['red'] = c[0]
  brdf_params['green'] = c[1]
  brdf_params['blue'] = c[2]  
  return render_disney_brdf_on_mesh(mesh, camera_pos, brdf_params, True)


# Here, diffuse BRDF is defined as (1/pi) * NdotL * c,
# where c is base color
def render_diffuse_brdf_on_point(light_pos, N, camera_pos, p, c):

  L = normalize(light_pos)
  N = normalize(N)
  irradiance = compute_irradiance(light_pos, N, camera_pos, p)
  radiance = irradiance * (1.0 / math.pi) * np.dot(N, L) * c

  radiance = np.power(radiance, 1.0 / 2.2)
  for i in range(3):
    radiance[i] = round(radiance[i] * 255.0) / 255.0

  return radiance  

def render_diffuse_brdf_on_mesh(mesh, camera_pos, c):
  if not mesh.has_vertex_normals():
    raise Exception("Mesh must have vertex normals")
  for i in range(len(mesh.vertices)):
    # N: surface normal
    N = normalize(mesh.vertex_normals[i])
    p = [mesh.vertices[i][0], mesh.vertices[i][1], mesh.vertices[i][2]]
    # L: light direction (same as view direction)
    L = camera_pos
    mesh.vertex_colors[i] = render_diffuse_brdf_on_point(camera_pos, N, camera_pos, p, c)
  return mesh  



def render_disney_brdf_image(diffuse_colors, xyz_coordinates, normals, camera_pos, reflectance_params, diffuse_approximation=False):

  height = len(diffuse_colors)
  width = len(diffuse_colors[0])

  render = np.zeros((height,width,3), np.float32)
  # all_zeros = np.zeros((height,width,3), np.float32)

  for i in range(height):
    for j in range(width):
      diffuse_color = diffuse_colors[i,j]
      p = xyz_coordinates[i,j]
      N = normalize(normals[i,j])
      

      if diffuse_color[0] == 70/255 and diffuse_color[1] == 70/255 and diffuse_color[2] == 70/255:
        render[i,j,:] = [70/255,70/255,70/255]
      else:

        brdf_params = copy.deepcopy(reflectance_params)

        brdf_params['red'] = diffuse_color[0]
        brdf_params['green'] = diffuse_color[1]
        brdf_params['blue'] = diffuse_color[2]

        radiance = render_disney_brdf_on_point(camera_pos, N, camera_pos, p, brdf_params, diffuse_approximation)

        # render = numpy.where(diffuse_color == [70/255,70/255,70/255], all_zeros, radiance)
        # render = tf.where(...)

        render[i,j,:] = radiance

  return render


  
def main():


  path = "models/toucan_0.5"
  fname = "{}/toucan_0.5_0_diffuse_colors_projected.png".format(path)  
  img = cv2.imread(fname, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    
  diffuse_colors = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
  diffuse_colors = diffuse_colors / 255.0

  x_offset = 0#1.1
  y_offset = 0#1.1
  z_offset = 0#0.5

  fname = "{}/toucan_0.5_0_geometry.exr".format(path)  
  img = cv2.imread(fname, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  xyz_coordinates = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  #xyz_coordinates = xyz_coordinates - [x_offset, y_offset, z_offset]
  xyz_coordinates = xyz_coordinates 

  fname = "{}/toucan_0.5_0_normals.exr".format(path)  
  img = cv2.imread(fname, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  normals = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

  # flamingo
  #camera_pos = [1.62 - x_offset, 0.07 - y_offset, 0.62 - z_offset]

  # toucan
  camera_pos = [1.7 - x_offset, 0.11 - y_offset, 0.7 - z_offset]
  
  brdf_params = {}
  brdf_params['metallic'] = 0.0
  brdf_params['subsurface'] = 0.0  
  brdf_params['specular'] = 0.5
  brdf_params['roughness'] = 0.0
  brdf_params['specularTint'] = 0.0
  brdf_params['anisotropic'] = 0.0
  brdf_params['sheen'] = 0.0
  brdf_params['sheenTint'] = 0.0
  brdf_params['clearcoat'] = 1.0
  brdf_params['clearcoatGloss'] = 1.0


  render = render_disney_brdf_image(diffuse_colors, xyz_coordinates, normals, camera_pos, brdf_params, False)
  render = render * 255.0
  
  
  render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)      
  

  cv2.imwrite("{}/toucan_0.5_0_render.png".format(path), render)



if __name__ == "__main__":
  main()