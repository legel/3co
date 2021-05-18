import math
import numpy as np
import open3d as o3d



##############################################################################
################################### BRDF #####################################
##############################################################################

PI = 3.14159265358979323846 # preserving this silliness for the sake of posterity

def sqr(x):
  return x*x

def clamp(x, a, b):
  return max(a, min(x, b))

def normalize(x):
	norm = np.linalg.norm(x)
	if norm == 0:
		return x
	return x / norm

def mix(x, y, a):
	return x * (1 - a) + y*a

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
  if (NdotL < 0 or NdotV < 0): 
    return np.zeros(3)
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
  
  return NdotL * (((1/PI) * mix(Fd, ss, subsurface)*Cdlin + Fsheen) * (1-metallic) + Gs*Fs*Ds + .25*clearcoat*Gr*Fr*Dr)

  # Innmann has the above leading NdotL, whereas original (below) does not?
  #return ((1/PI) * mix(Fd, ss, subsurface)*Cdlin + Fsheen) * (1-metallic) + Gs*Fs*Ds + .25*clearcoat*Gr*Fr*Dr


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
  L = normalize(light_pos)

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

  return radiance


# Adapted from Tensorflow Graphics example:
# https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/reflectance.ipynb
def compute_irradiance(light_pos, N, camera_pos, p):
  light_red = 1
  light_green = 1
  light_blue = 1
  light_pos = light_pos[:3] # in case additional camera info passed in
  L = normalize(light_pos)

  # as long as the object is centered at the origin, 
  # the following should result in a reasonable intensity
  light_intensity_scale = np.dot(light_pos, light_pos) 
  light_intensity = np.asarray([light_red, light_green, light_blue]) * light_intensity_scale  

  # Irradiance
  cosine_term = np.dot(N, L)
  cosine_term = max(0, cosine_term)    
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



  

# Can run this to render on a single mesh with brdf parameters set below
def main():
  fname = "test_mesh.ply"  
  mesh = o3d.io.read_triangle_mesh(fname)
  mesh.compute_vertex_normals()

	#               x    y   z    yaw  pitch
  camera_pos = [4.8, 0.0, 2.5, 90.0, 60.0]  
  brdf_params = {}
  brdf_params['red'] = 0.037
  brdf_params['green'] = 0.798
  brdf_params['blue'] = 0.6
  brdf_params['metallic'] = 0.0
  brdf_params['subsurface'] = 0.036
  #brdf_params['specular'] = 0.484
  brdf_params['specular'] = 0.0
  brdf_params['roughness'] = 0.197
  brdf_params['specularTint'] = 0.0
  brdf_params['anisotropic'] = 0.0
  brdf_params['sheen'] = 1.0
  brdf_params['sheenTint'] = 0.0
  brdf_params['clearcoat'] = 1.0
  brdf_params['clearcoatGloss'] = 1.0


  #mesh = render_diffuse_brdf_on_mesh(mesh,camera_pos[:3],np.asarray([0.8,0.0,0.0]))
  mesh = render_disney_brdf_on_mesh(mesh, camera_pos[:3], brdf_params)
  outfname = "test.ply"
  o3d.io.write_triangle_mesh(outfname, mesh)


if __name__ == "__main__":
  main()