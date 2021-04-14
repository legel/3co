import math
import numpy as np
import open3d as o3d


# [type] [name] [min val] [max val] [default val]
#baseColor = [0, 1, [.82, .67, .16] ]
#metallic = [0, 1, 0]
#subsurface = [0, 1, 0]
#specular = [0, 1, .5]
#roughness = [0, 1, .5]
#specularTint = [0, 1, 0]
#anisotropic = [0, 1, 0]
#sheen = [0, 1, 0]
#sheenTint = [0, 1, .5]
#clearcoat = [0, 1, 0]
#clearcoatGloss = [0, 1, 1]

PI = 3.14159265358979323846

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


def GTR2(NdotH, a):
  a2 = a*a
  t = 1 + (a2-1)*NdotH*NdotH
  return a2 / (PI * t*t)

def GTR2_aniso(NdotH, HdotX, HdotY, ax, ay):
  return 1 / ( PI * ax*ay * sqr( sqr(HdotX/ax) + sqr(HdotY/ay) + NdotH*NdotH ))

def smithG_GGX(Ndotv, alphaG):
  a = alphaG*alphaG
  b = Ndotv*Ndotv
  return 1/(Ndotv + math.sqrt(a + b - a*b))

def smithG_GGX_aniso(NdotV, VdotX, VdotY, ax, ay):
  return 1 / (NdotV + math.sqrt( sqr(VdotX*ax) + sqr(VdotY*ay) + sqr(NdotV) ))

def mon2lin(x):
  return np.asarray( [pow(x[0], 2.2), pow(x[1], 2.2), pow(x[2], 2.2)] )


def BRDF( L, V, N, X, Y, baseColor = np.asarray([.82, .67, .16]), metallic = 0, subsurface = 0, specular = 0.5,
	roughness = 0.5, specularTint = 0, anisotropic = 0, sheen = 0, sheenTint = 0.5, clearcoat = 0, clearcoatGloss = 1.0 ):


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
  aspect = math.sqrt(1-anisotropic*.9)
  ax = max(.001, sqr(roughness)/aspect)
  ay = max(.001, sqr(roughness)*aspect)
  Ds = GTR2_aniso(NdotH, np.dot(H, X), np.dot(H, Y), ax, ay)
  FH = SchlickFresnel(LdotH)
  Fs = mix(Cspec0, np.ones(3), FH)
  Gs = smithG_GGX_aniso(NdotL, np.dot(L, X), np.dot(L, Y), ax, ay)
  Gs = Gs * smithG_GGX_aniso(NdotV, np.dot(V, X), np.dot(V, Y), ax, ay)

  # sheen
  Fsheen = FH * sheen * Csheen

  # clearcoat (ior = 1.5 -> F0 = 0.04)
  Dr = GTR1(NdotH, mix(.1,.001,clearcoatGloss))
  Fr = mix(.04, 1, FH)
  Gr = smithG_GGX(NdotL, .25) * smithG_GGX(NdotV, .25)

  return ((1/PI) * mix(Fd, ss, subsurface)*Cdlin + Fsheen) * (1-metallic) + Gs*Fs*Ds + .25*clearcoat*Gr*Fr*Dr



def main():

  fname = "test_mesh.ply"  
  mesh = o3d.io.read_triangle_mesh(fname)
  mesh.compute_vertex_normals()
  mesh.compute_adjacency_list()

	#               x    y   z    yaw  pitch
  camera_pos = [2.8, 0.0, 2.5, 90.0, 60.0]

  for i in range(len(mesh.vertices)):
    
    # N: surface normal
    N = normalize(mesh.vertex_normals[i])

    # compute orthogonal vectors in surface tangent plane
    # note: any two orthogonal vectors on the plane will do. 
    # choose the first one arbitrarily
    # x: surface tangent
    U = normalize(np.random.rand(3))    
    X = normalize(np.cross(N, U))
      
    # y: surface bitangent
    Y = normalize(np.cross(N, X))

    vertex = mesh.vertices[i]
	  #  V: view direction        
    V = np.asarray(camera_pos[:3]) - np.asarray(vertex)	    

    #  L: light direction: same as view direction
    L = V


    metallic = 0.0
    subsurface = 0.0
    specular = 0.5
    roughness = 0.05
    specularTint = 0.5
    anisotropic = 0.5
    sheen = 0.5
    sheenTint = 0.5
    clearcoat = 0.5
    clearcoatGloss = 1.0
    
    baseColor = np.asarray([0.2, 0.5, 0.2])
    #baseColor = np.asarray(mesh.vertex_colors[i])
    render =  5*BRDF(L=L, V=V, N=N, X=X, Y=Y, baseColor=baseColor, metallic=metallic, subsurface=subsurface, specular=specular, roughness=roughness, specularTint=specularTint, anisotropic=anisotropic,sheen=sheen,sheenTint=sheenTint,clearcoat=clearcoat,clearcoatGloss=clearcoatGloss)

    mesh.vertex_colors[i] = render
      
  o3d.visualization.draw_geometries([mesh])

  outfname = "test.ply"  
  o3d.io.write_triangle_mesh(outfname, mesh)
  
  

if __name__ == "__main__":
  main()