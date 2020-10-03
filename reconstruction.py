import sys
from multipledispatch import dispatch
import math
import numpy as np
from scipy.spatial import ConvexHull
import csv2ply
import open3d as o3d
import os

class Point3D:
  def __init__(self, x, y, z, nx=0.0, ny=0.0, nz=0.0, r=0, g=0, b=0, valid=True):
    self.valid = valid
    self.orphan = True
    self.x = x
    self.y = y 
    self.z = z
    self.nx = nx
    self.ny = ny
    self.nz = nz
    self.r = r
    self.g = g
    self.b = b


  def __sub__(self, other):
    return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

  def __add__(self, other):
    return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

  def __div__(self, other):
    return Point3D(self.x/other, self.y/other, self.z/other)


# horizantal_pixels: 1824 * sensor_resolution
# vertical_pixels: 2280 * sensor_resolution
class GridCloud:
  vertical_pixels = 2280
  horizantal_pixels = 1824

  def __init__(self, sensor_resolution):
    self.sensor_resolution = sensor_resolution
    self.gc = []
    self.v_max = int(sensor_resolution*self.vertical_pixels)
    self.h_max = int(sensor_resolution*self.horizantal_pixels)
    self.max_valid_row = -1
    self.max_valid_col = -1

    for i in range(0, self.v_max):
      self.gc.append([])
      for i in range(0, self.h_max):
        self.gc[-1].append(Point3D(0,0,0,0,0,0,0,0,0,False))


  def get(self, v, h):
    return self.gc[v][h]

  # valid bounds may no longer be correct after using this function
  def set(self, v, h, p):
    v_i = v
    h_i = h
    self.gc[v_i][h_i].x = p.x
    self.gc[v_i][h_i].y = p.y
    self.gc[v_i][h_i].z = p.z
    self.gc[v_i][h_i].nx = p.nx
    self.gc[v_i][h_i].ny = p.ny
    self.gc[v_i][h_i].nz = p.nz
    self.gc[v_i][h_i].r = p.r
    self.gc[v_i][h_i].g = p.g
    self.gc[v_i][h_i].b = p.b
    self.gc[v_i][h_i].valid = p.valid

  def computeValidBounds(self):
    self.mm = []
    for i in range(0, self.v_max):
      j = 0
      while j < self.h_max and self.gc[i][j].valid is False:
        j = j + 1
      if j == self.h_max:
        self.mm.append((-1,-1))
      else:
        j_min = j
        j = self.h_max - 1
        while j > j_min and self.gc[i][j].valid is False:
          j = j - 1
        j_max = j
        self.mm.append((j_min,j_max))
    self.max_valid_row = self.maxValidRow()
    self.max_valid_col = self.maxValidCol()

  def maxValidCol(self):
    max_col = -1
    for i in range(0, len(self.mm)):
      if self.mm[i][1] > max_col:
        max_col = self.mm[i][1]
    return max_col

  def maxValidRow(self):
    max_row = -1
    for i in range(0, self.v_max):
      if self.mm[i][0] > -1:
        max_row = i
    return max_row

  def resetOrphanState(self):
    for i in range(0, self.v_max):
      for j in range(0, self.h_max):
        self.gc[i][j].orphan = True

  # write grid cloud as .csv
  def writeAsCSV(self, outfilename):
    f_out = open(outfilename, "w")
    f_out.write("h,v,x,y,z,r,g,b,nx,ny,nz\n")
    for i in range(0, self.v_max):
      for j in range(0, self.h_max):
        h = j
        v = i
        p = self.gc[i][j]
        if p.valid == True:
          f_out.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(h,v,p.x,p.y,p.z,p.nx,p.ny,p.nz,p.r,p.g,p.b))


class Mesh:

  def __init__(self, V, faces):
    self.V = V
    self.faces = faces
    self.T = []

  # collect the triangle indices each vertex is a part of 
  def computeT(self):
    self.T = []
    for f in self.faces:
      self.T.append([])
    i = 0
    for f in self.faces:
      self.T[f[0]].append(i)
      self.T[f[1]].append(i)
      self.T[f[2]].append(i)
      i = i + 1


  def writeAsPLY(self, f_out_name, vertices_only=False):

    V = self.V
    faces = self.faces

    fout = open(f_out_name, "w")
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex {}\n".format(str(len(V))))
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("property float nx\n")
    fout.write("property float ny\n")
    fout.write("property float nz\n")
    fout.write("property uchar red\n")
    fout.write("property uchar green\n")
    fout.write("property uchar blue\n")
    if vertices_only == False:
      fout.write("element face {}\n".format(str(len(faces))))
      fout.write("property list uchar int vertex_indices\n")
    #fout.write("element edge {}\n".format(str(len(faces)*3)))
    #fout.write("property int vertex1\n")
    #fout.write("property int vertex2\n")
    #fout.write("property uchar red\n")
    #fout.write("property uchar green\n")
    #fout.write("property uchar blue\n")
    fout.write("end_header\n")
    offsets = []
    for v in V:
      fout.write("{} {} {} {} {} {} {} {} {}\n".format(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]))
    if vertices_only == False:  
      for face in faces:
        fout.write("3 {} {} {}\n".format(face[0],face[1],face[2]))

      #for face in faces:
      #  fout.write("{} {} {} {} {}\n".format(face[0], face[1], 255, 255, 255))
      #  fout.write("{} {} {} {} {}\n".format(face[0], face[2], 255, 255, 255))
      #  fout.write("{} {} {} {} {}\n".format(face[1], face[2], 255, 255, 255))


  # only writes vertices (ignores faces)
  def writeAsCSV(self, f_out_name):

    fout = open(f_out_name, "w")
    fout.write("h,v,x,y,z,nx,ny,nz,r,g,b\n")
    for p in self.V:
      x = p[0]
      y = p[1]
      z = p[2]
      nx = p[3]
      ny = p[4]
      nz = p[5]
      r = p[6]
      g = p[7]
      b = p[8]
      h = p[9]
      v = p[10]
      fout.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(h,v,x,y,z,nx,ny,nz,r,g,b))

  def copy(self):
    cpy_V = []
    cpy_faces = []
    for v in self.V:
      #cpy_V.append([v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10]])
      cpy_V.append([v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]])
    for face in self.faces:
      cpy_faces.append([face[0],face[1],face[2]])

    return Mesh(cpy_V, cpy_faces)

def readMesh(fname):
  V = []
  faces = []

  omesh = o3d.io.read_triangle_mesh(fname)
  oV = np.asarray(omesh.vertices)
  ofaces = np.asarray(omesh.triangles)
  onormals = np.asarray(omesh.vertex_normals)

  for i in range(0, len(oV)):
    xyz = oV[i,:]
    nxnynz = [0.0, 0.0, 0.0]
    rgb = [128,128,128]
    v = []
    v.append(xyz[0])
    v.append(xyz[1])
    v.append(xyz[2])
    v = v + nxnynz
    v = v + rgb
    V.append(v)
  for i in range(0, len(ofaces)):
    faces.append( ofaces[i,:] )

  return Mesh(V, faces)




def mergeMeshes(mesh1, mesh2):
  merged = mesh1.copy()
  V = merged.V
  faces = merged.faces

  for v in mesh2.V:
    V.append(v)
  for face in mesh2.faces:
    faces.append([face[0]+len(mesh1.V), face[1]+len(mesh1.V), face[2]+len(mesh1.V)])

  return Mesh(V,faces)

# create grid cloud from .csv
def getGridCloud(fname, sensor_resolution):

  gc = GridCloud(sensor_resolution)
  fin = open(fname, "r")
  fin.readline()

  for line in fin:
    l = line.split(",")
    h = int(l[0])
    v = int(l[1])
    x = float(l[2])
    y = float(l[3])
    z = float(l[4])
    nx = float(l[5])
    ny = float(l[6])
    nz = float(l[7])
    r = int(l[8])
    g = int(l[9])
    b = int(l[10])
    gc.set(v, h, Point3D(x,y,z,nx,ny,nz,r,g,b))

  gc.computeValidBounds()
  return gc


def vectorsAngle(a, b):
  n = norm([a.x,a.y,a.z]) * norm([b.x,b.y,b.z])
  if n==0:
    print("norm is 0")
    print("{},{},{}".format(a.x,a.y,a.z))
    print("{},{},{}".format(b.x,b.y,b.z))
    quit()
  ratio = dot(a,b) / n 
  if ratio > 0.9999999:
    ratio = 0.9999999
  elif ratio < -0.9999999:
    ratio = -0.9999999
  return math.acos(ratio)

def overlapping(tri1, tri2, thresh):
  p1 = Point3D(tri1[0][0], tri1[0][1], tri1[0][2])
  p2 = Point3D(tri1[1][0], tri1[1][1], tri1[1][2])
  p3 = Point3D(tri1[2][0], tri1[2][1], tri1[2][2])
  tp1 = Point3D(tri2[0][0], tri2[0][1], tri2[0][2])
  tp2 = Point3D(tri2[1][0], tri2[1][1], tri2[1][2])
  tp3 = Point3D(tri2[2][0], tri2[2][1], tri2[2][2])
  tri = [tp1, tp2, tp3]
  p1_proj = projectPointToTriangle(p1, (tp1,tp2,tp3))
  if d(p1_proj, p1) < thresh and isInTriangle(p1_proj, tri):
    return True
  p2_proj = projectPointToTriangle(p2, (tp1,tp2,tp3))
  if d(p2_proj, p2) < thresh and isInTriangle(p2_proj, tri):
    return True
  p3_proj = projectPointToTriangle(p3, (tp1,tp2,tp3))
  if d(p3_proj, p3) < thresh and isInTriangle(p3_proj, tri):
    return True
  
  return False

def isInTriangle(p, tri):

  # not considered in triangle if p is one of the points of tri
  if (p.x == tri[0].x and p.y == tri[0].y and p.z == tri[0].z) or (p.x == tri[1].x and p.y == tri[1].y and p.z == tri[1].z) or (p.x == tri[2].x and p.y == tri[2].y and p.z == tri[2].z):
    return False
  
  theta1 = (vectorsAngle(p - tri[0], p - tri[1]))
  theta2 = (vectorsAngle(p - tri[0], p - tri[2]))
  theta3 = (vectorsAngle(p - tri[1], p - tri[2]))
  return abs(theta1 + theta2 + theta3 - 2.0*math.pi) < 0.001

def projectPointToTriangle(p, tri):

  #print("warning: projectPointToTriangle may not properly handle color or normals")

  # compute unit normal of triangle plane  
  qr = tri[1] - tri[0]
  qs = tri[2] - tri[0]
  n = cross(qr, qs)
  n = n / norm(n)

  # compute distance from p to triangle plane
  qp = p - tri[0]
  mag = np.dot([qp.x,qp.y,qp.z], n)

  # translate p to plane
  return p - scale(n,mag)

def dot(p1, p2):
  return np.dot([p1.x,p1.y,p1.z], [p2.x,p2.y,p2.z])

def norm(v):
  return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def cross(a, b):
  return np.cross([a.x,a.y,a.z],[b.x,b.y,b.z])

def scale(p, c):
  return Point3D(p[0]*c, p[1]*c, p[2]*c)


# maximum distance in 3d from a point p inside cube of edge length x to maximum distance vertex of cube (w.r.t p)
def cubeBound(x):
  return math.sqrt(  x*x * (1 + math.cos(math.radians(45))*math.cos(math.radians(45))) / (4*math.cos(radians(45)*math.cos(radians(45))) ) )


# euclidean distance for points represented as Point3D
@dispatch(object, object)
def d(p1,p2):
  return d((p1.x,p1.y,p1.z), (p2.x,p2.y,p2.z))

# euclidean distance for points represented as tuples
@dispatch(tuple, tuple)
def d(p1,p2):
  dx = (p1[0]-p2[0])*(p1[0]-p2[0])
  dy = (p1[1]-p2[1])*(p1[1]-p2[1])
  dz = (p1[2]-p2[2])*(p1[2]-p2[2])
  return math.sqrt(dx + dy + dz)

def pindex(v,h,cols):
  return v*cols + h

# max edge length for triangle represented as vertex coordinate tuple
@dispatch(list)
def maxEdgeLen(v):  
  return max(d(v[0],v[1]), d(v[0],v[2]), d(v[1],v[2]))

# max edge length for triangle represented as indices into vertex coordinate tuple
@dispatch(list, list)
def maxEdgeLen(tri_is, vs):
  return maxEdgeLen([vs[tri_is[0]], vs[tri_is[1]], vs[tri_is[2]]])


# Perform surface reconstruction given input GridCloud
# note: marks any vertices that do not form a face as invalid
def localSurfaceReconstruction(gc, d_thresh):
  
  gc.resetOrphanState()
  cols = gc.maxValidCol() + 1
  rows = gc.maxValidRow() + 1

  M = gc.gc
  P = {}
  V = []
  faces = []
  edges = []
  orphaned = []
  invalids = []
  k = 0

  for i in range(0, rows):
    for j in range(0, cols):   
      p = M[i][j]
      #V.append([p.x,p.y,p.z,p.r,p.g,p.b,p.nx,p.ny,p.nz,j,i])
      V.append([p.x,p.y,p.z,p.nx,p.ny,p.nz,p.r,p.g,p.b,j,i])
      P[pindex(i,j,cols)] = k
      orphaned.append(False)
      invalids.append(not p.valid)
      k = k + 1

  for i in range(0, rows):
    for j in range(0, cols):

      if M[i][j].valid == False:
        continue

      p1 = M[        i       ][         j         ]
      p2 = M[ (i+1) % (rows) ][         j         ]
      p3 = M[        i       ][  (j+1) % (cols)   ]
      p4 = M[ (i+1) % (rows) ][  (j+1) % (cols)   ]

      ij_points = []
      v_indices = []

      ij_points.append(p1)
      v_indices.append(P[pindex(i,j,cols)])
      v_indices.append(P[pindex( (i+1) % (rows),      j,          (cols))])
      v_indices.append(P[pindex(      i,         (j+1) % (cols),  (cols))])
      v_indices.append(P[pindex((i+1) % (rows), (j+1) % (cols),  (cols))])

      if p2.valid == True:
        ij_points.append(p2)
      if p3.valid == True:
        ij_points.append(p3)
      if p4.valid == True:
        ij_points.append(p4)



      count = len(ij_points)
      if count == 4: # possibly make 2 triangles
        tri1 = [0, 2, 1]
        tri2 = [1, 2, 3]
        tri3 = [0, 3, 1]
        tri4 = [0, 2, 3]

        tri1_d = maxEdgeLen(tri1, ij_points)
        tri2_d = maxEdgeLen(tri2, ij_points)
        tri3_d = maxEdgeLen(tri3, ij_points)
        tri4_d = maxEdgeLen(tri4, ij_points)

        # note: if can make 2 triangles, consider choosing the way to do it that
        # minimizes cross-length

        if tri1_d < d_thresh and tri2_d < d_thresh:
          faces.append([v_indices[tri1[0]], v_indices[tri1[1]], v_indices[tri1[2]]])
          faces.append([v_indices[tri2[0]], v_indices[tri2[1]], v_indices[tri2[2]]])
          p1.orphan = False
          p2.orphan = False
          p3.orphan = False
          p4.orphan = False
        elif tri3_d < d_thresh and tri4_d < d_thresh:
          faces.append([v_indices[tri3[0]], v_indices[tri3[1]], v_indices[tri3[2]]])
          faces.append([v_indices[tri4[0]], v_indices[tri4[1]], v_indices[tri4[2]]])
          p1.orphan = False
          p2.orphan = False
          p3.orphan = False
          p4.orphan = False
        elif tri1_d < d_thresh:
          faces.append([v_indices[tri1[0]], v_indices[tri1[1]], v_indices[tri1[2]]])
          p1.orphan = False
          p2.orphan = False
          p3.orphan = False
        elif tri2_d < d_thresh:
          faces.append([v_indices[tri2[0]], v_indices[tri2[1]], v_indices[tri2[2]]])
          p2.orphan = False
          p3.orphan = False
          p4.orphan = False
        elif tri3_d < d_thresh:
          faces.append([v_indices[tri3[0]], v_indices[tri3[1]], v_indices[tri3[2]]])
          p1.orphan = False
          p2.orphan = False
          p4.orphan = False
        elif tri4_d < d_thresh:
          faces.append([v_indices[tri4[0]], v_indices[tri4[1]], v_indices[tri4[2]]])
          p1.orphan = False
          p3.orphan = False
          p4.orphan = False

          
      elif count == 3: # possibly make 1 triangle
        z = maxEdgeLen(ij_points)
        if z < d_thresh:
          # get indices of vertices in V and add to faces
          # make sure in clockwise order... 0,2,3,1
          # make one of the following:
          # tri1 = [0, 2, 1]
          # tri3 = [0, 3, 1]
          # tri4 = [0, 2, 3]
          if p3.valid == True and p2.valid == True:
            faces.append([v_indices[0], v_indices[2], v_indices[1]])
          elif p4.valid == True and p2.valid == True:
            faces.append([v_indices[0], v_indices[3], v_indices[1]])
          elif p3.valid == True and p4.valid == True:
            faces.append([v_indices[0], v_indices[2], v_indices[3]])
          else:
            print("error in local meshing?")
            quit()

          ij_points[0].orphan = False
          ij_points[1].orphan = False
          ij_points[2].orphan = False

      # if p1 hasn't found a family by this point, he or she never will
      if p1.orphan == True:
        p1.valid = False
        #print(V[P[pindex(i,j,max_col)]][0])
        orphaned[P[pindex(i,j,cols)]] = True
  

  V_pruned = []
  offsets = [0]
  k = 0
  for v in V:
    if orphaned[k] == True or invalids[k] == True:
      offsets.append(offsets[-1] + 1)
    else:
      V_pruned.append(v)
      offsets.append(offsets[-1])
    k = k + 1
    
  n_pruned = offsets[-1]
  offsets.pop(0)

  # update vertex indices for faces to account for removed vertices
  for face in faces:
    face[0] = face[0] - offsets[face[0]]
    face[1] = face[1] - offsets[face[1]]
    face[2] = face[2] - offsets[face[2]]


  # update input gridcloud's valid bounds if any vertices were removed
  if n_pruned != 0:
    gc.computeValidBounds()
  gc.resetOrphanState()

  return Mesh(V_pruned, faces)



def localRemeshing(mesh, thresh):
  

  #mesh.computeT()

  tri_centers = []
  for face in mesh.faces:
    center_x = (mesh.V[face[0]][0] + mesh.V[face[1]][0] + mesh.V[face[2]][0])/3.0
    center_y = (mesh.V[face[0]][1] + mesh.V[face[1]][1] + mesh.V[face[2]][1])/3.0
    center_z = (mesh.V[face[0]][2] + mesh.V[face[1]][2] + mesh.V[face[2]][2])/3.0
    tri_centers.append([center_x, center_y, center_z])

  center_pc = o3d.geometry.PointCloud()
  center_pc.points = o3d.utility.Vector3dVector(np.array(tri_centers))
  center_pc_tree = o3d.geometry.KDTreeFlann(center_pc)


  # color all vertices grey
  #for idx in range(0, len(mesh.V)):
    #mesh.V[idx][3:6] = [0.5,0.5,0.5]
    #mesh.V[idx][6:9] = [128,128,128]

  for i in range(0, len(mesh.faces)):
    if i%100 == 0:
      print("{} of {}".format(i, len(mesh.faces)))

    face_i = mesh.faces[i]
    tri_i = [mesh.V[face_i[0]][:3], mesh.V[face_i[1]][:3], mesh.V[face_i[2]][:3]]
    #mesh.V[face_i[0]][6:9] = [0,0,255]
    #mesh.V[face_i[1]][6:9] = [0,0,255]
    #mesh.V[face_i[2]][6:9] = [0,0,255]

    # examine all triangle centers within radius thresh of v 
    [k, idx, _] = center_pc_tree.search_radius_vector_3d(center_pc.points[i], thresh)
    idx = idx[1:]


    for j in idx:
      face_j = mesh.faces[j]
      tri_j = [mesh.V[face_j[0]][:3], mesh.V[face_j[1]][:3], mesh.V[face_j[2]][:3]]

      if overlapping(tri_i, tri_j, thresh) == True:

        # lookup vertex indices of the two triangles
        v_i = [mesh.faces[i][0], mesh.faces[i][1], mesh.faces[i][2]]
        v_j = [mesh.faces[j][0], mesh.faces[j][1], mesh.faces[j][2]]

        # remove triangle from second mesh
        mesh.faces.pop(j)
        
        # color vertices of intersecting triangles red
        #for idx in v_i + v_j:
        #  mesh.V[idx][6:9] = [255,0,0]

  return mesh


def mergeRawPointClouds(files, fdir, dataset):
  V = []
  v_off = 0
  for f in files:
    f_in = open("{}/{}_{}.ply".format(fdir, dataset, f),"r")
    vertices = 0
    for i in range(0, 13):
      line = f_in.readline()
      if i==2:
        vertices = int(line.split(" ")[-1])
    # read vertices
    for i in range(0, vertices):
      line = f_in.readline()
      V.append(line)
    v_off = v_off + vertices

  # write merged ply
  f_out = open("{}/{}_merged.ply".format(fdir, dataset), "w")
  f_out.write("ply\n")
  f_out.write("format ascii 1.0\n")
  f_out.write("element vertex {}\n".format(v_off))
  f_out.write("property float x\n")
  f_out.write("property float y\n")
  f_out.write("property float z\n")
  f_out.write("property float nx\n")
  f_out.write("property float ny\n")
  f_out.write("property float nz\n")
  f_out.write("property uchar red\n")
  f_out.write("property uchar green\n")
  f_out.write("property uchar blue\n")

  f_out.write("end_header\n")
  for v in V:
    f_out.write("{}".format(v))


def mergeRawMeshes(files, fdir, dataset, resolution, thresh):
  meshes = []

  for f in files:
    f_in_name = "simulated_scanner_outputs/{}/{}_{}.csv".format(dataset,dataset,f)
    gc = getGridCloud(f_in_name, resolution)
    mesh = localSurfaceReconstruction(gc, thresh)
    mesh.writeAsPLY("{}/{}_raw_mesh_{}.ply".format(fdir, dataset,f))
    meshes.append(mesh)

  
  merged = meshes[0].copy()
  for mesh in meshes[1:]:
    merged = mergeMeshes(merged,mesh)


  merged.writeAsPLY("{}/{}_raw_meshes_merged.ply".format(fdir, dataset))




def reconstruction(files, resolution, thresh, voxel_size, use_im_remesh):

  meshes = []

  print("Performing local reconstruction for each point cloud...")
  for f in files:
    f_in_name = "{}.csv".format(f)
    gc = getGridCloud(f_in_name, resolution)
    mesh = localSurfaceReconstruction(gc, thresh)
    mesh.writeAsPLY("{}_mesh.ply".format(f))


  print("Reconstructing without overlapping faces with MeshLab VCG algorithm...")

  f_in = open("meshlab_script_template.mlx", "r")
  f_out = open("meshlab_script.mlx", "w")
  for line in f_in:
    f_out.write(line.replace("voxel_size", str(voxel_size)))

  f_out.close()

  command = "../meshlab/distrib/meshlabserver -s meshlab_script.mlx -i"
  for f in files:
    command = command + " {}_mesh.ply".format(f)
  command = command + " -o reconstructed.ply"
  os.system(command)


  # clean up
  os.system("rm meshlab_script.mlx")


  mesh = readMesh("reconstructed.ply")

  if use_im_remesh == True:
    mesh.writeAsPLY("reconstructed.ply")
    print("Remeshing with Instant Meshes...")
    target_face_count = int(len(mesh.faces)/10)
    command = "../instant-meshes/InstantMeshes reconstructed.ply -f {} -d -S 0 -r 6 -p 6 -o remesh.ply".format(target_face_count)
    os.system(command)
    mesh = readMesh("remesh.ply")

    # clean up 
    os.system("rm reconstructed.ply")
    os.system("rm remesh.ply")


  return mesh



def doReconstruction(fname, n_files, resolution, max_edge_len, voxel_size, use_im_remesh):

  files = []
  for i in range(n_files):
    files.append("{}_{}".format(fname, i))
  return reconstruction(files=files, resolution=resolution, thresh=max_edge_len, voxel_size=voxel_size, use_im_remesh=use_im_remesh)





# Usage:
# This script makes use of MeshLab and, optionally, Instant Meshes. The following directory
# structure must be used to make them accessible:
# -- This script is located in research/
# -- Complete MeshLab installation (built from source) must be located one directory above
#    so that meshlabserver is located in ../meshlab/distrib/
# -- Similarly, InstantMeshes executable must be located in ../instant-meshes/
#
# Parameters are as follows:
# --resolution: same as resolution used in optics.py. Use 1.0 for full resolution
# --fname: each .csv to be included in reconstruction needs to be accessible from this
#          directory by <fname>_<x>.csv, with <x> ranging from 0 to the total number of
#          files
# --n_files: total number of files to be included in reconstruction
# --max_edge_len: maximum edge length of triangles constructed in the local phase of
#                 reconstruction
# --voxel_size: size of voxel sides used in VCG. Smaller values will result in higher accuracy
#               but reconstruction will take more space and time and will produce a larger model
# --use_im_remesh: whether or not Instant Meshes will be used as a final step to remesh
#                  the mesh produced by VCG. Smooths the result and reduces the number of
#                  faces by 90%.
def main():

  resolution = 0.1
  #fname = "simulated_scanner_outputs/brownchair_{}/brownchair_{}".format(resolution, resolution)
  fname = "simulated_scanner_outputs/chalice_{}/chalice_{}".format(resolution, resolution)
  outfname = "{}_reconstructed.ply".format(fname)
  n_files = 12
  max_edge_len = 0.03
  voxel_size = 0.001
  use_im_remesh = True 

  print("Reconstruction initiated.")
  mesh = doReconstruction(fname, n_files, resolution, max_edge_len, voxel_size, use_im_remesh)
  print("Reconstruction complete. Writing result to {}".format(outfname))
  mesh.writeAsPLY(outfname)

if __name__ == "__main__":
  main()


