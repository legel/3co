import math
import numpy as np
import open3d as o3d
import disney_brdf
import path_planning
import os
import json


def main():

  path = path_planning.get_balustervase_path()
  dataset_num = 13
  ooi = "balustervase_0.2"
  if not os.path.isdir("models/{}/dataset_{}".format(ooi, dataset_num)):
    os.system("mkdir models/{}/dataset_{}".format(ooi, dataset_num))
  info = open("models/{}/dataset_{}/brdf_params.json".format(ooi, dataset_num), "w")
  
  
  brdf_params = {}
  brdf_params['red'] = 0.6
  brdf_params['green'] = 0.0
  brdf_params['blue'] = 0.3
  brdf_params['metallic'] = 0.0  
  brdf_params['subsurface'] = 0.0
  brdf_params['specular'] = 0.5
  brdf_params['roughness'] = 0.2
  brdf_params['specularTint'] = 0.0
  brdf_params['anisotropic'] = 0.0
  brdf_params['sheen'] = 0.0
  brdf_params['sheenTint'] = 0.0
  brdf_params['clearcoat'] = 0.0
  brdf_params['clearcoatGloss'] = 1.0

  js = json.dumps(brdf_params)
  info.write(js)
  info.close()


  path_dict = {}
  t = 0
  for camera_pos in path:
    
    path_dict[t] = (camera_pos[0], camera_pos[1], camera_pos[2], camera_pos[3], camera_pos[4], 0.0)    
    fname = "models/{}/{}_{}_mesh.ply".format(ooi,ooi,t)  
    mesh = o3d.io.read_triangle_mesh(fname)
    mesh.compute_vertex_normals()
    mesh.remove_triangles_by_index(range(50000, len(mesh.triangles)))
    mesh.remove_triangles_by_index(range(0,30000))
    mesh.remove_unreferenced_vertices()

    mesh = disney_brdf.render_disney_brdf_on_mesh(mesh,camera_pos[:3],brdf_params)
    outfname = "models/{}/dataset_{}/{}_{}_mesh_rendered.ply".format(ooi,dataset_num,ooi,t)
    o3d.io.write_triangle_mesh(outfname, mesh)
    t = t + 1

  info = open("models/{}/dataset_{}/camera_pos.json".format(ooi, dataset_num), "w")
  js = json.dumps(path_dict)
  info.write(js)
  info.close()
  

if __name__ == "__main__":
  main()