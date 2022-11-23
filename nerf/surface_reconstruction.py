import numpy as np
import math
import open3d as o3d

from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
from pytorch3d.io import IO
import torch

import imageio
from PIL import Image
import numpy as np
import random, math
import sys, os, shutil, copy, glob, json
import time, datetime
import argparse
import wandb
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.join(sys.path[0], '../..'))
from utils.pos_enc import encode_position
from utils.volume_op import volume_sampling, volume_rendering
from utils.lie_group_helper import convert3x4_4x4
from utils.training_utils import PolynomialDecayLearningRate, heatmap_to_pseudo_color, set_randomness, save_checkpoint
from models.intrinsics import CameraIntrinsicsModel
from models.poses import CameraPoseModel
from models.nerf_models import NeRFDensity, NeRFColor

import mcubes

from learn import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import    


def load_meshes(mesh_dir, base_fname, mesh_indices, format='glb'):

    meshes = []
    for i in mesh_indices:
        fname = "{}/{}_{}_luminosity_filtered_only.{}".format(mesh_dir, base_fname, i, format)
        print("Loading {}".format(fname))
        mesh =  o3d.io.read_triangle_mesh(fname)               
        meshes.append(mesh)
    return meshes

def set_bounds(vertices, vox_size):
    global min_x, min_y, min_z, max_x, max_y, max_z
    padding = 5 * vox_size
    
    all_vertices = []
    for view_vertices in vertices:
      all_vertices += np.asarray(view_vertices).tolist()

    all_vertices = torch.tensor(all_vertices)
             
    min_x = torch.min(all_vertices[:,0]) - padding    
    min_y = torch.min(all_vertices[:,1]) - padding
    min_z = torch.min(all_vertices[:,2]) - padding
    max_x = torch.max(all_vertices[:,0]) + padding
    max_y = torch.max(all_vertices[:,1]) + padding
    max_z = torch.max(all_vertices[:,2]) + padding

    print("{}, {}, {}, {}, {}, {}".format(min_x, min_y, min_z, max_x, max_y, max_z))

def construct_volume(vox_size):
    
    volume_x = torch.linspace(min_x, max_x, 1 + math.ceil( (max_x - min_x) / vox_size))
    volume_y = torch.linspace(min_y, max_y, 1 + math.ceil( (max_y - min_y) / vox_size))
    volume_z = torch.linspace(min_z, max_z, 1 + math.ceil( (max_z - min_z) / vox_size))    
    grid_x, grid_y, grid_z = torch.meshgrid(volume_x, volume_y, volume_z, indexing='ij')
    volume = torch.cat([grid_x.unsqueeze(3), grid_y.unsqueeze(3), grid_z.unsqueeze(3)], dim=3)
    return volume

def distance_to_mesh(p, face_vertices, dists):
            
    p_expand = p.unsqueeze(0).expand(face_vertices.size()[0], p.size()[0], 3)

    # compute distance from each voxel to closest face of mesh
    face_centers = torch.mean(face_vertices, dim=1)
    face_centers_expand = face_centers.unsqueeze(1).expand(p_expand.size()[0], p_expand.size()[1], 3)
    distance_to_face_centers = torch.sqrt( torch.sum((face_centers_expand-p_expand)**2,dim=2) )        
    min_distance_to_face_centers = torch.min(distance_to_face_centers, dim=0)[0]

    return torch.min( torch.cat([min_distance_to_face_centers.unsqueeze(1), dists.unsqueeze(1)], dim=1), dim=1)[0]        
        

def vcg(meshes, poses, pixel_directions_world, out_f_name):
    
    # define voxel grid    
    H = 480
    W = 640    
    near = 0.1
    far = 1.0        
    vox_size = 0.001
    save_sdf_frequency = 1
    all_vertices = [mesh.vertices for mesh in meshes]
    set_bounds(all_vertices, vox_size)
    volume = construct_volume(vox_size).to(torch.device('cuda:0'))
    sdf = torch.zeros(volume.size()[0], volume.size()[0], volume.size()[0]).to(torch.device('cuda:0'))
    sdf = sdf + float('Inf') # No fear
    n_processed_meshes = 0
    
    for cam_index in range(poses.size()[0]):
        
        # Select voxel indices that are within camera's line of sight
        corner_view_xy_directions = torch.stack([
          pixel_directions_world[cam_index, 0, 0, :2],
          pixel_directions_world[cam_index, H-1, 0, :2],
          pixel_directions_world[cam_index, 0, W-1, :2],
          pixel_directions_world[cam_index, H-1, W-1, :2]
        ])
        
        min_view_direction_x = torch.min(corner_view_xy_directions[:, 0])
        max_view_direction_x = torch.max(corner_view_xy_directions[:, 0])
        min_view_direction_y = torch.min(corner_view_xy_directions[:, 1])
        max_view_direction_y = torch.max(corner_view_xy_directions[:, 1])

        view_xyz = poses[cam_index][:3, 3]        
                
        x_view_bounds = torch.tensor([
          min_view_direction_x * far + view_xyz[0], 
          max_view_direction_x * far + view_xyz[0],
          min_view_direction_x * near + view_xyz[0], 
          max_view_direction_x * near + view_xyz[0],
        ])
        y_view_bounds = torch.tensor([
          min_view_direction_y * far + view_xyz[1], 
          max_view_direction_y * far + view_xyz[1],
          min_view_direction_y * near + view_xyz[1], 
          max_view_direction_y * near + view_xyz[1],
        ])
        x_lower = x_view_bounds.min()
        y_lower = y_view_bounds.min()
        x_upper = x_view_bounds.max()
        y_upper = y_view_bounds.max()        
        
        voxel_is_viewable_x = torch.logical_and(volume[:,:,:,0] >= x_lower, volume[:,:,:,0] <= x_upper)
        voxel_is_viewable_y = torch.logical_and(volume[:,:,:,1] >= y_lower, volume[:,:,:,1] <= y_upper)
        voxel_in_near_far_bounds = torch.logical_and ( 
          torch.sqrt(torch.sum((volume - view_xyz)**2, dim=3)) > near,
          torch.sqrt(torch.sum((volume - view_xyz)**2, dim=3)) < far,
        )

        voxel_is_viewable = torch.logical_and(voxel_is_viewable_x, voxel_is_viewable_y)
        voxel_is_viewable = torch.logical_and(voxel_is_viewable, voxel_in_near_far_bounds)        
        viewable_voxel_indices = torch.argwhere(voxel_is_viewable)           
        view_volume = volume[viewable_voxel_indices[:,0], viewable_voxel_indices[:,1], viewable_voxel_indices[:,2]]                

        # Unpack numpy data from open3D and convert to tensor
        # Angry note: very important to do it as done below, or else it can create a memory leak for some reason!
        faces = np.asarray(meshes[cam_index].triangles)
        vertices = np.asarray(meshes[cam_index].vertices)
        face_vertices = torch.tensor(vertices[faces]).float().to(torch.device('cuda:0'))          
        print("\nmesh {} out of {} ({} faces)".format(cam_index,  poses.size()[0], faces.size))
                                
        # Split data into batches based on estimate of GPU memory requirements per batch
        total_bytes_needed = (
          view_volume.size()[0] * face_vertices.size()[0] * 1 +
          view_volume.size()[0] * 3 +
          face_vertices.size()[0] * 3
        ) * 8
        total_bytes_needed *= 2 # overestimate by a bit
             
        bytes_per_batch = 4000000000 # 4gb
        n_batches = total_bytes_needed // bytes_per_batch + 1
        batch_size = view_volume.size()[0] // n_batches
                        
        print('batch size: {}'.format(batch_size))
                
        view_volume_batches = view_volume.split(batch_size)
        viewable_voxel_indices_batches = viewable_voxel_indices.split(batch_size)

        batch_number = 0
        for view_volume_batch, viewable_voxel_indices_batch in zip(view_volume_batches, viewable_voxel_indices_batches):          
          #print("batch {} out of {}".format(batch_number, view_volume.size()[0] // batch_size))
          
          dists = sdf[viewable_voxel_indices_batch[:,0], viewable_voxel_indices_batch[:,1], viewable_voxel_indices_batch[:,2]]
          dists = distance_to_mesh(view_volume_batch, face_vertices, dists)          
          sdf[viewable_voxel_indices_batch[:,0], viewable_voxel_indices_batch[:,1], viewable_voxel_indices_batch[:,2]] = dists
          batch_number += 1

        n_processed_meshes += 1

        if (n_processed_meshes % save_sdf_frequency == 0 or cam_index == poses.size()[0] - 1):
            print("Outputting sdf to {}".format(out_f_name))
            sdf = sdf.cpu()
            valid_sdf_indices = torch.argwhere(torch.logical_and(sdf != float('Inf'), sdf != 0))

            with open(out_f_name, "wb") as f:
              result = np.transpose(np.asarray([
                valid_sdf_indices[:,0].numpy(), 
                valid_sdf_indices[:,1].numpy(), 
                valid_sdf_indices[:,2].numpy(), 
                sdf[valid_sdf_indices[:,0].numpy(), valid_sdf_indices[:,1].numpy(), valid_sdf_indices[:,2].numpy()].numpy()
              ]))              
              print(result.shape)
              np.save(f, result)

            sdf = sdf.to(torch.device('cuda:0'))
        
    return sdf
                                          
def print_memory_usage():
  t = torch.cuda.get_device_properties(0).total_memory
  r = torch.cuda.memory_reserved(0)
  a = torch.cuda.memory_allocated(0)
  f = r-a
  
  print('total memory: {}'.format(t/1000000))
  print("reserved memory: {}".format(r/1000000))
  print("allocated memory: {}".format(a/1000000))
  print("reserved free memory: {}".format(f/1000000))
  print("__________________________________")


if __name__ == '__main__':
    
    with torch.no_grad():
        scene = SceneModel(args=parse_args(), experiment_args='test')

        mesh_indices = range(240)[::4]
        #mesh_indices = range(2)
        mesh_dir = './data/dragon_scale/hyperparam_experiments/pretrained_with_entropy_loss_200k/amazing_pointclouds/depth_view_pointcloud/meshes'
        meshes = load_meshes(mesh_dir=mesh_dir, base_fname='dragon_scale', mesh_indices = mesh_indices, format='glb')                    
        
        poses = scene.models["pose"](0)[mesh_indices]
        pixel_directions = scene.pixel_directions[mesh_indices]
        
        camera_world_rotation = poses[:, :3, :3].view(poses.size()[0], 1, 1, 3, 3) # (M, 1, 1, 3, 3)
        pixel_directions = pixel_directions.unsqueeze(4) # (M, H, W, 3, 1)       
        
        pixel_world_directions = torch.matmul(camera_world_rotation, pixel_directions).squeeze(4).squeeze(0)        
        pixel_world_directions = torch.nn.functional.normalize(pixel_world_directions, p=2, dim=3)  # (N_pixels, 3)        
        
        del scene        
        del camera_world_rotation
        del pixel_directions
        torch.cuda.empty_cache()

        out_f_name = 'sdf_120meshes_vox001.npy'        
        sdf = vcg(meshes, poses, pixel_world_directions, out_f_name)




    