import torch
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix, matrix_to_euler_angles, matrix_to_quaternion, axis_angle_to_quaternion, quaternion_multiply, matrix_to_axis_angle, axis_angle_to_matrix, euler_angles_to_matrix
from scipy.spatial.transform import Rotation
from torchsummary import summary
import cv2
import open3d as o3d
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

from learn import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import    


def rotate_pose_in_camera_space(scene, pose, dx, dy, dz):
    camera_rotation_matrix = pose[:3,:3]
    
    Rx = torch.tensor([
        [1.0,    0.0,      0.0       ],
        [0,   np.cos(dx), -np.sin(dx)],
        [0,   np.sin(dx),  np.cos(dx) ]
    ]).float()

    Ry = torch.tensor([
        [np.cos(dy),   0.0,   np.sin(dy)],
        [0,            1.0,   0.0       ],
        [-np.sin(dy),  0.0,   np.cos(dy)]
    ]).float()

    Rz = torch.tensor([
        [np.cos(dz), -np.sin(dz),   0.0 ],
        [np.sin(dz), np.cos(dz),    0.0 ],
        [0.0,          0.0,         1.0]
    ]).float()

    R = Rz @ Ry @ Rx
    new_camera_rotation_matrix = R @ camera_rotation_matrix

    new_pose = torch.zeros((4,4))
    new_pose[:3,:3] = new_camera_rotation_matrix
    new_pose[:3,3] = pose[:3,3]
    new_pose[3,3] = 1.0    
    return new_pose        


def translate_pose_in_global_space(scene, pose, d_x, d_y, d_z):            
    translation = torch.FloatTensor([d_x,d_y,d_z])

    new_pose = torch.clone(pose)
    new_pose[:3,:3] = pose[:3,:3]
    new_pose[:3,3] = pose[:3,3] + translation
    new_pose[3,3] = 1.0

    return new_pose

def translate_pose_in_camera_space(scene, pose, d_x, d_y, d_z):        
    camera_rotation_matrix = pose[:3,:3]
    translation = camera_rotation_matrix[:3,:3] @ torch.FloatTensor([d_x,d_y,d_z])
    
    new_pose = torch.clone(pose)
    new_pose[:3,:3] = pose[:3,:3]
    new_pose[:3,3] = pose[:3,3] + translation
    new_pose[3,3] = 1.0

    return new_pose    

def generate_spin_poses(scene, number_of_poses):

    object_xyzs = scene.xyz[torch.where(scene.rgbd[:, 3] < 0.5)]
    center_x = torch.mean(object_xyzs[:, 0:1])
    center_y = torch.mean(object_xyzs[:, 1:2])
    center_z = torch.mean(object_xyzs[:, 2:3])
                
    p_center = torch.tensor([center_x, center_y, center_z])
    #p_center = torch.tensor([0.0044, -0.2409, -0.2728])
    
    p_to = p_center # point to look at
    
    dx = 0.015
    dy = 0.35
    dz = -0.15
        
    p_from = p_to + torch.tensor([dx, dy, dz])
    v_forward = torch.nn.functional.normalize(p_from - p_to, dim=0, p=2).float()
    v_arbitrary = torch.tensor([0.0, 1.0, 0.0])
    v_right = torch.cross(v_arbitrary, v_forward, dim=0)
    v_right = torch.nn.functional.normalize(v_right, dim=0, p=2)
    v_up = torch.cross(v_forward, v_right, dim=0)
    v_up = torch.nn.functional.normalize(v_up, dim=0, p=2)
    
    initial_pose = torch.zeros(4,4)    
    initial_pose[0, :3] = v_right
    initial_pose[1, :3] = v_up
    initial_pose[2, :3] = v_forward
    initial_pose[3,  3] = 1.0
    initial_pose[:3, 3] = p_from          

    focal_length = scene.models['focal'](0)[0]
    scene.compute_ray_direction_in_camera_coordinates(focal_length.unsqueeze(0))
    pixel_directions = scene.pixel_directions[0].to(torch.device('cuda:0'))    

    #initial_pose = scene.models['pose'](0)[0].cpu()
    next_cam_pose = torch.clone(initial_pose)

    xyzs = []        
    new_pose = torch.clone(next_cam_pose)  
    poses = [new_pose]
    spin_radius = torch.abs(p_from[2] - p_to[2])
        
    for i in range(0, number_of_poses):
                        
        x = next_cam_pose[0,3]
        y = next_cam_pose[2,3]

        # convert to polar coordinates with center_pixel_xyz as origin     
        theta = torch.atan2(torch.FloatTensor([y - p_to[2]]),torch.FloatTensor([x - p_to[0]]))                       

        # rotate
        theta = theta + 2.0*np.pi/number_of_poses

        # convert back to cartesian coordinates
        xp = spin_radius * math.cos(theta) + p_to[0]
        yp = spin_radius * math.sin(theta) + p_to[2]
                

        # translate in global coordinates
        next_cam_pose = translate_pose_in_global_space(scene, next_cam_pose, (xp - x), 0.0, (yp - y))        

        # rotate in camera coordinates
        next_cam_pose = rotate_pose_in_camera_space(scene, next_cam_pose, 0.0, -1.0 * 2.0 * np.pi / number_of_poses, 0.0)                                
        xyzs.append(next_cam_pose[:,3])          
        new_pose = torch.clone(next_cam_pose)        
        poses.append(new_pose)         
    
    xyzs = torch.stack(xyzs, dim=0)

    # visualize path
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xs = xyzs[:, 0]
    ys = xyzs[:, 1]
    zs = xyzs[:, 2]
    ax.scatter(xs, zs, ys, marker='o')
    ax.scatter(xs[0], zs[0], ys[0], marker='o', c='red')
    ax.scatter(p_to[0], p_to[2], p_to[1], marker='o', c='green')
    plt.show()

    poses = torch.stack(poses, 0)    
    poses = poses.to(scene.device)    

    return poses


def imgs_to_video(video_dir, n_poses):
    cimgs = []
    dimgs = []
    for i in range(0, n_poses):
        fname1 = '{}/color_video_images/color_{}.png'.format(video_dir, i)
        #fname2 = '{}/depth_video_images/depth_{}.png'.format(video_dir, i)
        cimg = Image.open(fname1)
        #dimg = Image.open(fname2)
        cimgs.append(cimg)
        #dimgs.append(dimg)
    imageio.mimwrite('{}/color.mp4'.format(video_dir), cimgs, fps=60, quality=9)        
    #imageio.mimwrite('{}/depth.mp4'.format(video_dir), dimgs, fps=60, quality=9)        
    

def create_spin_video_images(scene, number_of_poses, video_dir):

    print("generating spin poses")
    poses = generate_spin_poses(scene, number_of_poses)
    render_poses(scene, poses, video_dir)


def render_poses(scene, poses, video_dir):

    color_out_dir = Path("{}/color_video_images/".format(experiment_dir))
    color_out_dir.mkdir(parents=True, exist_ok=True)
    depth_out_dir = Path("{}/depth_video_images/".format(experiment_dir))
    depth_out_dir.mkdir(parents=True, exist_ok=True)   
    color_images = []
    depth_images = []
    focal_length = scene.models["focal"](0)
        
    print('using focal length {}'.format(focal_length[0]))
    scene.compute_ray_direction_in_camera_coordinates(focal_length)

    for i,pose in enumerate(poses):
        index = i
        print('rendering pose {}'.format(i))
        render_result = scene.basic_render(pose, scene.pixel_directions[0], focal_length[0])
        render_result = render_on_background(scene, render_result['rendered_image_fine'], render_result['rendered_depth_fine'], pose, scene.pixel_directions[0])
        color_out_file_name = os.path.join(color_out_dir, "color_{}.png".format(index))                
        depth_out_file_name = os.path.join(depth_out_dir, "depth_{}.png".format(index))                        
        
        scene.save_render_as_png(render_result, color_out_file_name, depth_out_file_name)

        rendered_color_for_file = (render_result['rendered_image_fine'].cpu().numpy() * 255).astype(np.uint8)    
        rendered_depth_data = render_result['rendered_depth_fine'].cpu().numpy()         

        rendered_depth_for_file = heatmap_to_pseudo_color(rendered_depth_data)
        rendered_depth_for_file = (rendered_depth_for_file * 255).astype(np.uint8)     
        color_images.append(rendered_color_for_file)   
        depth_images.append(rendered_depth_for_file)



def render_on_background(scene, rgb, depth, pose, pixel_directions):

    camera_world_position = pose[:3, 3].view(1, 1, 1, 3)     # (1, 1, 1, 3)
    camera_world_rotation = pose[:3, :3].view(1, 1, 1, 3, 3) # (1, 1, 1, 3, 3)
    pixel_directions = pixel_directions.reshape(scene.H, scene.W, 3).unsqueeze(3).to(device=scene.device)
    pixel_directions = torch.nn.functional.normalize(pixel_directions, p=2, dim=2)
    rgb_img = rgb.reshape(scene.H, scene.W, 3).to(device=scene.device)
    depth_img = depth.reshape(scene.H, scene.W).to(device=scene.device)    
    xyz_coordinates = scene.derive_xyz_coordinates(camera_world_position, camera_world_rotation, pixel_directions, depth_img)    
    # dragon_scale
    center = torch.tensor([0.0044, -0.2409, -0.2728]).to(scene.device)
    corner = torch.tensor([-0.1450, -0.3655, -0.4069]).to(scene.device)
    # elastica_burgundy
    #center = torch.tensor([0.0058, -0.4112, -0.2170]).to(scene.device)    


    #mins = torch.tensor([-0.1934, -0.6670+0.25, -0.3852]).to(scene.device)*1.5
    #maxs = torch.tensor([0.2049, -0.1554+0.1, -0.0488+0.1]).to(scene.device)*1.5
    #bounding_box_condition = torch.logical_and(xyz_coordinates > mins, xyz_coordinates < maxs).float()
    #bounding_box_condition = torch.sum(bounding_box_condition, dim=2)    
    #bounding_box_condition_indices = torch.where(bounding_box_condition != 3.0)
    
    #print(bounding_box_condition_indices.size())
    
    bounding_sphere_radius = torch.sqrt( torch.sum( (center - corner)**2) ).to(scene.device)    
    bounding_box_condition =  (torch.sqrt(torch.sum( (xyz_coordinates - center)**2, dim=2)) > bounding_sphere_radius).to(scene.device)
    bounding_box_condition_indices = torch.where(bounding_box_condition)

    #rgb_img[bounding_box_condition_indices] = torch.tensor([0.8, 0.8, 0.8]).to(device=scene.device)
    #depth_img[bounding_box_condition_indices] = 1.0

    result = {
        'rendered_image_fine': rgb_img,
        'rendered_depth_fine': depth_img,
        'rendered_image_coarse': rgb_img,
        'rendered_depth_coarse': depth_img
    }

    return result


def render_all_training_images(scene, images_dir):
    
    focal_length = scene.models["focal"](0)
    scene.compute_ray_direction_in_camera_coordinates(focal_length)

    for image_id in range(0, 157):
        print("rendering image {}".format(image_id))
        color_out_file_name = '{}/color_renders/color_{}.png'.format(images_dir, image_id).zfill(4)
        depth_out_file_name = '{}/depth_renders/depth_{}.png'.format(images_dir, image_id).zfill(4)
        render_result = scene.render_prediction_for_train_image(image_id)
        scene.save_render_as_png(render_result, color_out_file_name, depth_out_file_name)


def color_mesh_with_nerf_colors(scene, mesh):
    
    # Using a very small measure distance makes things more robust to inaccurate normals. We really only
    # care about the general direction the normal is pointing.
    measure_distance = torch.tensor(0.001).to(torch.device('cuda:0'))        

    scene.near = torch.tensor([measure_distance/2.0]).to(torch.device('cuda:0')) 
    scene.far = measure_distance * (100)
    scene.far =scene.near + measure_distance*20.0
    scene.args.near_maximum_depth = 2000.0 * measure_distance
    scene.args.far_maximum_depth = scene.args.near_maximum_depth
    scene.args.percentile_of_samples_in_near_region = 1.0

    vertices = torch.tensor(np.asarray(mesh.vertices).tolist()).to(torch.device('cuda:0'))
    vertex_normals = torch.tensor(np.asarray(mesh.vertex_normals).tolist()).to(torch.device('cuda:0'))
    normals = torch.nn.functional.normalize(vertex_normals, dim=1, p=2)
    n_vertices = vertices.size()[0]
        
    print('n_vertices: ', n_vertices)
    print('vertex_normals size: ', vertex_normals.size()[0])

    min_x = -0.1519
    max_x = 0.1625

    min_y = -0.3678
    max_y = -0.1093

    min_z =  -0.4149
    max_z =  -0.1301

    vertices = vertices / 500.0
    vertices[:, 0] += min_x
    vertices[:, 1] += min_y
    vertices[:, 2] += min_z

    

    # construct local coordinate system for each camera (looking at vertex)
    # https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function                
    poses = torch.zeros(n_vertices,4,4).to(torch.device('cuda:0'))

    p_from = -normals * measure_distance + vertices
    p_to = vertices
    print('p_from: ', p_from)
    print('p_to: ', p_to)    
    v_forward = -torch.nn.functional.normalize(p_from - p_to, dim=1, p=2) 
    v_arbitrary = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0).expand(n_vertices, 3).to(torch.device('cuda:0'))
    v_right = -torch.cross(v_arbitrary, v_forward, dim=1)     
    v_right = torch.nn.functional.normalize(v_right, dim=1, p=2)
    v_up = torch.cross(v_forward, v_right, dim=1)
    v_up = torch.nn.functional.normalize(v_up, dim=1, p=2) 

    poses[:,0, :3] = v_right
    poses[:,1, :3] = v_up
    poses[:,2, :3] = v_forward
    poses[:,3,  3] = 1.0

    pixel_directions = torch.tensor([0.0,0.0,1.0]).unsqueeze(0).expand(n_vertices, 3).to(torch.device('cuda:0'))
    #pixel_directions = scene.pixel_directions[0][scene.H//2, scene.W//2].unsqueeze(0).expand(n_vertices, 3).to(torch.device('cuda:0'))
    focal_length = scene.models["focal"](0)    
    focal_lengths = focal_length[0].expand(n_vertices)
    
    poses[:, :3, 3] = p_from #poses[:, :3, 3] - measure_distance*poses[:,2,:3]
    render_result = scene.flat_render(poses, pixel_directions, focal_lengths)     
    vertex_colors = render_result['rendered_pixels'] 
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors.cpu())   
    mesh.vertices = o3d.utility.Vector3dVector(vertices.cpu())
    o3d.io.write_triangle_mesh('colored_mesh.ply', mesh, write_ascii = True)    

if __name__ == '__main__':
    
    with torch.no_grad():

        dynamic_args = {
            "number_of_samples_outward_per_raycast" : 360,
            "number_of_samples_outward_per_raycast_for_test_renders" : 360,
            "percentile_of_samples_in_near_region" : 0.8,
            "number_of_pixels_per_batch_for_test_renders" : 5000,            
            "near_maximum_depth" : 1.0,
            "skip_every_n_images_for_training" : 60,
            "use_sparse_fine_rendering" : False,
            #"base_directory" : '\'./data/orchid\'',
            #"base_directory" : '\'./data/dragon_scale\'',
            #"base_directory" : '\'./data/dragon_scale\'',            
            #"base_directory" : '\'./data/cactus\'',            
            #"pretrained_models_directory" : '\'./data/dragon_scale/hyperparam_experiments/5k_camerafixed/\'',                        
            "pretrained_models_directory" : '\'./data/cactus/hyperparam_experiments/from_cloud/cactus_run28/models/\'',                        
            "start_epoch" : 50001,
            "load_pretrained_models" : True,
        }
        scene = SceneModel(args=parse_args(), experiment_args='dynamic', dynamic_args=dynamic_args)          
            
        print (scene.args.near_maximum_depth)
        data_out_dir = "{}/videos".format(scene.args.base_directory)            
        experiment_label = "{}_{}".format(scene.start_time, 'spin_video')                            
        experiment_dir = Path(os.path.join(data_out_dir, experiment_label))

        n_poses = 200
        print("creating spin video images")        
        create_spin_video_images(scene, n_poses, experiment_dir)
        print("converting images to video")        
        imgs_to_video(experiment_dir, n_poses)
        print("video output to {}".format(experiment_dir))


        """
            fname = 'dragon_scale_tri.ply'
            #fname = './data/dragon_scale/hyperparam_experiments/pretrained_with_entropy_loss_200k/amazing_pointclouds/depth_view_pointcloud/meshes/dragon_scale_0_luminosity_filtered_only.glb'
            #print('Loading mesh: {}'.format(fname))
            mesh =  o3d.io.read_triangle_mesh(fname)
            color_mesh_with_nerf_colors(scene, mesh)

            #zoom_out_from_point(scene)
        """
