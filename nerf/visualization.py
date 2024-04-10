import torch
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix, matrix_to_euler_angles, matrix_to_quaternion, axis_angle_to_quaternion, quaternion_multiply, matrix_to_axis_angle, axis_angle_to_matrix, euler_angles_to_matrix
from scipy.spatial.transform import Rotation
from torchsummary import summary
import open3d as o3d
from PIL import Image
import numpy as np
import random, math
import sys, os, shutil, copy, glob, json
import time, datetime
import argparse
import wandb
from pathlib import Path
from tqdm import tqdm
import os

sys.path.append(os.path.join(sys.path[0], '../..'))
from utils.pos_enc import encode_position
from utils.volume_op import volume_sampling, volume_rendering
from utils.lie_group_helper import convert3x4_4x4
from utils.training_utils import PolynomialDecayLearningRate, heatmap_to_pseudo_color, set_randomness, save_checkpoint
from models.intrinsics import CameraIntrinsicsModel
from models.poses import CameraPoseModel
from models.nerf_models import NeRFDensity, NeRFColor

from learn import *
from utils.camera import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import    


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

set_randomness()
torch.set_float32_matmul_precision('high')
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose=False      

def rotate_pose_in_camera_space(pose, dx, dy, dz):
    camera_rotation_matrix = pose[:3,:3]
    
    Rx = torch.tensor([
        [1.0,    0.0,      0.0       ],
        [0,   torch.cos(dx), -torch.sin(dx)],
        [0,   torch.sin(dx),  torch.cos(dx) ]
    ]).to(torch.device('cuda:0')).float()

    Ry = torch.tensor([
        [torch.cos(dy),   0.0,   torch.sin(dy)],
        [0,            1.0,   0.0       ],
        [-torch.sin(dy),  0.0,   torch.cos(dy)]
    ]).to(torch.device('cuda:0')).float()

    Rz = torch.tensor([
        [torch.cos(dz), -torch.sin(dz),   0.0 ],
        [torch.sin(dz), torch.cos(dz),    0.0 ],
        [0.0,          0.0,         1.0]
    ]).to(torch.device('cuda:0')).float()

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

def translate_pose_in_camera_space(pose, d_x, d_y, d_z):        
    camera_rotation_matrix = pose[:3,:3]
    translation = camera_rotation_matrix[:3,:3] @ torch.FloatTensor([d_x,d_y,d_z]).to(pose.device)
    
    new_pose = torch.clone(pose)
    new_pose[:3,:3] = pose[:3,:3]
    new_pose[:3,3] = pose[:3,3] + translation
    new_pose[3,3] = 1.0

    return new_pose    


def get_center(scene, dataset=None):

    if dataset == "cactus":
        return torch.tensor([0.002148, -0.30666, -0.278652])
        return torch.tensor([0.032756, -0.31353, -0.308534])
    elif dataset == "dragon_scale":
        return torch.tensor([0.0044, -0.2409, -0.2728])
    else:
        object_xyzs = scene.xyz[torch.where(scene.rgbd[:, 3] < 0.5)]
        center_x = torch.mean(object_xyzs[:, 0:1])
        center_y = torch.mean(object_xyzs[:, 1:2])
        center_z = torch.mean(object_xyzs[:, 2:3])
        return torch.tensor([center_x, center_y, center_z])

def generate_spin_poses(scene, number_of_poses):
            
    # construct local coordinate system for each camera (looking at vertex)
    # https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function   
    p_center = get_center(scene, "cactus")    
    p_to = p_center # point to look at
    
    dx = 0.000001
    #dy = 0.15
    dy = 0.3
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
    ######################################################################

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
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #xs = xyzs[:, 0]
    #ys = xyzs[:, 1]
    #zs = xyzs[:, 2]
    #ax.scatter(xs, zs, ys, marker='o')
    #ax.scatter(xs[0], zs[0], ys[0], marker='o', c='red')
    #ax.scatter(p_to[0], p_to[2], p_to[1], marker='o', c='green')
    #plt.show()

    poses = torch.stack(poses, 0)    
    poses = poses.to(scene.device)    

    return (poses, p_center)


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
    poses, center = generate_spin_poses(scene, number_of_poses)
    render_poses(scene, poses, video_dir, center)


def render_poses(scene, poses, video_dir, center):

    color_out_dir = Path("{}/color_video_images/".format(experiment_dir))
    color_out_dir.mkdir(parents=True, exist_ok=True)
    depth_out_dir = Path("{}/depth_video_images/".format(experiment_dir))
    depth_out_dir.mkdir(parents=True, exist_ok=True)   
    color_images = []
    depth_images = []
    pp_x = scene.principal_point_x * (scene.args.W_for_test_renders / scene.W)
    pp_y = scene.principal_point_y * (scene.args.H_for_test_renders / scene.H)           
        
    focal_lengths = scene.models['focal'](0)[0].expand(scene.args.H_for_test_renders*scene.args.W_for_test_renders)*(scene.args.W_for_test_renders / scene.W)
    pixel_directions = compute_pixel_directions(focal_lengths, scene.pixel_rows_for_test_renders, scene.pixel_cols_for_test_renders, pp_y, pp_x).to(torch.device('cuda:0'))

    for i,pose in enumerate(poses):
        index = i
        print('rendering pose {}'.format(i))
        poses = pose.unsqueeze(0).expand(scene.args.H_for_test_renders*scene.args.W_for_test_renders,4,4)
        render_result = scene.render_prediction(poses, focal_lengths, scene.args.H_for_test_renders, scene.args.W_for_test_renders, principal_point_x=pp_x, principal_point_y=pp_y)        
        render_result = filter_background(scene, render_result['rendered_image_fine'], render_result['rendered_depth_fine'], render_result['entropy_coarse'], pose, pixel_directions, center.to(scene.device))
        color_out_file_name = os.path.join(color_out_dir, "color_{}.png".format(index))                
        depth_out_file_name = os.path.join(depth_out_dir, "depth_{}.png".format(index))                        
        
        scene.save_render_as_png(render_result, scene.args.H_for_test_renders, scene.args.W_for_test_renders, color_out_file_name, depth_out_file_name)

        rendered_color_for_file = (render_result['rendered_image_fine'].cpu().numpy() * 255).astype(np.uint8)    
        rendered_depth_data = render_result['rendered_depth_fine'].cpu().numpy()         
        

        rendered_depth_for_file = heatmap_to_pseudo_color(rendered_depth_data)
        rendered_depth_for_file = (rendered_depth_for_file * 255).astype(np.uint8)     
        color_images.append(rendered_color_for_file)   
        depth_images.append(rendered_depth_for_file)
    

def filter_background(scene, rgb, depth, entropy, pose, pixel_directions, center):

    
    #radius = torch.tensor([0.35]).to(scene.device)
    radius = torch.tensor([0.35]).to(scene.device)
    
    center = center.to(scene.device)
    rgb = rgb.to(scene.device)
    entropy = entropy.to(scene.device)
    min_y = -0.38

    camera_world_position = pose[:3, 3].view(1, 1, 1, 3)     # (1, 1, 1, 3)
    camera_world_rotation = pose[:3, :3].view(1, 1, 1, 3, 3) # (1, 1, 1, 3, 3)

    #pixel_directions = pixel_directions.reshape(scene.args.H_for_test_renders, scene.args.W_for_test_renders, 3).unsqueeze(3).to(device=scene.device)
    #pixel_directions = torch.nn.functional.normalize(pixel_directions, p=2, dim=2)
    #rgb_img = rgb.reshape(scene.args.H_for_test_renders, scene.args.W_for_test_renders, 3).to(device=scene.device)
    #depth_img = depth.reshape(scene.args.H_for_test_renders, scene.args.W_for_test_renders).to(device=scene.device)    

    xyz_coordinates = derive_xyz_coordinates(camera_world_position, camera_world_rotation, pixel_directions.to(scene.device), depth.to(scene.device), flattened=True).to(scene.device)

    center_radius_filter = torch.sqrt(torch.sum((xyz_coordinates - center.unsqueeze(0).expand(xyz_coordinates.size()[0], 3))**2,dim=1)) > radius

    rgb[center_radius_filter] = torch.tensor([0.8, 0.8, 0.8]).to(device=scene.device)
    depth[center_radius_filter] = 0.0

    entropy_filter = entropy > 999.0    
    rgb[entropy_filter] = torch.tensor([0.8, 0.8, 0.8]).to(device=scene.device)
    depth[entropy_filter] = 0.0    

    y_filter = xyz_coordinates[:,1] < min_y
    rgb[y_filter] = torch.tensor([0.8, 0.8, 0.8]).to(device=scene.device)
    depth[y_filter] = 0.0



    # dragon_scale
    #center = torch.tensor([0.0044, -0.2409, -0.2728]).to(scene.device)
    #corner = torch.tensor([-0.1450, -0.3655, -0.4069]).to(scene.device)
    # elastica_burgundy
    #center = torch.tensor([0.0058, -0.4112, -0.2170]).to(scene.device)    


    #mins = torch.tensor([-0.1934, -0.6670+0.25, -0.3852]).to(scene.device)*1.5
    #maxs = torch.tensor([0.2049, -0.1554+0.1, -0.0488+0.1]).to(scene.device)*1.5
    #bounding_box_condition = torch.logical_and(xyz_coordinates > mins, xyz_coordinates < maxs).float()
    #bounding_box_condition = torch.sum(bounding_box_condition, dim=2)    
    #bounding_box_condition_indices = torch.where(bounding_box_condition != 3.0)
    
    #print(bounding_box_condition_indices.size())
    
    #bounding_sphere_radius = torch.sqrt( torch.sum( (center - corner)**2) ).to(scene.device)    
    #bounding_box_condition =  (torch.sqrt(torch.sum( (xyz_coordinates - center)**2, dim=2)) > bounding_sphere_radius).to(scene.device)
    #bounding_box_condition_indices = torch.where(bounding_box_condition)

    #rgb_img[bounding_box_condition_indices] = torch.tensor([0.8, 0.8, 0.8]).to(device=scene.device)
    #depth_img[bounding_box_condition_indices] = 1.0

    result = {
        'rendered_image_fine': rgb.reshape(scene.args.H_for_test_renders, scene.args.W_for_test_renders,3),
        'rendered_depth_fine': depth.reshape(scene.args.H_for_test_renders, scene.args.W_for_test_renders)
        #'rendered_image_coarse': rgb_img,
        #'rendered_depth_coarse': depth_img
    }

    return result


def color_mesh_with_nerf_colors(scene, mesh):
    
    # Using a very small measure distance makes things more robust to inaccurate normals. We really only
    # care about the general direction the normal is pointing.

    #quit()
    measure_distance = torch.tensor(0.000002).to(torch.device('cuda:0'))        
    scene.near = torch.tensor(0.0).to(torch.device('cuda:0'))
    scene.far = torch.tensor(0.000004).to(torch.device('cuda:0'))
    #scene.near = torch.tensor(0.01).to(torch.device('cuda:0')) #torch.tensor([measure_distance/2.0]).to(torch.device('cuda:0'))     
    
    scene.args.near_maximum_depth = torch.tensor(0.01).to(torch.device('cuda:0'))
    scene.args.far_maximum_depth = torch.tensor(3.0).to(torch.device('cuda:0'))
    scene.args.percentile_of_samples_in_near_region = torch.tensor(0.9).to(torch.device('cuda:0'))

    vertices = torch.tensor(np.asarray(mesh.vertices).tolist()).to(torch.device('cuda:0'))        
    
    vertex_normals = torch.tensor(np.asarray(mesh.vertex_normals).tolist()).to(torch.device('cuda:0'))
    
    print (vertex_normals.shape)
    
    normals = torch.nn.functional.normalize(vertex_normals, dim=1, p=2)
    n_vertices = vertices.size()[0]
        
    print('n_vertices: ', n_vertices)
    print('vertex_normals size: ', vertex_normals.size()[0])
            
    poses = torch.zeros(n_vertices,4,4).to(torch.device('cuda:0'))
    #poses = torch.zeros(scene.args.W_for_test_renders * scene.args.H_for_test_renders,4,4).to(torch.device('cuda:0'))    
    
    # construct local coordinate system for each camera (looking at vertex)
    # https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function/framing-lookat-function.html   

    p_to = vertices    
    p_from = normals * measure_distance + vertices

    
    v_forward = torch.nn.functional.normalize(p_from - p_to, dim=1, p=2).float()
    

    v_arbitrary = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0).expand(n_vertices, 3).to(torch.device('cuda:0'))
    v_right = torch.cross(v_arbitrary, v_forward, dim=1)
    v_right = torch.nn.functional.normalize(v_right, dim=1, p=2)
    v_up = torch.cross(v_forward, v_right, dim=1)
    v_up = torch.nn.functional.normalize(v_up, dim=1, p=2)


    forward = torch.nn.functional.normalize(p_from - p_to, dim=1, p=2).float()
    right = -torch.nn.functional.normalize(torch.cross(v_up, forward, dim=1), dim=1, p=2).float()
    up = torch.nn.functional.normalize(torch.cross(forward, right, dim=1), dim=1, p=2).float()        
    
    #initial_pose = torch.zeros(4,4)    
    #initial_pose[0, :3] = v_right
    #initial_pose[1, :3] = v_up
    #initial_pose[2, :3] = v_forward
    #initial_pose[3,  3] = 1.0
    #initial_pose[:3, 3] = p_from     
    poses[:,  0, :3] = right
    poses[:,  1, :3] = up
    poses[:,  2, :3] = forward
    poses[:,  3,  3] = 1.0
    poses[:, :3,  3] = p_from    




    #pose = scene.models['pose'](0)[0]
    #print(pose)
    #poses = pose.unsqueeze(0).expand(scene.args.W_for_test_renders * scene.args.H_for_test_renders, 4, 4)


    
    #poses = scene.models['pose'](0)[0].unsqueeze(0).expand(scene.args.W_for_test_renders * scene.args.H_for_test_renders, 4, 4)

    #temp_pose = rotate_pose_in_camera_space(poses[100], torch.tensor(0.0).to(torch.device('cuda:0')), torch.tensor(-np.pi/2).to(torch.device('cuda:0')), torch.tensor(0.0).to(torch.device('cuda:0')))
    #temp_pose = rotate_pose_in_camera_space(poses[100], torch.tensor(0.0).to(torch.device('cuda:0')), torch.tensor(0.0).to(torch.device('cuda:0')), torch.tensor(0.0).to(torch.device('cuda:0')))
    #poses = temp_pose.unsqueeze(0).expand(scene.args.W_for_test_renders * scene.args.H_for_test_renders, 4, 4)
    #rows = scene.pixel_rows_for_test_renders
    #cols = scene.pixel_cols_for_test_renders
    #focal_lengths = (scene.models['focal'](0)[0] * (scene.args.W_for_test_renders / scene.W)).expand(scene.args.W_for_test_renders * scene.args.H_for_test_renders)

    pp_x = scene.principal_point_x * (scene.args.W_for_test_renders / float(scene.W))
    pp_y = scene.principal_point_y * (scene.args.H_for_test_renders / float(scene.H))
     
    focal_lengths = (scene.models['focal'](0)[0] * (scene.args.W_for_test_renders / float(scene.W))).expand(n_vertices)    
    rows = pp_y.float().expand(n_vertices)
    cols = pp_x.float().expand(n_vertices)
    scene.pixel_rows_for_test_renders = rows
    scene.pixel_cols_for_test_renders = cols    
    
    #pp_x = scene.principal_point_x * (scene.args.W_for_test_renders / scene.W)
    #pp_y = scene.principal_point_y * (scene.args.H_for_test_renders / scene.H)               

    #focal_lengths = (scene.models['focal'](0)[0] * (scene.args.W_for_test_renders / scene.W)).expand(n_vertices)    
    #rows = pp_y.int().expand(n_vertices)
    #cols = pp_x.int().expand(n_vertices)
    #scene.pixel_rows_for_test_renders = rows
    #scene.pixel_cols_for_test_renders = cols
    #pixel_directions = torch.tensor([0.0, 0.0, -1.0]).unsqueeze(0).expand(n_vertices, 3)

    #pixel_directions = compute_pixel_directions(focal_lengths=focal_lengths, pixel_rows=rows, pixel_cols=cols, principal_point_x=pp_x, principal_point_y=pp_y).to(torch.device('cuda:0'))    

    #pose = poses[0]    
    #poses = pose.unsqueeze(0).expand(scene.args.W_for_test_renders * scene.args.H_for_test_renders, 4, 4)
    #pose[:3, 3] = pose[:3, 3] + d * pose[2, :3]
    #render_result = scene.render_prediction(poses, focal_lengths, scene.args.H_for_test_renders, scene.args.W_for_test_renders, pp_x, pp_y)    
    #img = render_result['rendered_image_fine'].reshape(scene.args.H_for_test_renders, scene.args.W_for_test_renders, 3)

    #scene.save_render_as_png(render_result, scene.args.H_for_test_renders, scene.args.W_for_test_renders, 'vis_test/color_img_{}.png'.format(0), 'vis_test/depth_img_{}.png'.format(0))


    """
    d = 0.02
    for i in range (1, 100):

        pose = poses[0]
        #pose = translate_pose_in_camera_space(pose=pose, d_x=0.0, d_y=0.0, d_z=0.05)
        pose = rotate_pose_in_camera_space(pose.to(torch.device('cuda:0')), torch.tensor(0.0).to(torch.device('cuda:0')), torch.tensor(np.pi/16).to(torch.device('cuda:0')), torch.tensor(0.0).to(torch.device('cuda:0')))
        poses = pose.unsqueeze(0).expand(scene.args.W_for_test_renders * scene.args.H_for_test_renders, 4, 4)
        #pose[:3, 3] = pose[:3, 3] + d * pose[2, :3]
        render_result = scene.render_prediction(poses, focal_lengths, scene.args.H_for_test_renders, scene.args.W_for_test_renders, pp_x, pp_y)    
        #img = render_result['rendered_image_fine'].reshape(scene.args.H_for_test_renders, scene.args.W_for_test_renders, 3)

        scene.save_render_as_png(render_result, scene.args.H_for_test_renders, scene.args.W_for_test_renders, 'vis_test/color_img_{}.png'.format(i), 'vis_test/depth_img_{}.png'.format(i))
    
    """    
    render_result = scene.render_prediction(poses, focal_lengths, scene.args.H_for_test_renders, scene.args.W_for_test_renders, pp_x, pp_y)    
    #img = render_result['rendered_image_fine'].reshape(scene.args.H_for_test_renders, scene.args.W_for_test_renders, 3)
    #scene.save_render_as_png(render_result, scene.args.H_for_test_renders, scene.args.W_for_test_renders, 'meow.png', 'meow2.png')    
    
    vertex_colors = render_result['rendered_image_fine'] 
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors.cpu())   
    mesh.vertices = o3d.utility.Vector3dVector(vertices.cpu())
    o3d.io.write_triangle_mesh('colored_mesh.ply', mesh, write_ascii = True)    


if __name__ == '__main__':
    
    with torch.no_grad():

        dynamic_args = {
            "base_directory" : '\'./data/dragon_scale\'',
            "number_of_samples_outward_per_raycast" : 360,
            "number_of_samples_outward_per_raycast_for_test_renders" : 360,
            "density_neural_network_parameters" : 256,
            "percentile_of_samples_in_near_region" : 0.80,
            "number_of_pixels_per_batch_for_test_renders" : 1000,            
            #"H_for_test_renders" : 1440,
            #"W_for_test_renders" : 1920,
            "H_for_test_renders" : 240,
            "W_for_test_renders" : 320,            
            "near_maximum_depth" : 0.5,
            "skip_every_n_images_for_training" : 60,
            "use_sparse_fine_rendering" : False,
            "pretrained_models_directory" : '\'./data/dragon_scale/hyperparam_experiments/from_cloud/dragon_scale_run39/models\'',
            "start_epoch" : 500001,
            "load_pretrained_models" : True,            
        }



        scene = SceneModel(args=parse_args(), experiment_args='dynamic', dynamic_args=dynamic_args)          
            
        #print (scene.args.near_maximum_depth)
        #data_out_dir = "{}/videos".format(scene.args.base_directory)            
        #experiment_label = "{}_{}".format(scene.start_time, 'spin_video')                            
        #experiment_dir = Path(os.path.join(data_out_dir, experiment_label))

        #n_poses = 200
        #print("creating spin video images")        
        #create_spin_video_images(scene, n_poses, experiment_dir)
        #print("converting images to video")        
        #imgs_to_video(experiment_dir, n_poses)
        #print("video output to {}".format(experiment_dir))    
        
        fname = 'test/dragon_scale/3/dragon_high_ndc_mesh_in_nerf2.ply'
        #fname = 'test/dragon_scale/3/dragon_scale_mesh_3_bulb_removed(highres).ply'
        #fname = './data/dragon_scale/hyperparam_experiments/pretrained_with_entropy_loss_200k/amazing_pointclouds/depth_view_pointcloud/meshes/dragon_scale_0_luminosity_filtered_only.glb'
        #print('Loading mesh: {}'.format(fname))
        mesh =  o3d.io.read_triangle_mesh(fname)
        color_mesh_with_nerf_colors(scene, mesh)

        #zoom_out_from_point(scene)        
