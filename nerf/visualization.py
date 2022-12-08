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



def rotate_pose_in_camera_space(scene, pose, d_pitch, d_roll, d_yaw):
    camera_rotation_matrix = pose[:3,:3]
    
    pose_quaternion = matrix_to_quaternion(camera_rotation_matrix)
    new_quaternion = quaternion_multiply(pose_quaternion, axis_angle_to_quaternion(torch.FloatTensor([d_pitch,d_roll,d_yaw])))

    new_camera_rotation_matrix = quaternion_to_matrix(new_quaternion)
    new_camera_axis_angles = matrix_to_axis_angle(new_camera_rotation_matrix)

    new_pose = torch.zeros((4,4))
    new_pose[:3,:3] = new_camera_rotation_matrix
    new_pose[:3,3] = pose[:3,3]
    new_pose[3,3] = 1.0

    return new_pose        
    

def translate_pose_in_global_space(scene, pose, d_x, d_y, d_z):        
    camera_rotation_matrix = pose[:3,:3]
    translation = torch.FloatTensor([d_x,d_y,d_z])

    new_pose = torch.clone(pose)
    new_pose[:3,:3] = pose[:3,:3]
    new_pose[:3,3] = pose[:3,3] + translation
    new_pose[3,3] = 1.0

    return new_pose


def construct_pose(scene, pitch, yaw, roll, x, y, z):
    pose = torch.zeros((4,4))
    pose_quaternion = axis_angle_to_quaternion(torch.FloatTensor([pitch, yaw, roll]))
    pose_rotation_matrix = quaternion_to_matrix(pose_quaternion)
    pose[:3,:3] = pose_rotation_matrix
    pose[:3,3] = torch.FloatTensor([x,y,z])
    pose[3,3] = 1.0

    return pose        





def generate_spin_poses(scene, number_of_poses):

    # bottom of pot: y=-0.34
    # top of pot: y=-0.26
    # center of pot: x=(-0.0057 + 0.059) / 2
    # center of pot: z=(-0.212 + -0.267) / 2
    # [0.051322 -0.364376 -0.272635]
    # [0.004562 -0.364395 -0.222432]

    # dragon scale
    center_x = (0.051322 + 0.004562) / 2
    center_y = -0.26
    center_z = (-0.272635 -0.222432) / 2    
    initial_pitch = np.pi - np.pi/4.5
    r = 0.2
    dy = 0.2
    dx = 0
    dz = r  

    # elastica
    # center of soil surface: [-0.021638 -0.548057 -0.290823] [689,618]
    # top of stem
    #center_x = -0.0236
    #center_y = -0.278801
    #center_z = -0.203017            
    # center of soil surface + stem height
    #center_x = -0.021638
    #center_y = -0.278801
    #center_z = -0.290823
    #initial_pitch = np.pi - np.pi/3.4  # pointing down minus 30 degrees from horizon
    #r = 0.33
    #dy = 0.22
    #dx = 0
    #dz = r    
    
    initial_yaw = 0 # north
    initial_roll = 0 # horizon
    initial_y = center_y + dy
    initial_x = center_x + dx
    initial_z = center_z + dz    
    
    next_cam_pose = torch.zeros(4,4)
    matrix = euler_angles_to_matrix(torch.tensor([initial_pitch, initial_roll, initial_yaw]), "XYZ")
    next_cam_pose[:3, :3] = matrix
    next_cam_pose[:3, 3] = torch.tensor([initial_x, initial_y, initial_z])
                
    new_pose = torch.clone(next_cam_pose)  
    poses = [new_pose]
    pose_debug = []        
    
    center_pixel_xyz = torch.tensor([center_x, center_y, center_z])
    for i in range(0, number_of_poses):
                
        x = next_cam_pose[0,3]
        y = next_cam_pose[2,3]

        # convert to polar coordinates with center_pixel_xyz as origin     
        theta = torch.atan2(torch.FloatTensor([y - center_pixel_xyz[2]]),torch.FloatTensor([x - center_pixel_xyz[0]]))               

        # rotate
        theta = theta + 2.0*np.pi/number_of_poses

        # convert back to cartesian coordinates
        xp = r * math.cos(theta) + center_pixel_xyz[0]
        yp = r * math.sin(theta) + center_pixel_xyz[2]

        # translate then rotate        
        next_cam_pose = translate_pose_in_global_space(scene, next_cam_pose, (xp - x), 0.0, (yp - y))
        
        # convention: pitch, yaw, roll
        next_cam_pose = rotate_pose_in_camera_space(scene, next_cam_pose, np.pi - initial_pitch, 0.0, 0.0)
        next_cam_pose = rotate_pose_in_camera_space(scene, next_cam_pose, 0.0, 1.0 * 2.0 * np.pi / number_of_poses, 0.0)                    
        next_cam_pose = rotate_pose_in_camera_space(scene, next_cam_pose, -(np.pi - initial_pitch), 0.0, 0.0)
        
        new_pose = torch.clone(next_cam_pose)        
        pose_debug.append([new_pose[0,3], new_pose[2,3], new_pose[1,3]])                        
        poses.append(new_pose)         
    
    poses = torch.stack(poses, 0)    
    poses = convert3x4_4x4(poses).to(scene.device)

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
    focal_length_x, focal_length_y = scene.models["focal"](0)
        
    print('using focal length {}'.format(focal_length_x[0]))
    scene.compute_ray_direction_in_camera_coordinates(focal_length_x, focal_length_y)

    for i,pose in enumerate(poses):
        index = i
        print('rendering pose {}'.format(i))
        render_result = scene.basic_render(pose, scene.pixel_directions[0], focal_length_x[0])
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
    pixel_directions = pixel_directions.reshape(scene.H, scene.W, 3).unsqueeze(3)

    rgb_img = rgb.reshape(scene.H, scene.W, 3).to(device=scene.device)
    depth_img = depth.reshape(scene.H, scene.W).to(device=scene.device)    
    xyz_coordinates = scene.derive_xyz_coordinates(camera_world_position, camera_world_rotation, pixel_directions, depth_img)    
    # dragon_scale
    center = torch.tensor([0.0044, -0.2409, -0.2728]).to(scene.device)
    corner = torch.tensor([-0.1450, -0.3655, -0.4069]).to(scene.device)
    # elastica_burgundy
    #center = torch.tensor([0.0058, -0.4112, -0.2170]).to(scene.device)    
    #mins = torch.tensor([-0.1934, -0.6670, -0.3852]).to(scene.device)
    #maxs = torch.tensor([0.2049, -0.1554, -0.0488]).to(scene.device)
    #bounding_box_condition = torch.logical_and(xyz_coordinates > mins, xyz_coordinates < maxs).float()
    #bounding_box_condition = torch.sum(bounding_box_condition, dim=2)    
    #bounding_box_condition_indices = torch.where(bounding_box_condition != 3.0)
    
    #print(bounding_box_condition_indices.size())
    
    bounding_sphere_radius = torch.sqrt( torch.sum( (center - corner)**2) ).to(scene.device)    
    bounding_box_condition =  (torch.sqrt(torch.sum( (xyz_coordinates - center)**2, dim=2)) > bounding_sphere_radius).to(scene.device)
    bounding_box_condition_indices = torch.where(bounding_box_condition)

    rgb_img[bounding_box_condition_indices] = torch.tensor([0.8, 0.8, 0.8]).to(device=scene.device)
    depth_img[bounding_box_condition_indices] = 1.0

    result = {
        'rendered_image_fine': rgb_img,
        'rendered_depth_fine': depth_img,
        'rendered_image_coarse': rgb_img,
        'rendered_depth_coarse': depth_img
    }

    return result


def render_all_training_images(scene, images_dir):
    
    focal_length_x, focal_length_y = scene.models["focal"](0)
    scene.compute_ray_direction_in_camera_coordinates(focal_length_x, focal_length_y)

    for image_id in range(0, 157):
        print("rendering image {}".format(image_id))
        color_out_file_name = '{}/color_renders/color_{}.png'.format(images_dir, image_id).zfill(4)
        depth_out_file_name = '{}/depth_renders/depth_{}.png'.format(images_dir, image_id).zfill(4)
        render_result = scene.render_prediction_for_train_image(image_id)
        scene.save_render_as_png(render_result, color_out_file_name, depth_out_file_name)


def zoom_out_from_point(scene):

    #fname = 'dragon_scale_tri.ply'
    fname = './data/dragon_scale/hyperparam_experiments/pretrained_with_entropy_loss_200k/amazing_pointclouds/depth_view_pointcloud/meshes/dragon_scale_0_luminosity_filtered_only.glb'    
    mesh =  o3d.io.read_triangle_mesh(fname)
    vertices = torch.tensor(np.asarray(mesh.vertices).tolist()).to(torch.device('cuda:0'))
    vertex_normals = torch.tensor(np.asarray(mesh.vertex_normals).tolist()).to(torch.device('cuda:0'))    

    view_number = 0
    focus_point = vertices[view_number]
    normal = vertex_normals[view_number]
    normal = torch.nn.functional.normalize(normal, dim=0, p=2)
    print('normal: ',normal)
    # x (left/right) right = +
    # y (up/down) {is upside down} down = +
    # z (forward/back) forward = +
    pose = torch.zeros(4,4).to(torch.device('cuda:0'))

    measure_distance = 0.01
    scene.near = torch.tensor([measure_distance]).to(torch.device('cuda:0'))
    #scene.args.near_maximum_depth = torch.tensor([0.1])
    #scene.args.far_maximum_depth = torch.tensor([0.11])

    p_from = -normal * measure_distance + focus_point
    p_to = focus_point    
    print('p_from: ', p_from)
    print('p_to: ', p_to)    
    v_forward = -torch.nn.functional.normalize(p_from - p_to, dim=0, p=2) # original    
    v_arbitrary = torch.tensor([0.0, 1.0, 0.0]).to(torch.device('cuda:0'))
    v_right = -torch.cross(v_arbitrary, v_forward)     
    v_right = torch.nn.functional.normalize(v_right, dim=0, p=2)
    v_up = torch.cross(v_forward, v_right) # original    
    v_up =torch.nn.functional.normalize(v_up, dim=0, p=2) # original    
    pixel_directions = scene.pixel_directions[view_number]
    focal_length_x, focal_length_y = scene.models["focal"](0)    
    focal_length = focal_length_x[view_number]

    # for the pose, the coordinate transform matrix is the mapping from camera to world pose,
    # but xyz is the *world* offset

    # also, z is backwards

    
    n_poses = 100
    for i in range(n_poses):
        print(i)
        pose[0, :3] = v_right.clone()
        pose[1, :3] = v_up.clone()
        pose[2, :3] = v_forward.clone()
        pose[3,  3] = 1.0
        
        #pose = scene.models['pose']()[35]
        if i==0:
            print([pose[:3,:3]])
        #pose[:3, 3] = torch.matmul(pose[:3,:3], p_from.unsqueeze(1)).squeeze(1)
        pose[:3, 3] = p_from
        pose[:3, :3] = pose[:3,:3].inverse()

        

        xyz_camera = torch.tensor([0.0, 0.0, -i*measure_distance]).unsqueeze(1).to(torch.device('cuda:0'))
        
        pose[:3, 3] = pose[:3, 3] + torch.matmul(pose[:3,:3], xyz_camera).squeeze(1)

        #pose[:3, 3] = p_from - pose[2, :3] * i * measure_distance
        #pose[2, 3] = pose[2, 3] - measure_distance
        #pose[3,  3] = 1.0                

        
        print(pose)
        render_result = scene.basic_render(pose, pixel_directions, focal_length)     
        scene.save_render_as_png(render_result, color_file_name_fine='tmp/color_{}.png'.format(i), depth_file_name_fine='tmp/depth_{}.png'.format(i), color_file_name_coarse=None, depth_file_name_coarse=None)
        



def color_mesh_with_nerf_colors(scene, mesh):
    

    # bottom of pot: y=-0.34
    # top of pot: y=-                          
    # center of pot: x=(-0.0057 + 0.059) / 2 = 
    # center of pot: z=(-0.212 + -0.267) / 2 = 
    
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

    """

        # blender transformation:
        # x = -0.153701 m
        # y = -0.363557 m
        # z = -0.412759 m
        # scale = 0.0002

        # nerf:
        # right = x
        # up = y
        # forward = -z

        # blender:
        # right = x
        # up = z
        # forward = -y

        # 0.008674 -0.166496 -0.196280
        #0.008865 -0.209839 -0.139628] 


        # 0.009557 -0.170050 -0.189943
        # 0.008283 -0.168344 -0.194182

        vertices = vertices / 500.0

        x_offset = (-0.153701) + (0.008674  - 0.008865) + (0.009557 - 0.008283)
        y_offset = (-0.412759) + (-0.166496 - -0.209839) + (-0.170050 - -0.168344)
        z_offset = (-0.363557) + (-0.196280 - -0.139628) + (-0.189943 - -0.194182)
        
        
        vertices[:, 0] += x_offset
        vertices[:, 1] += y_offset
        vertices[:, 2] += z_offset
    """
    

        
    

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
    focal_length_x, focal_length_y = scene.models["focal"](0)    
    focal_lengths = focal_length_x[0].expand(n_vertices)

    
    poses[:, :3, 3] = p_from #poses[:, :3, 3] - measure_distance*poses[:,2,:3]
    render_result = scene.flat_render(poses, pixel_directions, focal_lengths)     
    vertex_colors = render_result['rendered_pixels'] 
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors.cpu())   
    mesh.vertices = o3d.utility.Vector3dVector(vertices.cpu())
    o3d.io.write_triangle_mesh('colored_mesh.ply', mesh, write_ascii = True)    

if __name__ == '__main__':
    
    with torch.no_grad():

        scene = SceneModel(args=parse_args(), experiment_args='test')
        scene.args.number_of_samples_outward_per_raycast = 128                   
        scene.args.use_sparse_fine_rendering = False
        """
                
            data_out_dir = "{}/videos".format(scene.args.base_directory)            
            #experiment_label = "{}_{}".format(scene.start_time, 'spin_video')                    
            experiment_label = '1667331314_spin_video'
            experiment_dir = Path(os.path.join(data_out_dir, experiment_label))

            n_poses = 479
            print("creating spin video images")        
            #create_spin_video_images(scene, n_poses, experiment_dir)
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
