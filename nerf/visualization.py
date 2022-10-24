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
    # initial_pitch = np.pi - np.pi/3.4  # pointing down minus 30 degrees from horizon



    r = 0.2
    dy = 0.2
    dx = 0
    dz = r    

    
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
        
        " pitch, yaw, roll "
        next_cam_pose = rotate_pose_in_camera_space(scene, next_cam_pose, np.pi - initial_pitch, 0.0, 0.0)
        next_cam_pose = rotate_pose_in_camera_space(scene, next_cam_pose, 0.0, 1.0 * 2.0 * np.pi / number_of_poses, 0.0)                    
        next_cam_pose = rotate_pose_in_camera_space(scene, next_cam_pose, -(np.pi - initial_pitch), 0.0, 0.0)
        
        new_pose = torch.clone(next_cam_pose)        
        pose_debug.append([new_pose[0,3], new_pose[2,3], new_pose[1,3]])                        
        poses.append(new_pose)         


    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #center_pixel_xyz = center_pixel_xyz.cpu()
    #ax.scatter([x[0] for x in pose_debug], [x[1] for x in pose_debug], [x[2] for x in pose_debug])
    #ax.scatter([center_pixel_xyz[0]], [center_pixel_xyz[2]], [center_pixel_xyz[1]], marker='o')
    #plt.show()
    
    poses = torch.stack(poses, 0)    
    poses = convert3x4_4x4(poses).to(scene.device)

    return poses


def imgs_to_video(video_dir, n_poses):
    cimgs = []
    dimgs = []
    for i in range(0, n_poses):
        fname1 = '{}/color_video/color_{}.png'.format(video_dir, i)
        fname2 = '{}/depth_video/depth_{}.png'.format(video_dir, i)
        cimg = Image.open(fname1)
        dimg = Image.open(fname2)
        cimgs.append(cimg)
        dimgs.append(dimg)
    imageio.mimwrite('{}/color.mp4'.format(video_dir), cimgs, fps=15, quality=9)        
    imageio.mimwrite('{}/depth.mp4'.format(video_dir), dimgs, fps=15, quality=9)        
    

def create_spin_video(scene, number_of_poses, video_dir):

    print("generating spin poses")
    poses = generate_spin_poses(scene, number_of_poses)
    create_video_from_poses(scene, poses, video_dir)


def create_video_from_poses(scene, poses, video_dir):


    color_out_dir = Path("{}/color_video/".format(experiment_dir))
    color_out_dir.mkdir(parents=True, exist_ok=True)
    depth_out_dir = Path("{}/depth_video/".format(experiment_dir))
    depth_out_dir.mkdir(parents=True, exist_ok=True)   

    color_images = []
    depth_images = []

    focal_length_x, focal_length_y = scene.models["focal"](0)
        
    print('using focal length {}'.format(focal_length_x[0]))
    scene.compute_ray_direction_in_camera_coordinates(focal_length_x, focal_length_y)

    for i,pose in enumerate(poses[62:]):
        index = i + 62
        print('rendering pose {}'.format(i))
        render_result = scene.render_prediction(pose=pose, train_image_index=0, max_depth=None)
        color_out_file_name = os.path.join(color_out_dir, "color_{}.png".format(index))                
        depth_out_file_name = os.path.join(depth_out_dir, "depth_{}.png".format(index))                        
        
        scene.save_render_as_png(render_result, color_out_file_name, depth_out_file_name)

        rendered_color_for_file = (render_result['rendered_image_fine'].cpu().numpy() * 255).astype(np.uint8)    
        rendered_depth_data = render_result['rendered_depth_fine'].cpu().numpy()         

        rendered_depth_for_file = heatmap_to_pseudo_color(rendered_depth_data)
        rendered_depth_for_file = (rendered_depth_for_file * 255).astype(np.uint8)     
        color_images.append(rendered_color_for_file)   
        depth_images.append(rendered_depth_for_file)

    imageio.mimwrite(os.path.join(color_out_dir, 'color.mp4'), color_images, fps=15, quality=9)
    imageio.mimwrite(os.path.join(depth_out_dir, 'depth.mp4'), depth_images, fps=15, quality=9)


def render_all_training_images(scene, images_dir):
    
    focal_length_x, focal_length_y = scene.models["focal"](0)
    scene.compute_ray_direction_in_camera_coordinates(focal_length_x, focal_length_y)

    for image_id in range(0, 157):
        print("rendering image {}".format(image_id))
        color_out_file_name = '{}/color_renders/color_{}.png'.format(images_dir, image_id).zfill(4)
        depth_out_file_name = '{}/depth_renders/depth_{}.png'.format(images_dir, image_id).zfill(4)
        render_result = scene.render_prediction_for_train_image(image_id)
        scene.save_render_as_png(render_result, color_out_file_name, depth_out_file_name)


if __name__ == '__main__':
    
    with torch.no_grad():
        scene = SceneModel(args=parse_args(), load_saved_args=True)
        scene.args.number_of_samples_outward_per_raycast = 1024
            
        data_out_dir = "{}/videos".format(scene.args.base_directory)            
        experiment_label = "{}_{}".format(scene.start_time, 'spin_video')                    
        experiment_dir = Path(os.path.join(data_out_dir, experiment_label))
 
        print("creating spin video images")
        #create_spin_video(scene, 120, experiment_dir)
        print("converting images to video")
        imgs_to_video(experiment_dir, 120)
        #render_all_training_images(scene)




















###################################### code for printing out weights and gradients of network input layer
#print(self.models['density'.])
"""
    print("density weights:")
    for i in range(128):
        print(self.models['geometry'].layers0[0].weight.size())
        quit()
        print("_________________________________")
        print("unit {}: ".format(i))
        print("weights:")
        print(self.models['geometry'].layers0[0].weight[i,:])
        print("gradient:")
        print(self.models['geometry'].layers0[0].weight.grad[i,:])
    
"""


"""
    xs = x[0, :, 0].detach().cpu().numpy()
    ys = x[0, :, 1].detach().cpu().numpy()
    zs = x[0, :, 2].detach().cpu().numpy()
    xs2 = depth_xyzs[0,:, 0].detach().cpu().numpy()
    ys2 = depth_xyzs[0,:, 1].detach().cpu().numpy()
    zs2 = depth_xyzs[0,:, 2].detach().cpu().numpy()        
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')    
    ax.scatter(xs, ys, zs, s=4, c='blue')    
    ax.scatter(xs2, ys2, zs2, s=4, c='green')        
    plt.show()    
"""    




########################## code for printing out features fromn pos_enc.py
#for i in [202]:    
#for i in range(0, sampling_depths[0].size()[0]-1):
"""
    for j in range(0,90//3):
        f = features[0][0,i, j*3 : j*3 + 3]                        
        v = features[1][0,i, j*3 : j*3 + 3]
"""
#f = features[0][0,i, 86:89]
#v = features[1][0,i, 86:89]
#print("depth {} feature {}: ({:8f}, {:8f}, {:8f} (var: {:8f}, {:8f}, {:8f}))".format(i, j, f[0].item(), f[1].item(), f[2].item(), v[0].item(), v[1].item(), v[2].item()))
        
#print("depth {} feature 86-89: ({:8f}, {:8f}, {:8f} (var: {:8f}, {:8f}, {:8f}))".format(i, f[0].item(), f[1].item(), f[2].item(), v[0].item(), v[1].item(), v[2].item()))


"""
    for i in range(0, 96//3):        
        print("depth {} feature {}: ({:8f}, {:8f}, {:8f})".format(10, i, (features[0][10,20,i].item()), (features[0][10,20,i+1].item()), (features[0][10,20,i+2].item())))
        print("depth {} feature {}: ({:8f}, {:8f}, {:8f})".format(50, i, features[0][10,100,i].item(), features[0][10,100,i+1].item(), features[0][10,100,i+2].item()))
        print("depth {} feature {}: ({:8f}, {:8f}, {:8f})".format(150, i, (features[0][10,300,i].item()), (features[0][10,300,i+1].item()), (features[0][10,300,i+2].item())))
        print("depth {} feature {}: ({:8f}, {:8f}, {:8f})".format(250, i, (features[0][10,500,i].item()), (features[0][10,500,i+1].item()), (features[0][10,500,i+2].item())))
    print("___________________________________________________________________")    
"""

""" test for explicitly treating focal_length as distance from lense to principal point
    #dp_x = (camera_coordinates_x - self.principal_point_x)
    #dp_y = (camera_coordinates_y - self.principal_point_y)
    #angles_x = torch.atan(dp_x / focal_length_x_rep)
    #angles_y = torch.atan(dp_y / focal_length_y_rep)    
    #camera_coordinates_directions_x = angles_x
    #camera_coordinates_directions_y = angles_y
"""

"""

    def debug_visualization(scene, rgb, xyz, depth_weights, rendered_image, downsample_ratio=50):
        # rgb := (N_pixels, N_samples, 3)
        # xyz :- (N_pixels, N_samples, 3)
        # depth_weights := (N_pixels, N_samples)

        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.clear()
        ax.set_title("Visualization of RGB + Depth Weights Per 128 raycast samples Per Pixel")
        all_visible_indices = []
        number_of_pixels, number_of_samples, _ = rgb.shape
        for sample in range(1, number_of_samples):
            sample_rgb = rgb[:,sample,:].cpu().detach().numpy()[int(number_of_pixels/3):int(2*number_of_pixels/3)][::downsample_ratio] # (N_pixels / 10, 3)
            sample_xyz = xyz[:,sample,:].cpu().detach().numpy()[int(number_of_pixels/3):int(2*number_of_pixels/3)][::downsample_ratio] # (N_pixels / 10, 3)
            sample_depth_weight = depth_weights[:,sample].cpu().detach().numpy()[int(number_of_pixels/3):int(2*number_of_pixels/3)][::downsample_ratio] # (N_pixels / 10)
            rgba = np.concatenate([sample_rgb, np.expand_dims(sample_depth_weight, axis=1)], axis=1) # (N_pixels / 10, 4)

            clearly_visible_point_indices = np.argwhere(rgba[:,3] > 0.05)
            selected_rgba = rgba[clearly_visible_point_indices,:]

            selected_x = sample_xyz[clearly_visible_point_indices,0]
            selected_y = sample_xyz[clearly_visible_point_indices,1]
            selected_z = sample_xyz[clearly_visible_point_indices,2]

            ax.scatter(selected_x, selected_y, selected_z, color=selected_rgba, marker=".", s=400)
            all_visible_indices.extend(clearly_visible_point_indices.flatten())
        
        # now show raycasts to get a sense of what the final colors are
        all_visible_indices_in_set = list(set(all_visible_indices))
        pixel_xyz_visible_start = xyz[int(number_of_pixels/3):int(2*number_of_pixels/3)].cpu().detach().numpy()[::downsample_ratio][all_visible_indices_in_set, 0, :]
        pixel_xyz_visible_end = xyz[int(number_of_pixels/3):int(2*number_of_pixels/3)].cpu().detach().numpy()[::downsample_ratio][all_visible_indices_in_set, -1, :]
        rendered_colors_of_visible_pixels = rendered_image[int(number_of_pixels/3):int(2*number_of_pixels/3), :].cpu().detach().numpy()[::downsample_ratio][all_visible_indices_in_set, :]

        pixel_raycast_x = np.stack([pixel_xyz_visible_start[:,0], pixel_xyz_visible_end[:,0]]) # (N_selected_pixels, 2)
        pixel_raycast_y = np.stack([pixel_xyz_visible_start[:,1], pixel_xyz_visible_end[:,1]]) # (N_selected_pixels, 2)
        pixel_raycast_z = np.stack([pixel_xyz_visible_start[:,2], pixel_xyz_visible_end[:,2]]) # (N_selected_pixels, 2)

        number_start_stop, number_of_pixels_to_raycast = pixel_raycast_x.shape
        for ray in range(number_of_pixels_to_raycast):
            xs = pixel_raycast_x[:,ray]
            ys = pixel_raycast_y[:,ray]
            zs = pixel_raycast_z[:,ray]
            color = rendered_colors_of_visible_pixels[ray,:]
            ax.plot3D(xs, ys, zs, color=color, linewidth=0.1)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        plt.show()
"""

"""
    def visualize_n_point_clouds(scene, n=None, save=False):
        if type(n) == type(None):
            n = scene.number_of_images
        pcds = []
        for image_number in range(n):
            pcd, xyz_coordinates, image_colors = scene.get_point_cloud(image_number)
            pcds.append(pcd)
        o3d.visualization.draw_geometries(pcds)
        if save:
            for image_number, pcd in enumerate(pcds):
                o3d.io.write_point_cloud("ground_truth_visualization_{}.ply".format(image_number), pcd)
"""