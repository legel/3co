import torch
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix, matrix_to_euler_angles, matrix_to_quaternion, axis_angle_to_quaternion, quaternion_multiply, matrix_to_axis_angle
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


def visualize_n_point_clouds(scene_model, n=None, save=False):
    if type(n) == type(None):
        n = scene_model.number_of_images
    pcds = []
    for image_number in range(n):
        pcd, xyz_coordinates, image_colors = scene_model.get_point_cloud(image_number)
        pcds.append(pcd)
    o3d.visualization.draw_geometries(pcds)
    if save:
        for image_number, pcd in enumerate(pcds):
            o3d.io.write_point_cloud("ground_truth_visualization_{}.ply".format(image_number), pcd)


"""
    def rotate_pose_in_global_space(pose, d_pitch, d_roll, d_yaw):
        pose_rotation_matrix = pose[:3,:3]
        pose_axis_angles = matrix_to_axis_angle(camera_rotation_matrix)
        new_axis_angles = pose_axis_angles + torch.FloatTensor([d_pitch, d_roll, d_yaw])
        new_quaternian = axis_angle_to_quaternion(new_axis_angles)
        new_camera_rotation_matrix = quaternion_to_matrix(new_quaternian)

        new_pose = torch.zeros((4,4))
        new_pose[:3,:3] = new_camera_rotation_matrix
        new_pose[:3,3] = pose[:3,3]
        new_pose[3,3] = 1.0

        return new_pose
"""

def rotate_pose_in_camera_space(scene_model, pose, d_pitch, d_roll, d_yaw):
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
    
def translate_pose_in_global_space(scene_model, pose, d_x, d_y, d_z):        
    camera_rotation_matrix = pose[:3,:3]
    translation = torch.FloatTensor([d_x,d_y,d_z])

    new_pose = torch.zeros((4,4))
    new_pose[:3,:3] = pose[:3,:3]
    new_pose[:3,3] = pose[:3,3] + translation
    new_pose[3,3] = 1.0

    return new_pose

def translate_pose_in_camera_space(scene_model, pose, d_x, d_y, d_z):        
    camera_rotation_matrix = pose[:3,:3]
    transformed_translation = torch.matmul(camera_rotation_matrix, torch.FloatTensor([d_x,d_y,d_z]))           

    new_pose = torch.zeros((4,4))
    new_pose[:3,:3] = pose[:3,:3]
    new_pose[:3,3] = pose[:3,3] + transformed_translation
    new_pose[3,3] = 1.0
                    
    return new_pose

def construct_pose(scene_model, pitch, yaw, roll, x, y, z):
    pose = torch.zeros((4,4))
    pose_quaternion = axis_angle_to_quaternion(torch.FloatTensor([pitch, yaw, roll]))
    pose_rotation_matrix = quaternion_to_matrix(pose_quaternion)
    pose[:3,:3] = pose_rotation_matrix
    pose[:3,3] = torch.FloatTensor([x,y,z])
    pose[3,3] = 1.0

    return pose        

def generate_spin_poses(scene_model, number_of_poses):

    #pose = scene_model.models['pose'](0)[0]
    initial_pose = scene_model.initial_poses[0]
    
    camera_xyz = initial_pose[:3, 3]
    
    # modify the first pose to obtain a good start position for the video
    pose = scene_model.models['pose'](0)[0]
    initial_pose_axis_angles = matrix_to_axis_angle(pose[:3,:3])
    initial_pitch = initial_pose_axis_angles[0]
    initial_yaw = initial_pose_axis_angles[1]
    initial_roll = initial_pose_axis_angles[2] 
    #d_pitch = 1.0 * (-np.pi/2 - initial_pitch + 3.0 * np.pi / 50 + 1.1*np.pi/30)
    d_pitch = np.pi/2
    d_yaw = 0.0
    d_roll = -1.0 * np.pi/10
    d_x = -0.15 #0.05 * 2  # + is move right, - is move left
    d_y = 0.05 * 13 # + is move down, - is move up
    d_z = -0.6 # + is move forward, - is move back        
    rotation = torch.FloatTensor([d_pitch, d_yaw, d_roll])        
    translation = torch.FloatTensor([d_x, d_y, d_z])
    pose = rotate_pose_in_camera_space(scene_model, pose, rotation[0], rotation[1], rotation[2])        
    pose = translate_pose_in_camera_space(scene_model, pose, translation[0], translation[1], translation[2])        
    

    # get the xyz coordinates of the center pixel of the (pre-modified) initial image
    pixel_indices_for_this_image = torch.argwhere(scene_model.image_ids_per_pixel == 0)
    pixel_rows = scene_model.pixel_rows[pixel_indices_for_this_image]
    pixel_cols = scene_model.pixel_cols[pixel_indices_for_this_image]
    rgbd = scene_model.rgbd[torch.squeeze(pixel_indices_for_this_image)].to(scene_model.device)  # (N_pixels, 4)

    sensor_depth = rgbd[:,3].to(scene_model.device) # (N_pixels) 
    center_pixel_row = pixel_rows[int(len(sensor_depth) / 2)]
    center_pixel_col = pixel_cols[int(len(sensor_depth) / 2)]        
    center_pixel_distance = sensor_depth[int(len(sensor_depth) / 2)]     

    camera_rotation_matrix = scene_model.models['pose'](0)[0][:3,:3] 
    ray_dir_world = torch.matmul(camera_rotation_matrix.view(1, 1, 3, 3), scene_model.pixel_directions.unsqueeze(3)).squeeze(3)  # (1, 1, 3, 3) * (H, W, 3, 1) -> (H, W, 3)    
    ray_dir_for_pixel = ray_dir_world[center_pixel_row, center_pixel_col, :] # (3) orientation for this pixel from the camera        

    camera_xyz = scene_model.models['pose'](0)[0][:3,3]
    pixel_xyz = camera_xyz + ray_dir_for_pixel * center_pixel_distance 
    center_pixel_xyz = pixel_xyz[0]
    center_pixel_xyz[0] = center_pixel_xyz[0] + -0.05
    center_pixel_xyz[2] = center_pixel_xyz[0] - 0.15

    pose[1,3] = center_pixel_xyz[1]

    poses = [pose]
    pose_debug = []
    next_cam_pose = np.zeros((4,4))
    next_cam_pose = pose[:4,:4]

    for i in range(0, number_of_poses):
                    
        x = next_cam_pose[0,3]
        y = next_cam_pose[2,3]

        # convert to polar coordinates with center_pixel_xyz as origin
        r = math.dist([pose[0,3],pose[1,3],pose[2,3]], [center_pixel_xyz[0], center_pixel_xyz[1], center_pixel_xyz[2]])                  
        #print(math.dist([x,next_cam_pose[1,3],y], [center_pixel_xyz[0], center_pixel_xyz[1], center_pixel_xyz[2]]) )
        theta = torch.atan2(torch.FloatTensor([y - center_pixel_xyz[2]]),torch.FloatTensor([x - center_pixel_xyz[0]]))       
        #print(theta)     

        # rotate
        theta = theta + 2.0*np.pi/number_of_poses

        # convert back to cartesian coordinates
        xp = r * math.cos(theta) + center_pixel_xyz[0]
        yp = r * math.sin(theta) + center_pixel_xyz[2]

        # translate then rotate
        next_cam_pose = translate_pose_in_global_space(scene_model, next_cam_pose, (xp - x), 0.0, (yp - y))
        next_cam_pose = rotate_pose_in_camera_space(scene_model, next_cam_pose, 0.0, 1.0 * 2.0 * np.pi / number_of_poses, 0.0)            

        pose_debug.append([next_cam_pose[0,3], next_cam_pose[2,3], next_cam_pose[1,3]])                        
        poses.append(next_cam_pose)            

    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #center_pixel_xyz = center_pixel_xyz.cpu()
    #ax.scatter([x[0] for x in pose_debug], [x[1] for x in pose_debug], [x[2] for x in pose_debug])
    #ax.scatter([center_pixel_xyz[0]], [center_pixel_xyz[2]], [center_pixel_xyz[1]], marker='o')
    #plt.show()

    poses = torch.stack(poses, 0)
    poses = convert3x4_4x4(poses).to(scene_model.device)

    return poses


def generate_zoom_poses(scene_model, pose, center_pixel_distance, center_pixel_row, center_pixel_col, sphere_angle, number_of_poses):
    camera_rotation_matrix = pose[:3, :3] # rotation matrix (3,3)
    camera_xyz = pose[:3, 3]  # translation vector (3)

    # transform rays from camera coordinate to world coordinate
    ray_dir_world = torch.matmul(camera_rotation_matrix.view(1, 1, 3, 3), scene_model.pixel_directions.unsqueeze(3)).squeeze(3)  # (1, 1, 3, 3) * (H, W, 3, 1) -> (H, W, 3)    
    ray_dir_for_pixel = ray_dir_world[center_pixel_row, center_pixel_col, :] # (3) orientation for this pixel from the camera

    pixel_xyz = camera_xyz + ray_dir_for_pixel * center_pixel_distance 
    pixel_xyz = pixel_xyz[0]

    pixel_x = pixel_xyz[0]
    pixel_y = pixel_xyz[1]
    pixel_z = pixel_xyz[2]

    # By subtracting the pixel_xyz from itself, and the camera_xyz, then we have the pixel_xyz become the origin
    cam_zoom_direction_xyz = camera_xyz - pixel_xyz
    zoom_distance_x = cam_zoom_direction_xyz[0]
    zoom_distance_y = cam_zoom_direction_xyz[1]
    zoom_distance_z = cam_zoom_direction_xyz[2]

    # define number_of_poses along a path from the original camera pose zooming into the object
    poses = []
    for i, zoom_percent in enumerate(np.linspace(0, 0.75, number_of_poses)):
        # then convert back to euclidian coordinates
        next_cam_x = camera_xyz[0] - zoom_distance_x * zoom_percent
        next_cam_y = camera_xyz[1] - zoom_distance_y * zoom_percent
        next_cam_z = camera_xyz[2] - zoom_distance_z * zoom_percent

        next_cam_pose = torch.zeros((3,4))
        next_cam_pose[:3,:3] = camera_rotation_matrix
        next_cam_pose[:3,3] = torch.tensor([next_cam_x, next_cam_y, next_cam_z])

        poses.append(next_cam_pose)

    poses = torch.stack(poses, 0)
    poses = convert3x4_4x4(poses).to(scene_model.device)

    return poses


def imgs_to_video():
    cimgs = []
    dimgs = []
    for i in range(0, 121):
        fname1 = 'data/temp2/spin_video/color/color_{}.png'.format(i)
        fname2 = 'data/temp2/spin_video/depth/depth_{}.png'.format(i)
        cimg = Image.open(fname1)
        dimg = Image.open(fname2)
        cimgs.append(cimg)
        dimgs.append(dimg)
    imageio.mimwrite('color.mp4', cimgs, fps=15, quality=9)        
    imageio.mimwrite('depth.mp4', dimgs, fps=15, quality=9)        

def create_spin_video(scene_model, number_of_poses, output_dir):

    poses = generate_spin_poses(scene_model, number_of_poses)
    create_video_from_poses(scene_model, poses, output_dir)


def create_video_from_poses(scene_model, poses, output_dir):

    color_images = []
    depth_images = []

    for i,pose in enumerate(poses):
        render_result = scene_model.render_prediction(pose)
        color_out_file_name = '{}/color/color_{}.png'.format(output_dir,i).zfill(4)
        depth_out_file_name = '{}/depth/depth_{}.png'.format(output_dir,i).zfill(4)        
        scene_model.save_render_as_png(render_result, color_out_file_name, depth_out_file_name)

        rendered_color_for_file = (render_result['rendered_image'].cpu().numpy() * 255).astype(np.uint8)    
        rendered_depth_data = render_result['rendered_depth'].cpu().numpy() 
        rendered_depth_for_file = heatmap_to_pseudo_color(rendered_depth_data)
        rendered_depth_for_file = (rendered_depth_for_file * 255).astype(np.uint8)     
        color_images.append(rendered_color_for_file)   
        depth_images.append(rendered_depth_for_file)

    imageio.mimwrite(os.path.join(output_dir, 'color.mp4'), color_images, fps=15, quality=9)
    imageio.mimwrite(os.path.join(output_dir, 'depth.mp4'), depth_images, fps=15, quality=9)



def render_all_training_images(scene_model):
    

    for image_id in range(0, 157):
        print("rendering image {}".format(image_id))
        color_out_file_name = 'data/temp2/color/color_{}.png'.format(image_id).zfill(4)
        depth_out_file_name = 'data/temp2/depth/depth_{}.png'.format(image_id).zfill(4)
        render_result = scene_model.render_prediction_for_train_image(image_id)
        scene_model.save_render_as_png(render_result, color_out_file_name, depth_out_file_name)
        #rendered_rgb = render_result['rendered_image'].reshape(scene_model.H, scene_model.W, 3)
        #rendered_depth = render_result['rendered_depth'].reshape(scene_model.H, scene_model.W)
        #rendered_color_image = (rendered_rgb.cpu().numpy() * 255).astype(np.uint8)               



def density_visualization(scene_model):

    test_image_id = 0
    test_pixel_row = 240
    test_pixel_col = 320

    # original image
    image, image_name = scene_model.load_image_data(image_id=test_image_id)
    print(image.size())
    print(image_name)        

    # rendered image
    out_file_name = 'data/pillow_small/density_visualization/'
    render_result = scene_model.render_prediction_for_train_image(test_image_id)
    rendered_rgb = render_result['rendered_image'].reshape(scene_model.H, scene_model.W, 3)
    rendered_depth = render_result['rendered_depth'].reshape(scene_model.H, scene_model.W)

    rendered_color_image = (rendered_rgb.cpu().numpy() * 255).astype(np.uint8)       

    plt.imshow(rendered_color_image)
    plt.show()
            
    # density vector
    density_row = 200
    density_col = 350
    density_pixel_index = density_col + scene_model.H * density_row
    #x = squeezed_density[density_pixel_index].cpu()
    #plt.scatter([i for i in range(0,len(x))], [x for x in x], s=10)
    #plt.show()        


def debug_visualization(scene_model, rgb, xyz, depth_weights, rendered_image, downsample_ratio=50):
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


if __name__ == '__main__':
    # Load a scene object with all data and parameters
    scene = SceneModel(args=parse_args(), load_saved_args=True)
    
    
    with torch.no_grad():
        #scene.test()
        #density_visualization(scene)
        #create_spin_video(scene, 120, 'data/temp2/spin_video')
        #imgs_to_video()
        render_all_training_images(scene)




















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