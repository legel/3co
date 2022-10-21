import torch
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix, matrix_to_euler_angles, matrix_to_quaternion, axis_angle_to_quaternion, quaternion_multiply, matrix_to_axis_angle
from scipy.spatial.transform import Rotation
from torchsummary import summary
import cv2
import open3d as o3d
from torch_cluster import grid_cluster
from pytorch3d.ops.knn import knn_points
from pytorch3d.io import IO
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
from utils.pos_enc import encode_position, encode_ipe
from utils.volume_op import volume_sampling, volume_rendering
from utils.lie_group_helper import convert3x4_4x4
from utils.training_utils import PolynomialDecayLearningRate, heatmap_to_pseudo_color, set_randomness, save_checkpoint
from models.intrinsics import CameraIntrinsicsModel
from models.poses import CameraPoseModel
from models.nerf_models import NeRFDensity, NeRFColor

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

set_randomness()

iterations_per_batch = 1

def parse_args():
    parser = argparse.ArgumentParser()

    # Define path to relevant data for training, and decide on number of images to use in training
    parser.add_argument('--base_directory', type=str, default='./data/dragon_scale', help='The base directory to load and save information from')
    parser.add_argument('--images_directory', type=str, default='color', help='The specific group of images to use during training')
    parser.add_argument('--images_data_type', type=str, default='jpg', help='Whether images are jpg or png')
    parser.add_argument('--skip_every_n_images_for_training', type=int, default=30, help='When loading all of the training data, ignore every N images')
    parser.add_argument('--save_models_frequency', type=int, default=25000, help='Save model every this number of epochs')
    parser.add_argument('--load_pretrained_models', type=bool, default=False, help='Whether to start training from models loaded with load_pretrained_models()')
    parser.add_argument('--pretrained_models_directory', type=str, default='./data/dragon_scale/hyperparam_experiments/1662755515_depth_loss_0.00075_to_0.0_k9_N1_NeRF_Density_LR_0.0005_to_0.0001_k4_N1_pose_LR_0.0025_to_1e-05_k9_N1', help='The directory storing models to load')

    # Define number of epochs, and timing by epoch for when to start training per network
    parser.add_argument('--start_epoch', default=0, type=int, help='Epoch on which to begin or resume training')
    parser.add_argument('--number_of_epochs', default=200001, type=int, help='Number of epochs for training, used in learning rate schedules')    
    parser.add_argument('--start_training_extrinsics_epoch', type=int, default=500, help='Set to epoch number >= 0 to init poses using estimates from iOS, and start refining them from this epoch.')
    parser.add_argument('--start_training_intrinsics_epoch', type=int, default=5000, help='Set to epoch number >= 0 to init focals using estimates from iOS, and start refining them from this epoch.')
    parser.add_argument('--start_training_color_epoch', type=int, default=0, help='Set to a epoch number >= 0 to start learning RGB NeRF on top of density NeRF.')
    parser.add_argument('--start_training_geometry_epoch', type=int, default=0, help='Set to a epoch number >= 0 to start learning RGB NeRF on top of density NeRF.')

    # Define evaluation/logging/saving frequency and parameters
    parser.add_argument('--test_frequency', default=100, type=int, help='Frequency of epochs to render an evaluation image')    
    parser.add_argument('--save_point_cloud_frequency', default=200002, type=int, help='Frequency of epochs to save point clouds')
    parser.add_argument('--save_depth_weights_frequency', default=200002, type=int, help='Frequency of epochs to save density depth weight visualizations')
    parser.add_argument('--log_frequency', default=1, type=int, help='Frequency of epochs to log outputs e.g. loss performance')        
    parser.add_argument('--number_of_test_images', default=2, type=int, help='Index in the training data set of the image to show during testing')
    parser.add_argument('--skip_every_n_images_for_testing', default=80, type=int, help='Skip every Nth testing image, to ensure sufficient test view diversity in large data set')    
    parser.add_argument('--number_of_pixels_per_batch_in_test_renders', default=128, type=int, help='Size in pixels of each batch input to rendering')
    parser.add_argument('--show_debug_visualization_in_testing', default=False, type=bool, help='Whether or not to show the cool Matplotlib 3D view of rays + weights + colors')
    parser.add_argument('--export_test_data_for_post_processing', default=False, type=bool, help='Whether to save in external files the final render RGB + weights for all samples for all images')
    parser.add_argument('--save_ply_point_clouds_of_sensor_data', default=False, type=bool, help='Whether to save a .ply file at start of training showing the initial projected sensor data in 3D global coordinates')
    parser.add_argument('--save_ply_point_clouds_of_sensor_data_with_learned_poses', default=False, type=bool, help='Whether to save a .ply file after loading a pre-trained model to see how the sensor data projects to 3D global coordinates with better poses')

    # Define learning rates, including start, stop, and two parameters to control curvature shape (https://arxiv.org/pdf/2004.05909v1.pdf)
    parser.add_argument('--nerf_density_lr_start', default=0.0030, type=float, help="Learning rate start for NeRF geometry network")
    parser.add_argument('--nerf_density_lr_end', default=0.0001, type=float, help="Learning rate end for NeRF geometry network")
    parser.add_argument('--nerf_density_lr_exponential_index', default=7, type=int, help="Learning rate speed of exponential decay (higher value = faster initial decay) for NeRF geometry network")
    parser.add_argument('--nerf_density_lr_curvature_shape', default=1, type=int, help="Learning rate shape of decay (lower value = faster initial decay) for NeRF geometry network")

    parser.add_argument('--nerf_color_lr_start', default=0.0030, type=float, help="Learning rate start for NeRF RGB (pitch,yaw) network")
    parser.add_argument('--nerf_color_lr_end', default=0.0001, type=float, help="Learning rate end for NeRF RGB (pitch,yaw) network")
    parser.add_argument('--nerf_color_lr_exponential_index', default=7, type=int, help="Learning rate speed of exponential decay (higher value = faster initial decay) for NeRF RGB (pitch,yaw) network")
    parser.add_argument('--nerf_color_lr_curvature_shape', default=1, type=int, help="Learning rate shape of decay (lower value = faster initial decay) for NeRF RGB (pitch,yaw) network")

    parser.add_argument('--focal_lr_start', default=0.00100, type=float, help="Learning rate start for NeRF-- camera intrinsics network")
    parser.add_argument('--focal_lr_end', default=0.00001, type=float, help="Learning rate end for NeRF-- camera intrinsics network")
    parser.add_argument('--focal_lr_exponential_index', default=9, type=int, help="Learning rate speed of exponential decay (higher value = faster initial decay) for NeRF-- camera intrinsics network")
    parser.add_argument('--focal_lr_curvature_shape', default=1, type=int, help="Learning rate shape of decay (lower value = faster initial decay) for NeRF-- camera intrinsics network")

    parser.add_argument('--pose_lr_start', default=0.00100, type=float, help="Learning rate start for NeRF-- camera extrinsics network")
    parser.add_argument('--pose_lr_end', default=0.00001, type=float, help="Learning rate end for NeRF-- camera extrinsics network")
    parser.add_argument('--pose_lr_exponential_index', default=9, type=int, help="Learning rate speed of exponential decay (higher value = faster initial decay) for NeRF-- camera extrinsics network")
    parser.add_argument('--pose_lr_curvature_shape', default=1, type=int, help="Learning rate shape of decay (lower value = faster initial decay) for NeRF-- camera extrinsics network")

    parser.add_argument('--depth_to_rgb_loss_start', default=0.0010, type=float, help="Learning rate start for ratio of loss importance between depth and RGB inverse rendering loss")
    parser.add_argument('--depth_to_rgb_loss_end', default=0.00000, type=float, help="Learning rate end for ratio of loss importance between depth and RGB inverse rendering loss")
    parser.add_argument('--depth_to_rgb_loss_exponential_index', default=9, type=int, help="Learning rate speed of exponential decay (higher value = faster initial decay) for ratio of loss importance between depth and RGB inverse rendering loss")
    parser.add_argument('--depth_to_rgb_loss_curvature_shape', default=1, type=int, help="Learning rate shape of decay (lower value = faster initial decay) for ratio of loss importance between depth and RGB inverse rendering loss")    

    parser.add_argument('--entropy_loss_tuning_start_epoch', type=float, default=1000000, help='epoch to start entropy loss tuning')
    parser.add_argument('--entropy_loss_tuning_end_epoch', type=float, default=1000000, help='epoch to end entropy loss tuning')
    parser.add_argument('--use_sparse_rendering_for_test_renders', type=bool, help='Whether or not to use sparse rendering technique in test renders')
    parser.add_argument('--use_sparse_fine_rendering_for_test_renders', type=bool, help='Whether or not to use sparse fine rendering technique in test renders')
    

    # Define parameters the determines the overall size and learning capacity of the neural networks and their encodings
    parser.add_argument('--density_neural_network_parameters', type=int, default=512, help='The baseline number of units that defines the size of the NeRF geometry network')
    parser.add_argument('--color_neural_network_parameters', type=int, default=512, help='The baseline number of units that defines the size of the NeRF RGB (pitch,yaw) network')
    parser.add_argument('--positional_encoding_fourier_frequencies', type=int, default=10, help='The number of frequencies that are generated for positional encoding of (x,y,z)')
    parser.add_argument('--directional_encoding_fourier_frequencies', type=int, default=10, help='The number of frequencies that are generated for positional encoding of (pitch, yaw)')

    # Define sampling parameters, including how many samples per raycast (outward), number of samples randomly selected per image, and (if masking is used) ratio of good to masked samples
    parser.add_argument('--pixel_samples_per_epoch', type=int, default=500, help='The number of rows of samples to randomly collect for each image during training')
    parser.add_argument('--number_of_samples_outward_per_raycast', type=int, default=1000, help='The number of samples per raycast to collect (linearly)')    
    parser.add_argument('--positional_encoding', type=str, default='nerf', help='Type of positional encoding to be used: nerf or mip')

    # Define depth sensor parameters
    parser.add_argument('--depth_sensor_error', type=float, default=0.5, help='If not using a heuristic for variance, hardcoded variance of Gaussian depth sensor model, in millimeters')
    parser.add_argument('--epsilon', type=float, default=0.0000001, help='Minimum value in log() for NeRF density weights going to 0')    
    parser.add_argument('--number_of_nearest_neighbors_to_use_in_knn_distance_metric_for_estimation_of_sensor_error', default=10, type=int, help='N for the KNN on every 3D point at start of optimization, of which distances for N points are used as metric of variance')    

    # Additional parameters on pre-processing of depth data and coordinate systems
    parser.add_argument('--maximum_depth', type=float, default=5.0, help='All depths below this value will be clipped to this value')
    parser.add_argument('--min_confidence', type=float, default=2.0, help='A value in [0,1,2] where 0 allows all depth data to be used, 2 filters the most and ignores that')

    # Depth sampling optimizations
    parser.add_argument('--n_depth_sampling_optimizations', type=int, default=2, help='For every epoch, for every set of pixels, do this many renders to find the best depth sampling distances')

    parsed_args = parser.parse_args()

    # Save parameters with Weights & Biases log
    wandb.init(project="nerf--", entity="3co", config=parsed_args)

    return parser.parse_args()


class SceneModel:
    def __init__(self, args, load_saved_args = False):

    
        self.args = args

        if load_saved_args:
            print("loading test args")
            self.load_saved_args_train()                    

        # initialize high-level arguments        
        self.epoch = self.args.start_epoch
        self.start_time = int(time.time()) 
        self.device = torch.device('cuda:0') 
        self.gpu2 = torch.device('cuda:1')        

        # # set cache directory
        os.environ['PYTORCH_KERNEL_CACHE_PATH'] = self.args.base_directory

        # set up location for saving experiment data
        self.create_experiment_directory()        
        self.save_experiment_parameters()

        # load all unique IDs (names without ".png") of images to self.image_ids
        self.load_all_images_ids()        

        # get camera intrinsics (same for all images)
        self.load_camera_intrinsics()

        # get camera extrinsics (for each image)
        self.load_camera_extrinsics()

        # define bounds (self.min_x, self.max_x), (self.min_y, self.max_y), (self.min_z, self.max_z) in which all points should initially project inside, or else not be included
        self.set_xyz_bounds_from_crop_of_image(index_to_filter=0, min_pixel_row=0, max_pixel_row=self.H, min_pixel_col=0, max_pixel_col=self.W) # pre-trained model with min_pixel_row=200, max_pixel_row=400, min_pixel_col=250, max_pixel_col=450; tight-clipping for small image is min_pixel_row=110, max_pixel_row=390, min_pixel_col=170, max_pixel_col=450

        # prepare test evaluation indices
        self.prepare_test_data()

        # now load only the necessary data that falls within bounds defined
        self.load_image_and_depth_data_within_xyz_bounds()
        
        # initialize all models        
        self.initialize_models()        
        self.initialize_learning_rates()  

        if self.args.export_test_data_for_post_processing:
            self.save_cam_xyz()        

        if self.args.load_pretrained_models:
            print("Loading pretrained models")
            # load pre-trained model
            self.load_pretrained_models()

            # compute the ray directions using the latest focal lengths
            focal_length_x, focal_length_y = self.models["focal"](0)
            self.compute_ray_direction_in_camera_coordinates(focal_length_x, focal_length_y)            
        else:            
            print("Training from scratch")
            

    #########################################################################
    ################ Loading and initial processing of data #################
    #########################################################################

    def prepare_test_data(self):
        self.test_image_indices = range(0, self.args.number_of_test_images * self.args.skip_every_n_images_for_testing, self.args.skip_every_n_images_for_testing)
        print("Test image indices are: {}".format([i for i in self.test_image_indices]))

    def load_all_images_ids(self):
        # get images in directory of RGB images
        path_to_images = "{}/{}".format(self.args.base_directory, self.args.images_directory)
        unsorted_image_names = glob.glob("{}/*.{}".format(path_to_images, self.args.images_data_type))

        # extract out numbers of their IDs, and sort images by numerical ID
        self.image_ids = np.asarray(sorted([int(image.split("/")[-1].replace(".{}".format(self.args.images_data_type),"")) for image in unsorted_image_names]))
        self.n_training_images = len(self.image_ids[::self.args.skip_every_n_images_for_training])

    def load_image_data(self, image_id):
        """
        Read the data from the Stray Scanner (https://docs.strayrobots.io/apps/scanner/format.html).
        Do this after running convert_to_open3d.py in order to get images, as well as camera_intrinsics.json
        """
        # recreate the image name
        image_name = "{}.{}".format(str(int(image_id)).zfill(6), self.args.images_data_type)

        # get image path in folder
        path_to_images = "{}/{}".format(self.args.base_directory, self.args.images_directory)
        image_path = os.path.join(path_to_images, image_name)

        # load image data, and collect indices of pixels that may be masked using alpha channel if image is a .png 
        image_data = imageio.imread(image_path)

        # clip out alpha channel, if it exists
        image_data = image_data[:, :, :3]  # (H, W, 3)

        # convert to torch format and normalize between 0-1.0
        image = torch.from_numpy(image_data).float() / 255 # (H, W, 3) torch.float32
        
        return image.to(device=self.device), image_name


    def load_depth_data(self, image_id):
        # load depth data from LIDAR systems

        # get folders to depth and confidence data
        depth_folder = os.path.join(self.args.base_directory, 'depth')
        confidence_folder = os.path.join(self.args.base_directory, 'confidence')

        # get paths to depth and confidence data
        confidence_path = os.path.join(confidence_folder, f'{image_id:06}.png')
        depth_path = os.path.join(depth_folder, f'{image_id:06}.png')

        # load confidence data, which Apple provides with only three possible confidence metrics: 0 (least confident), 1 (moderate confidence), 2 (most confident)
        confidence_data = np.array(Image.open(confidence_path))

        # we're now going to interpolate the confidence data, such that we will be able to have estimated confidence values for interpolated depths
        confidence_data = cv2.resize(confidence_data, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        confidence_data = torch.from_numpy(confidence_data).to(device=self.device)

        # read the 16 bit greyscale depth data which is formatted as an integer of millimeters
        depth_mm = cv2.imread(depth_path, -1).astype(np.float32)

        # convert data in millimeters to meters
        depth_m = depth_mm / (1000.0)  
        
        # set a cap on the maximum depth in meters; clips erroneous/irrelevant depth data from way too far out
        depth_m[depth_m > self.args.maximum_depth] = self.args.maximum_depth

        # resize to a resolution that e.g. may be higher, and equivalent to image data
        # to do: incorporate lower confidence into the interpolated depth metrics!
        resized_depth_meters = cv2.resize(depth_m, (self.W, self.H), interpolation=cv2.INTER_LINEAR)

        # NeRF requires bounds which can be used to constrain both the processed coordinate system data, as well as the ray sampling
        # for fast look-up, define nearest and farthest depth measured per image
        near_bound = np.min(resized_depth_meters)
        far_bound = np.max(resized_depth_meters)

        #depth_m[depth_m >= self.args.maximum_depth] = self.args.maximum_depth - 1.0 / self.args.number_of_samples_outward_per_raycast

        depth = torch.Tensor(resized_depth_meters).to(device=self.device) # (N_images, H_image, W_image)

        return depth, near_bound, far_bound, confidence_data


    def load_camera_intrinsics(self):
        # load camera instrinsics estimates from Apple's internal API
        camera_intrinsics_data = json.load(open(os.path.join(self.args.base_directory,'camera_intrinsics.json')))
        self.H = int(camera_intrinsics_data["height"])
        self.W = int(camera_intrinsics_data["width"])

        print("Image with size (H,W)=({},{})".format(self.H, self.W))

        # save camera intrinsics matrix for future learning
        camera_intrinsics_matrix = np.zeros(shape=(3,3))
        camera_intrinsics_matrix[0,0] = camera_intrinsics_data["intrinsic_matrix"][0] # fx (focal length)
        camera_intrinsics_matrix[1,1] = camera_intrinsics_data["intrinsic_matrix"][4] # fy (focal length)
        camera_intrinsics_matrix[0,2] = camera_intrinsics_data["intrinsic_matrix"][6] # ox (principal point) 320   W = 640
        camera_intrinsics_matrix[1,2] = camera_intrinsics_data["intrinsic_matrix"][7] # oy (principal point) 240   H = 480
        camera_intrinsics_matrix[2,2] = camera_intrinsics_data["intrinsic_matrix"][8] # 1.0

        self.camera_intrinsics = torch.Tensor(camera_intrinsics_matrix).to(device=self.device)

        self.initial_focal_length_x = self.camera_intrinsics[0,0].repeat(self.n_training_images)
        self.initial_focal_length_y = self.camera_intrinsics[1,1].repeat(self.n_training_images)

        self.principal_point_x = self.camera_intrinsics[0,2]
        self.principal_point_y = self.camera_intrinsics[1,2]
        
        self.compute_ray_direction_in_camera_coordinates(focal_length_x=self.initial_focal_length_x, focal_length_y=self.initial_focal_length_y)


    def load_camera_extrinsics(self):
        # load odometry data which includes estimates from Apple's ARKit in the form of: timestamp, frame, x, y, z, qx, qy, qz, qw
        poses = []
        odometry = np.loadtxt(os.path.join(self.args.base_directory, 'odometry.csv'), delimiter=',', skiprows=1)
        for line_index,line in enumerate(odometry):
            # unpack (x,y,z) data as a 1x3 translation vector
            translation_vector = np.asarray(line[2:5])

            # unpack quaternion data and convert it to a 3x3 rotation matrix
            qx=line[5]
            qy=line[6]
            qz=line[7]
            qw=line[8]
            r = float(qw)
            i = float(qx)
            j = float(qy)
            k = float(qz)
            quaternion = torch.tensor([r,i,j,k])
            rotation_matrix = quaternion_to_matrix(quaternion) # Rotation.from_quat(quaternion).as_matrix()

            # define pose initially as a 4x3 matrix
            pose = np.eye(4)
            pose[:3, :3] = rotation_matrix
            pose[:3, 3] = translation_vector

            poses.append(pose)

        poses = np.asarray(poses)
        rotations_translations = poses[:,:3,:] # get rotations and translations from the 4x4 matrix in a 4x3 matrix

        self.initial_poses = torch.Tensor(convert3x4_4x4(rotations_translations)).to(device=self.device) # (N, 4, 4)


    def create_point_cloud(self, xyz_coordinates, colors, label=0, flatten_xyz=True, flatten_image=True):
        pcd = o3d.geometry.PointCloud()
        if flatten_xyz:
            xyz_coordinates = torch.flatten(xyz_coordinates, start_dim=0, end_dim=1).cpu().detach().numpy()
        else:
            xyz_coordinates = xyz_coordinates.cpu().detach().numpy()
        if flatten_image:
            colors = torch.flatten(colors, start_dim=0, end_dim=1).cpu().detach().numpy()
        else:
            colors = colors.cpu().detach().numpy()

        pcd.points = o3d.utility.Vector3dVector(xyz_coordinates)

        # Load saved point cloud and visualize it
        pcd.estimate_normals()
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd


    def get_point_cloud(self, pose, depth, rgb, pixel_directions, label=0, save=False, remove_zero_depths=True, save_raw_xyz=False, dir='', entropy_image=None, unsparse_rendered_rgb_img=None, sparse_rendered_rgb_img=None, unsparse_depth=None):        

        camera_world_position = pose[:3, 3].view(1, 1, 1, 3)     # (1, 1, 1, 3)
        camera_world_rotation = pose[:3, :3].view(1, 1, 1, 3, 3) # (1, 1, 1, 3, 3)
        pixel_directions = pixel_directions.unsqueeze(3) # (H, W, 3, 1)        

        xyz_coordinates = self.derive_xyz_coordinates(camera_world_position, camera_world_rotation, pixel_directions, depth)
        
        xyz_coordinates = xyz_coordinates
        if save_raw_xyz:
            file_path = "{}/{}_xyz_raw.npy".format(dir, label)
            with open(file_path, "wb") as f:
                np.save(f, xyz_coordinates.cpu().detach().numpy())


        pixel_directions = pixel_directions.squeeze(3)        
        r = torch.sqrt(torch.sum(  (pixel_directions[60, 80] - pixel_directions[240, 320])**2, dim=0))                                                                

        depth_condition = (depth!=0.0).to(self.device)
        angle_condition = (torch.sqrt( torch.sum( (pixel_directions - pixel_directions[240, 320, :])**2, dim=2 ) ) < r).to(self.device)
        entropy_condition = (entropy_image < 1000.0).to(self.device)

        joint_condition = torch.logical_and(depth_condition, angle_condition).to(self.device)        
        joint_condition = torch.logical_and(joint_condition, entropy_condition).to(self.device)        

        sparse_depth = depth
        unsparse_depth = unsparse_depth

        if self.args.use_sparse_fine_rendering_for_test_renders:            
            sticker_condition = torch.abs(unsparse_depth - sparse_depth) < (self.far-self.near) / ( (self.args.number_of_samples_outward_per_raycast+1) * 4.0)
            joint_condition = torch.logical_and(joint_condition, sticker_condition)

        joint_condition_indices = torch.where(joint_condition)

        n_filtered_points = self.H * self.W - torch.sum(joint_condition.flatten())

        depth = depth[joint_condition_indices]
        rgb = rgb[joint_condition_indices]
        xyz_coordinates = xyz_coordinates[joint_condition_indices]

        print("filtered {}/{} points".format(n_filtered_points, self.W*self.H))

        pcd = self.create_point_cloud(xyz_coordinates, rgb, label="point_cloud_{}".format(label), flatten_xyz=False, flatten_image=False)
        if save:
            file_name = "{}/view_{}_training_data_{}.ply".format(dir, label, self.epoch-1)
            o3d.io.write_point_cloud(file_name, pcd)

        return pcd


    def visualize_mask(self, pixels_to_visualize, mask_index, colors=None):
        
        if type(colors) == type(None):
            filtered_mask = torch.where(pixels_to_visualize, 255, 0).cpu().numpy().astype(np.uint8)
        else:
            colors = (colors * 255).to(torch.long)
            filtered_r = torch.where(pixels_to_visualize, colors[:,:,0], 0) 
            filtered_g = torch.where(pixels_to_visualize, colors[:,:,1], 0) 
            filtered_b = torch.where(pixels_to_visualize, colors[:,:,2], 0) 
            filtered_mask = torch.stack([filtered_r,filtered_g,filtered_b], dim=2).cpu().numpy().astype(np.uint8)

        color_out_path = Path("{}/mask_for_filtering_{}.png".format(self.experiment_dir, mask_index))
        imageio.imwrite(color_out_path, filtered_mask)


    def save_cam_xyz(self):
        number_of_test_images_saved = 0
        # now loop through all of the data, and filter out (only load and save as necessary) based on whether the points land within our focus area
        for i, image_id in enumerate(self.image_ids[::self.args.skip_every_n_images_for_training]):
            if i in self.test_image_indices:
                poses = self.models["pose"](0)
                pose = poses[i, :, :]
                cam_xyz = pose[:3,3].cpu().detach().numpy()
                # print("Saving camera (x,y,z) for test pose {}: ({:.4f},{:.4f},{:.4f})".format(self.test_poses_processed, cam_xyz[0], cam_xyz[1], cam_xyz[2]))
                cam_xyz_file = "{}/cam_xyz_{}.npy".format(self.args.base_directory,i)
                with open(cam_xyz_file, "wb") as f:
                    np.save(f, cam_xyz)
                self.test_poses_processed += 1


    def get_xyz_inside_range(self, xyz_coordinates):
        x_coordinates = xyz_coordinates[:,:,0]
        y_coordinates = xyz_coordinates[:,:,1]
        z_coordinates = xyz_coordinates[:,:,2]

        x_inside_range = torch.logical_and(x_coordinates >= self.min_x, x_coordinates <= self.max_x)
        y_inside_range = torch.logical_and(y_coordinates >= self.min_y, y_coordinates <= self.max_y)
        z_inside_range = torch.logical_and(z_coordinates >= self.min_z, z_coordinates <= self.max_z)

        x_and_y_inside_range = torch.logical_and(x_inside_range, y_inside_range)
        y_and_z_inside_range = torch.logical_and(y_inside_range, z_inside_range)
        x_and_z_inside_range = torch.logical_and(x_inside_range, z_inside_range)

        xyz_inside_range = torch.logical_and(torch.logical_and(x_and_y_inside_range, y_and_z_inside_range), x_and_z_inside_range)

        return xyz_inside_range.to(device=self.device)


    def load_image_and_depth_data_within_xyz_bounds(self, visualize_masks=True):
        self.rgbd = []
        self.image_ids_per_pixel = []
        self.pixel_rows = []
        self.pixel_cols = []
        self.xyz_per_view = []
        self.confidence_per_pixel = []
        self.test_poses_processed = 0
        number_of_test_images_saved = 0        

        # now loop through all of the data, and filter out (only load and save as necessary) based on whether the points land within our focus area
        for i, image_id in enumerate(self.image_ids[::self.args.skip_every_n_images_for_training]):

            # get depth data for this image
            depth, near_bound, far_bound, confidence = self.load_depth_data(image_id=image_id) # (H, W)

            # get (x,y,z) coordinates for this image
            xyz_coordinates = self.get_sensor_xyz_coordinates(pose_data=self.initial_poses[i*self.args.skip_every_n_images_for_training], depth_data=depth, i=i) # (H, W, 3)            
            
            # now, reverse engineer which pixel coordinates are inside our convex hull of attention
            xyz_coordinates_on_or_off = self.get_xyz_inside_range(xyz_coordinates) # (H, W, 3) with True if (x,y,z) inside of previously set bounds, False if outside
            
            # get the indices of the pixel rows and pixel columns where the projected (x,y,z) point is inside the target convex hull region            
            pixel_indices_selected = torch.argwhere(xyz_coordinates_on_or_off)          

            pixel_rows_selected = pixel_indices_selected[:,0]
            pixel_cols_selected = pixel_indices_selected[:,1]
            self.pixel_rows.append(pixel_rows_selected)
            self.pixel_cols.append(pixel_cols_selected)

            # save general pixel indices one time here for rendering of all pixels, later
            if i == 0:
                all_pixel_indices = torch.argwhere(xyz_coordinates[:,:,0] >= torch.tensor(-np.inf))
                self.all_pixel_rows = all_pixel_indices[:,0]
                self.all_pixel_cols = all_pixel_indices[:,1]

            # get the corresponding (x,y,z) coordinates and depth values selected by the mask
            xyz_coordinates_selected = xyz_coordinates[pixel_rows_selected, pixel_cols_selected, :]
            self.xyz_per_view.append(xyz_coordinates_selected)

            # get the confidence of the selected pixels
            # confidence = confidence.cpu()
            selected_confidence = confidence[pixel_rows_selected, pixel_cols_selected]
            self.confidence_per_pixel.append(selected_confidence)

            depth_selected = depth[pixel_rows_selected, pixel_cols_selected] # (N selected)

            # now, load the (r,g,b) image and filter the pixels we're only focusing on
            image, image_name = self.load_image_data(image_id=image_id)
            rgb_selected = image[pixel_rows_selected, pixel_cols_selected, :] # (N selected, 3)

            # concatenate the (R,G,B) data with the Depth data to create a RGBD vector for each pixel
            rgbd_selected = torch.cat([rgb_selected, depth_selected.view(-1, 1)], dim=1)
            self.rgbd.append(rgbd_selected)

            # now, save this image index, multiplied by the number of pixels selected, in a global vector across all images 
            number_of_selected_pixels = torch.sum(xyz_coordinates_on_or_off)
            image_id_for_all_pixels = torch.full(size=[number_of_selected_pixels], fill_value=i)
            self.image_ids_per_pixel.append(image_id_for_all_pixels)

            # we need indices of the selected rows and cols for every image for post-processing
            if i in self.test_image_indices and self.args.export_test_data_for_post_processing:
                pixel_rows_to_save = pixel_rows_selected.cpu().numpy()
                pixel_cols_to_save = pixel_cols_selected.cpu().numpy()
                pixel_rows_file = "{}/selected_pixel_rows_{}.npy".format(self.args.base_directory,i)
                pixel_cols_file = "{}/selected_pixel_cols_{}.npy".format(self.args.base_directory,i)
                with open(pixel_rows_file, "wb") as f:
                    np.save(f, pixel_rows_to_save)
                with open(pixel_cols_file, "wb") as f:
                    np.save(f, pixel_cols_to_save)

            # script for visualizing mask
            if visualize_masks and i in self.test_image_indices:
                self.visualize_mask(pixels_to_visualize=xyz_coordinates_on_or_off, mask_index=i, colors=image)

            if self.args.save_ply_point_clouds_of_sensor_data and i in self.test_image_indices:
                pcd = self.get_point_cloud(dir=self.experiment_dir, pose=self.initial_poses[i*self.args.skip_every_n_images_for_training], depth=depth, rgb=image, pixel_directions=self.pixel_directions[i], label="raw_{}".format(image_id), save=True, save_raw_xyz=False)                


        # bring the data together
        self.xyz = torch.cat(self.xyz_per_view, dim=0).to(device=self.device) # to(device=torch.device('cpu')) # 

        self.rgbd = torch.cat(self.rgbd, dim=0)
        self.pixel_rows = torch.cat(self.pixel_rows, dim=0)
        self.pixel_cols = torch.cat(self.pixel_cols, dim=0)
        self.image_ids_per_pixel = torch.cat(self.image_ids_per_pixel, dim=0)
        self.confidence_per_pixel = torch.cat(self.confidence_per_pixel, dim=0)
        
        # and clean up
        self.number_of_pixels = self.image_ids_per_pixel.shape[0]
        self.near = torch.min(self.rgbd[:,3])
        self.far = torch.max(self.rgbd[:,3])
        print("The near bound is {:.3f} meters and the far bound is {:.3f} meters".format(self.near, self.far))
        self.initial_poses = self.initial_poses[::self.args.skip_every_n_images_for_training]
        
        print("Loaded {} images with {:,} pixels selected".format(i+1, self.number_of_pixels ))


    def derive_xyz_coordinates(self, camera_world_position, camera_world_rotation, pixel_directions, pixel_depths, flattened=False):
        
        if not flattened:
            # transform rays from camera coordinate to world coordinate
            # camera_world_rotation: [1,1,1,3,3]
            pixel_world_directions = torch.matmul(camera_world_rotation, pixel_directions).squeeze(4).squeeze(0)                                                            
            pixel_world_directions = torch.nn.functional.normalize(pixel_world_directions, p=2, dim=2)  # (N_pixels, 3)
            
            # Get sample position in the world (1, 1, 3) + (H, W, 3) * (H, W, 1) -> (H, W, 3)
            global_xyz = camera_world_position + pixel_world_directions * pixel_depths.unsqueeze(2)
            global_xyz = global_xyz.squeeze(0)

        else:
            pixel_directions_world = torch.matmul(camera_world_rotation, pixel_directions.unsqueeze(2)).squeeze(2)  # (N, 3, 3) * (N, 3, 1) -> (N, 3) .squeeze(3) 
            pixel_directions_world = torch.nn.functional.normalize(pixel_directions_world, p=2, dim=2)  # (N_pixels, 3)
            pixel_depth_samples_world_directions = pixel_directions_world * pixel_depths.unsqueeze(1).expand(-1,3) # (N_pixels, 3)
            global_xyz = camera_world_position + pixel_depth_samples_world_directions # (N_pixels, 3)

        return global_xyz
            

    def load_estimates_of_depth_sensor_error(self):
        self.average_nearest_neighbor_distance_per_pixel = []
        need_to_recompute_sensor_error_estimates = False
    
        for view, image_id in enumerate(self.image_ids[::self.args.skip_every_n_images_for_training]):
            error_metric_file_name = "image_id_{}_estimated_sensor_error_from_top_{}_knn_distances.npy".format(image_id, self.args.number_of_nearest_neighbors_to_use_in_knn_distance_metric_for_estimation_of_sensor_error)
            file_path = "{}/sensor_variance/{}".format(self.args.base_directory, error_metric_file_name)

            if os.path.exists(file_path):
                error_metric_data =  torch.from_numpy(np.load(file_path))
                self.average_nearest_neighbor_distance_per_pixel.append(error_metric_data)
            else:
                need_to_recompute_sensor_error_estimates = True
                break

        if need_to_recompute_sensor_error_estimates:
            # otherwise, let's go ahead and compute it

            sensor_variance_dir = Path("{}/sensor_variance".format(self.args.base_directory))
            sensor_variance_dir.mkdir(parents=True, exist_ok=True)
            
            print("Recomputing estimates of depth sensor error based on KNN for each point initially projected in 3D")
            number_of_views = len(self.xyz_per_view)
            for view, image_id in enumerate(self.image_ids[::self.args.skip_every_n_images_for_training]):
                this_view_xyz = self.xyz_per_view[view] # (N_pixels, 3)
                number_of_points_in_this_view = this_view_xyz.shape[0]

                # we compare against the view immediately prior and immediately after, as long as they exist
                comparison_views = []
                for comparison_view in [view-1, view+1]:
                    if comparison_view >= 0 and comparison_view <= number_of_views - 1:
                        comparison_views.append(comparison_view)

                # we will compute an error metric per pixel, which is the average distance to the nearest neighbor, for the nearby view(s)
                distances_to_nearest_neighbors_in_nearest_views = torch.zeros(size=[number_of_points_in_this_view]).to(device=self.device)

                for comparison_view in comparison_views:
                    # get (x,y,z) coordinates for this view and other view
                    other_view_xyz = self.xyz_per_view[comparison_view] # (N_pixels, 3)
                    number_of_points_in_other_view = other_view_xyz.shape[0]

                    distances, indices, nn = knn_points(p1=torch.unsqueeze(this_view_xyz, dim=0), p2=torch.unsqueeze(other_view_xyz, dim=0), K=self.args.number_of_nearest_neighbors_to_use_in_knn_distance_metric_for_estimation_of_sensor_error)
                    for nearest_neighbor_index in range(self.args.number_of_nearest_neighbors_to_use_in_knn_distance_metric_for_estimation_of_sensor_error):
                        nearest_neighbor_distance = distances[0,:,nearest_neighbor_index] # 0th batch, all points, i-th nearest neighbor
                        distances_to_nearest_neighbors_in_nearest_views += nearest_neighbor_distance

                average_nearest_neighbor_distance_per_pixel = distances_to_nearest_neighbors_in_nearest_views / (self.args.number_of_nearest_neighbors_to_use_in_knn_distance_metric_for_estimation_of_sensor_error * len(comparison_views))
                error_metric_file_name = "image_id_{}_estimated_sensor_error_from_top_{}_knn_distances.npy".format(image_id, self.args.number_of_nearest_neighbors_to_use_in_knn_distance_metric_for_estimation_of_sensor_error)
                print("Saving average of {} nearest neighbor distances (e.g. {}mm for pixels 0,1,2) for view {} vs. views {}, located in file {}".format(self.args.number_of_nearest_neighbors_to_use_in_knn_distance_metric_for_estimation_of_sensor_error,
                                                                                                                                              average_nearest_neighbor_distance_per_pixel[0:3] * 1000,
                                                                                                                                              view,
                                                                                                                                              comparison_views,
                                                                                                                                              error_metric_file_name))
                file_path = "{}/sensor_variance/{}".format(self.args.base_directory, error_metric_file_name)
                with open(file_path, "wb") as f:
                    np.save(f, average_nearest_neighbor_distance_per_pixel.cpu().numpy())

                self.average_nearest_neighbor_distance_per_pixel.append(average_nearest_neighbor_distance_per_pixel)
            
        self.xyz_per_view = None

        # now we need to flatten the sensor variance estimates, just like the rest of the data
        self.estimated_sensor_error = torch.cat(self.average_nearest_neighbor_distance_per_pixel, dim=0)


    def get_sensor_xyz_coordinates(self, i=None, pose_data=None, depth_data=None):

        # get camera world position and rotation
        camera_world_position = pose_data[:3, 3].view(1, 1, 1, 3)     # (1, 1, 1, 3)
        camera_world_rotation = pose_data[:3, :3].view(1, 1, 1, 3, 3) # (1, 1, 1, 3, 3)

        # get relative pixel orientations              
        pixel_directions = self.pixel_directions[i].unsqueeze(3) # (H, W, 3, 1)                        
        xyz_coordinates = self.derive_xyz_coordinates(camera_world_position, camera_world_rotation, pixel_directions, depth_data)

        return xyz_coordinates 


    def get_min_max_bounds(self, xyz_coordinates, padding=0.025):
        min_x = torch.min(xyz_coordinates[:,:,0]) - padding
        max_x = torch.max(xyz_coordinates[:,:,0]) + padding
        min_y = torch.min(xyz_coordinates[:,:,1]) - padding
        max_y = torch.max(xyz_coordinates[:,:,1]) + padding
        min_z = torch.min(xyz_coordinates[:,:,2]) - padding
        max_z = torch.max(xyz_coordinates[:,:,2]) + padding

        x_cm = torch.abs(max_x - min_x) * 100
        y_cm = torch.abs(max_y - min_y) * 100
        z_cm = torch.abs(max_z - min_z) * 100

        print("The bounds in meters of the selected region are: (min,max) := [x: ({:.3f},{:.3f}), y: ({:.3f},{:.3f}), z: ({:.3f},{:.3f})]".format(  min_x,
                                                                                                                                                    max_x,
                                                                                                                                                    min_y,
                                                                                                                                                    max_y,
                                                                                                                                                    min_z,
                                                                                                                                                    max_z))
        print("i.e. a rectangular prism of size {:.1f}cm x {:.1f}cm x {:.1f}cm".format(x_cm, y_cm, z_cm))

        return min_x, max_x, min_y, max_y, min_z, max_z


    def set_xyz_bounds_from_crop_of_image(self, index_to_filter, min_pixel_row, max_pixel_row, min_pixel_col, max_pixel_col):
        image_id = self.image_ids[index_to_filter]

        # get depth data for that image
        depth, near_bound, far_bound, confidence = self.load_depth_data(image_id=image_id)

        # now, get (x,y,z) coordinates for the first image
        
        xyz_coordinates = self.get_sensor_xyz_coordinates(pose_data=self.initial_poses[index_to_filter], depth_data=depth, i=image_id)

        # now filter both the xyz_coordinates and the image by the values in the top of this function
        if type(min_pixel_row) != type(None) or type(max_pixel_row) != type(None):
            xyz_coordinates = xyz_coordinates[min_pixel_row:max_pixel_row, :, :]
        if type(min_pixel_col) != type(None) or type(max_pixel_col) != type(None):
            xyz_coordinates = xyz_coordinates[:, min_pixel_col:max_pixel_col, :]

        # now define the bounds in (x,y,z) space through which we will filter all future pixels by their projected points
        self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z = self.get_min_max_bounds(xyz_coordinates, padding=100.0)
 

    def get_camera_pixel_directions(self, focal_length):

        camera_coordinates_y, camera_coordinates_x = torch.meshgrid(torch.arange(self.H, dtype=torch.float32, device=self.device),
                                                                    torch.arange(self.W, dtype=torch.float32, device=self.device),
                                                                    indexing='ij')  # (H, W)
                
        #camera_coordinates_y = camera_coordinates_y.expand(self.H, self.W)                  
        #camera_coordinates_x = camera_coordinates_x.expand(self.H, self.W)
        
        focal_length_rep = focal_length.unsqueeze(1).expand(self.H, self.W)        

        # This camera coordinate system is the one that matches with incoming data from Apple's ARKit        
        camera_coordinates_directions_x = (camera_coordinates_x - self.principal_point_x.cpu()) / focal_length_rep  # (H, W)
        camera_coordinates_directions_y = (camera_coordinates_y - self.principal_point_y.cpu()) / focal_length_rep  # (H, W)
        
        camera_coordinates_directions_z = torch.ones(self.H, self.W, dtype=torch.float32, device=self.device)  # (H, W)        
        camera_coordinates_pixel_directions = torch.stack([camera_coordinates_directions_x, camera_coordinates_directions_y, camera_coordinates_directions_z], dim=-1)  # (N_images, H, W, 3)        
        return camera_coordinates_pixel_directions


    # compute unnormalized ray directions for all pixels in dataset
    def compute_ray_direction_in_camera_coordinates(self, focal_length_x, focal_length_y):
        
        # Compute ray directions in the camera coordinate, which only depends on intrinsics. This could be further transformed to world coordinate later, using camera poses.
        camera_coordinates_y, camera_coordinates_x = torch.meshgrid(torch.arange(self.H, dtype=torch.float32, device=self.device),
                                                                    torch.arange(self.W, dtype=torch.float32, device=self.device),
                                                                    indexing='ij')  # (H, W)
        
        # (N_images, H, W)
        camera_coordinates_y = camera_coordinates_y.unsqueeze(0).expand(self.n_training_images, self.H, self.W)                  
        camera_coordinates_x = camera_coordinates_x.unsqueeze(0).expand(self.n_training_images, self.H, self.W)
        
        focal_length_x_rep = focal_length_x.unsqueeze(1).unsqueeze(2).expand(self.n_training_images, self.H, self.W)

        #######################################################################
        # focal_length_y forced to be same as focal_length_x for each image        
        focal_length_y_rep = focal_length_x.unsqueeze(1).unsqueeze(2).expand(self.n_training_images, self.H, self.W)        
        #######################################################################

        # This camera coordinate system is the one that matches with incoming data from Apple's ARKit        
        camera_coordinates_directions_x = (camera_coordinates_x - self.principal_point_x.cpu()) / focal_length_x_rep  # (N_images, H, W)
        camera_coordinates_directions_y = (camera_coordinates_y - self.principal_point_y.cpu()) / focal_length_y_rep  # (N_images, H, W)
        
        camera_coordinates_directions_z = torch.ones(self.n_training_images,self.H, self.W, dtype=torch.float32, device=self.device)  # (N_images, H, W)        
        camera_coordinates_pixel_directions = torch.stack([camera_coordinates_directions_x, camera_coordinates_directions_y, camera_coordinates_directions_z], dim=-1)  # (N_images, H, W, 3)        
        self.pixel_directions = camera_coordinates_pixel_directions.to(device=self.device)         
        
        self.focal_length_x = focal_length_x
        self.focal_length_y = focal_length_y        
        


    #########################################################################
    ####### Define model data structures and set learning parameters ########
    #########################################################################

    def initialize_models(self):
        self.models = {}

        # Load the relevant models
        self.models["focal"] = CameraIntrinsicsModel(self.H, self.W, self.initial_focal_length_x, self.initial_focal_length_y, self.n_training_images).to(device=self.device)
        self.models["pose"] = CameraPoseModel(self.initial_poses).to(device=self.device)
        self.models["geometry"] = NeRFDensity(self.args).to(device=self.device)
        self.models["color"] = NeRFColor(self.args).to(device=self.device)

        # Set up Weights & Biases logging on top of the network in order to record its structure
        wandb.watch(self.models["focal"])
        wandb.watch(self.models["pose"])
        wandb.watch(self.models["geometry"])
        wandb.watch(self.models["color"])        


    def get_polynomial_decay(self, start_value, end_value, exponential_index=1, curvature_shape=1):
        
        if self.epoch > self.args.number_of_epochs: # hack to handle cases where we go outside original epoch range, such as when we load and continue training a model
            return 0.000001
        else:
            return (start_value - end_value) * (1 - (self.epoch)**curvature_shape / self.args.number_of_epochs**curvature_shape)**exponential_index + end_value
        

    def create_polynomial_learning_rate_schedule(self, model):
        schedule = PolynomialDecayLearningRate(optimizer=self.optimizers[model], 
                                               total_steps=self.args.number_of_epochs + self.args.start_epoch, 
                                               start_value=self.learning_rates[model]["start"], 
                                               end_value=self.learning_rates[model]["end"], 
                                               exponential_index=self.learning_rates[model]["exponential_index"], 
                                               curvature_shape=self.learning_rates[model]["curvature_shape"],
                                               model_type=model,
                                               log_frequency=self.args.log_frequency)
        return schedule


    def initialize_learning_rates(self):
        self.optimizers = {}
        self.optimizers["geometry"] = torch.optim.Adam(self.models["geometry"].parameters(), lr=self.args.nerf_density_lr_start)
        self.optimizers["color"] = torch.optim.Adam(self.models["color"].parameters(), lr=self.args.nerf_color_lr_start)
        self.optimizers["focal"] = torch.optim.Adam(self.models["focal"].parameters(), lr=self.args.focal_lr_start)
        self.optimizers["pose"] = torch.optim.Adam(self.models["pose"].parameters(), lr=self.args.pose_lr_start)

        self.learning_rates = {}
        self.learning_rates["geometry"] = {"start": self.args.nerf_density_lr_start, "end": self.args.nerf_density_lr_end, "exponential_index": self.args.nerf_density_lr_exponential_index, "curvature_shape": self.args.nerf_density_lr_curvature_shape}
        self.learning_rates["color"] = {"start": self.args.nerf_color_lr_start, "end": self.args.nerf_color_lr_end, "exponential_index": self.args.nerf_color_lr_exponential_index, "curvature_shape": self.args.nerf_color_lr_curvature_shape}
        self.learning_rates["focal"] = {"start": self.args.focal_lr_start, "end": self.args.focal_lr_end, "exponential_index": self.args.focal_lr_exponential_index, "curvature_shape": self.args.focal_lr_curvature_shape}
        self.learning_rates["pose"] = {"start": self.args.pose_lr_start, "end": self.args.pose_lr_end, "exponential_index": self.args.pose_lr_exponential_index, "curvature_shape": self.args.pose_lr_curvature_shape}

        self.schedulers = {}
        self.schedulers["geometry"] = self.create_polynomial_learning_rate_schedule(model = "geometry")
        self.schedulers["color"] = self.create_polynomial_learning_rate_schedule(model = "color")
        self.schedulers["focal"] = self.create_polynomial_learning_rate_schedule(model = "focal")
        self.schedulers["pose"] = self.create_polynomial_learning_rate_schedule(model = "pose")

        self.learning_rate_histories = {}
        for topic in ["color", "geometry", "pose", "focal"]:
            self.learning_rate_histories[topic] = []

    def load_pretrained_models(self):        
        for model_name in self.models.keys():
            model_path = "{}/{}_{}.pth".format(self.args.pretrained_models_directory, model_name, self.args.start_epoch-1)            

            # load checkpoint data
            ckpt = torch.load(model_path, map_location=self.device)

            # load model from saved state
            model = self.models[model_name]                        
            weights = ckpt['model_state_dict']
            model.load_state_dict(weights, strict=True)            

            # load optimizer parameters
            optimizer = self.optimizers[model_name]
            state = ckpt['optimizer_state_dict']
            optimizer.load_state_dict(state)            

            # scheduler already has reference to optimizer but needs n_steps (epocs)
            scheduler = self.schedulers[model_name]      
            scheduler.n_steps = ckpt['epoch']                
                

    def save_models(self):
        for topic in ["color", "geometry", "pose", "focal"]:
            model = self.models[topic]
            optimizer = self.optimizers[topic]
            print("Saving {} model...".format(topic))            
            save_checkpoint(epoch=self.epoch-1, model=model, optimizer=optimizer, path=self.experiment_dir, ckpt_name='{}_{}'.format(topic, self.epoch-1))


    def save_point_clouds_with_sensor_depths(self):
        for i, image_id in enumerate(self.image_ids[::self.args.skip_every_n_images_for_training]):
            print("Saving with learned poses and intrinsics the raw sensor colors and sensor depth for view {}".format(i))
            image, _ = self.load_image_data(image_id=image_id)
            depth, _, _, _ = self.load_depth_data(image_id=image_id)
            pose = self.models['pose'](0)[i].to(device=self.device)
            self.get_point_cloud(pose=pose, depth=depth, rgb=image, pixel_directions=self.pixel_directions[image_id], label="raw_sensor_with_learned_poses_intrinsics_{}".format(i), save=True, save_raw_xyz=True)               


    #########################################################################
    ############ Helper functions for model usage and training ##############
    #########################################################################

    def sample_depths_linearly(self, number_of_pixels, add_noise=True):
        
        number_of_samples = self.args.number_of_samples_outward_per_raycast + 1

        raycast_distances = torch.linspace(self.near, self.far, number_of_samples).to(self.device)
        raycast_distances = raycast_distances.unsqueeze(0).expand(number_of_pixels, number_of_samples)

        if add_noise:
            depth_noise = torch.rand(number_of_pixels, number_of_samples, device=self.device, dtype=torch.float32)  # (N_pixels, N_samples)
            depth_noise = depth_noise * (self.far - self.near) / number_of_samples # (N_pixels, N_samples)
            raycast_distances = raycast_distances + depth_noise  # (N_pixels, N_samples)            

        return raycast_distances


    def resample_depths_from_nerf_weights(self, number_of_pixels, weights, depth_samples, resample_padding = 0.01, use_sparse_fine_rendering=False):
        double_edged_weights = [weights[..., :1], weights, weights[..., -1:]]
        weights_pad = torch.cat(double_edged_weights, dim=-1)
        weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
        weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

        # Add in a constant (the sampling function will renormalize the PDF).
        weights = weights_blur + resample_padding
        raycast_distances = depth_samples
        weight_based_depth_samples = self.resample_from_weights_probability_distribution(bins=raycast_distances.to(self.device), weights=weights.to(self.device), use_sparse_fine_rendering=use_sparse_fine_rendering)
    
        # equivalent of the official TensorFlow/JAX stop_gradient, to prevent exploding gradients
        weight_based_depth_samples = weight_based_depth_samples.detach()

        return weight_based_depth_samples


    # bins: all of the depth samples [N, pixels, number_of_samples_outward_per_raycast+1]
    # weights: [N_pixels, number_of_samples_outward_per_raycast]
    def resample_from_weights_probability_distribution(self, bins, weights, use_sparse_fine_rendering=False):
        # start off by inferring how many pixels we're computing for
        
        number_of_samples = self.args.number_of_samples_outward_per_raycast

        number_of_pixels = weights.shape[0]
        number_of_weights = weights.shape[1]

        if use_sparse_fine_rendering:
                                    
            weights = weights + 1e-9
            
            max_weight_indices = torch.argmax(weights.to(device=self.device), 1).to(device=self.device)
            pixel_rows = torch.arange(weights.shape[0]).unsqueeze(1).to(device=self.device)            
            max_weight_indices = max_weight_indices.unsqueeze(1).to(device=self.device)                                   
            rows_and_max_indices = torch.cat([pixel_rows, max_weight_indices], dim=1).to(device=self.device)                                                
            rows_and_max_indices = rows_and_max_indices
            bins = bins

            max_depths = bins[rows_and_max_indices[:, 0], rows_and_max_indices[:, 1]]
                        
            bin_length = 1.5 * (self.far-self.near) / number_of_samples                                  
            
            samples = max_depths.unsqueeze(1).expand(number_of_pixels, number_of_samples)
            sample_offsets_from_max = bin_length * torch.rand(number_of_pixels, number_of_samples-1, device=self.device, dtype=torch.float32)  # (N_pixels, N_samples)

            sample_offsets_from_max = sample_offsets_from_max - (bin_length) / 2.0            

            z = torch.zeros(number_of_pixels, 1).to(self.device)
            sample_offsets_from_max_with_zero = torch.cat( (sample_offsets_from_max, z), 1)

            samples = samples + sample_offsets_from_max_with_zero
            samples = torch.sort(samples, dim=1)[0]                        

            return samples                                    
        
        # prevent NaNs
        weights = weights + 1e-9
  
        # create probability distribution function by dividing by weights
        probability_distribution_function = weights / torch.sum(weights, dim=1).unsqueeze(1).expand(number_of_pixels, number_of_weights).to(self.device)

        # get cumulative distribution function, which is the increasing sum of probabilities over the range
        cumulative_distribution_function = torch.cumsum(probability_distribution_function, dim=1).to(self.device)

        # add zeros of same shape as number of pixels to start of the cumulative distribution function
        cumulative_distribution_function = torch.cat([torch.zeros(number_of_pixels,1).to(device=self.device), cumulative_distribution_function], dim=1) 

        # now, we prepare to sample from the probability distribution of the weights, uniformly
        uniform_samples = torch.linspace(0, 1, number_of_samples).to(device=self.device)
        uniform_samples = uniform_samples.unsqueeze(0).expand(number_of_pixels, number_of_samples)

        # we collect from our cumulative distribution function a set of indices corresponding to the CDF
        indices_of_samples_in_cdf = torch.searchsorted(sorted_sequence=cumulative_distribution_function, input=uniform_samples, side="right")

        indices_below = torch.maximum(input=torch.tensor(0), other=indices_of_samples_in_cdf - 1)
        indices_above = torch.minimum(input=torch.tensor(cumulative_distribution_function.shape[1] - 1), other=indices_of_samples_in_cdf)

        range_of_weighted_probabilities = torch.stack(tensors=[indices_below, indices_above], dim=-1) #.squeeze()

        weighted_cdf_min = torch.gather(input=cumulative_distribution_function, index=range_of_weighted_probabilities[:,:,0], dim=1)
        weighted_cdf_max = torch.gather(input=cumulative_distribution_function, index=range_of_weighted_probabilities[:,:,1], dim=1)
        weighted_cdf = torch.cat([weighted_cdf_min.unsqueeze(2), weighted_cdf_max.unsqueeze(2)], dim=2)

        weighted_bins_min = torch.gather(input=bins, index=range_of_weighted_probabilities[:,:,0], dim=1)
        weighted_bins_max = torch.gather(input=bins, index=range_of_weighted_probabilities[:,:,1], dim=1)
        weighted_bins = torch.cat([weighted_bins_min.unsqueeze(2), weighted_bins_max.unsqueeze(2)], dim=2)

        probability_ranges = (weighted_cdf[..., 1] - weighted_cdf[..., 0])

        filtered_probability_ranges = torch.where(probability_ranges < 1e-5, torch.ones_like(probability_ranges), probability_ranges)
        sample_distances = (uniform_samples - weighted_cdf[..., 0]) / filtered_probability_ranges
        samples = weighted_bins[..., 0] + sample_distances * (weighted_bins[..., 1] - weighted_bins[..., 0])                

        return samples.to(self.device)


    def export_geometry_data(self, depth_weight_per_sample, xyz_for_all_samples, test_view_number=0, top_n_depth_weights = 15):
        # wrap up NeRF and baseline data for offline post-processing
        print("Exporting data for test image {}...".format(test_view_number))
        
        x_per_sample = xyz_for_all_samples[:,:,0] # (N_pixels, N_samples)
        y_per_sample = xyz_for_all_samples[:,:,1]
        z_per_sample = xyz_for_all_samples[:,:,2]

        number_of_pixels = x_per_sample.shape[0]
        number_of_samples = x_per_sample.shape[1]

        sorted_depth_weights = torch.argsort(depth_weight_per_sample, dim=1, descending=True)

        all_xyz_depth = []
        all_xyz_slopes = []
        all_xyz_intercepts = []
        all_depth_weights = []

        all_x_slope = []
        all_y_slope = []
        all_z_slope = []

        all_x_intercept = []
        all_y_intercept = []
        all_z_intercept = []

        for pixel_index in range(0, number_of_pixels):                         
            # first, grab our "focus" geometry
            top_sample_indices = sorted_depth_weights[pixel_index,:top_n_depth_weights]

            # now, get the minimum and maximum (x,y,z) points along the ray, and save information for this line: we will fit only between these
            min_top_index = torch.min(top_sample_indices)
            max_top_index = torch.max(top_sample_indices)

            # min (x,y,z) along raycast
            x1 = x_per_sample[pixel_index, min_top_index]
            y1 = y_per_sample[pixel_index, min_top_index]
            z1 = z_per_sample[pixel_index, min_top_index]

            # max (x,y,z) along raycast
            x2 = x_per_sample[pixel_index, max_top_index]
            y2 = y_per_sample[pixel_index, max_top_index]
            z2 = z_per_sample[pixel_index, max_top_index]

            # save information for the line that is defined by the minimum to maximum points along the ray, i.e. x = x_slope * t + x_intercept where x_slope = x_2 - x_1 and x_intercept = x_1
            xyz_slope = torch.stack([x2 - x1, y2 - y1, z2 - z1], dim=0)
            all_xyz_slopes.append(xyz_slope)

            xyz_intercept = torch.stack([x1, y1, z1], dim=0)
            all_xyz_intercepts.append(xyz_intercept)

            # get all of the (x,y,z) for the top depth weights and save them for later analysis
            top_depth_x = x_per_sample[pixel_index, top_sample_indices]
            top_depth_y = y_per_sample[pixel_index, top_sample_indices]
            top_depth_z = z_per_sample[pixel_index, top_sample_indices]
            top_depth_xyz = torch.stack([top_depth_x, top_depth_y, top_depth_z], dim=0)
            all_xyz_depth.append(top_depth_xyz)

            # get the top weights and normalize them so that they sum to 1.0
            top_weights = depth_weight_per_sample[pixel_index, top_sample_indices]
            normalized_top_weights = torch.nn.functional.normalize(top_weights, p=1, dim=0)
            all_depth_weights.append(normalized_top_weights)

            if pixel_index % 1000 == 0:
                print("(TEST VIEW {}) {} Indices: {} with Min Index: {}, Max Index: {}\n >>> Weights: {}".format(test_view_number, pixel_index, top_sample_indices, min_top_index, max_top_index, top_weights))

        xyz_depths = torch.stack(all_xyz_depth, dim=0)
        xyz_slopes = torch.stack(all_xyz_slopes, dim=0)
        xyz_intercepts = torch.stack(all_xyz_intercepts, dim=0)
        depth_weights = torch.stack(all_depth_weights, dim=0)

        with open("{}/xyz_depths_view_{}.npy".format(self.args.geometry_data_out_dir, test_view_number), "wb") as f:
            np.save(f, xyz_depths.cpu().numpy())

        with open("{}/xyz_slopes_view_{}.npy".format(self.args.geometry_data_out_dir, test_view_number), "wb") as f:
            np.save(f, xyz_slopes.cpu().numpy())

        with open("{}/xyz_intercepts_view_{}.npy".format(self.args.geometry_data_out_dir, test_view_number), "wb") as f:
            np.save(f, xyz_intercepts.cpu().numpy())

        with open("{}/depth_weights_view_{}.npy".format(self.args.geometry_data_out_dir, test_view_number), "wb") as f:
            np.save(f, depth_weights.cpu().numpy())


    def render(self, poses, pixel_directions, sampling_depths, perturb_depths=False, rgb_image=None, use_sparse_rendering=False, pixel_focal_lengths_x=None):
        
        # poses := (N_pixels, 4, 4)
        # pixel_directions := (N_images, N_pixels, 3)
        # sampling_depths := (N_samples)

        # (N_pixels, N_sample, 3), (N_pixels, 3), (N_pixels, N_samples)                    
        pixel_xyz_positions, pixel_directions_world, resampled_depths = volume_sampling(poses=poses, pixel_directions=pixel_directions, sampling_depths=sampling_depths, perturb_depths=perturb_depths)
        pixel_directions_world = torch.nn.functional.normalize(pixel_directions_world, p=2, dim=1)  # (N_pixels, 3)
        
        xyz_position_encoding = encode_ipe(poses[:, :3, 3], pixel_xyz_positions, pixel_directions_world, sampling_depths, pixel_focal_lengths_x, self.principal_point_x, self.principal_point_y)                            
        xyz_position_encoding = xyz_position_encoding.to(self.device)
                            
        # encode direction: (H, W, N_sample, (2L+1)*C = 27)        
        angular_directional_encoding = encode_position(pixel_directions_world, levels=self.args.directional_encoding_fourier_frequencies)  # (N_pixels, 27)
        angular_directional_encoding = angular_directional_encoding.unsqueeze(1).expand(-1, sampling_depths.size()[1] - 1, -1)  # (N_pixels, N_sample, 27)

        # inference rgb and density using position and direction encoding.                
        #xyz_position_encoding = xyz_position_encoding.float()        
        xyz_position_encoding = xyz_position_encoding


        
        density, features = self.models["geometry"](xyz_position_encoding) # (N_pixels, N_sample, 1), # (N_pixels, N_sample, D)        

        rgb = self.models["color"](features, angular_directional_encoding)  # (N_pixels, N_sample, 4)        
        
        render_result = volume_rendering(rgb, density, resampled_depths[:, 0:sampling_depths.size()[1]-1])                

        depth_weights = render_result['weight']

        depth_map = render_result['depth_map']
        rgb_rendered = render_result['rgb_rendered']

        if use_sparse_rendering:               
            delta_pixel_densities = torch.zeros(depth_map.size()[0], self.args.number_of_samples_outward_per_raycast).to(device=self.device)
            max_density_indices = torch.argmax(depth_weights.to(device=self.device), 1).to(device=self.device)

            pixel_rows = torch.arange(delta_pixel_densities.shape[0]).unsqueeze(1).to(device=self.device)
            max_density_indices = max_density_indices.unsqueeze(1).to(device=self.device)                       

            raycast_sample_indices_of_max_density = torch.cat([pixel_rows, max_density_indices], dim=1).to(device=self.device)            
            delta_pixel_densities[raycast_sample_indices_of_max_density[:,0], raycast_sample_indices_of_max_density[:,1]] = 10000   
            sparse_render_result = volume_rendering(rgb, delta_pixel_densities.unsqueeze(2), resampled_depths[:, 0:self.args.number_of_samples_outward_per_raycast])
            depth_map = sparse_render_result['depth_map']
            rgb_rendered = sparse_render_result['rgb_rendered']
                
        result = {
            'rgb_rendered': rgb_rendered, # (N_pixels, 3)
            'pixel_xyz_positions': pixel_xyz_positions,    # (N_pixels, N_sample, 3)
            'depth_map': depth_map,       # (N_pixels)
            'depth_weights': render_result['weight'],      # (N_pixels, N_sample),
            'rgb': render_result['rgb'],                   # (N_pixels, N_sample, 3),
            'density': render_result['density'],                            # (N_pixels, N_sample),
            'alpha': render_result['alpha'],               # (N_pixels, N_sample),
            'acc_transmittance': render_result['acc_transmittance'], # (N_pixels, N_sample),
            'resampled_depths': resampled_depths,           # (N_samples)
            'distances': render_result['distances'],
        }

        return result


    # invoke current model for the pose and mask associated with train_image_index
    # for visual results, supply result to save_render_as_png
    def render_prediction_for_train_image(self, train_image_index, use_sparse_rendering=False):
        mask = torch.zeros(self.H, self.W)        

        pixel_indices_for_this_image = torch.argwhere(self.image_ids_per_pixel == train_image_index)        
        pixel_rows = self.pixel_rows[pixel_indices_for_this_image]
        pixel_cols = self.pixel_cols[pixel_indices_for_this_image]        
        mask[pixel_rows, pixel_cols] = 1
        mask = mask.flatten()
        pose = self.models['pose'](0)[train_image_index]

        return self.render_prediction(pose=pose, train_image_index=train_image_index, mask=mask, use_sparse_rendering=use_sparse_rendering)


    # invoke current model for a specific pose and 1d mask
    # for visual results, supply result to save_render_as_png
    # pixels filtered out by the mask are assigned value [0,0,0] 
    def render_prediction(self, pose, train_image_index, mask=None, use_sparse_rendering=False):        

        # copy pose to go with each individual pixel and match reference pixel directions with them
        if mask is not None:
            poses = pose.unsqueeze(0).expand(int(mask.sum().item()), -1, -1)
            pixel_directions = self.pixel_directions[train_image_index].flatten(start_dim=0, end_dim=1)[torch.argwhere(mask==1)]
            pixel_directions = pixel_directions.squeeze(1)                                                
            focal_lengths_x = self.focal_length_x[train_image_index].unsqueeze(0).expand(int(mask.sum().item()), 1)              
        else:
            poses = pose.unsqueeze(0).expand(self.W*self.H, -1, -1)
            pixel_directions = self.pixel_directions[train_image_index].flatten(start_dim=0, end_dim=1)                  
            focal_lengths_x = self.focal_length_x[train_image_index].unsqueeze(0).expand(self.W*self.H, 1)

        # split each of the rendering inputs into smaller batches, which speeds up GPU processing        
        poses_batches = poses.split(self.args.number_of_pixels_per_batch_in_test_renders)
        pixel_directions_batches = pixel_directions.split(self.args.number_of_pixels_per_batch_in_test_renders)
        focal_lengths_x_batches = focal_lengths_x.split(self.args.number_of_pixels_per_batch_in_test_renders)         

        rendered_image_fine_batches = []
        depth_image_fine_batches = []
        density_fine_batches = []
        depth_weights_fine_batches = []        
        resampled_depths_fine_batches = []

        rendered_image_coarse_batches = []
        depth_image_coarse_batches = []
        density_coarse_batches = []
        depth_weights_coarse_batches = []                
        resampled_depths_coarse_batches = []

        rendered_image_unsparse_fine_batches = []
        depth_image_unsparse_fine_batches = []

        if self.args.export_test_data_for_post_processing:
            pixel_xyz_positions_batches = []

        number_of_coarse_samples = self.args.number_of_samples_outward_per_raycast + 1
        number_of_fine_samples = self.args.number_of_samples_outward_per_raycast

        # for each batch, compute the render and extract out RGB and depth map                  
        for poses_batch, pixel_directions_batch, focal_lengths_x_batch in zip(poses_batches, pixel_directions_batches, focal_lengths_x_batches):

            # for resampling with test data, we will compute the NeRF-weighted resamples per batch
            for depth_sampling_optimization in range(self.args.n_depth_sampling_optimizations):
                # get the depth samples per pixel
                if depth_sampling_optimization == 0:
                    # if this is the first iteration, collect linear depth samples to query NeRF, uniformly in space                    
                    depth_samples_coarse = self.sample_depths_linearly(number_of_pixels=poses_batch.size()[0], add_noise=False) # (N_pixels, N_samples)                   
                    resampled_depths_coarse_batches.append(depth_samples_coarse[..., : number_of_coarse_samples].cpu())
                    rendered_data_coarse = self.render(poses=poses_batch, pixel_directions=pixel_directions_batch, sampling_depths=depth_samples_coarse, perturb_depths=False, use_sparse_rendering=use_sparse_rendering, pixel_focal_lengths_x=focal_lengths_x_batch.squeeze(1))  # (N_pixels, 3)
                else:
                    # if this is not the first iteration, then resample with the latest weights                                        
                    depth_samples_fine = self.resample_depths_from_nerf_weights(number_of_pixels=poses_batch.size()[0], weights=rendered_data_coarse['depth_weights'], depth_samples=depth_samples_coarse, use_sparse_fine_rendering=self.args.use_sparse_fine_rendering_for_test_renders)  # (N_pixels, N_samples)            
                    resampled_depths_fine_batches.append(depth_samples_fine[..., : number_of_fine_samples].cpu())
                    rendered_data_fine = self.render(poses=poses_batch, pixel_directions=pixel_directions_batch, sampling_depths=depth_samples_fine, perturb_depths=False, use_sparse_rendering=use_sparse_rendering, pixel_focal_lengths_x=focal_lengths_x_batch.squeeze(1))  # (N_pixels, 3)
                    
                    if self.args.use_sparse_fine_rendering_for_test_renders:
                        # point cloud filtering needs both the sparsely rendered data the "unsparsely" (normal) rendered data
                        depth_samples_fine = self.resample_depths_from_nerf_weights(number_of_pixels=poses_batch.size()[0], weights=rendered_data_coarse['depth_weights'], depth_samples=depth_samples_coarse, use_sparse_fine_rendering=False)  # (N_pixels, N_samples)            
                        unsparse_rendered_data_fine = self.render(poses=poses_batch, pixel_directions=pixel_directions_batch, sampling_depths=depth_samples_fine, perturb_depths=False, use_sparse_rendering=False, pixel_focal_lengths_x=focal_lengths_x_batch.squeeze(1))  # (N_pixels, 3)                        


            if self.args.n_depth_sampling_optimizations > 1:
                rendered_image_fine_batches.append(rendered_data_fine['rgb_rendered']) # (n_pixels_per_row, 3)
                depth_image_fine_batches.append(rendered_data_fine['depth_map']) # (n_pixels_per_row)             
                density_fine_batches.append(rendered_data_fine['density'])
                depth_weights_fine_batches.append(rendered_data_fine['depth_weights'])            
                if self.args.use_sparse_fine_rendering_for_test_renders:
                    rendered_image_unsparse_fine_batches.append(unsparse_rendered_data_fine['rgb_rendered'])
                    depth_image_unsparse_fine_batches.append(unsparse_rendered_data_fine['depth_map'])

            rendered_image_coarse_batches.append(rendered_data_coarse['rgb_rendered']) # (n_pixels_per_row, 3)
            depth_image_coarse_batches.append(rendered_data_coarse['depth_map']) # (n_pixels_per_row)             
            density_coarse_batches.append(rendered_data_coarse['density'])
            depth_weights_coarse_batches.append(rendered_data_coarse['depth_weights'])                        

            if self.args.export_test_data_for_post_processing:                
                pixel_xyz_positions_batches.append(rendered_data_fine['pixel_xyz_positions'])

        # combine batch results to compose full images
        if self.args.n_depth_sampling_optimizations > 1:
            rendered_image_data_fine = torch.cat(rendered_image_fine_batches, dim=0).cpu() # (N_pixels, 3)
            rendered_depth_data_fine = torch.cat(depth_image_fine_batches, dim=0).cpu()  # (N_pixels)
            density_data_fine = torch.cat(density_fine_batches, dim=0).cpu()  # (N_pixels, N_samples)
            depth_weights_data_fine = torch.cat(depth_weights_fine_batches, dim=0).cpu()  # (N_pixels, N_samples)
            resampled_depths_data_fine = torch.cat(resampled_depths_fine_batches, dim=0).cpu()  # (N_pixels, N_samples)
            if self.args.use_sparse_fine_rendering_for_test_renders:
                rendered_image_data_unsparse_fine = torch.cat(rendered_image_unsparse_fine_batches, dim=0).cpu()
                depth_image_data_unsparse_fine = torch.cat(depth_image_unsparse_fine_batches, dim=0).cpu()

        rendered_image_data_coarse = torch.cat(rendered_image_coarse_batches, dim=0).cpu() # (N_pixels, 3)
        rendered_depth_data_coarse = torch.cat(depth_image_coarse_batches, dim=0).cpu()  # (N_pixels)
        density_data_coarse = torch.cat(density_coarse_batches, dim=0).cpu()  # (N_pixels, N_samples)
        depth_weights_data_coarse = torch.cat(depth_weights_coarse_batches, dim=0).cpu()  # (N_pixels, N_samples)
        resampled_depths_data_coarse = torch.cat(resampled_depths_coarse_batches, dim=0).cpu()  # (N_pixels, N_samples)

        if self.args.export_test_data_for_post_processing:
            depth_weight_per_sample = torch.cat(depth_weights_fine_batches, dim=0)
            xyz_for_all_samples = torch.cat(pixel_xyz_positions_batches, dim=0)
            self.export_geometry_data(depth_weight_per_sample=depth_weight_per_sample, xyz_for_all_samples=xyz_for_all_samples, test_view_number=train_image_index, top_n_depth_weights = 10)

        if self.args.n_depth_sampling_optimizations > 1:
            rendered_image_fine = torch.zeros(self.H * self.W, 3).cpu()
            rendered_depth_fine = torch.zeros(self.H * self.W).cpu()
            density_fine = torch.zeros(self.H * self.W, number_of_fine_samples-1).cpu()
            depth_weights_fine = torch.zeros(self.H * self.W, number_of_fine_samples-1).cpu()
            resampled_depths_fine = torch.zeros(self.H * self.W, number_of_fine_samples).cpu()        
            if self.args.use_sparse_fine_rendering_for_test_renders:
                rendered_image_unsparse_fine = torch.zeros(self.H * self.W, 3).cpu()
                depth_image_unsparse_fine = torch.zeros(self.H * self.W).cpu()

        rendered_image_coarse = torch.zeros(self.H * self.W, 3).cpu()
        rendered_depth_coarse = torch.zeros(self.H * self.W).cpu()
        density_coarse = torch.zeros(self.H * self.W, number_of_coarse_samples-1).cpu()
        depth_weights_coarse = torch.zeros(self.H * self.W, number_of_coarse_samples-1).cpu()        
        resampled_depths_coarse = torch.zeros(self.H * self.W, number_of_coarse_samples).cpu()        

        # if we're using a mask, we need to find the right pixel positions
        if mask is not None:

            if self.args.n_depth_sampling_optimizations > 1:
                rendered_image_fine[torch.argwhere(mask==1).squeeze()] = rendered_image_data_fine.cpu()
                rendered_depth_fine[torch.argwhere(mask==1).squeeze()] = rendered_depth_data_fine.cpu()
                density_fine[torch.argwhere(mask==1).squeeze()] = density_data_fine.cpu()
                depth_weights_fine[torch.argwhere(mask==1).squeeze()] = depth_weights_data_fine.cpu()
                resampled_depths_fine[torch.argwhere(mask==1).squeeze()] = resampled_depths_data_fine.cpu()
                if self.args.use_sparse_fine_rendering_for_test_renders:
                    rendered_image_unsparse_fine[torch.argwhere(mask==1).squeeze()] = rendered_image_data_unsparse_fine.cpu()
                    depth_image_unsparse_fine[torch.argwhere(mask==1).squeeze()] = depth_image_data_unsparse_fine.cpu()

            rendered_image_coarse[torch.argwhere(mask==1).squeeze()] = rendered_image_data_coarse.cpu()
            rendered_depth_coarse[torch.argwhere(mask==1).squeeze()] = rendered_depth_data_coarse.cpu()
            density_coarse[torch.argwhere(mask==1).squeeze()] = density_data_coarse.cpu()
            depth_weights_coarse[torch.argwhere(mask==1).squeeze()] = depth_weights_data_coarse.cpu()
            resampled_depths_coarse[torch.argwhere(mask==1).squeeze()] = resampled_depths_data_coarse.cpu()
        else:

            if self.args.n_depth_sampling_optimizations > 1:
                rendered_image_fine = rendered_image_data_fine.cpu()
                rendered_depth_fine = rendered_depth_data_fine.cpu()
                density_fine = density_data_fine.cpu()
                depth_weights_fine = depth_weights_data_fine.cpu()
                resampled_depths_fine = resampled_depths_data_fine.cpu()
                if self.args.use_sparse_fine_rendering_for_test_renders:
                    rendered_image_unsparse_fine = rendered_image_data_unsparse_fine.cpu()
                    depth_image_unsparse_fine = depth_image_data_unsparse_fine.cpu()
                

            rendered_image_coarse = rendered_image_data_coarse.cpu()
            rendered_depth_coarse = rendered_depth_data_coarse.cpu()
            density_coarse = density_data_coarse.cpu()
            depth_weights_coarse = depth_weights_data_coarse.cpu()            
            resampled_depths_coarse = resampled_depths_data_coarse.cpu()

        render_result = {
            'rendered_image_coarse': rendered_image_coarse,
            'rendered_depth_coarse': rendered_depth_coarse,
            'density_coarse': density_coarse,
            'depth_weights_coarse': depth_weights_coarse,                        
            'resampled_depths_coarse': resampled_depths_coarse,
        }

        if self.args.n_depth_sampling_optimizations > 1:            
            render_result['rendered_image_fine'] = rendered_image_fine
            render_result['rendered_depth_fine'] = rendered_depth_fine
            render_result['density_fine'] = density_fine
            render_result['depth_weights_fine'] = depth_weights_fine       
            render_result['resampled_depths_fine'] = resampled_depths_fine

            if self.args.use_sparse_fine_rendering_for_test_renders:
                render_result['rendered_image_unsparse_fine'] = rendered_image_unsparse_fine
                render_result['depth_image_unsparse_fine'] = depth_image_unsparse_fine

        return render_result    


    # process raw rendered pixel data and save into images
    def save_render_as_png(self, render_result, color_file_name_fine, depth_file_name_fine, color_file_name_coarse, depth_file_name_coarse):
        
        rendered_rgb_coarse = render_result['rendered_image_coarse'].reshape(self.H, self.W, 3)
        rendered_depth_coarse = render_result['rendered_depth_coarse'].reshape(self.H, self.W)                
        rendered_color_for_file_coarse = (rendered_rgb_coarse.cpu().numpy() * 255).astype(np.uint8)    

        if self.args.n_depth_sampling_optimizations > 1:
            rendered_rgb_fine = render_result['rendered_image_fine'].reshape(self.H, self.W, 3)
            rendered_depth_fine = render_result['rendered_depth_fine'].reshape(self.H, self.W)
            rendered_color_for_file_fine = (rendered_rgb_fine.cpu().numpy() * 255).astype(np.uint8)    

        # get depth map and convert it to Turbo Color Map
        rendered_depth_data_coarse = rendered_depth_coarse.cpu().numpy() 
        rendered_depth_for_file_coarse = heatmap_to_pseudo_color(rendered_depth_data_coarse)
        rendered_depth_for_file_coarse = (rendered_depth_for_file_coarse * 255).astype(np.uint8)        
        imageio.imwrite(color_file_name_coarse, rendered_color_for_file_coarse)
        imageio.imwrite(depth_file_name_coarse, rendered_depth_for_file_coarse)                   

        if self.args.n_depth_sampling_optimizations > 1:
            rendered_depth_data_fine = rendered_depth_fine.cpu().numpy() 
            rendered_depth_for_file_fine = heatmap_to_pseudo_color(rendered_depth_data_fine)
            rendered_depth_for_file_fine = (rendered_depth_for_file_fine * 255).astype(np.uint8)        
            imageio.imwrite(color_file_name_fine, rendered_color_for_file_fine)
            imageio.imwrite(depth_file_name_fine, rendered_depth_for_file_fine)               


    def train(self, indices_of_random_pixels, iteration=0):
        # initialize whether each model is in training mode or else is just in evaluation mode (no gradient updates)
        if self.epoch >= self.args.start_training_color_epoch:
            self.models["color"].train()
        else:
            self.models["color"].eval()

        if self.epoch >= self.args.start_training_extrinsics_epoch:
            self.models["pose"].train()
        else:
            self.models["pose"].eval()

        if self.epoch >= self.args.start_training_intrinsics_epoch:
            self.models["focal"].train()
        else:
            self.models["focal"].eval()

        if self.epoch >= self.args.start_training_geometry_epoch:
            self.models["geometry"].train()
        else:
            self.models["geometry"].eval()

        # get the randomly selected RGBD data
        rgbd = self.rgbd[indices_of_random_pixels].to(self.device)  # (N_pixels, 4)

        # get the pixel rows and columns that we've selected (across all images)
        pixel_rows = self.pixel_rows[indices_of_random_pixels]
        pixel_cols = self.pixel_cols[indices_of_random_pixels]

        # unpack the image RGB data and the sensor depth
        rgb = rgbd[:,:3].to(self.device) # (N_pixels, 3)
        sensor_depth = rgbd[:,3].to(self.device) # (N_pixels) 
        sensor_depth_per_sample = sensor_depth.unsqueeze(1).expand(-1, self.args.number_of_samples_outward_per_raycast) # (N_pixels, N_samples) 

        n_pixels = self.args.pixel_samples_per_epoch        

        # get confidence data for each pixel, which can be used to weight the depth loss
        selected_confidences = self.confidence_per_pixel[indices_of_random_pixels].to(self.device)
        confidence_loss_weights = torch.where(selected_confidences >= self.args.min_confidence, 1, 0).to(self.device)
        number_of_pixels_with_confident_depths = torch.sum(confidence_loss_weights)

        # initialize our total weighted loss, which will be computed as the weighted sum of coarse and fine losses
        total_weighted_loss = torch.tensor(0.0).to(self.device)

        for depth_sampling_optimization in range(self.args.n_depth_sampling_optimizations):

            # get the camera intrinsics, train if this is not the coarse render
            if self.epoch >= self.args.start_training_intrinsics_epoch and depth_sampling_optimization > 0:
                focal_length_x, focal_length_y = self.models["focal"](0)
            else:
                with torch.no_grad():
                    focal_length_x, focal_length_y = self.models["focal"](0)
            
            self.compute_ray_direction_in_camera_coordinates(focal_length_x, focal_length_y)

            # get all camera poses from model
            if self.epoch >= self.args.start_training_extrinsics_epoch and depth_sampling_optimization > 0:
                poses = self.models["pose"](0) # (N_images, 4, 4)
            else:
                with torch.no_grad():
                    poses = self.models["pose"](0)  # (N_images, 4, 4)

            # get a tensor with the poses per pixel
            image_ids = self.image_ids_per_pixel[indices_of_random_pixels].to(self.device) # (N_pixels)
            selected_poses = poses[image_ids].to(self.device) # (N_pixels, 4, 4)

            # get the focal lengths for every pixel given the images that were actually selected for each pixel
            selected_focal_lengths_x = focal_length_x[image_ids].to(self.device)
            selected_focal_lengths_y = focal_length_y[image_ids].to(self.device)

            # get pixel directions
            pixel_directions_selected = self.pixel_directions[image_ids, pixel_rows, pixel_cols]  # (N_pixels, 3)        


            #####################| Sampling & Rendering |##################
            if depth_sampling_optimization == 0:
                # if this is the first iteration, collect linear depth samples to query NeRF, uniformly in space
                depth_samples = self.sample_depths_linearly(number_of_pixels=n_pixels, add_noise=True) # (N_pixels, N_samples)
            else:
                # if this is not the first iteration, then resample with the latest weights
                depth_samples = self.resample_depths_from_nerf_weights(number_of_pixels=n_pixels, weights=nerf_depth_weights, depth_samples=depth_samples)  # (N_pixels, N_samples)
            
            n_samples = depth_samples.size()[1]

            # render an image using selected rays, pose, sample intervals, and the network
            render_result = self.render(poses=selected_poses, pixel_directions=pixel_directions_selected, sampling_depths=depth_samples, perturb_depths=False, rgb_image=rgb, pixel_focal_lengths_x=selected_focal_lengths_x)  # (N_pixels, 3)

            rgb_rendered = render_result['rgb_rendered']         # (N_pixels, 3)
            nerf_depth_weights = render_result['depth_weights']  # (N_pixels, N_samples)
            nerf_depth = render_result['depth_map']              # (N_pixels) NeRF depth (weights x distances) for every pixel
            nerf_sample_bin_lengths_in = render_result['distances'] # (N_pixels, N_samples)            

            nerf_sample_bin_lengths = depth_samples[:, 1:] - depth_samples[:, :-1]
            nerf_depth_weights = nerf_depth_weights + self.args.epsilon            

            #####################| KL Loss |################################
            sensor_variance = self.args.depth_sensor_error
            kl_divergence_bins = -1 * torch.log(nerf_depth_weights) * torch.exp(-1 * (depth_samples[:, : n_samples - 1] * 1000 - sensor_depth_per_sample[:, : n_samples - 1] * 1000) ** 2 / (2 * sensor_variance)) * nerf_sample_bin_lengths * 1000                                
            confidence_weighted_kl_divergence_pixels = confidence_loss_weights * torch.sum(kl_divergence_bins, 1) # (N_pixels)
            depth_loss = torch.sum(confidence_weighted_kl_divergence_pixels) / number_of_pixels_with_confident_depths
            depth_to_rgb_importance = self.get_polynomial_decay(start_value=self.args.depth_to_rgb_loss_start, end_value=self.args.depth_to_rgb_loss_end, exponential_index=self.args.depth_to_rgb_loss_exponential_index, curvature_shape=self.args.depth_to_rgb_loss_curvature_shape)
            
            
            #####################| Entropy Loss |###########################
            if (self.epoch > self.args.entropy_loss_tuning_start_epoch and self.epoch < self.args.entropy_loss_tuning_end_epoch):
                entropy_depth_loss_weight = 0.0075
                pixels_weights_entropy = -1 * torch.sum(nerf_depth_weights * torch.log(nerf_depth_weights), dim=1)
                entropy_depth_loss = entropy_depth_loss_weight * torch.mean(pixels_weights_entropy)                
                depth_to_rgb_importance = 0.0
            else:
                entropy_depth_loss = 0.0
            ################################################################

            with torch.no_grad():
                # get a metric in Euclidian space that we can output via prints for human review/intuition; not actually used in backpropagation
                interpretable_depth_loss = confidence_loss_weights * torch.sum(nerf_depth_weights * torch.sqrt((depth_samples[:, : n_samples-1] * 1000 - sensor_depth_per_sample[:, : n_samples -1 ] * 1000) ** 2), dim=1)
                interpretable_depth_loss_per_confident_pixel = torch.sum(interpretable_depth_loss) / number_of_pixels_with_confident_depths

                # get a metric in (0-255) (R,G,B) space that we can output via prints for human review/intuition; not actually used in backpropagation
                interpretable_rgb_loss = torch.sqrt((rgb_rendered * 255 - rgb * 255) ** 2)
                interpretable_rgb_loss_per_pixel = torch.mean(interpretable_rgb_loss)

            # compute the mean squared difference between the RGB render of the neural network and the original image     
            rgb_loss = (rgb_rendered - rgb)**2
            rgb_loss = torch.mean(rgb_loss)

            # to-do: implement perceptual color difference minimizer
            # torch.norm(ciede2000_diff(rgb2lab_diff(inputs,self.device),rgb2lab_diff(adv_input,self.device),self.device).view(batch_size, -1),dim=1)
            fine_rgb_loss = 0
            fine_depth_loss = 0
            fine_interpretable_rgb_loss_per_pixel = 0
            fine_interpretable_depth_loss_per_confident_pixel = 0

            # following official mip-NeRF, if this is the coarse render, we only give 0.1 weight to the total loss contribution; if it is a fine, then 0.9
            if depth_sampling_optimization == 0:
                total_weighted_loss = 0.1 * (depth_to_rgb_importance * depth_loss + (1 - depth_to_rgb_importance) * rgb_loss + entropy_depth_loss)

                coarse_rgb_loss = rgb_loss
                coarse_depth_loss = depth_loss
                coarse_interpretable_rgb_loss_per_pixel = interpretable_rgb_loss_per_pixel
                coarse_interpretable_depth_loss_per_confident_pixel = interpretable_depth_loss_per_confident_pixel

            else:
                #depth_loss = 0
                #entropy_depth_loss = 0
                #total_weighted_loss += 0.9 * (depth_to_rgb_importance * depth_loss + (1 - depth_to_rgb_importance) * rgb_loss + entropy_depth_loss)
                total_weighted_loss += 0.9 * ((1 - depth_to_rgb_importance) * rgb_loss)

                fine_rgb_loss = rgb_loss
                #fine_depth_loss = depth_loss
                fine_depth_loss = 0
                fine_interpretable_rgb_loss_per_pixel = interpretable_rgb_loss_per_pixel
                fine_interpretable_depth_loss_per_confident_pixel = interpretable_depth_loss_per_confident_pixel

        for optimizer in self.optimizers.values():
            optimizer.zero_grad()   

        # release unused GPU memory
        torch.cuda.empty_cache()

        # backward propagate the gradients to update the values which are parameters to this loss
        total_weighted_loss.backward(create_graph=False, retain_graph=False)
        
        # step each optimizer forward once
        for optimizer in self.optimizers.values():
            optimizer.step()        
                  
        # update the learning rate schedulers
        for scheduler in self.schedulers.values():
            scheduler.step()

        if self.epoch % self.args.log_frequency == 0:
            wandb.log({"Coarse RGB Inverse Render Loss (0-255 per pixel)": coarse_interpretable_rgb_loss_per_pixel,
                       "Coarse Depth Sensor Loss (average millimeters error vs. sensor)": coarse_interpretable_depth_loss_per_confident_pixel,
                       "Fine RGB Inverse Render Loss (0-255 per pixel)": fine_interpretable_rgb_loss_per_pixel,
                       "Fine Depth Sensor Loss (average millimeters error vs. sensor)": fine_interpretable_depth_loss_per_confident_pixel               
                       })

        if self.epoch % self.args.log_frequency == 0:
            minutes_into_experiment = (int(time.time())-int(self.start_time)) / 60

        self.log_learning_rates()

        print("({} at {:.2f} min) - LOSS = {:.5f} -> RGB: C: {:.6f} ({:.3f}/255) | F: {:.6f} ({:.3f}/255), DEPTH: C: {:.6f} ({:.2f}mm) | F: {:.6f} ({:.2f}mm) w/ imp. {:.5f}, Entropy: {:.6f}".format(self.epoch, 
            minutes_into_experiment, 
            total_weighted_loss,
            (1 - depth_to_rgb_importance) * coarse_rgb_loss, 
            coarse_interpretable_rgb_loss_per_pixel, 
            (1 - depth_to_rgb_importance) * fine_rgb_loss, 
            fine_interpretable_rgb_loss_per_pixel,             
            depth_to_rgb_importance * coarse_depth_loss, 
            coarse_interpretable_depth_loss_per_confident_pixel,
            depth_to_rgb_importance * fine_depth_loss, 
            fine_interpretable_depth_loss_per_confident_pixel,            
            depth_to_rgb_importance,
            entropy_depth_loss
        ))
        
        # a new epoch has dawned
        self.epoch += 1


    def sample_next_batch(self):
        indices_of_random_pixels_for_this_epoch = random.sample(population=range(self.number_of_pixels), k=self.args.pixel_samples_per_epoch)                    
        return indices_of_random_pixels_for_this_epoch


    def create_experiment_directory(self):
        data_out_dir = "{}/hyperparam_experiments".format(self.args.base_directory)            
        experiment_params = "depth_loss_{}_to_{}_k{}_N{}_NeRF_Density_LR_{}_to_{}_k{}_N{}_pose_LR_{}_to_{}_k{}_N{}".format( self.args.depth_to_rgb_loss_start,
                                                                                                                            self.args.depth_to_rgb_loss_end,
                                                                                                                            self.args.depth_to_rgb_loss_exponential_index,
                                                                                                                            self.args.depth_to_rgb_loss_curvature_shape,
                                                                                                                            self.args.nerf_density_lr_start,
                                                                                                                            self.args.nerf_density_lr_end,
                                                                                                                            self.args.nerf_density_lr_exponential_index,
                                                                                                                            self.args.nerf_density_lr_curvature_shape,
                                                                                                                            self.args.pose_lr_start,
                                                                                                                            self.args.pose_lr_end,
                                                                                                                            self.args.pose_lr_exponential_index,
                                                                                                                            self.args.pose_lr_curvature_shape)

        experiment_label = "{}_{}".format(self.start_time, experiment_params)            
        experiment_dir = Path(os.path.join(data_out_dir, experiment_label))
        experiment_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir = experiment_dir

        self.color_out_dir = Path("{}/color_renders/".format(self.experiment_dir))
        self.color_out_dir.mkdir(parents=True, exist_ok=True)

        self.depth_out_dir = Path("{}/depth_renders/".format(self.experiment_dir))
        self.depth_out_dir.mkdir(parents=True, exist_ok=True)

        self.depth_weights_out_dir = Path("{}/depth_weights_visualization/".format(self.experiment_dir))
        self.depth_weights_out_dir.mkdir(parents=True, exist_ok=True)

        self.depth_view_out_dir = Path("{}/depth_view_pointcloud/".format(self.experiment_dir))
        self.depth_view_out_dir.mkdir(parents=True, exist_ok=True)  

        self.geometry_data_out_dir = Path("{}/geometry_data/".format(self.experiment_dir))
        self.geometry_data_out_dir.mkdir(parents=True, exist_ok=True)            

        self.learning_rates_out_dir = Path("{}/learning_rates/".format(self.experiment_dir))
        self.learning_rates_out_dir.mkdir(parents=True, exist_ok=True)              


    def save_experiment_parameters(self):
        param_dict = vars(self.args)
        f = open('{}/parameters.txt'.format(self.experiment_dir), 'w')
        for param_name in param_dict:
            f.write('{} = {}\n'.format(param_name, param_dict[param_name]))
        

    def log_learning_rates(self):
        for topic in ["color", "geometry", "pose", "focal"]:
            lr = self.schedulers[topic].polynomial_decay()
            if topic in self.learning_rate_histories:
                self.learning_rate_histories[topic].append(lr)
            else:
                self.learning_rate_histories[topic] = [lr]            
                

    def test(self):
        
        epoch = self.epoch - 1        
        for model in self.models.values():
            model.eval()

        # compute the ray directions using the latest focal lengths, derived for the first image
        focal_length_x, focal_length_y = self.models["focal"](0)
        self.compute_ray_direction_in_camera_coordinates(focal_length_x, focal_length_y)

        for image_index in self.test_image_indices:
            print("Rendering for view {}".format(image_index))
            render_result = self.render_prediction_for_train_image(image_index, self.args.use_sparse_rendering_for_test_renders)


            nerf_weights_coarse = render_result['depth_weights_coarse']
            density_coarse = render_result['density_coarse']                                
            depth_coarse = render_result['rendered_depth_coarse']            
            sampled_depths_coarse = render_result['resampled_depths_coarse']

            if self.args.n_depth_sampling_optimizations > 1:
                nerf_weights_fine = render_result['depth_weights_fine']
                density_fine = render_result['density_fine']
                depth_fine = render_result['rendered_depth_fine']            
                sampled_depths_fine = render_result['resampled_depths_fine']            
            pixel_indices = torch.argwhere(self.image_ids_per_pixel == image_index)
            this_image_rgbd = self.rgbd[pixel_indices].cpu().squeeze(1)            
                        
            # save rendered rgb and depth images
            out_file_suffix = str(image_index)
            color_file_name_fine = os.path.join(self.color_out_dir, str(out_file_suffix).zfill(4) + '_color_fine_{}.png'.format(epoch))
            depth_file_name_fine = os.path.join(self.depth_out_dir, str(out_file_suffix).zfill(4) + '_depth_fine_{}.png'.format(epoch))            
            color_file_name_coarse = os.path.join(self.color_out_dir, str(out_file_suffix).zfill(4) + '_color_coarse_{}.png'.format(epoch))
            depth_file_name_coarse = os.path.join(self.depth_out_dir, str(out_file_suffix).zfill(4) + '_depth_coarse_{}.png'.format(epoch))                        
            self.save_render_as_png(render_result, color_file_name_fine, depth_file_name_fine, color_file_name_coarse, depth_file_name_coarse)
            
            # save rendered depth as a pointcloud
            if epoch % self.args.save_point_cloud_frequency == 0:
                print("Saving .ply for view {}".format(image_index))
                depth_view_file_name = os.path.join(self.depth_view_out_dir, str(out_file_suffix).zfill(4) + '_depth_view_{}.png'.format(epoch))
                pose = self.models['pose'](0)[image_index].to(device=self.device)
                unsparse_rendered_rgb_img = None

                if self.args.n_depth_sampling_optimizations > 1: 
                    depth_img = render_result['rendered_depth_fine'].reshape(self.H, self.W).to(device=self.device)
                    rendered_rgb_img = render_result['rendered_image_fine'].reshape(self.H, self.W, 3).to(device=self.device)
                    if self.args.use_sparse_fine_rendering_for_test_renders:
                        unsparse_rendered_rgb_img = render_result['rendered_image_unsparse_fine'].reshape(self.H, self.W, 3).to(device=self.device)
                        unsparse_depth_img = render_result['depth_image_unsparse_fine'].reshape(self.H, self.W).to(device=self.device)
                    else:
                        unsparse_rendered_rgb_img = None
                        unsparse_depth_img = None
                else:
                    depth_img = render_result['rendered_depth_coarse'].reshape(self.H, self.W).to(device=self.device)
                    rendered_rgb_img = render_result['rendered_image_coarse'].reshape(self.H, self.W, 3).to(device=self.device)

                rgb = this_image_rgbd[:, :3]
                ground_truth_image = torch.zeros(self.H * self.W, 3).cpu()
                entropy_image = torch.zeros(self.H * self.W, 1).cpu()                
                mask = torch.zeros(self.H, self.W)        
                
                pixel_rows = self.pixel_rows[pixel_indices]
                pixel_cols = self.pixel_cols[pixel_indices]        
                mask[pixel_rows, pixel_cols] = 1
                mask = mask.flatten()
                
                ground_truth_image[torch.argwhere(mask==1).squeeze()] = rgb.cpu()   
                ground_truth_image = ground_truth_image.reshape(self.H, self.W, 3)
                                
                coarse_weights_entropy = -1 * torch.sum( (nerf_weights_coarse+self.args.epsilon) * torch.log(nerf_weights_coarse + self.args.epsilon), dim=1)
                entropy_image = coarse_weights_entropy.reshape(self.H, self.W)
                #entropy_image[torch.argwhere(mask==1).squeeze()] = coarse_weights_entropy[torch.argwhere(mask==1).squeeze()].unsqueeze(1)                

                this_image_pixel_directions = self.pixel_directions[image_index]

                


                self.get_point_cloud(pose=pose, depth=depth_img, rgb=ground_truth_image, pixel_directions=this_image_pixel_directions, label="_{}_{}".format(epoch,image_index), save=True, dir=self.depth_view_out_dir, entropy_image=entropy_image, sparse_rendered_rgb_img=rendered_rgb_img, unsparse_rendered_rgb_img=unsparse_rendered_rgb_img, unsparse_depth=unsparse_depth_img)               

            # save graphs of nerf density weights visualization
            if epoch % self.args.save_depth_weights_frequency == 0:               
                print("Creating depth weight graph for view {}".format(image_index))
                Path("{}/depth_weights_visualization/{}/image_{}/".format(self.experiment_dir, epoch, image_index)).mkdir(parents=True, exist_ok=True)
                start_pixel = 160 * 640                
                #start_pixel = 0
                
                for pixel_index in range(start_pixel, start_pixel+640):        
                    
                    # prevent out-of-bounds when image is masked
                    if this_image_rgbd.size()[0] < pixel_index:
                        continue

                    out_path = Path("{}/{}/".format(self.depth_weights_out_dir, epoch))
                    out_path.mkdir(parents=True, exist_ok=True)
                    sensor_depth = this_image_rgbd[pixel_index, 3]                    

                    #raycast_distances = np.array([x for x in torch.linspace(self.near, self.far, self.args.number_of_samples_outward_per_raycast).numpy()])                                                                                            
                    if self.args.n_depth_sampling_optimizations > 1:                    
                        weights_fine = nerf_weights_fine.squeeze(0)[pixel_index]                                        
                        densities_fine = density_fine.squeeze(0)[pixel_index]  
                        predicted_depth = depth_fine[pixel_index]                                        


                    image = ground_truth_image
                    downsized_image = torch.nn.functional.interpolate(image.permute(2, 0, 1).unsqueeze(0), size=(int(image.shape[0]/4), int(image.shape[1]/4)), mode='bilinear').squeeze().permute(1, 2, 0)
                    pixel_row_to_highlight = self.pixel_rows[pixel_indices[pixel_index]][0] 
                    pixel_col_to_highlight = self.pixel_cols[pixel_indices[pixel_index]][0]                    
                    pixel_row_to_highlight = int(pixel_row_to_highlight / 4)
                    pixel_col_to_highlight = int(pixel_col_to_highlight / 4)
                    downsized_image[pixel_row_to_highlight, pixel_col_to_highlight, :] = torch.tensor([1.0,0.0,0.0], dtype=torch.float32)     
                    downsized_image = (downsized_image * 255).to(torch.long)
                    downsized_image = downsized_image.cpu().numpy().astype(np.uint8)
                    downsized_path = Path("{}/downsized_image_{}.png".format(self.experiment_dir, image_index))
                    imageio.imwrite(downsized_path, downsized_image)                    

                    densities_coarse = density_coarse.squeeze(0)[pixel_index]  
                    weights_coarse = nerf_weights_coarse.squeeze(0)[pixel_index]                   
                    predicted_depth_coarse = depth_coarse[pixel_index]
                    coarse_weights_entropy = -1 * torch.sum( (weights_coarse+self.args.epsilon) * torch.log(weights_coarse + self.args.epsilon), dim=0)
                    fig, axs = plt.subplots(3,2)
                    offset = 0.02

                    avg_fine_sample_depth = torch.mean(sampled_depths_fine[pixel_index, : sampled_depths_fine.size()[1]-1])
                    if self.args.n_depth_sampling_optimizations > 1:
                        predicted_depth_fine = depth_fine[pixel_index]                    
                        axs[1,0].scatter(sampled_depths_fine[pixel_index, : sampled_depths_fine.size()[1]-1], weights_fine, s=1, marker='o', c='green')                    
                        axs[1,0].scatter([sensor_depth.item()], [0], s=30, marker='o', c='red')                    
                        axs[1,0].scatter([predicted_depth_fine.item()], [0], s=5, marker='o', c='blue')
                        #axs[1,0].set_xlim([avg_fine_sample_depth - offset, avg_fine_sample_depth + offset])
                        axs[1,0].set_xlim([0,4])
                                        
                    axs[0,0].scatter(sampled_depths_coarse[pixel_index, : sampled_depths_coarse.size()[1]-1], weights_coarse, s=1, marker='o', c='green')                    
                    axs[0,0].scatter([sensor_depth.item()], [0], s=30, marker='o', c='red')                    
                    axs[0,0].scatter([predicted_depth_coarse.item()], [0], s=5, marker='o', c='blue')                    
                    #axs[0,0].set_xlim([avg_fine_sample_depth - offset, avg_fine_sample_depth + offset])
                    axs[0,0].set_xlim([0,4])
                    if self.args.n_depth_sampling_optimizations > 1:
                                            
                        axs[1,1].scatter(sampled_depths_fine[pixel_index, : sampled_depths_fine.size()[1]-1], densities_fine, s=1, marker='o', c='green')                    
                        axs[1,1].scatter([sensor_depth.item()], [0], s=30, marker='o', c='red')                    
                        axs[1,1].scatter([predicted_depth_fine.item()], [0], s=5, marker='o', c='blue')
                        #axs[1,1].set_xlim([avg_fine_sample_depth - offset, avg_fine_sample_depth + offset])
                        axs[1,1].set_xlim([0,4])

                    
                    axs[0,1].scatter(sampled_depths_coarse[pixel_index, : sampled_depths_coarse.size()[1]-1], densities_coarse, s=1, marker='o', c='green')                    
                    axs[0,1].scatter([sensor_depth.item()], [0], s=30, marker='o', c='red')                    
                    axs[0,1].scatter([predicted_depth_coarse.item()], [0], s=5, marker='o', c='blue')                    
                    #axs[0,1].set_xlim([avg_fine_sample_depth - offset, avg_fine_sample_depth + offset])
                    axs[0,1].set_xlim([0,4])

                    text_kwargs = dict(ha='center', va='center', fontsize=8, color='black')
                    axs[2,0].text(0.5,0.5, 'coarse weights entropy: {:.6f} \npixel coordinates: ({},{}) '.format(coarse_weights_entropy, self.pixel_rows[pixel_indices[pixel_index]][0], self.pixel_cols[pixel_indices[pixel_index]][0]) ,  **text_kwargs)
                    axs[2,0].axis('off')


                    #axin = axs[2,1].inset_axes([0,0, 160,120],transform=axs[2,1].transData)  
                    axs[2,1].imshow(downsized_image)
                    axs[2,1].axis('off')
                    

                    out_file_suffix = 'image_{}/{}'.format(image_index, image_index)
                    depth_weights_file_name = os.path.join(out_path, str(out_file_suffix).zfill(4) + '_depth_weights_e{}_p{}.png'.format(epoch, pixel_index))
                    plt.savefig(depth_weights_file_name)
                    plt.close()
                    
            # save graphs of learning rate histories
            for topic in ["color", "geometry", "pose", "focal"]:                                            
                lrs = self.learning_rate_histories[topic]
                plt.figure()
                plt.plot([x for x in range(len(lrs))], lrs)
                plt.ylim(0.0, 0.001)
                plt.savefig('{}/{}.png'.format(self.learning_rates_out_dir, topic))
                plt.close()

        if self.args.export_test_data_for_post_processing:
            print("System exiting after exporting geometry data")
            sys.exit(0)   


    def load_saved_args_train(self):

        self.args.base_directory = './data/elastica_burgundy_2'
        #self.args.base_directory = './data/dragon_scale'
        self.args.images_directory = 'color'
        self.args.images_data_type = 'jpg'
        self.args.skip_every_n_images_for_training = 60    
        self.args.load_pretrained_models = False
        self.args.pretrained_models_directory = ''
        self.args.start_epoch = 1
        self.args.number_of_epochs = 1000000

        self.args.start_training_extrinsics_epoch = 500        
        self.args.start_training_intrinsics_epoch = 5000        
        self.args.start_training_color_epoch = 0
        self.args.start_training_geometry_epoch = 0
        self.args.entropy_loss_tuning_start_epoch = 1000000
        self.args.entropy_loss_tuning_end_epoch = 1000000

        self.args.nerf_density_lr_start = 0.0010
        self.args.nerf_density_lr_end = 0.000025
        self.args.nerf_density_lr_exponential_index = 4
        self.args.nerf_density_lr_curvature_shape = 1

        self.args.nerf_color_lr_start = 0.001
        self.args.nerf_color_lr_end = 0.000025
        self.args.nerf_color_lr_exponential_index = 2
        self.args.nerf_color_lr_curvature_shape = 1

        self.args.focal_lr_start = 0.001
        self.args.focal_lr_end = 0.00001
        self.args.focal_lr_exponential_index = 9
        self.args.focal_lr_exponential_index = 1

        self.args.pose_lr_start = 0.0001
        self.args.pose_lr_end = 0.0000025
        self.args.pose_lr_exponential_index = 9
        self.args.pose_lr_curvature_shape = 1

        self.args.depth_to_rgb_loss_start = 0.0075
        self.args.depth_to_rgb_loss_end = 0.0
        self.args.depth_to_rgb_loss_exponential_index = 9
        self.args.depth_to_rgb_loss_curvature_shape = 1

        self.args.density_neural_network_parameters = 256
        self.args.color_neural_network_parameters = 256
        #self.args.positional_encoding_fourier_frequencies = 8
        self.args.directional_encoding_fourier_frequencies = 8
        self.args.positional_encoding = 'mip'

        self.args.maximum_depth = 5.0 # prev 5.0
        self.args.epsilon = 0.0000001
        self.args.depth_sensor_error = 0.5
        self.args.min_confidence = 2.0

        ### test images
        self.args.skip_every_n_images_for_testing = 60
        self.args.number_of_test_images = 1    

        ### test frequency parameters
        self.args.test_frequency = 5000
        self.args.export_test_data_for_testing = False
        self.args.save_ply_point_clouds_of_sensor_data = False
        self.args.save_ply_point_clouds_of_sensor_data_with_learned_poses = False        
        self.args.save_point_cloud_frequency = 5000
        self.args.save_depth_weights_frequency = 5000
        self.args.log_frequency = 1
        self.args.save_models_frequency = 20000

        ### GPU parameters
        self.args.pixel_samples_per_epoch = 400
        self.args.number_of_pixels_per_batch_in_test_renders = 400
        self.args.number_of_samples_outward_per_raycast = 512

        self.args.use_sparse_rendering_for_test_renders = False
        self.args.use_sparse_fine_rendering_for_test_renders = False        


    def load_saved_args_test(self):

        self.load_saved_args_train()
        self.args.load_pretrained_models = True
        self.args.n_depth_sampling_optimizations = 2
        self.args.pretrained_models_directory = './models/dragon_scale_2/entropy_20000_1024'
        self.args.entropy_loss_tuning_start_epoch = 7800000000
        self.args.entropy_loss_tuning_end_epoch = 8200000000
        self.args.number_of_samples_outward_per_raycast = 1024
        self.args.number_of_pixels_per_batch_in_test_renders = 50
        self.args.pixel_samples_per_epoch = 200
        self.args.test_frequency = 1000
        self.args.save_depth_weights_frequency = 1
        self.args.save_point_cloud_frequency = 1
        self.args.start_epoch = 800001
        self.args.number_of_epochs = 40000
        self.args.save_models_frequency = 5000
        self.args.number_of_test_images = 100
        self.args.skip_every_n_images_for_testing = 2
        
        self.args.positional_encoding = 'mip'        

        self.args.use_sparse_rendering_for_test_renders = False
        self.args.use_sparse_fine_rendering_for_test_renders = True

        


if __name__ == '__main__':
    # Load a scene object with all data and parameters

    with torch.no_grad():
        scene = SceneModel(args=parse_args(), load_saved_args=True)

    while scene.epoch < scene.args.start_epoch + scene.args.number_of_epochs:    

        #with torch.no_grad():
        #    scene.test()
        #    quit()

        batch = scene.sample_next_batch()        
        #with torch.autograd.detect_anomaly():
        scene.train(batch) 
        
        if (scene.epoch-1) % scene.args.test_frequency == 0 and (scene.epoch-1) != 0:
            with torch.no_grad():
                scene.test()
       
        if (scene.epoch-1) % scene.args.save_models_frequency == 0 and (scene.epoch-1) !=  scene.args.start_epoch:
            scene.save_models()

        