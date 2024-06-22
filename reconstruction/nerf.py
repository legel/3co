import torch
import torch._dynamo
import numpy as np
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix
from pytorch3d.ops.knn import knn_points
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import cv2
import imageio.v2 as imageio
from PIL import Image

import matplotlib.pyplot as plt
import random, math
import sys, os, shutil, copy, glob, json
import time, datetime
import argparse
from pathlib import Path
from tqdm import tqdm
import gc
import trimesh

sys.path.append(os.path.join(sys.path[0], '../..'))

from utils.pos_enc import encode_position, encode_ipe
from utils.volume_op import volume_sampling, volume_rendering
from utils.lie_group_helper import convert3x4_4x4
from utils.training_utils import PolynomialDecayLearningRate, heatmap_to_pseudo_color, set_randomness, save_checkpoint
from utils.camera import *

from torch_efficient_distloss import eff_distloss # MipNeRF-360 distortion loss implementation from https://github.com/sunset1995/torch_efficient_distloss

from models.intrinsics import CameraIntrinsicsModel
from models.poses import CameraPoseModel
from models.nerf_models import NeRFDensity, NeRFColor
from models.ssan import SSAN_Geometry, SSAN_Appearance
from models.ngp import HashEmbedder, SHEncoder

class SceneModel:
    def __init__(self):
        self.parse_input_args()
        self.initialize_training_environment()
        self.prepare_testing()
        self.load_all_data_and_models()
        if self.args.train:
            self.sample_training_data()

    def parse_input_args(self):
        parser = argparse.ArgumentParser()

        # Define path to relevant data for training, and decide on number of images to use in training
        parser.add_argument('--base_directory', type=str, default='./data/spotted_purple_orchid', help='The base directory to load and save information from')
        parser.add_argument('--images_directory', type=str, default='color', help='The specific group of images to use during training')
        parser.add_argument('--images_data_type', type=str, default='jpg', help='Whether images are jpg or png')
        parser.add_argument('--H_for_training', type=int, default=192, help='The height in pixels that training images will be downsampled to')
        parser.add_argument('--W_for_training', type=int, default=256, help='The width in pixels that training images will be downsampled to')
        parser.add_argument('--number_of_pixels_in_training_dataset', type=int, default=480*640*100, help='The total number of pixels sampled from all images in training data')
        parser.add_argument('--resample_pixels_frequency', type=int, default=500, help='Resample training data every this number of epochs')
        parser.add_argument('--save_models_frequency', type=int, default=5000, help='Save model every this number of epochs')
        parser.add_argument('--load_pretrained_models', type=bool, default=False, help='Whether to start training from models loaded with load_pretrained_models()', action=argparse.BooleanOptionalAction)
        parser.add_argument('--pretrained_models_directory', type=str, default='./data/spotted_purple_orchid/hyperparam_experiments/168465057', help='The directory storing models to load')    
        parser.add_argument('--reset_learning_rates', type=bool, default=False, help='When loading pretrained models, whether to reset learning rate schedules instead of resuming them', action=argparse.BooleanOptionalAction)
        parser.add_argument('--H_for_test_renders', type=int, default=480, help='The image height used for test renders and pointclouds')
        parser.add_argument('--W_for_test_renders', type=int, default=640, help='The image width used for test renders and pointclouds')                
        parser.add_argument('--number_of_images_in_training_dataset', type=int, default=1024, help='The number of images that will be trained on in total for each dataset')                

        # Define the type of NeRF encoding (MiP or NGP), as well as type of coarse sampling, ranging from "naive" to "focused" to "sdf"
        parser.add_argument('--positional_encoding_framework', type=str, default='mip', help='If \'mip\' then use mip-NeRF encoding; if \'NGP\', then use hash encoding from NGP')
        parser.add_argument('--coarse_sampling_strategy', type=str, default='naive', help='If \'naive\' then coarse sampling is naively linearly sampled between global near and far estimates; if \'focused\', then sampling is focused to only include some parts of a pre-defined voxel-based grid; if \'sdf\', then Signed Surface Approximation Network will be trained')
        parser.add_argument('--extract_depth_probability_distributions', type=bool, default=False, help='Whether to extract the 16th, 50th, and 84th percentile depth distances per pixel during testing for later SDF training', action=argparse.BooleanOptionalAction)
        parser.add_argument('--visualize_depth_probability_distributions', type=bool, default=True, help='Whether to visualize the 16th, 50th, and 84th percentile depth distances per pixel during testing for later, for human review', action=argparse.BooleanOptionalAction)
        parser.add_argument('--extract_sdf_field', type=bool, default=False, help='Whether to extract previously trained SDF field', action=argparse.BooleanOptionalAction)
        parser.add_argument('--extract_sdf_frequency', type=int, default=500, help='Frequency to stop SDF training and output visualizations', action=argparse.BooleanOptionalAction)
        parser.add_argument('--export_extrinsics_intrinsics', type=bool, default=False, help='Whether to extract previously trained SDF field', action=argparse.BooleanOptionalAction)

        # Define parameters for leveraging knowledge of trusted vs. bad poses estimated by Apple
        parser.add_argument('--avoid_trusting_bad_pose_estimates', type=bool, default=False, help='Requires a pre-processing step to determine bad pose estimates', action=argparse.BooleanOptionalAction)

        # Define parameters for voxel-based sampling
        parser.add_argument('--object_voxel_xyz_data_file', type=str, default='/home/photon/sense/3cology/plantvine/scans/bird_of_paradise/trained_models/sdf/pointclouds/150mm_voxel_xyz_center.pt', help='File with information for prior voxelization of where the object is probably located')
        parser.add_argument('--object_voxel_size_in_meters', type=float, default=0.15, help='The size of a voxel used for estimating location of object in pixels')

        # Define number of epochs, and timing by epoch for when to start training per network
        parser.add_argument('--start_epoch', default=0, type=int, help='Epoch on which to begin or resume training')
        parser.add_argument('--number_of_epochs', default=500001, type=int, help='Number of epochs for training, used in learning rate schedules')    
        parser.add_argument('--start_training_extrinsics_epoch', type=int, default=100, help='Set to epoch number >= 0 to init poses using estimates from iOS, and start refining them from this epoch.')
        parser.add_argument('--start_training_intrinsics_epoch', type=int, default=1000, help='Set to epoch number >= 0 to init focals using estimates from iOS, and start refining them from this epoch.')
        parser.add_argument('--start_training_color_epoch', type=int, default=0, help='Set to a epoch number >= 0 to start learning RGB NeRF on top of density NeRF.')
        parser.add_argument('--start_training_geometry_epoch', type=int, default=0, help='Set to a epoch number >= 0 to start learning RGB NeRF on top of density NeRF.')
        parser.add_argument('--max_entropy_weight', type=float, default=0.002, help='Weight used for entropy loss.')

        # Define evaluation/logging/saving frequency and parameters
        parser.add_argument('--test_frequency', default=1000000, type=int, help='Frequency of epochs to render an evaluation image')        
        parser.add_argument('--log_frequency', default=1, type=int, help='Frequency of epochs to log outputs e.g. loss performance')        
        parser.add_argument('--visualize_poses', default=False, type=bool, help='Frequency of epochs to visualize poses')        
        parser.add_argument('--number_of_test_images', default=1, type=int, help='Index in the training data set of the image to show during testing')
        parser.add_argument('--skip_every_n_images_for_testing', default=1, type=int, help='Skip every Nth testing image, to ensure sufficient test view diversity in large data set')    
        parser.add_argument('--number_of_pixels_per_batch_in_test_renders', default=2048, type=int, help='Size in pixels of each batch input to rendering')                
        parser.add_argument('--train', default=True, type=bool, help='Whether or not to train', action=argparse.BooleanOptionalAction)                

        # Define learning rates, including start, stop, and two parameters to control curvature shape (https://arxiv.org/pdf/2004.05909v1.pdf)
        parser.add_argument('--nerf_density_lr_start', default=0.0005, type=float, help="Learning rate start for NeRF geometry network")
        parser.add_argument('--nerf_density_lr_end', default=0.0001, type=float, help="Learning rate end for NeRF geometry network")
        parser.add_argument('--nerf_density_lr_exponential_index', default=4, type=int, help="Learning rate speed of exponential decay (higher value = faster initial decay) for NeRF geometry network")
        parser.add_argument('--nerf_density_lr_curvature_shape', default=1, type=int, help="Learning rate shape of decay (lower value = faster initial decay) for NeRF geometry network")

        parser.add_argument('--nerf_color_lr_start', default=0.0005, type=float, help="Learning rate start for NeRF RGB (pitch,yaw) network")
        parser.add_argument('--nerf_color_lr_end', default=0.0001, type=float, help="Learning rate end for NeRF RGB (pitch,yaw) network")
        parser.add_argument('--nerf_color_lr_exponential_index', default=2, type=int, help="Learning rate speed of exponential decay (higher value = faster initial decay) for NeRF RGB (pitch,yaw) network")
        parser.add_argument('--nerf_color_lr_curvature_shape', default=1, type=int, help="Learning rate shape of decay (lower value = faster initial decay) for NeRF RGB (pitch,yaw) network")

        parser.add_argument('--focal_lr_start', default=0.0001, type=float, help="Learning rate start for NeRF-- camera intrinsics network")
        parser.add_argument('--focal_lr_end', default=0.0000025, type=float, help="Learning rate end for NeRF-- camera intrinsics network")
        parser.add_argument('--focal_lr_exponential_index', default=1, type=int, help="Learning rate speed of exponential decay (higher value = faster initial decay) for NeRF-- camera intrinsics network")
        parser.add_argument('--focal_lr_curvature_shape', default=1, type=int, help="Learning rate shape of decay (lower value = faster initial decay) for NeRF-- camera intrinsics network")

        parser.add_argument('--pose_lr_start', default=0.0001, type=float, help="Learning rate start for NeRF-- camera extrinsics network")
        parser.add_argument('--pose_lr_end', default=0.00001, type=float, help="Learning rate end for NeRF-- camera extrinsics network")
        parser.add_argument('--pose_lr_exponential_index', default=9, type=int, help="Learning rate speed of exponential decay (higher value = faster initial decay) for NeRF-- camera extrinsics network")
        parser.add_argument('--pose_lr_curvature_shape', default=1, type=int, help="Learning rate shape of decay (lower value = faster initial decay) for NeRF-- camera extrinsics network")

        parser.add_argument('--depth_to_rgb_loss_start', default=0.5, type=float, help="Learning rate start for ratio of loss importance between depth and RGB inverse rendering loss")
        parser.add_argument('--depth_to_rgb_loss_end', default=0.00001, type=float, help="Learning rate end for ratio of loss importance between depth and RGB inverse rendering loss")
        parser.add_argument('--depth_to_rgb_loss_exponential_index', default=9, type=int, help="Learning rate speed of exponential decay (higher value = faster initial decay) for ratio of loss importance between depth and RGB inverse rendering loss")
        parser.add_argument('--depth_to_rgb_loss_curvature_shape', default=1, type=int, help="Learning rate shape of decay (lower value = faster initial decay) for ratio of loss importance between depth and RGB inverse rendering loss")    

        parser.add_argument('--entropy_loss_tuning_start_epoch', type=float, default=10000, help='epoch to start entropy loss tuning')
        parser.add_argument('--entropy_loss_tuning_end_epoch', type=float, default=1000000, help='epoch to end entropy loss tuning')    
        
        # Special rendering and export parameters
        parser.add_argument('--use_sparse_fine_rendering', type=bool, default=False,  help='Whether or not to use sparse fine rendering technique in test renders', action=argparse.BooleanOptionalAction)
        parser.add_argument('--save_point_clouds_during_testing', type=bool, default=False,  help='Whether or not to save point clouds during testing', action=argparse.BooleanOptionalAction)
        parser.add_argument('--maximum_point_cloud_depth', type=float, default=1.0,  help='Maximum depth of point cloud points saved')
        parser.add_argument('--maximum_sparse_vs_unsparse_depth_difference', type=float, default=0.5,  help='Maximum difference in meters between sparse and unsparse depths...')

        # Define parameters the determine the overall size and learning capacity of the neural networks and their encodings
        parser.add_argument('--density_neural_network_parameters', type=int, default=256, help='The baseline number of units that defines the size of the NeRF geometry network')
        parser.add_argument('--color_neural_network_parameters', type=int, default=256, help='The baseline number of units that defines the size of the NeRF RGB (pitch,yaw) network')
        parser.add_argument('--directional_encoding_fourier_frequencies', type=int, default=8, help='The number of frequencies that are generated for positional encoding of (pitch, yaw)')

        # Define sampling parameters, including how many samples per raycast (outward), number of samples randomly selected per image, and (if masking is used) ratio of good to masked samples
        parser.add_argument('--pixel_samples_per_epoch', type=int, default=2500, help='The number of rows of samples to randomly collect for each image during training')
        parser.add_argument('--number_of_samples_outward_per_raycast', type=int, default=250, help='The number of samples per raycast to collect for each rendered pixel during training')        
        parser.add_argument('--number_of_samples_outward_per_raycast_for_test_renders', type=int, default=500, help='The number of samples per raycast to collect for each rendered pixel during testing')        

        # Define depth sensor parameters
        parser.add_argument('--depth_sensor_error', type=float, default=0.33, help='A rough estimate of the 1D-Gaussian-modeled depth sensor, in millimeters')
        parser.add_argument('--epsilon', type=float, default=1e-10, help='Minimum value in log() for NeRF density weights going to 0')    
        parser.add_argument('--min_depth_sensor_confidence', type=float, default=2.0, help='A value in [0,1,2] where 0 allows all depth data to be used, 2 filters the most and ignores that')

        # Additional parameters on pre-processing of depth data and coordinate systems
        parser.add_argument('--near_maximum_depth', type=float, default=1.0, help='A percent of all raycast samples will be dedicated between the minimum depth (determined by sensor value) and this value')
        parser.add_argument('--far_maximum_depth', type=float, default=6.0, help='The remaining percent of all raycast samples will be dedicated between the near_maximum_depth and this value')
        parser.add_argument('--percentile_of_samples_in_near_region', type=float, default=0.90, help='This is the percent that determines the ratio between near and far sampling')    

        # Depth sampling optimizations
        parser.add_argument('--n_depth_sampling_optimizations', type=int, default=2, help='For every epoch, for every set of pixels, do this many renders to find the best depth sampling distances')
        parser.add_argument('--coarse_weight', type=float, default=0.1, help='Weight between [0,1] for coarse loss')        

        self.args = parser.parse_args()


    def initialize_training_environment(self):
        self.parse_input_args()
        self.initialize_params()
        self.create_experiment_directory()        
        self.save_experiment_parameters()   


    def load_all_data_and_models(self):
        if self.args.avoid_trusting_bad_pose_estimates:
            self.load_pose_quality_information()

        self.load_all_images_ids()        
        self.load_camera_intrinsics()
        self.load_camera_extrinsics()

        if self.args.coarse_sampling_strategy in ["focused", "sdf"]:
           self.load_xyz_of_voxel_center_representing_object_occupancy()
           self.create_grid_for_fast_point_in_voxel_lookup()
                
        self.prepare_test_data()        

        self.initialize_models()        
        self.initialize_learning_rates()

        if self.args.load_pretrained_models:
            self.load_pretrained_models()                        
        else:            
            print("Training from scratch")
        
        if self.args.coarse_sampling_strategy == "focused":
            # create a filter for sampling training data based on whether pixels point to voxels occupied by object
            self.determine_which_pixels_see_object()

        self.preprocess_training_image_and_depth_data_once_for_fast_loading()

        self.create_blender_cameras_for_nerf2mesh()
        quit()


    def initialize_params(self):
        # initialize high-level arguments        
        self.epoch = self.args.start_epoch
        self.start_time = int(time.time())

        # set random seed for experiment reproducibility
        set_randomness()

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        torch.set_printoptions(precision=4, sci_mode=False)

        # set cache directory
        os.environ['PYTORCH_KERNEL_CACHE_PATH'] = self.args.base_directory

        torch.set_float32_matmul_precision('high')
        torch._dynamo.config.verbose=True        
        torch._dynamo.config.suppress_errors = False

        self.device = torch.device('cuda:0')    


    #########################################################################
    ################ Loading and initial processing of data #################
    #########################################################################
    def load_image_data(self, image_id):
        # recreate the image name
        image_name = "{}.{}".format(str(int(image_id)).zfill(6), self.args.images_data_type)

        # get image path in folder
        path_to_images = "{}/{}".format(self.args.base_directory, self.args.images_directory)
        image_path = os.path.join(path_to_images, image_name)

        # load image data, and collect indices of pixels that may be masked using alpha channel if image is a .png 
        image_data = np.array(Image.open(image_path))
        image_data = cv2.resize(image_data, (self.args.W_for_training, self.args.H_for_training))

        self.H = int(image_data.shape[0])
        self.W = int(image_data.shape[1])        
        
        # clip out alpha channel, if it exists
        image_data = image_data[:, :, :3]  # (H, W, 3)

        # convert to torch format and normalize between 0-1.0
        image = torch.from_numpy(image_data).to(dtype=torch.uint8) #.float() / 255 # (H, W, 3) torch.float32
        
        #return image.to(device=self.device), image_name
        return image.cpu(), image_name


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
        confidence_data = torch.from_numpy(confidence_data).cpu()

        # read the 16 bit greyscale depth data which is formatted as an integer of millimeters
        depth_mm = cv2.imread(depth_path, -1).astype(np.float32)

        # convert data in millimeters to meters
        depth_m = depth_mm / (1000.0)  
        
        # set a cap on the maximum depth in meters; clips erroneous/irrelevant depth data from way too far out
        depth_m[depth_m > self.args.far_maximum_depth] = self.args.far_maximum_depth

        # resize to a resolution that e.g. may be higher, and equivalent to image data
        resized_depth_meters = cv2.resize(depth_m, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
                
        # NeRF requires bounds which can be used to constrain both the processed coordinate system data, as well as the ray sampling
        near_bound = np.min(resized_depth_meters)
        far_bound = np.max(resized_depth_meters)        
        depth = torch.Tensor(resized_depth_meters).cpu() # (N_images, H_image, W_image)

        return depth, near_bound, far_bound, confidence_data


    def export_depth_data(self, source_depth_file_path, destination_depth_file_path, max_depth = 3.0, visualize_depth_data = True):        
        # read the 16 bit greyscale depth data which is formatted as an integer of millimeters
        depth_mm = cv2.imread(source_depth_file_path, -1).astype(np.float32)

        # convert data in millimeters to meters
        depth_m = depth_mm / (1000.0)  
        
        # set a cap on the maximum depth in meters; clips erroneous/irrelevant depth data from way too far out
        depth_m[depth_m > max_depth] = max_depth

        # resize to a resolution that e.g. may be higher, and equivalent to image data
        resized_depth_meters = cv2.resize(depth_m, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
                
        # convert back to 16 bit (saves memory)
        depth_mm = (1000 * resized_depth_meters).astype(np.uint16)

        # read and export confidence data too
        source_confidence_file_path = source_depth_file_path.replace("depth", "confidence")
        print("Exporting confidence data from {}".format(source_confidence_file_path))
        
        # load confidence data, which Apple provides with only three possible confidence metrics: 0 (least confident), 1 (moderate confidence), 2 (most confident)
        confidence_data = np.array(Image.open(source_confidence_file_path))

        # we're now going to interpolate the confidence data, such that we will be able to have estimated confidence values for interpolated depths
        confidence_data = cv2.resize(confidence_data, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        confidence_data = torch.from_numpy(confidence_data).cpu()

        destination_confidence_file_path = destination_depth_file_path.replace(".npy", "_confidence.npy")

        if visualize_depth_data:

            np_depth_data = heatmap_to_pseudo_color(depth_mm.astype(np.float32) / 1000)
            np_depth_data = (255 * np_depth_data).astype(np.uint8) #(255 * np_depth_data) / np.max(np_depth_data)
            depth_image = Image.fromarray(np_depth_data)

            destination_depth_image_path = destination_depth_file_path.replace(".npy",".png")
            print("Saving depth visualization to {}".format(destination_depth_image_path))
            depth_image.save(destination_depth_image_path)

        # quit()

        np.save(destination_depth_file_path, depth_mm)
        np.save(destination_confidence_file_path, confidence_data)


    def load_camera_intrinsics(self):
        # load camera instrinsics estimates from Apple's internal API
        camera_intrinsics_data = json.load(open(os.path.join(self.args.base_directory,'camera_intrinsics.json')))
        intrinsics_H = int(camera_intrinsics_data["height"])
        intrinsics_W = int(camera_intrinsics_data["width"])

        # save camera intrinsics matrix for future learning
        camera_intrinsics_matrix = np.zeros(shape=(3,3))
        camera_intrinsics_matrix[0,0] = camera_intrinsics_data["intrinsic_matrix"][0] # fx (focal length)
        camera_intrinsics_matrix[1,1] = camera_intrinsics_data["intrinsic_matrix"][4] # fy (focal length)
        camera_intrinsics_matrix[0,2] = camera_intrinsics_data["intrinsic_matrix"][6] # ox (principal point) 320   W = 640
        camera_intrinsics_matrix[1,2] = camera_intrinsics_data["intrinsic_matrix"][7] # oy (principal point) 240   H = 480
        camera_intrinsics_matrix[2,2] = camera_intrinsics_data["intrinsic_matrix"][8] # 1.0

        # scale intrinsics matrix values to match downsampled training data
        camera_intrinsics_matrix[0,0] = camera_intrinsics_data["intrinsic_matrix"][0] * (float(self.W) / float(intrinsics_W))
        camera_intrinsics_matrix[1,1] = camera_intrinsics_data["intrinsic_matrix"][4] * (float(self.H) / float(intrinsics_H))
        camera_intrinsics_matrix[0,2] = camera_intrinsics_data["intrinsic_matrix"][6] * (float(self.W) / float(intrinsics_W))
        camera_intrinsics_matrix[1,2] = camera_intrinsics_data["intrinsic_matrix"][7] * (float(self.H) / float(intrinsics_H))

        self.camera_intrinsics = torch.Tensor(camera_intrinsics_matrix).to(device=self.device)

        # save_point
        self.initial_focal_length = self.camera_intrinsics[0,0].repeat(self.args.number_of_images_in_training_dataset)                
        #self.initial_focal_length = self.camera_intrinsics[0,0].repeat(len(self.image_ids[::self.skip_every_n_images_for_training]))                

        self.principal_point_x = self.camera_intrinsics[0,2]
        self.principal_point_y = self.camera_intrinsics[1,2]


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
            rotation_matrix = quaternion_to_matrix(quaternion)

            to_backward_z_axis = torch.tensor([
                [1.0,  0.0,  0.0],
                [0.0, -1.0,  0.0],
                [0.0,  0.0, -1.0]
            ])

            rotation_matrix =  rotation_matrix @ to_backward_z_axis
            
            pose = np.eye(4)
            pose[:3, :3] = rotation_matrix
            pose[:3, 3] = translation_vector            

            poses.append(pose)

        poses = np.asarray(poses)
        rotations_translations = poses[:,:3,:] # get rotations and translations from the 4x4 matrix in a 4x3 matrix                
        self.all_initial_poses = torch.Tensor(convert3x4_4x4(rotations_translations)).cpu() # (N, 4, 4)  
        self.selected_initial_poses = self.all_initial_poses[::self.skip_every_n_images_for_training][:self.args.number_of_images_in_training_dataset]

        # initial_poses = self.selected_initial_poses.clone()
        # for bad_pose_index in self.bad_pose_indices:
        #     current_index = bad_pose_index
        #     while current_index in self.bad_pose_indices:
        #         if current_index <= 0:
        #             current_index = 0
        #             break
        #         current_index = current_index - 1
        #     print("Initializing bad pose {} to pose {}".format(bad_pose_index, current_index))
        #     self.selected_initial_poses[bad_pose_index] = initial_poses[current_index]




    # def load_multi_view_depth_confidences(self):
    #     self.bad_pose_indices = torch.load("{}/bad_pose_indices.pt".format(self.directory_for_fast_reload_of_training_params))


    def load_pose_quality_information(self):
        self.bad_pose_indices = torch.load("{}/bad_pose_indices.pt".format(self.directory_for_fast_reload_of_training_params))


    def find_bad_poses(self, image_ids_sampled):
        # Reshape image_ids_sampled for broadcasting
        image_ids_reshaped = image_ids_sampled.unsqueeze(-1)
        
        # Check each element of image_ids_sampled against each element of bad_pose_indices
        comparison = image_ids_reshaped == self.bad_pose_indices
        
        # Sum along the second dimension and convert to binary tensor
        mask = (comparison.sum(dim=-1) > 0).long()
        
        return mask


    def get_mask_filter(self, all_image_ids, image_ids_to_mask_with_1):
        # Reshape image_ids_sampled for broadcasting
        all_image_ids = all_image_ids.unsqueeze(-1)
        
        # Check each element of image_ids_sampled against each element of bad_pose_indices
        comparison = all_image_ids == image_ids_to_mask_with_1
        
        # Sum along the second dimension and convert to binary tensor
        mask = (comparison.sum(dim=-1) > 0).long()
        
        return mask


    def export_camera_extrinsics_and_intrinsics_from_trained_model(self, output_directory, epoch, save=True, print_debug_info=False, skip_saving=False):
        # gather the latest poses ("camera extrinsics")


        print("Exporting camera extrinsics...")
        camera_extrinsics = self.models['pose']()([0]).cpu() #self.all_initial_poses[::self.skip_every_n_images_for_training][:self.args.number_of_images_in_training_dataset] #self.models['pose']()([0]).cpu()    
       
        # gather the latest focal lengths (f_x and f_y, which are currently identical)
        focal_length = self.models['focal']()([0]).cpu()      

        print("Focal length {}: {}". format(focal_length.shape, focal_length))

        # get the number of cameras represented
        n_cameras = camera_extrinsics.shape[0]
        
        # format camera intrinsics for export
        camera_intrinsics = torch.zeros(size=(n_cameras,3,3), dtype=torch.float32)
        
        print("Camera intrinsics {}: {}".format(camera_intrinsics.shape, camera_intrinsics))

        camera_intrinsics[:,0,0] = focal_length
        camera_intrinsics[:,1,1] = focal_length
        camera_intrinsics[:,0,2] = self.principal_point_x
        camera_intrinsics[:,1,2] = self.principal_point_y
        
        # use index [2,2] in intrinsics to store the image index (of full dataset) that the extrinsics/intrinsics correspond to
        image_indices = torch.arange(start=0, end=self.args.number_of_images_in_training_dataset) * self.skip_every_n_images_for_training
        camera_intrinsics[:, 2, 2] = image_indices.to(torch.float32)

        # use indices [1,0] and [2,0] in intrinsics to store the height and width of the iamge data that the extrinsics/intrinsics correspond to
        camera_intrinsics[:, 1, 0] = self.H
        camera_intrinsics[:, 2, 0] = self.W
                
        if not skip_saving:
            camera_intrinsics_dir = '{}camera_intrinsics_{}.pt'.format(output_directory, epoch)
            camera_extrinsics_dir = '{}camera_extrinsics_{}.pt'.format(output_directory, epoch)

            print("Saving camera intrinsics and extrinsics to {} and {}".format(camera_extrinsics_dir, camera_intrinsics_dir))
            torch.save(camera_intrinsics, camera_intrinsics_dir)
            torch.save(camera_extrinsics, camera_extrinsics_dir)

        if print_debug_info:
            print('image indices: ')
            print(camera_intrinsics[:5, 2, 2])
            print('H and W:')
            print(camera_intrinsics[:5, 1, 0])
            print(camera_intrinsics[:5, 2, 0])
            print('fl x and y: ')
            print(camera_intrinsics[:5, 0, 0])
            print(camera_intrinsics[:5, 1, 1])
            print('pp_x and pp_y: ')
            print(camera_intrinsics[:5, 0, 2])
            print(camera_intrinsics[:5, 1, 2])        

        return (camera_extrinsics, camera_intrinsics)


    def create_blender_cameras_for_nerf2mesh(self):
        
        camera_extrinsics, camera_intrinsics = self.export_camera_extrinsics_and_intrinsics_from_trained_model(output_directory=None, epoch=None, skip_saving=True)
        n_cameras = camera_extrinsics.shape[0]

        n_val_test_samples = 10

        import math
        frequency_of_val_test_images = math.floor(n_cameras / n_val_test_samples)

        project_name = self.args.base_directory.split("/")[-1]

        # make directories for BAA-NGP preparing other processing scripts
        baangp_directory = Path("/home/photon/sense/3cology/research/baa-ngp/baangp/data/nerf_synthetic/{}".format(project_name))
        baangp_directory.mkdir(parents=True, exist_ok=True)

        baangp_color_directory = Path("/home/photon/sense/3cology/research/baa-ngp/baangp/data/nerf_synthetic/{}/train".format(project_name))
        baangp_color_directory.mkdir(parents=True, exist_ok=True)

        baangp_color_val_directory = Path("/home/photon/sense/3cology/research/baa-ngp/baangp/data/nerf_synthetic/{}/val".format(project_name))
        baangp_color_val_directory.mkdir(parents=True, exist_ok=True)

        baangp_color_test_directory = Path("/home/photon/sense/3cology/research/baa-ngp/baangp/data/nerf_synthetic/{}/test".format(project_name))
        baangp_color_test_directory.mkdir(parents=True, exist_ok=True)

        baangp_color_directory = Path("/home/photon/sense/3cology/research/baa-ngp/baangp/data/nerf_synthetic/{}/train".format(project_name))
        baangp_color_directory.mkdir(parents=True, exist_ok=True)

        baangp_depth_directory = Path("/home/photon/sense/3cology/research/baa-ngp/baangp/data/nerf_synthetic/{}/depth".format(project_name))
        baangp_depth_directory.mkdir(parents=True, exist_ok=True)

        # make directories for NeRF2Mesh
        nerf2mesh_directory = Path("/home/photon/sense/3cology/research/nerf2mesh/data/{}".format(project_name))
        nerf2mesh_directory.mkdir(parents=True, exist_ok=True)

        nerf2mesh_color_directory = Path("/home/photon/sense/3cology/research/nerf2mesh/data/{}/train".format(project_name))
        nerf2mesh_color_directory.mkdir(parents=True, exist_ok=True)

        nerf2mesh_color_val_directory = Path("/home/photon/sense/3cology/research/nerf2mesh/data/{}/val".format(project_name))
        nerf2mesh_color_val_directory.mkdir(parents=True, exist_ok=True)

        nerf2mesh_color_test_directory = Path("/home/photon/sense/3cology/research/nerf2mesh/data/{}/test".format(project_name))
        nerf2mesh_color_test_directory.mkdir(parents=True, exist_ok=True)

        nerf2mesh_depth_directory = Path("/home/photon/sense/3cology/research/nerf2mesh/data/{}/depth".format(project_name))
        nerf2mesh_depth_directory.mkdir(parents=True, exist_ok=True)

        cameras = []
        test_cameras = []

        print("Camera intrinsics {}: {}".format(camera_intrinsics.shape, camera_intrinsics))
        print("Camera extrinsics {}: {}".format(camera_extrinsics.shape, camera_extrinsics))

        print("Exporting camera data in Blender format for import to nerf2mesh")
        for cam_i, image_id in enumerate(self.image_ids[::self.skip_every_n_images_for_training][:self.args.number_of_images_in_training_dataset]):
            
            print("Cam {}, Image ID {}".format(cam_i, image_id))

            fl_x = camera_intrinsics[cam_i, 0, 0].item()
            fl_y = camera_intrinsics[cam_i, 1, 1].item()

            cx = camera_intrinsics[cam_i, 0, 2].item()
            cy = camera_intrinsics[cam_i, 1, 2].item()
            image_index = camera_intrinsics[cam_i, 2, 2]

            original_rgb_name = '{}.jpg'.format(str(image_id).zfill(6))
            color_image_source_path = "{}/color/{}".format(self.args.base_directory, original_rgb_name)
            
            # export image data to downstream projects
            color_image_dest_baangp_path = "{}/{}".format(baangp_color_directory, original_rgb_name)
            if not os.path.islink(color_image_dest_baangp_path):
                os.symlink(color_image_source_path, color_image_dest_baangp_path)

            color_image_dest_nerf2mesh_path = "{}/{}".format(nerf2mesh_color_directory, original_rgb_name)
            if not os.path.islink(color_image_dest_nerf2mesh_path):
                os.symlink(color_image_source_path, color_image_dest_nerf2mesh_path)

            # export depth data to downstream projects
            depth_file_name = original_rgb_name.replace(".jpg", ".png")
            source_file_path = "{}/depth/{}".format(self.args.base_directory, depth_file_name)
            output_depth_file_name = depth_file_name.replace(".png", ".npy")
            baangp_depth_file_path = "{}/{}".format(baangp_depth_directory, output_depth_file_name)
            nerf2mesh_depth_file_path = "{}/{}".format(nerf2mesh_depth_directory, output_depth_file_name)

            self.export_depth_data(source_depth_file_path=source_file_path, destination_depth_file_path=baangp_depth_file_path)
            self.export_depth_data(source_depth_file_path=source_file_path, destination_depth_file_path=nerf2mesh_depth_file_path, visualize_depth_data=False)

            ground_truth_rgb_img_fname = './train/{}'.format(original_rgb_name)

            # swap y and z axes
            permutation_matrix = torch.tensor([
                [1.0,  0.0,  0.0],
                [0.0,  0.0,  1.0],
                [0.0,  1.0,  0.0]
            ])

            transform_matrix = permutation_matrix @ camera_extrinsics[cam_i, :3, :3]

            transform_matrix_out = torch.zeros((4,4))

            xyz_coor = camera_extrinsics[cam_i, :3, 3]            

            transform_matrix_out[:3, :3] = transform_matrix            
            transform_matrix_out[0, 3] = xyz_coor[0]
            transform_matrix_out[1, 3] = xyz_coor[2]
            transform_matrix_out[2, 3] = xyz_coor[1]
            transform_matrix_out[3, 3] = 1.0

            transform_matrix_out = transform_matrix_out.tolist()

            camera = {
                'fl_x' : fl_x,
                'fl_y' : fl_y,
                'cx' : cx,
                'cy' : cy,
                'frames' : []
            }

            frame = {
                'file_path' : ground_truth_rgb_img_fname,
                'transform_matrix' : transform_matrix_out,
            }

            camera['frames'].append(frame)

            cameras.append(camera)

            if cam_i % frequency_of_val_test_images == 0:
    
                # export image data to downstream projects
                color_image_dest_baangp_path = "{}/{}".format(baangp_color_val_directory, original_rgb_name)
                if not os.path.islink(color_image_dest_baangp_path):
                    os.symlink(color_image_source_path, color_image_dest_baangp_path)

                color_image_dest_baangp_path = "{}/{}".format(baangp_color_test_directory, original_rgb_name)
                if not os.path.islink(color_image_dest_baangp_path):
                    os.symlink(color_image_source_path, color_image_dest_baangp_path)

                color_image_dest_nerf2mesh_path = "{}/{}".format(nerf2mesh_color_val_directory, original_rgb_name)
                if not os.path.islink(color_image_dest_nerf2mesh_path):
                    os.symlink(color_image_source_path, color_image_dest_nerf2mesh_path)

                color_image_dest_nerf2mesh_path = "{}/{}".format(nerf2mesh_color_test_directory, original_rgb_name)
                if not os.path.islink(color_image_dest_nerf2mesh_path):
                    os.symlink(color_image_source_path, color_image_dest_nerf2mesh_path)

                test_cameras.append(camera)

            print("  -> {} (fx,fy) = ({},{}), R/T = {}\n".format(ground_truth_rgb_img_fname, fl_x, fl_y, transform_matrix_out))

        print('Exporting tranforms to {}'.format(self.args.base_directory))



        with open('{}/{}'.format(self.args.base_directory, 'transforms_train.json'), 'w') as f:
            json.dump(cameras, f, indent=4)

        with open('{}/{}'.format(baangp_directory, 'transforms_train.json'), 'w') as f:
            json.dump(cameras, f, indent=4)

        with open('{}/{}'.format(nerf2mesh_directory, 'transforms_train.json'), 'w') as f:
            json.dump(cameras, f, indent=4)

        with open('{}/{}'.format(self.args.base_directory, 'transforms_test.json'), 'w') as f:
            json.dump(test_cameras, f, indent=4)

        with open('{}/{}'.format(baangp_directory, 'transforms_test.json'), 'w') as f:
            json.dump(test_cameras, f, indent=4)

        with open('{}/{}'.format(nerf2mesh_directory, 'transforms_test.json'), 'w') as f:
            json.dump(test_cameras, f, indent=4)

        with open('{}/{}'.format(self.args.base_directory, 'transforms_val.json'), 'w') as f:
            json.dump(test_cameras, f, indent=4)

        with open('{}/{}'.format(baangp_directory, 'transforms_val.json'), 'w') as f:
            json.dump(test_cameras, f, indent=4)

        with open('{}/{}'.format(nerf2mesh_directory, 'transforms_val.json'), 'w') as f:
            json.dump(test_cameras, f, indent=4)


    def export_data_for_nerf2mesh():
        # psuedo code for steps:
        # <list of image ids> = self.image_ids[::skip_every_n_images_for_training][::skip_every_n_images_for_testing]
        # <list of image_ids> and <resolution>:
        # for each image_id in <list of image_ids>,
        #       transfer ground truth rgb image for image_id, downsampled to <resolution>, to 
        #           '{nerf2mesh_data_dir}/{dataset_name}/ground_truth_rgb/' as str(image_id).zfill(6).jpg
        #       render depth image for image_id at <resolution> using pretrained model, transfer to
        #           '{nerf2mesh_data_dir}/{dataset_name}/depth_images/' as str(image_id).zfill(6).png
        #       convert above depth images to numpy, save to 
        #           '{nerf2mesh_data_dir}/{dataset_name}/depths, as str(image_id).zfill(6).npy
        # compute transforms using self.create_blender_cameras_for_nerf2mesh(),
        #       transfer to '{nerf2mesh_data_dir}/{dataset_name}/ as tranforms_train.npy, transforms_test.npy, tranforms_val.npy
        #       (note that this includes the filenames of gt_rgb and depth that nerf2mesh will look for)
        #
        # NOTES: 
        # -- we use "INTER_LINEAR" interpolation to upsample/downsample data in our nerf training/testing, while nerf2mesh uses
        #    "BILINEAR", probably worth testing whether we should match them
        # -- could probably make .jpg or .png for the exported images a choice, not sure whether it would break something in nerf2mesh though
        # -- to save compute, we might be able to get away with rendering depth images at the sensor resolution and upsampling them,
        #    but this is dubious since they're trained at the higher resolution
        # -- no need for confidence data unless we want to try to make sensor data work consistently and/or want to express
        #    confidence for our learned dephts        
        quit()



    def preprocess_training_image_and_depth_data_once_for_fast_loading(self, save_point_cloud=False):
        # make a directory for fast reload of data for this training mode and resolution
        if not os.path.exists(self.directory_for_fast_reload_of_training_params):

            os.makedirs(self.directory_for_fast_reload_of_training_params)

            if save_point_cloud:
                os.makedirs("{}/point_clouds".format(self.directory_for_fast_reload_of_training_params))

            all_depths = []
            all_near_bounds = []
            all_far_bounds = []
            all_confidence = []
            all_images = []
            all_xyz = []

            for i, image_id in enumerate(self.image_ids[::self.skip_every_n_images_for_training][:self.args.number_of_images_in_training_dataset]):
                depth, near_bound, far_bound, confidence = self.load_depth_data(image_id=image_id)
                image, _ = self.load_image_data(image_id=image_id)
                # xyz = get_sensor_xyz_coordinates(self.all_initial_poses[image_id], depth, self.H, self.W, self.principal_point_x, self.principal_point_y, self.initial_focal_length[0].expand(self.H*self.W).cpu()) # (H, W, 3)

                pixel_rows_and_cols = torch.meshgrid(torch.arange(self.H, dtype=torch.float32, device=self.device),
                                                     torch.arange(self.W, dtype=torch.float32, device=self.device),
                                                     indexing='ij'
                )  # (H, W)

                rows = pixel_rows_and_cols[0].flatten()
                cols = pixel_rows_and_cols[1].flatten()    
                
                pixel_directions = compute_pixel_directions(self.initial_focal_length[0].expand(self.H*self.W).cpu(), rows, cols, self.principal_point_x, self.principal_point_y) # (H, W, 3, 1)

                # change to selected_initial_poses
                poses_for_sampling = self.selected_initial_poses[i].unsqueeze(0).expand(self.H*self.W, -1, -1).to(self.device) #.unsqueeze(0).expand(self.H*self.W, -1, -1).to(self.device)
                pixel_directions_for_sampling = pixel_directions.to(device=self.device)
                depths_for_sampling = depth.unsqueeze(1).reshape(self.H*self.W, 1).to(self.device)

                xyz, pixel_directions_world, resampled_depths = volume_sampling(poses=poses_for_sampling, 
                                                                                pixel_directions=pixel_directions_for_sampling, 
                                                                                sampling_depths=depths_for_sampling, 
                                                                                perturb_depths=False)
                xyz = xyz.squeeze()

                depths_in_mm = (depth * 1000).to(dtype=torch.int16)

                print("Preprocessing data for image {} of {}".format(i, self.args.number_of_images_in_training_dataset))
                all_depths.append(depths_in_mm.unsqueeze(0))
                all_near_bounds.append(torch.tensor([[near_bound]]))
                all_far_bounds.append(torch.tensor([[far_bound]]))
                all_confidence.append(confidence.unsqueeze(0))
                all_images.append(image.unsqueeze(0))
                all_xyz.append(xyz.view(-1, 3))

                if save_point_cloud:
                    imageio.imwrite("{}/point_clouds/{}.png".format(self.directory_for_fast_reload_of_training_params, image_id), image.detach().cpu().numpy().astype(np.uint8))     
                    pcd = self.create_point_cloud(xyz.view(-1, 3), image.view(-1, 3).to(torch.float32) / 255, normals=None, flatten_xyz=False, flatten_image=False)
                    o3d.io.write_point_cloud("{}/point_clouds/{}.ply".format(self.directory_for_fast_reload_of_training_params, image_id), pcd, write_ascii = True)

            depths = torch.cat(all_depths, dim=0)
            near_bounds = torch.cat(all_near_bounds, dim=0)
            far_bounds = torch.cat(all_far_bounds, dim=0)
            confidences = torch.cat(all_confidence, dim=0)
            images = torch.cat(all_images, dim=0)
            all_xyz = torch.cat(all_xyz, dim=0)

            xyz_min_values, xyz_max_range = self.compute_normalization_parameters(all_xyz)
            print("From all (x,y,z) data estimated with sensors, min (x,y,z) values = {} with max dimensional range {} (meters)".format(xyz_min_values, xyz_max_range))

            torch.save(depths, "{}/depths.pt".format(self.directory_for_fast_reload_of_training_params))
            torch.save(near_bounds, "{}/near_bounds.pt".format(self.directory_for_fast_reload_of_training_params))
            torch.save(far_bounds, "{}/far_bounds.pt".format(self.directory_for_fast_reload_of_training_params))
            torch.save(confidences, "{}/confidences.pt".format(self.directory_for_fast_reload_of_training_params))
            torch.save(images, "{}/images.pt".format(self.directory_for_fast_reload_of_training_params))
            torch.save(xyz_min_values, "{}/xyz_min_values.pt".format(self.directory_for_fast_reload_of_training_params))
            torch.save(xyz_max_range, "{}/xyz_max_range.pt".format(self.directory_for_fast_reload_of_training_params))
            torch.save(self.selected_initial_poses, "{}/poses.pt".format(self.directory_for_fast_reload_of_training_params))
            torch.save(self.principal_point_x, "{}/principal_point_x.pt".format(self.directory_for_fast_reload_of_training_params))
            torch.save(self.principal_point_y, "{}/principal_point_y.pt".format(self.directory_for_fast_reload_of_training_params))
            torch.save(self.initial_focal_length[0], "{}/focal_length.pt".format(self.directory_for_fast_reload_of_training_params))

            depths_by_train_image = torch.load("{}/depths.pt".format(self.directory_for_fast_reload_of_training_params)).to(device=self.device)
            self.near = torch.min(depths_by_train_image).to(dtype=torch.float32) / 1000
            self.xyz_min_values = torch.load("{}/xyz_min_values.pt".format(self.directory_for_fast_reload_of_training_params)).to(device=self.device)
            self.xyz_max_range = torch.load("{}/xyz_max_range.pt".format(self.directory_for_fast_reload_of_training_params)).to(device=self.device)



    def load_all_images_ids(self):
        # get images in directory of RGB images
        path_to_images = "{}/{}".format(self.args.base_directory, self.args.images_directory)
        unsorted_image_names = glob.glob("{}/*.{}".format(path_to_images, self.args.images_data_type))

        # extract out numbers of their IDs, and sort images by numerical ID
        self.image_ids = np.asarray(sorted([int(image.split("/")[-1].replace(".{}".format(self.args.images_data_type),"")) for image in unsorted_image_names]))
        self.skip_every_n_images_for_training = int(len(self.image_ids) / self.args.number_of_images_in_training_dataset)                
        
        # load the first image to establish initial height and width of image data because it gets downsampled
        self.load_image_data(0)


    def prepare_test_data(self):        
        self.test_image_indices = range(0, self.args.number_of_test_images * self.args.skip_every_n_images_for_testing, self.args.skip_every_n_images_for_testing)

        if self.args.coarse_sampling_strategy == "sdf": 

            self.test_image_depth_maps = torch.zeros(size=(len(self.test_image_indices), self.H, self.W, 3))
            
            # reconstruct path to previously saved depth maps at 16th, 50th, 84th percentiles 
            data_out_dir = "{}/trained_models".format(self.args.base_directory)            
            experiment_label = "naive"
            experiment_dir = Path(os.path.join(data_out_dir, experiment_label))
            depth_out_dir = Path("{}/depth_renders/".format(experiment_dir))
            epoch = 50001 #self.args.start_epoch

            print("Loading previously extracted depth maps at 16th, 50th, and 84th percentiles of weight distributions, for faster SDF training...")
            # load previously computed depth data for percentiles
            for nth_image, test_image_index in enumerate(self.test_image_indices):

                depth_file_name_fine = os.path.join(depth_out_dir, str(test_image_index).zfill(4) + '_depth_fine_{}.png'.format(epoch))            

                for nth_percentile, percentile in enumerate([0.16, 0.50, 0.84]):
                    depth_map_for_percentile_file_name = depth_file_name_fine.replace(".png", "_{}th_percentile.pt".format(int(100*percentile)))
                    depth_map_for_percentile_file_name = depth_map_for_percentile_file_name.split("/")[-1]                    
                    depth_map_for_percentile_file_path = "{}/depth_percentiles_{}x{}/{}".format(self.args.base_directory, self.H, self.W, depth_map_for_percentile_file_name)
                    depth_map_for_percentile = torch.load(depth_map_for_percentile_file_path)

                    self.test_image_depth_maps[nth_image, :, :, nth_percentile] = depth_map_for_percentile

                    

    def compute_normalization_parameters(self, xyz_coordinates):
        # Step 1: Calculate the min and max for each dimension
        min_values, _ = torch.min(xyz_coordinates, dim=0)
        max_values, _ = torch.max(xyz_coordinates, dim=0)
        
        # Step 2: Determine the dimension with the largest range
        ranges = max_values - min_values
        max_range, _ = torch.max(ranges, dim=0)

        return min_values, max_range

    def normalize_coordinates(self, xyz_coordinates, min_values, max_range):        
        normalized_coordinates = 2 * (xyz_coordinates - min_values) / max_range - 1
        return normalized_coordinates

    def convert_distance_to_normalized(self, distance, max_xyz_range):
        return distance / (max_xyz_range / 2)

    def convert_normalized_distance_to_unnormalized_distance(self, normalized_distance, max_xyz_range):
        unnormalized_distance = normalized_distance * (max_xyz_range / 2)
        return unnormalized_distance

    def denormalize_coordinates(self, normalized_coordinates, min_values, max_range):
        # Convert back to the original range
        original_coordinates = ((normalized_coordinates + 1) * max_range / 2) + min_values
        return original_coordinates


    #########################################################################
    ####### Voxelization logic for "focused" coarse sampling strategy #######
    #########################################################################
    def load_xyz_of_voxel_center_representing_object_occupancy(self):
        self.voxel_xyz_centers = torch.load(self.args.object_voxel_xyz_data_file).to(device=self.device)
        print("Loaded ({}mm)^3 voxel centers representing object occupancy of shape {}: {}...".format(self.args.object_voxel_size_in_meters*1000, 
                                                                                                      self.voxel_xyz_centers.shape,
                                                                                                      self.voxel_xyz_centers))

        self.min_voxel_x = torch.min(self.voxel_xyz_centers[:,0])
        self.max_voxel_x = torch.max(self.voxel_xyz_centers[:,0])
        self.min_voxel_y = torch.min(self.voxel_xyz_centers[:,1])
        self.max_voxel_y = torch.max(self.voxel_xyz_centers[:,1])
        self.min_voxel_z = torch.min(self.voxel_xyz_centers[:,2])
        self.max_voxel_z = torch.max(self.voxel_xyz_centers[:,2])


    def determine_which_pixels_see_object(self, save_xray_of_voxels_hit=False, save_color_coded_depth_bounds=False):
        near_pixel_depths_filepath = "{}/near_pixel_sampling_depth_in_mm.pt".format(self.args.base_directory)
        far_pixel_depths_filepath = "{}/far_pixel_sampling_depth_in_mm.pt".format(self.args.base_directory)

        # self.pixels_that_see_object = torch.zeros(self.args.number_of_images_in_training_dataset, self.H, self.W)
        self.sample_image = torch.zeros(self.H, self.W).to(device=self.device)

        if not os.path.isfile(near_pixel_depths_filepath) or not os.path.isfile(far_pixel_depths_filepath):

            # temporary settings for now
            self.near = torch.tensor(0.0)

            pixels_per_image = self.H * self.W
            
            # gather the latest focal length and poses for all images
            focal_length = self.models['focal']()([0])                        
            poses = self.models['pose']()([0])
        
            # initialize a 16 bit representation of the depth bounds per pixel: 16 bit signed integers range −32,768 to 32,767; here, each integer represents a millimeter
            self.near_pixel_sampling_depth_in_mm = torch.full(size=(self.args.number_of_images_in_training_dataset, self.H, self.W), fill_value=-1, dtype=torch.int16, device=self.device)  
            self.far_pixel_sampling_depth_in_mm = torch.full(size=(self.args.number_of_images_in_training_dataset, self.H, self.W), fill_value=-1, dtype=torch.int16, device=self.device)  

            self.directory_for_fast_reload_of_training_params = "{}/{}_{}x{}".format(self.args.base_directory, self.args.coarse_sampling_strategy, self.args.H_for_training, self.args.W_for_training)
            images_by_train_image = torch.load("{}/images.pt".format(self.directory_for_fast_reload_of_training_params)).to(device=self.device)

            print("Determining for every pixel the distribution (if any) of voxels occupied by object...")     
            # now loop through all of the data, and filter out (only load and save as necessary) based on whether the points land within our focus area
            for i, image_id in enumerate(self.image_ids[::self.skip_every_n_images_for_training][:self.args.number_of_images_in_training_dataset]):
                # create a dense vector of image IDs for all pixels
                image_id_for_all_pixels = torch.full(size=[self.H * self.W], fill_value=i)
                
                # extract out the pose and focal length for this image 
                pose = poses[image_id_for_all_pixels]
                focal_length = focal_length[image_id_for_all_pixels]

                # gather all pixel row and col indices 
                pixel_indices = torch.argwhere(self.sample_image==0)                                                            
                pixel_rows = pixel_indices[:,0]
                pixel_cols = pixel_indices[:,1]

                pixel_directions = compute_pixel_directions(focal_length, pixel_rows, pixel_cols, self.principal_point_x, self.principal_point_y)

                depth_samples = self.sample_depths_near_linearly_far_nonlinearly(number_of_pixels=pixels_per_image, add_noise=False, pixel_directions=pixel_directions, poses=pose, force_naive=True) # (N_pixels, N_samples)            

                # pixel_directions = torch.nn.functional.normalize(pixel_directions, p=2, dim=1)

                pixel_xyz_positions, pixel_directions_world, resampled_depths = volume_sampling(poses=pose, pixel_directions=pixel_directions, sampling_depths=depth_samples, perturb_depths=False)

                valid_pixel_indices, near_depth_bound, far_depth_bound, sum_of_occupied_voxel_hits = self.get_bounds_of_depths_per_pixel_that_focus_on_object(xyz_samples=pixel_xyz_positions, depth_samples=depth_samples)
                
                if len(valid_pixel_indices) > 0:

                    # gather the corresponding pixel rows and cols for the pixel indices that do see the object
                    valid_pixel_rows = pixel_rows[valid_pixel_indices]
                    valid_pixel_cols = pixel_cols[valid_pixel_indices]

                    self.near_pixel_sampling_depth_in_mm[i, valid_pixel_rows, valid_pixel_cols] = (near_depth_bound * 1000).to(dtype=torch.int16)
                    self.far_pixel_sampling_depth_in_mm[i, valid_pixel_rows, valid_pixel_cols] = (far_depth_bound * 1000).to(dtype=torch.int16)

                    if save_xray_of_voxels_hit:
                        color_out_path = Path("{}/{}_voxel_xray.png".format(self.experiment_dir, image_id))  
                        print("  -> Saving X-Ray visualization of voxels occupied by object to {}".format(color_out_path))              
                        image = np.zeros(shape=(self.H,self.W, 3))
                        sum_of_occupied_voxel_hits_on_cpu = sum_of_occupied_voxel_hits.to(device="cpu")
                        print("Sum of occupied pixels {}: {}".format(sum_of_occupied_voxel_hits_on_cpu.shape, sum_of_occupied_voxel_hits_on_cpu))
                        image[pixel_rows.cpu(), pixel_cols.cpu()] = sum_of_occupied_voxel_hits_on_cpu.unsqueeze(1).repeat(1,3).numpy() * 255
                        image = image.astype(np.uint8) 
                        imageio.imwrite(color_out_path, image)     

                    elif save_color_coded_depth_bounds:
                        valid_pixel_rows = pixel_rows[valid_pixel_indices].cpu()
                        valid_pixel_cols = pixel_cols[valid_pixel_indices].cpu()

                        heatmap_near_bound = heatmap_to_pseudo_color(near_depth_bound.cpu(), min_val=0.0, max_val=1.5)
                        heatmap_far_bound = heatmap_to_pseudo_color(far_depth_bound.cpu(), min_val=0.0, max_val=1.5)

                        near_color_out_path = Path("{}/near_{}_depth.png".format(self.experiment_dir, image_id))  
                        far_color_out_path = Path("{}/far_{}_depth.png".format(self.experiment_dir, image_id))  

                        print(" ({}/{}) {}.png -> Saving near depth color-coded visualization of voxels occupied by object to {}".format(i+1, self.args.number_of_images_in_training_dataset, image_id, near_color_out_path))              
                        print(" ({}/{}) {}.png -> Saving far depth color-coded visualization of voxels occupied by object to {}\n".format(i+1, self.args.number_of_images_in_training_dataset, image_id, far_color_out_path))              

                        ground_truth_color_image = (images_by_train_image[i].to(dtype=torch.float32) / 255).cpu().numpy()

                        near_image = np.zeros(shape=(self.H,self.W, 3))
                        # near_image[:,:,3] = 1.0
                        near_image[valid_pixel_rows, valid_pixel_cols, :] = heatmap_near_bound * 0.5 + ground_truth_color_image[valid_pixel_rows, valid_pixel_cols, :] * 0.5
                        near_image = near_image * 255 
                        near_image = near_image.astype(np.uint8) 
                        imageio.imwrite(near_color_out_path, near_image)

                        far_image = np.zeros(shape=(self.H,self.W, 3))
                        # far_image[:,:,3] = 1.0
                        far_image[valid_pixel_rows, valid_pixel_cols, :] = heatmap_far_bound * 0.5 + ground_truth_color_image[valid_pixel_rows, valid_pixel_cols, :] * 0.5
                        far_image = far_image * 255
                        far_image = far_image.astype(np.uint8) 
                        imageio.imwrite(far_color_out_path, far_image)
                        
                    else:
                        print("  -> {} of {} image depth (near, far) bounds processed, based on overlap with voxels occupied by object".format(i+1, self.args.number_of_images_in_training_dataset))

            print("Saving near_pixel_sampling_depth_in_mm.pt and far_pixel_sampling_depth_in_mm.pt for faster run from same pre-trained model directory")
            torch.save(self.near_pixel_sampling_depth_in_mm, near_pixel_depths_filepath)    
            torch.save(self.far_pixel_sampling_depth_in_mm, far_pixel_depths_filepath)

        else:
            self.near_pixel_sampling_depth_in_mm = torch.load(near_pixel_depths_filepath).to(device=self.device)
            self.far_pixel_sampling_depth_in_mm = torch.load(far_pixel_depths_filepath).to(device=self.device)
            print("Loaded previously computed near_pixel_sampling_depth_in_mm.pt ({}) and far_pixel_sampling_depth_in_mm.pt ({})".format(self.near_pixel_sampling_depth_in_mm.shape, self.far_pixel_sampling_depth_in_mm.shape))     


    def create_grid_for_fast_point_in_voxel_lookup(self, expansion_buffer=0.01):
        # Transform voxel center coordinates to integer-based grid coordinates
        object_voxel_grid_unnormalized = (self.voxel_xyz_centers / self.args.object_voxel_size_in_meters).float().round().long()
        print("Created object voxel grid {}: {}".format(object_voxel_grid_unnormalized.shape, object_voxel_grid_unnormalized))
        
        # Determine the bounds of the grid, in integer-based grid coordinates
        self.min_voxel_grid_xyz = object_voxel_grid_unnormalized.min(dim=0)[0] - (torch.tensor(expansion_buffer).to(device=self.device) / self.args.object_voxel_size_in_meters).float().round().long()
        self.max_voxel_grid_xyz = object_voxel_grid_unnormalized.max(dim=0)[0] + (torch.tensor(expansion_buffer).to(device=self.device) / self.args.object_voxel_size_in_meters).float().round().long()
 
        # Determine the minima and maxima of the object, in metric coordinates
        self.min_object_xyz = self.voxel_xyz_centers.min(dim=0)[0] - expansion_buffer
        self.max_object_xyz = self.voxel_xyz_centers.max(dim=0)[0] + expansion_buffer

        # Create a tensor to represent the grid
        grid_size = self.max_voxel_grid_xyz - self.min_voxel_grid_xyz + 1

        self.object_voxel_grid = torch.zeros(*grid_size, dtype=torch.bool, device=self.device)

        indices_of_grid_center_coordinates = tuple((object_voxel_grid_unnormalized - self.min_voxel_grid_xyz).T)

        # Set the elements corresponding to the cube center coordinates to one
        self.object_voxel_grid[indices_of_grid_center_coordinates] = 1


    def get_bounds_of_depths_per_pixel_that_focus_on_object(self, xyz_samples, depth_samples=None):
        # Transform sample coordinates to grid coordinates
        xyz_samples_in_voxel_grid_coordinates = (xyz_samples / self.args.object_voxel_size_in_meters).float().round().long() - self.min_voxel_grid_xyz.unsqueeze(0)

        # Create masks for valid indices: if indices are less than 0, or greater than maximum size of the voxel grid representing the object, then they're out of bounds
        grid_shape_tensor = torch.tensor(self.object_voxel_grid.shape, device=self.device)
        valid_masks = (xyz_samples_in_voxel_grid_coordinates >= 0) & (xyz_samples_in_voxel_grid_coordinates < grid_shape_tensor)

        # All dimensions need to be valid
        valid_indices = valid_masks.all(dim=-1)

        if torch.sum(valid_indices) == 0:
            if type(depth_samples) != type(None):
                # if no pixels see the object, then just return empty tensors
                return torch.tensor([]), torch.tensor([]), torch.tensor([]), sum_of_occupied_voxel_hits
            else:
                return torch.tensor([])

        # Initialize output tensor with zeros (indicating all samples are outside the cubes)
        xyz_samples_are_in_a_voxel_occupied_by_object = torch.zeros(xyz_samples.shape[:-1], dtype=torch.bool, device=self.device)

        # Unwrap indices for tensor-based indexing across (x,y,z) channels
        unwrapped_xyz_coordinates = tuple(xyz_samples_in_voxel_grid_coordinates[valid_indices].T)

        # Check which valid samples are inside a cube
        xyz_samples_are_in_a_voxel_occupied_by_object[valid_indices] = self.object_voxel_grid[unwrapped_xyz_coordinates]

        # determine which indices hit voxels
        indices_of_samples_in_occupied_voxels = torch.argwhere(xyz_samples_are_in_a_voxel_occupied_by_object)

        if type(depth_samples) == type(None):
            return indices_of_samples_in_occupied_voxels[:,0]

        # gather total counts of voxel hits per pixel
        sum_of_occupied_voxel_hits = torch.sum(xyz_samples_are_in_a_voxel_occupied_by_object, dim=1) / 100.0

        # break out indices into the pixels that hit voxels, and the corresponding raycast samples that did
        pixel_indices_in_occupied_voxels = indices_of_samples_in_occupied_voxels[:,0]
        raycast_sample_indices_in_occupied_voxels = indices_of_samples_in_occupied_voxels[:,1]

        # determine the (start, end) points from one pixel to the next, first by checking where the pixel index changes with a diff
        number_of_candidate_depths = pixel_indices_in_occupied_voxels.shape[0]
        pixel_to_pixel_id_diff = torch.diff(pixel_indices_in_occupied_voxels) 
        pixel_to_pixel_transition_indices = torch.argwhere(pixel_to_pixel_id_diff != 0)[:,0]   

        # then compose two lists, one of "starts" and another of "ends", which points to the global start and end index of each pixel's raycasts in voxels
        pixel_index_starts = torch.cat([torch.tensor([0]).to(device=device), pixel_to_pixel_transition_indices + 1])
        pixel_index_ends   = torch.cat([pixel_to_pixel_transition_indices + 1, torch.tensor([number_of_candidate_depths]).to(device=device)])

        # gather just one instance of each of the pixel indices that actually have raycast samples landing in voxels occupied by object
        if len(pixel_indices_in_occupied_voxels) > 0:
            pixel_indices_that_see_voxels_occupied_by_object = pixel_indices_in_occupied_voxels[pixel_index_starts]

            # now, let's get the first raycast that hits the object for this pixel
            first_raycast_sample_inside_voxels_occupied_by_object = raycast_sample_indices_in_occupied_voxels[pixel_index_starts]
            
            # we gather the last raycast in a similar fashion; note that we subtract 1 from the pixel_index_ends, because those end bounds were designed for slicing
            last_raycast_sample_inside_voxels_occupied_by_object = raycast_sample_indices_in_occupied_voxels[pixel_index_ends - 1]

            if type(depth_samples) != type(None):
                # gather the depths in question, for just one pixel (since there is no noise, they're all the same)
                depths = depth_samples[0,:]

                # it is very possible that the pixel only sees one voxel, which means that the first = last raycast depth; that case highlights why it is valuable to add a buffer to the depths            
                buffer = torch.where(first_raycast_sample_inside_voxels_occupied_by_object == last_raycast_sample_inside_voxels_occupied_by_object, self.args.object_voxel_size_in_meters / 4, 0.0)

                near_depth_bound_for_pixels_that_see_object = depths[first_raycast_sample_inside_voxels_occupied_by_object] - buffer #- self.args.object_voxel_size_in_meters / 4

                # clamp to zero in this case if the near bound is below 0
                near_depth_bound_for_pixels_that_see_object = torch.where(near_depth_bound_for_pixels_that_see_object < 0, 0, near_depth_bound_for_pixels_that_see_object)

                far_depth_bound_for_pixels_that_see_object = depths[last_raycast_sample_inside_voxels_occupied_by_object] + buffer #+ self.args.object_voxel_size_in_meters / 4
            
                return pixel_indices_that_see_voxels_occupied_by_object, near_depth_bound_for_pixels_that_see_object, far_depth_bound_for_pixels_that_see_object, sum_of_occupied_voxel_hits

            else:

                return pixel_indices_that_see_voxels_occupied_by_object
        else:

            if type(depth_samples) != type(None):
                # if no pixels see the object, then just return empty tensors
                return torch.tensor([]), torch.tensor([]), torch.tensor([]), sum_of_occupied_voxel_hits
            else:
                return torch.tensor([])


    #########################################################################
    ############ Initialize models and set learning parameters ##############
    #########################################################################
    def initialize_models(self):
        self.models = {}                
        
        model = torch.compile(CameraIntrinsicsModel(self.H, self.W, self.initial_focal_length, self.args.number_of_images_in_training_dataset))
        self.models["focal"] = model
        model.to(self.device)

        model = torch.compile(CameraPoseModel(self.selected_initial_poses))                                
        self.models["pose"] = model
        model.to(self.device)

        if self.args.coarse_sampling_strategy == "sdf":
            model = torch.compile(SSAN_Geometry())
            self.models["ssan_geometry"] = model
            model.to(self.device)

            model = torch.compile(SSAN_Appearance(self.args))
            self.models["ssan_appearance"] = model
            model.to(self.device)

        elif self.args.coarse_sampling_strategy in ["naive", "focused"]:        
            model =  torch.compile(NeRFDensity(self.args))
            self.models["geometry"] = model
            model.to(self.device)

            model = torch.compile(NeRFColor(self.args))
            self.models["color"] = model
            model.to(self.device)

        if self.args.positional_encoding_framework == "NGP":
            self.xyz_max_values = self.xyz_min_values + self.xyz_max_range
            model = HashEmbedder(bounding_box=[self.xyz_min_values, self.xyz_max_values])
            self.models["ngp_positional_encoding"] = model
            model.to(self.device)
            
            model = SHEncoder(input_dim=3, degree=4)
            self.models["ngp_directional_encoding"] = model
            model.to(self.device)



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
        self.optimizers["focal"] = torch.optim.AdamW(self.models["focal"].parameters(), lr=self.args.focal_lr_start)
        self.optimizers["pose"] = torch.optim.AdamW(self.models["pose"].parameters(), lr=self.args.pose_lr_start)

        if self.args.positional_encoding_framework == "NGP":
            self.optimizers["ngp_positional_encoding"] = torch.optim.AdamW(self.models["ngp_positional_encoding"].parameters(), lr=0.01, betas=(0.9, 0.99),  eps=1e-16)
            # self.optimizers["ngp_directional_encoding"] = torch.optim.AdamW(self.models["ngp_directional_encoding"].parameters(), lr=self.args.nerf_color_lr_start, betas=(0.9, 0.999))

        if self.args.coarse_sampling_strategy == "sdf":
            self.optimizers["ssan_geometry"] = torch.optim.AdamW(self.models["ssan_geometry"].parameters(), lr=0.01)
            self.optimizers["ssan_appearance"] = torch.optim.AdamW(self.models["ssan_appearance"].parameters(), lr=0.001) 

        elif self.args.coarse_sampling_strategy in ["naive", "focused"]:        
            self.optimizers["geometry"] = torch.optim.AdamW(self.models["geometry"].parameters(), lr=self.args.nerf_density_lr_start, betas=(0.9, 0.999))
            self.optimizers["color"] = torch.optim.AdamW(self.models["color"].parameters(), lr=self.args.nerf_color_lr_start, betas=(0.9, 0.999))

        self.learning_rates = {}
        self.learning_rates["focal"] = {"start": self.args.focal_lr_start, "end": self.args.focal_lr_end, "exponential_index": self.args.focal_lr_exponential_index, "curvature_shape": self.args.focal_lr_curvature_shape}
        self.learning_rates["pose"] = {"start": self.args.pose_lr_start, "end": self.args.pose_lr_end, "exponential_index": self.args.pose_lr_exponential_index, "curvature_shape": self.args.pose_lr_curvature_shape}
        
        if self.args.coarse_sampling_strategy == "sdf":
            self.learning_rates["ssan_geometry"] = {"start": 0.001, "end": 0.0005, "exponential_index": 4, "curvature_shape": 1}
            self.learning_rates["ssan_appearance"] = {"start": 0.001, "end": 0.0005, "exponential_index": 4, "curvature_shape": 1}

        elif self.args.coarse_sampling_strategy in ["naive", "focused"]:       
            self.learning_rates["geometry"] = {"start": self.args.nerf_density_lr_start, "end": self.args.nerf_density_lr_end, "exponential_index": self.args.nerf_density_lr_exponential_index, "curvature_shape": self.args.nerf_density_lr_curvature_shape}
            self.learning_rates["color"] = {"start": self.args.nerf_color_lr_start, "end": self.args.nerf_color_lr_end, "exponential_index": self.args.nerf_color_lr_exponential_index, "curvature_shape": self.args.nerf_color_lr_curvature_shape}

        if self.args.positional_encoding_framework == "NGP":
            self.learning_rates["ngp_positional_encoding"] = {"start": 0.01, "end": 0.005, "exponential_index": 4, "curvature_shape": 1}
            # self.learning_rates["ngp_directional_encoding"] = {"start": 0.0001, "end": 0.00005, "exponential_index": 4, "curvature_shape": 1}            

        self.schedulers = {}
        self.schedulers["focal"] = self.create_polynomial_learning_rate_schedule(model = "focal")
        self.schedulers["pose"] = self.create_polynomial_learning_rate_schedule(model = "pose")

        if self.args.coarse_sampling_strategy == "sdf":
            self.schedulers["ssan_geometry"] = self.create_polynomial_learning_rate_schedule(model = "ssan_geometry")
            self.schedulers["ssan_appearance"] = self.create_polynomial_learning_rate_schedule(model = "ssan_appearance")
        elif self.args.coarse_sampling_strategy in ["naive", "focused"]:       
            self.schedulers["geometry"] = self.create_polynomial_learning_rate_schedule(model = "geometry")
            self.schedulers["color"] = self.create_polynomial_learning_rate_schedule(model = "color")

        if self.args.positional_encoding_framework == "NGP":
            self.schedulers["ngp_positional_encoding"] = self.create_polynomial_learning_rate_schedule(model = "ngp_positional_encoding")
            # self.schedulers["ngp_directional_encoding"] = self.create_polynomial_learning_rate_schedule(model = "ngp_directional_encoding")           

        self.learning_rate_histories = {}

        self.model_topics = ["pose", "focal"]
        if self.args.coarse_sampling_strategy == "sdf":
            self.model_topics.extend(["ssan_geometry", "ssan_appearance"])
        elif self.args.coarse_sampling_strategy in ["naive", "focused"]:       
            self.model_topics.extend(["color", "geometry"])


        if self.args.coarse_sampling_strategy == "NGP":
            self.model_topics.extend(["ngp_positional_encoding", "ngp_directional_encoding"])

        for topic in self.model_topics:
            self.learning_rate_histories[topic] = []


    def load_pretrained_models(self):        
        print('Loading pretrained model at {}/[model]_{}.pth'.format(self.args.pretrained_models_directory, self.args.start_epoch-1))
        for model_name in self.models.keys():

            if not self.args.extract_sdf_field:

                if model_name in ["ssan_geometry", "ssan_appearance"]:
                    continue

            model_path = "{}/{}_{}.pth".format(self.args.pretrained_models_directory, model_name, self.args.start_epoch-1)            
                        
            # load checkpoint data
            ckpt = torch.load(model_path, map_location=self.device)
            
            # # this fixes loading for a small set of runs (orchid-204, cactus-204, philodendron-204, cycad-204, red_berry_bonsai-204)
            # if model_name == 'focal':
            #     ckpt['model_state_dict']['_orig_mod.fx'] = ckpt['model_state_dict']['_orig_mod.fx'][:self.args.number_of_images_in_training_dataset]
            # elif model_name == 'pose':
            #     ckpt['model_state_dict']['_orig_mod.t'] = ckpt['model_state_dict']['_orig_mod.t'][:self.args.number_of_images_in_training_dataset]
            #     ckpt['model_state_dict']['_orig_mod.r'] = ckpt['model_state_dict']['_orig_mod.r'][:self.args.number_of_images_in_training_dataset]
            #     ckpt['model_state_dict']['_orig_mod.poses'] = ckpt['model_state_dict']['_orig_mod.poses'][:self.args.number_of_images_in_training_dataset]

            # load model from saved state
            model = self.models[model_name]                                
            weights = ckpt['model_state_dict']
            model.load_state_dict(weights, strict=True)            
            
            if self.args.reset_learning_rates == False:
                # load optimizer parameters
                optimizer = self.optimizers[model_name]
                state = ckpt['optimizer_state_dict']
                optimizer.load_state_dict(state)            
                
                # scheduler already has reference to optimizer but needs n_steps (epocs)
                scheduler = self.schedulers[model_name]      
                scheduler.n_steps = ckpt['epoch']
            

    def save_models(self):
        for topic in self.model_topics:
            model = self.models[topic]
            optimizer = self.optimizers[topic]
            print("Saving {} model...".format(topic))                        
            save_checkpoint(epoch=self.epoch-1, model=model, optimizer=optimizer, path=self.experiment_dir, ckpt_name='{}_{}'.format(topic, self.epoch-1))


    def prepare_testing(self):
        if self.args.coarse_sampling_strategy == "sdf":
            directory_label_with_precomputed_training_params = "naive"
        elif self.args.coarse_sampling_strategy in ["naive", "focused"]: 
            directory_label_with_precomputed_training_params = "naive"


        self.directory_for_fast_reload_of_training_params = "{}/{}_{}x{}".format(self.args.base_directory, directory_label_with_precomputed_training_params, self.args.H_for_training, self.args.W_for_training)

        # create meshgrid representing rows and cols, which will be used for rendering full images
        pixel_rows_and_cols_for_test_renders = torch.meshgrid(
            torch.arange(self.args.H_for_test_renders, dtype=torch.float32, device=self.device),
            torch.arange(self.args.W_for_test_renders, dtype=torch.float32, device=self.device),
            indexing='ij'
        )  # (H, W)
     
        self.pixel_rows_for_test_renders = pixel_rows_and_cols_for_test_renders[0].flatten()
        self.pixel_cols_for_test_renders = pixel_rows_and_cols_for_test_renders[1].flatten()


        if os.path.isdir(self.directory_for_fast_reload_of_training_params):
            depths_by_train_image = torch.load("{}/depths.pt".format(self.directory_for_fast_reload_of_training_params)).to(device=self.device)
            self.near = torch.min(depths_by_train_image).to(dtype=torch.float32) / 1000
            self.xyz_min_values = torch.load("{}/xyz_min_values.pt".format(self.directory_for_fast_reload_of_training_params)).to(device=self.device)
            self.xyz_max_range = torch.load("{}/xyz_max_range.pt".format(self.directory_for_fast_reload_of_training_params)).to(device=self.device)

    #########################################################################
    ############################# Sampling ##################################
    #########################################################################
    def sample_training_data(self, visualize_sampled_pixels=False, save_ply_point_clouds_of_sensor_data=False, n_train_poses_per_sampling=32):
        self.rgbd = []
        # self.depths_per_image = []
        self.image_ids_per_pixel = []
        self.relative_image_index_per_pixel = []
        self.pixel_rows = []
        self.pixel_cols = []
        self.confidence_per_pixel = []                      
        neighbor_distance_per_pixel = []
    
        if self.args.coarse_sampling_strategy == "focused":
            self.near_pixel_sampling_depth_in_mm_for_selected_pixels = []
            self.far_pixel_sampling_depth_in_mm_for_selected_pixels = []
            
        # load preprocessed data for sampling
        depths_by_train_image = torch.load("{}/depths.pt".format(self.directory_for_fast_reload_of_training_params)).to(device=self.device)
        near_bounds_by_train_image = torch.load("{}/near_bounds.pt".format(self.directory_for_fast_reload_of_training_params)).to(device=self.device)
        far_bounds_by_train_image = torch.load("{}/far_bounds.pt".format(self.directory_for_fast_reload_of_training_params)).to(device=self.device)
        confidences_by_train_image = torch.load("{}/confidences.pt".format(self.directory_for_fast_reload_of_training_params)).to(device=self.device)
        images_by_train_image = torch.load("{}/images.pt".format(self.directory_for_fast_reload_of_training_params)).to(device=self.device)

        if self.args.coarse_sampling_strategy == "focused":
            self.near_pixel_sampling_depth_in_mm = self.near_pixel_sampling_depth_in_mm.to(device=self.device)

        # now loop through all of the data, and filter out (only load and save as necessary) based on whether the points land within our focus area

        train_image_ids = self.image_ids[::self.skip_every_n_images_for_training][:self.args.number_of_images_in_training_dataset]

        # sub-sample images from total poses, so that we can more efficiently learn poses for subset of images
        n_train_images = train_image_ids.shape[0]
        
        all_valid_image_indices =  torch.from_numpy(np.asarray([i for i in range(len(train_image_ids))])) # if i not in self.bad_pose_indices
        
        samples_of_valid_image_indices = torch.randperm(len(all_valid_image_indices))[:n_train_poses_per_sampling]  # Get n_train_poses_per_sampling random indices
        print("Sampled valid indices {}: {}".format(samples_of_valid_image_indices.shape, samples_of_valid_image_indices))

        neighbor_relative_index = torch.where(samples_of_valid_image_indices != len(all_valid_image_indices) - 1, 1 , -1)
        print("Relative neighbor indices {}: {}".format(neighbor_relative_index.shape, neighbor_relative_index))
        neighbor_valid_indices = samples_of_valid_image_indices + neighbor_relative_index
        print("Neighbor valid indices {}: {}".format(neighbor_valid_indices.shape, neighbor_valid_indices))
        neighbor_pairwise_image_indices_for_training = torch.stack((samples_of_valid_image_indices, neighbor_valid_indices), dim=1).view(-1)

        valid_image_indices = all_valid_image_indices[neighbor_pairwise_image_indices_for_training]
        self.neighbor_pairwise_image_indices_for_training = valid_image_indices

        print("Valid image indices (neighbor pairwise image indices) {}: {}".format(self.neighbor_pairwise_image_indices_for_training.shape, self.neighbor_pairwise_image_indices_for_training))

        #print("Random indices: {}".format(random_indices))

        #valid_image_indices = random_indices #train_image_ids[indices]
        #print("Valid image indices {}: {}".format(valid_image_indices))
        # if self.args.coarse_sampling_strategy in ["naive","focused"]:
        #     valid_image_indices = [i for i in range(len(train_image_ids))]
        # elif self.args.coarse_sampling_strategy == "sdf":
        #     valid_image_indices = self.test_image_indices 

        n_pixels_in_training_dataset = self.args.number_of_pixels_in_training_dataset
        n_images = len(valid_image_indices)
        n_pixels_to_sample_per_image = min(n_pixels_in_training_dataset // n_images, self.H*self.W)
        image_size = self.H * self.W

        # valid_image_ids = valid_image_ids[valid_image_indices]
        # print("Image IDs to be sampled for training {}: {}".format(valid_image_ids.shape, valid_image_ids))
        n_images_processed = -1

        for i, image_id in enumerate(train_image_ids):
            
            if i not in valid_image_indices:
                continue
            else:
                n_images_processed += 1

            # initialize selected pixels of image to zero
            selected_pixels_in_image_canvas = torch.zeros(self.H, self.W)
                
            # get depth data for this image
            depth = depths_by_train_image[i]
            near_bound = near_bounds_by_train_image[i]
            far_bound = far_bounds_by_train_image[i]
            confidence = confidences_by_train_image[i]

            # select a uniformly random subset of those pixels
            all_indices = torch.arange(image_size) #torch.tensor(range(xyz_coordinates.size()[0] * xyz_coordinates.size()[1]))

            if self.args.coarse_sampling_strategy == "focused":
                # determine which pixels actually see the object
                pixel_indices_that_see_voxels_occupied_by_object = torch.argwhere(self.near_pixel_sampling_depth_in_mm[i,:,:] != -1) #[:,0]

                pixel_rows_that_see_voxels_occupied_by_object = pixel_indices_that_see_voxels_occupied_by_object[:,0]
                pixel_cols_that_see_voxels_occupied_by_object = pixel_indices_that_see_voxels_occupied_by_object[:,1]

                # skip this image if there are no pixels that see the object
                number_of_pixels_that_see_object = len(pixel_rows_that_see_voxels_occupied_by_object)
                if number_of_pixels_that_see_object == 0: 
                    continue

                # if there are not many pixels that see the object, then look to only sample those
                if number_of_pixels_that_see_object < n_pixels_to_sample_per_image:
                    number_of_pixels_to_sample_for_era = number_of_pixels_that_see_object
                else:
                    number_of_pixels_to_sample_for_era = n_pixels_to_sample_per_image

                randomly_selected_indices_that_see_object = torch.randint(high = number_of_pixels_that_see_object, size=(number_of_pixels_to_sample_for_era,))
                
                pixel_indices_selected = pixel_indices_that_see_voxels_occupied_by_object[randomly_selected_indices_that_see_object, :]


            else:
                pixel_indices_selected = all_indices[ torch.randint(high = image_size, size=(n_pixels_to_sample_per_image,))]
                selected = torch.zeros(image_size)
                selected[pixel_indices_selected] = 1
                selected = selected.reshape(self.H, self.W)            
                pixel_indices_selected = torch.argwhere(selected==1)            
                number_of_selected_pixels = pixel_indices_selected.size()[0]

            # get the rows and cols of the selected pixel rows and pixel columns
            pixel_rows_selected = pixel_indices_selected[:,0]
            pixel_cols_selected = pixel_indices_selected[:,1]

            self.pixel_rows.append(pixel_rows_selected)
            self.pixel_cols.append(pixel_cols_selected)       

            if self.args.coarse_sampling_strategy == "focused":
                near_pixel_sampling_depth_in_mm_for_this_image = self.near_pixel_sampling_depth_in_mm[i, pixel_rows_selected, pixel_cols_selected]
                far_pixel_sampling_depth_in_mm_for_this_image = self.far_pixel_sampling_depth_in_mm[i, pixel_rows_selected, pixel_cols_selected]

                self.near_pixel_sampling_depth_in_mm_for_selected_pixels.append(near_pixel_sampling_depth_in_mm_for_this_image)
                self.far_pixel_sampling_depth_in_mm_for_selected_pixels.append(far_pixel_sampling_depth_in_mm_for_this_image)

            # get the confidence of the selected pixels            
            selected_confidence = confidence[pixel_rows_selected, pixel_cols_selected]
            self.confidence_per_pixel.append(selected_confidence)

            # convert depth back to meters after we have sub-sampled
            depth_selected = depth[pixel_rows_selected, pixel_cols_selected].to(dtype=torch.float32) / 1000 # (N selected)            

            # now, load the (r,g,b) image and filter only the pixels we're focusing on; convert 8 bit image back to 0-1.0  
            image = images_by_train_image[i].to(dtype=torch.float32) / 255

            rgb_selected = image[pixel_rows_selected, pixel_cols_selected, :] # (N selected, 3)   
            number_of_selected_pixels = rgb_selected.shape[0]         

            # measure how different the selected pixels are from their neighboring pixels
            # (there's probably a more elegant way to do this with built-in tensor functions)
            up = pixel_rows_selected - 1                        
            down = pixel_rows_selected + 1
            left = pixel_cols_selected - 1
            right = pixel_cols_selected + 1
            
            up[torch.argwhere(up < 0)] = 0
            down[torch.argwhere(down > self.H-1)] = self.H-1
            left[torch.argwhere(left < 0)] = 0
            right[torch.argwhere(right > self.W-1)] = self.W-1

            patches = (
                torch.stack([
                    image[up, left],
                    image[up, pixel_cols_selected],
                    image[up, right],
                    image[pixel_rows_selected, left],
                    image[pixel_rows_selected, right],
                    image[down, left],
                    image[down, pixel_cols_selected],
                    image[down, right]
                ], dim=1)
            )

            rgb_selected_expand = rgb_selected.unsqueeze(1).expand(number_of_selected_pixels, 8, 3)                        
            avg_pixel_neighbor_rgb_dist = torch.sqrt(torch.sum((rgb_selected_expand - patches)**2, dim=2))            
            avg_pixel_neighbor_rgb_dist = torch.mean(avg_pixel_neighbor_rgb_dist, dim=1)      
            neighbor_distance_per_pixel.append(avg_pixel_neighbor_rgb_dist)                                    

            # concatenate the (R,G,B) data with the Depth data to create a RGBD vector for each pixel            
            rgbd_selected = torch.cat([rgb_selected, depth_selected.view(-1, 1)], dim=1)
            # self.depths_per_image.append(depth_selected)
            self.rgbd.append(rgbd_selected)                                    

            # now, save this image index, multiplied by the number of pixels selected, in a global vector across all images             
            image_id_for_all_pixels = torch.full(size=[number_of_selected_pixels], fill_value=i)
            self.image_ids_per_pixel.append(image_id_for_all_pixels)
            
            relative_image_index_for_all_pixels = torch.full(size=[number_of_selected_pixels], fill_value=n_images_processed)
            self.relative_image_index_per_pixel.append(relative_image_index_for_all_pixels)

            if visualize_sampled_pixels:
                color_out_path = Path("{}/mask_for_filtering_{}.png".format(self.experiment_dir, image_id))                
                image = np.zeros(shape=(self.H,self.W,3))
                image[pixel_rows_selected, pixel_cols_selected] = ((rgbd_selected[:, :3]).cpu().numpy() * 255)
                image = image.astype(np.uint8) 
                imageio.imwrite(color_out_path, image)                

            # if i % 100 == 0:
            #     print("  -> {} of {} images pre-processed".format(i, self.args.number_of_images_in_training_dataset))

        print("  -> Finished pre-processing image data")

        # bring the data together
        self.rgbd = torch.cat(self.rgbd, dim=0)                        
        self.pixel_rows = torch.cat(self.pixel_rows, dim=0)#.unsqueeze(1)
        self.pixel_cols = torch.cat(self.pixel_cols, dim=0)#.unsqueeze(1)     

        if self.args.coarse_sampling_strategy == "focused":
            self.near_pixel_sampling_depth_in_mm_for_selected_pixels = torch.cat(self.near_pixel_sampling_depth_in_mm_for_selected_pixels, dim=0) 
            self.far_pixel_sampling_depth_in_mm_for_selected_pixels = torch.cat(self.far_pixel_sampling_depth_in_mm_for_selected_pixels, dim=0) 

        self.near = torch.min(self.rgbd[:,3])
        self.far = torch.max(self.rgbd[:,3])        

        self.image_ids_per_pixel = torch.cat(self.image_ids_per_pixel, dim=0).cpu()
        self.relative_image_index_per_pixel = torch.cat(self.relative_image_index_per_pixel, dim=0).cpu()

        self.confidence_per_pixel = torch.cat(self.confidence_per_pixel, dim=0)
        neighbor_distance_per_pixel = torch.cat(neighbor_distance_per_pixel, dim=0)        

        # compute sampling weights
        self.depth_based_pixel_sampling_weights = torch.ones(self.rgbd.size()[0]).cpu()    
        self.depth_based_pixel_sampling_weights = (1 / ((self.rgbd[:,3] + self.near) ** (0.66))).cpu() # bias sampling of closer pixels probabilistically                        
        
        max_rgb_distance = np.sqrt(3)
        steepness = 20.0
        neighbor_rgb_distance_sampling_weights = torch.log2( (steepness * neighbor_distance_per_pixel / max_rgb_distance + 1.0))        
        self.depth_based_pixel_sampling_weights = neighbor_rgb_distance_sampling_weights 
                    
        print("Loaded {} images with {:,} pixels selected".format(len(valid_image_indices), self.image_ids_per_pixel.shape[0] ))


    def interpolate_depths_linearly(self, starts, ends, L):
        L = int(L)
        N = starts.size(0)

        # Reshape the tensors to be (N, 1) so that broadcasting can work
        starts = starts.view(N, 1)
        ends = ends.view(N, 1)

        # Generate the L values between 0 and 1
        t = torch.linspace(0, 1, L).view(1, L).to(device=self.device)

        # Perform the interpolation and return the result
        interpolated_depths = (1 - t) * starts + t * ends

        return interpolated_depths


    def sample_depths_near_linearly_far_nonlinearly(self, number_of_pixels, add_noise=True, test_render=False, pixel_directions=None, poses=None, near_depths=None, far_depths=None, force_naive=False):    
        n_samples = self.args.number_of_samples_outward_per_raycast + 1
        if test_render:
            n_samples = self.args.number_of_samples_outward_per_raycast_for_test_renders + 1

        percentile_of_samples_in_near_region = self.args.percentile_of_samples_in_near_region

        near_min_focus = near_depths #self.near  #   0.091
        near_max_focus = far_depths #self.args.near_maximum_depth # 0.5
        far_max_focus = self.args.far_maximum_depth # 3.0        

        # set additional arguments from sanity checks / identities
        if self.args.coarse_sampling_strategy == "focused" and not force_naive:
            near_min_focus = near_min_focus
            far_min_focus = near_max_focus

        elif self.args.coarse_sampling_strategy == "naive" or force_naive:
            near_min_focus = torch.maximum(self.near, torch.tensor(0.0))
            far_min_focus = self.args.near_maximum_depth        

        # determine number of samples in near region vs. far region
        n_samples_near = torch.floor(torch.tensor(n_samples * percentile_of_samples_in_near_region)) + 1
        n_samples_far = int(n_samples - n_samples_near + 1)     

        if self.args.coarse_sampling_strategy == "focused" and not force_naive:
            near_bins = self.interpolate_depths_linearly(starts=near_min_focus, ends=near_max_focus, L=n_samples_near).to(self.device)

            # compute sample distance for the far region, where the far min is equal to the near max
            far_focus_base = (far_max_focus/far_min_focus)**(1/n_samples_far)

            far_focus_base = far_focus_base.unsqueeze(1).repeat(1,n_samples_far)

            far_sample_numbers = torch.arange(start=0, end=n_samples_far).to(self.device)

            exponentiated_far_focus_base = far_focus_base ** far_sample_numbers  

            far_min_focus = far_min_focus.unsqueeze(1).repeat(1,n_samples_far)

            far_bins = far_min_focus * exponentiated_far_focus_base 

            bins = torch.cat([near_bins, far_bins[:,1:] ], dim=1).to(self.device) 

            # we continue by expanding out the sample distances in the same way as the previous linear depth sampling
            bin_sizes = torch.diff(bins)

            bins = bins[:, :-1]

        elif self.args.coarse_sampling_strategy == "naive" or force_naive:
            near_bins = torch.linspace(near_min_focus, far_min_focus, int(n_samples_near)).to(self.device)        

            # compute sample distance for the far region, where the far min is equal to the near max
            far_focus_base = (far_max_focus/far_min_focus)**(1/n_samples_far)
            far_sample_numbers = torch.arange(start=0, end=n_samples_far).to(self.device)
            far_bins = far_min_focus * far_focus_base ** far_sample_numbers  

            bins = torch.cat([near_bins, far_bins[1:] ]).to(self.device) 

            # we continue by expanding out the sample distances in the same way as the previous linear depth sampling
            bin_sizes = torch.diff(bins)
            bins = bins[:-1]
            bins = bins.unsqueeze(0).expand(number_of_pixels, n_samples-1)

        # we make sure to enable sampling of *any* distance within the bins that have been created from this non-linear discretization
        if add_noise:
            # generate random numbers between [0,1) in the shape of (number_of_pixels, number_of_samples); this is the "entropy" that allows us to sample across everywhere in every bin
            depth_noise = torch.rand(number_of_pixels, n_samples-1, device=self.device, dtype=torch.float32)

            # now shift each sample by [0,1) * bin distances 
            bins = bins + depth_noise * bin_sizes

        return bins


    def sample_depths_nonlinearly(self, number_of_pixels, add_noise=True, test_render=False):
        # unwrap values from arguments
        n_samples = self.args.number_of_samples_outward_per_raycast + 1
        if test_render:
            n_samples = self.args.number_of_samples_outward_per_raycast_for_test_renders + 1
        near_min_focus = self.near  #   0.091
        near_max_focus = self.args.near_maximum_depth # 0.5
        far_max_focus = self.args.far_maximum_depth # 3.0
        percentile_of_samples_in_near_region = self.args.percentile_of_samples_in_near_region

        # set additional arguments from sanity checks / identities
        near_min_focus = torch.maximum(near_min_focus, torch.tensor(0.0))
        far_min_focus = near_max_focus

        # determine number of samples in near region vs. far region
        n_samples_near = torch.floor(torch.tensor(n_samples * percentile_of_samples_in_near_region))
        n_samples_far = n_samples - n_samples_near

        # near_focus_base and far_focus_base are solutions for the problem of (min_focus) * (focus_base) ** (n_samples) = max_focus, expressed as a continuous exponential growth depending on the sample_number (see below)
        near_focus_base = (near_max_focus/near_min_focus)**(1/n_samples_near)
        far_focus_base = (far_max_focus/far_min_focus)**(1/n_samples_far)
        
        # we will return this
        sample_distances = []

        # compute sample distances for the near region, based on above equation, where distance = min_focus * (focus_base ** sample_number)
        near_sample_numbers = torch.arange(start=0, end=n_samples_near).to(self.device)
        near_distances = near_min_focus * near_focus_base ** near_sample_numbers
        sample_distances.append(near_distances)

        # compute sample distance for the far region, where the far min is equal to the near max
        far_sample_numbers = torch.arange(start=0, end=n_samples_far).to(self.device)
        far_distances = far_min_focus * far_focus_base ** far_sample_numbers
        sample_distances.append(far_distances)

        # combine the near and far sample distances
        sample_distances = torch.cat(sample_distances).to(self.device)

        # we continue by expanding out the sample distances in the same way as the previous linear depth sampling
        sample_distances = sample_distances.unsqueeze(0).expand(number_of_pixels, n_samples)        

        # we make sure to enable sampling of *any* distance within the bins that have been created from this non-linear discretization
        if add_noise:
            # generate random numbers between [0,1) in the shape of (number_of_pixels, number_of_samples); this is the "entropy" that allows us to sample across everywhere in every bin
            depth_noise = torch.rand(number_of_pixels, n_samples, device=self.device, dtype=torch.float32)

            # now we need to get the actual bin distances, which have been non-linearly generated from the sampling strategy above; time for a diff (subtraction of neighboring points in a vector)
            bin_distances = torch.diff(sample_distances)

            # add for the 0th sample a 0.0 noise, such that the total number of bin distances equals total depth samples, and the first sample is equal to the minimum depth (i.e. near_min_focus)
            bin_distances = torch.cat([torch.zeros(size=(number_of_pixels,1)).to(self.device), bin_distances], dim=1) 

            # now shift each sample by [0,1) * bin distances 
            noise_shifted_bin_distances = depth_noise * bin_distances
            sample_distances = sample_distances + noise_shifted_bin_distances

            # just in case there is an error with one of the bin lengths being wrong (should be impossible), we sort
            sample_distances = torch.sort(sample_distances, dim=1)[0]

        return sample_distances


    def sample_depths_linearly(self, number_of_pixels, add_noise=True, test_render=False):
        number_of_samples = self.args.number_of_samples_outward_per_raycast + 1
        if test_render:
            number_of_samples = self.args.number_of_samples_outward_per_raycast_for_test_renders + 1

        raycast_distances = torch.linspace(self.near.item(), self.far.item(), number_of_samples).to(self.device)
        raycast_distances = raycast_distances.unsqueeze(0).expand(number_of_pixels, number_of_samples)

        if add_noise:
            depth_noise = torch.rand(number_of_pixels, number_of_samples, device=self.device, dtype=torch.float32)  # (N_pixels, N_samples)
            depth_noise = depth_noise * (self.far.item() - self.near.item()) / number_of_samples # (N_pixels, N_samples)
            raycast_distances = raycast_distances + depth_noise  # (N_pixels, N_samples)            

        return raycast_distances


    def resample_depths_from_nerf_weights(self, weights, depth_samples, resample_padding = 0.01, use_sparse_fine_rendering=False, test_render=False):
        double_edged_weights = [weights[..., :1], weights, weights[..., -1:]]
        weights_pad = torch.cat(double_edged_weights, dim=-1)
        weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
        weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

        # Add in a constant (the sampling function will renormalize the PDF).
        weights = weights_blur + resample_padding
        raycast_distances = depth_samples
        weight_based_depth_samples = self.resample_from_weights_probability_distribution(bins=raycast_distances.to(self.device), weights=weights.to(self.device), use_sparse_fine_rendering=use_sparse_fine_rendering, test_render=test_render)
    
        # equivalent of the official TensorFlow/JAX stop_gradient, to prevent exploding gradients
        weight_based_depth_samples = weight_based_depth_samples.detach()

        return weight_based_depth_samples


    # bins: all of the depth samples [N_pixels, number_of_samples_outward_per_raycast + 1]
    # weights:                       [N_pixels, number_of_samples_outward_per_raycast    ]
    # notes: -when using sparse fine rendering, add_noise should be passed in as False when coarse samples are obtained from sample_depths_linearly
    #        -here, the number of depth samples returned matches the number of bins
    def resample_from_weights_probability_distribution(self, bins, weights, use_sparse_fine_rendering=False, test_render=False):

        if self.args.positional_encoding_framework == "NGP":
            weights = weights[:,:-1]

        number_of_samples = self.args.number_of_samples_outward_per_raycast + 1
        if test_render:
            number_of_samples = self.args.number_of_samples_outward_per_raycast_for_test_renders + 1

        number_of_pixels = weights.shape[0]
        number_of_weights = weights.shape[1]

        if use_sparse_fine_rendering:                   
            weights = weights + 1e-9
            max_weight_indices = torch.argmax(weights.to(device=self.device), 1).to(device=self.device)

            pixel_rows = torch.arange(weights.shape[0]).unsqueeze(1).to(device=self.device)            
            max_weight_indices = max_weight_indices.unsqueeze(1).to(device=self.device)    
            rows_and_max_indices = torch.cat([pixel_rows, max_weight_indices], dim=1).to(device=self.device)                                                            

            max_depths = bins[rows_and_max_indices[:, 0], rows_and_max_indices[:, 1]]
                                                                                  
            bin_length = (self.args.near_maximum_depth - self.near.item()) / (number_of_samples)
            samples = torch.linspace(-bin_length/2.0, bin_length/2.0, number_of_samples).to(self.device)
            
            samples = samples.unsqueeze(0).expand(number_of_pixels, number_of_samples)

            samples = samples + max_depths.unsqueeze(1).expand(number_of_pixels, number_of_samples)

            # don't try to sample outside of [self.near, self.far]            
            out_of_bounds_indices = torch.argwhere(samples > self.args.far_maximum_depth)
            samples[out_of_bounds_indices[:,0], out_of_bounds_indices[:,1]] = self.args.far_maximum_depth
            out_of_bounds_indices = torch.argwhere(samples < self.near)
            samples[out_of_bounds_indices[:,0], out_of_bounds_indices[:,1]] = self.near.item()     
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
        uniform_samples = uniform_samples.unsqueeze(0).expand(number_of_pixels, number_of_samples).contiguous()

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

        if self.args.positional_encoding_framework == "NGP":
            samples = samples[:,:-1]

        return samples.to(self.device)


    def extract_cumulative_distribution_percentiles_from_depths(self, depth_weights, depths, percentiles=[0.16, 0.50, 0.84]):
        number_of_pixels = depth_weights.shape[0]
        number_of_weights = depth_weights.shape[1]

        # first, get the average depth between the "fence posts" depths that were defined for mip-NeRF Gaussian means
        depths = (depths[:,:-1] + depths[:,1:]) / 2

        # save output tensor which will contain the interpolated depth at target cumulative distribution percentiles
        depths_per_percentiles = torch.zeros(number_of_pixels, len(percentiles)).to(device=self.device)

        # create probability distribution function by dividing by weights
        probability_distribution_function = depth_weights / torch.sum(depth_weights, dim=1).unsqueeze(1).expand(number_of_pixels, number_of_weights).to(self.device)

        # get cumulative distribution function, which is the increasing sum of probabilities over the range
        cumulative_distribution_function = torch.cumsum(probability_distribution_function, dim=1).to(self.device)

        for i, percentile in enumerate(percentiles):
            # interpolate the depths from the weights, based on the target depth percentiles (e.g. 16th, 50th, 84th)
            # start by computing the distance
            distances_to_target_depth_percentile = torch.abs(cumulative_distribution_function - percentile)

            # sort the indices of the depth samples (e.g. 1 to 200) for each pixel in increasing order in terms of their distances to the target percentiles
            nearest_to_furthest_depth_weight_indices = torch.argsort(distances_to_target_depth_percentile, dim=1)

            # gather the closest two depth samples for each target percentile
            nearest_depth_weight_index = nearest_to_furthest_depth_weight_indices[:,0]
            pixel_indices = torch.arange(number_of_pixels)
            nearest_depth_weight = cumulative_distribution_function[pixel_indices, nearest_depth_weight_index]

            # for the second nearest, we need to make sure it properly bounds the percentile, so that the percentile is in between the two
            # this depends on whether the nearest depth weight is above or below the target percentile; if above, then get the next one below, and vice versa 
            second_nearest_depth_weight_index = torch.where(nearest_depth_weight - percentile > 0, nearest_depth_weight_index - 1, nearest_depth_weight_index + 1)
            second_nearest_depth_weight = cumulative_distribution_function[pixel_indices, second_nearest_depth_weight_index]

            # based on algebra, derive equation where (coefficient_a * nearest_1st_depth_weight) + (coefficient_b * nearest_2nd_depth_weight) = percentile such that (coefficient_a + coefficient_b) = 1
            # i.e. determine coefficients so a weighted average of the depth percentiles adds up to target percentile 
            # coefficient_b = torch.abs(nearest_depth_weight - percentile) / torch.abs(nearest_depth_weight - second_nearest_depth_weight)
            
            coefficient_b = torch.where(torch.abs(nearest_depth_weight - second_nearest_depth_weight) > 1e-6, torch.abs(nearest_depth_weight - percentile) / torch.abs(nearest_depth_weight - second_nearest_depth_weight), 0.5)

            coefficient_a = 1 - coefficient_b

            nearest_depth = depths[pixel_indices, nearest_depth_weight_index]
            second_nearest_depth = depths[pixel_indices, second_nearest_depth_weight_index]
            derived_depth = coefficient_a * nearest_depth + coefficient_b * second_nearest_depth
            # derived_percentile = coefficient_a * nearest_depth_weight + coefficient_b * second_nearest_depth_weight

            depths_per_percentiles[:,i] = derived_depth

        return depths_per_percentiles


    def gradient(self, y, x, grad_outputs=None):
        if grad_outputs is None:
            grad_outputs = torch.ones_like(y)
        grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
        return grad


    #########################################################################
    ############################## Rendering ################################
    #########################################################################
    def compute_pixel_world_width(self, pixel_directions, pixel_focal_lengths, sampling_depths):
        pixels_shifted_x = (pixel_directions[:, 0] + (1.0 / pixel_focal_lengths)) 
        pixels_shifted_y = pixel_directions[:, 1]
        pixels_shifted_z = pixel_directions[:, 2]
        
        neighbor_pixel_directions = torch.stack([pixels_shifted_x, pixels_shifted_y, pixels_shifted_z], dim=-1)        
        neighbor_pixel_directions = neighbor_pixel_directions / torch.sqrt(torch.sum(neighbor_pixel_directions ** 2,dim=1)).unsqueeze(1).to(torch.device('cuda:0'))
        
        pixel_directions_neighbor_distances = torch.sqrt(torch.sum( (pixel_directions - neighbor_pixel_directions)**2,dim=1)).to(torch.device('cuda:0'))
        pixel_directions_neighbor_distances = pixel_directions_neighbor_distances.unsqueeze(1).expand(sampling_depths.size(0), sampling_depths.size(1)-1).to(torch.device('cuda:0'))

        pixel_world_widths = pixel_directions_neighbor_distances
        return pixel_world_widths


    def visualize_poses(self, poses, size=0.05, bound=1, points=None):
        # poses: [B, 4, 4]

        axes = trimesh.creation.axis(axis_length=4)
        box = trimesh.primitives.Box(extents=[2*bound]*3).as_outline()
        box.colors = np.array([[128, 128, 128]] * len(box.entities))
        objects = [axes, box]    

        if bound > 1:
            unit_box = trimesh.primitives.Box(extents=[2]*3).as_outline()
            unit_box.colors = np.array([[128, 128, 128]] * len(unit_box.entities))
            objects.append(unit_box)

            for pose in poses:
                # a camera is visualized with 8 line segments.
                pos = pose[:3, 3]
                a = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
                b = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
                c = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
                d = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]

                dir = (a + b + c + d) / 4 - pos
                dir = dir / (np.linalg.norm(dir) + 1e-8)
                o = pos + dir * 0.05

                segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
                segs = trimesh.load_path(segs)
                objects.append(segs)

        if points is not None:
            print('[visualize points]', points.shape, points.dtype, points.min(0), points.max(0))
            colors = np.zeros((points.shape[0], 4), dtype=np.uint8)
            colors[:, 2] = 255 # blue
            colors[:, 3] = 30 # transparent
            objects.append(trimesh.PointCloud(points, colors))

        # tmp: verify mesh matches the points
        # mesh = trimesh.load('trial_garden_colmap/mesh_stage0/mesh.ply')
        # objects.append(mesh)

        scene = trimesh.Scene(objects)
        scene.set_camera(distance=bound, center=[0, 0, 0])
        scene.show()


    def normalize_coordinates(self, xyz_coordinates, min_xyz_values, max_xyz_range):        
        normalized_coordinates = 2 * (xyz_coordinates - min_xyz_values) / max_xyz_range - 1
        return normalized_coordinates

    # the core rendering function
    def render(self, poses, pixel_directions, sampling_depths, pixel_focal_lengths, perturb_depths=False, testing=False):
        # pixel_directions = torch.nn.functional.normalize(pixel_directions, p=2, dim=1)
        pixel_xyz_positions, pixel_directions_world, resampled_depths = volume_sampling(poses=poses, pixel_directions=pixel_directions, sampling_depths=sampling_depths, perturb_depths=perturb_depths)
        # pixel_directions_world = torch.nn.functional.normalize(pixel_directions_world, p=2, dim=1)  # (N_pixels, 3)        
        pixel_world_widths = self.compute_pixel_world_width(pixel_directions, pixel_focal_lengths, sampling_depths)
        
        # encode direction: (H, W, N_sample, (2L+1)*C = 27)        
        angular_directional_encoding = encode_position(pixel_directions_world, levels=self.args.directional_encoding_fourier_frequencies)  # (N_pixels, 27)

        # sample geometry density network
        if self.args.coarse_sampling_strategy in ["naive", "focused"]:

            if self.args.positional_encoding_framework == "mip":
                angular_directional_encoding = angular_directional_encoding.unsqueeze(1).expand(-1, sampling_depths.size()[1] - 1, -1)  # (N_pixels, N_sample, 27)   

                # encode with MipNeRF Gaussian positional encoding
                xyz_position_encoding, derived_pixel_xyz = encode_ipe(origin_xyz=poses[:, :3, 3], 
                                                                      pixel_directions=pixel_directions_world,
                                                                      sampling_depths=sampling_depths, 
                                                                      pixel_world_widths=pixel_world_widths, 
                                                                      xyz_min_values=self.xyz_min_values, 
                                                                      xyz_max_range=self.xyz_max_range)
                xyz_position_encoding = xyz_position_encoding.to(self.device)
                resampled_depths = resampled_depths[:, : -1]

            elif self.args.positional_encoding_framework == "NGP":
                # encode with learnable hash encoding from Instant Neural Graphics Primitives
                xyz_position_encoding = self.models["ngp_positional_encoding"](x=pixel_xyz_positions) # (N_pixels, N_sample, 32)
                angular_directional_encoding = self.models["ngp_directional_encoding"](input=pixel_directions_world)
                angular_directional_encoding = angular_directional_encoding.unsqueeze(1).expand(-1, sampling_depths.size()[1], -1)  # (N_pixels, N_sample, 16)   


            density, features = self.models["geometry"]()([xyz_position_encoding]) # (N_pixels, N_sample, 1), # (N_pixels, N_sample, D)    
            rgb = self.models["color"]()([features, angular_directional_encoding])  # (N_pixels, N_sample, 4)         

            render_result = volume_rendering(rgb=rgb, density=density, depths=resampled_depths)       

            result = {
                'rgb_rendered': render_result['rgb_rendered'],  # (N_pixels, 3)
                'depth_map': render_result['depth_map'],        # (N_pixels)
                'depth_weights': render_result['weight'],       # (N_pixels, N_sample),
                'distances': render_result['distances'],
            }

            return result

        elif self.args.coarse_sampling_strategy == "sdf":
            if self.epoch - self.args.start_epoch > 1000:
                small_constant = 0.5
            else:
                small_constant = 0.0

            if self.epoch % 1000 == 0:
                original_sampling_depths = sampling_depths[:,1].clone()

            with torch.no_grad():
                # for the 50th percentile, achieve better estimate of zero level set by recomputing position based on 4 projections toward SDF gradient
                for i in range(4):
                     # normalize (x,y,z) values between -1 and 1
                    normalized_pixel_xyz_positions_50th_percentile = self.normalize_coordinates(xyz_coordinates=pixel_xyz_positions[:, 1], min_xyz_values=self.xyz_min_values, max_xyz_range=self.xyz_max_range)

                    # apply classic NeRF position encoding
                    xyz_position_encoding_50th_percentile = encode_position(input=normalized_pixel_xyz_positions_50th_percentile, levels=8, inc_input=True)

                    # produce SDF, normals, and appearance features from the SSAN geometry MLP
                    sdf_50th_percentile = self.models["ssan_geometry"]()([xyz_position_encoding_50th_percentile.unsqueeze(1), False])

                    unnormalized_sdf_distance = self.convert_normalized_distance_to_unnormalized_distance(normalized_distance=sdf_50th_percentile[:,0], max_xyz_range=self.xyz_max_range)

                    sampling_depths[:, 1] = sampling_depths[:, 1] + small_constant * unnormalized_sdf_distance

                    # resample new (x,y,z) position with updated 50 percentile sampling depth
                    pixel_xyz_positions_50th_percentile, pixel_directions_world_50th_percentile, _ = volume_sampling(poses=poses, pixel_directions=pixel_directions, sampling_depths=sampling_depths[:,1].unsqueeze(1), perturb_depths=perturb_depths)
                    
                    pixel_directions_world_50th_percentile = torch.nn.functional.normalize(pixel_directions_world_50th_percentile, p=2, dim=1)  # (N_pixels, 3)        

                    # encode direction: (H, W, N_sample, (2L+1)*C = 27)        
                    angular_directional_encoding_50th_percentile = encode_position(pixel_directions_world_50th_percentile, levels=self.args.directional_encoding_fourier_frequencies)  # (N_pixels, 27)

                    angular_directional_encoding[:, 1] = angular_directional_encoding_50th_percentile[:,0]
                    pixel_xyz_positions[:, 1] = pixel_xyz_positions_50th_percentile[:,0]

            if self.epoch % 1000 == 0:
                sampling_projected_distance = torch.abs(sampling_depths[:,1] - original_sampling_depths)
                average_sampling_projected_distance = torch.mean(sampling_projected_distance)
                print("    -> After projection, average sampling depth change (in meters) by SDF model = {:.5f}, e.g. {}\n".format(average_sampling_projected_distance, sampling_projected_distance))

            if not testing:
                # track (x,y,z) gradient, used to constrain normals
                pixel_xyz_positions = pixel_xyz_positions.clone().detach().requires_grad_(True)

            # normalize (x,y,z) values between -1 and 1
            normalized_pixel_xyz_positions = self.normalize_coordinates(xyz_coordinates=pixel_xyz_positions, min_xyz_values=self.xyz_min_values, max_xyz_range=self.xyz_max_range)

            # apply classic NeRF position encoding
            xyz_position_encoding = encode_position(input=normalized_pixel_xyz_positions, levels=8, inc_input=True)

            # produce SDF, normals, and appearance features from the SSAN geometry MLP
            sdf_50th_percentile, predicted_normals_50th_percentile, appearance_features_50th_percentile = self.models["ssan_geometry"]()([xyz_position_encoding[:,1].unsqueeze(1)])

            xyz_encoding_without_50th_percentile = torch.cat( [ xyz_position_encoding[:,0].unsqueeze(1), xyz_position_encoding[:,2:] ], dim = 1 )

            sdf_without_50th_percentile = self.models["ssan_geometry"]()([xyz_encoding_without_50th_percentile, False])

            sdf_16th_percentile = sdf_without_50th_percentile[:,0]
            sdf_84th_percentile = sdf_without_50th_percentile[:,1]
            sdf_other_samples = sdf_without_50th_percentile[:,2:]

            sdf = torch.cat( [sdf_16th_percentile.unsqueeze(1), sdf_50th_percentile, sdf_84th_percentile.unsqueeze(1), sdf_other_samples], dim = 1) 

            if not testing:
                # track gradient of (x,y,z) positions for TSDF Eikonal regulization
                # e.g. see https://github.com/lioryariv/volsdf/blob/main/code/model/network.py
                sdf_xyz_gradient = self.gradient(sdf, pixel_xyz_positions)

                # minimize second derivative of TSDF, following https://research.nvidia.com/labs/dir/neuralangelo/paper.pdf
                sdf_xyz_second_gradient = self.gradient(sdf_xyz_gradient, pixel_xyz_positions)
            else:
                sdf_xyz_gradient = None
                sdf_xyz_second_gradient = None

            # expand angular directional encoding per sample
            angular_directional_encoding_50th_percentile = angular_directional_encoding_50th_percentile.unsqueeze(1).expand(-1, 1, -1)  # (N_pixels, 1, 27)   

            # concatenate inputs for the appearance network
            color_input_encoding_50th_percentile = torch.cat([appearance_features_50th_percentile, angular_directional_encoding_50th_percentile, predicted_normals_50th_percentile], dim=-1)

            # compute an (R,G,B) value for each of the samples
            rgb_50th_percentile = self.models["ssan_appearance"]()([color_input_encoding_50th_percentile]) # (N_pixels, N_sample, 3)        

            result = {}
            result['pixel_xyz_positions'] = pixel_xyz_positions
            result['sdf'] = sdf
            result['predicted_normals_50th_percentile'] = predicted_normals_50th_percentile[:,0]
            result['pixel_directions'] = pixel_directions_world
            result['sdf_xyz_gradient'] = sdf_xyz_gradient
            result['sdf_xyz_second_gradient'] = sdf_xyz_second_gradient
            result['rgb_50th_percentile'] = rgb_50th_percentile[:,0]
            # result['valid_pixel_indices'] = valid_pixel_indices

            return result


    # invoke current model for input poses/focal lengths
    # for visual results, supply result to save_render_as_png
    # pose : (N, 4, 4)
    # focal_lengths: (N, 1)
    def render_prediction(self, poses, focal_lengths, principal_point_x, principal_point_y, image_index):
        number_of_pixels = poses.shape[0]

        pixel_directions = compute_pixel_directions(
            focal_lengths, 
            self.pixel_rows_for_test_renders, 
            self.pixel_cols_for_test_renders, 
            principal_point_x=principal_point_x, 
            principal_point_y=principal_point_y
        )

        if self.args.coarse_sampling_strategy == "naive":
            # establish naive linear sampling regime composed primarily of sampling in a near region, with some samples in far region 
            near_depths = torch.tensor([self.near], dtype=torch.float32, device=self.device).expand(number_of_pixels)
            far_depths = torch.tensor([self.args.near_maximum_depth], dtype=torch.float32, device=self.device).expand(number_of_pixels)
        
        elif self.args.coarse_sampling_strategy == "focused":
            flattened_image_near_sampling_depth = self.near_pixel_sampling_depth_in_mm[image_index,:,:].flatten()
            flattened_image_far_sampling_depth = self.far_pixel_sampling_depth_in_mm[image_index,:,:].flatten()
            flattened_focused_pixel_indices = torch.argwhere(flattened_image_near_sampling_depth != -1)[:,0]
            near_depths = flattened_image_near_sampling_depth[flattened_focused_pixel_indices].to(dtype=torch.float32, device="cpu") / 1000.0
            far_depths = flattened_image_far_sampling_depth[flattened_focused_pixel_indices].to(dtype=torch.float32, device="cpu") / 1000.0

            image_near_sampling_depth = self.near_pixel_sampling_depth_in_mm[image_index,:,:]
            focused_pixel_indices = torch.argwhere(image_near_sampling_depth != -1) 
            focused_pixel_rows = focused_pixel_indices[:,0]
            focused_pixel_cols = focused_pixel_indices[:,1]

            poses = poses[flattened_focused_pixel_indices]
            pixel_directions = pixel_directions[flattened_focused_pixel_indices]
            focal_lengths = focal_lengths[flattened_focused_pixel_indices]

        elif self.args.coarse_sampling_strategy == "sdf":

            test_image_index = int(image_index / self.args.skip_every_n_images_for_testing)
            print("Processing test image {}".format(test_image_index))

            self.sample_image = torch.zeros(self.H, self.W)
            pixel_indices = torch.argwhere(self.sample_image==0)                                                            
            pixel_rows = pixel_indices[:,0]
            pixel_cols = pixel_indices[:,1]

            n_pixels = pixel_rows.shape[0]

            test_image_index_for_all_pixels = torch.full(fill_value=test_image_index, size=(n_pixels,))

            depth_samples_50th_percentile = self.test_image_depth_maps[test_image_index_for_all_pixels, pixel_rows, pixel_cols, 1].to(device=self.device)


        # batch the data
        poses_batches = poses.split(self.args.number_of_pixels_per_batch_in_test_renders)
        pixel_directions_batches = pixel_directions.split(self.args.number_of_pixels_per_batch_in_test_renders)
        focal_lengths_batches = focal_lengths.split(self.args.number_of_pixels_per_batch_in_test_renders)      


        if self.args.coarse_sampling_strategy in ["naive", "focused"]:
            near_depths_batches = near_depths.split(self.args.number_of_pixels_per_batch_in_test_renders)
            far_depths_batches = far_depths.split(self.args.number_of_pixels_per_batch_in_test_renders)
        elif self.args.coarse_sampling_strategy == "sdf":
            depth_samples_50th_percentile_batches = depth_samples_50th_percentile.split(self.args.number_of_pixels_per_batch_in_test_renders)


        if self.args.coarse_sampling_strategy in ["naive", "focused"]:

            rendered_image_fine_batches = []
            depth_image_fine_batches = []
            rendered_image_unsparse_fine_batches = []
            depth_image_unsparse_fine_batches = [] 
            depths_per_percentiles_batches = []       
            depth_weights_coarse_batches = []
            
            # for each batch, compute the render and extract RGB and depth map
            for poses_batch, pixel_directions_batch, focal_lengths_batch, near_depths_batch, far_depths_batch in zip(poses_batches, pixel_directions_batches, focal_lengths_batches, near_depths_batches, far_depths_batches):

                poses_batch = poses_batch.to(self.device)
                pixel_directions_batch = pixel_directions_batch.to(self.device)
                focal_lengths_batch = focal_lengths_batch.to(self.device)            
                near_depths_batch = near_depths_batch.to(self.device)
                far_depths_batch = far_depths_batch.to(self.device)

                if poses_batch.shape[0] == 0:
                    break   

                # for resampling with test data, we will compute the NeRF-weighted resamples per batch
                for depth_sampling_optimization in range(self.args.n_depth_sampling_optimizations):
                    # get the depth samples per pixel
                    if depth_sampling_optimization == 0:
                        # if this is the first iteration, collect linear depth samples to query NeRF, uniformly in space                                        
                        depth_samples_coarse = self.sample_depths_near_linearly_far_nonlinearly(number_of_pixels=poses_batch.size()[0], add_noise=False, test_render=True, pixel_directions=pixel_directions_batch, poses=poses_batch, near_depths=near_depths_batch, far_depths=far_depths_batch) # (N_pixels, N_samples)
                        rendered_data_coarse = self.render(poses=poses_batch, pixel_directions=pixel_directions_batch, sampling_depths=depth_samples_coarse, pixel_focal_lengths=focal_lengths_batch, perturb_depths=False)  # (N_pixels, 3)
                    else:
                        # if this is not the first iteration, then resample with the latest weights         
                        depth_samples_fine = self.resample_depths_from_nerf_weights(weights=rendered_data_coarse['depth_weights'], depth_samples=depth_samples_coarse, use_sparse_fine_rendering=self.args.use_sparse_fine_rendering, test_render=True)  # (N_pixels, N_samples)
                        rendered_data_fine = self.render(poses=poses_batch, pixel_directions=pixel_directions_batch, sampling_depths=depth_samples_fine, pixel_focal_lengths=focal_lengths_batch, perturb_depths=False)  # (N_pixels, 3)
                        
                        if self.args.use_sparse_fine_rendering:
                            # point cloud filtering needs both the sparsely rendered data the "unsparsely" (normal) rendered data
                            depth_samples_unsparse_fine = self.resample_depths_from_nerf_weights(weights=rendered_data_coarse['depth_weights'], depth_samples=depth_samples_coarse, use_sparse_fine_rendering=False, test_render=True)  # (N_pixels, N_samples)                                    
                            unsparse_rendered_data_fine = self.render(poses=poses_batch, pixel_directions=pixel_directions_batch, sampling_depths=depth_samples_unsparse_fine, pixel_focal_lengths=focal_lengths_batch, perturb_depths=False)  # (N_pixels, 3)                        

                        if self.args.extract_depth_probability_distributions:
                            if self.args.use_sparse_fine_rendering:
                                depths_per_percentiles = self.extract_cumulative_distribution_percentiles_from_depths(depth_weights=unsparse_rendered_data_fine['depth_weights'], depths=depth_samples_unsparse_fine, percentiles=[0.16, 0.50, 0.84])
                            else:
                                depths_per_percentiles = self.extract_cumulative_distribution_percentiles_from_depths(depth_weights=rendered_data_fine['depth_weights'], depths=depth_samples_fine, percentiles=[0.16, 0.50, 0.84])

                        
                rendered_image_fine_batches.append(rendered_data_fine['rgb_rendered'].cpu()) # (n_pixels_per_row, 3)                
                depth_image_fine_batches.append(rendered_data_fine['depth_map'].cpu()) # (n_pixels_per_row)             
                                    
                if self.args.use_sparse_fine_rendering:
                    rendered_image_unsparse_fine_batches.append(unsparse_rendered_data_fine['rgb_rendered'].cpu())
                    depth_image_unsparse_fine_batches.append(unsparse_rendered_data_fine['depth_map'].cpu())


                if self.args.extract_depth_probability_distributions:
                    depths_per_percentiles_batches.append(depths_per_percentiles)

                depth_weights_coarse_batches.append(rendered_data_coarse['depth_weights'].cpu())  
            
            if len(rendered_image_fine_batches) > 0:

                # combine batch results to compose full images                
                rendered_image_data_fine = torch.cat(rendered_image_fine_batches, dim=0).cpu() # (N_pixels, 3)            
                rendered_depth_data_fine = torch.cat(depth_image_fine_batches, dim=0).cpu()  # (N_pixels)                
                
                if self.args.use_sparse_fine_rendering:
                    rendered_image_data_unsparse_fine = torch.cat(rendered_image_unsparse_fine_batches, dim=0).cpu()
                    depth_image_data_unsparse_fine = torch.cat(depth_image_unsparse_fine_batches, dim=0).cpu()
                
                if self.args.extract_depth_probability_distributions:
                    all_depths_per_percentiles = torch.cat(depths_per_percentiles_batches, dim=0).cpu()

                depth_weights_data_coarse = torch.cat(depth_weights_coarse_batches, dim=0).cpu()  # (N_pixels, N_samples)

                render_result = {
                     'rendered_image_fine': rendered_image_data_fine,
                     'rendered_depth_fine': rendered_depth_data_fine,
                     'depth_weights_coarse': depth_weights_data_coarse,
                }
            
                if self.args.use_sparse_fine_rendering:
                    render_result['rendered_image_unsparse_fine'] = rendered_image_data_unsparse_fine
                    render_result['depth_image_unsparse_fine'] = depth_image_data_unsparse_fine

                render_result['pixel_directions'] = pixel_directions

                if self.args.coarse_sampling_strategy == "focused":
                    render_result['flattened_focused_pixel_indices'] = flattened_focused_pixel_indices
                    render_result['focused_pixel_rows'] = focused_pixel_rows
                    render_result['focused_pixel_cols'] = focused_pixel_cols

                if self.args.extract_depth_probability_distributions:
                    render_result['depths_per_percentiles'] = all_depths_per_percentiles

                return render_result    
            
            else:

                return None

        elif self.args.coarse_sampling_strategy == "sdf":

            rendered_image_batches = []
            depth_image_batches = []

            # for each batch, compute the render and extract RGB and depth map
            for poses_batch, pixel_directions_batch, focal_lengths_batch, depth_samples_50th_percentile_batch in zip(poses_batches, pixel_directions_batches, focal_lengths_batches, depth_samples_50th_percentile_batches):
                # do render with SDF network
                render_result = self.render(poses=poses_batch, pixel_directions=pixel_directions_batch, sampling_depths=depth_samples_50th_percentile_batch.unsqueeze(1), pixel_focal_lengths=focal_lengths_batch, perturb_depths=False, testing=True)
                rendered_image_batches.append(render_result['rgb'].cpu())
                depth_image_batches.append(depth_samples_50th_percentile_batch.cpu())

            if len(rendered_image_batches) > 0:
                # combine batch results to compose full images                
                rendered_image = torch.cat(rendered_image_batches, dim=0).cpu() # (N_pixels, 3)            
                depth_image = torch.cat(depth_image_batches, dim=0).cpu()  # (N_pixels)                
                render_result = {'rendered_image': rendered_image, 'depth_image': depth_image, 'pixel_directions': pixel_directions}

                return render_result
            else:
                return None



    # process raw rendered pixel data and save into images
    def save_render_as_png(self, render_result, H, W, color_file_name_fine, depth_file_name_fine, color_file_name_coarse=None, depth_file_name_coarse=None): 
        if self.args.coarse_sampling_strategy == "naive":
            rendered_rgb_fine = render_result['rendered_image_fine'].reshape(H, W, 3)
            rendered_depth_fine = render_result['rendered_depth_fine'].reshape(H, W)
        elif self.args.coarse_sampling_strategy == "focused":
            rendered_rgb_fine = torch.zeros(self.H, self.W, 3) #.to(device=self.device)
            rendered_depth_fine = torch.zeros(self.H, self.W) #.to(device=self.device)

            rows = render_result['focused_pixel_rows']
            cols = render_result['focused_pixel_cols']

            rendered_rgb_fine[rows, cols, :] = render_result['rendered_image_fine']
            rendered_depth_fine[rows, cols] = render_result['rendered_depth_fine']

        elif self.args.coarse_sampling_strategy == "sdf":
            rendered_rgb_fine = render_result['rendered_image'].reshape(H, W, 3)
            rendered_depth_fine = render_result['depth_image'].reshape(H, W)

        if self.args.extract_depth_probability_distributions and self.args.visualize_depth_probability_distributions:
            percentiles = [0.16, 0.50, 0.84]
            number_of_percentiles = len(percentiles)

            if self.args.coarse_sampling_strategy != "focused":
                depths_per_percentiles = render_result['depths_per_percentiles'].reshape(H, W, number_of_percentiles)
            else:
                depths_per_percentiles = torch.zeros(self.H, self.W, number_of_percentiles) #.to(device=self.device)s
                depths_per_percentiles[rows, cols, :] = render_result['depths_per_percentiles']

        rendered_color_for_file_fine = (rendered_rgb_fine.cpu().numpy() * 255).astype(np.uint8)            

        # get depth map and convert it to Turbo Color Map
        if color_file_name_coarse is not None:
            rendered_rgb_coarse = render_result['rendered_image_coarse'].reshape(H, W, 3)            
            rendered_color_for_file_coarse = (rendered_rgb_coarse.cpu().numpy() * 255).astype(np.uint8)                
            imageio.imwrite(color_file_name_coarse, rendered_color_for_file_coarse)  

            rendered_depth_coarse = render_result['rendered_depth_coarse'].reshape(H, W)                
            rendered_depth_data_coarse = rendered_depth_coarse.cpu().numpy() 
            rendered_depth_for_file_coarse = heatmap_to_pseudo_color(rendered_depth_data_coarse)
            rendered_depth_for_file_coarse = (rendered_depth_for_file_coarse * 255).astype(np.uint8)        
            imageio.imwrite(depth_file_name_coarse, rendered_depth_for_file_coarse)  
        
        rendered_depth_data_fine = rendered_depth_fine.cpu().numpy()         
        rendered_depth_for_file_fine = heatmap_to_pseudo_color(rendered_depth_data_fine)
        rendered_depth_for_file_fine = (rendered_depth_for_file_fine * 255).astype(np.uint8)

        
        rendered_color_for_file_fine = cv2.rotate(rendered_color_for_file_fine, cv2.ROTATE_90_CLOCKWISE)
        rendered_depth_for_file_fine = cv2.rotate(rendered_depth_for_file_fine, cv2.ROTATE_90_CLOCKWISE)


        imageio.imwrite(color_file_name_fine, rendered_color_for_file_fine)
        imageio.imwrite(depth_file_name_fine, rendered_depth_for_file_fine)         

        if self.args.extract_depth_probability_distributions and self.args.visualize_depth_probability_distributions:
            
            for i,percentile in enumerate(percentiles):
                depth_percentile_png_file_name = depth_file_name_fine.replace(".png", "_{}th_percentile.png".format(int(100*percentile)))

                depth_percentile_torch_file_name = depth_file_name_fine.replace(".png", "_{}th_percentile.pt".format(int(100*percentile)))
                depth_percentile_torch_file_name = depth_percentile_torch_file_name.split("/")[-1]
                depth_percentile_torch_file_path = "{}/depth_percentiles_{}x{}/{}".format(self.args.base_directory, self.H, self.W, depth_percentile_torch_file_name)

                depths_for_percentile = depths_per_percentiles[:,:,i]
                torch.save(depths_for_percentile, depth_percentile_torch_file_path)

                depths_for_percentile = depths_for_percentile.cpu().numpy()         
                rendered_depth_for_percentile = heatmap_to_pseudo_color(depths_for_percentile)
                rendered_depth_for_percentile = (rendered_depth_for_percentile * 255).astype(np.uint8)        
                imageio.imwrite(depth_percentile_png_file_name, rendered_depth_for_percentile)  


    #########################################################################
    ############################## Training #################################
    #########################################################################
    def train(self, indices_of_random_pixels):
        # initialize whether each model is in training mode or else is just in evaluation mode (no gradient updates)
        
        if self.epoch >= self.args.start_training_extrinsics_epoch:
            self.models["pose"].train()
        else:
            self.models["pose"].eval()     

        if self.epoch >= self.args.start_training_intrinsics_epoch:
            self.models["focal"].train()
        else:
            self.models["focal"].eval()

        if self.args.coarse_sampling_strategy == "sdf":
            # train our SSAN
            self.models["ssan_geometry"].train()
            self.models["ssan_appearance"].train()

            # keep our pre-trained NeRF models fixed (saves memory)
            self.models["pose"].eval()            
            self.models["focal"].eval()

        elif self.args.coarse_sampling_strategy in ["naive", "focused"]:       

            if self.epoch >= self.args.start_training_geometry_epoch:
                self.models["geometry"].train()
            else:
                self.models["geometry"].eval()

            if self.epoch >= self.args.start_training_color_epoch:
                self.models["color"].train()
            else:
                self.models["color"].eval()


        if self.args.positional_encoding_framework == "NGP":
            self.models["ngp_positional_encoding"].train()

        # get the pixel rows and columns that we've selected (across all images)
        pixel_rows = self.pixel_rows[indices_of_random_pixels]
        pixel_cols = self.pixel_cols[indices_of_random_pixels]

        n_pixels = indices_of_random_pixels.size()[0]

        # get the randomly selected RGBD data        
        rgbd = self.rgbd[indices_of_random_pixels].to(self.device)  # (N_pixels, 4)

        # unpack the image RGB data and the sensor depth
        rgb = rgbd[:,:3].to(self.device)         # (N_pixels, 3)

        if self.args.coarse_sampling_strategy in ["naive", "focused"]:

            if self.args.coarse_sampling_strategy == "naive":
                # establish naive linear sampling regime composed primarily of sampling in a near region, with some samples in far region 
                selected_near_depths = torch.tensor([self.args.near_maximum_depth], dtype=torch.float32, device=self.device).expand(len(indices_of_random_pixels))
                selected_far_depths = torch.tensor([self.near], dtype=torch.float32, device=self.device).expand(len(indices_of_random_pixels))

            elif self.args.coarse_sampling_strategy == "focused":
                # leverage estimated prior knowledge of where the object is to focus coarse sampling only there
                selected_near_depths = self.near_pixel_sampling_depth_in_mm_for_selected_pixels[indices_of_random_pixels].to(dtype=torch.float32, device=self.device) / 1000.0
                selected_far_depths = self.far_pixel_sampling_depth_in_mm_for_selected_pixels[indices_of_random_pixels].to(dtype=torch.float32, device=self.device) / 1000.0

  
            sensor_depth = rgbd[:,3].to(self.device) # (N_pixels) 
            sensor_depth_per_sample = sensor_depth.unsqueeze(1).expand(-1, self.args.number_of_samples_outward_per_raycast) # (N_pixels, N_samples)   


            # ignore depth loss for low-confidence pixels
            selected_confidences = self.confidence_per_pixel[indices_of_random_pixels].to(self.device)
            confidence_loss_weights = torch.where(selected_confidences >= self.args.min_depth_sensor_confidence, 1, 0).to(self.device)
            number_of_pixels_with_confident_depths = torch.sum(confidence_loss_weights)

            # don't try to fit sensor depths that are fixed to max value
            ignore_max_sensor_depths = torch.where(sensor_depth < self.args.far_maximum_depth, 1, 0).to(self.device)


        # initialize our total weighted loss, which will be computed as the weighted sum of coarse and fine losses
        total_weighted_loss = torch.tensor(0.0).to(self.device)
        
        focal_length = self.models['focal']()([0])                        
        poses = self.models['pose']()([0])
                

        # get a tensor with the poses per pixel
        image_ids = self.image_ids_per_pixel[indices_of_random_pixels] # (N_pixels)                
        selected_poses = poses[image_ids]                              # (N_pixels, 4, 4)

        # extract out a mask of bad poses in the selected poses, where 1 means pose is good, 0 means pose is in bad poses list
        unique_image_ids_for_epoch = torch.unique(image_ids)

        currently_training_poses_mask = self.get_mask_filter(all_image_ids=torch.arange(start=0, end=poses.shape[0]), image_ids_to_mask_with_1=unique_image_ids_for_epoch).to(self.device)
        # print("Currently training poses mask {} (sum {}): {}".format(currently_training_poses_mask.shape, torch.sum(currently_training_poses_mask), currently_training_poses_mask))

        bad_poses_mask = self.find_bad_poses(image_ids).to(device=self.device)
        bad_poses_mask_per_pose = self.find_bad_poses(torch.arange(start=0, end=poses.shape[0])).to(device=self.device)

        # print("Bad poses mask {}: {}".format(bad_poses_mask.shape, bad_poses_mask))
        # print("Bad poses mask per pose {}: {}".format(bad_poses_mask_per_pose.shape, bad_poses_mask_per_pose))

        good_poses_mask = 1 - bad_poses_mask
        good_poses_mask_per_pose = 1 - bad_poses_mask_per_pose

        # get the focal lengths and pixel directions for every pixel given the images that were actually selected for each pixel
        selected_focal_lengths = focal_length[image_ids]
        
        pixel_directions_selected = compute_pixel_directions(selected_focal_lengths, pixel_rows, pixel_cols, self.principal_point_x, self.principal_point_y)
 
        if self.args.coarse_sampling_strategy in ["naive", "focused"]:

            for depth_sampling_optimization in range(self.args.n_depth_sampling_optimizations):
                
                #####################| Sampling & Rendering |##################

                if depth_sampling_optimization == 0:
                    # if this is the first iteration, collect linear depth samples to query NeRF, uniformly in space
                    depth_samples_coarse = self.sample_depths_near_linearly_far_nonlinearly(number_of_pixels=n_pixels, add_noise=True, pixel_directions=pixel_directions_selected, poses=selected_poses, near_depths=selected_near_depths, far_depths=selected_far_depths) # (N_pixels, N_samples)
                    depth_samples = depth_samples_coarse.clone()
                else:
                    # if this is not the first iteration, then resample with the latest weights
                    depth_samples = self.resample_depths_from_nerf_weights(weights=nerf_depth_weights, depth_samples=depth_samples, use_sparse_fine_rendering=self.args.use_sparse_fine_rendering)  # (N_pixels, N_samples)                

                    if self.args.use_sparse_fine_rendering:
                        # point cloud filtering needs both the sparsely rendered data and the "unsparsely" (normal) rendered data
                        depth_samples_unsparse_fine = self.resample_depths_from_nerf_weights(weights=nerf_depth_weights, depth_samples=depth_samples, use_sparse_fine_rendering=False)  # (N_pixels, N_samples)                                    
                        unsparse_rendered_data_fine = self.render(poses=selected_poses, pixel_directions=pixel_directions_selected, sampling_depths=depth_samples_unsparse_fine, pixel_focal_lengths=selected_focal_lengths, perturb_depths=False)  # (N_pixels, 3)                        
                        unsparse_fine_depth_map = unsparse_rendered_data_fine["depth_map"]

                n_samples = depth_samples.size()[1]

                # render an image using selected rays, pose, sample intervals, and the network            
                render_result = self.render(poses=selected_poses, pixel_directions=pixel_directions_selected, sampling_depths=depth_samples, pixel_focal_lengths=selected_focal_lengths, perturb_depths=False)  # (N_pixels, 3)    
                rgb_rendered = render_result['rgb_rendered']         # (N_pixels, 3)                                                
                nerf_depth_weights = render_result['depth_weights']  # (N_pixels, N_samples)

                nerf_sample_bin_lengths = depth_samples[:, 1:] - depth_samples[:, :-1]
                nerf_depth_weights = nerf_depth_weights + 1e-7



                #################### Pose Regularlizations ######################

                # #### Penalize pose-to-pose distances, exponentially
                
                # print("Unique image IDs {}: {}".format(unique_image_ids_for_epoch.shape, unique_image_ids_for_epoch))

                # unique_poses_for_epoch = poses[unique_image_ids_for_epoch] 
                # print("Unique poses for epoch {}: {}".format(unique_image_ids_for_epoch.shape, unique_image_ids_for_epoch))

                N_pose_neighbors = len(self.neighbor_pairwise_image_indices_for_training) // 2
                print("N pose neighbors: {}".format(N_pose_neighbors))

                pose_indices_a = self.neighbor_pairwise_image_indices_for_training[::2]
                pose_indices_b = self.neighbor_pairwise_image_indices_for_training[1::2]

                print("Pose indices A {}: {}".format(pose_indices_a.shape, pose_indices_a))
                print("Pose indices B {}: {}".format(pose_indices_b.shape, pose_indices_b))

                poses_a = poses[pose_indices_a]
                poses_b = poses[pose_indices_b]

                print("Poses A {}: {}".format(poses_a.shape, poses_a))
                print("Poses B {}: {}".format(poses_b.shape, poses_b))



                xyz_a, _, _ = volume_sampling(poses=selected_poses, pixel_directions=pixel_directions, sampling_depths=depth_samples, perturb_depths=False)


                ### The experiment was to implement the neighboring pose ICP loss.
                ### As-is, we've got neighboring poses sampled.
                ### We'd want to resample N poses every M epochs (resample_training_data)
                ### Right now, we need to extract some (or all) depth samples + pixel directions (or recompute them) and then do volume_sampling as above, then KNN on the two (x,y,z) sets
                ### That by itself could be a sufficiently meaningful constraint to improve pose learning
                ### However, ablation tests in the Nope-NeRF suggest that an (r,g,b) re-projection error of point clouds is most useful; as-is, the reprojection code appears to require NDC, but could be tested. 
                ### It sometimes seems like there is an endless struggle, but recall what is no longer a struggle, see what is really left. Finish this.




                quit()



                xyz_1 = poses[:-1,:3, 3]
                xyz_2 = poses[1:, :3, 3]

                # n_poses = poses.shape[0]
                pose_to_pose_distances = torch.sum((xyz_1 - xyz_2)**2, dim=-1)
                first_to_last_pose_to_pose_distances = torch.cat([pose_to_pose_distances, torch.tensor([0.0]).to(device=self.device)])
                last_to_first_pose_to_pose_distances = torch.cat([torch.tensor([0.0]).to(device=self.device), pose_to_pose_distances])
                combined_pose_to_pose_distances = first_to_last_pose_to_pose_distances + last_to_first_pose_to_pose_distances
                
                # print("Currently training poses mask {}: {}".format(currently_training_poses_mask.shape, currently_training_poses_mask))
                # print("Bad poses mask {}: {}".format(bad_poses_mask.shape, bad_poses_mask))
                # print("Combined pose to pose distances {}: {}".format(combined_pose_to_pose_distances.shape, combined_pose_to_pose_distances))

                masked_pose_to_pose_distances = torch.mean(currently_training_poses_mask * bad_poses_mask_per_pose * combined_pose_to_pose_distances) #torch.mean(pose_to_pose_distances) #/ n_poses
                # print("Masked pose to pose distance {}: {}".format(masked_pose_to_pose_distances.shape, masked_pose_to_pose_distances))

                # plt.plot(pose_to_pose_distances.detach().cpu().numpy())
                # plt.show()
                
                # confidence_loss_weights = confidence_loss_weights * good_poses_mask

                original_pose_xyz = self.selected_initial_poses[:, :3, 3].to(device=self.device)
                learned_pose_xyz = poses[:, :3, 3]
                original_to_learned_pose_distances = torch.sum((original_pose_xyz - learned_pose_xyz)**2, dim=-1)

                # print("Currently training poses mask {}: {}".format(currently_training_poses_mask.shape, currently_training_poses_mask))
                # print("Good poses mask {}: {}".format(good_poses_mask_per_pose.shape, good_poses_mask_per_pose))
                # print("Original to learned {}: {}".format(original_to_learned_pose_distances.shape, original_to_learned_pose_distances))

                masked_original_to_learned_pose_distances = currently_training_poses_mask * good_poses_mask_per_pose * original_to_learned_pose_distances
                # print("Masked original learned to original pose distance {}: {}".format(masked_original_to_learned_pose_distances.shape, masked_original_to_learned_pose_distances))

                average_original_to_learned_pose_distance = torch.mean(masked_original_to_learned_pose_distances)

                # print("Pose-to-Pose Distances {}: Min = {:.6f}, Mean = {:.6f}, Max = {:.6f}\n{}".format(pose_to_pose_distances.shape, torch.min(pose_to_pose_distances), torch.mean(pose_to_pose_distances), torch.max(pose_to_pose_distances), pose_to_pose_distances))

                # print("Original pose to pose distances {}: {}".format(self.models['pose'].original_pose_to_pose_distances.shape, self.models['pose'].original_pose_to_pose_distances))





                ### Need to visualize movement of poses...
                ### Need trusted poses to be stable, while bad poses need to go fast near trusted poses







                #####################| KL Loss |################################
                # sensor_variance = self.args.depth_sensor_error
                # kl_divergence_bins = -1 * torch.log2(nerf_depth_weights) * torch.exp(-1 * (depth_samples[:, : n_samples - 1] * 1000 - sensor_depth_per_sample[:, : n_samples - 1] * 1000) ** 2 / (2 * sensor_variance)) * nerf_sample_bin_lengths * 1000                                
                # confidence_weighted_kl_divergence_pixels = ignore_max_sensor_depths * confidence_loss_weights * torch.sum(kl_divergence_bins, 1) # (N_pixels)
                #depth_loss = torch.sum(confidence_weighted_kl_divergence_pixels) / number_of_pixels_with_confident_depths

                if self.args.positional_encoding_framework == "NGP":
                    depth_loss = confidence_loss_weights * torch.sum(nerf_depth_weights * (depth_samples - sensor_depth_per_sample) ** 2, dim=1)
                elif self.args.positional_encoding_framework == "mip":
                    depth_loss = confidence_loss_weights * torch.sum(nerf_depth_weights * (depth_samples[:, : n_samples-1] - sensor_depth_per_sample[:, : n_samples -1 ]) ** 2, dim=1)

                depth_loss = torch.sum(depth_loss) / number_of_pixels_with_confident_depths

                depth_to_rgb_importance = self.get_polynomial_decay(start_value=self.args.depth_to_rgb_loss_start, end_value=self.args.depth_to_rgb_loss_end, exponential_index=self.args.depth_to_rgb_loss_exponential_index, curvature_shape=self.args.depth_to_rgb_loss_curvature_shape)
                


                #####################| Entropy Loss |###########################
                entropy_depth_loss = 0.0
                n_near_samples = int(self.args.percentile_of_samples_in_near_region * self.args.number_of_samples_outward_per_raycast)
                mean_entropy = torch.mean(-1 * torch.sum(nerf_depth_weights * torch.log2(nerf_depth_weights), dim=1))
                if (self.epoch >= self.args.entropy_loss_tuning_start_epoch and self.epoch <= self.args.entropy_loss_tuning_end_epoch):                                                
                    epoch = torch.tensor([self.epoch]).float().to(self.device).float()
                    entropy_loss_weight = (torch.tanh(0.01 * epoch / 1000.0) / (1.0 / self.args.max_entropy_weight)).item()
                    entropy_depth_loss = entropy_loss_weight * mean_entropy                
                    # depth_to_rgb_importance = 0.0                                        
                ################################################################

                with torch.no_grad():
                    if self.args.positional_encoding_framework == "NGP":
                        interpretable_depth_loss = confidence_loss_weights * torch.sum(nerf_depth_weights * torch.sqrt((depth_samples * 1000 - sensor_depth_per_sample * 1000) ** 2), dim=1)
                    elif self.args.positional_encoding_framework == "mip":
                        interpretable_depth_loss = confidence_loss_weights * torch.sum(nerf_depth_weights * torch.sqrt((depth_samples[:, : n_samples-1] * 1000 - sensor_depth_per_sample[:, : n_samples -1 ] * 1000) ** 2), dim=1)

                    # get a metric in Euclidian space that we can output via prints for human review/intuition; not actually used in backpropagation
                    interpretable_depth_loss_per_confident_pixel = torch.sum(interpretable_depth_loss) / number_of_pixels_with_confident_depths

                    # get a metric in (0-255) (R,G,B) space that we can output via prints for human review/intuition; not actually used in backpropagation
                    interpretable_rgb_loss = torch.sqrt((rgb_rendered * 255 - rgb * 255) ** 2)
                    interpretable_rgb_loss_per_pixel = torch.mean(interpretable_rgb_loss)  

                
                #####################| MipNeRF-360 Distortion Loss |###########################
                sample_midpoints = (depth_samples[:, 1:] + depth_samples[:, :-1]) / 2

                # only apply distortion loss to coarse sampling
                if depth_sampling_optimization == 0:
                    if self.args.positional_encoding_framework == "NGP":
                        distortion_loss = 0.001 * eff_distloss(nerf_depth_weights[:,:-1], sample_midpoints, nerf_sample_bin_lengths)
                    elif self.args.positional_encoding_framework == "mip":
                        distortion_loss = 0.001 * eff_distloss(nerf_depth_weights, sample_midpoints, nerf_sample_bin_lengths)
                else:
                    distortion_loss = 0.000
                ###############################################################################

                # compute the mean squared difference between the RGB render of the neural network and the original image                             
                rgb_loss = (rgb_rendered - rgb)**2
                rgb_loss = torch.mean(rgb_loss)            

                fine_rgb_loss = 0
                fine_depth_loss = 0
                fine_interpretable_rgb_loss_per_pixel = 0
                fine_interpretable_depth_loss_per_confident_pixel = 0            

                pose_to_pose_loss = 0.0 # masked_pose_to_pose_distances * 200
                learned_pose_loss = 0.0 # average_original_to_learned_pose_distance * 200



                ### Can't really expect to suddenly get the loss adjustments + learning rates (0.1 -> 0.00001?) working properly on first shot.
                ### Need to visualize poses and debug aggressively until there is a clear indication that things are working as intended.
                ### Beware of warping space to "stretch" around "very close poses" with MLP 



                # following official mip-NeRF, if this is the coarse render, we only give 0.1 weight to the total loss contribution; if it is a fine render, then 0.9
                if depth_sampling_optimization == 0:
                    total_weighted_loss = self.args.coarse_weight * (depth_to_rgb_importance * depth_loss + (1 - depth_to_rgb_importance) * rgb_loss + entropy_depth_loss + distortion_loss) + pose_to_pose_loss + learned_pose_loss            
                    coarse_rgb_loss = rgb_loss
                    coarse_depth_loss = depth_loss
                    coarse_interpretable_rgb_loss_per_pixel = interpretable_rgb_loss_per_pixel
                    coarse_interpretable_depth_loss_per_confident_pixel = interpretable_depth_loss_per_confident_pixel
                    coarse_distortion_loss = distortion_loss
                else:
                    # note: KL divergence loss is not used for fine iteration even though it's computed above                
                    total_weighted_loss += (1.0 - self.args.coarse_weight)* ((1 - depth_to_rgb_importance) * rgb_loss + distortion_loss) + pose_to_pose_loss + learned_pose_loss                
                    fine_rgb_loss = rgb_loss        
                    fine_depth_loss = 0
                    fine_interpretable_rgb_loss_per_pixel = interpretable_rgb_loss_per_pixel
                    fine_interpretable_depth_loss_per_confident_pixel = interpretable_depth_loss_per_confident_pixel
                    fine_distortion_loss = distortion_loss

                if self.args.coarse_sampling_strategy == "sdf":
                    total_weighted_loss += +  0.001 * sdf_losses


        elif self.args.coarse_sampling_strategy == "sdf":

            relative_image_indices = self.relative_image_index_per_pixel[indices_of_random_pixels] # (N_pixels)                
            depth_samples_16th_50th_84th_percentiles = self.test_image_depth_maps[relative_image_indices, pixel_rows, pixel_cols, :].to(device=self.device)

            n_percentiles = depth_samples_16th_50th_84th_percentiles.shape[1]
            n_samples_in_between_percentiles = 4 #4

            # print("For SDF training, we've extracted depth maps of 16th, 50th, 84th percentiles {}: {}".format(depth_samples_16th_50th_84th_percentiles.shape, depth_samples_16th_50th_84th_percentiles))

            # Now, save these depth samples, and also add to them a few uniformly randomly sampled values in between those bounds
            depth_samples = torch.zeros(size=(n_pixels, n_percentiles + n_samples_in_between_percentiles)).to(device=self.device)
            depth_samples[:,:3] = depth_samples_16th_50th_84th_percentiles

            # Create random samples between 0 and 1
            random_samples = torch.rand((n_pixels, n_samples_in_between_percentiles)).to(device=self.device)

            # print("Random samples {}: {}".format(random_samples.shape, random_samples))

            # Scale and shift the random samples to be in the intervals
            depth_samples[:,3] = random_samples[:, 0] * depth_samples[:, 1]  # 0 to 16th percentile
            depth_samples[:,4] = random_samples[:, 1] * depth_samples[:, 1]  # 0 to 16th percentile    # depth_samples[:, 1] + 5 * random_samples[:, 1] * (depth_samples[:, 2] - depth_samples[:, 1])  # middle to 5x end
            depth_samples[:,5] = random_samples[:, 2] * depth_samples[:, 1]  # 0 to 16th percentile
            depth_samples[:,6] = random_samples[:, 3] * depth_samples[:, 1]  # 0 to 16th percentile

            # print("Depth samples 1 {}: {}".format(depth_samples[:,3].shape, depth_samples[:,3]))
            # print("Depth samples 2 {}: {}".format(depth_samples[:,4].shape, depth_samples[:,4]))
            # print("Depth samples 3 {}: {}".format(depth_samples[:,5].shape, depth_samples[:,5]))
            # print("Depth samples 4 {}: {}".format(depth_samples[:,6].shape, depth_samples[:,6]))
            # print("50 percentile depth {}: {}".format(depth_samples[:, 1].shape, depth_samples[:, 1]))

            #other_sample_1_positive_epsilon = depth_samples[:,0] / depth_samples[:,3]

            # # Scale and shift so random samples are outside the intervals
            #depth_samples[:,5] = depth_samples[:, 0] * random_samples[:, 2] # collect random sample between 0 and 16th percentile 
            #depth_samples[:,6] = depth_samples[:, 2] * (random_samples[:, 3] * 3 + torch.tensor(1.0).to(device=self.device))  # collect random sample between 1x 84th percentile and 4x 84th percentile


            # pixel_directions = torch.nn.functional.normalize(pixel_directions_selected, p=2, dim=1)
            pixel_xyz_positions, pixel_directions_world, resampled_depths = volume_sampling(poses=selected_poses, pixel_directions=pixel_directions, sampling_depths=depth_samples, perturb_depths=False)
            valid_pixel_indices = self.get_bounds_of_depths_per_pixel_that_focus_on_object(xyz_samples=pixel_xyz_positions)

            all_indices = torch.arange(n_pixels).to(device=self.device)
            
            # concatenate the original subset and all possible indices
            concatenated = torch.cat((valid_pixel_indices, all_indices))

            # find the unique elements in the concatenated tensor
            complement, counts = torch.unique(concatenated, return_counts=True)

            # keep only elements that appear once
            invalid_pixel_indices = complement[counts == 1]

            depth_samples = depth_samples[valid_pixel_indices, :]

            render_result = self.render(poses=selected_poses[valid_pixel_indices], pixel_directions=pixel_directions_selected[valid_pixel_indices], sampling_depths=depth_samples, pixel_focal_lengths=selected_focal_lengths[valid_pixel_indices], perturb_depths=False)

            # valid_pixel_indices = render_result['valid_pixel_indices']
            
            #####################| SDF Isosurface (sum(-D,0,D) = 0) Loss #####################################
            sdf_isosurface_loss_coefficient = 30000000
            sdf = render_result['sdf']
            # positive_depth_sample_distance = depth_samples[:,1] - depth_samples[:,0]
            # negative_depth_sample_distance = depth_samples[:,2] - depth_samples[:,1]
            # positive_depth_sample_distance_normalized = self.convert_distance_to_normalized(distance=positive_depth_sample_distance, max_xyz_range=self.xyz_max_range)
            # negative_depth_sample_distance_normalized = self.convert_distance_to_normalized(distance=negative_depth_sample_distance, max_xyz_range=self.xyz_max_range)
            
            n_valid_pixels = valid_pixel_indices.shape[0]
            # valid_sdf_16th_percentile = sdf[valid_pixel_indices, 0]
            # valid_sdf_50th_percentile = sdf[valid_pixel_indices, 1]
            # valid_sdf_84th_percentile = sdf[valid_pixel_indices, 2]
            # print("{} valid pixels: {} SDF for 16th%".format(n_valid_pixels, valid_sdf_16th_percentile))

            epsilon = torch.tensor(0.005).to(device=self.device)
            sdf_isosurface_loss_per_pixel = (sdf[:,0] - epsilon)**2  + sdf[:,1]**2 + (sdf[:,2] + epsilon)**2
            # sdf_isosurface_loss_per_pixel = (sdf[:,0] - positive_depth_sample_distance_normalized)**2  + sdf[:,1]**2 + (sdf[:,2] + negative_depth_sample_distance_normalized)**2

            sdf_isosurface_loss = torch.sum(sdf_isosurface_loss_per_pixel) / n_valid_pixels * sdf_isosurface_loss_coefficient
            ##################################################################################################

            # #####################| SDF Null Outside Loss #####################################
            sdf_close_up_loss_coefficient = 1
            #sdf = render_result['sdf']
            #near_outside_sdf = sdf[:,5]
            #far_outside_sdf = sdf[:,6]

            sdf_close_up_1 = sdf[:,3]
            sdf_close_up_2 = sdf[:,4]
            sdf_close_up_3 = sdf[:,5]
            sdf_close_up_4 = sdf[:,6]

            # print("SDF close up 1 {}: {}".format(sdf_close_up_1.shape, sdf_close_up_1))
            # print("SDF close up 2 {}: {}".format(sdf_close_up_2.shape, sdf_close_up_2))
            # print("SDF close up 3 {}: {}".format(sdf_close_up_3.shape, sdf_close_up_3))
            # print("SDF close up 4 {}: {}".format(sdf_close_up_4.shape, sdf_close_up_4))

            other_depth_sample_epsilon_ratio_1 = (depth_samples[:, 1] - depth_samples[:,3] + 1e-5) / (depth_samples[:, 1] - depth_samples[:,0] + 1e-5)  
            other_depth_sample_epsilon_ratio_2 = (depth_samples[:, 1] - depth_samples[:,4] + 1e-5) / (depth_samples[:, 1] - depth_samples[:,0] + 1e-5)
            other_depth_sample_epsilon_ratio_3 = (depth_samples[:, 1] - depth_samples[:,5] + 1e-5) / (depth_samples[:, 1] - depth_samples[:,0] + 1e-5)
            other_depth_sample_epsilon_ratio_4 = (depth_samples[:, 1] - depth_samples[:,6] + 1e-5) / (depth_samples[:, 1] - depth_samples[:,0] + 1e-5)

            # print("50 percentile - sample 1 (numerator) {}".format((depth_samples[:, 1] - depth_samples[:,3]) ))
            # print("50 percentile - sample 2 (numerator) {}".format((depth_samples[:, 1] - depth_samples[:,4]) ))
            # print("50 percentile - sample 3 (numerator) {}".format((depth_samples[:, 1] - depth_samples[:,5]) ))
            # print("50 percentile - sample 4 (numerator) {}".format((depth_samples[:, 1] - depth_samples[:,6]) ))

            # print("50 percentile - sample 1 (numerator) + 1e-5 {}".format((depth_samples[:, 1] - depth_samples[:,3] + 1e-5 ) ))
            # print("50 percentile - sample 2 (numerator) + 1e-5 {}".format((depth_samples[:, 1] - depth_samples[:,4] + 1e-5 ) ))
            # print("50 percentile - sample 3 (numerator) + 1e-5 {}".format((depth_samples[:, 1] - depth_samples[:,5] + 1e-5 ) ))
            # print("50 percentile - sample 4 (numerator) + 1e-5 {}".format((depth_samples[:, 1] - depth_samples[:,6] + 1e-5 ) ))

            # print("50 percentile - 16th percentile (denominator) {}".format(depth_samples[:, 1] - depth_samples[:,0]))
            # print("50 percentile - 16th percentile + 1e-5 (denominator) {}".format(depth_samples[:, 1] - depth_samples[:,0] + 1e-5))

            # print("Sample 1 epsilon ratio {}".format(other_depth_sample_epsilon_ratio_1))
            # print("Sample 2 epsilon ratio {}".format(other_depth_sample_epsilon_ratio_2))
            # print("Sample 3 epsilon ratio {}".format(other_depth_sample_epsilon_ratio_3))
            # print("Sample 4 epsilon ratio {}".format(other_depth_sample_epsilon_ratio_4))

            sdf_close_up_loss_1 = torch.mean((sdf_close_up_1 - other_depth_sample_epsilon_ratio_1 * epsilon)**2)
            sdf_close_up_loss_2 = torch.mean((sdf_close_up_2 - other_depth_sample_epsilon_ratio_2 * epsilon)**2)
            sdf_close_up_loss_3 = torch.mean((sdf_close_up_3 - other_depth_sample_epsilon_ratio_3 * epsilon)**2)
            sdf_close_up_loss_4 = torch.mean((sdf_close_up_4 - other_depth_sample_epsilon_ratio_4 * epsilon)**2)

            # print("Squared distance to other depth sample 1 {}".format((sdf_close_up_1 - other_depth_sample_epsilon_ratio_1 * epsilon)**2))
            # print("Squared distance to other depth sample 2 {}".format((sdf_close_up_2 - other_depth_sample_epsilon_ratio_2 * epsilon)**2))
            # print("Squared distance to other depth sample 3 {}".format((sdf_close_up_3 - other_depth_sample_epsilon_ratio_3 * epsilon)**2))
            # print("Squared distance to other depth sample 4 {}".format((sdf_close_up_4 - other_depth_sample_epsilon_ratio_4 * epsilon)**2))

            sdf_close_up_loss = (sdf_close_up_loss_1 + sdf_close_up_loss_2 + sdf_close_up_loss_3 + sdf_close_up_loss_4) * sdf_close_up_loss_coefficient

            # print("{} invalid pixels: {}".format(n_invalid_pixels, invalid_pixel_indices))
            # print("The invalid SDF are {}: {}".format(invalid_sdf.shape, invalid_sdf))
            #invalid_epsilon = torch.tensor(0.006).to(device=self.device)
            #sdf_invalid_loss_per_pixel = (near_outside_sdf - invalid_epsilon)**2 + (far_outside_sdf - invalid_epsilon)**2
            # sdf_invalid_loss = 0.0 #torch.sum(sdf_invalid_loss_per_pixel) / n_valid_pixels * sdf_invalid_loss_coefficient
            # ##################################################################################################

            #####################| SDF Eikonal Loss ||∇|| = 1 |###############################################
            eikonal_curvature_loss_coefficient = 10000
            sdf_xyz_gradient = render_result['sdf_xyz_gradient'] #[valid_pixel_indices, :]
            eikonal_curvature_loss = torch.mean((torch.linalg.norm(sdf_xyz_gradient, ord=2, dim=-1) - 0.1)**2) * eikonal_curvature_loss_coefficient
            ##################################################################################################

            #####################| SDF Smooth Curvature Loss min(|∇^2|) |#####################################
            curvature_smoothness_loss_coefficient = 0.0005
            sdf_xyz_second_gradient = render_result['sdf_xyz_second_gradient'] #[valid_pixel_indices, :]
            curvature_smoothness_loss = torch.mean(torch.linalg.norm(sdf_xyz_second_gradient, ord=1, dim=-1)) * curvature_smoothness_loss_coefficient
            ##################################################################################################

            #####################| SDF Surface Minimization Loss min(|∇^2|) |#################################
            surface_minimization_loss_coefficient = 0
            sdf = render_result['sdf']
            surface_minimization_loss = torch.mean(torch.exp(-1e6 * sdf**2)) * surface_minimization_loss_coefficient
            ##################################################################################################

            #####################| SDF Normal Consistency Loss at 50th percentile (1st index) |##############################################
            normal_consistency_loss_coefficient = 50 #100
            gradient_normals_50th_percentile = sdf_xyz_gradient[:,1] 
            predicted_normals_50th_percentile = render_result['predicted_normals_50th_percentile'] #[valid_pixel_indices,:]
            normal_consistency_loss = torch.mean(torch.linalg.norm(gradient_normals_50th_percentile - predicted_normals_50th_percentile, ord=2, dim=-1)) * normal_consistency_loss_coefficient
            ##################################################################################################

            #####################| SDF Normal Orientation Loss at 50th percentile (1st index) |##############################################
            normal_orientation_loss_coefficient = 0.025 #1
            pixel_directions_50th_percentile = render_result['pixel_directions'] #[valid_pixel_indices,:]
            normal_orientation_dot_product = pixel_directions_50th_percentile[:,0] * gradient_normals_50th_percentile[:,0] + pixel_directions_50th_percentile[:,1] * gradient_normals_50th_percentile[:,1] + pixel_directions_50th_percentile[:,2] * gradient_normals_50th_percentile[:,2]
            normal_orientation_loss = torch.max(torch.zeros(size=(n_valid_pixels,)).to(device=self.device), normal_orientation_dot_product)
            normal_orientation_loss = torch.sum(normal_orientation_loss) * normal_orientation_loss_coefficient
            ##################################################################################################

            #####################| RGB Inverse Rendering Loss |###############################################
            rgb_loss_coefficient = 250
            rgb_50th_percentile = render_result['rgb_50th_percentile']
            rgb_loss = (rgb_50th_percentile - rgb[valid_pixel_indices])**2 
            total_rgb_loss = torch.mean(rgb_loss) * rgb_loss_coefficient
            ##################################################################################################

            total_weighted_loss += sdf_isosurface_loss + eikonal_curvature_loss + curvature_smoothness_loss + surface_minimization_loss + normal_consistency_loss + normal_orientation_loss + total_rgb_loss + sdf_close_up_loss


        for optimizer in self.optimizers.values():
            optimizer.zero_grad() 

        # release unused GPU memory (for memory usage monitoring purposes)
        torch.cuda.empty_cache()
              
        ## backward propagate the gradients to update the values which are parameters to this loss        
        total_weighted_loss.backward(create_graph=False, retain_graph=False)       
        

        if self.args.coarse_sampling_strategy == "sdf":
            # train our SSAN only
            self.optimizers["ssan_geometry"].step()
            self.optimizers["ssan_appearance"].step()

            self.schedulers["ssan_geometry"].step()
            self.schedulers["ssan_appearance"].step()

        elif self.args.coarse_sampling_strategy in ["naive", "focused"]:

            # step each optimizer forward once if currently training
            if self.epoch >= self.args.start_training_color_epoch:
                self.optimizers['color'].step()               
            if self.epoch >= self.args.start_training_extrinsics_epoch:
                self.optimizers['pose'].step()            
            if self.epoch >= self.args.start_training_intrinsics_epoch:
                self.optimizers['focal'].step()            
            if self.epoch >= self.args.start_training_geometry_epoch:
                self.optimizers['geometry'].step()

            # advance all schedulers to preserve schedule consistency
            for scheduler in self.schedulers.values():
                scheduler.step()            

        if self.epoch % self.args.log_frequency == 0:
            minutes_into_experiment = (int(time.time())-int(self.start_time)) / 60

        report = "({} at {:.1f} min)".format(self.epoch, minutes_into_experiment)
        report += " - L={:.5f}".format(total_weighted_loss) 

        if self.args.coarse_sampling_strategy in ["naive", "focused"]:
            weighted_rgb_loss = (1.0 - self.args.coarse_weight)* (1 - depth_to_rgb_importance) * rgb_loss + self.args.coarse_weight * (1 - depth_to_rgb_importance) * rgb_loss
            weighted_depth_loss = self.args.coarse_weight * (depth_to_rgb_importance) * depth_loss

            report += "-> RGB=(C:{:.2f}/255".format(coarse_interpretable_rgb_loss_per_pixel)
            report += ", F:{:.2f}/255".format(fine_interpretable_rgb_loss_per_pixel)
            report += ", #:{:.5f}),".format(weighted_rgb_loss)
            report += " DEPTH=(C:{:.2f}mm".format(coarse_interpretable_depth_loss_per_confident_pixel)
            report += ", F:{:.2f}mm".format(fine_interpretable_depth_loss_per_confident_pixel)
            report += ", #:{:.8f})".format(weighted_depth_loss)
            report += ", Entropy=µ:{:.6f}, #:{:.6f}".format(mean_entropy, entropy_depth_loss)
            report += ", Distortion=(C:{:.6f}, F:{:.6f})".format(coarse_distortion_loss, fine_distortion_loss)
            report += ", ΔT(1->2->3...)={:.6f}, ΔT(est->lrn)={:.6f}".format(pose_to_pose_loss, learned_pose_loss)

        if self.args.coarse_sampling_strategy == "sdf":
            report += ", RGB:{:.5f}/255,".format(torch.mean(torch.sqrt(rgb_loss))*255)

            average_16th_percentile = torch.mean(sdf[:,0])
            average_50th_percentile = torch.mean(sdf[:,1])
            average_84th_percentile = torch.mean(sdf[:,2])

            if average_16th_percentile > 0:
                sign_16_percentile = "+"
            else:
                sign_16_percentile = ""
            if average_50th_percentile > 0:
                sign_50_percentile = "+"
            else:
                sign_50_percentile = ""
            if average_84th_percentile > 0:
                sign_84_percentile = "+"
            else:
                sign_84_percentile = ""
            
            # vs_1 = positive_depth_sample_distance_normalized[0]
            # vs_2 = negative_depth_sample_distance_normalized[0]
            report += " SDF:{:.5f} (e.g. {}{:.2f},{}{:.2f},{}{:.2f}),".format(sdf_isosurface_loss, sign_16_percentile, 200*average_16th_percentile, sign_50_percentile, 200*average_50th_percentile, sign_84_percentile, 200*average_84th_percentile)


            mean_eikonal_magnitude = torch.mean(torch.linalg.norm(sdf_xyz_gradient[:,:5], ord=2, dim=-1))

            # if sdf_xyz_gradient[0,1,0] > 0:
            #     sign_x = "+"
            # else:
            #     sign_x = ""
            # if sdf_xyz_gradient[0,1,1] > 0:
            #     sign_y = "+"
            # else:
            #     sign_y = ""
            # if sdf_xyz_gradient[0,1,2] > 0:
            #     sign_z = "+"
            # else:
            #     sign_z = ""        

            mean_sdf_close_up_epsilon_ratio = torch.mean( torch.mean(torch.abs(sdf[:,3:]), dim=1) / (torch.abs(sdf[:,0]) + 0.000001)  )

            #man_eik_loss = torch.mean((torch.linalg.norm(sdf_xyz_gradient, ord=2, dim=-1) - 10)**2)
            report += " Near Empty Space: {:.5f} ({:.3f}),".format(sdf_close_up_loss, mean_sdf_close_up_epsilon_ratio)
            # report += " Eikonal:{:.5f} (e.g. {}{:.3f},{}{:.3f},{}{:.3f}),".format(eikonal_curvature_loss, sign_x, sdf_xyz_gradient[0,1,0], sign_y, sdf_xyz_gradient[0,1,1], sign_z, sdf_xyz_gradient[0,1,2])
            report += " Eikonal:{:.5f} ({:.4f})".format(eikonal_curvature_loss, mean_eikonal_magnitude)
            report += " Smooth:{:.8f})".format(curvature_smoothness_loss)
            # report += " min(Surf):{:.2f})".format(surface_minimization_loss)
            report += " Nm Cons.:{:.3f}".format(normal_consistency_loss)
            report += " Nm Dir.:{:.3f})".format(normal_orientation_loss)
        
        # print the report of losses
        print(report)

        # a new epoch has dawned
        self.epoch += 1


    def sample_next_batch(self, weighted=True):
        if weighted:
            # subsample set of pixels to do weighted sampling from                        
            # technically this is sampling with replacement, but it shouldn't matter much, and using torch.randperm instead is extremely inefficient
            randomly_sampled_pixel_indices = torch.randint(self.image_ids_per_pixel.shape[0], (self.args.pixel_samples_per_epoch * 10,)).to(device="cpu")
            
            # get 1000x number of samples to collect and do weighted sampling from this subset            
            subsampled_depth_weights = self.depth_based_pixel_sampling_weights[randomly_sampled_pixel_indices]            
            subsampled_indices_from_weights = torch.multinomial(input=subsampled_depth_weights, num_samples=self.args.pixel_samples_per_epoch, replacement=False).to(device="cpu")
            
            # now we need to grab from our global pixel indices which ones we actually selected, after weighted sampling
            indices_of_random_pixels_for_this_epoch = randomly_sampled_pixel_indices[subsampled_indices_from_weights]            
                                                                              
        else:            
            indices_of_random_pixels_for_this_epoch = torch.tensor(random.sample(population=range(self.image_ids_per_pixel.shape[0]), k=self.args.pixel_samples_per_epoch))            

        return indices_of_random_pixels_for_this_epoch
            
    
    #########################################################################
    ############################# Experiments ###############################
    #########################################################################
    def create_experiment_directory(self):
        data_out_dir = "{}/trained_models".format(self.args.base_directory)            
                
        #experiment_label = "{}".format( int(str(self.start_time)[:9]) )            
        experiment_label = self.args.coarse_sampling_strategy
        experiment_dir = Path(os.path.join(data_out_dir, experiment_label))

        # if not self.args.load_pretrained_models:
        #     if experiment_dir.exists() and experiment_dir.is_dir():
        #         shutil.rmtree(experiment_dir)

        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_dir = experiment_dir

        print("Experiment directory: {}".format(self.experiment_dir))

        self.color_out_dir = Path("{}/color_renders/".format(self.experiment_dir))
        self.color_out_dir.mkdir(parents=True, exist_ok=True)

        self.depth_out_dir = Path("{}/depth_renders/".format(self.experiment_dir))
        self.depth_out_dir.mkdir(parents=True, exist_ok=True)

        self.depth_weights_out_dir = Path("{}/depth_weights_visualization/".format(self.experiment_dir))
        self.depth_weights_out_dir.mkdir(parents=True, exist_ok=True)

        self.pointcloud_out_dir = Path("{}/pointclouds/".format(self.experiment_dir))
        self.pointcloud_out_dir.mkdir(parents=True, exist_ok=True)  

        self.geometry_data_out_dir = Path("{}/geometry_data/".format(self.experiment_dir))
        self.geometry_data_out_dir.mkdir(parents=True, exist_ok=True)            

        self.learning_rates_out_dir = Path("{}/learning_rates/".format(self.experiment_dir))
        self.learning_rates_out_dir.mkdir(parents=True, exist_ok=True)              

        self.sampling_data_out_dir = Path("{}/sampling_data/".format(self.experiment_dir))
        self.sampling_data_out_dir.mkdir(parents=True, exist_ok=True)                     


    def save_experiment_parameters(self):
        param_dict = vars(self.args)
        f = open('{}/parameters.txt'.format(self.experiment_dir), 'w')
        for param_name in param_dict:
            f.write('{} = {}\n'.format(param_name, param_dict[param_name]))
        

    def test(self):
        epoch = self.epoch - 1        
        for model in self.models.values():
            model.eval()

        H = self.args.H_for_test_renders
        W = self.args.W_for_test_renders        
        
        all_trained_focal_lengths = self.models["focal"]()([0])

        test_image_indices = self.test_image_indices        

        test_image_indices = [144] #[144, 250]
        for i, image_index in enumerate(test_image_indices):

            # image_index = 152
            print("Rendering test image {}".format(image_index))

            pp_x = self.principal_point_x * (float(W) / float(self.W))
            pp_y = self.principal_point_y * (float(H) / float(self.H))

            # always render                                   
            focal_lengths = all_trained_focal_lengths[image_index].expand(int(H*W)) * (float(H) / float(self.H))

            poses = self.models['pose']()([0])[image_index].unsqueeze(0).expand(W*H, -1, -1)
                            
            render_result = self.render_prediction(poses, focal_lengths, pp_x, pp_y, image_index)

            if type(render_result) == type(None):
                continue
                                
            # save rendered rgb and depth images
            out_file_suffix = str(image_index)
            color_file_name_fine = os.path.join(self.color_out_dir, str(out_file_suffix).zfill(4) + '_color_fine_{}.png'.format(epoch))
            depth_file_name_fine = os.path.join(self.depth_out_dir, str(out_file_suffix).zfill(4) + '_depth_fine_{}.png'.format(epoch))     

            if self.args.use_sparse_fine_rendering:
                color_file_name_fine = os.path.join(self.color_out_dir, str(out_file_suffix).zfill(4) + 'sparse_color_fine_{}.png'.format(epoch))
                depth_file_name_fine = os.path.join(self.depth_out_dir, str(out_file_suffix).zfill(4) + 'sparse_depth_fine_{}.png'.format(epoch))                            
                    
            if self.args.save_point_clouds_during_testing:

                if self.args.coarse_sampling_strategy == "focused":
                    focused_indices = render_result['flattened_focused_pixel_indices']
                    poses = poses[focused_indices]
                    focal_lengths = focal_lengths[focused_indices]

                if self.args.coarse_sampling_strategy in ["naive", "focused"]:
                    self.export_point_clouds_during_testing(poses=poses, 
                                                            focal_lengths=focal_lengths, 
                                                            pixel_directions=render_result['pixel_directions'],
                                                            depths=render_result['rendered_depth_fine'],
                                                            unsparse_depths=render_result['depth_image_unsparse_fine'], 
                                                            colors=render_result['rendered_image_fine'],
                                                            view_number=image_index)
                elif self.args.coarse_sampling_strategy == "sdf":
                    self.export_point_clouds_during_testing(poses=poses, 
                                                            focal_lengths=focal_lengths, 
                                                            pixel_directions=render_result['pixel_directions'],
                                                            depths=render_result['depth_image'],
                                                            unsparse_depths=render_result['depth_image'], 
                                                            colors=render_result['rendered_image'],
                                                            view_number=image_index)                    
            else:
                print("  -> Rendered Image {} of {} ({})".format(i+1, len(test_image_indices), image_index))            
                self.save_render_as_png(render_result, H, W, color_file_name_fine, depth_file_name_fine)            

            # break

            

        torch.cuda.empty_cache()

        if not self.args.train:
            quit()


    ##################################################################
    ###################### MEM utils #################################
    ##################################################################

    def print_memory_usage(self):

        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a
        
        print('total memory: {}'.format(t/1000000))
        print("reserved memory: {}".format(r/1000000))
        print("allocated memory: {}".format(a/1000000))
        print("reserved free memory: {}".format(f/1000000))

        self.mem_report()
        print("__________________________________")
    
    # https://gist.github.com/Stonesjtu/368ddf5d9eb56669269ecdf9b0d21cbe
    # https://gist.github.com/Stonesjtu
    def mem_report(self):
        '''Report the memory usage of the tensor.storage in pytorch
        Both on CPUs and GPUs are reported'''

        def _mem_report(tensors, mem_type):
            '''Print the selected tensors of type
            There are two major storage types in our major concern:
                - GPU: tensors transferred to CUDA devices
                - CPU: tensors remaining on the system memory (usually unimportant)
            Args:
                - tensors: the tensors of specified type
                - mem_type: 'CPU' or 'GPU' in current implementation '''
            print('Storage on %s' %(mem_type))
            print('-'*LEN)
            total_numel = 0
            total_mem = 0
            visited_data = []
            for tensor in tensors:
                if tensor.is_sparse:
                    continue
                # a data_ptr indicates a memory block allocated
                data_ptr = tensor._storage().data_ptr()
                if data_ptr in visited_data:
                    continue
                visited_data.append(data_ptr)

                numel = tensor._storage().size()
                total_numel += numel
                element_size = tensor._storage().element_size()
                mem = numel*element_size /1024/1024 # 32bit=4Byte, MByte
                total_mem += mem
                element_type = type(tensor).__name__
                size = tuple(tensor.size())

                print('%s\t\t%s\t\t%.2f' % (
                    element_type,
                    size,
                    mem) )
            print('-'*LEN)
            print('Total Tensors: %d \tUsed Memory Space: %.2f MBytes' % (total_numel, total_mem) )
            print('-'*LEN)

        LEN = 65
        print('='*LEN)
        objects = gc.get_objects()
        print('%s\t%s\t\t\t%s' %('Element type', 'Size', 'Used MEM(MBytes)') )
        tensors = [obj for obj in objects if torch.is_tensor(obj)]
        cuda_tensors = [t for t in tensors if t.is_cuda]
        host_tensors = [t for t in tensors if not t.is_cuda]
        _mem_report(cuda_tensors, 'GPU')
        _mem_report(host_tensors, 'CPU')
        print('='*LEN)


    ##################################################################
    ##################### Exports ##############################
    ##################################################################
    # def export_camera_extrinsics_and_intrinsics_from_trained_model(self, dir, epoch, save=True):
    #     # gather the latest poses ("camera extrinsics")
    #     camera_extrinsics = self.models['pose']()([0]).cpu()    
       
    #     # gather the latest focal lengths (f_x and f_y, which are currently identical)
    #     focal_length = self.models['focal']()([0]).cpu()      

    #     # get the number of cameras represented
    #     n_cameras = camera_extrinsics.shape[0]

    #     print("{} cameras in export...".format(n_cameras))
        
    #     # format camera intrinsics for export
    #     camera_intrinsics = torch.zeros(size=(n_cameras,3,3), dtype=torch.float32)
    #     camera_intrinsics[:,0,0] = focal_length
    #     camera_intrinsics[:,1,1] = focal_length
    #     camera_intrinsics[:,0,2] = self.principal_point_x
    #     camera_intrinsics[:,1,2] = self.principal_point_y
        
    #     # use index [2,2] in intrinsics to store the image index that the extrinsics/intrinsics correspond to
    #     image_indices = self.image_ids[::self.skip_every_n_images_for_training][:self.args.number_of_images_in_training_dataset] #torch.arange(start=0, end=self.args.number_of_images_in_training_dataset) * self.skip_every_n_images_for_training

    #     camera_intrinsics[:, 2, 2] = torch.tensor(image_indices).to(torch.float32)
    #     torch.save(camera_intrinsics, '{}/camera_intrinsics_{}.pt'.format(dir, epoch))
    #     torch.save(camera_extrinsics, '{}/camera_extrinsics_{}.pt'.format(dir, epoch))

    #     return (camera_extrinsics, camera_intrinsics)


    def export_point_clouds_during_testing(self, poses, focal_lengths, pixel_directions, depths, unsparse_depths, colors, view_number):

        xyz_coordinates, _, _ = volume_sampling(poses=poses.to(device=self.device), pixel_directions=pixel_directions.to(device=self.device), sampling_depths=depths.unsqueeze(1).to(device=self.device), perturb_depths=False)
        
        xyz_coordinates = xyz_coordinates[:,0,:]

        # filter points by maximum depth allowable (naturally carves out object only during object-focused scans)
        point_indices_below_maximum_depth = depths <= self.args.maximum_point_cloud_depth
        
        # filter points if sparse fine render depth is too distant from unsparse fine render depth
        point_indices_with_sparse_rendering_consistent = torch.abs(unsparse_depths - depths) <= self.args.maximum_sparse_vs_unsparse_depth_difference

        valid_point_indices = point_indices_below_maximum_depth & point_indices_with_sparse_rendering_consistent

        xyz_coordinates = xyz_coordinates[valid_point_indices]
        colors = colors[valid_point_indices].squeeze()

        print("({}) Saving {} of {} (x,y,z) points ({} close enough, {} with consistent sparse rendering)".format(view_number, torch.sum(valid_point_indices), depths.shape[0], torch.sum(point_indices_below_maximum_depth), torch.sum(point_indices_with_sparse_rendering_consistent)))

        if torch.sum(valid_point_indices) > 0:
            pcd = self.create_point_cloud(xyz_coordinates, colors, normals=None, flatten_xyz=False, flatten_image=False)
            o3d.io.write_point_cloud("{}/pointclouds/{}.ply".format(self.experiment_dir, view_number), pcd, write_ascii = True)   


    # create a point cloud using open3d
    def create_point_cloud(self, xyz_coordinates, colors, normals=None, flatten_xyz=True, flatten_image=True):
        pcd = o3d.geometry.PointCloud()
        if flatten_xyz:
            xyz_coordinates = torch.flatten(xyz_coordinates, start_dim=0, end_dim=1).cpu().detach().numpy()
        else:
            xyz_coordinates = xyz_coordinates.cpu().detach().numpy()
        if flatten_image:
            colors = torch.flatten(colors, start_dim=0, end_dim=1).cpu().detach().numpy()
            if normals is not None:
                normals = torch.flatten(normals, start_dim=0, end_dim=1).cpu().detach().numpy()
        else:
            colors = colors.cpu().detach().numpy()
            if normals is not None:
                normals = normals.cpu().detach().numpy()

        pcd.points = o3d.utility.Vector3dVector(xyz_coordinates)

        # Load saved point cloud and visualize it
        pcd.estimate_normals()
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)

        return pcd

    def export_50th_percentile_depth_map_point_clouds(self):
        all_trained_focal_lengths = self.models["focal"]()([0])

        H = self.args.H_for_test_renders
        W = self.args.W_for_test_renders

        for i, image_index in enumerate(self.test_image_indices):

            pp_x = self.principal_point_x * (float(W) / float(self.W))
            pp_y = self.principal_point_y * (float(H) / float(self.H))

            focal_lengths = all_trained_focal_lengths[image_index].expand(int(H*W)) * (float(H) / float(self.H))

            poses = self.models['pose']()([0])[image_index].unsqueeze(0).expand(W*H, -1, -1)
                            
            render_result = self.render_prediction(poses, focal_lengths, pp_x, pp_y, image_index)


    def get_xyz_bounds_from_cropped_point_cloud(self):
        path_to_cropped_merged_point_cloud = "{}/pointclouds/all_cropped.ply".format(self.experiment_dir)

        # for file_number, file_name in enumerate(glob.glob("{}/*.ply".format(point_clouds_directory))):
        pcd = o3d.io.read_point_cloud(path_to_cropped_merged_point_cloud)
        xyz = torch.from_numpy(np.asarray(pcd.points)).to(device=device)

        min_x = torch.min(xyz[:,0]) - 0.1
        max_x = torch.max(xyz[:,0]) + 0.1

        min_y = torch.min(xyz[:,1]) - 0.1
        max_y = torch.max(xyz[:,1]) + 0.1

        min_z = torch.min(xyz[:,2]) - 0.1
        max_z = torch.max(xyz[:,2]) + 0.1

        print("Loaded cropped point cloud bounds [min,max] := (x: [{},{}], y: [{},{}], z: [{},{}])".format(min_x, max_x, min_y, max_y, min_z, max_z))

        return min_x, max_x, min_y, max_y, min_z, max_z


    def get_voxel_center_coordinates(self, min_x, max_x, min_y, max_y, min_z, max_z, voxel_size=0.001):
        # Compute the number of voxels along each axis
        num_x = torch.ceil((max_x - min_x) / voxel_size).int()
        num_y = torch.ceil((max_y - min_y) / voxel_size).int()
        num_z = torch.ceil((max_z - min_z) / voxel_size).int()

        # Create a range of values for each axis
        x_range = torch.linspace(min_x + voxel_size / 2, max_x - voxel_size / 2, num_x)
        y_range = torch.linspace(min_y + voxel_size / 2, max_y - voxel_size / 2, num_y)
        z_range = torch.linspace(min_z + voxel_size / 2, max_z - voxel_size / 2, num_z)

        # Create a 3D grid of voxel center coordinates
        x, y, z = torch.meshgrid(x_range, y_range, z_range)

        # Reshape the grid into an (N, 3) tensor
        voxel_center_coordinates = torch.stack((x, y, z), dim=-1).reshape(-1, 3)
        
        return voxel_center_coordinates


    def get_sdf_xyz_samples(self):
        min_x, max_x, min_y, max_y, min_z, max_z = self.get_xyz_bounds_from_cropped_point_cloud()

        voxel_center_coordinates = self.get_voxel_center_coordinates(min_x, max_x, min_y, max_y, min_z, max_z)

        return voxel_center_coordinates

    def gather_sdf_coordinates_nearby_surfaces(self, xyz_coordinates, maximum_distance_to_surface = 0.001):

        valid_pixel_indices = self.get_bounds_of_depths_per_pixel_that_focus_on_object(xyz_samples=xyz_coordinates)

        if len(valid_pixel_indices) == 0:
            return None, None

        xyz_coordinates = xyz_coordinates[valid_pixel_indices]

        # normalize (x,y,z) values between -1 and 1
        normalized_voxel_xyz_positions = self.normalize_coordinates(xyz_coordinates=xyz_coordinates, min_xyz_values=self.xyz_min_values, max_xyz_range=self.xyz_max_range)

        # apply classic NeRF position encoding
        xyz_position_encoding = encode_position(input=normalized_voxel_xyz_positions.unsqueeze(1), levels=8, inc_input=True)

        # produce SDF, normals, and appearance features from the SSAN geometry MLP
        sdf = self.models["ssan_geometry"]()([xyz_position_encoding, False]).squeeze()

        # print("  -> Output SDF {}: {}".format(sdf.shape, sdf))

        indices_close_to_surface = torch.argwhere(torch.abs(sdf) < maximum_distance_to_surface)[:,0]

        final_indices_close_to_surface = valid_pixel_indices[indices_close_to_surface]

        return final_indices_close_to_surface, sdf[indices_close_to_surface]


    def save_sdf_data(self, xyz, sdf, save_point_cloud_of_all_points=True, truncation_limit=0.001):
    
        # good_indices = filter_out_too_distant_and_too_sharply_changing_tsdf_values(tsdf)
        # good_tsdf, good_tsdf_indices = get_good_tsdf(tsdf=tsdf, good_indices=good_indices)
        # good_tsdf_xyz = get_good_tsdf_xyz(good_indices=good_tsdf_indices)

        if save_point_cloud_of_all_points:
            colors = heatmap_to_pseudo_color(sdf.cpu().numpy(), min_val=-truncation_limit, max_val=truncation_limit)

            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(xyz.cpu().numpy())
            point_cloud.colors = o3d.utility.Vector3dVector(colors)

            print("Saving point cloud with {:,} points...".format(xyz.shape[0]))
            o3d.io.write_point_cloud("sdf_{}.ply".format(self.epoch - 1), point_cloud)

        torch.save(sdf, "sdf.pt")
        torch.save(xyz, "sdf_xyz.pt")


    # create a point cloud using open3d
    def export_sdf_field(self, distance_filter=0.0001):
        with torch.no_grad():

            voxel_xyz = self.get_sdf_xyz_samples()
            coordinates_per_batch = 1000000
            voxel_xyz_batches = voxel_xyz.split(coordinates_per_batch)
            n_batches = len(voxel_xyz_batches)

            print("Created {} batches of voxel (x,y,z) coordinates to query SDF network with".format(n_batches))

            all_xyz = []
            all_sdf = []
            for batch_number, voxel_xyz_batch in enumerate(voxel_xyz_batches):            
                voxel_xyz_batch = voxel_xyz_batch.to(device=self.device)
                indices_close_to_surface, sdf = self.gather_sdf_coordinates_nearby_surfaces(xyz_coordinates=voxel_xyz_batch, maximum_distance_to_surface=distance_filter)

                if type(indices_close_to_surface) != type(None) and sdf.shape[0] >= 3:
                    
                    print("({} of {}) {:,} of {:,} (x,y,z) coordinates nearby surface: e.g. {:.4f}, {:.4f}, {:.4f}, ...".format(batch_number, 
                                                                                                                                n_batches, 
                                                                                                                                len(indices_close_to_surface), 
                                                                                                                                coordinates_per_batch, 
                                                                                                                                sdf[0],
                                                                                                                                sdf[1],
                                                                                                                                sdf[2]))


                    all_xyz.append(voxel_xyz_batch[indices_close_to_surface])
                    all_sdf.append(sdf)

                else:
                     print("({} of {}) {:,} of {:,} (x,y,z) coordinates nearby surface".format(batch_number, 
                                                                                                n_batches, 
                                                                                                0, 
                                                                                                coordinates_per_batch))                   

                # if batch_number > 100:
                #     break


            all_xyz = torch.cat(all_xyz, dim=0)
            all_sdf = torch.cat(all_sdf, dim=0)

            self.save_sdf_data(xyz=all_xyz, sdf=all_sdf, truncation_limit=distance_filter)





    ##################################################################
    ##################### Main function ##############################
    ##################################################################
if __name__ == '__main__':
    
    with torch.no_grad():
        scene = SceneModel()

    while scene.epoch < scene.args.start_epoch + scene.args.number_of_epochs:    
        
        # if scene.epoch == scene.args.start_epoch:
        #     with torch.no_grad():                                            
        #         scene.test()
        #         quit()
                        
        with torch.no_grad():
            if scene.args.train:
                if scene.epoch != 1 and scene.epoch != scene.args.start_epoch and (scene.epoch-1) % scene.args.resample_pixels_frequency == 0:
                    print('Resampling training data...')
                    scene.sample_training_data(visualize_sampled_pixels=False)

                batch = scene.sample_next_batch(weighted=True)
            
        # if scene.args.extract_sdf_field or ((scene.epoch-1) % scene.args.extract_sdf_frequency == 0 and scene.epoch >= scene.args.start_epoch + scene.args.extract_sdf_frequency - 1):
        #     scene.export_sdf_field()

        # if scene.args.export_extrinsics_intrinsics:
        #     print("Exporting camera extrinsics and intrinsics...")
        #     scene.export_camera_extrinsics_and_intrinsics_from_trained_model(scene.experiment_dir, scene.epoch)
        #     print("  -> Finished exporting camera extrinsics and intrinsics")
        #     quit()

        if scene.args.train:
            scene.train(batch) 
       
        if (scene.epoch-1) % scene.args.save_models_frequency == 0 and (scene.epoch-1) !=  scene.args.start_epoch:
            scene.save_models()

        if (scene.epoch-1) % scene.args.test_frequency == 0 and (scene.epoch-1) != 0:
            with torch.no_grad():                
                scene.test()