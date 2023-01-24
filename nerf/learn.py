import torch
import torch._dynamo
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix, matrix_to_euler_angles, matrix_to_quaternion, axis_angle_to_quaternion, quaternion_multiply, matrix_to_axis_angle
from scipy.spatial.transform import Rotation
from torchsummary import summary
import cv2
import open3d as o3d
#from pytorch3d.io import IO
import imageio.v2 as imageio
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

import gc


#from torchviz import make_dot, make_dot_from_trace
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


set_randomness()

iterations_per_batch = 1

def parse_args():
    parser = argparse.ArgumentParser()

    # Define path to relevant data for training, and decide on number of images to use in training
    parser.add_argument('--base_directory', type=str, default='./data/dragon_scale', help='The base directory to load and save information from')
    parser.add_argument('--images_directory', type=str, default='color', help='The specific group of images to use during training')
    parser.add_argument('--images_data_type', type=str, default='jpg', help='Whether images are jpg or png')
    parser.add_argument('--skip_every_n_images_for_training', type=int, default=30, help='When loading all of the training data, ignore every N images')
    parser.add_argument('--number_of_pixels_in_training_dataset', type=int, default=480*640*100, help='The total number of pixels sampled from all images in training data')
    parser.add_argument('--resample_pixels_frequency', type=int, default=5000, help='Resample training data every this number of epochs')
    parser.add_argument('--save_models_frequency', type=int, default=25000, help='Save model every this number of epochs')
    parser.add_argument('--load_pretrained_models', type=bool, default=False, help='Whether to start training from models loaded with load_pretrained_models()')
    parser.add_argument('--pretrained_models_directory', type=str, default='./data/dragon_scale/hyperparam_experiments/1662755515_depth_loss_0.00075_to_0.0_k9_N1_NeRF_Density_LR_0.0005_to_0.0001_k4_N1_pose_LR_0.0025_to_1e-05_k9_N1', help='The directory storing models to load')    
    parser.add_argument('--reset_learning_rates', type=bool, default=False, help='When loading pretrained models, whether to reset learning rate schedules instead of resuming them')
    
    # Define number of epochs, and timing by epoch for when to start training per network
    parser.add_argument('--start_epoch', default=0, type=int, help='Epoch on which to begin or resume training')
    parser.add_argument('--number_of_epochs', default=200001, type=int, help='Number of epochs for training, used in learning rate schedules')    
    parser.add_argument('--start_training_extrinsics_epoch', type=int, default=500, help='Set to epoch number >= 0 to init poses using estimates from iOS, and start refining them from this epoch.')
    parser.add_argument('--start_training_intrinsics_epoch', type=int, default=5000, help='Set to epoch number >= 0 to init focals using estimates from iOS, and start refining them from this epoch.')
    parser.add_argument('--start_training_color_epoch', type=int, default=0, help='Set to a epoch number >= 0 to start learning RGB NeRF on top of density NeRF.')
    parser.add_argument('--start_training_geometry_epoch', type=int, default=0, help='Set to a epoch number >= 0 to start learning RGB NeRF on top of density NeRF.')
    parser.add_argument('--entropy_loss_weight', type=float, default=0.005, help='The weight used for entropy loss.')


    # Define evaluation/logging/saving frequency and parameters
    parser.add_argument('--test_frequency', default=100, type=int, help='Frequency of epochs to render an evaluation image')    
    parser.add_argument('--save_point_cloud_frequency', default=200002, type=int, help='Frequency of epochs to save point clouds')
    parser.add_argument('--save_depth_weights_frequency', default=200002, type=int, help='Frequency of epochs to save density depth weight visualizations')
    parser.add_argument('--log_frequency', default=1, type=int, help='Frequency of epochs to log outputs e.g. loss performance')        
    parser.add_argument('--number_of_test_images', default=2, type=int, help='Index in the training data set of the image to show during testing')
    parser.add_argument('--skip_every_n_images_for_testing', default=80, type=int, help='Skip every Nth testing image, to ensure sufficient test view diversity in large data set')    
    parser.add_argument('--number_of_pixels_per_batch_in_test_renders', default=128, type=int, help='Size in pixels of each batch input to rendering')    
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
    parser.add_argument('--use_sparse_fine_rendering', type=bool, help='Whether or not to use sparse fine rendering technique in test renders')

    # Define parameters the determines the overall size and learning capacity of the neural networks and their encodings
    parser.add_argument('--density_neural_network_parameters', type=int, default=512, help='The baseline number of units that defines the size of the NeRF geometry network')
    parser.add_argument('--color_neural_network_parameters', type=int, default=512, help='The baseline number of units that defines the size of the NeRF RGB (pitch,yaw) network')
    parser.add_argument('--directional_encoding_fourier_frequencies', type=int, default=10, help='The number of frequencies that are generated for positional encoding of (pitch, yaw)')

    # Define sampling parameters, including how many samples per raycast (outward), number of samples randomly selected per image, and (if masking is used) ratio of good to masked samples
    parser.add_argument('--pixel_samples_per_epoch', type=int, default=500, help='The number of rows of samples to randomly collect for each image during training')
    parser.add_argument('--number_of_samples_outward_per_raycast', type=int, default=1000, help='The number of samples per raycast to collect for each rendered pixel during training')        
    parser.add_argument('--number_of_samples_outward_per_raycast_for_test_renders', type=int, default=1000, help='The number of samples per raycast to collect for each rendered pixel during testing')        

    # Define depth sensor parameters
    parser.add_argument('--depth_sensor_error', type=float, default=0.5, help='If not using a heuristic for variance, hardcoded variance of Gaussian depth sensor model, in millimeters')
    parser.add_argument('--epsilon', type=float, default=0.0000001, help='Minimum value in log() for NeRF density weights going to 0')    

    # Additional parameters on pre-processing of depth data and coordinate systems    
    parser.add_argument('--min_confidence', type=float, default=2.0, help='A value in [0,1,2] where 0 allows all depth data to be used, 2 filters the most and ignores that')

    # Additional parameters on pre-processing of depth data and coordinate systems
    parser.add_argument('--near_maximum_depth', type=float, default=1.0, help='A percent of all raycast samples will be dedicated between the minimum depth (determined by sensor value) and this value')
    parser.add_argument('--far_maximum_depth', type=float, default=3.0, help='The remaining percent of all raycast samples will be dedicated between the near_maximum_depth and this value')
    parser.add_argument('--percentile_of_samples_in_near_region', type=float, default=0.90, help='This is the percent that determines the ratio between near and far sampling')    

    # Depth sampling optimizations
    parser.add_argument('--n_depth_sampling_optimizations', type=int, default=2, help='For every epoch, for every set of pixels, do this many renders to find the best depth sampling distances')
    parser.add_argument('--coarse_weight', type=float, default=0.1, help='Weight between [0,1] for coarse loss')    
    

    parsed_args = parser.parse_args()

    # Save parameters with Weights & Biases log
    wandb.init(project="nerf--", entity="3co", config=parsed_args)

    return parser.parse_args()


class SceneModel:
    def __init__(self, args, experiment_args = None):
    
        self.args = args
        self.load_args(experiment_args)

        # initialize high-level arguments        
        self.epoch = self.args.start_epoch
        self.start_time = int(time.time()) 
        self.device = torch.device('cuda:0')         

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

        
        torch.set_float32_matmul_precision('high')
        torch._dynamo.config.verbose=True        
        self.initialize_models()        
        self.initialize_learning_rates()  

        if self.args.load_pretrained_models:
            print("Loading pretrained models")
            # load pre-trained model
            self.load_pretrained_models()

            # compute the ray directions using the latest focal lengths
            focal_length = self.models["focal"](0)
            self.compute_ray_direction_in_camera_coordinates(focal_length)            
        else:            
            print("Training from scratch")
            

        # now load only the necessary data that falls within bounds defined
        self.load_image_and_depth_data_within_xyz_bounds()
                
        # initialize all models        


        
        #self.save_cam_xyz()    
        #quit()



    #########################################################################
    ################ Loading and initial processing of data #################
    #########################################################################

    def load_args(self, experiment_args = None):        
        if experiment_args == 'train': 
            print('\n------------------------------------------------------')
            print('------------------- Using train args -----------------')                       
            print('------------------------------------------------------')
            self.load_saved_args_train()
        elif experiment_args == 'test':
            print('\n------------------------------------------------------')
            print('------------------- Using test args ------------------')                       
            print('------------------------------------------------------')
            self.load_saved_args_test()            
        else:
            print('\n------------------------------------------------------')
            print('--------- No args provided: using default args -------')
            print('------------------------------------------------------')

    def prepare_test_data(self):
        self.test_image_indices = range(0, self.args.number_of_test_images * self.args.skip_every_n_images_for_testing, self.args.skip_every_n_images_for_testing)
        #print("Test image indices are: {}".format([i for i in self.test_image_indices]))

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
        #confidence_data = torch.from_numpy(confidence_data).to(device=self.device)
        confidence_data = torch.from_numpy(confidence_data).cpu()

        # read the 16 bit greyscale depth data which is formatted as an integer of millimeters
        depth_mm = cv2.imread(depth_path, -1).astype(np.float32)

        # convert data in millimeters to meters
        depth_m = depth_mm / (1000.0)  
        
        # set a cap on the maximum depth in meters; clips erroneous/irrelevant depth data from way too far out
        depth_m[depth_m > self.args.far_maximum_depth] = self.args.far_maximum_depth

        # resize to a resolution that e.g. may be higher, and equivalent to image data
        # to do: incorporate lower confidence into the interpolated depth metrics!
        resized_depth_meters = cv2.resize(depth_m, (self.W, self.H), interpolation=cv2.INTER_LINEAR)

        # NeRF requires bounds which can be used to constrain both the processed coordinate system data, as well as the ray sampling
        # for fast look-up, define nearest and farthest depth measured per image
        near_bound = np.min(resized_depth_meters)
        far_bound = np.max(resized_depth_meters)        

        #depth = torch.Tensor(resized_depth_meters).to(device=self.device) # (N_images, H_image, W_image)
        depth = torch.Tensor(resized_depth_meters).cpu() # (N_images, H_image, W_image)

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

        self.initial_focal_length = self.camera_intrinsics[0,0].repeat(self.n_training_images)                

        self.principal_point_x = self.camera_intrinsics[0,2]
        self.principal_point_y = self.camera_intrinsics[1,2]
        
        self.compute_ray_direction_in_camera_coordinates(focal_length=self.initial_focal_length)        


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
        
        self.all_initial_poses = torch.Tensor(convert3x4_4x4(rotations_translations)).cpu() # (N, 4, 4)        

    def create_point_cloud(self, xyz_coordinates, colors, normals, label=0, flatten_xyz=True, flatten_image=True):
        pcd = o3d.geometry.PointCloud()
        if flatten_xyz:
            xyz_coordinates = torch.flatten(xyz_coordinates, start_dim=0, end_dim=1).cpu().detach().numpy()
        else:
            xyz_coordinates = xyz_coordinates.cpu().detach().numpy()
        if flatten_image:
            colors = torch.flatten(colors, start_dim=0, end_dim=1).cpu().detach().numpy()
            normals = torch.flatten(normals, start_dim=0, end_dim=1).cpu().detach().numpy()
        else:
            colors = colors.cpu().detach().numpy()
            normals = normals.cpu().detach().numpy()

        pcd.points = o3d.utility.Vector3dVector(xyz_coordinates)

        # Load saved point cloud and visualize it
        pcd.estimate_normals()
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.normals = o3d.utility.Vector3dVector(normals)

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
        
        # now loop through all of the data, and filter out (only load and save as necessary) based on whether the points land within our focus area
        for i, image_id in enumerate(self.image_ids[::self.args.skip_every_n_images_for_training]):
            if i in self.test_image_indices:
                poses = self.models["pose"](0)
                pose = poses[i, :, :]
                cam_xyz = pose[:3,3].cpu().detach().numpy()
                # print("Saving camera (x,y,z) for test pose {}: ({:.4f},{:.4f},{:.4f})".format(self.test_poses_processed, cam_xyz[0], cam_xyz[1], cam_xyz[2]))
                cam_xyz_file = "{}/cam_xyz_{}.npy".format(self.experiment_dir,i)
                with open(cam_xyz_file, "wb") as f:
                    np.save(f, cam_xyz)
                


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

        #return xyz_inside_range.to(device=self.device)
        return xyz_inside_range.cpu()


    def load_image_and_depth_data_within_xyz_bounds(self, visualize_masks=True):
        self.rgbd = []
        self.image_ids_per_pixel = []
        self.pixel_rows = []
        self.pixel_cols = []
        self.xyz_per_view = []
        self.confidence_per_pixel = []                      
        neighbor_distance_per_pixel = []
    
        n_pixels_in_training_dataset = self.args.number_of_pixels_in_training_dataset
        n_images = len(self.image_ids[::self.args.skip_every_n_images_for_training])
        n_pixels_per_image = n_pixels_in_training_dataset // n_images

        print('n pixels in training dataset: ', n_pixels_in_training_dataset)
        print('n_images: ', n_images)
        print('n_pixels_per_image: ', n_pixels_per_image)

        # now loop through all of the data, and filter out (only load and save as necessary) based on whether the points land within our focus area
        for i, image_id in enumerate(self.image_ids[::self.args.skip_every_n_images_for_training]):
        #for i, image_id in enumerate(self.image_ids):                        

            # get depth data for this image
            depth, near_bound, far_bound, confidence = self.load_depth_data(image_id=image_id) # (H, W)

            # get (x,y,z) coordinates for this image
            xyz_coordinates = self.get_sensor_xyz_coordinates(pose_data=self.all_initial_poses[i*self.args.skip_every_n_images_for_training], depth_data=depth, i=i) # (H, W, 3)                                    
            
            # select only pixels whose estimated xyz coordinates fall within bounding box
            xyz_coordinates_on_or_off = self.get_xyz_inside_range(xyz_coordinates) # (H, W, 3) with True if (x,y,z) inside of previously set bounds, False if outside

            # select a uniformly random subset of those pixels            
            pixel_indices_selected = torch.argwhere(xyz_coordinates_on_or_off)[torch.randperm(torch.argwhere(xyz_coordinates_on_or_off).size()[0])[:n_pixels_per_image]]
            number_of_selected_pixels = pixel_indices_selected.size()[0]
                        
            # get the indices of the pixel rows and pixel columns where the projected (x,y,z) point is inside the target convex hull region
            pixel_rows_selected = pixel_indices_selected[:,0]
            pixel_cols_selected = pixel_indices_selected[:,1]
            self.pixel_rows.append(pixel_rows_selected)
            self.pixel_cols.append(pixel_cols_selected)            

            # get the corresponding (x,y,z) coordinates and depth values selected by the mask
            xyz_coordinates_selected = xyz_coordinates[pixel_rows_selected, pixel_cols_selected, :]
            self.xyz_per_view.append(xyz_coordinates_selected)

            # get the confidence of the selected pixels            
            selected_confidence = confidence[pixel_rows_selected, pixel_cols_selected]
            self.confidence_per_pixel.append(selected_confidence)

            depth_selected = depth[pixel_rows_selected, pixel_cols_selected] # (N selected)

            # now, load the (r,g,b) image and filter the pixels we're only focusing on
            image, image_name = self.load_image_data(image_id=image_id)
            rgb_selected = image[pixel_rows_selected, pixel_cols_selected, :] # (N selected, 3)            

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
            self.rgbd.append(rgbd_selected)            

            # now, save this image index, multiplied by the number of pixels selected, in a global vector across all images 
            #number_of_selected_pixels = torch.sum(xyz_coordinates_on_or_off)            
            image_id_for_all_pixels = torch.full(size=[number_of_selected_pixels], fill_value=i)
            self.image_ids_per_pixel.append(image_id_for_all_pixels)

            if self.args.save_ply_point_clouds_of_sensor_data and i in self.test_image_indices:
                pcd = self.get_point_cloud(dir=self.experiment_dir, pose=self.all_initial_poses[i*self.args.skip_every_n_images_for_training], depth=depth, rgb=image, pixel_directions=self.pixel_directions[i], label="raw_{}".format(image_id), save=True, save_raw_xyz=False)                
                

        # bring the data together        
        self.xyz = torch.cat(self.xyz_per_view, dim=0).cpu() # to(device=torch.device('cpu')) # 
        self.rgbd = torch.cat(self.rgbd, dim=0)                
        self.pixel_rows = torch.cat(self.pixel_rows, dim=0)
        self.pixel_cols = torch.cat(self.pixel_cols, dim=0)
        
        self.image_ids_per_pixel = torch.cat(self.image_ids_per_pixel, dim=0).cpu()
        self.confidence_per_pixel = torch.cat(self.confidence_per_pixel, dim=0)

        neighbor_distance_per_pixel = torch.cat(neighbor_distance_per_pixel, dim=0)

        
        # apply sampling weights        
        self.near = torch.min(self.rgbd[:,3])
        self.far = torch.max(self.rgbd[:,3])
        print("The near bound is {:.3f} meters and the far bound is {:.3f} meters".format(self.near, self.far))        

        self.depth_based_pixel_sampling_weights = (1 / (self.rgbd[:,3] ** (0.33))).cpu() # bias sampling of closer pixels probabilistically                
        max_depth_weight = torch.max(self.depth_based_pixel_sampling_weights)
        
        print("Max depth sampling weight of {:.2f} at {:.4f}m. More examples of depth vs. sampling weight: {:.2f}m = {:.2f}, {:.2f}m = {:.2f}, {:.2f}m = {:.2f}, {:.2f}m = {:.2f}, {:.2f}m = {:.2f}, {:.2f}m = {:.2f}, {:.2f}m = {:.2f}, {:.2f}m = {:.2f}, ...".format(
                                                                                                                            max_depth_weight,
                                                                                                                            self.near,
                                                                                                                            self.rgbd[:,3][0*100000],
                                                                                                                            self.depth_based_pixel_sampling_weights[0*100000],
                                                                                                                            self.rgbd[:,3][1*100000],
                                                                                                                            self.depth_based_pixel_sampling_weights[1*100000],
                                                                                                                            self.rgbd[:,3][2*100000],
                                                                                                                            self.depth_based_pixel_sampling_weights[2*100000],
                                                                                                                            self.rgbd[:,3][3*100000],
                                                                                                                            self.depth_based_pixel_sampling_weights[3*100000],
                                                                                                                            self.rgbd[:,3][4*100000],
                                                                                                                            self.depth_based_pixel_sampling_weights[4*100000],
                                                                                                                            self.rgbd[:,3][5*100000],
                                                                                                                            self.depth_based_pixel_sampling_weights[5*100000],
                                                                                                                            self.rgbd[:,3][6*100000],
                                                                                                                            self.depth_based_pixel_sampling_weights[6*100000],
                                                                                                                            self.rgbd[:,3][7*100000],
                                                                                                                            self.depth_based_pixel_sampling_weights[7*100000]))


                        

        max_rgb_distance = np.sqrt(3)
        steepness = 20.0
        
        neighbor_rgb_distance_sampling_weights = torch.log2( (steepness * neighbor_distance_per_pixel / max_rgb_distance + 1.0))
        self.depth_based_pixel_sampling_weights = self.depth_based_pixel_sampling_weights * neighbor_rgb_distance_sampling_weights
        #print(neighbor_rgb_distance_sampling_weights[:10])
        

        
        print("Loaded {} images with {:,} pixels selected".format(i+1, self.image_ids_per_pixel.shape[0] ))


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
        xyz_coordinates = self.get_sensor_xyz_coordinates(pose_data=self.all_initial_poses[index_to_filter * self.args.skip_every_n_images_for_training], depth_data=depth, i=image_id)

        # now filter both the xyz_coordinates and the image by the values in the top of this function
        if type(min_pixel_row) != type(None) or type(max_pixel_row) != type(None):
            xyz_coordinates = xyz_coordinates[min_pixel_row:max_pixel_row, :, :]
        if type(min_pixel_col) != type(None) or type(max_pixel_col) != type(None):
            xyz_coordinates = xyz_coordinates[:, min_pixel_col:max_pixel_col, :]

        # now define the bounds in (x,y,z) space through which we will filter all future pixels by their projected points
        self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z = self.get_min_max_bounds(xyz_coordinates, padding=100.0)


    #########################################################################
    ####################### Camera helper functions #########################
    #########################################################################

    def get_sensor_xyz_coordinates(self, i=None, pose_data=None, depth_data=None):

        # get camera world position and rotation
        camera_world_position = pose_data[:3, 3].view(1, 1, 1, 3)     # (1, 1, 1, 3)
        camera_world_rotation = pose_data[:3, :3].view(1, 1, 1, 3, 3) # (1, 1, 1, 3, 3)

        # get relative pixel orientations              
        pixel_directions = self.pixel_directions[i].unsqueeze(3) # (H, W, 3, 1)   
                
        xyz_coordinates = self.derive_xyz_coordinates(camera_world_position, camera_world_rotation, pixel_directions, depth_data)

        return xyz_coordinates 

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


    # compute *unnormalized* ray directions for all pixels in dataset
    def compute_ray_direction_in_camera_coordinates(self, focal_length):

        

        
        # Compute ray directions in the camera coordinate, which only depends on intrinsics. This will be further transformed to world coordinate later, using camera poses.
        #camera_coordinates_y, camera_coordinates_x = torch.meshgrid(torch.arange(self.H, dtype=torch.float32, device=self.device),
        #                                                            torch.arange(self.W, dtype=torch.float32, device=self.device),
        #                                                            indexing='ij')  # (H, W)
        camera_coordinates_y, camera_coordinates_x = torch.meshgrid(torch.arange(self.H, dtype=torch.float32),
                                                                    torch.arange(self.W, dtype=torch.float32),
                                                                    indexing='ij')  # (H, W)        
        
        # (N_images, H, W)
    
        camera_coordinates_y = camera_coordinates_y.unsqueeze(0).expand(self.n_training_images, self.H, self.W)                  
        camera_coordinates_x = camera_coordinates_x.unsqueeze(0).expand(self.n_training_images, self.H, self.W)
        
        focal_length_rep = focal_length.cpu().unsqueeze(1).unsqueeze(2).expand(self.n_training_images, self.H, self.W)                                

        # Apple's camera coordinate system:
        # x = right
        # y = up
        # z = forward (note, z represents "forward" in our environment, whereas it typically represents "backward" in other environments)
        #radial_distortion_x = 1.0 + 0.000001 * torch.abs(camera_coordinates_x - self.principal_point_x.cpu())**2
        #radial_distortion_y = 1.0 + 0.000001 * torch.abs(camera_coordinates_y - self.principal_point_y.cpu())**2




        #camera_coordinates_directions_x = radial_distortion_x * (camera_coordinates_x - self.principal_point_x.cpu()) / focal_length_rep  # (N_images, H, W)
        #camera_coordinates_directions_y = radial_distortion_y * (camera_coordinates_y - self.principal_point_y.cpu()) / focal_length_rep  # (N_images, H, W)
        camera_coordinates_directions_x = (camera_coordinates_x - self.principal_point_x.cpu()) / focal_length_rep  # (N_images, H, W)
        camera_coordinates_directions_y = (camera_coordinates_y - self.principal_point_y.cpu()) / focal_length_rep  # (N_images, H, W)        
                
        #camera_coordinates_directions_z = torch.ones(self.n_training_images,self.H, self.W, dtype=torch.float32, device=self.device)  # (N_images, H, W)        
        camera_coordinates_directions_z = torch.ones(self.n_training_images,self.H, self.W, dtype=torch.float32)  # (N_images, H, W)        
        camera_coordinates_pixel_directions = torch.stack([camera_coordinates_directions_x, camera_coordinates_directions_y, camera_coordinates_directions_z], dim=-1)  # (N_images, H, W, 3)        

        #self.pixel_directions = camera_coordinates_pixel_directions.to(device=self.device)         
        self.pixel_directions = camera_coordinates_pixel_directions.cpu()
        
        self.focal_length = focal_length.cpu()                
    

    def compute_pixel_directions(self, focal_lengths, pixel_rows, pixel_cols):
        
        n_pixels = focal_lengths.size()[0]                   

        pixel_directions_x = (pixel_cols.to(self.device) - self.principal_point_x.to(self.device)) / focal_lengths
        pixel_directions_y = (pixel_rows.to(self.device) - self.principal_point_y.to(self.device)) / focal_lengths
        pixel_directions_z = torch.ones(n_pixels, dtype=torch.float32, device=self.device)

        pixel_directions = torch.stack([pixel_directions_x, pixel_directions_y, pixel_directions_z], dim=1)                               
        return pixel_directions


    def get_point_cloud(self, pose, depth, rgb, pixel_directions, label=0, save=False, remove_zero_depths=True, save_raw_xyz=False, dir='', entropy_image=None, unsparse_rendered_rgb_img=None, sparse_rendered_rgb_img=None, unsparse_depth=None, max_depth_filter_image=None,save_in_image_coordinates=True):        

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

        max_depth_condition = (max_depth_filter_image == 1).to(self.device)
        n_filtered_points = self.H * self.W - torch.sum(max_depth_condition.flatten())
        print("filtering {}/{} points with max depth condition".format(n_filtered_points, self.W*self.H))

        depth_condition = (depth != 0).to(self.device)
        n_filtered_points = self.H * self.W - torch.sum(depth_condition.flatten())
        print("filtering {}/{} points with depth=0 condition".format(n_filtered_points, self.W*self.H))

        angle_condition = (torch.sqrt( torch.sum( (pixel_directions - pixel_directions[240, 320, :])**2, dim=2 ) ) < r).to(self.device)
        n_filtered_points = self.H * self.W - torch.sum(angle_condition.flatten())
        print("filtering {}/{} points with angle condition".format(n_filtered_points, self.W*self.H))

        #entropy_condition = (entropy_image < 3.0).to(self.device)
        entropy_condition = (entropy_image < 3.0).to(self.device)
        n_filtered_points = self.H * self.W - torch.sum(entropy_condition.flatten())
        print("filtering {}/{} points with entropy condition".format(n_filtered_points, self.W*self.H))

        joint_condition = torch.logical_and(max_depth_condition, depth_condition).to(self.device)        
        joint_condition = torch.logical_and(joint_condition, angle_condition).to(self.device)        
        joint_condition = torch.logical_and(joint_condition, entropy_condition).to(self.device)        

        sparse_depth = depth      

        if self.args.use_sparse_fine_rendering:            
            sticker_condition = torch.abs(unsparse_depth - sparse_depth) < 0.01 #< 8.0 * (self.far-self.near) / ( (self.args.number_of_samples_outward_per_raycast + 1))            
            joint_condition = torch.logical_and(joint_condition, sticker_condition)
            n_filtered_points = self.H * self.W - torch.sum(sticker_condition.flatten())
            print("filtering {}/{} points with sticker condition".format(n_filtered_points, self.W*self.H))            
    
        n_filtered_points = self.H * self.W - torch.sum(joint_condition.flatten())
        print("{}/{} total points filtered with intersection of conditions".format(n_filtered_points, self.W*self.H))        

        if save_raw_xyz:

            if save_in_image_coordinates:
                # save geometry as .npy file accessible in image coordinates (H,W,3)
                xyz_coordinates_as_image = xyz_coordinates.reshape(self.H, self.W, 3)

                joint_condition_1_or_0 = torch.where(joint_condition, 1.0, 0.0)

                xyz_data = xyz_coordinates_as_image.to(device=self.device)
                entropy_data = entropy_image.unsqueeze(2).to(device=self.device)
                filter_data = joint_condition_1_or_0.unsqueeze(2).to(device=self.device)

                # save entropy data in the 4th channel
                xyz_entropy_filter_image = torch.cat([xyz_data, entropy_data, filter_data], dim=2)

                print("Saving (x,y,z) coordinates as image with shape {}".format(xyz_entropy_filter_image.shape))
                file_path = "{}/{}_xyz_entropy_per_pixel.npy".format(dir, label)
                with open(file_path, "wb") as f:
                    np.save(f, xyz_entropy_filter_image.cpu().detach().numpy())

                # save normalized geometry coordinates as image (quick visual verification)
                max_coordinate = torch.max(xyz_coordinates_as_image)
                min_coordinate = torch.min(xyz_coordinates_as_image)
                normalized_xyz_image_data = (xyz_coordinates_as_image - min_coordinate) / (max_coordinate - min_coordinate)
                geometry_image_data = (normalized_xyz_image_data.cpu().numpy() * 255).astype(np.uint8)    
                geometry_image_file_path = "{}/{}_geometry_data_color_coded.png".format(dir, label)
                imageio.imwrite(geometry_image_file_path, geometry_image_data)

                # save entropy as image (quick visual verification)
                rendered_entropy = heatmap_to_pseudo_color(entropy_image.cpu().detach().numpy(), min_val=0.0, max_val=2.0)
                entropy_image_data = (rendered_entropy * 255).astype(np.uint8)    
                entropy_image_file_path = "{}/{}_entropy_data_color_coded.png".format(dir, label)
                imageio.imwrite(entropy_image_file_path, entropy_image_data)

                # save filter as image (quick visual verification)
                rendered_filter_data = heatmap_to_pseudo_color(joint_condition_1_or_0.cpu().detach().numpy())
                filter_image_data = (rendered_filter_data * 255).astype(np.uint8)    
                filter_image_file_path = "{}/{}_filter_color_coded.png".format(dir, label)
                imageio.imwrite(filter_image_file_path, filter_image_data)

                # save colors, extra verification
                rgb_as_image = rgb.reshape(self.H, self.W, 3)   
                color_image_data = (rgb_as_image.cpu().numpy() * 255).astype(np.uint8)    
                color_file_path = "{}/{}_color_data_restructured_same_as_geometry_data.png".format(dir, label)
                imageio.imwrite(color_file_path, color_image_data)

            else:
                file_path = "{}/{}_xyz_raw.npy".format(dir, label)
                with open(file_path, "wb") as f:
                    np.save(f, xyz_coordinates.cpu().detach().numpy())


        joint_condition_indices = torch.where(joint_condition)

        n_filtered_points = self.H * self.W - torch.sum(joint_condition.flatten())

        rgb = rgb.to(device=self.device)
       
        depth = depth[joint_condition_indices]
        rgb = rgb[joint_condition_indices]
                
        normals = torch.zeros(self.H, self.W, 3)
        normals = pose[:3, 3] - xyz_coordinates
        normals = torch.nn.functional.normalize(normals, p=2, dim=2)        
        normals = normals[joint_condition_indices]

        xyz_coordinates = xyz_coordinates[joint_condition_indices]
        
        pcd = self.create_point_cloud(xyz_coordinates, rgb, normals, label="point_cloud_{}".format(label), flatten_xyz=False, flatten_image=False)
        if save:
            file_name = "{}/view_{}_training_data_{}.ply".format(dir, label, self.epoch-1)
            o3d.io.write_point_cloud(file_name, pcd, write_ascii = True)

        return pcd             
        

    #########################################################################
    ############ Initialize models and set learning parameters ##############
    #########################################################################

    def initialize_models(self):
        # TODO: resolve issues with using mode='reduce-overhead' or 'max-autotune' for geometry and color models
        self.models = {}                

        # "[WARNING] skipping cudagraphs due to multiple devices"
        self.models["focal"] = torch.compile(CameraIntrinsicsModel(self.H, self.W, self.initial_focal_length, self.n_training_images).to(device=self.device), mode='reduce-overhead')
        #self.models["focal"] = CameraIntrinsicsModel(self.H, self.W, self.initial_focal_length, self.n_training_images).to(device=self.device)
        
        #self.models["pose"] = torch.compile(CameraPoseModel(self.all_initial_poses[::self.args.skip_every_n_images_for_training]).to(device=self.device))
        self.models["pose"] = CameraPoseModel(self.all_initial_poses[::self.args.skip_every_n_images_for_training]).to(device=self.device)
        

        self.models["geometry"] = torch.compile(NeRFDensity(self.args).to(device=self.device))
        #self.models["geometry"] = NeRFDensity(self.args).to(device=self.device)
                        
        # "[WARNING] skipping cudagraphs due to complex input striding"
        self.models["color"] = torch.compile(NeRFColor(self.args).to(device=self.device))
        #self.models["color"] = NeRFColor(self.args).to(device=self.device)
        
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
        print('Loading pretrained model at {}/[model]_{}.pth'.format(self.args.pretrained_models_directory, self.args.start_epoch))
        for model_name in self.models.keys():
            model_path = "{}/{}_{}.pth".format(self.args.pretrained_models_directory, model_name, self.args.start_epoch-1)            
                        
            # load checkpoint data
            ckpt = torch.load(model_path, map_location=self.device)

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
    ############################# Sampling ##################################
    #########################################################################

    def sample_depths_near_linearly_far_nonlinearly(self, number_of_pixels, add_noise=True, test_render=False):    
        n_samples = self.args.number_of_samples_outward_per_raycast + 1
        if test_render:
            n_samples = self.args.number_of_samples_outward_per_raycast_for_test_renders + 1

        percentile_of_samples_in_near_region = self.args.percentile_of_samples_in_near_region            
        near_min_focus = self.near  #   0.091
        near_max_focus = self.args.near_maximum_depth # 0.5
        far_max_focus = self.args.far_maximum_depth # 3.0        

        # set additional arguments from sanity checks / identities
        near_min_focus = torch.maximum(near_min_focus, torch.tensor(0.0))
        far_min_focus = near_max_focus        

        # determine number of samples in near region vs. far region
        n_samples_near = torch.floor(torch.tensor(n_samples * percentile_of_samples_in_near_region))
        n_samples_far = n_samples - n_samples_near
                
        sample_distances = torch.linspace(near_min_focus, far_min_focus, int(n_samples_near)).to(self.device)
        #sample_distances = sample_distances.unsqueeze(0).expand(number_of_pixels, n_samples_near.int())

        # compute sample distance for the far region, where the far min is equal to the near max
        far_focus_base = (far_max_focus/far_min_focus)**(1/n_samples_far)
        far_sample_numbers = torch.arange(start=0, end=n_samples_far).to(self.device)
        far_distances = far_min_focus * far_focus_base ** far_sample_numbers
        sample_distances = torch.cat([sample_distances, far_distances]).to(self.device)
        #sample_distances.append(far_distances)

        # combine the near and far sample distances
        #sample_distances = torch.cat(sample_distances).to(self.device)

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

        """
            print(sample_distances[0])
            #d = sample_distances[0][torch.argwhere(sample_distances[0] < 0.5)] 
            d = sample_distances[0]
            plt.scatter( d.cpu().numpy(), torch.ones(d.size()[0]).cpu().numpy() , s=3)
            plt.show()
            quit()
        """

        return sample_distances


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

    def resample_depths_from_nerf_weights(self, number_of_pixels, weights, depth_samples, resample_padding = 0.01, use_sparse_fine_rendering=False, test_render=False):
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

        return samples.to(self.device)


    #########################################################################
    ############################## Rendering ################################
    #########################################################################

    # the core rendering function
    def render(self, poses, pixel_directions, sampling_depths, pixel_focal_lengths, perturb_depths=False):
        
        # poses := (N_pixels, 4, 4)
        # pixel_directions := (N_pixels, 3)
        # sampling_depths := (N_samples)

        # (N_pixels, N_sample, 3), (N_pixels, 3), (N_pixels, N_samples)                    
        
        pixel_directions = torch.nn.functional.normalize(pixel_directions, p=2, dim=1)
        pixel_xyz_positions, pixel_directions_world, resampled_depths = volume_sampling(poses=poses, pixel_directions=pixel_directions, sampling_depths=sampling_depths, perturb_depths=perturb_depths)
        pixel_directions_world = torch.nn.functional.normalize(pixel_directions_world, p=2, dim=1)  # (N_pixels, 3)
        
        xyz_position_encoding = encode_ipe(poses[:, :3, 3], pixel_xyz_positions, pixel_directions_world, sampling_depths, pixel_focal_lengths, self.principal_point_x, self.principal_point_y)                            
        xyz_position_encoding = xyz_position_encoding.to(self.device)
                            
        # encode direction: (H, W, N_sample, (2L+1)*C = 27)        
        angular_directional_encoding = encode_position(pixel_directions_world, levels=self.args.directional_encoding_fourier_frequencies)  # (N_pixels, 27)
        angular_directional_encoding = angular_directional_encoding.unsqueeze(1).expand(-1, sampling_depths.size()[1] - 1, -1)  # (N_pixels, N_sample, 27)                    
        
        density, features = self.models["geometry"](xyz_position_encoding) # (N_pixels, N_sample, 1), # (N_pixels, N_sample, D)                                
        
        rgb = self.models["color"](features, angular_directional_encoding)  # (N_pixels, N_sample, 4)        
        
        render_result = volume_rendering(rgb, density, resampled_depths[:, : -1])                

        depth_weights = render_result['weight']
        
        depth_map = render_result['depth_map']
        rgb_rendered = render_result['rgb_rendered']
                
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


    def flat_render(self, poses, pixel_directions, focal_lengths):

        n_pixels = poses.size()[0]
        
        # split each of the rendering inputs into batches for better GPU usage        
        poses_batches = poses.split(self.args.number_of_pixels_per_batch_in_test_renders)
        pixel_directions_batches = pixel_directions.split(self.args.number_of_pixels_per_batch_in_test_renders)
        focal_lengths_batches = focal_lengths.split(self.args.number_of_pixels_per_batch_in_test_renders)                 

        rendered_image_fine_batches = []        

        for poses_batch, pixel_directions_batch, focal_lengths_batch in zip(poses_batches, pixel_directions_batches, focal_lengths_batches):

            # for resampling with test data, we will compute the NeRF-weighted resamples per batch
            for depth_sampling_optimization in range(self.args.n_depth_sampling_optimizations):
                # get the depth samples per pixel
                if depth_sampling_optimization == 0:
                    # if this is the first iteration, collect linear depth samples to query NeRF, uniformly in space                    
                    depth_samples_coarse = self.sample_depths_linearly(number_of_pixels=poses_batch.size()[0], add_noise=False, test_render=True) # (N_pixels, N_samples)                                                           
                    #depth_samples_coarse = self.sample_depths_nonlinearly(number_of_pixels=poses_batch.size()[0], add_noise=False, test_render=True) # (N_pixels, N_samples)                                                           
                    rendered_data_coarse = self.render(poses=poses_batch, pixel_directions=pixel_directions_batch, sampling_depths=depth_samples_coarse, pixel_focal_lengths=focal_lengths_batch, perturb_depths=False)  # (N_pixels, 3)
                else:
                    # if this is not the first iteration, then resample with the latest weights                                        
                    depth_samples_fine = self.resample_depths_from_nerf_weights(number_of_pixels=poses_batch.size()[0], weights=rendered_data_coarse['depth_weights'], depth_samples=depth_samples_coarse, use_sparse_fine_rendering=self.args.use_sparse_fine_rendering, test_render=True)  # (N_pixels, N_samples)                                
                    rendered_data_fine = self.render(poses=poses_batch, pixel_directions=pixel_directions_batch, sampling_depths=depth_samples_fine, pixel_focal_lengths=focal_lengths_batch, perturb_depths=False)  # (N_pixels, 3)
                                        
            if self.args.n_depth_sampling_optimizations > 1:
                rendered_image_fine_batches.append(rendered_data_fine['rgb_rendered']) # (n_pixels_per_row, 3)                

        if self.args.n_depth_sampling_optimizations > 1:
            rendered_image_data_fine = torch.cat(rendered_image_fine_batches, dim=0)#.cpu() # (N_pixels, 3)            

        if self.args.n_depth_sampling_optimizations > 1:                                
            rendered_image_fine = rendered_image_data_fine#.cpu()
        
        render_result = {
            'rendered_pixels': rendered_image_fine            
        }                    
        
        return render_result            


    # Render just the fine and coarse rgb/depth images, all on GPU
    # view_pixel_directions: assumed to be computed with respect to view_focal_length
    # view_focal_length: only needed by IPE computation    
    def basic_render(self, pose, view_pixel_directions, view_focal_length, test_render=False):

        poses = pose.unsqueeze(0).expand(self.W*self.H, -1, -1).to(self.device)
        pixel_directions = view_pixel_directions.flatten(start_dim=0, end_dim=1).to(self.device)
        focal_lengths = view_focal_length.unsqueeze(0).expand(self.W*self.H, 1).to(self.device)

        # split each of the rendering inputs into batches for better GPU usage        
        poses_batches = poses.split(self.args.number_of_pixels_per_batch_in_test_renders)
        pixel_directions_batches = pixel_directions.split(self.args.number_of_pixels_per_batch_in_test_renders)
        focal_lengths_batches = focal_lengths.split(self.args.number_of_pixels_per_batch_in_test_renders)                 

        rendered_image_fine_batches = []
        depth_image_fine_batches = []
        rendered_image_coarse_batches = []
        depth_image_coarse_batches = []

        for poses_batch, pixel_directions_batch, focal_lengths_batch in zip(poses_batches, pixel_directions_batches, focal_lengths_batches):

            # for resampling with test data, we will compute the NeRF-weighted resamples per batch
            for depth_sampling_optimization in range(self.args.n_depth_sampling_optimizations):
                # get the depth samples per pixel
                if depth_sampling_optimization == 0:
                    # if this is the first iteration, collect linear depth samples to query NeRF, uniformly in space                    
                    depth_samples_coarse = self.sample_depths_near_linearly_far_nonlinearly(number_of_pixels=poses_batch.size()[0], add_noise=False, test_render=test_render) # (N_pixels, N_samples)                                       
                    rendered_data_coarse = self.render(poses=poses_batch, pixel_directions=pixel_directions_batch, sampling_depths=depth_samples_coarse, pixel_focal_lengths=focal_lengths_batch.squeeze(1), perturb_depths=False)  # (N_pixels, 3)
                else:
                    # if this is not the first iteration, then resample with the latest weights                                        
                    depth_samples_fine = self.resample_depths_from_nerf_weights(number_of_pixels=poses_batch.size()[0], weights=rendered_data_coarse['depth_weights'], depth_samples=depth_samples_coarse, use_sparse_fine_rendering=self.args.use_sparse_fine_rendering, test_render=test_render)  # (N_pixels, N_samples)                                
                    rendered_data_fine = self.render(poses=poses_batch, pixel_directions=pixel_directions_batch, sampling_depths=depth_samples_fine, pixel_focal_lengths=focal_lengths_batch.squeeze(1), perturb_depths=False)  # (N_pixels, 3)
                                        
            if self.args.n_depth_sampling_optimizations > 1:
                rendered_image_fine_batches.append(rendered_data_fine['rgb_rendered']) # (n_pixels_per_row, 3)
                depth_image_fine_batches.append(rendered_data_fine['depth_map']) # (n_pixels_per_row)             

            rendered_image_coarse_batches.append(rendered_data_coarse['rgb_rendered']) # (n_pixels_per_row, 3)
            depth_image_coarse_batches.append(rendered_data_coarse['depth_map']) # (n_pixels_per_row)             

        if self.args.n_depth_sampling_optimizations > 1:
            rendered_image_data_fine = torch.cat(rendered_image_fine_batches, dim=0) # (N_pixels, 3)
            rendered_depth_data_fine = torch.cat(depth_image_fine_batches, dim=0)  # (N_pixels)

        rendered_image_data_coarse = torch.cat(rendered_image_coarse_batches, dim=0) # (N_pixels, 3)
        rendered_depth_data_coarse = torch.cat(depth_image_coarse_batches, dim=0)  # (N_pixels)

        if self.args.n_depth_sampling_optimizations > 1:
            rendered_image_fine = torch.zeros(self.H * self.W, 3)
            rendered_depth_fine = torch.zeros(self.H * self.W)

        rendered_image_coarse = torch.zeros(self.H * self.W, 3)
        rendered_depth_coarse = torch.zeros(self.H * self.W)

        if self.args.n_depth_sampling_optimizations > 1:
            rendered_image_fine = rendered_image_data_fine
            rendered_depth_fine = rendered_depth_data_fine
            
        rendered_image_coarse = rendered_image_data_coarse
        rendered_depth_coarse = rendered_depth_data_coarse

        render_result = {
            'rendered_image_coarse': rendered_image_coarse,
            'rendered_depth_coarse': rendered_depth_coarse,
        }

        if self.args.n_depth_sampling_optimizations > 1:            
            render_result['rendered_image_fine'] = rendered_image_fine
            render_result['rendered_depth_fine'] = rendered_depth_fine

        return render_result    


    # invoke current model for the pose and mask associated with train_image_index
    # for visual results, supply result to save_render_as_png
    def render_prediction_for_train_image(self, train_image_index):
        mask = torch.zeros(self.H, self.W)        

        pixel_indices_for_this_image = torch.argwhere(self.image_ids_per_pixel == train_image_index)

        pixel_rows = self.pixel_rows[pixel_indices_for_this_image]
        pixel_cols = self.pixel_cols[pixel_indices_for_this_image]        
        mask[pixel_rows, pixel_cols] = 1
        mask = mask.flatten()
        pose = self.models['pose'](0)[train_image_index]

        return self.render_prediction(pose=pose, train_image_index=train_image_index, mask=None)


    # invoke current model for a specific pose and 1d mask
    # for visual results, supply result to save_render_as_png
    # pixels filtered out by the mask are assigned value [0,0,0] 
    # train_image_index 
    def render_prediction(self, pose, train_image_index, mask=None, max_depth=None):                
        
        # n_samples: the number of queries to NeRF model        
        n_samples = self.args.number_of_samples_outward_per_raycast_for_test_renders

        # copy pose to go with each individual pixel and match reference pixel directions with them
        if mask is not None:
            poses = pose.unsqueeze(0).expand(int(mask.sum().item()), -1, -1)
            pixel_directions = self.pixel_directions[train_image_index].flatten(start_dim=0, end_dim=1)[torch.argwhere(mask==1)]
            pixel_directions = pixel_directions.squeeze(1)                                                
            focal_lengths = self.focal_length[train_image_index].unsqueeze(0).expand(int(mask.sum().item()), 1)              
        else:
            poses = pose.unsqueeze(0).expand(self.W*self.H, -1, -1)
            pixel_directions = self.pixel_directions[train_image_index].flatten(start_dim=0, end_dim=1)                  
            focal_lengths = self.focal_length[train_image_index].unsqueeze(0).expand(self.W*self.H, 1)


                

        # split each of the rendering inputs into batches for better GPU usage        
        poses_batches = poses.split(self.args.number_of_pixels_per_batch_in_test_renders)
        pixel_directions_batches = pixel_directions.split(self.args.number_of_pixels_per_batch_in_test_renders)
        focal_lengths_batches = focal_lengths.split(self.args.number_of_pixels_per_batch_in_test_renders)         

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

        # for each batch, compute the render and extract RGB and depth map
        for poses_batch, pixel_directions_batch, focal_lengths_batch in zip(poses_batches, pixel_directions_batches, focal_lengths_batches):

            poses_batch = poses_batch.to(self.device)
            pixel_directions_batch = pixel_directions_batch.to(self.device)
            focal_lengths_batch = focal_lengths_batch.to(self.device)

            # for resampling with test data, we will compute the NeRF-weighted resamples per batch
            for depth_sampling_optimization in range(self.args.n_depth_sampling_optimizations):
                # get the depth samples per pixel
                if depth_sampling_optimization == 0:
                    # if this is the first iteration, collect linear depth samples to query NeRF, uniformly in space                    
                    depth_samples_coarse = self.sample_depths_near_linearly_far_nonlinearly(number_of_pixels=poses_batch.size()[0], add_noise=False, test_render=True) # (N_pixels, N_samples)                              
                    resampled_depths_coarse_batches.append(depth_samples_coarse.cpu())
                    rendered_data_coarse = self.render(poses=poses_batch, pixel_directions=pixel_directions_batch, sampling_depths=depth_samples_coarse, pixel_focal_lengths=focal_lengths_batch.squeeze(1), perturb_depths=False)  # (N_pixels, 3)
                else:
                    # if this is not the first iteration, then resample with the latest weights         
                    depth_samples_fine = self.resample_depths_from_nerf_weights(number_of_pixels=poses_batch.size()[0], weights=rendered_data_coarse['depth_weights'], depth_samples=depth_samples_coarse, use_sparse_fine_rendering=self.args.use_sparse_fine_rendering, test_render=True)  # (N_pixels, N_samples)
                    resampled_depths_fine_batches.append(depth_samples_fine.cpu()) 
                    rendered_data_fine = self.render(poses=poses_batch, pixel_directions=pixel_directions_batch, sampling_depths=depth_samples_fine, pixel_focal_lengths=focal_lengths_batch.squeeze(1), perturb_depths=False)  # (N_pixels, 3)
                    
                    if self.args.use_sparse_fine_rendering:
                        # point cloud filtering needs both the sparsely rendered data the "unsparsely" (normal) rendered data
                        depth_samples_unsparse_fine = self.resample_depths_from_nerf_weights(number_of_pixels=poses_batch.size()[0], weights=rendered_data_coarse['depth_weights'], depth_samples=depth_samples_coarse, use_sparse_fine_rendering=False, test_render=True)  # (N_pixels, N_samples)                                    
                        unsparse_rendered_data_fine = self.render(poses=poses_batch, pixel_directions=pixel_directions_batch, sampling_depths=depth_samples_unsparse_fine, pixel_focal_lengths=focal_lengths_batch.squeeze(1), perturb_depths=False)  # (N_pixels, 3)                        


            if self.args.n_depth_sampling_optimizations > 1:
                rendered_image_fine_batches.append(rendered_data_fine['rgb_rendered'].cpu()) # (n_pixels_per_row, 3)                
                depth_image_fine_batches.append(rendered_data_fine['depth_map'].cpu()) # (n_pixels_per_row)             
                density_fine_batches.append(rendered_data_fine['density'].cpu())
                depth_weights_fine_batches.append(rendered_data_fine['depth_weights'].cpu())                                     
                                
                if self.args.use_sparse_fine_rendering:
                    rendered_image_unsparse_fine_batches.append(unsparse_rendered_data_fine['rgb_rendered'].cpu())
                    depth_image_unsparse_fine_batches.append(unsparse_rendered_data_fine['depth_map'].cpu())

            rendered_image_coarse_batches.append(rendered_data_coarse['rgb_rendered'].cpu()) # (n_pixels_per_row, 3)
            depth_image_coarse_batches.append(rendered_data_coarse['depth_map'].cpu()) # (n_pixels_per_row)                         
            density_coarse_batches.append(rendered_data_coarse['density'].cpu())            
            depth_weights_coarse_batches.append(rendered_data_coarse['depth_weights'].cpu())                        
        

        # combine batch results to compose full images
        if self.args.n_depth_sampling_optimizations > 1:            
            rendered_image_data_fine = torch.cat(rendered_image_fine_batches, dim=0).cpu() # (N_pixels, 3)            
            rendered_depth_data_fine = torch.cat(depth_image_fine_batches, dim=0).cpu()  # (N_pixels)            
            density_data_fine = torch.cat(density_fine_batches, dim=0).cpu()  # (N_pixels, N_samples)            
            depth_weights_data_fine = torch.cat(depth_weights_fine_batches, dim=0).cpu()  # (N_pixels, N_samples)            
            resampled_depths_data_fine = torch.cat(resampled_depths_fine_batches, dim=0).cpu()  # (N_pixels, N_samples)
            if self.args.use_sparse_fine_rendering:
                rendered_image_data_unsparse_fine = torch.cat(rendered_image_unsparse_fine_batches, dim=0).cpu()
                depth_image_data_unsparse_fine = torch.cat(depth_image_unsparse_fine_batches, dim=0).cpu()

        rendered_image_data_coarse = torch.cat(rendered_image_coarse_batches, dim=0).cpu() # (N_pixels, 3)
        rendered_depth_data_coarse = torch.cat(depth_image_coarse_batches, dim=0).cpu()  # (N_pixels)        
        density_data_coarse = torch.cat(density_coarse_batches, dim=0).cpu()  # (N_pixels, N_samples)
        depth_weights_data_coarse = torch.cat(depth_weights_coarse_batches, dim=0).cpu()  # (N_pixels, N_samples)
        resampled_depths_data_coarse = torch.cat(resampled_depths_coarse_batches, dim=0).cpu()  # (N_pixels, N_samples)
        
        if self.args.n_depth_sampling_optimizations > 1:
            rendered_image_fine = torch.zeros(self.H * self.W, 3).cpu()
            rendered_depth_fine = torch.zeros(self.H * self.W).cpu()
            density_fine = torch.zeros(self.H * self.W, n_samples).cpu()
            depth_weights_fine = torch.zeros(self.H * self.W, n_samples).cpu()
            resampled_depths_fine = torch.zeros(self.H * self.W, n_samples+1).cpu()        
            if self.args.use_sparse_fine_rendering:
                rendered_image_unsparse_fine = torch.zeros(self.H * self.W, 3).cpu()
                depth_image_unsparse_fine = torch.zeros(self.H * self.W).cpu()

        rendered_image_coarse = torch.zeros(self.H * self.W, 3).cpu()
        rendered_depth_coarse = torch.zeros(self.H * self.W).cpu()
        density_coarse = torch.zeros(self.H * self.W, n_samples).cpu()
        depth_weights_coarse = torch.zeros(self.H * self.W, n_samples).cpu()        
        resampled_depths_coarse = torch.zeros(self.H * self.W, n_samples+1).cpu()        

        # if we're using a mask, we need to find the right pixel positions
        if mask is not None:
            if self.args.n_depth_sampling_optimizations > 1:
                rendered_image_fine[torch.argwhere(mask==1).squeeze()] = rendered_image_data_fine.cpu()                
                rendered_depth_fine[torch.argwhere(mask==1).squeeze()] = rendered_depth_data_fine.cpu()
                if max_depth is not None:
                    rendered_image_fine[torch.argwhere(rendered_depth_fine > max_depth)] = torch.tensor([1.0, 1.0, 1.0])

                density_fine[torch.argwhere(mask==1).squeeze()] = density_data_fine.cpu()
                depth_weights_fine[torch.argwhere(mask==1).squeeze()] = depth_weights_data_fine.cpu()
                resampled_depths_fine[torch.argwhere(mask==1).squeeze()] = resampled_depths_data_fine.cpu()
                if self.args.use_sparse_fine_rendering:
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
                if max_depth is not None:
                    rendered_image_fine[torch.argwhere(rendered_depth_fine > max_depth)] = torch.tensor([1.0, 1.0, 1.0])                
                density_fine = density_data_fine.cpu()
                depth_weights_fine = depth_weights_data_fine.cpu()
                resampled_depths_fine = resampled_depths_data_fine.cpu()
                if self.args.use_sparse_fine_rendering:
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

            if self.args.use_sparse_fine_rendering:
                render_result['rendered_image_unsparse_fine'] = rendered_image_unsparse_fine
                render_result['depth_image_unsparse_fine'] = depth_image_unsparse_fine

        return render_result    


    # process raw rendered pixel data and save into images
    def save_render_as_png(self, render_result, color_file_name_fine, depth_file_name_fine, color_file_name_coarse=None, depth_file_name_coarse=None):
        
        rendered_rgb_coarse = render_result['rendered_image_coarse'].reshape(self.H, self.W, 3)
        rendered_depth_coarse = render_result['rendered_depth_coarse'].reshape(self.H, self.W)                
        rendered_color_for_file_coarse = (rendered_rgb_coarse.cpu().numpy() * 255).astype(np.uint8)    

        if self.args.n_depth_sampling_optimizations > 1:
            rendered_rgb_fine = render_result['rendered_image_fine'].reshape(self.H, self.W, 3)
            rendered_depth_fine = render_result['rendered_depth_fine'].reshape(self.H, self.W)
            rendered_color_for_file_fine = (rendered_rgb_fine.cpu().numpy() * 255).astype(np.uint8)    

        # get depth map and convert it to Turbo Color Map
        if color_file_name_coarse is not None:
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


    #########################################################################
    ############################## Training #################################
    #########################################################################

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
        

        """
            self.models["color"].train()
            self.models["color"].eval()        
            self.models["geometry"].train()
            self.models["geometry"].eval()
            self.models["pose"].train()
            self.models["pose"].eval()
            self.models["focal"].train()
            self.models["focal"].eval()        
        """

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

        # don't try to fit sensor depths that are fixed to max value
        ignore_max_sensor_depths = torch.where(sensor_depth < self.args.far_maximum_depth, 1, 0).to(self.device)

        # initialize our total weighted loss, which will be computed as the weighted sum of coarse and fine losses
        total_weighted_loss = torch.tensor(0.0).to(self.device)
        
        focal_length = self.models['focal'](0).cpu()
        poses = self.models['pose'](0).cpu()

        print(poses[0])
        print(focal_length[0])

        #print(list(self.models['pose'].parameters())[0].grad)

        

        #self.compute_ray_direction_in_camera_coordinates(focal_length)

        
        # get a tensor with the poses per pixel
        #image_ids = self.image_ids_per_pixel[indices_of_random_pixels].to(self.device) # (N_pixels)
        image_ids = self.image_ids_per_pixel[indices_of_random_pixels] # (N_pixels)        
        
        selected_poses = poses[image_ids].to(self.device) # (N_pixels, 4, 4)

        # get the focal lengths and pixel directions for every pixel given the images that were actually selected for each pixel
        selected_focal_lengths = focal_length[image_ids].to(self.device)    

        pixel_directions_selected = self.compute_pixel_directions(selected_focal_lengths, pixel_rows, pixel_cols)
        #pixel_directions_selected = self.pixel_directions[image_ids, pixel_rows, pixel_cols].to(self.device)  # (N_pixels, 3)                

        for depth_sampling_optimization in range(self.args.n_depth_sampling_optimizations):

            # get the camera intrinsics, train if this is not the coarse render
            #if self.epoch >= self.args.start_training_intrinsics_epoch and depth_sampling_optimization > 0:
            #    focal_length = self.models["focal"](0)                
            #    print('focal length: ', torch.mean(focal_length))
            #else:
            #    with torch.no_grad():
            #        focal_length = self.models["focal"](0)
                        
            # get all camera poses from model
            #if self.epoch >= self.args.start_training_extrinsics_epoch and depth_sampling_optimization > 0:
            #    poses = self.models["pose"](0) # (N_images, 4, 4)
            #else:
            #    with torch.no_grad():
            #        poses = self.models["pose"](0)  # (N_images, 4, 4)



            #####################| Sampling & Rendering |##################
            if depth_sampling_optimization == 0:
                # if this is the first iteration, collect linear depth samples to query NeRF, uniformly in space
                depth_samples = self.sample_depths_near_linearly_far_nonlinearly(number_of_pixels=n_pixels, add_noise=True) # (N_pixels, N_samples)
            else:
                # if this is not the first iteration, then resample with the latest weights
                depth_samples = self.resample_depths_from_nerf_weights(number_of_pixels=n_pixels, weights=nerf_depth_weights, depth_samples=depth_samples)  # (N_pixels, N_samples)                
            
            n_samples = depth_samples.size()[1]

            # render an image using selected rays, pose, sample intervals, and the network
            
            render_result = self.render(poses=selected_poses, pixel_directions=pixel_directions_selected, sampling_depths=depth_samples, pixel_focal_lengths=selected_focal_lengths, perturb_depths=False)  # (N_pixels, 3)
            

            rgb_rendered = render_result['rgb_rendered']         # (N_pixels, 3)
            nerf_depth_weights = render_result['depth_weights']  # (N_pixels, N_samples)
            nerf_depth = render_result['depth_map']              # (N_pixels) NeRF depth (weights x distances) for every pixel
            nerf_sample_bin_lengths_in = render_result['distances'] # (N_pixels, N_samples)            

            nerf_sample_bin_lengths = depth_samples[:, 1:] - depth_samples[:, :-1]
            nerf_depth_weights = nerf_depth_weights + self.args.epsilon            

            #####################| KL Loss |################################
            sensor_variance = self.args.depth_sensor_error
            kl_divergence_bins = -1 * torch.log(nerf_depth_weights) * torch.exp(-1 * (depth_samples[:, : n_samples - 1] * 1000 - sensor_depth_per_sample[:, : n_samples - 1] * 1000) ** 2 / (2 * sensor_variance)) * nerf_sample_bin_lengths * 1000                                
            confidence_weighted_kl_divergence_pixels = ignore_max_sensor_depths * confidence_loss_weights * torch.sum(kl_divergence_bins, 1) # (N_pixels)
            depth_loss = torch.sum(confidence_weighted_kl_divergence_pixels) / number_of_pixels_with_confident_depths
            depth_to_rgb_importance = self.get_polynomial_decay(start_value=self.args.depth_to_rgb_loss_start, end_value=self.args.depth_to_rgb_loss_end, exponential_index=self.args.depth_to_rgb_loss_exponential_index, curvature_shape=self.args.depth_to_rgb_loss_curvature_shape)

            
            #####################| Entropy Loss |###########################
            entropy_depth_loss = 0.0
            mean_entropy = torch.mean(-1 * torch.sum(nerf_depth_weights * torch.log(nerf_depth_weights), dim=1))
            if (self.epoch >= self.args.entropy_loss_tuning_start_epoch and self.epoch <= self.args.entropy_loss_tuning_end_epoch):                                                
                entropy_depth_loss = self.args.entropy_loss_weight * mean_entropy
                depth_to_rgb_importance = 0.0                                            
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
                total_weighted_loss = self.args.coarse_weight * (depth_to_rgb_importance * depth_loss + (1 - depth_to_rgb_importance) * rgb_loss + entropy_depth_loss)

                coarse_rgb_loss = rgb_loss
                coarse_depth_loss = depth_loss
                coarse_interpretable_rgb_loss_per_pixel = interpretable_rgb_loss_per_pixel
                coarse_interpretable_depth_loss_per_confident_pixel = interpretable_depth_loss_per_confident_pixel

            else:
                # Note: KL divergence loss is not used for fine iteration even though it's computed above                
                total_weighted_loss += (1.0 - self.args.coarse_weight)* ((1 - depth_to_rgb_importance) * rgb_loss)

                fine_rgb_loss = rgb_loss                
                fine_depth_loss = 0
                fine_interpretable_rgb_loss_per_pixel = interpretable_rgb_loss_per_pixel
                fine_interpretable_depth_loss_per_confident_pixel = interpretable_depth_loss_per_confident_pixel



        
        for optimizer in self.optimizers.values():
            optimizer.zero_grad() 

        # release unused GPU memory (for memory usage monitoring purposes)
        torch.cuda.empty_cache()

              

        # backward propagate the gradients to update the values which are parameters to this loss
        total_weighted_loss.backward(create_graph=False, retain_graph=False)
        
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
            wandb.log({"Coarse RGB Inverse Render Loss (0-255 per pixel)": coarse_interpretable_rgb_loss_per_pixel,
                       "Coarse Depth Sensor Loss (average millimeters error vs. sensor)": coarse_interpretable_depth_loss_per_confident_pixel,
                       "Fine RGB Inverse Render Loss (0-255 per pixel)": fine_interpretable_rgb_loss_per_pixel,
                       "Fine Depth Sensor Loss (average millimeters error vs. sensor)": fine_interpretable_depth_loss_per_confident_pixel               
                       })

        if self.epoch % self.args.log_frequency == 0:
            minutes_into_experiment = (int(time.time())-int(self.start_time)) / 60

        self.log_learning_rates()

        print("({} at {:.2f} min) - LOSS = {:.5f} -> RGB: C: {:.6f} ({:.3f}/255) | F: {:.6f} ({:.3f}/255), DEPTH: C: {:.6f} ({:.2f}mm) | F: {:.6f} ({:.2f}mm) w/ imp. {:.5f}, Entropy: {:.6f} (loss: {:.6f})".format(self.epoch, 
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
            mean_entropy,
            entropy_depth_loss
        ))
        
        # a new epoch has dawned
        self.epoch += 1        


    def sample_next_batch(self, weighted=True):

        if weighted:
            # subsample set of pixels to do weighted sampling from                        
            # technically this is sampling with replacement, but it shouldn't matter much, and using torch.randperm instead is extremely inefficient
            randomly_sampled_pixel_indices = torch.randint(self.image_ids_per_pixel.shape[0], (self.args.pixel_samples_per_epoch * 1000,) )
            
            # get 1000x number of samples to collect and do weighted sampling from this subset            
            subsampled_depth_weights = self.depth_based_pixel_sampling_weights[randomly_sampled_pixel_indices]            
            subsampled_indices_from_weights = torch.multinomial(input=subsampled_depth_weights, num_samples=self.args.pixel_samples_per_epoch, replacement=False)
            
            # now we need to grab from our global pixel indices which ones we actually selected, after weighted sampling
            indices_of_random_pixels_for_this_epoch = randomly_sampled_pixel_indices[subsampled_indices_from_weights]            
                                    
            print('Avg (sensor) depth of samples: ', torch.mean(self.rgbd[indices_of_random_pixels_for_this_epoch, 3] ))
            
        else:            
            indices_of_random_pixels_for_this_epoch = random.sample(population=self.pixel_indices_below_max_depth, k=self.args.pixel_samples_per_epoch)            

        #return indices_of_random_pixels_for_this_epoch.to(self.device)
        return indices_of_random_pixels_for_this_epoch
            


    #########################################################################
    ############################# Experiments ###############################
    #########################################################################

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
        focal_length = self.models["focal"](0)
        # intial: 529.9991
        # learned: 541.8873
        # test: 600
        focal_length = focal_length  #+ (529.9991 - 541.8873)
        self.compute_ray_direction_in_camera_coordinates(focal_length)        

        #for image_index in np.array(self.test_image_indices):
        #for image_index in [self.test_image_indices[i] for i in [0,106,148]  ]:
        for image_index in [self.test_image_indices[i] for i in [0,159,222]  ]:
                                    
            pixel_indices = torch.argwhere(self.image_ids_per_pixel == image_index)
            this_image_rgbd = self.rgbd[pixel_indices].cpu().squeeze(1)
            rgb = this_image_rgbd[:, :3]

            ground_truth_image = torch.zeros(self.H * self.W, 3).cpu()
            mask = torch.zeros(self.H, self.W)                    
            pixel_rows = self.pixel_rows[pixel_indices]
            pixel_cols = self.pixel_cols[pixel_indices]        
            mask[pixel_rows, pixel_cols] = 1
            mask = mask.flatten()                
                        
            ground_truth_image[torch.argwhere(mask==1).squeeze()] = rgb.cpu()
            ground_truth_image = ground_truth_image.reshape(self.H, self.W, 3)            
            
            # always render              
            print("Rendering for image {}".format(image_index))            

            #render_result = self.render_prediction_for_train_image(train_image_index=image_index)            
            render_result = self.basic_render(self.models['pose'](0)[image_index], self.pixel_directions[image_index], self.models['focal'](0)[image_index], test_render=True)

            #nerf_weights_coarse = render_result['depth_weights_coarse']
            #density_coarse = render_result['density_coarse']                                
            depth_coarse = render_result['rendered_depth_coarse']            
            #sampled_depths_coarse = render_result['resampled_depths_coarse']

            if self.args.n_depth_sampling_optimizations > 1:
                #nerf_weights_fine = render_result['depth_weights_fine']
                #density_fine = render_result['density_fine']
                depth_fine = render_result['rendered_depth_fine']            
                #sampled_depths_fine = render_result['resampled_depths_fine']         
                                
            # save rendered rgb and depth images
            out_file_suffix = str(image_index)
            color_file_name_fine = os.path.join(self.color_out_dir, str(out_file_suffix).zfill(4) + '_color_fine_{}.png'.format(epoch))
            depth_file_name_fine = os.path.join(self.depth_out_dir, str(out_file_suffix).zfill(4) + '_depth_fine_{}.png'.format(epoch))            
            color_file_name_coarse = os.path.join(self.color_out_dir, str(out_file_suffix).zfill(4) + '_color_coarse_{}.png'.format(epoch))
            depth_file_name_coarse = os.path.join(self.depth_out_dir, str(out_file_suffix).zfill(4) + '_depth_coarse_{}.png'.format(epoch))                        
            #self.save_render_as_png(render_result, color_file_name_fine, depth_file_name_fine, color_file_name_coarse, depth_file_name_coarse)
            self.save_render_as_png(render_result, color_file_name_fine, depth_file_name_fine, None, None)
            
            # optional: save rendered depth as a pointcloud
            if epoch % self.args.save_point_cloud_frequency == 0 and epoch != 0:
                print("Saving .ply for view {}".format(image_index))
                depth_view_file_name = os.path.join(self.depth_view_out_dir, str(out_file_suffix).zfill(4) + '_depth_view_{}.png'.format(epoch))
                pose = self.models['pose'](0)[image_index].to(device=self.device)
                unsparse_rendered_rgb_img = None

                if self.args.n_depth_sampling_optimizations > 1: 
                    depth_img = render_result['rendered_depth_fine'].reshape(self.H, self.W).to(device=self.device)
                    rendered_rgb_img = render_result['rendered_image_fine'].reshape(self.H, self.W, 3).to(device=self.device)
                    if self.args.use_sparse_fine_rendering:
                        unsparse_rendered_rgb_img = render_result['rendered_image_unsparse_fine'].reshape(self.H, self.W, 3).to(device=self.device)
                        unsparse_depth_img = render_result['depth_image_unsparse_fine'].reshape(self.H, self.W).to(device=self.device)
                    else:
                        unsparse_rendered_rgb_img = None
                        unsparse_depth_img = None
                else:
                    depth_img = render_result['rendered_depth_coarse'].reshape(self.H, self.W).to(device=self.device)
                    rendered_rgb_img = render_result['rendered_image_coarse'].reshape(self.H, self.W, 3).to(device=self.device)

                entropy_image = torch.zeros(self.H * self.W, 1) #.cpu()  
                
                pixel_indices_filter = torch.zeros(self.rgbd.size()[0]).to(device=self.device)
                pixel_indices_filter[self.pixel_indices_below_max_depth] = 1                    
                filtered_pixel_indices = torch.argwhere(torch.logical_and(self.image_ids_per_pixel == image_index, pixel_indices_filter == 1))            
                filtered_pixel_rows = self.pixel_rows[filtered_pixel_indices]
                filtered_pixel_cols = self.pixel_cols[filtered_pixel_indices]

                max_depth_filter_image = torch.zeros(self.H, self.W) #.cpu()                
                max_depth_filter_image[filtered_pixel_rows, filtered_pixel_cols] = 1
                #max_depth_filter_image = max_depth_filter_image.reshape(self.H, self.W)                                              
                                
                coarse_weights_entropy = -1 * torch.sum( (nerf_weights_coarse+self.args.epsilon) * torch.log(nerf_weights_coarse + self.args.epsilon), dim=1)
                entropy_image = coarse_weights_entropy.reshape(self.H, self.W)                
                this_image_pixel_directions = self.pixel_directions[image_index]                                

                self.get_point_cloud(pose=pose, depth=depth_img, rgb=ground_truth_image, pixel_directions=this_image_pixel_directions, label="_{}_{}".format(epoch,image_index), save=True, remove_zero_depths=True, save_raw_xyz=True, dir=self.depth_view_out_dir, entropy_image=entropy_image, sparse_rendered_rgb_img=rendered_rgb_img, unsparse_rendered_rgb_img=unsparse_rendered_rgb_img, unsparse_depth=unsparse_depth_img, max_depth_filter_image=max_depth_filter_image)               

            # optional: save graphs of nerf density weights visualization
            if epoch % self.args.save_depth_weights_frequency == 0 and epoch != 0:               
                print("Creating depth weight graph for view {}".format(image_index))
                Path("{}/depth_weights_visualization/{}/image_{}/".format(self.experiment_dir, epoch, image_index)).mkdir(parents=True, exist_ok=True)
                view_row = 155
                #view_row = 240+10
                pixel_rows = self.pixel_rows[pixel_indices]
                pixel_cols = self.pixel_cols[pixel_indices]                

                start_pixel = torch.sum(torch.where(pixel_rows.squeeze(1) < view_row, 1, 0))
                end_pixel = torch.sum(torch.where(pixel_rows.squeeze(1) < view_row + 1, 1, 0))

                for pixel_index in range(start_pixel, end_pixel):                                                

                    out_path = Path("{}/{}/".format(self.depth_weights_out_dir, epoch))
                    out_path.mkdir(parents=True, exist_ok=True)
                    sensor_depth = this_image_rgbd[pixel_index, 3]                    

                    if self.args.n_depth_sampling_optimizations > 1:                    
                        weights_fine = nerf_weights_fine.squeeze(0)[pixel_index]                                        
                        densities_fine = density_fine.squeeze(0)[pixel_index]  
                        predicted_depth = depth_fine[pixel_index]                                        

                    image = ground_truth_image
                    downsized_image = torch.nn.functional.interpolate(image.permute(2, 0, 1).unsqueeze(0), size=(int(image.shape[0]/4), int(image.shape[1]/4)), mode='bilinear').squeeze().permute(1, 2, 0)                    
                    pixel_row_to_highlight = pixel_rows[pixel_index][0]                                    
                    pixel_col_to_highlight = pixel_cols[pixel_index][0]                    
                    pixel_row_to_highlight = int(pixel_row_to_highlight / 4)
                    pixel_col_to_highlight = int(pixel_col_to_highlight / 4)
                    downsized_image[pixel_row_to_highlight, pixel_col_to_highlight, :] = torch.tensor([1.0,0.0,0.0], dtype=torch.float32)     
                    downsized_image = (downsized_image * 255).to(torch.long)
                    downsized_image = downsized_image.cpu().numpy().astype(np.uint8)
                    downsized_path = Path("{}/downsized_image_{}.png".format(self.experiment_dir, image_index))
                    imageio.imwrite(downsized_path, downsized_image)                    

                    densities_coarse = density_coarse.squeeze(0)[pixel_index]  
                    weights_coarse = nerf_weights_coarse.squeeze(0)[pixel_index]                   
                    weights_fine = nerf_weights_fine.squeeze(0)[pixel_index]
                    predicted_depth_coarse = depth_coarse[pixel_index]
                    coarse_weights_entropy = -1 * torch.sum( (weights_coarse+self.args.epsilon) * torch.log(weights_coarse + self.args.epsilon), dim=0)
                    fig, axs = plt.subplots(3,2)                    

                    max_depth = 0.5
                    avg_fine_sample_depth = torch.mean(sampled_depths_fine[pixel_index, : sampled_depths_fine.size()[1]-1])
                    if self.args.n_depth_sampling_optimizations > 1:
                        predicted_depth_fine = depth_fine[pixel_index]                    
                        axs[1,0].scatter(sampled_depths_fine[pixel_index, : sampled_depths_fine.size()[1]-1], weights_fine, s=1, marker='o', c='green')                    
                        axs[1,0].scatter([sensor_depth.item()], [0], s=30, marker='o', c='red')                    
                        axs[1,0].scatter([predicted_depth_fine.item()], [0], s=5, marker='o', c='blue')
                        #axs[1,0].set_xlim([avg_fine_sample_depth - offset, avg_fine_sample_depth + offset])
                        axs[1,0].set_xlim([0,max_depth])
                                        
                    axs[0,0].scatter(sampled_depths_coarse[pixel_index, : sampled_depths_coarse.size()[1]-1], weights_coarse, s=1, marker='o', c='green')                    
                    axs[0,0].scatter([sensor_depth.item()], [0], s=30, marker='o', c='red')                    
                    axs[0,0].scatter([predicted_depth_coarse.item()], [0], s=5, marker='o', c='blue')                    
                    #axs[0,0].set_xlim([avg_fine_sample_depth - offset, avg_fine_sample_depth + offset])
                    axs[0,0].set_xlim([0,max_depth])
                    if self.args.n_depth_sampling_optimizations > 1:                                            
                        axs[1,1].scatter(sampled_depths_fine[pixel_index, : sampled_depths_fine.size()[1]-1], densities_fine, s=1, marker='o', c='green')                    
                        axs[1,1].scatter([sensor_depth.item()], [0], s=30, marker='o', c='red')                    
                        axs[1,1].scatter([predicted_depth_fine.item()], [0], s=5, marker='o', c='blue')
                        #axs[1,1].set_xlim([avg_fine_sample_depth - offset, avg_fine_sample_depth + offset])
                        axs[1,1].set_xlim([0,max_depth])
                    
                    axs[0,1].scatter(sampled_depths_coarse[pixel_index, : sampled_depths_coarse.size()[1]-1], densities_coarse, s=1, marker='o', c='green')                    
                    axs[0,1].scatter([sensor_depth.item()], [0], s=30, marker='o', c='red')                    
                    axs[0,1].scatter([predicted_depth_coarse.item()], [0], s=5, marker='o', c='blue')                    
                    #axs[0,1].set_xlim([avg_fine_sample_depth - offset, avg_fine_sample_depth + offset])
                    axs[0,1].set_xlim([0,max_depth])

                    coarse_weights_sum = torch.sum(weights_coarse, 0)
                    fine_weights_sum = torch.sum(weights_fine, 0)
                    text_kwargs = dict(ha='center', va='center', fontsize=8, color='black')   

                    depth_diff = 1337.69420
                    if self.args.use_sparse_fine_rendering:
                        predicted_depth_unsparse_fine = (render_result['depth_image_unsparse_fine'])[pixel_index]
                        depth_diff = torch.abs(predicted_depth_unsparse_fine - predicted_depth_fine)
                    
                    axs[2,0].text(0.5,0.5, 'coarse weights entropy: {:.6f}\npixel coordinates: ({},{})\ncoarse weights sum: {:.6f}\nfine weights sum: {:.6f}\nsparse/unsparse depth diff: {:.6f}'.format(coarse_weights_entropy, self.pixel_rows[pixel_indices[pixel_index]][0], self.pixel_cols[pixel_indices[pixel_index]][0], coarse_weights_sum, fine_weights_sum, depth_diff) ,  **text_kwargs)
                    axs[2,0].axis('off')

                    axs[2,1].imshow(downsized_image)
                    axs[2,1].axis('off')
                    
                    out_file_suffix = 'image_{}/{}'.format(image_index, image_index)
                    depth_weights_file_name = os.path.join(out_path, str(out_file_suffix).zfill(4) + '_depth_weights_e{}_p{}.png'.format(epoch, pixel_index))
                    plt.savefig(depth_weights_file_name)
                    plt.close()
                    
            # save graphs of learning rate histories
            for topic in ["color", "geometry", "pose", "focal"]:                                            
                lrs = self.learning_rate_histories[topic]
                if len(lrs) > 0:
                    plt.figure()
                    plt.plot([x for x in range(len(lrs))], lrs)
                    plt.ylim(0.0, max(lrs)*1.1)
                    plt.savefig('{}/{}.png'.format(self.learning_rates_out_dir, topic))
                    plt.close()

        if self.args.export_test_data_for_post_processing:
            print("System exiting after exporting geometry data")
            sys.exit(0)   

        torch.cuda.empty_cache()

    def load_saved_args_train(self):
        
        self.args.base_directory = './data/orchid'
        self.args.images_directory = 'color'
        self.args.images_data_type = 'jpg'            
        self.args.load_pretrained_models = False
        self.args.reset_learning_rates = False
        self.args.pretrained_models_directory = '/home/ubuntu/research/nerf/data/petrified_bonsai/hyperparam_experiments/1673093521_depth_loss_0.01_to_0.0_k9_N1_NeRF_Density_LR_0.001_to_0.0001_k4_N1_pose_LR_0.0005_to_1e-05_k9_N1'
        self.args.start_epoch = 1
        self.args.number_of_epochs = 500000
        

        #self.args.start_training_extrinsics_epoch = 500        
        #self.args.start_training_intrinsics_epoch = 5000
        self.args.start_training_extrinsics_epoch = 500
        self.args.start_training_intrinsics_epoch = 1000

        self.args.start_training_color_epoch = 0
        self.args.start_training_geometry_epoch = 0
        self.args.entropy_loss_tuning_start_epoch = 5001
        self.args.entropy_loss_tuning_end_epoch = 1000000
        self.args.entropy_loss_weight = 0.001

        self.args.nerf_density_lr_start = 0.0005
        #self.args.nerf_density_lr_end = 0.000025
        self.args.nerf_density_lr_end = 0.0001
        self.args.nerf_density_lr_exponential_index = 4
        self.args.nerf_density_lr_curvature_shape = 1

        self.args.nerf_color_lr_start = 0.0005
        self.args.nerf_color_lr_end = 0.0001
        self.args.nerf_color_lr_exponential_index = 2
        self.args.nerf_color_lr_curvature_shape = 1

        self.args.focal_lr_start = 0.0001        
        self.args.focal_lr_end = 0.0000025
        self.args.focal_lr_exponential_index = 9
        self.args.focal_lr_exponential_index = 1

        self.args.pose_lr_start = 0.0003
        self.args.pose_lr_end = 0.00001
        self.args.pose_lr_exponential_index = 9
        self.args.pose_lr_curvature_shape = 1

        self.args.depth_to_rgb_loss_start = 0.01        
        self.args.depth_to_rgb_loss_end = 0.0        

        self.args.depth_to_rgb_loss_exponential_index = 9
        self.args.depth_to_rgb_loss_curvature_shape = 1

        self.args.density_neural_network_parameters = 256
        self.args.color_neural_network_parameters = 256
        self.args.directional_encoding_fourier_frequencies = 8
        
        self.args.epsilon = 0.0000001
        #self.args.depth_sensor_error = 0.5
        self.args.depth_sensor_error = 0.5
        self.args.min_confidence = 2.0

        ### test images
        self.args.skip_every_n_images_for_testing = 1
        self.args.number_of_test_images = 250

        ### test frequency parameters
        self.args.test_frequency = 5000
        self.args.export_test_data_for_testing = False
        self.args.save_ply_point_clouds_of_sensor_data = False
        self.args.save_ply_point_clouds_of_sensor_data_with_learned_poses = False        
        self.args.save_point_cloud_frequency = 1000000
        self.args.save_depth_weights_frequency = 5000000000
        self.args.log_frequency = 1
        self.args.save_models_frequency = 5000

        

        # training
        self.args.pixel_samples_per_epoch = 1000
        self.args.number_of_samples_outward_per_raycast = 360
        self.args.skip_every_n_images_for_training = 60
        self.args.number_of_pixels_in_training_dataset = 640 * 480 * 200
        self.args.resample_pixels_frequency = 5000        

        # testing
        self.args.number_of_pixels_per_batch_in_test_renders = 5000
        self.args.number_of_samples_outward_per_raycast_for_test_renders = self.args.number_of_samples_outward_per_raycast
        

        self.args.use_sparse_fine_rendering = False        
                
        self.args.near_maximum_depth = 0.5
        self.args.far_maximum_depth = 3.00  
        self.args.percentile_of_samples_in_near_region = 0.80 


    def load_saved_args_test(self):        

        self.load_saved_args_train()        
        self.args.load_pretrained_models = True
        self.args.n_depth_sampling_optimizations = 2

        self.args.pretrained_models_directory = './data/orchid/hyperparam_experiments/385k_180samples'
        self.args.reset_learning_rates = False # start and end indices of learning rate schedules become {0, number_of_epochs}
                
        self.args.start_epoch = 200001
        self.args.number_of_epochs = 1

        self.args.save_models_frequency = 999999999        
        self.args.number_of_test_images = 250

        self.args.skip_every_n_images_for_testing = 1

        self.args.near_maximum_depth = 0.5
        self.args.far_maximum_depth = 3.00

        self.args.number_of_samples_outward_per_raycast_for_test_renders = 180

        self.args.number_of_pixels_per_batch_in_test_renders = 4000
        self.args.test_frequency = 1
        self.args.save_depth_weights_frequency = 1000000000
        self.args.save_point_cloud_frequency = 1000000000

        self.args.use_sparse_fine_rendering = False
        
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

# Based on https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10

    def plot_grad_flow(named_parameters):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    ## MEM utils ##
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


if __name__ == '__main__':
    # Load a scene object with all data and parameters

    with torch.no_grad():
        scene = SceneModel(args=parse_args(), experiment_args='train')

    while scene.epoch < scene.args.start_epoch + scene.args.number_of_epochs:    

        
        if scene.epoch == scene.args.start_epoch:
            with torch.no_grad():                                            
                #scene.test()
                print("")
                
                
        #scene.print_memory_usage()
        
        if scene.epoch != 1 and (scene.epoch-1) % scene.args.resample_pixels_frequency == 0:
            print('Resampling training data...')
            scene.load_image_and_depth_data_within_xyz_bounds(visualize_masks=False)


        batch = scene.sample_next_batch(weighted=True)
        
        # note: detect_anomaly is incompatible with Inductor
        #with torch.autograd.detect_anomaly():  # no need to enable this because there are never errors
        scene.train(batch) 


        #network = make_dot(scene.models['pose'](0), params=dict(scene.models['pose'].named_parameters()), show_attrs=True, show_saved=True)
        #print(network)
        #quit()
       
        if (scene.epoch-1) % scene.args.save_models_frequency == 0 and (scene.epoch-1) !=  scene.args.start_epoch:
            scene.save_models()


        if (scene.epoch-1) % scene.args.test_frequency == 0 and (scene.epoch-1) != 0:
            with torch.no_grad():                
                scene.test()
