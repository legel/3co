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
from scipy.spatial.transform import Rotation as R

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
from functorch.compile import make_boxed_func

#from torchviz import make_dot, make_dot_from_trace
from torch.autograd import Variable

import gc
from utils.camera import *

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

set_randomness()

def parse_args():
    parser = argparse.ArgumentParser()

    # Define path to relevant data for training, and decide on number of images to use in training
    parser.add_argument('--base_directory', type=str, default='./data/dragon_scale', help='The base directory to load and save information from')
    parser.add_argument('--images_directory', type=str, default='color', help='The specific group of images to use during training')
    parser.add_argument('--images_data_type', type=str, default='jpg', help='Whether images are jpg or png')
    parser.add_argument('--number_of_images_for_training', type=int, default=120, help='The number of images sampled from all images for training')
    parser.add_argument('--H_for_training', type=int, default=480, help='The height in pixels that training images will be downsampled to')
    parser.add_argument('--W_for_training', type=int, default=640, help='The width in pixels that training images will be downsampled to')
    parser.add_argument('--number_of_pixels_in_training_dataset', type=int, default=480*640*100, help='The total number of pixels sampled from all images in training data')
    parser.add_argument('--resample_pixels_frequency', type=int, default=5000, help='Resample training data every this number of epochs')
    parser.add_argument('--save_models_frequency', type=int, default=25000, help='Save model every this number of epochs')
    parser.add_argument('--load_pretrained_models', type=bool, default=False, help='Whether to start training from models loaded with load_pretrained_models()')
    parser.add_argument('--pretrained_models_directory', type=str, default='./data/dragon_scale/hyperparam_experiments/1662755515_depth_loss_0.00075_to_0.0_k9_N1_NeRF_Density_LR_0.0005_to_0.0001_k4_N1_pose_LR_0.0025_to_1e-05_k9_N1', help='The directory storing models to load')    
    parser.add_argument('--reset_learning_rates', type=bool, default=False, help='When loading pretrained models, whether to reset learning rate schedules instead of resuming them')
    parser.add_argument('--H_for_test_renders', type=int, default=480, help='The image height used for test renders and pointclouds')
    parser.add_argument('--W_for_test_renders = 640', type=int, default=640, help='The image width used for test renders and pointclouds')                
    
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
    parser.add_argument('--save_depth_weights_frequency', default=200002, type=int, help='Frequency of epochs to save density depth weight visualizations')
    parser.add_argument('--log_frequency', default=1, type=int, help='Frequency of epochs to log outputs e.g. loss performance')        
    parser.add_argument('--number_of_test_images', default=2, type=int, help='Index in the training data set of the image to show during testing')
    parser.add_argument('--skip_every_n_images_for_testing', default=80, type=int, help='Skip every Nth testing image, to ensure sufficient test view diversity in large data set')    
    parser.add_argument('--number_of_pixels_per_batch_in_test_renders', default=128, type=int, help='Size in pixels of each batch input to rendering')                

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

    # Define parameters the determine the overall size and learning capacity of the neural networks and their encodings
    parser.add_argument('--density_neural_network_parameters', type=int, default=256, help='The baseline number of units that defines the size of the NeRF geometry network')
    parser.add_argument('--color_neural_network_parameters', type=int, default=256, help='The baseline number of units that defines the size of the NeRF RGB (pitch,yaw) network')
    parser.add_argument('--directional_encoding_fourier_frequencies', type=int, default=10, help='The number of frequencies that are generated for positional encoding of (pitch, yaw)')

    # Define sampling parameters, including how many samples per raycast (outward), number of samples randomly selected per image, and (if masking is used) ratio of good to masked samples
    parser.add_argument('--pixel_samples_per_epoch', type=int, default=1000, help='The number of rows of samples to randomly collect for each image during training')
    parser.add_argument('--number_of_samples_outward_per_raycast', type=int, default=360, help='The number of samples per raycast to collect for each rendered pixel during training')        
    parser.add_argument('--number_of_samples_outward_per_raycast_for_test_renders', type=int, default=360, help='The number of samples per raycast to collect for each rendered pixel during testing')        

    # Define depth sensor parameters
    parser.add_argument('--depth_sensor_error', type=float, default=0.5, help='A rough estimate of the 1D-Gaussian-modeled depth sensor, in millimeters')
    parser.add_argument('--epsilon', type=float, default=1e-10, help='Minimum value in log() for NeRF density weights going to 0')    

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
    def __init__(self, args, experiment_args = None, dynamic_args = None):
        self.sample_distances = []

        self.args = args
        self.load_args(experiment_args, dynamic_args)

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
                
        # prepare test evaluation indices
        self.prepare_test_data()        
        
        torch.set_float32_matmul_precision('high')
        torch._dynamo.config.verbose=True        
        torch._dynamo.config.suppress_errors = False

        self.initialize_models()        
        self.initialize_learning_rates()  

        if self.args.load_pretrained_models:
            print("Loading pretrained models")
            self.load_pretrained_models()                        
            #self.export_camera_extrinsics_and_intrinsics_from_trained_model('./', '500000')
            #quit()
        else:            
            print("Training from scratch")
            
        # now sample the first set of pixels to train on
        self.sample_training_data(visualize_sampled_pixels=False)
        


    #########################################################################
    ################ Loading and initial processing of data #################
    #########################################################################


    def load_args(self, experiment_args = None, dynamic_args = None):        
        if experiment_args == 'train': 
            print('\n------------------------------------------------------')
            print('------------------- Using train args -----------------')                       
            print('------------------------------------------------------')
            self.load_saved_args_train()
        elif experiment_args == 'dynamic':
            print('\n------------------------------------------------------')
            print('------------------- Using dynamic args ---------------')                       
            print('------------------------------------------------------')            
            if dynamic_args == None:
                sys.exit("dynamic_args must be provided for dynamic experiment")
            self.load_saved_args_train()
            for k, v in dynamic_args.items():
                exec(f"self.args.{k} = {v}")
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
        n_images = len(self.image_ids)        
        self.skip_every_n_images_for_training = int(n_images / self.args.number_of_images_in_training_dataset)                

        print('n images: ', n_images)
        print('skip every: ', self.skip_every_n_images_for_training)
        #print('n selected images', len(self.image_ids[::self.skip_every_n_images_for_training][]) )

        
        
        # load the first image to establish initial height and width of image data because it gets downsampled
        self.load_image_data(0)
    

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
        image_data = np.array(Image.open(image_path))
        image_data = cv2.resize(image_data, (self.args.W_for_training, self.args.H_for_training))

        self.H = int(image_data.shape[0])
        self.W = int(image_data.shape[1])        
        
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
        near_bound = np.min(resized_depth_meters)
        far_bound = np.max(resized_depth_meters)        
        depth = torch.Tensor(resized_depth_meters).cpu() # (N_images, H_image, W_image)

        return depth, near_bound, far_bound, confidence_data


    def load_camera_intrinsics(self):
        # load camera instrinsics estimates from Apple's internal API
        camera_intrinsics_data = json.load(open(os.path.join(self.args.base_directory,'camera_intrinsics.json')))
        intrinsics_H = int(camera_intrinsics_data["height"])
        intrinsics_W = int(camera_intrinsics_data["width"])

        print("Image with size (H,W)=({},{})".format(self.H, self.W))

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


    def export_camera_extrinsics_and_intrinsics_from_trained_model(self, dir, epoch, save=True):
        # gather the latest poses ("camera extrinsics")
        camera_extrinsics = self.models['pose']()([0]).cpu()    
       
        # gather the latest focal lengths (f_x and f_y, which are currently identical)
        focal_length = self.models['focal']()([0]).cpu()      

        # get the number of cameras represented
        n_cameras = camera_extrinsics.shape[0]
        
        # format camera intrinsics for export
        camera_intrinsics = torch.zeros(size=(n_cameras,3,3), dtype=torch.float32)
        camera_intrinsics[:,0,0] = focal_length
        camera_intrinsics[:,1,1] = focal_length
        camera_intrinsics[:,0,2] = self.principal_point_x
        camera_intrinsics[:,1,2] = self.principal_point_y
        
        # use index [2,2] in intrinsics to store the image index that the extrinsics/intrinsics correspond to
        image_indices = torch.arange(start=0, end=self.args.number_of_images_in_training_dataset) * self.skip_every_n_images_for_training
        
        camera_intrinsics[:, 2, 2] = image_indices.to(torch.float32)

        torch.save(camera_intrinsics, '{}/camera_intrinsics_{}.pt'.format(dir, epoch))
        torch.save(camera_extrinsics, '{}/camera_extrinsics_{}.pt'.format(dir, epoch))

        print(camera_intrinsics[:, 2,2])
        quit()

        return (camera_extrinsics, camera_intrinsics)


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


    def sample_training_data(self, visualize_sampled_pixels=False, save_ply_point_clouds_of_sensor_data=False):
        self.rgbd = []
        self.image_ids_per_pixel = []
        self.pixel_rows = []
        self.pixel_cols = []
        self.xyz_per_view = []
        self.confidence_per_pixel = []                      
        neighbor_distance_per_pixel = []
        self.sample_image = torch.zeros(self.H, self.W)
    
        n_pixels_in_training_dataset = self.args.number_of_pixels_in_training_dataset
        n_images = self.args.number_of_images_in_training_dataset
        n_pixels_per_image = min(n_pixels_in_training_dataset // n_images, self.H*self.W)

        # create meshgrid representing rows and cols, which will be used for rendering full images
        pixel_rows_and_cols_for_test_renders = torch.meshgrid(
            torch.arange(self.args.H_for_test_renders, dtype=torch.float32, device=self.device),
            torch.arange(self.args.W_for_test_renders, dtype=torch.float32, device=self.device),
            indexing='ij'
        )  # (H, W)
     
        self.pixel_rows_for_test_renders = pixel_rows_and_cols_for_test_renders[0].flatten()
        self.pixel_cols_for_test_renders = pixel_rows_and_cols_for_test_renders[1].flatten()            

        # now loop through all of the data, and filter out (only load and save as necessary) based on whether the points land within our focus area
        for i, image_id in enumerate(self.image_ids[::self.skip_every_n_images_for_training][:self.args.number_of_images_in_training_dataset]):
        

            # get depth data for this image
            depth, near_bound, far_bound, confidence = self.load_depth_data(image_id=image_id) # (H, W)

            # get (x,y,z) coordinates for this image                                    
            xyz_coordinates = get_sensor_xyz_coordinates(self.all_initial_poses[i*self.skip_every_n_images_for_training], depth, self.H, self.W, self.principal_point_x, self.principal_point_y, self.initial_focal_length[0].expand(self.H*self.W).cpu()) # (H, W, 3)

            # select a uniformly random subset of those pixels
            all_indices = torch.tensor(range(xyz_coordinates.size()[0] * xyz_coordinates.size()[1]))     
            
            # randperm is more accurate but extremely slow
            #pixel_indices_selected = all_indices[ torch.randperm(xyz_coordinates.size()[0] * xyz_coordinates.size()[1])[:n_pixels_per_image] ]
            pixel_indices_selected = all_indices[ torch.randint(high = xyz_coordinates.size()[0] * xyz_coordinates.size()[1], size=(n_pixels_per_image,))]
            selected = torch.zeros(xyz_coordinates.size()[0]*xyz_coordinates.size()[1])
            selected[pixel_indices_selected] = 1
            selected = selected.reshape(xyz_coordinates.size()[0], xyz_coordinates.size(1))            
            pixel_indices_selected = torch.argwhere(selected==1)            
            number_of_selected_pixels = pixel_indices_selected.size()[0]
                                                
            # get the rows and cols of the selected pixel rows and pixel columns
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
            
            # now, load the (r,g,b) image and filter only the pixels we're focusing on
            image, _ = self.load_image_data(image_id=image_id)

            # set last dimension to be channels dimension (reshape changes ordering in this case)                                     
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
            image_id_for_all_pixels = torch.full(size=[number_of_selected_pixels], fill_value=i)
            self.image_ids_per_pixel.append(image_id_for_all_pixels)
            
            # optional visualizations of training data
            if save_ply_point_clouds_of_sensor_data and i in self.test_image_indices:
                pcd = self.create_point_cloud(xyz_coordinates=xyz_coordinates_selected, colors=image[pixel_rows_selected, pixel_cols_selected], normals=None, label="point_cloud_raw_{}".format(image_id), flatten_xyz=False, flatten_image=False)                
                file_name = "{}/view_{}.ply".format(self.experiment_dir, "pc_raw_{}".format(image_id), self.epoch-1)
                o3d.io.write_point_cloud(file_name, pcd, write_ascii = True)
            if visualize_sampled_pixels:
                color_out_path = Path("{}/mask_for_filtering_{}.png".format(self.experiment_dir, image_id))                
                image = np.zeros(shape=(self.H,self.W,3))
                image[pixel_rows_selected, pixel_cols_selected] = ((rgbd_selected[:, :3]).cpu().numpy() * 255).astype(np.uint8) 
                imageio.imwrite(color_out_path, image)                
        

        # bring the data together        
        self.xyz = torch.cat(self.xyz_per_view, dim=0).cpu() # to(device=torch.device('cpu')) # 
        self.rgbd = torch.cat(self.rgbd, dim=0)                        
        self.pixel_rows = torch.cat(self.pixel_rows, dim=0)#.unsqueeze(1)
        self.pixel_cols = torch.cat(self.pixel_cols, dim=0)#.unsqueeze(1)                
        self.near = torch.min(self.rgbd[:,3])
        self.far = torch.max(self.rgbd[:,3])        

        self.image_ids_per_pixel = torch.cat(self.image_ids_per_pixel, dim=0).cpu()
        self.confidence_per_pixel = torch.cat(self.confidence_per_pixel, dim=0)
        neighbor_distance_per_pixel = torch.cat(neighbor_distance_per_pixel, dim=0)        

        print('\n=== Average sensor depth of non-weighted samples: === {}\n'.format(torch.mean(self.rgbd[:, 3] )))                
        print("The near bound is {:.3f} meters and the far bound is {:.3f} meters".format(self.near, self.far))        

        # compute sampling weights
        self.depth_based_pixel_sampling_weights = torch.ones(self.rgbd.size()[0]).cpu()    
        self.depth_based_pixel_sampling_weights = (1 / ((self.rgbd[:,3]+self.near) ** (0.66))).cpu() # bias sampling of closer pixels probabilistically                        
        
        max_rgb_distance = np.sqrt(3)
        steepness = 20.0
        neighbor_rgb_distance_sampling_weights = torch.log2( (steepness * neighbor_distance_per_pixel / max_rgb_distance + 1.0))        
        self.depth_based_pixel_sampling_weights = self.depth_based_pixel_sampling_weights * neighbor_rgb_distance_sampling_weights        
                    
        print("Loaded {} images with {:,} pixels selected".format(i+1, self.image_ids_per_pixel.shape[0] ))


    #########################################################################
    ############ Initialize models and set learning parameters ##############
    #########################################################################

    def initialize_models(self):
                
        self.models = {}                
        
        model = torch.compile(CameraIntrinsicsModel(self.H, self.W, self.initial_focal_length, self.args.number_of_images_in_training_dataset))
        self.models["focal"] = model
        model.to(self.device)
        
        model = torch.compile(CameraPoseModel(self.all_initial_poses[::self.skip_every_n_images_for_training][:self.args.number_of_images_in_training_dataset]))                                
        self.models["pose"] = model
        model.to(self.device)

        model =  torch.compile(NeRFDensity(self.args))
        self.models["geometry"] = model
        model.to(self.device)

        model = torch.compile(NeRFColor(self.args))
        self.models["color"] = model
        model.to(self.device)

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
        print('Loading pretrained model at {}/[model]_{}.pth'.format(self.args.pretrained_models_directory, self.args.start_epoch-1))
        for model_name in self.models.keys():

            model_path = "{}/{}_{}.pth".format(self.args.pretrained_models_directory, model_name, self.args.start_epoch-1)            
                        
            # load checkpoint data
            ckpt = torch.load(model_path, map_location=self.device)
            
            # this fixes loading for a small set of runs (orchid-204, cactus-204, philodendron-204, cycad-204, red_berry_bonsai-204)
            if model_name == 'focal':
                ckpt['model_state_dict']['_orig_mod.fx'] = ckpt['model_state_dict']['_orig_mod.fx'][:self.args.number_of_images_in_training_dataset]
            elif model_name == 'pose':
                ckpt['model_state_dict']['_orig_mod.t'] = ckpt['model_state_dict']['_orig_mod.t'][:self.args.number_of_images_in_training_dataset]
                ckpt['model_state_dict']['_orig_mod.r'] = ckpt['model_state_dict']['_orig_mod.r'][:self.args.number_of_images_in_training_dataset]
                ckpt['model_state_dict']['_orig_mod.poses'] = ckpt['model_state_dict']['_orig_mod.poses'][:self.args.number_of_images_in_training_dataset]


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
        for i, image_id in enumerate(self.image_ids[::self.skip_every_n_images_for_training][:self.args.number_of_images_in_training_dataset]):
            print("Saving with learned poses and intrinsics the raw sensor colors and sensor depth for view {}".format(i))
            image, _ = self.load_image_data(image_id=image_id)
            depth, _, _, _ = self.load_depth_data(image_id=image_id)
            pose = self.models['pose']()([0])[i].to(device=self.device)
            focal_lengths = self.models['focal']()(0)[image_id].expand(self.H*self.W)            
            pixel_directions = compute_pixel_directions(focal_lengths, self.pixel_rows_for_test_renders, self.pixel_cols_for_test_renders, self.H, self.W)
            self.generate_point_cloud(pose=pose, depth=depth, rgb=image, pixel_directions=pixel_directions, label="raw_sensor_with_learned_poses_intrinsics_{}".format(i), save=True, save_raw_xyz=True)               


    #########################################################################
    ############################# Sampling ##################################
    #########################################################################

    # note: currently a minor bug that allows far samples to be zero if the region is too small
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
        n_samples_near = torch.floor(torch.tensor(n_samples * percentile_of_samples_in_near_region)) + 1
        n_samples_far = n_samples - n_samples_near + 1                
        bins = torch.linspace(near_min_focus, far_min_focus, int(n_samples_near)).to(self.device)        

        # compute sample distance for the far region, where the far min is equal to the near max
        far_focus_base = (far_max_focus/far_min_focus)**(1/n_samples_far)
        far_sample_numbers = torch.arange(start=0, end=n_samples_far).to(self.device)
        far_bins = far_min_focus * far_focus_base ** far_sample_numbers                
        bins = torch.cat([bins, far_bins[1:] ]).to(self.device) 

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

        #print('render')
        #a = torch.sum(  (pixel_directions - torch.nn.functional.normalize(pixel_directions, p=2, dim=1))**2, dim = 1)
        #a = torch.sqrt(a)
        #print(a.max())             
        #print(a.mean())
        #print("______")
        
        pixel_directions = torch.nn.functional.normalize(pixel_directions, p=2, dim=1)

        pixel_xyz_positions, pixel_directions_world, resampled_depths = volume_sampling(poses=poses, pixel_directions=pixel_directions, sampling_depths=sampling_depths, perturb_depths=perturb_depths)
        pixel_directions_world = torch.nn.functional.normalize(pixel_directions_world, p=2, dim=1)  # (N_pixels, 3)
        
        # compute pixel world width
        pixels_shifted_x = (pixel_directions[:, 0] + (1.0 / pixel_focal_lengths)) 
        pixels_shifted_y = pixel_directions[:, 1]
        pixels_shifted_z = pixel_directions[:, 2]
        
        neighbor_pixel_directions = torch.stack([pixels_shifted_x, pixels_shifted_y, pixels_shifted_z], dim=-1)        
        neighbor_pixel_directions = neighbor_pixel_directions / torch.sqrt(torch.sum(neighbor_pixel_directions ** 2,dim=1)).unsqueeze(1).to(torch.device('cuda:0'))
        
        pixel_directions_neighbor_distances = torch.sqrt(torch.sum( (pixel_directions - neighbor_pixel_directions)**2,dim=1)).to(torch.device('cuda:0'))
        pixel_directions_neighbor_distances = pixel_directions_neighbor_distances.unsqueeze(1).expand(sampling_depths.size(0), sampling_depths.size(1)-1).to(torch.device('cuda:0'))

        pixel_world_widths = pixel_directions_neighbor_distances

        xyz_position_encoding = encode_ipe(poses[:, :3, 3], pixel_xyz_positions, pixel_directions_world, sampling_depths, pixel_world_widths)
        xyz_position_encoding = xyz_position_encoding.to(self.device)
                            
        # encode direction: (H, W, N_sample, (2L+1)*C = 27)        
        angular_directional_encoding = encode_position(pixel_directions_world, levels=self.args.directional_encoding_fourier_frequencies)  # (N_pixels, 27)
        angular_directional_encoding = angular_directional_encoding.unsqueeze(1).expand(-1, sampling_depths.size()[1] - 1, -1)  # (N_pixels, N_sample, 27)                    

        density, features = self.models["geometry"]()([xyz_position_encoding]) # (N_pixels, N_sample, 1), # (N_pixels, N_sample, D)                                        

        rgb = self.models["color"]()([features, angular_directional_encoding])  # (N_pixels, N_sample, 4)        
        
        #result = make_dot(self.models["color"]()([features, x]), params=dict(list(self.models["color"].named_parameters())))
        #result.render('graph', format='png')
                
        render_result = volume_rendering(rgb, density, resampled_depths[:, : -1])          

        depth_map = render_result['depth_map']
        rgb_rendered = render_result['rgb_rendered']

        result = {
            'rgb_rendered': rgb_rendered, # (N_pixels, 3)
            #'pixel_xyz_positions': pixel_xyz_positions,    # (N_pixels, N_sample, 3)
            'depth_map': depth_map,       # (N_pixels)
            'depth_weights': render_result['weight'],      # (N_pixels, N_sample),
            #'rgb': render_result['rgb'],                   # (N_pixels, N_sample, 3),
            'density': render_result['density'],                            # (N_pixels, N_sample),
            #'alpha': render_result['alpha'],               # (N_pixels, N_sample),
            #'acc_transmittance': render_result['acc_transmittance'], # (N_pixels, N_sample),
            #'resampled_depths': resampled_depths,           # (N_samples)
            'distances': render_result['distances'],
        }

        return result


    # invoke current model for input poses/focal lengths
    # for visual results, supply result to save_render_as_png
    # pose : (N, 4, 4)
    # focal_lengths: (N, 1)
    def render_prediction(self, poses, focal_lengths, principal_point_x, principal_point_y):
        
        pixel_directions = compute_pixel_directions(
            focal_lengths, 
            self.pixel_rows_for_test_renders, 
            self.pixel_cols_for_test_renders, 
            principal_point_x=principal_point_x, 
            principal_point_y=principal_point_y
        )

        # batch the data
        poses_batches = poses.split(self.args.number_of_pixels_per_batch_in_test_renders)
        pixel_directions_batches = pixel_directions.split(self.args.number_of_pixels_per_batch_in_test_renders)
        focal_lengths_batches = focal_lengths.split(self.args.number_of_pixels_per_batch_in_test_renders)         

        rendered_image_fine_batches = []
        depth_image_fine_batches = []

        rendered_image_unsparse_fine_batches = []
        depth_image_unsparse_fine_batches = []        

        depth_weights_coarse_batches = []
        
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
                    rendered_data_coarse = self.render(poses=poses_batch, pixel_directions=pixel_directions_batch, sampling_depths=depth_samples_coarse, pixel_focal_lengths=focal_lengths_batch, perturb_depths=False)  # (N_pixels, 3)
                else:
                    # if this is not the first iteration, then resample with the latest weights         
                    depth_samples_fine = self.resample_depths_from_nerf_weights(weights=rendered_data_coarse['depth_weights'], depth_samples=depth_samples_coarse, use_sparse_fine_rendering=self.args.use_sparse_fine_rendering, test_render=True)  # (N_pixels, N_samples)
                    rendered_data_fine = self.render(poses=poses_batch, pixel_directions=pixel_directions_batch, sampling_depths=depth_samples_fine, pixel_focal_lengths=focal_lengths_batch, perturb_depths=False)  # (N_pixels, 3)
                    
                    if self.args.use_sparse_fine_rendering:
                        # point cloud filtering needs both the sparsely rendered data the "unsparsely" (normal) rendered data
                        depth_samples_unsparse_fine = self.resample_depths_from_nerf_weights(weights=rendered_data_coarse['depth_weights'], depth_samples=depth_samples_coarse, use_sparse_fine_rendering=False, test_render=True)  # (N_pixels, N_samples)                                    
                        unsparse_rendered_data_fine = self.render(poses=poses_batch, pixel_directions=pixel_directions_batch, sampling_depths=depth_samples_unsparse_fine, pixel_focal_lengths=focal_lengths_batch, perturb_depths=False)  # (N_pixels, 3)                        

                    
            rendered_image_fine_batches.append(rendered_data_fine['rgb_rendered'].cpu()) # (n_pixels_per_row, 3)                
            depth_image_fine_batches.append(rendered_data_fine['depth_map'].cpu()) # (n_pixels_per_row)             
                                
            if self.args.use_sparse_fine_rendering:
                rendered_image_unsparse_fine_batches.append(unsparse_rendered_data_fine['rgb_rendered'].cpu())
                depth_image_unsparse_fine_batches.append(unsparse_rendered_data_fine['depth_map'].cpu())

            depth_weights_coarse_batches.append(rendered_data_coarse['depth_weights'].cpu())  
        

        # combine batch results to compose full images                
        rendered_image_data_fine = torch.cat(rendered_image_fine_batches, dim=0).cpu() # (N_pixels, 3)            
        rendered_depth_data_fine = torch.cat(depth_image_fine_batches, dim=0).cpu()  # (N_pixels)                
        
        if self.args.use_sparse_fine_rendering:
            rendered_image_data_unsparse_fine = torch.cat(rendered_image_unsparse_fine_batches, dim=0).cpu()
            depth_image_data_unsparse_fine = torch.cat(depth_image_unsparse_fine_batches, dim=0).cpu()
                    
        depth_weights_data_coarse = torch.cat(depth_weights_coarse_batches, dim=0).cpu()  # (N_pixels, N_samples)

        render_result = {
             'rendered_image_fine': rendered_image_data_fine,
             'rendered_depth_fine': rendered_depth_data_fine,
             'depth_weights_coarse': depth_weights_data_coarse,
        }
    
        if self.args.use_sparse_fine_rendering:
            render_result['rendered_image_unsparse_fine'] = rendered_image_data_unsparse_fine
            render_result['depth_image_unsparse_fine'] = depth_image_data_unsparse_fine

        return render_result    


    # process raw rendered pixel data and save into images
    def save_render_as_png(self, render_result, H, W, color_file_name_fine, depth_file_name_fine, color_file_name_coarse=None, depth_file_name_coarse=None):
                
        rendered_rgb_fine = render_result['rendered_image_fine'].reshape(H, W, 3)
        rendered_depth_fine = render_result['rendered_depth_fine'].reshape(H, W)
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
        
        a = rendered_depth_fine.flatten(start_dim=0, end_dim=1)
        a[torch.where(a > 0.5)[0]] =  0.8
        rendered_depth_fine = a.reshape(H, W)
        rendered_depth_data_fine = rendered_depth_fine.cpu().numpy() 
        
        #rendered_depth_data_fine[torch.where(rendered_depth_data_fine)]
        
        
        rendered_depth_for_file_fine = heatmap_to_pseudo_color(rendered_depth_data_fine)
        rendered_depth_for_file_fine = (rendered_depth_for_file_fine * 255).astype(np.uint8)        
        imageio.imwrite(color_file_name_fine, rendered_color_for_file_fine)
        imageio.imwrite(depth_file_name_fine, rendered_depth_for_file_fine)         


    #########################################################################
    ############################## Training #################################
    #########################################################################

    def train(self, indices_of_random_pixels):
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
        n_pixels = indices_of_random_pixels.size()[0]
                
        # ignore depth loss for low-confidence pixels
        selected_confidences = self.confidence_per_pixel[indices_of_random_pixels].to(self.device)
        confidence_loss_weights = torch.where(selected_confidences >= self.args.min_confidence, 1, 0).to(self.device)
        number_of_pixels_with_confident_depths = torch.sum(confidence_loss_weights)

        # don't try to fit sensor depths that are fixed to max value
        ignore_max_sensor_depths = torch.where(sensor_depth < self.args.far_maximum_depth, 1, 0).to(self.device)

        # initialize our total weighted loss, which will be computed as the weighted sum of coarse and fine losses
        total_weighted_loss = torch.tensor(0.0).to(self.device)
        
        focal_length = self.models['focal']()([0])                        
        poses = self.models['pose']()([0])
                
        # get a tensor with the poses per pixel
        image_ids = self.image_ids_per_pixel[indices_of_random_pixels] # (N_pixels)                
        selected_poses = poses[image_ids] # (N_pixels, 4, 4)

        # get the focal lengths and pixel directions for every pixel given the images that were actually selected for each pixel
        selected_focal_lengths = focal_length[image_ids]
        
        pixel_directions_selected = compute_pixel_directions(selected_focal_lengths, pixel_rows, pixel_cols, self.principal_point_x, self.principal_point_y)

        
        for depth_sampling_optimization in range(self.args.n_depth_sampling_optimizations):
            
            #####################| Sampling & Rendering |##################
            if depth_sampling_optimization == 0:
                # if this is the first iteration, collect linear depth samples to query NeRF, uniformly in space
                depth_samples = self.sample_depths_near_linearly_far_nonlinearly(number_of_pixels=n_pixels, add_noise=True) # (N_pixels, N_samples)
            else:
                # if this is not the first iteration, then resample with the latest weights
                depth_samples = self.resample_depths_from_nerf_weights(weights=nerf_depth_weights, depth_samples=depth_samples)  # (N_pixels, N_samples)                


            n_samples = depth_samples.size()[1]

            # render an image using selected rays, pose, sample intervals, and the network            
            render_result = self.render(poses=selected_poses, pixel_directions=pixel_directions_selected, sampling_depths=depth_samples, pixel_focal_lengths=selected_focal_lengths, perturb_depths=False)  # (N_pixels, 3)    
            
            rgb_rendered = render_result['rgb_rendered']         # (N_pixels, 3)                                                

            nerf_depth_weights = render_result['depth_weights']  # (N_pixels, N_samples)
            
            nerf_sample_bin_lengths = depth_samples[:, 1:] - depth_samples[:, :-1]
            nerf_depth_weights = nerf_depth_weights + 1e-7

            #####################| KL Loss |################################
            sensor_variance = self.args.depth_sensor_error
            kl_divergence_bins = -1 * torch.log2(nerf_depth_weights) * torch.exp(-1 * (depth_samples[:, : n_samples - 1] * 1000 - sensor_depth_per_sample[:, : n_samples - 1] * 1000) ** 2 / (2 * sensor_variance)) * nerf_sample_bin_lengths * 1000                                
            confidence_weighted_kl_divergence_pixels = ignore_max_sensor_depths * confidence_loss_weights * torch.sum(kl_divergence_bins, 1) # (N_pixels)
            depth_loss = torch.sum(confidence_weighted_kl_divergence_pixels) / number_of_pixels_with_confident_depths
            depth_to_rgb_importance = self.get_polynomial_decay(start_value=self.args.depth_to_rgb_loss_start, end_value=self.args.depth_to_rgb_loss_end, exponential_index=self.args.depth_to_rgb_loss_exponential_index, curvature_shape=self.args.depth_to_rgb_loss_curvature_shape)
            
            #####################| Entropy Loss |###########################
            entropy_depth_loss = 0.0
            n_near_samples = int(self.args.percentile_of_samples_in_near_region * self.args.number_of_samples_outward_per_raycast)
            mean_entropy = torch.mean(-1 * torch.sum(nerf_depth_weights * torch.log2(nerf_depth_weights), dim=1))
            if (self.epoch >= self.args.entropy_loss_tuning_start_epoch and self.epoch <= self.args.entropy_loss_tuning_end_epoch):                                                
                epoch = torch.tensor([self.epoch]).float().to(self.device).float()
                entropy_loss_weight = (torch.tanh(0.01 * epoch / 1000.0) / (1.0 / self.args.max_entropy_weight)).item()
                entropy_depth_loss = entropy_loss_weight * mean_entropy                
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
                # note: KL divergence loss is not used for fine iteration even though it's computed above                
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
              
        print("({} at {:.2f} min) - LOSS = {:.5f} -> RGB: C: {:.6f} ({:.3f}/255) | F: {:.6f} ({:.3f}/255), DEPTH: C: {:.6f} ({:.2f}mm) | F: {:.6f} ({:.2f}mm) w/ imp. {:.5f}, Entropy: {:.6f} (loss: {:.6f})".format(
            self.epoch, 
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
                                    
            print('Avg (sensor) depth in batch: ', torch.mean(self.rgbd[indices_of_random_pixels_for_this_epoch, 3] ))            

            image_ids = self.image_ids_per_pixel[indices_of_random_pixels_for_this_epoch]
            pixel_indices_for_test_image = indices_of_random_pixels_for_this_epoch[torch.where(image_ids == 0)]
            pixel_rows = self.pixel_rows[pixel_indices_for_test_image]
            pixel_cols = self.pixel_cols[pixel_indices_for_test_image]
            pixel_coordinates = pixel_rows * self.W + pixel_cols
            hist = torch.histc( pixel_coordinates.float(), bins=self.W*self.H, min=0, max=self.W*self.H ).float()
            hist = hist.reshape(self.H, self.W)     
            self.sample_image = self.sample_image + hist                                           
        else:            
            indices_of_random_pixels_for_this_epoch = random.sample(population=self.pixel_indices_below_max_depth, k=self.args.pixel_samples_per_epoch)            

        return indices_of_random_pixels_for_this_epoch
            
    


    #########################################################################
    ############################# Experiments ###############################
    #########################################################################

    def create_experiment_directory(self):
        data_out_dir = "{}/hyperparam_experiments".format(self.args.base_directory)            
                
        experiment_label = "{}".format( int(str(self.start_time)[:9]) )            
        experiment_dir = Path(os.path.join(data_out_dir, experiment_label))
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

        H = self.args.H_for_test_renders
        W = self.args.W_for_test_renders        
        
        all_trained_focal_lengths = self.models["focal"]()([0])

        test_image_indices = self.test_image_indices        

        for image_index in test_image_indices:
                                          
            pp_x = self.principal_point_x * (float(W) / float(self.W))
            pp_y = self.principal_point_y * (float(H) / float(self.H))

            # always render              
            print("Rendering for image {}".format(image_index))            
                     
            focal_lengths_for_this_img = all_trained_focal_lengths[image_index].expand(int(H*W)) * (float(H) / float(self.H))

            poses_for_this_img = self.models['pose']()([0])[image_index].unsqueeze(0).expand(W*H, -1, -1)
                            
            render_result = self.render_prediction(poses_for_this_img, focal_lengths_for_this_img, pp_x, pp_y)
                                
            # save rendered rgb and depth images
            out_file_suffix = str(image_index)
            color_file_name_fine = os.path.join(self.color_out_dir, str(out_file_suffix).zfill(4) + '_color_fine_{}.png'.format(epoch))
            depth_file_name_fine = os.path.join(self.depth_out_dir, str(out_file_suffix).zfill(4) + '_depth_fine_{}.png'.format(epoch))            
            if self.args.use_sparse_fine_rendering:
                color_file_name_fine = os.path.join(self.color_out_dir, str(out_file_suffix).zfill(4) + 'sparse_color_fine_{}.png'.format(epoch))
                depth_file_name_fine = os.path.join(self.depth_out_dir, str(out_file_suffix).zfill(4) + 'sparse_depth_fine_{}.png'.format(epoch))                            

            
            self.save_render_as_png(render_result, H, W, color_file_name_fine, depth_file_name_fine)            
                                
            # save graphs of learning rate histories
            for topic in ["color", "geometry", "pose", "focal"]:                                            
                lrs = self.learning_rate_histories[topic]
                if len(lrs) > 0:
                    plt.figure()
                    plt.plot([x for x in range(len(lrs))], lrs)
                    plt.ylim(0.0, max(lrs)*1.1)
                    plt.savefig('{}/{}.png'.format(self.learning_rates_out_dir, topic))
                    plt.close()


            # optional: save graphs of nerf density weights visualization
            #if epoch % self.args.save_depth_weights_frequency == 0 and epoch != 0:               
                #print("Creating depth weight graph for view {}".format(image_index))
                #os.makesdirs('{}/depth_weights_visualization'.format(self.experiment_dir), exist_ok=True)
                #Path("{}/depth_weights_visualization/{}/image_{}/".format(self.experiment_dir, epoch, image_index)).mkdir(parents=True, exist_ok=True)
                #view_row = 240                
                #pixel_indices = torch.argwhere(self.image_ids_per_pixel == image_index)
                #pixel_rows = self.pixel_rows[pixel_indices]
                #pixel_cols = self.pixel_cols[pixel_indices]                
#
                #start_pixel = torch.sum(torch.where(pixel_rows.squeeze(1) < view_row, 1, 0))
                #end_pixel = torch.sum(torch.where(pixel_rows.squeeze(1) < view_row + 1, 1, 0))
                #this_image_rgbd = self.rgbd[pixel_indices].cpu().squeeze(1)  
                #ground_truth_image = self.load_image_data(image_index * self.skip_every_n_images_for_training)
                #for pixel_index in range(start_pixel, end_pixel):                                                
#
                    #out_path = Path("{}/{}/".format(self.depth_weights_out_dir, epoch))
                    #out_path.mkdir(parents=True, exist_ok=True)
#
                    #sensor_depth = this_image_rgbd[pixel_index, 3]                    
#
                    #nerf_weights_coarse = render_result['depth_weights_coarse']
#                    
                    ##weights_fine = nerf_weights_fine.squeeze(0)[pixel_index]                                                                           
#
                    #image = ground_truth_image
                    #downsized_image = torch.nn.functional.interpolate(image.permute(2, 0, 1).unsqueeze(0), size=(int(image.shape[0]/4), int(image.shape[1]/4)), mode='bilinear').squeeze().permute(1, 2, 0)                    
                    #pixel_row_to_highlight = pixel_rows[pixel_index][0]                                    
                    #pixel_col_to_highlight = pixel_cols[pixel_index][0]                    
                    #pixel_row_to_highlight = int(pixel_row_to_highlight / 4)
                    #pixel_col_to_highlight = int(pixel_col_to_highlight / 4)
                    #downsized_image[pixel_row_to_highlight, pixel_col_to_highlight, :] = torch.tensor([1.0,0.0,0.0], dtype=torch.float32)     
                    #downsized_image = (downsized_image * 255).to(torch.long)
                    #downsized_image = downsized_image.cpu().numpy().astype(np.uint8)
                    #downsized_path = Path("{}/downsized_image_{}.png".format(self.experiment_dir, image_index))
                    #imageio.imwrite(downsized_path, downsized_image)                    
#
                    ##densities_coarse = density_coarse.squeeze(0)[pixel_index]  
                    #weights_coarse = nerf_weights_coarse.squeeze(0)[pixel_index]                   
                    ##weights_fine = nerf_weights_fine.squeeze(0)[pixel_index]
                    ##predicted_depth_coarse = depth_coarse[pixel_index]
                    #coarse_weights_entropy = -1 * torch.sum( (weights_coarse+self.args.epsilon) * torch.log(weights_coarse + self.args.epsilon), dim=0)
                    #fig, axs = plt.subplots(3,2)                    
#
                    #max_depth = 0.5
                    #avg_fine_sample_depth = torch.mean(sampled_depths_fine[pixel_index, : sampled_depths_fine.size()[1]-1])
                    #if self.args.n_depth_sampling_optimizations > 1:
                        #predicted_depth_fine = depth_fine[pixel_index]                    
                        #axs[1,0].scatter(sampled_depths_fine[pixel_index, : sampled_depths_fine.size()[1]-1], weights_fine, s=1, marker='o', c='green')                    
                        #axs[1,0].scatter([sensor_depth.item()], [0], s=30, marker='o', c='red')                    
                        #axs[1,0].scatter([predicted_depth_fine.item()], [0], s=5, marker='o', c='blue')
                        ##axs[1,0].set_xlim([avg_fine_sample_depth - offset, avg_fine_sample_depth + offset])
                        #axs[1,0].set_xlim([0,max_depth])
#                                        
                    #axs[0,0].scatter(sampled_depths_coarse[pixel_index, : sampled_depths_coarse.size()[1]-1], weights_coarse, s=1, marker='o', c='green')                    
                    #axs[0,0].scatter([sensor_depth.item()], [0], s=30, marker='o', c='red')                    
                    #axs[0,0].scatter([predicted_depth_coarse.item()], [0], s=5, marker='o', c='blue')                    
                    ##axs[0,0].set_xlim([avg_fine_sample_depth - offset, avg_fine_sample_depth + offset])
                    #axs[0,0].set_xlim([0,max_depth])
                    #if self.args.n_depth_sampling_optimizations > 1:                                            
                        #axs[1,1].scatter(sampled_depths_fine[pixel_index, : sampled_depths_fine.size()[1]-1], densities_fine, s=1, marker='o', c='green')                    
                        #axs[1,1].scatter([sensor_depth.item()], [0], s=30, marker='o', c='red')                    
                        #axs[1,1].scatter([predicted_depth_fine.item()], [0], s=5, marker='o', c='blue')
                        ##axs[1,1].set_xlim([avg_fine_sample_depth - offset, avg_fine_sample_depth + offset])
                        #axs[1,1].set_xlim([0,max_depth])
#                    
                    #axs[0,1].scatter(sampled_depths_coarse[pixel_index, : sampled_depths_coarse.size()[1]-1], densities_coarse, s=1, marker='o', c='green')                    
                    #axs[0,1].scatter([sensor_depth.item()], [0], s=30, marker='o', c='red')                    
                    #axs[0,1].scatter([predicted_depth_coarse.item()], [0], s=5, marker='o', c='blue')                    
                    ##axs[0,1].set_xlim([avg_fine_sample_depth - offset, avg_fine_sample_depth + offset])
                    #axs[0,1].set_xlim([0,max_depth])
#
                    #coarse_weights_sum = torch.sum(weights_coarse, 0)
                    #fine_weights_sum = torch.sum(weights_fine, 0)
                    #text_kwargs = dict(ha='center', va='center', fontsize=8, color='black')   
#
                    #depth_diff = 1337.69420
                    #if self.args.use_sparse_fine_rendering:
                        #predicted_depth_unsparse_fine = (render_result['depth_image_unsparse_fine'])[pixel_index]
                        #depth_diff = torch.abs(predicted_depth_unsparse_fine - predicted_depth_fine)
#                    
                    #axs[2,0].text(0.5,0.5, 'coarse weights entropy: {:.6f}\npixel coordinates: ({},{})\ncoarse weights sum: {:.6f}\nfine weights sum: {:.6f}\nsparse/unsparse depth diff: {:.6f}'.format(coarse_weights_entropy, self.pixel_rows[pixel_indices[pixel_index]][0], self.pixel_cols[pixel_indices[pixel_index]][0], coarse_weights_sum, fine_weights_sum, depth_diff) ,  **text_kwargs)
                    #axs[2,0].axis('off')
#
                    #axs[2,1].imshow(downsized_image)
                    #axs[2,1].axis('off')
#                    
                    #out_file_suffix = 'image_{}/{}'.format(image_index, image_index)
                    #depth_weights_file_name = os.path.join(out_path, str(out_file_suffix).zfill(4) + '_depth_weights_e{}_p{}.png'.format(epoch, pixel_index))
                    #plt.savefig(depth_weights_file_name)
                    #plt.close()




            
        self.export_camera_extrinsics_and_intrinsics_from_trained_model(self.experiment_dir, epoch)

        torch.cuda.empty_cache()


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
    ########################## Parameters ############################
    ##################################################################        

    def load_saved_args_train(self):
            
        self.args.base_directory = './data/cactus'
        self.args.images_directory = 'color'
        self.args.images_data_type = 'jpg'            
        self.args.load_pretrained_models = True
        self.args.reset_learning_rates = False
        #self.args.pretrained_models_directory = './data/dragon_scale/hyperparam_experiments/from_cloud/dragon_scale_run39/models'
        self.args.pretrained_models_directory = './data/cactus/hyperparam_experiments/from_cloud/cactus_run204-2/models'
        self.args.start_epoch = 500001
        self.args.number_of_epochs = 500000
        
        #self.args.start_training_extrinsics_epoch = 500        
        #self.args.start_training_intrinsics_epoch = 5000
        self.args.start_training_extrinsics_epoch = 500
        self.args.start_training_intrinsics_epoch = 1000
        self.args.start_training_color_epoch = 0
        self.args.start_training_geometry_epoch = 0
        self.args.entropy_loss_tuning_start_epoch = 10000
        self.args.entropy_loss_tuning_end_epoch = 1000000
        self.args.max_entropy_weight = 0.002

        self.args.nerf_density_lr_start = 0.0005
        self.args.nerf_density_lr_end = 0.0001
        self.args.nerf_density_lr_exponential_index = 4
        self.args.nerf_density_lr_curvature_shape = 1

        self.args.nerf_color_lr_start = 0.0005
        self.args.nerf_color_lr_end = 0.0001
        self.args.nerf_color_lr_exponential_index = 2
        self.args.nerf_color_lr_curvature_shape = 1

        self.args.focal_lr_start = 0.0001        
        self.args.focal_lr_end = 0.0000025
        self.args.focal_lr_exponential_index = 1
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
        
        self.args.depth_sensor_error = 0.5
        self.args.min_confidence = 2.0

        ### test images
        self.args.skip_every_n_images_for_testing = 50
        self.args.number_of_test_images = 5

        ### test frequency parameters
        self.args.test_frequency = 5000
        self.args.export_test_data_for_testing = False    
        self.args.save_point_cloud_frequency = 1000000
        self.args.save_depth_weights_frequency = 5000000000
        self.args.log_frequency = 1
        self.args.save_models_frequency = 5000
        
        # training
        self.args.resample_pixels_frequency = 5000
        self.args.pixel_samples_per_epoch = 1024
        ###########################self.args.number_of_samples_outward_per_raycast = 360
        self.args.number_of_samples_outward_per_raycast = 360
        #self.args.number_of_images_in_training_dataset = 2048
        self.args.number_of_images_in_training_dataset = 120
        self.args.number_of_pixels_in_training_dataset = 640 * 480 * 256
        self.args.H_for_training = 480
        self.args.W_for_training = 640
        #self.args.H_for_training = 1440 # principal points and focal lengths will be rescaled to account for this
        #self.args.W_for_training = 1920
        

        # testing
        self.args.number_of_pixels_per_batch_in_test_renders = 2500
        self.args.number_of_samples_outward_per_raycast_for_test_renders = self.args.number_of_samples_outward_per_raycast
        
        self.args.use_sparse_fine_rendering = False        
                
        #self.args.near_maximum_depth = 0.5
        self.args.near_maximum_depth = 1.0
        self.args.far_maximum_depth = 3.00  
        self.args.percentile_of_samples_in_near_region = 0.80 

        #self.args.H_for_test_renders = 1440
        #self.args.W_for_test_renders = 1920        
        self.args.H_for_test_renders = 480 # principal points and focal lengths created for training will be rescaled for test renders
        self.args.W_for_test_renders = 640


    ##################################################################
    ##################### Main function ##############################
    ##################################################################
if __name__ == '__main__':
    
    with torch.no_grad():
        scene = SceneModel(args=parse_args(), experiment_args='train')

    while scene.epoch < scene.args.start_epoch + scene.args.number_of_epochs:    
        
        if scene.epoch == scene.args.start_epoch:
            with torch.no_grad():                                            
                scene.test()
                quit()
                print("")
                
        #scene.print_memory_usage()
        
        with torch.no_grad():
            if scene.epoch != 1 and (scene.epoch-1) % scene.args.resample_pixels_frequency == 0:

                if scene.sample_image.sum() > 0:
                    sample_image = scene.sample_image
                    sample_image = sample_image / sample_image.max()
                    sample_image = heatmap_to_pseudo_color(sample_image.cpu().detach().numpy(), min_val=0.0, max_val=1.0)
                    sample_image = (sample_image * 255).astype(np.uint8)
                    sample_image_path = "{}/sampling_data/sample_image_{}.png".format(scene.experiment_dir, scene.epoch-1)
                    imageio.imwrite(sample_image_path, sample_image)

                                        
                    #s = torch.cat(scene.sample_distances, dim=1)
                    #s = torch.sort(s, dim=0)[0]

                    #hist = torch.histc(s, bins=1000, min=0, max=4.0)

                    #total_samples = s.size()[0]
                    #cdf = []
                    #for i in range (0, total_samples):
                    #    cdf.append( (torch.sum(s[:i]) / total_samples).cpu().numpy()) 
                    #print(cdf)
                    #plt.figure()
                    #plt.subplot(211)

                    #plt.scatter(s.cpu().numpy(), np.zeros((total_samples, s.size()[1])), s=1)
                    #plt.plot(np.linspace(0.0, 4.0, num=10), cdf)
                    #plt.subplot(212)
                    #plt.bar(np.linspace(0.0, 4.0, num=1000), hist.cpu(), align='center', width=0.02)
                    #plt.suptitle(scene.args.base_directory.split('/')[-1])
                    #plt.show()                    

                print('Resampling training data...')
                scene.sample_training_data(visualize_sampled_pixels=False)

            batch = scene.sample_next_batch(weighted=True)
        
        
        #with torch.autograd.detect_anomaly():  # no need to enable this because there are never errors
        # note: detect_anomaly is incompatible with Inductor
        scene.train(batch) 
       
        if (scene.epoch-1) % scene.args.save_models_frequency == 0 and (scene.epoch-1) !=  scene.args.start_epoch:
            scene.save_models()

        if (scene.epoch-1) % scene.args.test_frequency == 0 and (scene.epoch-1) != 0:
            with torch.no_grad():                
                scene.test()