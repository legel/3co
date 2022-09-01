import torch
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix, matrix_to_euler_angles, matrix_to_quaternion, axis_angle_to_quaternion, quaternion_multiply, matrix_to_axis_angle
from scipy.spatial.transform import Rotation
from torchsummary import summary
import cv2
import open3d as o3d
from pytorch3d.ops.knn import knn_points
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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

set_randomness()

def parse_args():
    parser = argparse.ArgumentParser()

    # Define path to relevant data for training, and decide on number of images to use in training
    parser.add_argument('--base_directory', type=str, default='./data/dragon_scale', help='The base directory to load and save information from')
    parser.add_argument('--images_directory', type=str, default='color', help='The specific group of images to use during training')
    parser.add_argument('--images_data_type', type=str, default='jpg', help='Whether images are jpg or png')
    parser.add_argument('--skip_every_n_images_for_training', type=int, default=60, help='When loading all of the training data, ignore every N images')
    parser.add_argument('--save_models_frequency', type=int, default=50000, help='Save model every this number of epochs')
    parser.add_argument('--load_pretrained_models', type=bool, default=False, help='Whether to start training from models loaded with load_pretrained_models()')

    # Define number of epochs, and timing by epoch for when to start training per network
    parser.add_argument('--number_of_epochs', default=100001, type=int, help='Number of epochs for training, used in learning rate schedules')
    parser.add_argument('--early_termination_epoch', default=100001, type=int, help='kill training early at this epoch (even if learning schedule not finished')
    parser.add_argument('--start_training_extrinsics_epoch', type=int, default=500, help='Set to epoch number >= 0 to init poses using estimates from iOS, and start refining them from this epoch.')
    parser.add_argument('--start_training_intrinsics_epoch', type=int, default=5000, help='Set to epoch number >= 0 to init focals using estimates from iOS, and start refining them from this epoch.')
    parser.add_argument('--start_training_color_epoch', type=int, default=0, help='Set to a epoch number >= 0 to start learning RGB NeRF on top of density NeRF.')
    parser.add_argument('--start_training_geometry_epoch', type=int, default=0, help='Set to a epoch number >= 0 to start learning RGB NeRF on top of density NeRF.')

    # Define evaluation/logging/saving frequency and parameters
    parser.add_argument('--test_frequency', default=2500, type=int, help='Frequency of epochs to render an evaluation image')
    parser.add_argument('--visualize_point_cloud_frequency', default=200001, type=int, help='Frequency of epochs to visualize point clouds')
    parser.add_argument('--save_point_cloud_frequency', default=20000, type=int, help='Frequency of epochs to save point clouds')
    parser.add_argument('--log_frequency', default=1, type=int, help='Frequency of epochs to log outputs e.g. loss performance')
    parser.add_argument('--render_test_video_frequency', default=50000, type=int, help='Frequency of epochs to log outputs e.g. loss performance')
    parser.add_argument('--spherical_radius_of_test_video', default=1, type=int, help='Radius of sampled poses around the evaluation pose for video')
    parser.add_argument('--number_of_poses_in_test_video', default=10, type=int, help='Number of poses in test video to render for the total animation')
    parser.add_argument('--number_of_test_images', default=2, type=int, help='Index in the training data set of the image to show during testing')
    parser.add_argument('--skip_every_n_images_for_testing', default=80, type=int, help='Skip every Nth testing image, to ensure sufficient test view diversity in large data set')    
    parser.add_argument('--number_of_pixels_per_batch_in_test_renders', default=128, type=int, help='Size in pixels of each batch input to rendering')
    parser.add_argument('--show_debug_visualization_in_testing', default=False, type=bool, help='Whether or not to show the cool Matplotlib 3D view of rays + weights + colors')
    parser.add_argument('--export_test_data_for_post_processing', default=False, type=bool, help='Whether to save in external files the final render RGB + weights for all samples for all images')
    parser.add_argument('--save_ply_point_clouds_of_sensor_data', default=True, type=bool, help='Whether to save a .ply file at start of training showing the initial projected sensor data in 3D global coordinates')
    parser.add_argument('--save_ply_point_clouds_of_sensor_data_with_learned_poses', default=True, type=bool, help='Whether to save a .ply file after loading a pre-trained model to see how the sensor data projects to 3D global coordinates with better poses')
    parser.add_argument('--recompute_sensor_variance_from_initial_data', default=True, type=bool, help='If True, then it starts the optimization by computing and saving files representing estimate of sensor error, based on the KNN distance of each 3D point; if False, looks to load previous computation from saved file')
    parser.add_argument('--number_of_nearest_neighbors_to_use_in_knn_distance_metric_for_estimation_of_sensor_error', default=10, type=int, help='N for the KNN on every 3D point at start of optimization, of which distances for N points are used as metric of variance')

    # Define learning rates, including start, stop, and two parameters to control curvature shape (https://arxiv.org/pdf/2004.05909v1.pdf)
    parser.add_argument('--nerf_density_lr_start', default=0.0010, type=float, help="Learning rate start for NeRF geometry network")
    parser.add_argument('--nerf_density_lr_end', default=0.0001, type=float, help="Learning rate end for NeRF geometry network")
    parser.add_argument('--nerf_density_lr_exponential_index', default=6, type=int, help="Learning rate speed of exponential decay (higher value = faster initial decay) for NeRF geometry network")
    parser.add_argument('--nerf_density_lr_curvature_shape', default=1, type=int, help="Learning rate shape of decay (lower value = faster initial decay) for NeRF geometry network")

    parser.add_argument('--nerf_color_lr_start', default=0.001, type=float, help="Learning rate start for NeRF RGB (pitch,yaw) network")
    parser.add_argument('--nerf_color_lr_end', default=0.0001, type=float, help="Learning rate end for NeRF RGB (pitch,yaw) network")
    parser.add_argument('--nerf_color_lr_exponential_index', default=4, type=int, help="Learning rate speed of exponential decay (higher value = faster initial decay) for NeRF RGB (pitch,yaw) network")
    parser.add_argument('--nerf_color_lr_curvature_shape', default=1, type=int, help="Learning rate shape of decay (lower value = faster initial decay) for NeRF RGB (pitch,yaw) network")

    parser.add_argument('--focal_lr_start', default=0.00100, type=float, help="Learning rate start for NeRF-- camera intrinsics network")
    parser.add_argument('--focal_lr_end', default=0.0001, type=float, help="Learning rate end for NeRF-- camera intrinsics network")
    parser.add_argument('--focal_lr_exponential_index', default=9, type=int, help="Learning rate speed of exponential decay (higher value = faster initial decay) for NeRF-- camera intrinsics network")
    parser.add_argument('--focal_lr_curvature_shape', default=1, type=int, help="Learning rate shape of decay (lower value = faster initial decay) for NeRF-- camera intrinsics network")

    parser.add_argument('--pose_lr_start', default=0.00100, type=float, help="Learning rate start for NeRF-- camera extrinsics network")
    parser.add_argument('--pose_lr_end', default=0.0001, type=float, help="Learning rate end for NeRF-- camera extrinsics network")
    parser.add_argument('--pose_lr_exponential_index', default=9, type=int, help="Learning rate speed of exponential decay (higher value = faster initial decay) for NeRF-- camera extrinsics network")
    parser.add_argument('--pose_lr_curvature_shape', default=1, type=int, help="Learning rate shape of decay (lower value = faster initial decay) for NeRF-- camera extrinsics network")

    parser.add_argument('--depth_to_rgb_loss_start', default=0.0075, type=float, help="Learning rate start for ratio of loss importance between depth and RGB inverse rendering loss")
    parser.add_argument('--depth_to_rgb_loss_end', default=0.0, type=float, help="Learning rate end for ratio of loss importance between depth and RGB inverse rendering loss")
    parser.add_argument('--depth_to_rgb_loss_exponential_index', default=9, type=int, help="Learning rate speed of exponential decay (higher value = faster initial decay) for ratio of loss importance between depth and RGB inverse rendering loss")
    parser.add_argument('--depth_to_rgb_loss_curvature_shape', default=1, type=int, help="Learning rate shape of decay (lower value = faster initial decay) for ratio of loss importance between depth and RGB inverse rendering loss")

    # Define parameters the determines the overall size and learning capacity of the neural networks and their encodings
    parser.add_argument('--density_neural_network_parameters', type=int, default=512, help='The baseline number of units that defines the size of the NeRF geometry network')
    parser.add_argument('--color_neural_network_parameters', type=int, default=512, help='The baseline number of units that defines the size of the NeRF RGB (pitch,yaw) network')
    parser.add_argument('--positional_encoding_fourier_frequencies', type=int, default=10, help='The number of frequencies that are generated for positional encoding of (x,y,z)')
    parser.add_argument('--directional_encoding_fourier_frequencies', type=int, default=10, help='The number of frequencies that are generated for positional encoding of (pitch, yaw)')

    # Define sampling parameters, including how many samples per raycast (outward), number of samples randomly selected per image, and (if masking is used) ratio of good to masked samples
    parser.add_argument('--pixel_samples_per_epoch', type=int, default=512, help='The number of rows of samples to randomly collect for each image during training')
    parser.add_argument('--number_of_samples_outward_per_raycast', type=int, default=1028, help='The number of samples per raycast to collect (linearly)')
    parser.add_argument('--percent_of_sensor_depth_as_standard_deviation', type=float, default=0.003, help='The standard deviation of sampling by depth')
    parser.add_argument('--gaussian_sampling_around_depth_sensor', type=bool, default=False, help='An unproven technique that could be useful if refined')
    parser.add_argument('--depth_sensor_error', type=float, default=0.05 * 1000, help='Standard deviation of Gaussian depth sensor model, in millimeters')
    parser.add_argument('--epsilon', type=float, default=0.0001, help='Minimum value in log() for NeRF density weights going to 0')

    # Additional parameters on pre-processing of depth data and coordinate systems
    parser.add_argument('--maximum_depth', type=float, default=5.0, help='All depths below this value will be clipped to this value')
    parser.add_argument('--filter_depth_by_confidence', type=int, default=0, help='A value in [0,1,2] where 0 allows all depth data to be used, 2 filters the most and ignores that')

    parsed_args = parser.parse_args()

    # Save parameters with Weights & Biases log
    wandb.init(project="nerf--", entity="3co", config=parsed_args)

    return parser.parse_args()


class SceneModel:
    def __init__(self, args):
        # initialize high-level arguments
        self.args = args
        self.epoch = 1
        self.start_time = int(time.time()) 
        self.device = torch.device('cuda:0') 

        # get camera intrinsics (same for all images)
        self.load_camera_intrinsics()

        # get camera extrinsics (for each image)
        self.load_camera_extrinsics()

        # load all unique IDs (names without ".png") of images to self.image_ids
        self.load_all_images_ids()

        # define bounds (self.min_x, self.max_x), (self.min_y, self.max_y), (self.min_z, self.max_z) in which all points should initially project inside, or else not be included
        self.set_xyz_bounds_from_crop_of_image(index_to_filter=0, min_pixel_row=0, max_pixel_row=self.H, min_pixel_col=0, max_pixel_col=self.W) # pre-trained model with min_pixel_row=200, max_pixel_row=400, min_pixel_col=250, max_pixel_col=450; tight-clipping for small image is min_pixel_row=110, max_pixel_row=390, min_pixel_col=170, max_pixel_col=450

        # prepare test evaluation indices
        self.prepare_test_data()

        # now load only the necessary data that falls within bounds defined
        self.load_image_and_depth_data_within_xyz_bounds()

        # compute/load estimates of sensor variance
        self.load_estimates_of_depth_sensor_error()

        # initialize all models
        self.initialize_models()
        self.initialize_learning_rates()

        if self.args.load_pretrained_models:
            print("Loading pretrained models")
            # load pre-trained model
            self.load_pretrained_models()

            # compute the ray directions using the latest focal lengths, derived for the first image
            focal_length_x, focal_length_y = self.models["focal"](0)
            self.compute_ray_direction_in_camera_coordinates(focal_length_x, focal_length_y)

            if self.args.save_ply_point_clouds_of_sensor_data_with_learned_poses:
                self.save_point_clouds_with_sensor_depths()

            self.save_cam_xyz()
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

        # read the 16 bit greyscale depth data which is formatted as an integer of millimeters
        depth_mm = cv2.imread(depth_path, -1).astype(np.float32)

        # convert data in millimeters to meters
        depth_m = depth_mm / (1000.0)  

        # # filter by confidence
        # if confidence_data is not None:
        #     depth_m[confidence_data < self.args.filter_depth_by_confidence] = 0.0

        # set a cap on the maximum depth in meters; clips erroneous/irrelevant depth data from way too far out
        depth_m[depth_m > self.args.maximum_depth] = self.args.maximum_depth

        # resize to a resolution that e.g. may be higher, and equivalent to image data
        # to do: incorporate lower confidence into the interpolated depth metrics!
        resized_depth_meters = cv2.resize(depth_m, (self.W, self.H), interpolation=cv2.INTER_NEAREST_EXACT)

        # NeRF requires bounds which can be used to constrain both the processed coordinate system data, as well as the ray sampling
        # for fast look-up, define nearest and farthest depth measured per image
        near_bound = np.min(resized_depth_meters)
        far_bound = np.max(resized_depth_meters)

        depth = torch.Tensor(resized_depth_meters).to(device=self.device) # (N_images, H_image, W_image)

        return depth, near_bound, far_bound


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

        self.initial_focal_length_x = self.camera_intrinsics[0,0]
        self.initial_focal_length_y = self.camera_intrinsics[1,1]
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


    def get_point_cloud(self, pose, depth, rgb, label=0, save=False, remove_zero_depths=True, save_raw_xyz=False):
        camera_world_position = pose[:3, 3].view(1, 1, 3)     # (1, 1, 3)
        camera_world_rotation = pose[:3, :3].view(1, 1, 3, 3) # (1, 1, 3, 3)
        pixel_directions = self.pixel_directions.unsqueeze(3) # (H, W, 3, 1)

        xyz_coordinates = self.derive_xyz_coordinates(camera_world_position, camera_world_rotation, pixel_directions, depth)

        if save_raw_xyz:
            file_path = "{}/{}_xyz_raw.npy".format(self.args.base_directory, label)
            with open(file_path, "wb") as f:
                np.save(f, xyz_coordinates.cpu().detach().numpy())

        if remove_zero_depths:
            non_zero_depth = torch.where(depth!=0.0)
            depth = depth[non_zero_depth]
            pixel_directions = pixel_directions[non_zero_depth]
            rgb = rgb[non_zero_depth]
            xyz_coordinates = xyz_coordinates[non_zero_depth]

        pcd = self.create_point_cloud(xyz_coordinates, rgb, label="point_cloud_{}".format(label), flatten_xyz=False, flatten_image=False)
        if save:
            file_name = "{}/view_{}_training_data_{}.ply".format(self.args.base_directory, label, self.epoch-1)
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

        color_out_path = Path("{}/mask_for_filtering_{}.png".format(self.args.base_directory, mask_index))
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
        self.initial_selected_xyz = []
        self.test_poses_processed = 0
        number_of_test_images_saved = 0

        # now loop through all of the data, and filter out (only load and save as necessary) based on whether the points land within our focus area
        for i, image_id in enumerate(self.image_ids[::self.args.skip_every_n_images_for_training]):

            # get depth data for this image
            depth, near_bound, far_bound = self.load_depth_data(image_id=image_id) # (H, W)

            # get (x,y,z) coordinates for this image
            xyz_coordinates = self.get_sensor_xyz_coordinates(pose_data=self.initial_poses[i*self.args.skip_every_n_images_for_training], depth_data=depth) # (H, W, 3)

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

            if self.args.recompute_sensor_variance_from_initial_data:
                self.initial_selected_xyz.append(xyz_coordinates_selected)

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
                pcd = self.get_point_cloud(pose=self.initial_poses[i*self.args.skip_every_n_images_for_training], depth=depth, rgb=image, label="raw_{}".format(image_id), save=True)


        # bring the data together
        self.rgbd = torch.cat(self.rgbd, dim=0)
        self.pixel_rows = torch.cat(self.pixel_rows, dim=0)
        self.pixel_cols = torch.cat(self.pixel_cols, dim=0)
        self.image_ids_per_pixel = torch.cat(self.image_ids_per_pixel, dim=0)
        
        # and clean up
        self.number_of_pixels = self.image_ids_per_pixel.shape[0]
        self.near = torch.min(self.rgbd[:,3])
        self.far = torch.max(self.rgbd[:,3])
        print("The near bound is {:.3f} meters and the far bound is {:.3f} meters".format(self.near, self.far))
        self.initial_poses = self.initial_poses[::self.args.skip_every_n_images_for_training]


        print("Loaded {} images with {:,} pixels selected".format(i+1, self.number_of_pixels ))


    def load_estimates_of_depth_sensor_error(self):
        self.average_nearest_neighbor_distance_per_pixel = []

        if self.args.recompute_sensor_variance_from_initial_data:
            sensor_variance_dir = Path("{}/sensor_variance".format(self.args.base_directory))
            sensor_variance_dir.mkdir(parents=True, exist_ok=True)
            
            print("Recomputing estimates of depth sensor error based on KNN for each point initially projected in 3D")
            number_of_views = len(self.initial_selected_xyz)
            for view, image_id in enumerate(self.image_ids[::self.args.skip_every_n_images_for_training]):
                this_view_xyz = self.initial_selected_xyz[view] # (N_pixels, 3)
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
                    other_view_xyz = self.initial_selected_xyz[comparison_view] # (N_pixels, 3)
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

            # clear data from RAM
            self.initial_selected_xyz = None

        else:
            # otherwise we presume that we've already computed the above, and try to load the data directly
            for view, image_id in enumerate(self.image_ids[::self.args.skip_every_n_images_for_training]):
                error_metric_file_name = "image_id_{}_estimated_sensor_error_from_top_{}_knn_distances.npy".format(image_id, self.args.number_of_nearest_neighbors_to_use_in_knn_distance_metric_for_estimation_of_sensor_error)
                file_path = "{}/sensor_variance/{}".format(self.args.base_directory, error_metric_file_name)
                error_metric_data =  torch.from_numpy(np.load(file_path))
                self.average_nearest_neighbor_distance_per_pixel.append(error_metric_data)

        # now we need to flatten the sensor variance estimates, just like the rest of the data
        self.estimated_sensor_error = torch.cat(self.average_nearest_neighbor_distance_per_pixel, dim=0)


    def derive_xyz_coordinates(self, camera_world_position, camera_world_rotation, pixel_directions, pixel_depths, flattened=False):
        if not flattened:
            # transform rays from camera coordinate to world coordinate
            pixel_world_directions = torch.matmul(camera_world_rotation, pixel_directions).squeeze(3)  # (1, 1, 3, 3) * (H, W, 3, 1) -> (H, W, 3)

            # Get sample position in the world (1, 1, 3) + (H, W, 3) * (H, W, 1) -> (H, W, 3)
            global_xyz = camera_world_position + pixel_world_directions * pixel_depths.unsqueeze(2)

        # else:
        #     pixel_directions_world = torch.matmul(poses[:,:3, :3], pixel_directions.unsqueeze(2)).squeeze(2)  # (N, 3, 3) * (N, 3, 1) -> (N, 3) .squeeze(3) 
        #     poses_xyz = poses[:, :3, 3]  # the translation vectors (N, 3)
        #     pixel_depth_samples_world_directions = pixel_directions_world.unsqueeze(1) * resampled_depths.unsqueeze(2) # (N_pixels, N_samples, 3)
        #     pixel_xyz_positions = poses_xyz.unsqueeze(1).expand(N_pixels, N_samples, 3) + pixel_depth_samples_world_directions # (N_pixels, N_samples, 3)


        return global_xyz






    def get_sensor_xyz_coordinates(self, i=None, pose_data=None, depth_data=None):
        if type(i) != type(None):
            pose_data = self.initial_poses[i,:,:].to(self.device)
        elif type(pose_data) == type(None):
            print("Requires index i or pose data")
            return None

        # get camera world position and rotation
        camera_world_position = pose_data[:3, 3].view(1, 1, 3)     # (1, 1, 3)
        camera_world_rotation = pose_data[:3, :3].view(1, 1, 3, 3) # (1, 1, 3, 3)

        # get relative pixel orientations
        pixel_directions = self.pixel_directions.unsqueeze(3) # (H, W, 3, 1)

        if type(i) != type(None):
            depth_data = self.depths[i,:,:].to(self.device) # (H, W, 1)
        elif type(depth_data) == type(None):
            print("Requires index i or depth data")
            return None

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
        depth, near_bound, far_bound = self.load_depth_data(image_id=image_id)

        # now, get (x,y,z) coordinates for the first image
        xyz_coordinates = self.get_sensor_xyz_coordinates(pose_data=self.initial_poses[index_to_filter], depth_data=depth)

        # now filter both the xyz_coordinates and the image by the values in the top of this function
        if type(min_pixel_row) != type(None) or type(max_pixel_row) != type(None):
            xyz_coordinates = xyz_coordinates[min_pixel_row:max_pixel_row, :, :]
        if type(min_pixel_col) != type(None) or type(max_pixel_col) != type(None):
            xyz_coordinates = xyz_coordinates[:, min_pixel_col:max_pixel_col, :]

        # now define the bounds in (x,y,z) space through which we will filter all future pixels by their projected points
        self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z = self.get_min_max_bounds(xyz_coordinates, padding=1.0)
 


    def compute_ray_direction_in_camera_coordinates(self, focal_length_x, focal_length_y):
        # Compute ray directions in the camera coordinate, which only depends on intrinsics. This could be further transformed to world coordinate later, using camera poses.
        camera_coordinates_y, camera_coordinates_x = torch.meshgrid(torch.arange(self.H, dtype=torch.float32, device=self.device),
                                                                    torch.arange(self.W, dtype=torch.float32, device=self.device),
                                                                    indexing='ij')  # (H, W)

        # Use OpenGL coordinate in 3D:
        #   camera_coordinates_x points to right
        #   camera_coordinates_y points to up
        #   camera_coordinates_z points to backward
        #
        # The coordinate of the top left corner of an image should be (-0.5W, 0.5H, -1.0).
        camera_coordinates_directions_x = (camera_coordinates_x - self.principal_point_x) / focal_length_x  # (H, W) # self.W   0.5 * self.principal_point_x
        camera_coordinates_directions_y = (camera_coordinates_y - self.principal_point_y) / focal_length_y  # (H, W) # self.H  0.5 * self.principal_point_y
        camera_coordinates_directions_z = torch.ones(self.H, self.W, dtype=torch.float32, device=self.device)  # (H, W)
        camera_coordinates_pixel_directions = torch.stack([camera_coordinates_directions_x, camera_coordinates_directions_y, camera_coordinates_directions_z], dim=-1)  # (H, W, 3)

        self.pixel_directions = camera_coordinates_pixel_directions.to(device=self.device) 



    #########################################################################
    ####### Define model data structures and set learning parameters ########
    #########################################################################

    def initialize_models(self):
        self.models = {}

        # Load the relevant models
        self.models["focal"] = CameraIntrinsicsModel(self.H, self.W, self.initial_focal_length_x, self.initial_focal_length_y).to(device=self.device)
        self.models["pose"] = CameraPoseModel(self.initial_poses).to(device=self.device)
        self.models["geometry"] = NeRFDensity(self.args).to(device=self.device)
        self.models["color"] = NeRFColor(self.args).to(device=self.device)

        # Set up Weights & Biases logging on top of the network in order to record its structure
        wandb.watch(self.models["focal"])
        wandb.watch(self.models["pose"])
        wandb.watch(self.models["geometry"])
        wandb.watch(self.models["color"])


    def get_polynomial_decay(self, start_value, end_value, exponential_index=1, curvature_shape=1):
        return (start_value - end_value) * (1 - self.epoch**curvature_shape / self.args.number_of_epochs**curvature_shape)**exponential_index + end_value


    def create_polynomial_learning_rate_schedule(self, model):
        schedule = PolynomialDecayLearningRate(optimizer=self.optimizers[model], 
                                               total_steps=self.args.number_of_epochs, 
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


    def load_pretrained_models(self, path="models", epoch=100001):
        for model_name in self.models.keys():
            model = self.models[model_name]
            #model_path = "{}/{}_{}.pth".format(path, model_name, epoch)
            model_path = "models/pillow_small_model_1/{}_{}.pth".format(model_name, epoch)
            model = self.load_model(model_path=model_path, model=model)
            self.models[model_name] = model.to(device=self.device)


    def load_model(self, model_path, model):
        ckpt = torch.load(model_path, map_location=self.device)
        weights = ckpt['model_state_dict']
        model.load_state_dict(weights, strict=True)
        return model

    def save_models(self):
        for topic in ["color", "geometry", "pose", "focal"]:
            model = self.models[topic]
            optimizer = self.optimizers[topic]
            print("Saving {} model...".format(topic))
            save_checkpoint(epoch=self.epoch, model=model, optimizer=optimizer, path=self.experiment_dir, ckpt_name='{}_{}'.format(topic, self.epoch))

    def save_point_clouds_with_sensor_depths(self):
        for i, image_id in enumerate(self.image_ids[::self.args.skip_every_n_images_for_training]):
            print("Saving with learned poses and intrinsics the raw sensor colors and sensor depth for view {}".format(i))
            image, _ = self.load_image_data(image_id=image_id)
            depth, _, _ = self.load_depth_data(image_id=image_id)
            pose = self.models['pose'](0)[i].to(device=self.device)
            self.get_point_cloud(pose=pose, depth=depth, rgb=image, label="raw_sensor_with_learned_poses_intrinsics_{}".format(i), save=True, save_raw_xyz=True)               


    #########################################################################
    ############ Helper functions for model usage and training ##############
    #########################################################################

    def get_raycast_samples_per_pixel(self, number_of_pixels, sensor_depth=None, add_noise=True):

        if self.args.gaussian_sampling_around_depth_sensor:
            # sample about sensor depth, with increasing standard deviations for longer depths

            # randomly shift the center of the mean, potentially to help prevent overfitting of depth sensor data
            if add_noise:
                zero_mean = torch.normal(mean=torch.zeros(number_of_pixels).to(self.device), std=0.001).to(self.device)
            else:
                zero_mean = torch.normal(mean=torch.zeros(number_of_pixels).to(self.device), std=0.00).to(self.device)

            # an empirically-derived standard deviation for sampling, which penalizes depths that are further away (samples a wider distribution around further depths)
            depth_dependent_standard_deviation = ((sensor_depth * 1000) ** 1.4) / 1000 * self.args.percent_of_sensor_depth_as_standard_deviation
            sampled_standard_deviations = torch.normal(mean=zero_mean, std=depth_dependent_standard_deviation)

            # collect from -1 to 1 the baseline samples per raycast
            raycast_distances = torch.linspace(-1, 1, self.args.number_of_samples_outward_per_raycast).to(self.device)
            raycast_distances = raycast_distances.unsqueeze(0).expand(number_of_pixels, self.args.number_of_samples_outward_per_raycast)

            # now, take the absolute value for each of the random standard deviations defined above, and consider 2x that value to be the (min,max) sampling distance, with linear values in between 
            max_depth_sample_range = torch.abs(sampled_standard_deviations).unsqueeze(1).expand(number_of_pixels, self.args.number_of_samples_outward_per_raycast) * 2 

            # multiply [-1, -.9, ..., .9, 1] x max range value
            depth_samples = raycast_distances * max_depth_sample_range

            # given the *relative* range of samples, now just add that to the actual depth samples 
            final_samples = sensor_depth.unsqueeze(1).expand(number_of_pixels, self.args.number_of_samples_outward_per_raycast) + depth_samples

            clipped_final_samples = torch.clip(input=final_samples, min=0.0, max=2.0)

            return clipped_final_samples

        else:
            raycast_distances = torch.linspace(self.near, self.far, self.args.number_of_samples_outward_per_raycast).to(self.device)
            raycast_distances = raycast_distances.unsqueeze(0).expand(number_of_pixels, self.args.number_of_samples_outward_per_raycast)

            if add_noise:
                depth_noise = torch.rand((number_of_pixels, self.args.number_of_samples_outward_per_raycast), device=self.device, dtype=torch.float32)  # (N_pixels, N_samples)
                depth_noise = depth_noise * (self.far - self.near) / self.args.number_of_samples_outward_per_raycast # (N_pixels, N_samples)
                raycast_distances = raycast_distances + depth_noise  # (N_pixels, N_samples)

            return raycast_distances


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

        with open("{}/xyz_depths_view_{}.npy".format(self.args.base_directory, test_view_number), "wb") as f:
            np.save(f, xyz_depths.cpu().numpy())

        with open("{}/xyz_slopes_view_{}.npy".format(self.args.base_directory, test_view_number), "wb") as f:
            np.save(f, xyz_slopes.cpu().numpy())

        with open("{}/xyz_intercepts_view_{}.npy".format(self.args.base_directory, test_view_number), "wb") as f:
            np.save(f, xyz_intercepts.cpu().numpy())

        with open("{}/depth_weights_view_{}.npy".format(self.args.base_directory, test_view_number), "wb") as f:
            np.save(f, depth_weights.cpu().numpy())


    def render(self, poses, pixel_directions, sampling_depths, perturb_depths=False, rgb_image=None):
        # poses := (N_pixels, 4, 4)
        # pixel_directions := (N_pixels, 3)
        # sampling_depths := (N_samples)

        # (N_pixels, N_sample, 3), (N_pixels, 3), (N_pixels, N_samples)            
        pixel_xyz_positions, pixel_directions_world, resampled_depths = volume_sampling(poses=poses, pixel_directions=pixel_directions, sampling_depths=sampling_depths, perturb_depths=perturb_depths)

        # encode position: (H, W, N_sample, (2L+1)*C = 63)
        xyz_position_encoding = encode_position(pixel_xyz_positions, levels=self.args.positional_encoding_fourier_frequencies)
        # encode direction: (H, W, N_sample, (2L+1)*C = 27)
        pixel_directions_world = torch.nn.functional.normalize(pixel_directions_world, p=2, dim=1)  # (N_pixels, 3)
        angular_directional_encoding = encode_position(pixel_directions_world, levels=self.args.directional_encoding_fourier_frequencies)  # (N_pixels, 27)
        angular_directional_encoding = angular_directional_encoding.unsqueeze(1).expand(-1, self.args.number_of_samples_outward_per_raycast, -1)  # (N_pixels, N_sample, 27)

        # inference rgb and density using position and direction encoding.
        
        density, features = self.models["geometry"](xyz_position_encoding) # (N_pixels, N_sample, 1), # (N_pixels, N_sample, D)
        rgb = self.models["color"](features, angular_directional_encoding)  # (N_pixels, N_sample, 4)

        render_result = volume_rendering(rgb, density, resampled_depths)

        result = {
            'rgb_rendered': render_result['rgb_rendered'], # (N_pixels, 3)
            'pixel_xyz_positions': pixel_xyz_positions,    # (N_pixels, N_sample, 3)
            'depth_map': render_result['depth_map'],       # (N_pixels)
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
    def render_prediction_for_train_image(self, train_image_index):

        mask = torch.zeros(self.H, self.W)        

        pixel_indices_for_this_image = torch.argwhere(self.image_ids_per_pixel == train_image_index)        
        pixel_rows = self.pixel_rows[pixel_indices_for_this_image]
        pixel_cols = self.pixel_cols[pixel_indices_for_this_image]        
        mask[pixel_rows, pixel_cols] = 1
        mask = mask.flatten()
        pose = self.models['pose'](0)[train_image_index]

        return self.render_prediction(pose=pose, train_image_index=train_image_index, mask=mask)
        

    # invoke current model for a specific pose and 1d mask
    # for visual results, supply result to save_render_as_png
    # pixels filtered out by the mask are assigned value [0,0,0] 
    def render_prediction(self, pose, train_image_index, mask=None):        

        # copy pose to go with each individual pixel and match reference pixel directions with them
        if mask is not None:
            poses = pose.unsqueeze(0).expand(int(mask.sum().item()), -1, -1)
            pixel_directions = self.pixel_directions.flatten(start_dim=0, end_dim=1)[torch.argwhere(mask==1)]
            pixel_directions = pixel_directions.squeeze()                        
        else:
            poses = pose.unsqueeze(0).expand(self.W*self.H, -1, -1)
            pixel_directions = self.pixel_directions.flatten(start_dim=0, end_dim=1)
    
        depth_samples = self.get_raycast_samples_per_pixel(number_of_pixels=len(poses), add_noise=False)                

        # split each of the rendering inputs into smaller batches, which speeds up GPU processing
        poses_batches = poses.split(self.args.number_of_pixels_per_batch_in_test_renders)
        pixel_directions_batches = pixel_directions.split(self.args.number_of_pixels_per_batch_in_test_renders)
        depth_samples_batches = depth_samples.split(self.args.number_of_pixels_per_batch_in_test_renders)         

        rendered_image_batches = []
        depth_image_batches = []
        density_batches = []

        if self.args.export_test_data_for_post_processing:
            depth_weights_batches = []
            pixel_xyz_positions_batches = []

        # for each batch, compute the render and extract out RGB and depth map                  
        for poses_batch, pixel_directions_batch, depth_samples_batch in zip(poses_batches, pixel_directions_batches, depth_samples_batches):                                   
            rendered_data = self.render(poses=poses_batch, pixel_directions=pixel_directions_batch, sampling_depths=depth_samples_batch, perturb_depths=False)  # (N_pixels, 3)            
            rendered_image = rendered_data['rgb_rendered'] # (n_pixels_per_row, 3)
            rendered_depth = rendered_data['depth_map'] # (n_pixels_per_row)
            density = rendered_data['density']
            rendered_image_batches.append(rendered_image)
            depth_image_batches.append(rendered_depth)             
            density_batches.append(density)

            if self.args.export_test_data_for_post_processing:
                depth_weights_batches.append(rendered_data['depth_weights'])
                pixel_xyz_positions_batches.append(rendered_data['pixel_xyz_positions'])

        # combine batch results to compose full images
        rendered_image_data = torch.cat(rendered_image_batches, dim=0) # (N_pixels, 3)
        rendered_depth_data = torch.cat(depth_image_batches, dim=0)  # (N_pixels)
        density_data = torch.cat(density_batches, dim=0)  # (N_pixels, N_samples)

        if self.args.export_test_data_for_post_processing:
            depth_weight_per_sample = torch.cat(depth_weights_batches, dim=0)
            xyz_for_all_samples = torch.cat(pixel_xyz_positions_batches, dim=0)
            self.export_geometry_data(depth_weight_per_sample=depth_weight_per_sample, xyz_for_all_samples=xyz_for_all_samples, test_view_number=train_image_index, top_n_depth_weights = 10)

        rendered_image = torch.zeros(self.H * self.W, 3)
        rendered_depth = torch.zeros(self.H * self.W)
        density = torch.zeros(self.H * self.W, len(density_data[0]))

        # if we're using a mask, we need to find the right pixel positions
        if mask is not None:
            rendered_image[torch.argwhere(mask==1).squeeze()] = rendered_image_data.cpu()
            rendered_depth[torch.argwhere(mask==1).squeeze()] = rendered_depth_data.cpu()
            density[torch.argwhere(mask==1).squeeze()] = density_data.cpu()
        else:
            rendered_image = rendered_image_data.cpu()
            rendered_depth = rendered_depth_data.cpu()
            density = density_data.cpu()

        render_result = {
            'rendered_image': rendered_image,
            'rendered_depth': rendered_depth,
            'density': density,
        }

        return render_result    


    # process raw rendered pixel data and save into images
    def save_render_as_png(self, render_result, color_file_name, depth_file_name):

        rendered_rgb = render_result['rendered_image'].reshape(self.H, self.W, 3)
        rendered_depth = render_result['rendered_depth'].reshape(self.H, self.W)

        rendered_color_for_file = (rendered_rgb.cpu().numpy() * 255).astype(np.uint8)    

        # get depth map and convert it to Turbo Color Map
        rendered_depth_data = rendered_depth.cpu().numpy() 
        rendered_depth_for_file = heatmap_to_pseudo_color(rendered_depth_data)
        rendered_depth_for_file = (rendered_depth_for_file * 255).astype(np.uint8)

        imageio.imwrite(color_file_name, rendered_color_for_file)
        imageio.imwrite(depth_file_name, rendered_depth_for_file)   


    def train(self):

        if self.epoch == self.args.early_termination_epoch:
            print("Terminating early to speed up hyperparameter search")
            sys.exit(0)

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


        # shuffle training batch indices
        indices_of_random_pixels = random.sample(population=range(self.number_of_pixels), k=self.args.pixel_samples_per_epoch)

        # get the randomly selected RGBD data
        rgbd = self.rgbd[indices_of_random_pixels].to(self.device)  # (N_pixels, 4)

        # get the camera intrinsics
        if self.epoch >= self.args.start_training_intrinsics_epoch:
            focal_length_x, focal_length_y = self.models["focal"](0)
        else:
            with torch.no_grad():
                focal_length_x, focal_length_y = self.models["focal"](0)
        
        self.compute_ray_direction_in_camera_coordinates(focal_length_x, focal_length_y)

        number_of_pixels = self.args.pixel_samples_per_epoch
        number_of_raycast_samples = self.args.number_of_samples_outward_per_raycast

        # get all camera poses from model
        if self.epoch >= self.args.start_training_extrinsics_epoch:
            poses = self.models["pose"](0) # (N_images, 4, 4)
        else:
            with torch.no_grad():
                poses = self.models["pose"](0)  # (N_images, 4, 4)

        # get a tensor with the poses per pixel
        image_ids = self.image_ids_per_pixel[indices_of_random_pixels].to(self.device) # (N_pixels)
        selected_poses = poses[image_ids].to(self.device) # (N_pixels, 4, 4)

        # get the pixel rows and columns that we've selected (across all images)
        pixel_rows = self.pixel_rows[indices_of_random_pixels]
        pixel_cols = self.pixel_cols[indices_of_random_pixels]

        # unpack the image RGB data and the sensor depth
        rgb = rgbd[:,:3].to(self.device) # (N_pixels, 3)
        sensor_depth = rgbd[:,3].to(self.device) # (N_pixels) 
        
        # get pixel directions
        pixel_directions_selected = self.pixel_directions[pixel_rows, pixel_cols]  # (N_pixels, 3)

        # sample about sensor depth, with increasing standard deviations for longer depths
        depth_samples = self.get_raycast_samples_per_pixel(number_of_pixels=sensor_depth.shape[0], add_noise=True) # (N_pixels, N_samples) 
        sensor_depth_per_sample = sensor_depth.unsqueeze(1).expand(-1,depth_samples.shape[1]) # (N_pixels, N_samples) 

        # render an image using selected rays, pose, sample intervals, and the network
        render_result = self.render(poses=selected_poses, pixel_directions=pixel_directions_selected, sampling_depths=depth_samples, perturb_depths=False, rgb_image=rgb)  # (N_pixels, 3)

        rgb_rendered = render_result['rgb_rendered']  # (N_pixels, 3)
        nerf_depth_weights = render_result['depth_weights'] # (N_pixels, N_samples)
        nerf_sample_bin_lengths = render_result['distances'] # (N_pixels, N_samples)

        nerf_depth_weights = nerf_depth_weights + self.args.epsilon

        # get the estimated sensor error for these randomly selected pixels
        sensor_error = self.estimated_sensor_error[indices_of_random_pixels].to(self.device) # (N_pixels)
        sensor_error = sensor_error.unsqueeze(1).expand(number_of_pixels, number_of_raycast_samples) # (N_pixels, N_samples)

        kl_divergence_bins = -1 * torch.log(nerf_depth_weights) * torch.exp(-1 * (depth_samples * 1000 - sensor_depth_per_sample * 1000) ** 2 / (2 * sensor_error)) * nerf_sample_bin_lengths * 1000                                
        kl_divergence_pixels = torch.sum(kl_divergence_bins, 1)
        depth_loss = torch.mean(kl_divergence_pixels)
        
        # get a metric in Euclidian space that we can output via prints for human review/intuition; not actually used in backpropagation
        interpretable_depth_loss = torch.sum(nerf_depth_weights * torch.sqrt((depth_samples * 1000 - sensor_depth_per_sample * 1000) ** 2), dim=1)
        interpretable_depth_loss_per_pixel = torch.mean(interpretable_depth_loss)

        # get a metric in (0-255) (R,G,B) space that we can output via prints for human review/intuition; not actually used in backpropagation
        interpretable_rgb_loss = torch.sqrt((rgb_rendered * 255 - rgb * 255) ** 2)
        interpretable_rgb_loss_per_pixel = torch.mean(interpretable_rgb_loss)

        # compute the mean squared difference between the RGB render of the neural network and the original image     
        rgb_loss = (rgb_rendered - rgb)**2
        rgb_loss = torch.mean(rgb_loss)

        # to-do: implement perceptual color difference minimizer
        #  torch.norm(ciede2000_diff(rgb2lab_diff(inputs,self.device),rgb2lab_diff(adv_input,self.device),self.device).view(batch_size, -1),dim=1)

        # get the relative importance between depth and RGB loss
        depth_to_rgb_importance = self.get_polynomial_decay(start_value=self.args.depth_to_rgb_loss_start, end_value=self.args.depth_to_rgb_loss_end, exponential_index=self.args.depth_to_rgb_loss_exponential_index, curvature_shape=self.args.depth_to_rgb_loss_curvature_shape)

        # compute loss and backward propagate the gradients to update the values which are parameters to this loss
        weighted_loss = depth_to_rgb_importance * depth_loss + (1 - depth_to_rgb_importance) * rgb_loss
        unweighted_loss = rgb_loss + depth_loss
        weighted_loss.backward()

        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad()

        if self.epoch % self.args.log_frequency == 0:
            wandb.log({"RGB Inverse Render Loss (0-255 per pixel)": interpretable_rgb_loss_per_pixel,
                       "Depth Sensor Loss (average millimeters error vs. sensor)": interpretable_depth_loss_per_pixel,
                       })

        if self.epoch % self.args.log_frequency == 0:
            minutes_into_experiment = (int(time.time())-int(self.start_time)) / 60
            print("({} at {:.2f} minutes) - Total Loss: {:.6f}, RGB Loss: {:.6f} ({:.3f} of 255), Depth Loss: {:.6f} ({:3f} mm), Focal Length X: {:.3f}, Focal Length Y: {:.3f}".format(self.epoch, 
                                                                                                                                                                                        minutes_into_experiment, 
                                                                                                                                                                                        weighted_loss,
                                                                                                                                                                                        (1 - depth_to_rgb_importance) * rgb_loss, 
                                                                                                                                                                                        interpretable_rgb_loss_per_pixel, 
                                                                                                                                                                                        depth_to_rgb_importance * depth_loss, 
                                                                                                                                                                                        interpretable_depth_loss_per_pixel, 
                                                                                                                                                                                        focal_length_x, 
                                                                                                                                                                                        focal_length_y))
        # update the learning rate schedulers
        for scheduler in self.schedulers.values():
            scheduler.step()

        # a new epoch has dawned
        self.epoch += 1


    def test(self):
        epoch = self.epoch - 1

        if epoch == 0:
            return

        for model in self.models.values():
            model.eval()

        # compute the ray directions using the latest focal lengths, derived for the first image
        focal_length_x, focal_length_y = self.models["focal"](0)
        self.compute_ray_direction_in_camera_coordinates(focal_length_x, focal_length_y)

        for image_index in self.test_image_indices:
            render_result = self.render_prediction_for_train_image(image_index)

            # define file saving parameters
            image_out_dir = "{}/hyperparam_experiments".format(self.args.base_directory)
            number_of_samples_outward_per_raycasts = "{}".format(self.args.number_of_samples_outward_per_raycast)
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
            
            experiment_dir = Path(os.path.join(image_out_dir, experiment_label))
            experiment_dir.mkdir(parents=True, exist_ok=True)
            self.experiment_dir = experiment_dir

            out_file_suffix = str(image_index)
            color_out_dir = Path("{}/color_nerf_{}/".format(experiment_dir, str(epoch)))
            color_out_dir.mkdir(parents=True, exist_ok=True)
            depth_out_dir = Path("{}/depth_nerf_{}/".format(experiment_dir, str(epoch)))
            depth_out_dir.mkdir(parents=True, exist_ok=True)
            color_file_name = os.path.join(color_out_dir, str(out_file_suffix).zfill(4) + '_color.png')
            depth_file_name = os.path.join(depth_out_dir, str(out_file_suffix).zfill(4) + '_depth.png')

            self.save_render_as_png(render_result, color_file_name, depth_file_name) 

            if epoch % self.args.save_point_cloud_frequency == 0:
                print("Saving .ply for view {}".format(image_index))
                pose = self.models['pose'](0)[image_index].to(device=self.device)
                depth = render_result['rendered_depth'].reshape(self.H, self.W).to(device=self.device)
                rgb = render_result['rendered_image'].reshape(self.H, self.W, 3).to(device=self.device)
                self.get_point_cloud(pose=pose, depth=depth, rgb=rgb, label="10000k_KL_divergence_view_{}".format(image_index), save=True)               


if __name__ == '__main__':

    # Load a scene object with all data and parameters
    scene = SceneModel(args=parse_args())

    with torch.no_grad():
        scene.test()

    for epoch in range(1, scene.args.number_of_epochs + 1):
        scene.train()

        if epoch % scene.args.test_frequency == 0:
            with torch.no_grad():
                scene.test()
       
        if epoch % scene.args.save_models_frequency == 0 and epoch > 0:
            scene.save_models()