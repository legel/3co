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
import math

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

class CameraDataExporter:
    def __init__(self):
        # parse arguments and initialize basic systems
        self.parse_input_args()

        # load data for exporting
        self.load_all_images_ids()        
        self.load_camera_intrinsics()
        self.load_camera_extrinsics()
        self.initialize_models()        

        # export data
        self.export_camera_data()

    def parse_input_args(self):
        parser = argparse.ArgumentParser()

        # Define path to relevant data for training, and decide on number of images to use in training
        parser.add_argument('--base_directory', type=str, default='./data/spotted_purple_orchid', help='The base directory to load and save information from')
        parser.add_argument('--images_directory', type=str, default='color', help='The specific group of images to use during training')
        parser.add_argument('--images_data_type', type=str, default='jpg', help='Whether images are jpg or png')
        parser.add_argument('--H_for_training', type=int, default=480, help='The height in pixels that training images will be downsampled to')
        parser.add_argument('--W_for_training', type=int, default=640, help='The width in pixels that training images will be downsampled to')            
        parser.add_argument('--number_of_images_in_training_dataset', type=int, default=320, help='The number of images that will be trained on in total for each dataset')                

        self.args = parser.parse_args()


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


    def export_camera_data(self):
        
        camera_extrinsics, camera_intrinsics = self.export_camera_extrinsics_and_intrinsics_from_trained_model(output_directory=None, epoch=None, skip_saving=True)
        n_cameras = camera_extrinsics.shape[0]

        n_val_test_samples = 10

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


    def load_all_images_ids(self):
        # get images in directory of RGB images
        path_to_images = "{}/{}".format(self.args.base_directory, self.args.images_directory)
        unsorted_image_names = glob.glob("{}/*.{}".format(path_to_images, self.args.images_data_type))

        # extract out numbers of their IDs, and sort images by numerical ID
        self.image_ids = np.asarray(sorted([int(image.split("/")[-1].replace(".{}".format(self.args.images_data_type),"")) for image in unsorted_image_names]))
        self.skip_every_n_images_for_training = int(len(self.image_ids) / self.args.number_of_images_in_training_dataset)                
        
        # load the first image to establish initial height and width of image data because it gets downsampled
        self.load_image_data(0)


    def initialize_models(self):
        self.models = {}                

        model = torch.compile(CameraIntrinsicsModel(self.H, self.W, self.initial_focal_length, self.args.number_of_images_in_training_dataset))
        self.models["focal"] = model
        model.to(self.device)

        model = torch.compile(CameraPoseModel(self.selected_initial_poses))                                
        self.models["pose"] = model
        model.to(self.device)


if __name__ == '__main__':
    
    with torch.no_grad():
        scene = SceneModel()