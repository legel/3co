import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import imageio

import json

from utils.comp_ray_dir import comp_ray_dir_cam
from utils.pose_utils import center_poses
from utils.pos_enc import encode_position
from utils.lie_group_helper import convert3x4_4x4, SO3_to_quat
from scipy.spatial.transform import Rotation
import random

import time

from PIL import Image
import cv2

maximum_meters_for_unit_distance = 5.0

def resize_imgs(imgs, new_h, new_w):
    """
    :param imgs:    (N, H, W, 3)            torch.float32 RGB
    :param new_h:   int/torch int
    :param new_w:   int/torch int
    :return:        (N, new_H, new_W, 3)    torch.float32 RGB
    """
    imgs = imgs.permute(0, 3, 1, 2)  # (N, 3, H, W)
    imgs = F.interpolate(imgs, size=(new_h, new_w), mode='bilinear')  # (N, 3, new_H, new_W)
    imgs = imgs.permute(0, 2, 3, 1)  # (N, new_H, new_W, 3)

    return imgs  # (N, new_H, new_W, 3) torch.float32 RGB


def load_imgs(image_dir, img_ids, new_h, new_w):
    img_names = np.array(sorted(os.listdir(image_dir)))  # all image names
    img_names = img_names[img_ids]  # image name for this split

    img_paths = [os.path.join(image_dir, n) for n in img_names]

    img_list = []
    for p in tqdm(img_paths):
        img = imageio.imread(p)[:, :, :3]  # (H, W, 3) np.uint8
        img_list.append(img)
    img_list = np.stack(img_list)  # (N, H, W, 3)
    img_list = torch.from_numpy(img_list).float() / 255  # (N, H, W, 3) torch.float32
    img_list = resize_imgs(img_list, new_h, new_w)
    return img_list, img_names


def load_split(img_dir, img_ids, H, W, load_img):
    N_imgs = img_ids.shape[0]

    if load_img:
        imgs, img_names = load_imgs(img_dir, img_ids, H, W)  # (N, H, W, 3) torch.float32
    else:
        imgs, img_names = None, None

    result = {
        'imgs': imgs,  # (N, H, W, 3) torch.float32
        'img_names': img_names,  # (N, )
        'N_imgs': N_imgs,
    }
    return result


def load_confidence(path):
    return np.array(Image.open(path))



def resize_depth(depth, resize_H, resize_W):
    out = cv2.resize(depth, (resize_W, resize_H), interpolation=cv2.INTER_NEAREST_EXACT)
    # out[out < 10] = 0
    return out


def load_depth_data(path, confidence=None, filter_level=0, max_depth = 1.0, resize_H=480, resize_W=640):
    # read the 16 bit greyscale depth data which is formatted as an integer of millimeters
    depth_mm = cv2.imread(path, -1)

    # convert data in millimeters to meters, and ensure that the maximum meters allowed is 1.0 in depth (so e.g. 5 meters = 1.0 in depth)
    depth_m = depth_mm.astype(np.float32) / (1000.0 * maximum_meters_for_unit_distance)

    # filter by confidence
    if confidence is not None:
        depth_m[confidence < filter_level] = 0.0

    # set a cap on the maximum depth in meters; clips erroneous/irrelevant depth data from way too far out
    depth_m[depth_m > max_depth] = max_depth

    # resize to a resolution that e.g. may be higher, and equivalent to image data
    resized_depth_meters = resize_depth(depth_m, resize_H=resize_H, resize_W=resize_W)

    return resized_depth_meters


def read_meta(in_dir, use_ndc, data_type, num_img_to_load, skip):
    """
    Read the data from the Stray Scanner (https://docs.strayrobots.io/apps/scanner/format.html).
    Do this after running convert_to_open3d.py in order to get images, as well as camera_intrinsics.json
    """

    # load pre-splitted train/val ids
    img_ids = np.loadtxt(os.path.join(in_dir, data_type + '_ids.txt'), dtype=np.int32, ndmin=1)
    if num_img_to_load == -1:
        img_ids = img_ids[::skip]
        print('Loading all available {0:6d} images'.format(len(img_ids)))
    elif num_img_to_load > len(img_ids):
        print('Required {0:4d} images but only {1:4d} images available. '
              'Exit'.format(num_img_to_load, len(img_ids)))
        exit()
    else:
        img_ids = img_ids[:num_img_to_load:skip]

    # print("Loading image IDs: {}".format(img_ids))

    # load camera instrinsics estimates from Apple's internal API
    camera_intrinsics_data = json.load(open(os.path.join(in_dir,'camera_intrinsics.json')))
    H = camera_intrinsics_data["height"]
    W = camera_intrinsics_data["width"]
    F = camera_intrinsics_data["intrinsic_matrix"][0]

    # load odometry data which includes estimates from Apple's ARKit in the form of: timestamp, frame, x, y, z, qx, qy, qz, qw
    poses = []
    odometry = np.loadtxt(os.path.join(in_dir, 'odometry.csv'), delimiter=',', skiprows=1)
    for i,line in enumerate(odometry):

        # ignore data not in the list of images we want to process
        if i not in img_ids:
            continue

        position = np.asarray(line[2:5]) / 5.0 # convert position data in meters to position in 5 meters (so 1.0 of translation equals 5 meters)
        quaternion = line[5:]        
        T_WC = np.eye(4)
        rotation_matrix = Rotation.from_quat(quaternion).as_matrix()
        T_WC[:3, :3] = rotation_matrix
        T_WC[:3, 3] = position
        # # as shown in point cloud processing scripts, we want to get camera2world by taking the inverse of the world2camera 
        T_CW = np.linalg.inv(T_WC)
        poses.append(T_CW)

    # load depth data from LIDAR systems
    depths = []
    bounds = []
    depth_folder = os.path.join(in_dir, 'depth')
    confidence_folder = os.path.join(in_dir, 'confidence')
    for img_id in img_ids:
        confidence_path = os.path.join(confidence_folder, f'{img_id:06}.png')
        depth_path = os.path.join(depth_folder, f'{img_id:06}.png')

        confidence = load_confidence(confidence_path)

        # filter depth by confidence, with only three possible confidence metrics: 0 (least confident), 1 (moderate confidence), 2 (most confident)
        # in practice, a lot of the "least confident" data has useful information, so it does not necessarily make sense to filter from neural network
        depth = load_depth_data(path=depth_path, confidence=confidence, filter_level=0, resize_H=H, resize_W=W)

        # NeRF requires bounds which can be used to constrain both the processed coordinate system data, as well as the ray sampling
        bounds_for_this_image = np.asarray([np.min(depth), np.max(depth)])

        # if i % 500 == 0:
        #     print("Loading depth image {} of {}: min {:.3f}m, max {:.3f}m".format(i, len(poses), bounds_for_this_image[0], bounds_for_this_image[1]))

        bounds.append(bounds_for_this_image)
        depths.append(depth)

    poses = np.asarray(poses)   # (N_images, ...)
    depths = np.asarray(depths) # (N_images, H_image, W_image)
    bounds = np.asarray(bounds) # (N_images, 2)

    rotations_translations = poses[:,:3,:] # get rotations and translations from the 4x4 matrix in a 4x3 matrix

    # print("Rotations and translations for first image after centering:")
    # print(rotations_translations[0,:,:])

    rotations_translations, pose_avg = center_poses(rotations_translations)  # pose_avg @ c2ws -> centred c2ws

    # print("Rotations and translations for first image after centering:")
    # print(rotations_translations[0,:,:])

    # if use_ndc:
    #     # correct scale so that the nearest depth is at a little more than 1.0
    #     # See https://github.com/bmild/nerf/issues/34

    #     near_original = bounds.min()
    #     # print("Minimum of the bounds, near_original: {}".format(near_original))
    #     scale_factor = near_original * 0.75  # 0.75 is the default parameter
    #     # the nearest depth is at 1/0.75=1.33
    #     bounds /= scale_factor

    #     # print("Scale factor is {}".format(scale_factor))

    #     # print("We scaled the bounds and now they are: {}".format(bounds))
    #     # print("Before rescaling c2ws:\n{}".format(rotations_translations[0,:,:]))
    #     rotations_translations[..., 3] /= scale_factor
    #     # print("After rescaling c2ws:\n{}".format(rotations_translations[0,:,:]))
    
    c2ws = convert3x4_4x4(rotations_translations)  # (N, 4, 4)

    results = {
        'img_ids': img_ids, # (N_images)
        'c2ws': c2ws,       # (N, 4, 4) np
        'bounds': bounds,   # (N_images, 2) np
        'depths': depths,
        'H': int(H),        # scalar
        'W': int(W),        # scalar
        'focal': F,            # scalar
        'pose_avg': pose_avg,  # (4, 4) np
    }

    return results


class DataLoaderWithSmartphone:
    """
    Most useful fields:
        self.c2ws:          (N_imgs, 4, 4)      torch.float32
        self.imgs           (N_imgs, H, W, 4)   torch.float32
        self.ray_dir_cam    (H, W, 3)           torch.float32
        self.H              scalar
        self.W              scalar
        self.N_imgs         scalar
    """
    def __init__(self, base_dir, scene_name, data_type, res_ratio, num_img_to_load, skip, use_ndc, load_img=True, percent_of_data_to_use=1):
        """
        :param base_dir:
        :param scene_name:
        :param data_type:   'train' or 'val'.
        :param res_ratio:   int [1, 2, 4] etc to resize images to a lower resolution.
        :param num_img_to_load/skip: control frame loading in temporal domain.
        :param use_ndc      True/False, just centre the poses and scale them.
        :param load_img:    True/False. If set to false: only count number of images, get H and W,
                            but do not load imgs. Useful when vis poses or debug etc.
        :param percent_of_data_to_use   if 1.0 then 100% of images will be read in and poses found, else whatever fraction entered
        """
        my_devices = torch.device('cuda:' + "0")

        self.unix_time = int(time.time()) # unique ID for this experiment

        self.base_dir = base_dir
        self.scene_name = scene_name
        self.data_type = data_type
        self.res_ratio = res_ratio
        self.num_img_to_load = num_img_to_load
        self.skip = skip
        self.use_ndc = use_ndc
        self.load_img = load_img

        self.scene_dir = os.path.join(self.base_dir, self.scene_name)
        self.img_dir = os.path.join(self.scene_dir, 'images')

        # all meta info
        meta = read_meta(in_dir=self.scene_dir, use_ndc=self.use_ndc, data_type=self.data_type, num_img_to_load=self.num_img_to_load, skip=self.skip)

        self.img_ids = meta["img_ids"]
        self.c2ws = meta['c2ws']  # (N, 4, 4) all camera pose
        self.H = meta['H']
        self.W = meta['W']
        self.focal = float(meta['focal'])
        self.total_N_imgs = self.c2ws.shape[0]

        self.depths = torch.Tensor(meta['depths'])
        self.bounds = meta['bounds']

        # total_images_to_use = int(self.total_N_imgs * percent_of_data_to_use)
        # indices_of_selected_random_images = random.sample(range(self.total_N_imgs), total_images_to_use)
        # print(indices_of_selected_random_images)

        if self.res_ratio > 1:
            self.H = self.H // self.res_ratio
            self.W = self.W // self.res_ratio
            self.focal /= self.res_ratio

        '''Load train/val split'''
        split_results = load_split(img_dir=self.img_dir, img_ids=self.img_ids, H=self.H, W=self.W, load_img=self.load_img)
                           
        self.imgs = split_results['imgs']  # (N, H, W, 3) torch.float32
        self.img_names = split_results['img_names']  # (N, )
        self.N_imgs = split_results['N_imgs']

        # print("Image ID's for actual selected images is: {}".format(self.img_ids))

        # generate cam ray dir.
        self.ray_dir_cam = comp_ray_dir_cam(self.H, self.W, self.focal)  # (H, W, 3) torch.float32

        # convert np to torch.
        self.c2ws = torch.from_numpy(self.c2ws).float().to(device=my_devices)  # (N, 4, 4) torch.float32
        self.ray_dir_cam = self.ray_dir_cam.float().to(device=my_devices)  # (H, W, 3) torch.float32

        self.near = []
        self.far = []
        for i, (near_bound, far_bound) in enumerate(self.bounds):
            # print("(Image {}) Near bound: {}, Far bound: {}".format(i, near_bound, far_bound))
            self.near.append(near_bound)
            self.far.append(far_bound)

        self.near = np.asarray(self.near)
        self.far = np.asarray(self.far)

        # encode the camera positions and camera directions in the same fashion as will be done for the camera rays
        # note: this is only used for the PoseNet 
        self.camera_position_encodings = []
        self.camera_direction_encodings = []

        for c2w in self.c2ws:
            cam_direction = c2w[:3, :3]
            cam_position = c2w[:3, 3]

            # convert direction to quaternion vector
            cam_direction = torch.Tensor(SO3_to_quat(cam_direction.cpu().detach().numpy())).to(device=my_devices) # (3,3) to (4)

            # encode position: (H, W, N_sample, (2L+1)*C = 63) where L = 10, C = 3 of (x,y,z)
            cam_position_encoding = encode_position(cam_position, levels=10, inc_input=True) # (63)
            cam_position_encoding = torch.unsqueeze(cam_position_encoding, 0) # (1, 63)
            self.camera_position_encodings.append(cam_position_encoding)
            
            # encode position: (H, W, N_sample, (2L+1)*C = 36) where L = 4, C = 4 of quaternions (quat_x, quat_y, quat_z, quat_w) 
            cam_direction_encoding = encode_position(cam_direction, levels=4, inc_input=True) # (36)
            cam_direction_encoding = torch.unsqueeze(cam_direction_encoding, 0) # (1, 36)
            self.camera_direction_encodings.append(cam_direction_encoding)

        self.camera_position_encodings = torch.cat(self.camera_position_encodings, dim=0).to(device=my_devices)
        self.camera_direction_encodings = torch.cat(self.camera_direction_encodings, dim=0).to(device=my_devices)



if __name__ == '__main__':
    scene_name = 'LLFF/fern'
    use_ndc = True
    scene = DataLoaderWithSmartphone(base_dir='/your/data/path',
                                     scene_name=scene_name,
                                     data_type='train',
                                     res_ratio=8,
                                     num_img_to_load=-1,
                                     skip=1,
                                     use_ndc=use_ndc,
                                     percent_of_data_to_use=0.01)
