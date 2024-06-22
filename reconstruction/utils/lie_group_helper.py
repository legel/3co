import numpy as np
import torch
from scipy.spatial.transform import Rotation as RotLib
from ahrs import Quaternion, DCM
from ahrs.utils import angular_distance
#from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix
# from pytorch3d.transforms.so3 import so3_relative_angle

import math
from typing import Tuple

import torch
import sys

DEFAULT_ACOS_BOUND: float = 1.0 - 1e-4



def create_from_axis_angle(xx, yy, zz, a):    
    # Here we calculate the sin( theta / 2) once for optimization
    factor = np.sin( a / 2.0 )

    # Calculate the x, y and z of the quaternion
    x = xx * factor
    y = yy * factor
    z = zz * factor

    # Calcualte the w value by cos( theta / 2 )
    w = np.cos( a / 2.0 )

    result = torch.tensor([x, y, z, w])
    return torch.nn.functional.normalize(result, p=2, dim=0)


def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor([[0,0,0,1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate([input, np.array([[0,0,0,1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output



def vec2skew_batch(v):
    """
    :param v:  (N, 3, ) torch tensor
    :return:   (N, 3, 3)
    """
    number_of_samples = v.shape[0]
    zero = torch.zeros((number_of_samples,1), dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([ zero,    -v[:,2:3],   v[:,1:2]], dim=1)  # (N, 3, 1)
    skew_v1 = torch.cat([ v[:,2:3],   zero,    -v[:,0:1]], dim=1)  # (N, 3, 1)
    skew_v2 = torch.cat([-v[:,1:2],   v[:,0:1],   zero], dim=1)    # (N, 3, 1)
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=1)       # (N, 3, 3)
    #skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=2)       # (N, 3, 3)
    
    
    #skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=1)       # (N, 3, 3)
    
    return skew_v  # (N, 3, 3)

def Exp_batch(r):
    """so(3) vector to SO(3) matrix
    :param r: (N, 3, ) axis-angle, torch tensor
    :return:  (N, 3, 3)
    """
    batch_size = r.shape[0]

    skew_r = vec2skew_batch(r)  # (N, 3, 3)
    norm_r = r.norm() + torch.tensor([1e-15]).to(torch.device('cuda:0'))

    eye = torch.eye(3, dtype=torch.float32, device=r.device)    
    batch_eye = eye.repeat(batch_size, 1, 1)
    a = skew_r @ skew_r

    R = batch_eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
        
    return R

def make_pose(r, t):
    """
    :param r:  (N, 3, ) axis-angle             torch tensor
    :param t:  (N, 3, ) translation vector     torch tensor
    :return:   (N, 4, 4)
    """
    
    R = Exp_batch(r)  # (N, 3, 3)    
    # R = axis_angle_to_matrix(r)
             
    # Note, Exp_batch(r) seems to be equivalent to pytorch3d's axis_angle_to_matrix(-r)
    pose = torch.cat([R, t.unsqueeze(2)], dim=2)  # (N, 3, 4)    
    pose = convert3x4_4x4(pose)  # (N, 4, 4)        

    return pose


def vec2skew(v):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    zero = torch.zeros(1, dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([ zero,    -v[2:3],   v[1:2]])  # (3, 1)
    skew_v1 = torch.cat([ v[2:3],   zero,    -v[0:1]])
    skew_v2 = torch.cat([-v[1:2],   v[0:1],   zero])
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
    return skew_v  # (3, 3)

def Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    skew_r = vec2skew(r)  # (3, 3)
    norm_r = r.norm() + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
    return R

def make_c2w(r, t):
    """
    :param r:  (3, ) axis-angle             torch tensor
    :param t:  (3, ) translation vector     torch tensor
    :return:   (4, 4)
    """
    R = Exp(r)  # (3, 3)
    c2w = torch.cat([R, t.unsqueeze(1)], dim=1)  # (3, 4)
    c2w = convert3x4_4x4(c2w)  # (4, 4)
    return c2w






