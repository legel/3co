import torch
import torch.nn as nn
from utils.lie_group_helper import make_pose, make_c2w
import sys

class CameraPoseModel(nn.Module):
    def __init__(self, poses):
        super(CameraPoseModel, self).__init__()

        self.num_cams = poses.shape[0]
        self.poses = nn.Parameter(poses, requires_grad=False)
        self.r = nn.Parameter(torch.zeros(size=(self.num_cams, 3), dtype=torch.float32, device=torch.device('cuda:0')), requires_grad=True)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(self.num_cams, 3), dtype=torch.float32, device=torch.device('cuda:0')), requires_grad=True)  # (N, 3)        


    def forward(self, i=None):
        r = self.r  # (N, 3) axis-angle
        t = self.t  # (N, 3)        

        delta_pose = make_pose(r, t)  # (N, 4, 4)
        poses = delta_pose @ self.poses   
        
        #rotation_matrix[:, 2] = -rotation_matrix[:, 2]
        #rotation_matrix[2, :3] = rotation_matrix[2, :3]
        #translation_vector[2] = -translation_vector[2] 
        #poses[:, :, 2] = -poses[:, :, 2]
        #poses[:, 2, :3] = poses[:, 2]       
        return poses

        #c2w = make_c2w(r, t)
        #c2w = c2w @ self.poses[i]
        #return c2w        