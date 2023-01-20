import torch
import torch.nn as nn
from utils.lie_group_helper import make_pose, make_c2w
import sys

class CameraPoseModel(nn.Module):
    def __init__(self, poses):
        super(CameraPoseModel, self).__init__()

        self.num_cams = poses.shape[0]
        #self.poses = nn.Parameter(poses.clone(), requires_grad=False).to(torch.device('cuda:0'))
        self.poses = poses.clone().to(torch.device('cuda:0'))

        
        #self.r = nn.Parameter(torch.ones(size=(self.num_cams, 3), dtype=torch.float32).to(torch.device('cuda:0')), requires_grad=True)  # (N, 3)
        self.r = nn.Parameter(torch.ones(size=(self.num_cams, 3), dtype=torch.float32).to(torch.device('cuda:0')), requires_grad=True)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(self.num_cams, 3), dtype=torch.float32).to(torch.device('cuda:0')), requires_grad=True)  # (N, 3)        

    def forward(self):
        r = self.r  # (N, 3) axis-angle
        t = self.t  # (N, 3)
        #delta_pose = make_pose(r, t)  # (N, 4, 4)

        poses = self.poses.clone()
        for i in range (self.num_cams):
            #p = make_c2w(r[i], t[i]) @ self.poses[i]                        
            poses[i] = make_c2w(r[i], t[i]) @ self.poses[i]


        
        #print("batch:")
        #print(delta_pose[0])
        #print(r[0])

        #print("original:")
        #print(  make_c2w(r[0], t[0]) )

        # learn a delta pose between init pose and target pose        
        #poses = delta_pose @ self.poses        
        

        



        return poses