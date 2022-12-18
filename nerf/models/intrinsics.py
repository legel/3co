import torch
import torch.nn as nn
import numpy as np

class CameraIntrinsicsModel(nn.Module):
    def __init__(self, H, W, focal_length, n_cameras):
        super(CameraIntrinsicsModel, self).__init__()
        self.H = torch.tensor(H, requires_grad=False)
        self.W = torch.tensor(W, requires_grad=False)
        focal_length = focal_length.expand(n_cameras)
        coe = torch.sqrt(focal_length / self.W)#.to(torch.device('cuda:0'))        
        self.fx = nn.Parameter(coe, requires_grad=True)        

    def forward(self, i=None): # i is a dummy parameter 
        focal_length = self.fx**2 * self.W
                
        return focal_length
