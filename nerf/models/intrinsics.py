import torch
import torch.nn as nn
import numpy as np

class CameraIntrinsicsModel(nn.Module):
    def __init__(self, H, W, focal_length_x, focal_length_y, n_cameras):
        super(CameraIntrinsicsModel, self).__init__()
        self.H = torch.tensor(H, requires_grad=False)
        self.W = torch.tensor(W, requires_grad=False)

        #focal_length_x = focal_length_x.cpu().detach().numpy()
        #focal_length_y = focal_length_y.cpu().detach().numpy()

        focal_length_x = focal_length_x.expand(n_cameras)
        focal_length_y = focal_length_y.expand(n_cameras)

        #coe_x = torch.tensor(torch.sqrt(focal_length_x / self.W), requires_grad=False)

        #coe_x = torch.sqrt(focal_length_x / self.W).to(torch.device('cuda:0'))        
        #coe_y = torch.tensor(torch.sqrt(focal_length_y / self.H), requires_grad=False)

        coe_x = torch.sqrt(focal_length_x / self.W)
        coe_y = torch.sqrt(focal_length_y / self.H)

        self.fx = nn.Parameter(coe_x, requires_grad=True)
        self.fy = nn.Parameter(coe_y, requires_grad=True)

    def forward(self, i=None): # explanation for i: reasons  
        focal_length_x = self.fx**2 * self.W
        focal_length_y = self.fy**2 * self.H
        
        return focal_length_x, focal_length_y
