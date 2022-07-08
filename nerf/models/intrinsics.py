import torch
import torch.nn as nn
import numpy as np

class CameraIntrinsicsModel(nn.Module):
    def __init__(self, H, W, focal_length_x, focal_length_y):
        super(CameraIntrinsicsModel, self).__init__()
        self.H = torch.tensor(H, requires_grad=False)
        self.W = torch.tensor(W, requires_grad=False)

        focal_length_x = focal_length_x.cpu().detach().numpy()
        focal_length_y = focal_length_y.cpu().detach().numpy()

        coe_x = torch.tensor(np.sqrt(focal_length_x / float(W)), requires_grad=False).float()
        coe_y = torch.tensor(np.sqrt(focal_length_y / float(H)), requires_grad=False).float()

        self.fx = nn.Parameter(coe_x, requires_grad=True)
        self.fy = nn.Parameter(coe_y, requires_grad=True)

    def forward(self, i=None):  # the i=None is just to enable multi-gpu training
        focal_length_x = self.fx**2 * self.W
        focal_length_y = self.fy**2 * self.H
        
        return focal_length_x, focal_length_y
