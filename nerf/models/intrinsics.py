import torch
import torch.nn as nn
import numpy as np
from functorch.compile import make_boxed_func

class CameraIntrinsicsModel(nn.Module):

    def __init__(self, H, W, focal_length, n_cameras):

        super(CameraIntrinsicsModel, self).__init__()
        self.H = torch.tensor(H, requires_grad=False, device=torch.device('cuda:0'))
        self.W = torch.tensor(W, requires_grad=False, device=torch.device('cuda:0'))
        #self.H = torch.tensor(H, requires_grad=False)
        #self.W = torch.tensor(W, requires_grad=False)        
        #self.H = H
        #self.W = W
        
        focal_length = focal_length.expand(n_cameras)
        
        coe = torch.sqrt(focal_length / self.W)
        #coe = coe.to(torch.device('cuda:0'))
        self.fx = nn.Parameter(coe, requires_grad=True)          


    #def forward(self, i=None):
    def forward(self):
    
        #focal_length = self.fx**2 * self.W                
        #return focal_length        
            
        def f(i=None):
            focal_length = self.fx**2 * self.W                
            return focal_length
            
        return make_boxed_func(f)
        
        
