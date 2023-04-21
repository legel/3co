from operator import ne
from pyexpat.errors import XML_ERROR_INCORRECT_ENCODING
import torch
import numpy as np
import matplotlib.pyplot as plt

def get_number_of_encoded_dimensions(number_of_fourier_frequencies):
    encoded_dimensions = (2 * number_of_fourier_frequencies + 1) * 3  # (2L + 1) * 3
    return encoded_dimensions

def encode_position(input, levels, inc_input=True):
    """
    For each scalar, we encode it using a series of sin() and cos() functions with different frequency.
        - With L pairs of sin/cos function, each scalar is encoded to a vector that has 2L elements. Concatenating with
          itself results in 2L+1 elements.
        - With C channels, we get C(2L+1) channels output.

    :param input:   (..., C)            torch.float32
    :param levels:  scalar L            int
    :return:        (..., C*(2L+1))     torch.float32
    """

    # this is already doing 'log_sampling' in the official code.
    result_list = [input] if inc_input else []
    for i in range(levels):
        temp = 2.0**i * input  # (..., C)
        result_list.append(torch.sin(temp))  # (..., C)
        result_list.append(torch.cos(temp))  # (..., C)

    result_list = torch.cat(result_list, dim=-1)  # (..., C*(2L+1)) The list has (2L+1) elements, with (..., C) shape each.
    return result_list  # (..., C*(2L+1))


def encode_ipe(origin_xyz, depth_xyzs, pixel_directions, sampling_depths, pixel_world_widths):
    
    N_pixels = sampling_depths.size(0)            
    
    inscribed_circle_radius_factor = 2.0 / torch.sqrt(torch.tensor([12.0])).to(torch.device('cuda:0'))
    radii = pixel_world_widths * inscribed_circle_radius_factor        
    
    # straddle copy of depth samples over itself, shifted by one index    
    t0 = sampling_depths[:, : -1]    
    t1 = torch.zeros(sampling_depths.size(0), sampling_depths.size(1)-1).to(torch.device('cuda:0'))
    t1[:, :] = sampling_depths[:, 1:]            

    # compute parameters of gaussian approximating conical frustrum and lift to world coordinates        
    means, covs = conical_frustum_to_gaussian(pixel_directions, t0, t1, radii[:,:])

    # offset mean by camera origin  
    x = means + origin_xyz.unsqueeze(1).expand(N_pixels, sampling_depths.size(1)-1, 3)
    x_cov_diag = covs        

    ##### compute expectation of fourier-encoded gaussian ####
    min_deg = 0  # 0 is default from mip-nerf code
    max_deg = 10  # 16 is default from mip-nerf code    
    scales = torch.tensor(np.array([2**i for i in range(min_deg, max_deg)])).to(torch.device('cuda:0'))
    shape = list(x.shape[:-1]) + [-1]    

    y = torch.reshape(x[..., None, :] * scales[:,None], shape)    
    y_var = torch.reshape(x_cov_diag[..., None, :] * scales[:,None]**2, shape)        

    pi = torch.tensor(np.pi).to(torch.device('cuda:0'))    

    # compute final features: expectation of sine and cosine 
    features = expected_sin(
        torch.cat([y, y + 0.5 * pi], dim=-1), # cos(y) =  sin(y + pi/2)
        torch.cat([y_var] * 2, dim=-1)
    )        
    
    return features[0]    

    

def conical_frustum_to_gaussian(d, t0, t1, base_radius):
    """
        Approximate a conical frustum as a Gaussian distribution (mean+cov).
        Assumes the ray is originating from the origin, and base_radius is the
        radius at dist=1. Doesn't assume `d` is normalized.
        Args:
        d: jnp.float32 3-vector, the axis of the cone
        t0: float, the starting distance of the frustum.
        t1: float, the ending distance of the frustum.
        base_radius: float, the scale of the radius as a function of distance.
        Returns:
        a Gaussian (mean and covariance).
    """
    
    mu = (t0 + t1) / 2
    hw = (t1 - t0) / 2
    t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
    t_var = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) / (3 * mu**2 + hw**2)**2)    
    r_var = base_radius**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 * (hw**4) / (3 * mu**2 + hw**2))
        
    return lift_gaussian(d, t_mean, t_var, r_var)


# Lift a Gaussian defined along a ray to 3D coordinates.
def lift_gaussian(d, t_mean, t_var, r_var):
    
    mean = d[:, ..., None, :] * t_mean[:, ..., None]        

    epsilon = torch.tensor([1e-10]).to(torch.device('cuda:0'))

    d_mag_sq = torch.max(epsilon, torch.sum(d**2, dim=1)).unsqueeze(1)
    #d_mag_sq = torch.sum(d**2, dim=1).unsqueeze(1)
    d_outer_diag = d**2    
    null_outer_diag = 1 - d_outer_diag / d_mag_sq        
    t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]    
    xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
    cov_diag = t_cov_diag + xy_cov_diag    

    return mean, cov_diag


# Estimates mean and variance of sin(z), z ~ N(x, var).
def expected_sin(x, x_var):
    
    # When the variance is wide, shrink sin towards zero.
    y = torch.exp(-0.5 * x_var) * torch.sin(x) # safe_sin?
    #y_var = torch.max(torch.tensor([0.0]).to(torch.device('cuda:0')), 0.5 * (1 - torch.exp(-2 * x_var) * torch.cos(2 * x)) - y**2) # safe_cos?
    y_var = torch.clamp(0.5 * (1 - torch.exp(-2 * x_var) * torch.cos(2 * x)) - y**2, min=0) # safe_cos?
    return y, y_var
