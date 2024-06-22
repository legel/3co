import numpy as np
import torch
import open3d as o3d

def create_point_cloud(xyz, rgb):
    pcd = o3d.geometry.PointCloud()
    xyz = xyz.cpu().detach().numpy()
    rgb = rgb.cpu().detach().numpy()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals()
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd

view_numbers = [0]
top_n_candidates = 5

for view_number in view_numbers:

    rgbdxyz = torch.from_numpy(np.load("rgbdxyz_{}.npy".format(view_number))).to(device=torch.device('cuda:0'))

    r = rgbdxyz[:,:,0] # (N_pixels, N_samples) = (307200, 1028)
    g = rgbdxyz[:,:,1] # (N_pixels, N_samples) = (307200, 1028)
    b = rgbdxyz[:,:,2] # (N_pixels, N_samples) = (307200, 1028)
    d = rgbdxyz[:,:,3] # (N_pixels, N_samples) = (307200, 1028)
    x = rgbdxyz[:,:,4] # (N_pixels, N_samples) = (307200, 1028)
    y = rgbdxyz[:,:,5] # (N_pixels, N_samples) = (307200, 1028)
    z = rgbdxyz[:,:,6] # (N_pixels, N_samples) = (307200, 1028)

    number_of_pixels = r.shape[0]
    number_of_samples = r.shape[1]

    print("Getting indices for sorting 300,000 vectors...")
    sorted_depth_weights = torch.argsort(d, dim=1, descending=True)

    all_rgb = []
    all_xyz = []

    for pixel_index in range(0, number_of_pixels, 5): # 
        pixel_index = pixel_index # 100
        
        # first, grab our "focus" geometry
        top_sample_indices = sorted_depth_weights[pixel_index,:top_n_candidates]

        # now, get the color we will render the final derived point 
        nerf_r = torch.sum(d[pixel_index,:] * r[pixel_index,:], dim=0) #.cpu().numpy()
        nerf_g = torch.sum(d[pixel_index,:] * g[pixel_index,:], dim=0) #.cpu().numpy()
        nerf_b = torch.sum(d[pixel_index,:] * b[pixel_index,:], dim=0) #.cpu().numpy()
        nerf_rgb = torch.stack([nerf_r, nerf_g, nerf_b], dim=0)
        all_rgb.append(nerf_rgb)

        # get the top weights and normalize them so that they sum to 1.0
        top_weights = d[pixel_index, top_sample_indices]
        normalized_top_weights = torch.nn.functional.normalize(top_weights, p=1, dim=0)

        # get the top (x,y,z) points
        top_x = x[pixel_index, top_sample_indices]
        top_y = y[pixel_index, top_sample_indices]
        top_z = z[pixel_index, top_sample_indices]

        # now, let's compute a final geometric point based on the normalized sum of the top (x,y,z) points that have been filtered
        final_top_x = torch.sum(top_x * normalized_top_weights, dim=0) #.cpu().numpy()
        final_top_y = torch.sum(top_y * normalized_top_weights, dim=0) #.cpu().numpy()
        final_top_z = torch.sum(top_z * normalized_top_weights, dim=0) #.cpu().numpy()

        final_xyz = torch.stack([final_top_x, final_top_y, final_top_z], dim=0)
        all_xyz.append(final_xyz)

        if pixel_index % 10000 == 0:
            print("(CLOUD {}) {} Indices: {}".format(view_number, pixel_index, top_sample_indices))
            print("(CLOUD {}) {} Weights: {}\n".format(view_number, pixel_index, d[pixel_index, top_sample_indices]))

    rgb = torch.stack(all_rgb, dim=0)
    xyz = torch.stack(all_xyz, dim=0)

    pcd = create_point_cloud(xyz=xyz, rgb=rgb)
    o3d.io.write_point_cloud("nerf_derived_geometry_{}_top{}.ply".format(view_number, top_n_candidates), pcd)


