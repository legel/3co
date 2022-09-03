import numpy as np
import torch
import torch.nn as nn
import open3d as o3d
from torch_cluster import grid_cluster
from utils.training_utils import PolynomialDecayLearningRate
import time
import sys
from PIL import Image
from pytorch3d.ops.knn import knn_points
import os
from os.path import exists

project_base_directory = "data/dragon_scale"
number_of_epochs = 500
device = torch.device('cuda:0') 

views = [v for v in range(0,240,2)]

load_pretrained_hypotheses = False

class DepthOptimizer(nn.Module):
    def __init__(self, all_xyz_slopes, all_xyz_intercepts, all_xyz_depths, all_depth_weights, all_adjacent_pixel_indices, all_indices_of_nonborder_pixels, all_rgb, all_cam_xyz, all_sensor_xyz):
        super(DepthOptimizer, self).__init__()
        self.current_step = 0
        self.xyz_slopes = nn.ParameterList()
        self.xyz_intercepts = nn.ParameterList()
        self.xyz_depths = nn.ParameterList()
        self.depth_weights = nn.ParameterList()       
        self.adjacent_pixel_indices = nn.ParameterList()
        self.indices_of_nonborder_pixels = nn.ParameterList()
        self.depth_hypotheses = nn.ParameterList()
        self.rgb = nn.ParameterList()
        self.cam_xyz = nn.ParameterList()
        self.sensor_xyz = nn.ParameterList()
        self.number_of_views = len(all_xyz_slopes)
        self.number_of_top_depth_weights = all_depth_weights[0].shape[1]

        print("Working with {} views containing a total of {} weights".format(self.number_of_views, self.number_of_top_depth_weights))

        for view in range(self.number_of_views):
            number_of_pixels_in_view = all_xyz_slopes[view].shape[0]

            # load all of our parameters needed for computing (x,y,z) coordinates for the line segments that we're going to shift depth hypotheses along
            xyz_slopes = nn.Parameter(torch.from_numpy(all_xyz_slopes[view]), requires_grad=False)			# (N_pixels, 3)
            xyz_intercepts = nn.Parameter(torch.from_numpy(all_xyz_intercepts[view]), requires_grad=False) 	# (N_pixels, 3)
            xyz_depths = nn.Parameter(torch.from_numpy(all_xyz_depths[view]), requires_grad=False) 			# (N_pixels, 3, N_top_depth_weights)
            depth_weights = nn.Parameter(torch.from_numpy(all_depth_weights[view]), requires_grad=False) 		# (N_pixels, N_top_depth_weights)
            adjacent_pixel_indices = nn.Parameter(torch.from_numpy(all_adjacent_pixel_indices[view]), requires_grad=False) # (N_pixels, 4)
            indices_of_nonborder_pixels = nn.Parameter(torch.from_numpy(all_indices_of_nonborder_pixels[view]), requires_grad=False) # (N_pixels_non_border)
            rgb = nn.Parameter(torch.from_numpy(all_rgb[view]), requires_grad=False)
            cam_xyz = nn.Parameter(torch.from_numpy(all_cam_xyz[view]), requires_grad=False)
            sensor_xyz = nn.Parameter(torch.from_numpy(all_sensor_xyz[view]), requires_grad=False)

            # prior to setting up our initial hypothesis, let's just initialize with the top NeRF weight distance
            nerf_best_estimate_ray_distances = self.compute_ray_distance_from_xyz(xyz_slope=xyz_slopes, xyz_intercept=xyz_intercepts, xyz_point=xyz_depths[:,:,0])

            # now, our actual list of parameters, which should be updated at each step, are defined here
            depth_hypotheses = nn.Parameter(nerf_best_estimate_ray_distances, requires_grad=True) # (N_pixels)

            if load_pretrained_hypotheses:
                depth_hypotheses = self.load_depth_hypotheses(view=view)

            # these parameters are all constant
            self.xyz_slopes.append(xyz_slopes)
            self.xyz_intercepts.append(xyz_intercepts)
            self.xyz_depths.append(xyz_depths)
            self.depth_weights.append(depth_weights)
            self.adjacent_pixel_indices.append(adjacent_pixel_indices)
            self.indices_of_nonborder_pixels.append(indices_of_nonborder_pixels)
            self.rgb.append(rgb)
            self.cam_xyz.append(cam_xyz)
            self.sensor_xyz.append(sensor_xyz)

            # these are the parameters that get updated from gradient descent
            self.depth_hypotheses.append(depth_hypotheses)

        self.xyz = [view for view in range(self.number_of_views)]

        for view in range(self.number_of_views):
            self.recompute_xyz_hypotheses(view)
        
    def recompute_xyz_hypotheses(self, view):
        xyz = self.xyz_slopes[view] * self.depth_hypotheses[view].unsqueeze(1).expand(-1,3) + self.xyz_intercepts[view]
        xyz = torch.nan_to_num(xyz, nan=0.0)
        self.xyz[view] = xyz

    def load_depth_hypotheses(self, view, epoch=99):
        import_file = "{}/depth_hypotheses_epoch_{}_for_view_{}.npy".format(project_base_directory, view, epoch)
        print("Loading pretrained hypotheses for view {} from {}".format(view, import_file))
        depth_hypotheses = torch.from_numpy(np.load(import_file))
        depth_hypotheses = nn.Parameter(depth_hypotheses, requires_grad=True)
        return depth_hypotheses

    def save_depth_hypotheses(self, epoch):
        for view in range(self.number_of_views):
            depth_hypotheses = self.depth_hypotheses[view].cpu().detach().numpy()
            with open("{}/depth_hypotheses_epoch_{}_for_view_{}.npy".format(project_base_directory, epoch, view), "wb") as f:
                np.save(f, depth_hypotheses)

    def create_point_cloud(self, view, detach_xyz=True):
        xyz = self.xyz[view]
        rgb = self.rgb[view].cpu().detach().numpy()
        pcd = o3d.geometry.PointCloud()
        if detach_xyz:
            xyz = xyz.cpu().detach().numpy()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.estimate_normals()
        if type(rgb) != type(None):
            pcd.colors = o3d.utility.Vector3dVector(rgb)
        return pcd

    def create_polynomial_learning_rate_schedule(self, optimizer):
        schedule = PolynomialDecayLearningRate(optimizer=optimizer, 
                                               total_steps=number_of_epochs, 
                                               start_value=0.10, 
                                               end_value=0.0001, 
                                               exponential_index=6, 
                                               curvature_shape=1,
                                               model_type="mesh_optimizer",
                                               log_frequency=1)
        return schedule

    def compute_ray_distance_from_xyz(self, xyz_slope, xyz_intercept, xyz_point):
        # derive the t-value for each raycast, per-pixel, assumes slope is not flat for both x and y dimensions... (need to include z here as well for ultra-robustness)
        nerf_x_best_estimate_ray_distances = (xyz_point[:,0] - xyz_intercept[:,0]) / xyz_slope[:,0]
        nerf_y_best_estimate_ray_distances = (xyz_point[:,1] - xyz_intercept[:,1]) / xyz_slope[:,1]
        flat_x_slope_indices = torch.argwhere(xyz_slope[:,0] == 0.0)
        flat_y_slope_indices = torch.argwhere(xyz_slope[:,1] == 0.0)
        nerf_best_estimate_ray_distances = nerf_x_best_estimate_ray_distances
        nerf_best_estimate_ray_distances[flat_x_slope_indices] = nerf_y_best_estimate_ray_distances[flat_x_slope_indices]
        return torch.abs(nerf_best_estimate_ray_distances)

    def compute_depthwise_weighted_loss(self, view):
        xyz = self.xyz[view]                     # (N_pixels, 3)
        xyz_depths = self.xyz_depths[view]       # (N_pixels, 3, N_top_depth_weights)
        depth_weights = self.depth_weights[view] # (N_pixels, N_top_depth_weights)
        number_of_pixels_for_view = xyz.shape[0]
        expanded_xyz = xyz.unsqueeze(2).expand(-1, -1, self.number_of_top_depth_weights) # (N_pixels, 3, N_top_depth_weights)
        distance_between_hypothesis_and_depths = torch.sum((expanded_xyz * 1000 - xyz_depths * 1000) ** 2, dim=1) # (N_pixels, N_top_depth_weights)
        distance_weighted_by_neural_network = distance_between_hypothesis_and_depths * depth_weights # (N_pixels, N_top_depth_weights)
        average_distance_weighted_by_neural_network = torch.sum(distance_weighted_by_neural_network) / (number_of_pixels_for_view * self.number_of_top_depth_weights) # (1)
        depthwise_weighted_loss = average_distance_weighted_by_neural_network
        return depthwise_weighted_loss #/ self.number_of_views

    def compute_sensor_depth_deviation_loss(self, view):
        hypothesis_xyz = self.xyz[view] # (N_pixels, 3)
        sensor_xyz = self.sensor_xyz[view] # (N_pixels, 3)
        number_of_pixels_for_view = hypothesis_xyz.shape[0]
        distance_between_hypothesis_and_sensor_depths = torch.sum((hypothesis_xyz * 1000 - sensor_xyz * 1000) ** 2, dim=1) # (N_pixels)
        average_distance_between_hypothesis_and_sensor_depths = torch.sum(distance_between_hypothesis_and_sensor_depths) / number_of_pixels_for_view
        return average_distance_between_hypothesis_and_sensor_depths

    def compute_normal_angle(self, a, b, c):
        # a, b, c are all (x,y,z) coordinates, the angle returned is between them ABC
        ba = a - b
        bc = c - b
        cosine_numerator = torch.sum(ba*bc, axis=1)
        cosine_denominator_1 = torch.linalg.norm(ba, axis=1) + 0.0000001
        cosine_denominator_2 = torch.linalg.norm(bc, axis=1) + 0.0000001
        cosine_angle = (cosine_numerator + 0.0000001) / ((cosine_denominator_1 * cosine_denominator_2) + 0.0000001)
        angles = torch.arccos(cosine_angle)
        degree_angles = torch.rad2deg(angles)
        # degrees_from_perpendicular = (degree_angles - 90.0)**2
        # no_nansense_degrees = torch.nan_to_num(degree_angles, nan=90.0)
        return degree_angles

    def compute_neighbor_normal_loss(self, view):
        xyz = self.xyz[view] # (N_pixels, 3)
        adjacent_pixels = self.adjacent_pixel_indices[view] # (N_pixels_nonborder, 4)
        indices_of_nonborder_pixels = self.indices_of_nonborder_pixels[view] # (N_pixels_nonborder)
        nonborder_xyz = xyz[indices_of_nonborder_pixels] # (N_pixels_nonborder, 3)
        number_of_nonborder_pixels = nonborder_xyz.shape[0]
        top_pixel_xyz = xyz[adjacent_pixels[:,0]] # (N_pixels_nonborder, 3)
        bottom_pixel_xyz = xyz[adjacent_pixels[:,1]] # (N_pixels_nonborder, 3)
        left_pixel_xyz = xyz[adjacent_pixels[:,2]] # (N_pixels_nonborder, 3)
        right_pixel_xyz = xyz[adjacent_pixels[:,3]] # (N_pixels_nonborder, 3)
        cam_xyz = self.cam_xyz[view] # (3)
        cam_xyz = torch.unsqueeze(cam_xyz, dim=0).expand(number_of_nonborder_pixels,3) # (N_pixels_nonborder, 3)
        top_pixel_angle = self.compute_normal_angle(a=cam_xyz, b=nonborder_xyz, c=top_pixel_xyz)
        bottom_pixel_angle = self.compute_normal_angle(a=cam_xyz, b=nonborder_xyz, c=bottom_pixel_xyz)
        left_pixel_angle = self.compute_normal_angle(a=cam_xyz, b=nonborder_xyz, c=left_pixel_xyz)
        right_pixel_angle = self.compute_normal_angle(a=cam_xyz, b=nonborder_xyz, c=right_pixel_xyz)
        top_bottom_angle_distances = (top_pixel_angle + bottom_pixel_angle - 180.0)**2 
        left_right_angle_distances = (left_pixel_angle + right_pixel_angle - 180.0)**2
        average_pixel_to_pixel_angle_deviation = torch.sum((top_bottom_angle_distances + left_right_angle_distances) / 2) / number_of_nonborder_pixels
        normal_angle_deviation_neighbor_loss = average_pixel_to_pixel_angle_deviation
        return normal_angle_deviation_neighbor_loss #/ self.number_of_views

    def compute_pixelwise_neighbor_loss(self, view):
        xyz = self.xyz[view] # (N_pixels, 3)
        adjacent_pixels = self.adjacent_pixel_indices[view] # (N_pixels_nonborder, 4)
        indices_of_nonborder_pixels = self.indices_of_nonborder_pixels[view] # (N_pixels_nonborder)
        nonborder_xyz = xyz[indices_of_nonborder_pixels] # (N_pixels_nonborder, 3)
        number_of_nonborder_pixels = nonborder_xyz.shape[0]
        top_pixel_xyz = xyz[adjacent_pixels[:,0]] # (N_pixels_nonborder, 3)
        bottom_pixel_xyz = xyz[adjacent_pixels[:,1]] # (N_pixels_nonborder, 3)
        left_pixel_xyz = xyz[adjacent_pixels[:,2]] # (N_pixels_nonborder, 3)
        right_pixel_xyz = xyz[adjacent_pixels[:,3]] # (N_pixels_nonborder, 3)
        distance_top = torch.sum((top_pixel_xyz - nonborder_xyz) ** 2, dim=1) # (N_pixels_nonborder)
        distance_bottom = torch.sum((bottom_pixel_xyz - nonborder_xyz) ** 2, dim=1) # (N_pixels_nonborder)
        distance_left = torch.sum((left_pixel_xyz - nonborder_xyz) ** 2, dim=1) # (N_pixels_nonborder)
        distance_right = torch.sum((right_pixel_xyz - nonborder_xyz) ** 2, dim=1) # (N_pixels_nonborder)
        combined_pixelwise_neighbor_distances = distance_top + distance_bottom + distance_left + distance_right # (N_pixels_nonborder)
        # combined_pixelwise_neighbor_distances = torch.nan_to_num(combined_pixelwise_neighbor_distances, nan=0.1)
        average_distance_to_neighboring_pixel = torch.sum(combined_pixelwise_neighbor_distances) / (number_of_nonborder_pixels * 4) # (1)
        pixelwise_neighbor_loss = average_distance_to_neighboring_pixel
        return pixelwise_neighbor_loss #/ self.number_of_views

    def compute_multiview_geometry_loss(self, view):
        voxelwise_geometry_multiview_loss = torch.tensor(0.0).to(device=device)
        comparisons = 2
        comparison_views = []
        for comparison_view in [view-1, view+1]:
            if comparison_view == view:
                continue
            if comparison_view >= 0 and comparison_view <= self.number_of_views - 1:
                comparison_views.append(comparison_view)
        for comparison_view in comparison_views:
            # get colors and (x,y,z) coordinates for other views
            other_view_xyz = self.xyz[comparison_view] # (N_pixels, 3)
            total_number_of_other_points = other_view_xyz.shape[0]
            xyz = self.xyz[view] # (N_pixels, 3)
            number_of_points_in_this_view = xyz.shape[0]
            distances, indices, nn = knn_points(p1=torch.unsqueeze(xyz, dim=0), p2=torch.unsqueeze(other_view_xyz, dim=0), K=1)
            distances = distances[0,:,0]
            sum_of_distances_to_nearest_neighbors = (torch.sum(distances) / number_of_points_in_this_view)
            voxelwise_geometry_multiview_loss += sum_of_distances_to_nearest_neighbors
        return voxelwise_geometry_multiview_loss / comparisons

    def forward(self, view):
        depthwise_weighted_loss = self.compute_depthwise_weighted_loss(view)
        pixelwise_neighbor_loss = self.compute_pixelwise_neighbor_loss(view)
        voxelwise_multiview_geometry_loss = self.compute_multiview_geometry_loss(view)
        sensor_depth_deviation_loss = self.compute_sensor_depth_deviation_loss(view)
        self.current_step += 1
        return depthwise_weighted_loss, pixelwise_neighbor_loss, voxelwise_multiview_geometry_loss, sensor_depth_deviation_loss


def save_adjacent_4_pixels_for_every_index_plus_rgb_per_pixel(view_number=0, width=640, height=480, rgb_and_indices=True):
    print("Pre-computing indices for view {} (one time, then in the future not needed)...".format(view_number))

    sensor_depths = np.load('{}/raw_sensor_with_learned_poses_intrinsics_{}_xyz_raw.npy'.format(project_base_directory, view_number))
    pixel_rows = np.load('{}/selected_pixel_rows_{}.npy'.format(project_base_directory, view_number))
    pixel_cols = np.load('{}/selected_pixel_cols_{}.npy'.format(project_base_directory, view_number))
    rgb_image =  np.asarray(Image.open("{}/mask_for_filtering_{}.png".format(project_base_directory, view))) / 255 # (H, W, 3)

    indices = np.arange(width * height)
    indices_as_image = np.reshape(indices, newshape=(height,width))

    all_adjacent_pixel_indices = []
    indices_of_nonborder_pixels = []

    # create pixel_rows and pixel_cols dictionaries for fast look-up
    pixel_rows_dict = {}
    for pixel_row in pixel_rows:
        pixel_rows_dict[pixel_row] = 1

    pixel_cols_dict = {}
    for pixel_col in pixel_cols:
        pixel_cols_dict[pixel_col] = 1

    good_pixels = {}
    global_index = 0
    for pixel_row, pixel_col in zip(pixel_rows, pixel_cols):
        good_global_index = indices_as_image[pixel_row,pixel_col]
        good_pixels[good_global_index] = True

    global_to_local_index = {}
    for local_index, (pixel_row, pixel_col) in enumerate(zip(pixel_rows, pixel_cols)):
        global_index = indices_as_image[pixel_row,pixel_col]
        global_to_local_index[global_index] = local_index

    number_of_pixels = len(pixel_rows)
    all_pixel_rgb = []
    all_sensor_depths = []
    for local_index_count, (pixel_row, pixel_col) in enumerate(zip(pixel_rows, pixel_cols)):
        pixel_rgb = rgb_image[pixel_row, pixel_col, :]
        sensor_depth = sensor_depths[pixel_row, pixel_col, :]
        all_sensor_depths.append(sensor_depth)
        all_pixel_rgb.append(pixel_rgb)

        if rgb_and_indices:
            global_index = indices_as_image[pixel_row,pixel_col]
            # if this is not a border pixel, i.e. our loss metric will only compute on inside pixels
            if pixel_col not in [0,width-1] and pixel_row not in [0,height-1] and local_index_count not in [0,number_of_pixels-1]:

                top_pixel_index = indices_as_image[pixel_row-1,pixel_col]
                bot_pixel_index = indices_as_image[pixel_row+1,pixel_col]
                lef_pixel_index = indices_as_image[pixel_row,pixel_col-1]
                rig_pixel_index = indices_as_image[pixel_row,pixel_col+1]

                # if the neighboring pixels actually have values in our dataset
                if good_pixels.get(top_pixel_index, False) and good_pixels.get(bot_pixel_index, False) and good_pixels.get(lef_pixel_index, False) and good_pixels.get(rig_pixel_index, False):
                    # then we must get the 4 global pixel indices of those pixels that are adjacent in image space
                    # we need to first put the global pixel index (for all pixels) into a local pixel index (with cropped)
                    local_top_index = global_to_local_index[top_pixel_index]
                    local_bot_index = global_to_local_index[bot_pixel_index]
                    local_lef_index = global_to_local_index[lef_pixel_index]
                    local_rig_index = global_to_local_index[rig_pixel_index]

                    adjacent_pixel_indices = np.asarray([local_top_index, local_bot_index, local_lef_index, local_rig_index])
                    all_adjacent_pixel_indices.append(adjacent_pixel_indices)

                    indices_of_nonborder_pixels.append(local_index_count)

    all_pixel_rgb = np.asarray(all_pixel_rgb)
    all_sensor_depths = np.asarray(all_sensor_depths)

    if rgb_and_indices:
        all_adjacent_pixel_indices = np.asarray(all_adjacent_pixel_indices)

    if rgb_and_indices:
        with open("{}/adjacent_pixel_indices_for_view_{}.npy".format(project_base_directory, view_number), "wb") as f:
            np.save(f, all_adjacent_pixel_indices)

        with open("{}/indices_of_nonborder_pixels_for_view_{}.npy".format(project_base_directory, view_number), "wb") as f:
            np.save(f, indices_of_nonborder_pixels)

    with open("{}/flattened_pixel_rgb_for_view_{}.npy".format(project_base_directory, view_number), "wb") as f:
        np.save(f, all_pixel_rgb)

    with open("{}/flattened_pixel_sensor_depth_for_view_{}.npy".format(project_base_directory, view_number), "wb") as f:
        np.save(f, all_sensor_depths)


if __name__ == "__main__":
    all_adjacent_pixel_indices = []
    all_xyz_depths = []
    all_xyz_slopes = []
    all_xyz_intercepts = []
    all_depth_weights = []
    all_rgb = []
    all_cam_xyz = []
    all_indices_of_nonborder_pixels = []
    all_sensor_xyz = []

    print("Loading data for optimization of depths across views {}".format(views))
    for view in views:
        # pre-compute indices for faster training
        file_to_probe_for = "{}/adjacent_pixel_indices_for_view_{}.npy".format(project_base_directory, view)
        if not exists(file_to_probe_for):
            save_adjacent_4_pixels_for_every_index_plus_rgb_per_pixel(view_number=view, width=640, height=480, rgb_and_indices=True)

        # load data from export files
        adjacent_pixel_indices = np.load("{}/adjacent_pixel_indices_for_view_{}.npy".format(project_base_directory, view)) 
        xyz_depths = np.load("{}/xyz_depths_view_{}.npy".format(project_base_directory, view)) 
        xyz_slopes = np.load("{}/xyz_slopes_view_{}.npy".format(project_base_directory, view)) 
        xyz_intercepts = np.load("{}/xyz_intercepts_view_{}.npy".format(project_base_directory, view)) 
        depth_weights = np.load("{}/depth_weights_view_{}.npy".format(project_base_directory, view)) 
        cam_xyz = np.load("{}/cam_xyz_{}.npy".format(project_base_directory, view)) 
        rgb = np.load("{}/flattened_pixel_rgb_for_view_{}.npy".format(project_base_directory, view)) 
        sensor_xyz = np.load("{}/flattened_pixel_sensor_depth_for_view_{}.npy".format(project_base_directory, view)) 
        indices_of_nonborder_pixels = np.load("{}/indices_of_nonborder_pixels_for_view_{}.npy".format(project_base_directory, view)) 
        all_adjacent_pixel_indices.append(adjacent_pixel_indices)
        all_xyz_depths.append(xyz_depths)
        all_xyz_slopes.append(xyz_slopes)
        all_xyz_intercepts.append(xyz_intercepts)
        all_depth_weights.append(depth_weights)
        all_rgb.append(rgb)
        all_indices_of_nonborder_pixels.append(indices_of_nonborder_pixels)
        all_cam_xyz.append(cam_xyz)
        all_sensor_xyz.append(sensor_xyz)

    # load differentiable depth optimizer model
    model = DepthOptimizer( all_xyz_slopes, 
                            all_xyz_intercepts, 
                            all_xyz_depths, 
                            all_depth_weights, 
                            all_adjacent_pixel_indices, 
                            all_indices_of_nonborder_pixels, 
                            all_rgb, 
                            all_cam_xyz,
                            all_sensor_xyz)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = model.create_polynomial_learning_rate_schedule(optimizer=optimizer)
    model = model.to(device=device)

    for i,view in enumerate(views):
        model.recompute_xyz_hypotheses(i)

    start_time = int(time.time())
    for epoch in range(number_of_epochs):
        number_of_views = len(views)

        # make sure model is learning
        model.train()

        all_depthwise_weighted_loss = torch.tensor(0.0).to(device=device)
        all_pixelwise_neighbor_loss = torch.tensor(0.0).to(device=device)
        all_voxelwise_multiview_geometry_loss = torch.tensor(0.0).to(device=device)
        all_sensor_depth_deviation_loss = torch.tensor(0.0).to(device=device)

        for i,view in enumerate(views):
            # get latest loss from mesh model hypotheses
            depthwise_weighted_loss, pixelwise_neighbor_loss, voxelwise_multiview_geometry_loss, sensor_depth_deviation_loss = model(view=i) # voxelwise_multiview_color_loss
            all_depthwise_weighted_loss += depthwise_weighted_loss
            all_pixelwise_neighbor_loss += pixelwise_neighbor_loss
            all_voxelwise_multiview_geometry_loss += voxelwise_multiview_geometry_loss
            all_sensor_depth_deviation_loss += sensor_depth_deviation_loss

        loss = all_depthwise_weighted_loss / 10 + all_pixelwise_neighbor_loss * 1000 + all_voxelwise_multiview_geometry_loss * 100000 + all_sensor_depth_deviation_loss / 10000
        
        minutes_into_experiment = (int(time.time())-int(start_time)) / 60
        if epoch > 0:
            print("({} at {:.2f} min.) - LOSSES: Total = {:.6f}, NeRF Depth = {:.6f}, Sensor Depth = {:.6f}, Pixel Eucl. = {:.6f}, Multiview Geom. = {:.8f}".format(epoch,
                                                                                                                                minutes_into_experiment, 
                                                                                                                                loss,
                                                                                                                                all_depthwise_weighted_loss / 10,
                                                                                                                                all_sensor_depth_deviation_loss / 10000,
                                                                                                                                all_pixelwise_neighbor_loss * 1000,
                                                                                                                                all_voxelwise_multiview_geometry_loss * 100000,
                                                                                                                                ))

        # backpropagate gradients using loss
        loss.backward() 

        # recompute hypotheses (x,y,z) coordinates from current depth values
        for i,view in enumerate(views):
            model.recompute_xyz_hypotheses(i)

        # take a step forward in optimizer
        optimizer.step()
        # reset gradients
        optimizer.zero_grad()
        # step forward with learning rate scheduler
        scheduler.step()

        # # save point clouds
        # if epoch % 100 == 0 and epoch != 0:
        #     print("Saving point clouds")
        #     for i,view in enumerate(views):
        #         pcd = model.create_point_cloud(view=i)
        #         o3d.io.write_point_cloud("per_view_mesh_{}_with_{}_views_{}_epoch.ply".format(view, len(views), epoch), pcd)

        # save hypotheses in case future training is desired after program stop
        if epoch % 99 == 0 and epoch != 0:                
            print("Saving depth_hypotheses")
            model.save_depth_hypotheses(epoch)

    # save point clouds and NumPy files with data used for next step of mesh extraction
    print("Finished optimization. Now saving point clouds to .ply files.")
    for i,view in enumerate(views):
        xyz = model.xyz[i]
        rgb = all_rgb[i]

        with open("{}/per_view_optimized_xyz_for_view_{}.npy".format(project_base_directory, view), "wb") as f:
            np.save(f, xyz.cpu().detach().numpy())

        with open("{}/per_view_optimized_rgb_for_view_{}.npy".format(project_base_directory, view), "wb") as f:
            np.save(f, rgb)

        pcd = model.create_point_cloud(view=i)
        o3d.io.write_point_cloud("{}/per_view_mesh_{}_with_{}_views.ply".format(project_base_directory, view, len(views)), pcd)