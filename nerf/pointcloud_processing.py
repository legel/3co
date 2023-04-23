import torch
#import torch._dynamo
import numpy as np
import open3d as o3d
import os
import point_cloud_utils as pcu
import sys
import open3d as o3d
from utils.training_utils import set_randomness

from learn import *

sys.path.append(os.path.join(sys.path[0], '../..'))

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

set_randomness()
torch.set_float32_matmul_precision('high')
#torch._dynamo.config.verbose=True      
torch._dynamo.config.suppress_errors = True

device = torch.device('cuda:0')         

# # set cache directory
os.environ['PYTORCH_KERNEL_CACHE_PATH'] = './'



def generate_point_clouds():

    dynamic_args = {
        "base_directory" : '\'./data/orchid\'',
        "number_of_samples_outward_per_raycast" : 720,
        "number_of_samples_outward_per_raycast_for_test_renders" : 720,
        "density_neural_network_parameters" : 256,
        "percentile_of_samples_in_near_region" : 0.8,
        "number_of_pixels_per_batch_for_test_renders" : 1000,   
        #"H_for_test_renders" : 1440,
        #"W_for_test_renders" : 1920,
        "H_for_test_renders" : 1440,
        "W_for_test_renders" : 1920,            
        "near_maximum_depth" : 0.5,
        "number_of_images_in_training_dataset" : 120,
        "number_of_test_images" : 120,
        "number_of_pixels_in_training_dataset" : 640 * 480 * 256,
        "skip_every_n_images_for_testing" : 1,
        "use_sparse_fine_rendering" : True,
        "pretrained_models_directory" : '\'./data/orchid/hyperparam_experiments/from_cloud/orchid_run204/models\'',        
        #"pretrained_models_directory" : '\'./data/cactus/hyperparam_experiments/from_cloud/cactus_run43/models\'',        
        "start_epoch" : 500001,
        "load_pretrained_models" : True,            
        "number_of_epochs" : 1,            
    }  
    with torch.no_grad():
        scene = SceneModel(args=parse_args(), experiment_args='dynamic', dynamic_args=dynamic_args)
    
    H = scene.args.H_for_test_renders
    W = scene.args.W_for_test_renders       


    epoch = scene.epoch - 1        
    for model in scene.models.values():
        model.eval()
    
    H = scene.args.H_for_test_renders
    W = scene.args.W_for_test_renders        
    
    all_focal_lengths = scene.models["focal"]()([0])
    
    for image_index in scene.test_image_indices:

        ground_truth_rgb_img, _ = scene.load_image_data(image_index * scene.skip_every_n_images_for_training)

        pp_x = scene.principal_point_x * (W / scene.W)
        pp_y = scene.principal_point_y * (H / scene.H)           

        # always render              
        print("Rendering for image {}".format(image_index))            
                    
        focal_lengths_for_this_img = all_focal_lengths[image_index].expand(H*W) * (H / scene.H)
        pixel_directions_for_this_image = compute_pixel_directions(
            focal_lengths_for_this_img, 
            scene.pixel_rows_for_test_renders, 
            scene.pixel_cols_for_test_renders, 
            pp_x, 
            pp_y
        )

        poses_for_this_img = scene.models['pose']()([0])[image_index].unsqueeze(0).expand(W*H, -1, -1)                            
        render_result = scene.render_prediction(poses_for_this_img, focal_lengths_for_this_img, pp_x, pp_y)            
        depth_fine = render_result['rendered_depth_fine']            
                            
        # save rendered rgb and depth images
        out_file_suffix = str(image_index)
        color_file_name_fine = os.path.join(scene.color_out_dir, str(out_file_suffix).zfill(4) + '_color_fine_{}.png'.format(epoch))
        depth_file_name_fine = os.path.join(scene.depth_out_dir, str(out_file_suffix).zfill(4) + '_depth_fine_{}.png'.format(epoch))                            
        if scene.args.use_sparse_fine_rendering:
            color_file_name_fine = os.path.join(scene.color_out_dir, str(out_file_suffix).zfill(4) + 'sparse_color_fine_{}.png'.format(epoch))
            depth_file_name_fine = os.path.join(scene.depth_out_dir, str(out_file_suffix).zfill(4) + 'sparse_depth_fine_{}.png'.format(epoch))                            
        
        scene.save_render_as_png(render_result, H, W, color_file_name_fine, depth_file_name_fine, None, None)
    
        print("Saving .ply for view {}".format(image_index))                        
        pose = scene.models['pose']()([0])[image_index].cpu()
        unsparse_rendered_rgb_img = None
        
        rendered_depth_img = render_result['rendered_depth_fine'].reshape(H, W).to(device=scene.device)
        rendered_rgb_img = render_result['rendered_image_fine'].reshape(H, W, 3).to(device=scene.device)
        
        if scene.args.use_sparse_fine_rendering:
            unsparse_rendered_rgb_img = render_result['rendered_image_unsparse_fine'].reshape(H, W, 3).cpu()
            unsparse_depth_img = render_result['depth_image_unsparse_fine'].reshape(H, W).cpu()
            color_file_name_unsparse_fine = os.path.join(scene.color_out_dir, str(out_file_suffix).zfill(4) + '_color_fine_{}.png'.format(epoch))
            depth_file_name_unsparse_fine = os.path.join(scene.depth_out_dir, str(out_file_suffix).zfill(4) + '_depth_fine_{}.png'.format(epoch))                                    
            render_result['rendered_image_fine'] = unsparse_rendered_rgb_img
            render_result['rendered_depth_fine'] = unsparse_depth_img
            scene.save_render_as_png(render_result, H, W, color_file_name_unsparse_fine, depth_file_name_unsparse_fine, None, None)
        else:
            unsparse_rendered_rgb_img = None
            unsparse_depth_img = None                

        depth_weights = render_result['depth_weights_coarse']
        entropy_coarse = -1 * torch.sum( (depth_weights+scene.args.epsilon) * torch.log2(depth_weights + scene.args.epsilon), dim=1)
        entropy_image = torch.zeros(H * W, 1)                                        
        entropy_image = entropy_coarse.reshape(H, W)                              
        
        # resize ground truth rgb to match output resolution
        ground_truth_rgb_img_resized = ground_truth_rgb_img.cpu().numpy()
        ground_truth_rgb_img_resized = cv2.resize(ground_truth_rgb_img_resized, (W, H), interpolation=cv2.INTER_LINEAR)
        ground_truth_rgb_img_resized = torch.from_numpy(ground_truth_rgb_img_resized)

        camera_world_position = pose[:3, 3].view(1, 1, 1, 3)     # (1, 1, 1, 3)
        camera_world_rotation = pose[:3, :3].view(1, 1, 1, 3, 3) # (1, 1, 1, 3, 3)     
        xyz_coordinates = derive_xyz_coordinates(camera_world_position, camera_world_rotation, pixel_directions_for_this_image.cpu(), depth_fine, flattened=True)
        xyz_coordinates = xyz_coordinates.reshape(H,W,3).cpu()

        pcd = generate_point_cloud(
            pose=pose.cpu(), 
            depth=rendered_depth_img.cpu(), 
            gt_rgb=ground_truth_rgb_img_resized.cpu(), 
            pixel_directions=pixel_directions_for_this_image.cpu(), 
            H = H,
            W = W,
            xyz_coordinates=xyz_coordinates,
            entropy_image=entropy_image.cpu(), 
            sparse_rendered_rgb_img=rendered_rgb_img, 
            unsparse_rendered_rgb_img=unsparse_rendered_rgb_img, 
            unsparse_depth=unsparse_depth_img,
            max_depth_filter_image=None,
            use_sparse_fine_rendering=scene.args.use_sparse_fine_rendering,
        )

        label="_{}_{}".format(epoch,image_index)        
        file_name = "{}/pointclouds/view_{}.ply".format(scene.experiment_dir, label)
        print(file_name)
        o3d.io.write_point_cloud(file_name, pcd, write_ascii = True)        


def density_query(scene, xyz, pixel_directions):

    ray_distance = torch.tensor([0.0, 0.001]).to(scene.device)
    #origin_xyz = xyz + ray_distance[1]
    origin_xyz = xyz
    origin_xyz = origin_xyz.unsqueeze(0)
    depth_xyzs = xyz.unsqueeze(0)
    pixel_directions = pixel_directions.unsqueeze(0)
    sampling_depths = ray_distance.unsqueeze(0)
    focal_lengths = scene.models['focal']()([0])[0].unsqueeze(0)
    pp_x = scene.principal_point_x
    pp_y = scene.principal_point_y

    features = encode_ipe(origin_xyz, depth_xyzs, pixel_directions, sampling_depths, focal_lengths_x=focal_lengths, principal_point_x=pp_x, principal_point_y=pp_y)    
    density, features = scene.models["geometry"]()([features])

    return density


def generate_point_cloud(
    pose, 
    depth, 
    gt_rgb, 
    pixel_directions, 
    H,
    W,
    xyz_coordinates,
    entropy_image=None, 
    unsparse_rendered_rgb_img=None, 
    sparse_rendered_rgb_img=None, 
    unsparse_depth=None, 
    max_depth_filter_image=None,    
    use_sparse_fine_rendering=True,
):        
    
    pixel_directions = pixel_directions.reshape(H, W, 3)        
    r = torch.sqrt(torch.sum(  (pixel_directions[H//8, W//8] - pixel_directions[H//2, W//2])**2, dim=0))

    if max_depth_filter_image == None:
        max_depth_filter_image = torch.ones(H,W)

    max_depth_condition = (max_depth_filter_image == 1).cpu()
    n_filtered_points = H * W - torch.sum(max_depth_condition.flatten())
    print("filtering {}/{} points with max depth condition".format(n_filtered_points, W*H))

    depth_condition = (depth != 0).cpu()
    n_filtered_points = H * W - torch.sum(depth_condition.flatten())
    print("filtering {}/{} points with depth=0 condition".format(n_filtered_points, W*H))

    #angle_condition = (torch.sqrt( torch.sum( (pixel_directions - pixel_directions[H//2, W//2, :])**2, dim=2 ) ) < r)
    angle_condition = (torch.sqrt( torch.sum( (pixel_directions - pixel_directions[H//2, W//2, :])**2, dim=2 ) ) < r*100.0)
    n_filtered_points = H * W - torch.sum(angle_condition.flatten())
    print("filtering {}/{} points with angle condition".format(n_filtered_points, W*H))
    
    #entropy_condition = (entropy_image < 2.0).cpu()
    entropy_condition = (entropy_image < 2.0).cpu()
    n_filtered_points = H * W - torch.sum(entropy_condition.flatten())
    print("filtering {}/{} points with entropy condition".format(n_filtered_points, W*H))

    joint_condition = torch.logical_and(max_depth_condition, depth_condition).cpu()
    joint_condition = torch.logical_and(joint_condition, angle_condition).cpu()
    joint_condition = torch.logical_and(joint_condition, entropy_condition).cpu()

    if use_sparse_fine_rendering:            
        sparse_depth = depth
        #sticker_condition = torch.abs(unsparse_depth - sparse_depth) < 0.005
        sticker_condition = torch.abs(unsparse_depth - sparse_depth) < 50.0
        joint_condition = torch.logical_and(joint_condition, sticker_condition)
        n_filtered_points = H * W - torch.sum(sticker_condition.flatten())
        print("filtering {}/{} points with sticker condition".format(n_filtered_points, W*H))            

    n_filtered_points = H * W - torch.sum(joint_condition.flatten())
    print("{}/{} total points filtered with intersection of conditions".format(n_filtered_points, W*H))        

    joint_condition_indices = torch.where(joint_condition)
    n_filtered_points = H * W - torch.sum(joint_condition.flatten())
    gt_rgb = gt_rgb.to(device=device)
    depth = depth[joint_condition_indices]
    gt_rgb = gt_rgb[joint_condition_indices]

    xyz_coordinates = xyz_coordinates[joint_condition_indices]
    
    pcd = create_point_cloud(xyz_coordinates, gt_rgb, flatten_xyz=False, flatten_image=False)

    return pcd             


def create_point_cloud(xyz_coordinates, colors, flatten_xyz=True, flatten_image=True):
    pcd = o3d.geometry.PointCloud()
    if flatten_xyz:
        xyz_coordinates = torch.flatten(xyz_coordinates, start_dim=0, end_dim=1).cpu().detach().numpy()
    else:
        xyz_coordinates = xyz_coordinates.cpu().detach().numpy()
    if flatten_image:
        colors = torch.flatten(colors, start_dim=0, end_dim=1).cpu().detach().numpy()
    else:
        colors = colors.cpu().detach().numpy()

    pcd.points = o3d.utility.Vector3dVector(xyz_coordinates)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd
    

def pc_to_plane_distances(pc, plane):

    epsilon = 0.000001
    i = torch.ones(pc.size()[0],3).to(device)                
    distances = torch.sqrt(torch.sum(  (pc * plane[:3].unsqueeze(0).expand(pc.size()[0],3) + plane[3:])**2,  dim=1))
    flat = torch.tensor([0.0, 1.0, 0.0], requires_grad=False).to(device=device)

    distances = distances + torch.sqrt(  torch.sum(  (plane[:3] - flat)**2,  dim=0) + epsilon )
    return distances

def merge_and_filter_by_center_radius(directory):

    filter_radius = 0.5
    pointclouds = []
    centers = []
    file_names = os.listdir(directory)
    for file_name in file_names:        
        if file_name.split('.')[-1] == 'ply':
            pc = o3d.io.read_point_cloud(directory + "/{}".format(file_name.split('/')[-1]))
            centers.append(pc.get_center())
    
    centers = torch.tensor(centers)
    avg_xyz = torch.sum(centers,dim=0) / len(centers)
    print('avg_xyz: ', avg_xyz)

    merged_points = []
    merged_colors = []

    for file_name in file_names:        
        if file_name.split('.')[-1] == 'ply':
            pc = o3d.io.read_point_cloud(directory + "/{}".format(file_name.split('/')[-1]))
            points = torch.tensor(np.asarray(pc.points))
            colors = torch.tensor(np.asarray(pc.colors))
            filter = torch.argwhere(torch.sqrt(torch.sum((points - avg_xyz)**2, dim=1)) < filter_radius).squeeze(1)
                                    
            filtered_points = points[filter]
            filtered_colors = colors[filter]            

            merged_points.append(filtered_points)
            merged_colors.append(filtered_colors)

    merged_points = torch.cat(merged_points, dim=0).cpu().detach().numpy()    
    merged_colors = torch.cat(merged_colors, dim=0).cpu().detach().numpy()
        
    pc.points = o3d.utility.Vector3dVector(merged_points)    
    pc.colors = o3d.utility.Vector3dVector(merged_colors)

    cl, idx = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)        
    pc.points = o3d.utility.Vector3dVector(merged_points[idx])    
    pc.colors = o3d.utility.Vector3dVector(merged_colors[idx])    

    f_out_name = '{}/{}'.format(directory, 'filtered_and_merged.ply')
    return pc


def remove_plane(pointcloud, threshold=0.005, init_n=3, iterations=1000):

    w, idx = pointcloud.segment_plane(threshold, init_n, iterations)

    points = np.asarray(pointcloud.points)    
    colors = np.asarray(pointcloud.colors)    
    
    points_below_plane = points[:, 1] <= w[3]    

    points = np.delete(points, idx,0)    
    colors = np.delete(colors, idx,0)
    
    points_below_plane = np.squeeze(np.argwhere( points[:, 1] <= -w[3] + 0.025), axis=1)        

    points = np.delete(points, points_below_plane,0)        
    colors = np.delete(colors, points_below_plane,0)    

    pointcloud.points = o3d.utility.Vector3dVector(np.squeeze(np.asarray([points]), axis=0))    
    pointcloud.colors = o3d.utility.Vector3dVector(np.squeeze(np.asarray([colors]), axis=0))
            
    return pointcloud


def poisson_disk_resampling(pointcloud, number_of_samples, radius=None):
    
    pc = pointcloud
    points = np.squeeze(np.asarray([pc.points]), axis=0)    
    colors = np.squeeze(np.asarray([pc.colors]), axis=0)
    idx = pcu.downsample_point_cloud_poisson_disk(points, num_samples=number_of_samples)
    pc.points = o3d.utility.Vector3dVector(points[idx])    
    pc.colors = o3d.utility.Vector3dVector(colors[idx])    

    #cl, idx = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)        
    pc.points = o3d.utility.Vector3dVector(points[idx])    
    pc.colors = o3d.utility.Vector3dVector(colors[idx])    

    return pc


if __name__ == '__main__':

    with torch.no_grad():
       generate_point_clouds()
       quit()

    #generate_point_clouds()

    path = '/home/rob/research_code/3co/research/nerf/data/dragon_scale/hyperparam_experiments/from_cloud/dragon_scale_run39/pointclouds/pointclouds_nofilter_highres'

    print('merging and filtering by center radius...')
    pc = merge_and_filter_by_center_radius(path)
    f_out_name = 'test/merged_and_filtered.ply'    
    o3d.io.write_point_cloud(f_out_name, pc)  

    print('removing plane...')
    pc = remove_plane(pc)
    f_out_name = 'test/plane_removed.ply'    
    o3d.io.write_point_cloud(f_out_name, pc)      

    print('performing poisson disk resampling...')    
    pc = poisson_disk_resampling(pc, int(len(pc.points)*0.1))
    f_out_name = 'test/downsampled.ply'    
    o3d.io.write_point_cloud(f_out_name, pc)      




