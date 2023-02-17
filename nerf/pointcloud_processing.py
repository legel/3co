import torch
import torch._dynamo
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
torch._dynamo.config.verbose=True      

device = torch.device('cuda:0')         

# # set cache directory
os.environ['PYTORCH_KERNEL_CACHE_PATH'] = './'



def generate_point_clouds():

    dynamic_args = {
        "base_directory" : '\'./data/cactus\'',
        "number_of_samples_outward_per_raycast" : 360,
        "number_of_samples_outward_per_raycast_for_test_renders" : 360,
        "density_neural_network_parameters" : 256,
        "percentile_of_samples_in_near_region" : 0.8,
        "number_of_pixels_per_batch_for_test_renders" : 5000,            
        #"H_for_test_renders" : 1440,
        #"W_for_test_renders" : 1920,
        "H_for_test_renders" : 480,
        "W_for_test_renders" : 640,            
        "near_maximum_depth" : 0.5,
        "skip_every_n_images_for_training" : 60,
        "skip_every_n_images_for_testing" : 1,
        "use_sparse_fine_rendering" : True,
        #"pretrained_models_directory" : '\'./data/dragon_scale/hyperparam_experiments/from_cloud/dragon_scale_run39/models\'',        
        "pretrained_models_directory" : '\'./data/cactus/hyperparam_experiments/from_cloud/cactus_run43/models\'',        
        "start_epoch" : 255001,
        "load_pretrained_models" : True,            
        "number_of_epochs" : 1,    
        "number_of_test_images" : 500,        
    }  

    
    scene = SceneModel(args=parse_args(), experiment_args='dynamic', dynamic_args=dynamic_args)

    epoch = scene.epoch - 1        
    for model in scene.models.values():
        model.eval()
    
    H = scene.args.H_for_test_renders
    W = scene.args.W_for_test_renders        
    
    all_focal_lengths = scene.models["focal"](0)

    test_image_indices = scene.test_image_indices
    sub_test_image_indices = test_image_indices
    # sub_test_image_indices = [0,159,222]        
    for image_index in [scene.test_image_indices[i] for i in sub_test_image_indices]:        
                                
        pixel_indices = torch.argwhere(scene.image_ids_per_pixel == image_index)                                    
        ground_truth_rgb_img, image_name = scene.load_image_data(image_index * scene.args.skip_every_n_images_for_training)                        
        #sensor_depth, near_bound, far_bound, confidence = scene.load_depth_data(image_index) # (H, W)                        
                                        
        pixel_rows = scene.pixel_rows[pixel_indices]
        pixel_cols = scene.pixel_cols[pixel_indices]        

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

        poses_for_this_img = scene.models['pose'](0)[image_index].unsqueeze(0).expand(W*H, -1, -1)                            
        render_result = scene.render_prediction(poses_for_this_img, focal_lengths_for_this_img, H, W, pp_x, pp_y)            
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
        pointcloud_file_name = os.path.join(scene.pointcloud_out_dir, str(out_file_suffix).zfill(4) + '_depth_view_{}.png'.format(epoch))
        pose = scene.models['pose'](0)[image_index].cpu()
        unsparse_rendered_rgb_img = None
        
        rendered_depth_img = render_result['rendered_depth_fine'].reshape(H, W).to(device=scene.device)
        rendered_rgb_img = render_result['rendered_image_fine'].reshape(H, W, 3).to(device=scene.device)
        if scene.args.use_sparse_fine_rendering:
            unsparse_rendered_rgb_img = render_result['rendered_image_unsparse_fine'].reshape(H, W, 3).to(device=scene.device)
            unsparse_depth_img = render_result['depth_image_unsparse_fine'].reshape(H, W).to(device=scene.device)
            color_file_name_unsparse_fine = os.path.join(scene.color_out_dir, str(out_file_suffix).zfill(4) + '_color_fine_{}.png'.format(epoch))
            depth_file_name_unsparse_fine = os.path.join(scene.depth_out_dir, str(out_file_suffix).zfill(4) + '_depth_fine_{}.png'.format(epoch))                                    
            render_result['rendered_image_fine'] = unsparse_rendered_rgb_img
            render_result['rendered_depth_fine'] = unsparse_depth_img
            scene.save_render_as_png(render_result, H, W, color_file_name_unsparse_fine, depth_file_name_unsparse_fine, None, None)                        
        else:
            unsparse_rendered_rgb_img = None
            unsparse_depth_img = None                

        entropy_image = torch.zeros(H * W, 1)                                
        entropy_coarse = render_result['entropy_coarse']            
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
            sparse_rendered_rgb_img=rendered_rgb_img.cpu(), 
            unsparse_rendered_rgb_img=unsparse_rendered_rgb_img.cpu(), 
            unsparse_depth=unsparse_depth_img.cpu(),
            max_depth_filter_image=None,
            use_sparse_fine_rendering=True
        )

        label="_{}_{}".format(epoch,image_index)
        file_name = "{}/pointclouds/view_{}.ply".format(scene.experiment_dir, label)
        o3d.io.write_point_cloud(file_name, pcd, write_ascii = True)        


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
    use_sparse_fine_rendering=True
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

    angle_condition = (torch.sqrt( torch.sum( (pixel_directions - pixel_directions[H//2, W//2, :])**2, dim=2 ) ) < r)
    n_filtered_points = H * W - torch.sum(angle_condition.flatten())
    print("filtering {}/{} points with angle condition".format(n_filtered_points, W*H))
    
    entropy_condition = (entropy_image < 2.0).cpu()
    n_filtered_points = H * W - torch.sum(entropy_condition.flatten())
    print("filtering {}/{} points with entropy condition".format(n_filtered_points, W*H))

    joint_condition = torch.logical_and(max_depth_condition, depth_condition).cpu()
    joint_condition = torch.logical_and(joint_condition, angle_condition).cpu()
    joint_condition = torch.logical_and(joint_condition, entropy_condition).cpu()

    if use_sparse_fine_rendering:            
        sparse_depth = depth
        sticker_condition = torch.abs(unsparse_depth - sparse_depth) < 0.005
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

    normals = torch.zeros(H, W, 3)
    normals = pose[:3, 3] - xyz_coordinates
    normals = torch.nn.functional.normalize(normals, p=2, dim=2)        
    normals = normals[joint_condition_indices]

    xyz_coordinates = xyz_coordinates[joint_condition_indices]
    
    pcd = create_point_cloud(xyz_coordinates, gt_rgb, normals, flatten_xyz=False, flatten_image=False)

    return pcd             


def create_point_cloud(xyz_coordinates, colors, normals=None, flatten_xyz=True, flatten_image=True):
    pcd = o3d.geometry.PointCloud()
    if flatten_xyz:
        xyz_coordinates = torch.flatten(xyz_coordinates, start_dim=0, end_dim=1).cpu().detach().numpy()
    else:
        xyz_coordinates = xyz_coordinates.cpu().detach().numpy()
    if flatten_image:
        colors = torch.flatten(colors, start_dim=0, end_dim=1).cpu().detach().numpy()
        if normals is not None:
            normals = torch.flatten(normals, start_dim=0, end_dim=1).cpu().detach().numpy()
    else:
        colors = colors.cpu().detach().numpy()
        if normals is not None:
            normals = normals.cpu().detach().numpy()

    pcd.points = o3d.utility.Vector3dVector(xyz_coordinates)

    # force open3d to include normals data structure
    pcd.estimate_normals()
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)

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
    merged_normals = []
    merged_colors = []

    for file_name in file_names:        
        if file_name.split('.')[-1] == 'ply':
            pc = o3d.io.read_point_cloud(directory + "/{}".format(file_name.split('/')[-1]))
            points = torch.tensor(np.asarray(pc.points))
            normals = torch.tensor(np.asarray(pc.normals))
            colors = torch.tensor(np.asarray(pc.colors))
            filter = torch.argwhere(torch.sqrt(torch.sum((points - avg_xyz)**2, dim=1)) < filter_radius).squeeze(1)
                                    
            filtered_points = points[filter]
            filtered_normals = normals[filter]
            filtered_colors = colors[filter]            

            merged_points.append(filtered_points)
            merged_normals.append(filtered_normals)
            merged_colors.append(filtered_colors)

    merged_points = torch.cat(merged_points, dim=0).cpu().detach().numpy()
    merged_normals = torch.cat(merged_normals, dim=0).cpu().detach().numpy()
    merged_colors = torch.cat(merged_colors, dim=0).cpu().detach().numpy()
        
    pc.points = o3d.utility.Vector3dVector(merged_points)
    pc.normals = o3d.utility.Vector3dVector(merged_normals)
    pc.colors = o3d.utility.Vector3dVector(merged_colors)

    cl, idx = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)        
    pc.points = o3d.utility.Vector3dVector(merged_points[idx])
    pc.normals = o3d.utility.Vector3dVector(merged_normals[idx])
    pc.colors = o3d.utility.Vector3dVector(merged_colors[idx])    

    f_out_name = '{}/{}'.format(directory, 'filtered_and_merged.ply')
    return pc

"""
    def remove_plane(pointcloud, distance_threshold):

        distance_threshold = torch.tensor([distance_threshold]).to(device)
        #pc = o3d.io.read_point_cloud(pointcloud_path)
        pc = pointcloud
        points = torch.tensor(np.asarray(pc.points), requires_grad=False).float().to(device)

        # represent plane as ax + by + cz + d    
        center_xyz = torch.sum(points, dim=0) / points.size()[0]    
        min_y = torch.min(points[:,1])

        # initialize plane as flat and near center of pointcloud
        plane = torch.zeros(6, dtype=torch.float32, device=device)
        plane[:3] = torch.tensor([0.0, 1.0, 0.0])
        plane[3:] = torch.tensor([center_xyz[0], min_y, center_xyz[2]])

        plane = plane.requires_grad_()
        
        opt = torch.optim.Adam([plane], lr=0.01)
        max_epochs = 10000
        min_epochs = 100
        stop_condition = 0.0000001
        converged = False
        loss_history = np.array([])
        i = 0
        while i < max_epochs and converged == False:    
            opt.zero_grad()
            loss = torch.sqrt(torch.sum(pc_to_plane_distances(points, plane), dim=0) / points.size()[0])        
            loss.backward()
            opt.step()
            loss_history = np.append(loss_history, loss.cpu().detach().numpy())
            print(plane)
            print(loss)

            if i > min_epochs and (loss_history[-100:].sum()/100.0 - loss_history[-30:].sum()/30.0) < stop_condition:
                converged = True        
            i=i+1            

        print('final loss: ', loss)
        print('plane: ({:.3f}, {:.3f}, {:.3f}) ; ({:.3f}, {:.3f}, {:.3f})'.format(plane[0],plane[1],plane[2],plane[3],plane[4],plane[5])),     

        distances = pc_to_plane_distances(points, plane)    

        normals = torch.tensor(np.asarray(pc.normals)).to(device)
        colors = torch.tensor(np.asarray(pc.colors)).to(device)
        filter = torch.argwhere(distances > distance_threshold).squeeze(1)
        print('filtered {}/{} points '.format(points.size()[0] - filter.size()[0], points.size()[0]))

        filtered_points = points[filter].cpu().detach().numpy()
        filtered_normals = normals[filter].cpu().detach().numpy()
        filtered_colors = colors[filter].cpu().detach().numpy()       
            
        pc.points = o3d.utility.Vector3dVector(filtered_points)
        pc.normals = o3d.utility.Vector3dVector(filtered_normals)
        pc.colors = o3d.utility.Vector3dVector(filtered_colors)

        return pc
"""        


def remove_plane(pointcloud, threshold=0.005, init_n=3, iterations=1000):

    w, idx = pointcloud.segment_plane(threshold, init_n, iterations)

    points = np.asarray(pointcloud.points)
    normals = np.asarray(pointcloud.normals)
    colors = np.asarray(pointcloud.colors)    
    
    points_below_plane = points[:, 1] <= w[3]    

    points = np.delete(points, idx,0)    
    normals = np.delete(normals, idx,0)
    colors = np.delete(colors, idx,0)
    
    points_below_plane = np.squeeze(np.argwhere( points[:, 1] <= -w[3]), axis=1)        

    points = np.delete(points, points_below_plane,0)    
    normals = np.delete(normals, points_below_plane,0)
    colors = np.delete(colors, points_below_plane,0)    

    pointcloud.points = o3d.utility.Vector3dVector(np.squeeze(np.asarray([points]), axis=0))
    pointcloud.normals = o3d.utility.Vector3dVector(np.squeeze(np.asarray([normals]), axis=0))
    pointcloud.colors = o3d.utility.Vector3dVector(np.squeeze(np.asarray([colors]), axis=0))
            
    return pointcloud


def poisson_disk_resampling(pointcloud, number_of_samples, radius=None):
    
    pc = pointcloud
    points = np.squeeze(np.asarray([pc.points]), axis=0)
    normals = np.squeeze(np.asarray([pc.normals]), axis=0)
    colors = np.squeeze(np.asarray([pc.colors]), axis=0)
    idx = pcu.downsample_point_cloud_poisson_disk(points, num_samples=number_of_samples)
    
    pc.points = o3d.utility.Vector3dVector(points[idx])
    pc.normals = o3d.utility.Vector3dVector(normals[idx])
    pc.colors = o3d.utility.Vector3dVector(colors[idx])
    return pc


if __name__ == '__main__':

    path = '/home/rob/research_code/3co/research/nerf/data/dragon_scale/hyperparam_experiments/from_cloud/cactus_run38/pointclouds'

    #print('merging and filtering by center radius...')
    #pc = merge_and_filter_by_center_radius(path)
    #f_out_name = 'test/merged_and_filtered.ply'    
    #o3d.io.write_point_cloud(f_out_name, pc)  

    #print('removing plane...')
    #pc = remove_plane(pc)
    #f_out_name = 'test/plane_removed.ply'    
    #o3d.io.write_point_cloud(f_out_name, pc)      

    #print('performing poisson disk resampling...')    
    #pc = poisson_disk_resampling(pc, int(len(pc.points)*0.1))
    #f_out_name = 'test/downsampled.ply'    
    #o3d.io.write_point_cloud(f_out_name, pc)      



    with torch.no_grad():
        generate_point_clouds()
