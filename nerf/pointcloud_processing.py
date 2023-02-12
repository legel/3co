import torch
import torch._dynamo
import numpy as np
import open3d as o3d
import os
import point_cloud_utils as pcu
from utils.training_utils import set_randomness

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

set_randomness()
torch.set_float32_matmul_precision('high')
torch._dynamo.config.verbose=True      

device = torch.device('cuda:0')         

# # set cache directory
os.environ['PYTORCH_KERNEL_CACHE_PATH'] = './'

def pc_to_plane_distances(pc, plane):

    epsilon = 0.000001
    i = torch.ones(pc.size()[0],3).to(device)                
    distances = torch.sqrt(torch.sum(  (pc * plane[:3].unsqueeze(0).expand(pc.size()[0],3) + plane[3:])**2,  dim=1))
    flat = torch.tensor([0.0, 1.0, 0.0], requires_grad=False).to(device=device)

    distances = distances + torch.sqrt(  torch.sum(  (plane[:3] - flat)**2,  dim=0) + epsilon )
    return distances

def merge_and_filter_by_center_radius(directory):

    filter_radius = 0.25
    pointclouds = []
    centers = []
    file_names = os.listdir(directory)
    for file_name in file_names:        
        if file_name.split('.')[-1] == 'ply':
            pc = o3d.io.read_point_cloud(directory + "/{}".format(file_name.split('/')[-1]))
            centers.append(pc.get_center())
    
    centers = torch.tensor(centers)
    avg_xyz = torch.sum(centers,dim=0) / len(centers)

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
    f_out_name = '{}/{}'.format(directory, 'filtered_and_merged.ply')
    return pc


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

    path = '/home/rob/research_code/3co/research/nerf/data/cactus_large/hyperparam_experiments/from_cloud/cactus_large_runtest/pointclouds'

    print('merging and filtering by center radius...')
    pc = merge_and_filter_by_center_radius(path)
    f_out_name = 'test/merged_and_filtered.ply'    
    o3d.io.write_point_cloud(f_out_name, pc)  

    print('performing poisson disk resampling...')    
    pc = poisson_disk_resampling(pc, int(len(pc.points)*0.1))
    f_out_name = 'test/downsampled.ply'    
    o3d.io.write_point_cloud(f_out_name, pc)      

    print('removing plane...')
    pc = remove_plane(pc, distance_threshold=0.075)
    f_out_name = 'test/plane_removed.ply'    
    o3d.io.write_point_cloud(f_out_name, pc)      
