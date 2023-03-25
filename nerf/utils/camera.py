import torch

device = torch.device('cuda:0')

def get_sensor_xyz_coordinates(pose_data, depth_data, H, W, principal_point_x, principal_point_y, focal_lengths):

    # get camera world position and rotation
    camera_world_position = pose_data[:3, 3].view(1, 1, 1, 3)     # (1, 1, 1, 3)
    camera_world_rotation = pose_data[:3, :3].view(1, 1, 1, 3, 3) # (1, 1, 1, 3, 3)

    # create meshgrid representing rows and cols for *all* image pixels (i.e., before weighted pixel sampling)
    pixel_rows_and_cols = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=device),
                                            torch.arange(W, dtype=torch.float32, device=device),
                                            indexing='ij'
    )  # (H, W)

    rows = pixel_rows_and_cols[0].flatten()
    cols = pixel_rows_and_cols[1].flatten()    

    # get relative pixel orientations
    pixel_directions = compute_pixel_directions(focal_lengths, rows, cols, principal_point_x, principal_point_y) # (H, W, 3, 1)
    
    xyz_coordinates = derive_xyz_coordinates(camera_world_position, camera_world_rotation, pixel_directions, depth_data.flatten(),flattened=True).reshape(H,W,3)

    return xyz_coordinates 

def derive_xyz_coordinates(camera_world_position, camera_world_rotation, pixel_directions, pixel_depths, flattened=False):
    
    if not flattened:
        # transform rays from camera coordinate to world coordinate
        # camera_world_rotation: [1,1,1,1,3,3]    
        pixel_world_directions = torch.matmul(camera_world_rotation, pixel_directions).squeeze(4).squeeze(0)                                                                        
        pixel_world_directions = torch.nn.functional.normalize(pixel_world_directions, p=2, dim=2)  # (N_pixels, 3)
                    
        # Get sample position in the world (1, 1, 3) + (H, W, 3) * (H, W, 1) -> (H, W, 3)
        global_xyz = camera_world_position + pixel_world_directions * pixel_depths.unsqueeze(2)            
        global_xyz = global_xyz.squeeze(0)
    else:                        
        pixel_directions_world = torch.matmul(camera_world_rotation.squeeze(0).squeeze(0), pixel_directions.unsqueeze(2)).squeeze(2)  # (N, 3, 3) * (N, 3, 1) -> (N, 3) .squeeze(3) 
        pixel_directions_world = torch.nn.functional.normalize(pixel_directions_world, p=2, dim=1)  # (N_pixels, 3)            
        pixel_depth_samples_world_directions = pixel_directions_world * pixel_depths.unsqueeze(1).expand(-1,3) # (N_pixels, 3)                        
        
        global_xyz = camera_world_position.squeeze(0).squeeze(0) + pixel_depth_samples_world_directions # (N_pixels, 3)                                    
        
    return global_xyz      


# TODO: store these parameters in a "camera" class
def compute_pixel_directions(focal_lengths, pixel_rows, pixel_cols, principal_point_x, principal_point_y):

    # Our camera coordinate system matches Apple's camera coordinate system:
    # x = right
    # y = up
    # -z = forward        

    dev = focal_lengths.device
    n_pixels = focal_lengths.size()[0]                   
    
    pp_x = principal_point_x
    pp_y = principal_point_y        

    pixel_directions_x = (pixel_cols.to(dev) - pp_x.to(dev)) / focal_lengths        
    pixel_directions_y = -(pixel_rows.to(dev) - pp_y.to(dev)) / focal_lengths
    pixel_directions_z = -torch.ones(n_pixels, dtype=torch.float32, device=dev)

    pixel_directions = torch.stack([pixel_directions_x.unsqueeze(1), pixel_directions_y.unsqueeze(1), pixel_directions_z.unsqueeze(1)], dim=1).to(dev)

    pixel_directions = pixel_directions.squeeze(2)
    pixel_directions = torch.nn.functional.normalize(pixel_directions, p=2, dim=1)
        
    return pixel_directions