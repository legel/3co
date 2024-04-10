import trimesh
from PIL import Image
import numpy as np
import sys
import torch
import imageio
import os


device = torch.device('cuda:0')

###
### functions for recomputing UV map data structures ###
### 

@torch.jit.script
def edge_function(p: torch.Tensor, v0: torch.Tensor, v1: torch.Tensor, equal_shapes: bool = False):

    v0_u = v0[:,0]
    v0_v = v0[:,1]
    v1_u = v1[:,0]
    v1_v = v1[:,1]
    if equal_shapes:
        p_u = p[:,0]
        p_v = p[:,1]
    else:
        number_of_vertices = v0_u.shape[0]
        p_u = p[0].repeat(number_of_vertices) 
        p_v = p[1].repeat(number_of_vertices)
    
    return (p_u - v0_u) * (v1_v - v0_v) - (p_v  - v0_v) * (v1_u - v0_u)

@torch.jit.script
def barycentric_coordinates(p: torch.Tensor, v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor, kEpsilon: float = 1e-8):
    """
    Compute the barycentric coordinates of a point relative to a triangle.
    Args:
        p: Coordinates of a point.
        v0, v1, v2: Coordinates of the triangle vertices.
    Returns
        bary: (w0, w1, w2) barycentric coordinates in the range [0, 1].
    """
    area = edge_function(v2, v0, v1, equal_shapes=True) + kEpsilon  # 2 x face area.
    w0 = edge_function(p, v1, v2) / area
    w1 = edge_function(p, v2, v0) / area
    w2 = edge_function(p, v0, v1) / area

    return torch.cat([w0.unsqueeze(1), w1.unsqueeze(1), w2.unsqueeze(1)], dim=1) #.to(device=device)


# @torch.jit.script
def get_texture_hits(image_uv_coordinates, triangles, colors, normals):

    # get image dimensions
    image_width = image_uv_coordinates.shape[0]
    image_height = image_uv_coordinates.shape[1]

    image_texels_hit = torch.zeros((image_width, image_height), dtype=torch.long, device="cuda:0")

    colors_uv = torch.zeros((image_width, image_height, 3), dtype=torch.float32, device="cuda:0")
    normals_uv = torch.zeros((image_width, image_height, 3), dtype=torch.float32, device="cuda:0")

    v0 = triangles[:,0,:]
    v1 = triangles[:,1,:]
    v2 = triangles[:,2,:]

    image_uv_coordinates = image_uv_coordinates   
    
    u_0_to_u_1_distance = torch.abs(image_uv_coordinates[0, 0][0] - image_uv_coordinates[1, 0][0])
    buffer_multiplier = 20
    
    # # Loop over every texel, define triangle index colors
    for col in range(image_width):
        print('processing col {}'.format(col))
        # get the U (columnwise 0.0-1.0) coordinate for this column
        texel_u_coordinate = image_uv_coordinates[col, 0][0]

        # get the U coordinate for the first vertex of all triangles
        v0_u_coordinate = v0[:,0]

        # compute the distance between the U coordinates
        dist_v0_u_to_texel_u = torch.abs(v0_u_coordinate - texel_u_coordinate)

        # get the triangle indices that are relevant to work with
        qualified_triangle_indices = torch.argwhere(dist_v0_u_to_texel_u < u_0_to_u_1_distance * buffer_multiplier)[:,0]

        for row in range(image_height):
            point_uv = image_uv_coordinates[col, row]
            
            # find (hopfeully) all intersections of texels and uv-triangles
            left_u = point_uv[0] - (u_0_to_u_1_distance / 2)            
            right_u = point_uv[0] + (u_0_to_u_1_distance / 2)
            top_v = point_uv[1] - (u_0_to_u_1_distance / 2)          
            bot_v = point_uv[1] + (u_0_to_u_1_distance / 2)
            
            u_condition_left = v0[qualified_triangle_indices, 0] > left_u
            u_condition_right = v0[qualified_triangle_indices, 0] < right_u
            v_condition_top = v0[qualified_triangle_indices, 1] > top_v
            v_condition_bot = v0[qualified_triangle_indices, 1] < bot_v

            vertex_condition_0 = torch.logical_and(torch.logical_and(torch.logical_and(u_condition_left, u_condition_right), v_condition_top), v_condition_bot)
            
            u_condition_left = v1[qualified_triangle_indices, 0] > left_u
            u_condition_right = v1[qualified_triangle_indices, 0] < right_u
            v_condition_top = v1[qualified_triangle_indices, 1] > top_v
            v_condition_bot = v1[qualified_triangle_indices, 1] < bot_v            

            vertex_condition_1 = torch.logical_and(torch.logical_and(torch.logical_and(u_condition_left, u_condition_right), v_condition_top), v_condition_bot)                        
            
            u_condition_left = v2[qualified_triangle_indices, 0] > left_u
            u_condition_right = v2[qualified_triangle_indices, 0] < right_u
            v_condition_top = v2[qualified_triangle_indices, 1] > top_v
            v_condition_bot = v2[qualified_triangle_indices, 1] < bot_v            

            vertex_condition_2 = torch.logical_and(torch.logical_and(torch.logical_and(u_condition_left, u_condition_right), v_condition_top), v_condition_bot)                                    

            triangle_intersects_texel_condition  = torch.logical_or(torch.logical_or(vertex_condition_0, vertex_condition_1), vertex_condition_2)
            triangle_intersects_texel = torch.argwhere(triangle_intersects_texel_condition)
            
            bary = barycentric_coordinates(point_uv, v0[qualified_triangle_indices], v1[qualified_triangle_indices], v2[qualified_triangle_indices])                        
            texel_center_is_inside_triangle = torch.argwhere(torch.min(bary, dim=1).values >= 0.0)
            
            triangle_indices = []        
            if len(triangle_intersects_texel) > 0:
                triangle_indices.append(qualified_triangle_indices[triangle_intersects_texel[:, 0]])
                
            if len(texel_center_is_inside_triangle) > 0:
                triangle_indices.append(qualified_triangle_indices[texel_center_is_inside_triangle[:, 0]])
                
            if len(triangle_indices) > 0:

                triangle_indices = torch.cat(triangle_indices)            
                
                unique_triangle_indices = torch.unique(triangle_indices)                                
                n_triangle_intersections = unique_triangle_indices.shape[0]
                            
                colors_uv[col, row] = torch.sum(colors[unique_triangle_indices], dim=0) / n_triangle_intersections                
                normals_uv[col, row] = torch.sum(normals[unique_triangle_indices], dim=0) / n_triangle_intersections                                
                image_texels_hit[col, row] = len(triangle_indices)                


    return image_texels_hit, colors_uv, normals_uv
            

def load_image(image_path):
    image_data = imageio.imread(image_path)
    # clip out alpha channel, if it exists
    image_data = image_data[:, :, :3]  # (H, W, 3)
    # convert to torch format and normalize between 0-1.0
    image = torch.from_numpy(image_data).to(device=device) # (H, W, 3) torch.float32
    return image

def prepare_image_based_texture(texture_data, image_name):
    # convert Torch data to a PIL image
    image_data = Image.fromarray(texture_data.cpu().numpy(), mode='RGB')
    # save image for visualization
    image_data.save(image_name)
    # convert the image back to a NumPy array
    image_array = np.array(image_data)
    # rotate the array 90 degrees to the right
    rotated_array = np.rot90(image_array, k=1)
    # reload the rotated image array back to a PIL image    
    image_data = Image.fromarray(rotated_array)

    return image_data


def uv_to_texel_index(input_uv, texture_size):
    uv_indices_2d = torch.floor(input_uv[..., :] * texture_size).int()
    uv_indices_1d = uv_indices_2d[..., 0] * texture_size + uv_indices_2d[..., 1]
    return uv_indices_1d

def convert_1d_coor_to_2d(coor, image_width):
    rows = (coor / image_width).int()
    cols = (coor % image_width).int()
    return torch.cat([rows.unsqueeze(1), cols.unsqueeze(1)], dim=1)


mesh_filename = 'test/dragon_scale/3/dragon_triangulated_beautified_remeshed_smooth_10_ratio_0.70_instant_meshed_uv_mapped_angle_0.80_island_margin_0.003_face_weight_0.0.ply'

# load a mesh, extract faces indices and UV for processing
mesh = trimesh.load(mesh_filename, process=False)
vertices = mesh.vertices

# load face normals (float32 in [-1,1]) generated from blender script
face_normals = torch.zeros((vertices.shape[0], 3), dtype=torch.float32)
face_normals_f_name = 'dragon_winding_fixed_unseen_faces_removed_blender-remesh_im_100k_vertices.csv'
face_normals_f_name = 'dragon_triangulated_beautified_remeshed_smooth_10_ratio_0.70_instant_meshed_uv_mapped_angle_0.80_island_margin_0.003_face_weight_0.0.csv'
with open(face_normals_f_name, 'r') as f:
    for i, line in enumerate(f):
        normals_i = line.split(',')
        face_normals[i] = torch.tensor([float(normals_i[0]), float(normals_i[1]), float(normals_i[2])])
    face_normals = torch.nn.functional.normalize(face_normals, dim=1, p=2).cpu()
    torch.save(face_normals, 'face_normals_dragon_triangulated_beautified_remeshed_smooth_10_ratio_0.70_instant_meshed_uv_mapped_angle_0.80_island_margin_0.003_face_weight_0.0')

faces = torch.from_numpy(np.asarray(mesh.faces))
uv = torch.from_numpy(np.asarray(mesh.visual.uv))

face_colors_filename = 'face_colors_dragon_triangulated_beautified_remeshed_smooth_10_ratio_0.70_instant_meshed_uv_mapped_angle_0.80_island_margin_0.003_face_weight_0.0.pt'

# incoming: float, [0,1]
face_colors = (torch.load(face_colors_filename)).to(dtype=torch.float32, device=device)
            
triangles_with_uv_coordinates = uv[faces]

# define UV map textures size
image_width = 2048
image_height = 2048

# Create a tensor of texel indices
texel_indices = torch.meshgrid(torch.arange(image_width), torch.arange(image_height))
texel_indices_tensor = torch.stack(texel_indices, dim=-1)

# Convert the texel indices to probe coordinates
image_uv_coordinates = texel_indices_tensor.float() / torch.tensor([image_width, image_height]).float()

# shift coordinates by one-half of the size of the texel, lower and to the right, so points represent middle of texel
col_shift, row_shift = image_uv_coordinates[1,1,:]
image_uv_coordinates[:,:,0] = image_uv_coordinates[:,:,0] + col_shift/2
image_uv_coordinates[:,:,1] = image_uv_coordinates[:,:,1] + row_shift/2


project_name = 'initial_textures/t1/'
os.makedirs(project_name, exist_ok=True)

# convenience loading / saving of intermediate data for debugging purposes
load_precomputed_data = True

if load_precomputed_data:
    image_texels_hit = torch.load("{}/saved_image_texels_hit_{}_x_{}.pt".format(project_name, image_width, image_height))
    color_texture = torch.load("{}/saved_colors_{}_x_{}.pt".format(project_name, image_width, image_height))
    normals_texture = (torch.load("{}/saved_normals_{}_x_{}.pt".format(project_name, image_width, image_height)))
else:
    image_texels_hit, color_texture, normals_texture = get_texture_hits(
            image_uv_coordinates=image_uv_coordinates.to(device=device), 
            triangles=triangles_with_uv_coordinates.to(device=device), 
            colors=face_colors,
            normals=face_normals
    )    


torch.save(image_texels_hit, "{}/saved_image_texels_hit_{}_x_{}.pt".format(project_name, image_width, image_height))
torch.save(color_texture, "{}/saved_colors_{}_x_{}.pt".format(project_name, image_width, image_height))
torch.save(normals_texture, "{}/saved_normals_{}_x_{}.pt".format(project_name, image_width, image_height))





# gather textures. Note that roughness and metallic are packed into same texture, with roughness = G, metallic = R
roughness_texture = torch.zeros((image_width, image_height,3), dtype=torch.float32, device=device)
roughness_texture[:,:,1] = 1.0

# approximate textures for missed texels
missed_texels = torch.stack(torch.where(image_texels_hit == 0), dim=1)
hit_texels = torch.stack(torch.where(image_texels_hit > 0), dim=1)

missed_texels_color = torch.zeros( (missed_texels.shape[0], 3), dtype=torch.float32, device=device)
missed_texels_normals = torch.zeros( (missed_texels.shape[0], 3), dtype=torch.float32, device=device)
missed_texels_u = (missed_texels[:,0] / image_width)
missed_texels_v = (missed_texels[:,1] / image_height)

missed_texels_uv = torch.stack([missed_texels_u, missed_texels_v],dim=1).to(device)
batch_size = 1000
missed_texels_uv_batches = torch.split(missed_texels_uv, batch_size)
missed_texels_batches = torch.split(missed_texels, batch_size)
v_0 = triangles_with_uv_coordinates[:, 0, :].to(device)

for batch in range(len(missed_texels_uv_batches)):

    # for each missed texel, find the closest uv-triangle in mesh faces
    # (just the first vertex of each triangle is considered; empirically not much
    #  improvement to consider all three)
    missed_texels_uv_batch = missed_texels_uv_batches[batch]    
    missed_texels_batch = missed_texels_batches[batch]
    dists = torch.cdist(missed_texels_uv_batch.float(), v_0.float(), p=2)
    min_indices = dists.min(dim=1)[1]

    min_texel_indices = uv_to_texel_index(v_0[min_indices], image_width)        
    min_texel_indices_2d = convert_1d_coor_to_2d(min_texel_indices, image_width)    

    # find closest texel that was hit, measured in terms of UV coordinate euclidean distnace
    res = torch.cdist(min_texel_indices_2d.float(), hit_texels.float(), p=2).min(dim=1)    
    min_surrogate_texel_indices = res[1]
    
    surrogate_texel_indices = hit_texels[min_surrogate_texel_indices]    
    missed_texels_color [(batch*batch_size):((batch+1)*batch_size)] = color_texture[surrogate_texel_indices[:,0], surrogate_texel_indices[:,1]]
    missed_texels_normals [(batch*batch_size):((batch+1)*batch_size)] = normals_texture[surrogate_texel_indices[:,0], surrogate_texel_indices[:,1]]
    
color_texture[ missed_texels[:, 0], missed_texels[:, 1], :3 ] = missed_texels_color
normals_texture[ missed_texels[:, 0], missed_texels[:, 1], :3 ] = missed_texels_normals

color_texture = (255*color_texture).to(dtype=torch.int8)
roughness_texture = (255*roughness_texture).to(dtype=torch.int8)

colors_filename = "{}colors_texture_{}x{}.png".format(project_name, image_width, image_height)
metallic_roughness_filename = "{}metallic_roughness_texture_{}x{}.png".format(project_name, image_width, image_height)
normals_filename = "{}normals_texture_{}x{}.pt".format(project_name, image_width, image_height)

# save normals in float32 texture space
torch.save(normals_texture, normals_filename)

# create pbr materials for gltf
color_image_data = prepare_image_based_texture(texture_data=color_texture, image_name=colors_filename)
metallic_roughness_image_data = prepare_image_based_texture(texture_data=roughness_texture, image_name=metallic_roughness_filename)
pbr_material = trimesh.visual.material.PBRMaterial(baseColorTexture=color_image_data, metallicRoughnessTexture=metallic_roughness_image_data) # ignoring normal maps for now

# gather UV data, ensure material is assigned to mesh
uv = mesh.visual.uv
color_visuals = trimesh.visual.TextureVisuals(uv=uv, image=color_image_data, material=pbr_material)
mesh.visual = color_visuals

# export mesh as glTF
output_gltf_filename = "{}_{}x{}.glb".format(project_name, image_width, image_height)
mesh.export(output_gltf_filename, file_type='glb')