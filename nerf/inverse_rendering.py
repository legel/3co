import torch
import torch.nn as nn
import struct
from learn import *
from PIL import Image
import imageio.v2 as imageio
import numpy as np
import random, math
import open3d as o3d
import trimesh
import sys, os, shutil, copy, glob, json
from utils.camera import *
from pbr_rendering import *
import math

sys.path.append(os.path.join(sys.path[0], '../..'))

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    AmbientLights,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesVertex
)

from pytorch3d.vis.plotly_vis import plot_batch_individually
from functorch.compile import make_boxed_func
import gc

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch._dynamo.config.verbose = True
torch.set_float32_matmul_precision('high')

set_randomness()

device = torch.device('cuda:0')

texture_size = 2048
dataset = 'dragon_scale'
start_time = int(time.time())
experiment_label = "{}".format( int(str(start_time)[:9]) )
experiment_dir = './data/{}/inverse_rendering/{}'.format(dataset, experiment_label)
os.makedirs(experiment_dir, exist_ok=True)

mesh_render_dir = '{}/mesh_renders'.format(experiment_dir)
os.makedirs(mesh_render_dir, exist_ok=True)

gradient_dir = '{}/gradients'.format(experiment_dir)
os.makedirs(gradient_dir, exist_ok=True)

input_data_dir = './data/{}/inverse_rendering'.format(dataset)
os.makedirs(input_data_dir, exist_ok=True)

texture_visualization_dir = '{}/texture_visualizations'
os.makedirs(texture_visualization_dir, exist_ok=True)

gltf_dir = '{}/gltfs'.format(experiment_dir)
os.makedirs(gltf_dir, exist_ok=True)

models_dir = '{}/models'.format(experiment_dir)
os.makedirs(models_dir, exist_ok=True)


#################################################################################################
################# Rasterizer ####################################################################
#################################################################################################

def load_mesh(mesh_file_path):
    # Load the PLY file using Open3D
    mesh = o3d.io.read_triangle_mesh(mesh_file_path)

    # Extract the vertex colors as a numpy array
    vertex_colors = np.asarray(mesh.vertex_colors)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # assign colors based on geometry
    min_x = np.min(vertices[:,0])
    max_x = np.max(vertices[:,0])
    min_y = np.min(vertices[:,1])
    max_y = np.max(vertices[:,1])
    min_z = np.min(vertices[:,2])
    max_z = np.max(vertices[:,2])

    if len(vertex_colors) == 0:
        vertex_colors = vertices.copy()
        vertex_colors[:,0] = (vertex_colors[:,0] - min_x) / (max_x - min_x)
        vertex_colors[:,1] = (vertex_colors[:,1] - min_y) / (max_y - min_y)
        vertex_colors[:,2] = (vertex_colors[:,2] - min_z) / (max_z - min_z)

    # Convert the numpy array to a PyTorch tensor
    vertex_colors_tensor = torch.from_numpy(vertex_colors).float().to(device=device)
    vertices_tensor = torch.from_numpy(vertices).float().to(device=device)
    triangles_tensor = torch.from_numpy(triangles).long().to(device=device)

    rgb_texture = TexturesVertex(verts_features=[vertex_colors_tensor]).to(torch.device('cuda:0'))
    mesh = Meshes(verts=[vertices_tensor], faces=[triangles_tensor], textures=rgb_texture).to(torch.device('cuda:0'))

    return mesh


def correct_camera_extrinsics_for_pytorch3d(extrinsics):
    # in rob we trust
    R = np.array([
        [-1,  0,  0,  0],
        [ 0,  1,  0,  0],
        [ 0,  0, -1,  0],
        [ 0,  0,  0,  1]
    ])
    
    extrinsics[:3,:3] =  extrinsics[:3,:3] @ R[:3,:3] # giv'm a left hook
    extrinsics[:3, 3] = -extrinsics[:3, 3] @ extrinsics[:3,:3] # giv'm an uppercut
    
    return extrinsics # knockout


def render_mesh(extrinsics, intrinsics, mesh, image_size, show_interactive_3D_figure=False, save_renders=True):

    intrinsics[0,0] = intrinsics[0,0] * (float(image_size[0]) / 1440.0)
    intrinsics[1,1] = intrinsics[1,1] * (float(image_size[1]) / 1920.0)

    intrinsics[0,2] = intrinsics[0,2] * (float(image_size[0]) / 1440.0)
    intrinsics[1,2] = intrinsics[1,2] * (float(image_size[1]) / 1920.0)


    # Create the camera object with the given intrinsics and pose
    cameras = PerspectiveCameras(
                R=extrinsics[:3, :3].unsqueeze(0).to(device=device),
                T=extrinsics[:3, 3].unsqueeze(0).to(device=device), 
                image_size=(image_size,), 
                principal_point=torch.tensor([intrinsics[0,2], intrinsics[1,2]]).unsqueeze(0), 
                focal_length=torch.tensor([intrinsics[0,0], intrinsics[1,1]]).unsqueeze(0), 
                device=device, 
                in_ndc=False 
            )
    
    lights = AmbientLights(ambient_color=(1.0,1.0,1.0), device=device)
    raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1) # Rasterizer all set up
    mesh_rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings) # Rasterizer ready to go
    
    if show_interactive_3D_figure:
        fig = plot_batch_individually([cameras, mesh])
        fig.show()
    
    fragments = mesh_rasterizer.forward(mesh)        
    pixels_to_mesh_face_indices = fragments.pix_to_face
    bary_coords = fragments.bary_coords
    # unused parameters that may be useful at some point:
    # zbuf = fragments.zbuf
    # pix_dists = fragments.dists
    
    pixels_to_mesh_face_indices = pixels_to_mesh_face_indices[0,:,:,0] # for the 0th mesh, for all H,W pixels, get the first face index
    
    pixels_that_hit_face = len(torch.where(pixels_to_mesh_face_indices != -1)[0])
    print("          -> Pixel Raycast Hits: {:,} Faces".format(pixels_that_hit_face))

    if save_renders:
        soft_phong_shader = SoftPhongShader(device=device, cameras=cameras, lights=lights) # Shader all systems go

        # Create a Phong renderer by composing a rasterizer and a shader
        renderer = MeshRenderer(rasterizer=mesh_rasterizer, shader=soft_phong_shader) # Mesh rendering ready
        image = renderer(mesh.to(device=device))  # Boom! Render that mesh :)
        image = (image.detach().cpu().numpy()[0, ..., :3] * 255).astype(np.uint8)

        return image, pixels_to_mesh_face_indices, bary_coords

    else:
        return None, pixels_to_mesh_face_indices, bary_coords
    

def rasterize(mesh, camera_extrinsics, camera_intrinsics, H, W, save_result=False, save_renders=True):
    
    number_of_poses = camera_extrinsics.shape[0]
    
    number_of_rows = H
    number_of_cols = W
        
    posepix2face = torch.full( (number_of_poses, H*W), fill_value=-1, dtype=torch.int32, device=device)
    pose_bary_coords = torch.full( (number_of_poses, H*W, 3), fill_value=-1, dtype=torch.float32, device=device)
    
    for pose_index in range(0,number_of_poses):

        extrinsics = correct_camera_extrinsics_for_pytorch3d(camera_extrinsics[pose_index,:,:])
        intrinsics = camera_intrinsics[pose_index,:,:]

        intrinsics = torch.from_numpy(intrinsics) 
        extrinsics = torch.from_numpy(extrinsics)
        
        print("{}.png ({}.jpg)".format(pose_index, pose_index*60))

        image, pix2face, bary_coords = render_mesh( 
            extrinsics=extrinsics, 
            intrinsics=intrinsics, 
            mesh=mesh,
            image_size=(number_of_rows, number_of_cols),
            show_interactive_3D_figure=False,
            save_renders=save_renders
        )

        # (n_poses, H*W)                
        posepix2face[pose_index] = pix2face.reshape(H*W)
        pose_bary_coords[pose_index] = bary_coords[0, :, :, :].reshape(H*W,3)

        if save_renders:
            im = Image.fromarray(image)
            im.save('{}/{}.png'.format(input_data_dir, pose_index))                                

    if save_result:
        save_rasterizer_result(
            '{}/posepix2face_dragon_scale_test.pt'.format(input_data_dir), posepix2face,
            '{}/bary_coords_dragon_scale_test.pt'.format(input_data_dir), pose_bary_coords
        )

    return posepix2face, pose_bary_coords
        

def save_rasterizer_result(posepix2face_f_name, posepix2face, bary_coords_f_name, bary_coords):
        
    torch.save(posepix2face, posepix2face_f_name)    
    torch.save(bary_coords, bary_coords_f_name)



###########################################################################################
################## Inverse Rendering ######################################################
###########################################################################################                   

class DisneyBRDFModel(nn.Module):

    def __init__(self, n_texels, initial_textures=None):

        super(DisneyBRDFModel, self).__init__()
        # roughness, red, green, blue are overwritten below in the case where initial_textures=True
        brdf_params = {}
        brdf_params['metallic'] = 0.0
        brdf_params['subsurface'] = 0.0  
        brdf_params['specular'] = 0.5
        brdf_params['roughness'] = 1.0
        brdf_params['specularTint'] = 0.0
        brdf_params['anisotropic'] = 0.0
        brdf_params['sheen'] = 0.0
        brdf_params['sheenTint'] = 0.0
        brdf_params['clearcoat'] = 0.0
        brdf_params['clearcoatGloss'] = 0.0
        brdf_params['red'] = 0.3
        brdf_params['green'] = 0.5
        brdf_params['blue'] = 0.35

        brdf_params_tensor = torch.tensor([
            brdf_params['metallic'], brdf_params['subsurface'], brdf_params['specular'],
            brdf_params['roughness'], brdf_params['specularTint'], brdf_params['anisotropic'],
            brdf_params['sheen'], brdf_params['sheenTint'], brdf_params['clearcoat'],
            brdf_params['clearcoatGloss'], brdf_params['red'], brdf_params['green'], brdf_params['blue'],
        ])
            
        brdf_params_tensor_expand = brdf_params_tensor.repeat( (n_texels, 1) )

        if initial_textures is not None:            
            brdf_params_tensor_expand[:, -3:] = initial_textures[0].flatten(start_dim=0, end_dim=1)
            brdf_params_tensor_expand[:, 3] = initial_textures[1].flatten(start_dim=0, end_dim=1)[:, 1]

            brdf_params_tensor_expand[:] = torch.asin(brdf_params_tensor_expand * 2.0 - 1)
            self.initial_brdf_params = brdf_params_tensor_expand.to(torch.device('cuda:0'))            
            self.initial_brdf_params.requires_grad = False                                                    

        initial_delta = torch.zeros( (brdf_params_tensor_expand.shape[0], 13))        
        self.brdf_params = nn.Parameter(initial_delta, requires_grad=True)        
        self.frozen_indices = torch.zeros((13,), requires_grad=False, dtype=torch.uint8, device=device)
        self.frozen_param_values = torch.zeros((n_texels, 13,), dtype=torch.float32, requires_grad=False, device=device)
        self.active_indices = torch.ones((13,), requires_grad=False, dtype=torch.uint8, device=device)


    def set_mode(self, optimization='all'):
        
        with torch.no_grad():        

            # swap frozen and active indices; retrieve their values by calling forward() 
            #   to keep both in computational graph
            self.frozen_param_values[:, torch.where(self.active_indices)[0]] = self.forward()([0])[:, torch.where(self.active_indices)[0]].detach().clone()

            if optimization == 'diffuse':
                print('DisneyBRDFModel: setting mode to diffuse')                                                
                self.active_indices[:10] = 0
                self.active_indices[-3:] = 1                
                self.frozen_indices[:10] = 1
                self.frozen_indices[-3:] = 0
            elif optimization == 'reflectance':
                print('DisneyBRDFModel: setting mode to reflectance')
                self.active_indices[:10] = 1
                self.active_indices[-3:] = 0                
                self.frozen_indices[:10] = 0
                self.frozen_indices[-3:] = 1 
            else:                
                self.active_indices[:] = 1
                self.frozen_indices[:] = 0                                    


    def forward(self):        

        # because pytorch 2.0 just needed some boilerplate
        def f(i=None):
        
            brdf = (torch.sin(self.brdf_params + self.initial_brdf_params) + 1) / 2.0        
            with torch.no_grad():                        
                brdf[:, 0] = 0
                brdf[:, 1] = 0
                brdf[:, 2] = 0.5
                brdf[:, 3] = torch.clamp(brdf[:, 3], min=0, max=1.0)
                brdf[:, 4] = 0
                brdf[:, 5] = 0
                brdf[:, 6] = 0
                brdf[:, 7] = 0
                brdf[:, 8] = 0
                brdf[:, 9] = 0
                brdf[:, -3:] = torch.clamp(brdf[:, -3:], min=0, max=1)
                brdf[:, torch.where(self.frozen_indices)[0]] = self.frozen_param_values[:, torch.where(self.frozen_indices)[0]]
            return brdf
        
        return make_boxed_func(f)
            

    def visualize_texture_gradient(self, epoch, batch_n):
     
        pose_i = 0
        brdf_params = self.brdf_params
        grad = brdf_params.grad
        if grad == None:
            print("grad is none")
            return        
            
        print("Visualizing gradients")
        vis_grad = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        brdf_param_names = self.get_brdf_param_names()
        
        for p in vis_grad:

            grad_p = grad[:, p]
            rendered_grad = torch.zeros((texture_size*texture_size), dtype=torch.float32, device=device)
            rendered_grad[:] = grad_p
                        
            min_grad = grad_p.min()                
            max_grad = grad_p.max()
            
            rendered_grad = rendered_grad - min_grad            
            rendered_grad[torch.where(rendered_grad !=0)[0]] = rendered_grad[torch.where(rendered_grad !=0)[0]] / max_grad
            rendered_grad = rendered_grad.reshape(texture_size, texture_size)
            rendered_grad = (rendered_grad.cpu().numpy() * 255).astype(np.uint8)
            f_name = '{}/{}/grad_pose{}_epoch{}_batch{}.png'.format(gradient_dir, brdf_param_names[p], pose_i, epoch, batch_n)
            os.makedirs('{}/{}'.format(gradient_dir,brdf_param_names[p]), exist_ok=True)
            imageio.imwrite(f_name, rendered_grad)


    def get_brdf_param_names(self):
        return [
            'metallic', 'subsurface', 'specular', 'roughness', 'specularTint', 'anisotropic',
            'sheen', 'sheenTint', 'clearcoat', 'clearcoatGloss', 'red', 'green', 'blue',
        ]


class TexelSamples:
    
    def __init__(self, texel_index, rgb_samples, cam_xyzs, surface_xyz, face_indices):
        self.texel_index = texel_index # index in texture coordinates
        self.rgb_samples = rgb_samples
        self.cam_xyzs = cam_xyzs                     
        self.surface_xyzs = surface_xyz
        self.face_indices = face_indices

    def add_sample(self, rgb_sample, cam_xyz, surface_xyz, face_index):
        self.rgb_samples.append(rgb_sample)
        self.cam_xyzs.append(cam_xyz)
        self.surface_xyzs.append(surface_xyz)
        self.face_indices.append(face_index)


def uv_to_texel_index(input_uv):

    uv_indices_2d = torch.floor(input_uv[..., :] * texture_size).int()
    uv_indices_1d = uv_indices_2d[..., 0] * texture_size + uv_indices_2d[..., 1]
    return uv_indices_1d 

def texel_index_to_uv(texel_i):

    texel_width = torch.tensor(1.0 / texture_size, device=device)
    u = texel_width * torch.floor(texel_i[..., :]/texture_size) + texel_width/2.0
    v = texel_width * torch.fmod(texel_i[..., :], texture_size) + texel_width/2.0

    uv_coords = torch.stack([u,v], dim=-1)    
    return uv_coords


def render_training_views_with_uvmap_and_textures(cam_xyzs, vertex_xyzs, bary_coords, uv, textures, posepix2face, faces, lights, epoch):
    
    triangles_with_uv_coordinates = uv[faces].to(torch.device('cuda:0'))
    triangles_with_xyz_coordinates = vertex_xyzs[faces].to(torch.device('cuda:0'))     
    cam_xyzs = cam_xyzs.to(torch.device('cuda:0'))
    bary_coords = bary_coords.cpu()

    render_h = 1440
    render_w = 1920                
    n_poses = cam_xyzs.shape[0]

    for i in range(n_poses):
        pix2face = posepix2face[i].to(torch.device('cuda:0'))
        hit_indices = torch.where(pix2face != -1)[0].cpu()
        rendered_rgb = torch.zeros((render_h * render_w, 3), dtype=torch.float32, device=device)
        
        # we need to batch here to deal with matmul's memory wasteage
        # TODO: precompute this
        batch_size = 5000
        view_bary_coords = bary_coords[i, hit_indices].split(batch_size)
        hit_indices_batches = hit_indices.split(batch_size)
        n_batches = len(hit_indices_batches)
                
        for batch in range(n_batches):
            
            hit_indices_batch = hit_indices_batches[batch].to(torch.device('cuda:0'))
            view_triangles_with_uv_coordinates = triangles_with_uv_coordinates[pix2face[hit_indices_batch]]
            view_triangles_with_xyz_coordinates = triangles_with_xyz_coordinates[pix2face[hit_indices_batch]]
            view_bary_coords_batch = view_bary_coords[batch].to(torch.device('cuda:0'))
            
            view_cam_xyz = cam_xyzs[i].unsqueeze(0).expand(view_triangles_with_uv_coordinates.shape[0], 3)
            
            view_uv = torch.matmul(view_bary_coords_batch, view_triangles_with_uv_coordinates)
            view_xyz = torch.matmul(view_bary_coords_batch, view_triangles_with_xyz_coordinates)

            view_uv = torch.mean(view_uv, dim=1)
            view_xyz = torch.mean(view_xyz, dim=1)

            render_result = render_brdf_with_uvmap_and_textures(view_cam_xyz, view_xyz, view_uv, textures, lights)
            rendered_rgb[hit_indices_batch] = render_result

        rendered_rgb = rendered_rgb.reshape(render_h,render_w,3)
        rendered_rgb = (rendered_rgb.cpu().numpy() * 255).astype(np.uint8)
        label = str(i).zfill(4)
        
        f_name = './{}/view_{}_epoch_{}.png'.format(mesh_render_dir, label, str(epoch).zfill(6))
        imageio.imwrite(f_name, rendered_rgb)


def render_brdf_with_uvmap_and_textures(cam_xyzs, surface_xyzs, uv_coords, textures, lights):
    
    n_samples = uv_coords.shape[0]                    
    
    colors = textures[0].to(torch.device('cuda:0'))
    roughness = textures[1].to(torch.device('cuda:0'))
    normals = textures[2].to(torch.device('cuda:0'))
    cam_xyzs = cam_xyzs.to(torch.device('cuda:0'))
    surface_xyzs = surface_xyzs.to(torch.device('cuda:0'))
    
    # sample each texture using grid_sample()        
    interp_colors = sample_texture(colors, uv_coords).to(torch.device('cuda:0'))
    interp_normals = sample_texture(normals, uv_coords).to(torch.device('cuda:0'))
    interp_roughness = sample_texture(roughness, uv_coords).to(torch.device('cuda:0'))

    # ensure interpololated colors in [0, 255] and normalize for brdf rendering
    interp_colors = torch.clamp(interp_colors, min=0, max=255)
    interp_colors = interp_colors/255.0

    # ensure interpolated normals in [-1,1] and that they are normalized
    interp_normals = torch.clamp(interp_normals, min=-1, max=1)        
    interp_normals = torch.nn.functional.normalize(interp_normals, dim=1, p=2.0)
        
    # ensure roughness in [0, 255] and normalize for brdf computation
    interp_roughness = torch.clamp(interp_roughness, min=0, max=255)
    interp_roughness = interp_roughness/255.0        

    # TODO: surface_xyz should be interpolated as well!

    # construct final (brdf parameters, hit surface xyz, camera xyz) batch for rendering
    brdf_params = torch.zeros((n_samples, 13), dtype=torch.float32, device=device)

    brdf_params[:, 2] = 0.0   # specular
    brdf_params[:, 3] = interp_roughness[:, 1]
    brdf_params[:, 8] = 0.0 # clearcoat
    brdf_params[:, 9] = 0.0 # clearcoatgloss
    brdf_params[:, -3:] = interp_colors        
        
    render_result = render_brdf(interp_normals, cam_xyzs, surface_xyzs, brdf_params, lights)

    return render_result


def sample_texture(texture, coords):
    texture = texture.permute((2,1,0)).unsqueeze(0).float()    
    uv_coordinates_normalized = 2.0 * coords - 1.0    
    grid = uv_coordinates_normalized.unsqueeze(0).unsqueeze(0).view(1, 1, coords.shape[0], 2)    
    sampled_texture = torch.nn.functional.grid_sample(texture, grid, mode='bicubic', padding_mode='border', align_corners=True)    
    sampled_texture = sampled_texture.squeeze(0).squeeze(1).permute((1,0))
    
    return sampled_texture

def create_textures_from_brdf(brdf_params, normals_img):
    
    textures = [
        brdf_params[:, -3:].reshape(texture_size, texture_size, 3).to(torch.device('cuda:0')) * 255, 
        brdf_params[:, [0, 3, 0]].reshape(texture_size, texture_size, 3).to(torch.device('cuda:0')) * 255,
        normals_img.to(torch.device('cuda:0'))
    ]

    return textures
    

def prepare_image_based_texture(texture_data):
    # convert Torch data to a PIL image
    image_data = Image.fromarray(texture_data.cpu().numpy(), mode='RGB')
    
    # convert the image back to a NumPy array
    image_array = np.array(image_data)

    # rotate the array 90 degrees to the right
    rotated_array = np.rot90(image_array, k=1)

    # reload the rotated image array back to a PIL image
    image_data = Image.fromarray(rotated_array)

    return image_data


def save_gltf(textures, uv, output_gltf_filename):
    
    
    color_texture = (textures[0]).to(dtype=torch.uint8)    
    metallic_roughness_texture = (textures[1]).to(dtype=torch.uint8)
    metallic_roughness_texture[:, :, 0] = 0
    metallic_roughness_texture[:, :, 2] = 0
    color_image_data = prepare_image_based_texture(color_texture)
    metallic_roughness_image_data = prepare_image_based_texture(metallic_roughness_texture)    

    pbr_material = trimesh.visual.material.PBRMaterial(baseColorTexture=color_image_data, metallicRoughnessTexture=metallic_roughness_image_data) #, alphaMode='MASK', alphaCutoff=1.0) # ignoring normal maps for now

    # gather UV data, assign materials to mesh
    uv = mesh.visual.uv
    color_visuals = trimesh.visual.TextureVisuals(uv=uv, image=color_image_data, material=pbr_material)
    mesh.visual = color_visuals

    # export mesh as glTF    
    mesh.export(output_gltf_filename, file_type='glb')    


def print_brdf(brdf_params, index):

    names = [
        'metallic', 'subsurface', 'specular', 'roughness', 
        'specularTint','anisotropic', 'sheen', 'sheenTint', 
        'clearcoat', 'clearcoatGloss', 'red', 'green', 'blue'
    ]

    #torch.set_printoptions(precision=5)    
    for i, name in enumerate(names):        
        print('{}: {}'.format(name, brdf_params[index,i]))
    #torch.set_printoptions(precision='default')

     
def load_image(image_path):
    image_data = imageio.imread(image_path)

    # clip out alpha channel, if it exists
    image_data = image_data[:, :, :3]  # (H, W, 3)

    # convert to torch format and normalize between 0-1.0
    image = torch.from_numpy(image_data).to(dtype=torch.uint8, device=device) # (H, W, 3) torch.float32

    return image


def load_image_data(image_directory = "./data/dragon_scale/color", pose_indices=None, H=1440, W=1920, skip_every_n_images_for_training=60):
    
    rgb_from_photos = torch.zeros(size=(pose_indices.shape[0], H, W, 3), dtype=torch.uint8, device=torch.device('cpu'))

    for i, pose_index in enumerate(pose_indices):
        image_name = str(int(pose_index * skip_every_n_images_for_training)).zfill(6)
        image_path = "{}/{}.jpg".format(image_directory, image_name)
        rgb_from_photo = load_image(image_path)
        print("Loaded (R,G,B) data from photo {}.jpg of shape {}".format(image_name, rgb_from_photo.shape))
        rgb_from_photos[i,:,:,:] = rgb_from_photo    
    return rgb_from_photos

def get_face_normals(faces, blender_normals_csv_f_name):

    # load face normals (float32 in [-1,1]) generated from blender script
    face_normals = torch.zeros((faces.shape[0], 3), dtype=torch.float32)    
    
    with open(blender_normals_csv_f_name, 'r') as f:
        for i, line in enumerate(f):
            normals_i = line.split(',')
            face_normals[i] = torch.tensor([float(normals_i[0]), float(normals_i[1]), float(normals_i[2])])
        face_normals = torch.nn.functional.normalize(face_normals, dim=1, p=2).cpu()
    
    return face_normals


def get_initial_colors_and_normals(texels, face_normals):    
    texel_colors = torch.zeros( (texture_size*texture_size, 3), dtype=torch.uint8, device=torch.device('cpu'))
    texel_normals = torch.zeros( (texture_size*texture_size, 3), dtype=torch.float32, device=torch.device('cpu'))
    hit_texels = torch.zeros ((texture_size*texture_size,), dtype=torch.uint8, device=torch.device('cpu'))
    for i, texel_i in enumerate(texels):
        texel = texels[texel_i] 
        rgb = torch.stack(texel.rgb_samples, dim=0).float()
        face_indices = torch.stack(texel.face_indices, dim=0)
        normals = face_normals[face_indices]
        mean_rgb = torch.mean(rgb, dim=0)
        diff_from_mean_rgb = torch.sum((rgb-mean_rgb.unsqueeze(0))**2, dim=1)
        closest_rgb_index = torch.argmin(diff_from_mean_rgb) 

        texel_colors[int(texel_i)] = rgb[closest_rgb_index].to(torch.uint8)
        texel_normals[int(texel_i)] = normals[closest_rgb_index]
        hit_texels[int(texel_i)] = 1
            
    texel_colors[ torch.where(hit_texels==0)[0], : ] = 70
    return texel_colors, texel_normals
        

def compute_luminosity(rgb):    
    return (rgb * torch.tensor([0.2126, 0.7152, 0.0722]).unsqueeze(0))        


# sample_type: 'diffuse', 'reflectance', or 'all'
def sample_training_data(texels, max_samples_per_texel, sample_type='all'):

    dev = torch.device('cpu')
    rgb_tensor = torch.zeros((len(texels), max_samples_per_texel, 3), dtype=torch.uint8, device=dev)    
    cam_xyz_tensor = torch.zeros((len(texels), max_samples_per_texel, 3), dtype=torch.float32, device=dev)        
    surface_xyz_tensor = torch.zeros((len(texels), max_samples_per_texel, 3), dtype=torch.float32, device=dev)        
    texel_indices_tensor = torch.zeros((len(texels),), dtype=torch.int32, device=dev)
    
    for i, texel_i in enumerate(texels):

        texel = texels[texel_i]          
        rgb = torch.stack(texel.rgb_samples, dim=0)                
        cam_xyz = torch.stack(texel.cam_xyzs, dim=0)    
        surface_xyz = torch.stack(texel.surface_xyzs, dim=0)    
        texel_index = texel.texel_index        

        lum = compute_luminosity(rgb.to(torch.float32)/255.0)
        lum = lum.sum(dim=1)
        _, indices = torch.sort(lum, dim=0)
        
        # diffuse: take only the bottom quartile of samples in terms of luminosity
        if sample_type == 'diffuse':       
            low = 0
            high = int(rgb.shape[0] / 4) + 1
        # take reflectance: only the top three quartiles of samples in terms of luminosity    
        elif sample_type == 'reflectance':
            if rgb.shape[0] == 1:
                low = 0
                high = 1
            else:
                low = int(rgb.shape[0] / 4) + 1
                high = rgb.shape[0]
        else:
            low = 0
            high = rgb.shape[0]

        rgb = rgb[indices[low:high]]
        cam_xyz = cam_xyz[indices[low:high]]
        surface_xyz = surface_xyz[indices[low:high]]        
        lum = lum[indices[low:high]]

        # remove outliers, but only if enough remaining samples for the concept of outlier to make sense
        if rgb.shape[0] < 5:
            selected_samples = torch.arange(rgb.shape[0])
        else:
            Ctint = torch.zeros(rgb.shape[0], 3)
            Ctint[torch.where(lum > 0.0001)[0]] = ((rgb.to(torch.float32)/255.0)**2.2)[torch.where(lum > 0.0001)[0]] / lum[torch.where(lum > 0.0001)[0]].unsqueeze(1)
            Ctint_dev = torch.sqrt(torch.sum( ((Ctint - torch.mean(Ctint,dim=0))**2)  , dim=1))

            q1 = torch.quantile(Ctint_dev, 0.25)
            q3 = torch.quantile(Ctint_dev, 0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            selected_samples = torch.where( torch.logical_and(Ctint_dev >= lower_bound, Ctint_dev <= upper_bound) )[0]


        rgb = rgb[selected_samples]
        cam_xyz = cam_xyz[selected_samples]    
        surface_xyz = surface_xyz[selected_samples]                        
        
        # finally, uniformly randomly select max_samples_per_texel samples from the non-filtered indices
        #   --> lots of potential improvement possible here, e.g. using data augmentation instead of
        #       uniformly random sampling
        sample_indices = torch.randint(low=0, high=rgb.shape[0], size=(max_samples_per_texel,))
        rgb_tensor[i] = rgb[sample_indices]                        
        cam_xyz_tensor[i] = cam_xyz[sample_indices]        
        surface_xyz_tensor[i] = surface_xyz[sample_indices]
        texel_indices_tensor[i] = texel_index
        
        
    return rgb_tensor, cam_xyz_tensor, surface_xyz_tensor, texel_indices_tensor


def rgb_loss(x1, x2):
    
    loss = torch.mean( (x1 - x2) ** 2)    
    return loss


def make_lights_cube(center, l_distance):
    
    l_distance = torch.tensor(0.5, device=device)      
    center = center.to(torch.device('cuda:0'))
    n_lights = 8
    lights = torch.zeros((n_lights,3), dtype=torch.float32, device=device)
    lights[0] = center + l_distance * torch.nn.functional.normalize(torch.tensor([ 1,  1,  1], device=device)-center, dim=0, p=2.0)
    lights[1] = center + l_distance * torch.nn.functional.normalize(torch.tensor([ 1,  1, -1], device=device)-center, dim=0, p=2.0)
    lights[2] = center + l_distance * torch.nn.functional.normalize(torch.tensor([ 1, -1,  1], device=device)-center, dim=0, p=2.0)
    lights[3] = center + l_distance * torch.nn.functional.normalize(torch.tensor([ 1, -1, -1], device=device)-center, dim=0, p=2.0)
    lights[4] = center + l_distance * torch.nn.functional.normalize(torch.tensor([-1,  1,  1], device=device)-center, dim=0, p=2.0)
    lights[5] = center + l_distance * torch.nn.functional.normalize(torch.tensor([-1,  1, -1], device=device)-center, dim=0, p=2.0)
    lights[6] = center + l_distance * torch.nn.functional.normalize(torch.tensor([-1, -1,  1], device=device)-center, dim=0, p=2.0)
    lights[7] = center + l_distance * torch.nn.functional.normalize(torch.tensor([-1, -1, -1], device=device)-center, dim=0, p=2.0)
            
    return lights


def train(models, optimizers, texels, max_samples_per_texel, lights, visualize_gradient, extrinsics, verts, uv, normals_img):

    # set total number of 'diffuse' and 'reflectance' optimization steps
    number_of_steps = 1000

    # set epoch interval for switching optimization mode from 'diffuse' to 'reflectance' or vice-versa
    number_of_inner_epochs = 40
    test_frequency = 20
    device = torch.device('cuda:0')
    uv = uv.to(torch.device('cuda:0'))
    verts = verts.to(torch.device('cuda:0'))    
    extrinsics = extrinsics.to(torch.device('cuda:0'))
    lights = lights.to(torch.device('cuda:0'))

    for step in range(number_of_steps):

        print("Step {}".format(step))
        if step % 2 == 0: 
            optimization = 'diffuse'
        else:
            optimization = 'reflectance'

        print('Sampling training data...')
        with torch.no_grad():
            rgbs, cam_xyzs, surface_xyzs, texel_indices = sample_training_data(texels, max_samples_per_texel, sample_type=optimization)
        n_texels = rgbs.shape[0]
            
        # start in diffuse optimization mode        
        models['brdf'].set_mode(optimization)

        # massage everything into (n_faces * max_samples_per_texel, 3) tensors
        rgbs_data = rgbs.float().flatten(start_dim=0, end_dim=1)
        cam_xyzs_data = cam_xyzs.flatten(start_dim=0, end_dim=1)    
        surface_xyzs_data = surface_xyzs.flatten(start_dim=0, end_dim=1)
        texel_indices_data = texel_indices.unsqueeze(1).expand(n_texels, max_samples_per_texel).flatten(start_dim=0, end_dim=1)# texel_indices.unsqueeze(1).expand(n_texels, max_samples_per_texel).flatten(start_dim=0, end_dim=1)        
                
        # now all data is shape [n_texels * n_samples_per_texel, 3], batch it up
        n_batches = 100
        batch_size = int ((np.ceil(n_texels) / n_batches) * max_samples_per_texel)
        rgbs_data = torch.split(rgbs_data, batch_size)
        cam_xyzs_data = torch.split(cam_xyzs_data, batch_size)
        surface_xyzs_data = torch.split(surface_xyzs_data, batch_size)
        texel_indices_data = torch.split(texel_indices_data, batch_size)        
    
        n_batches = len(rgbs_data)        
                
        for epoch in range (number_of_inner_epochs):
        
            for optimizer in optimizers.values():
                optimizer.zero_grad()

            avg_loss_for_epoch = 0.0

            # test on the first epoch to see intitial conditions
            if epoch==0 and step==0:
                with torch.no_grad():            
                    brdf_params_test = models['brdf']()([0]).detach()            
                    textures_test = create_textures_from_brdf(brdf_params_test, normals_img)
                    test(textures_test, epoch + step*number_of_inner_epochs)                                        

            # create textures from newest pbr parameters to be used throughout the following batches
            brdf = models['brdf']()([0])
            textures = create_textures_from_brdf(brdf, normals_img)

            for batch_n in range(n_batches):
                            
                # load batch to GPU for each parameter required for rendering
                rgbs_batch = rgbs_data[batch_n].to(torch.device('cuda:0'))
                cam_xyzs_batch = cam_xyzs_data[batch_n].to(torch.device('cuda:0'))
                surface_xyzs_batch = surface_xyzs_data[batch_n].to(torch.device('cuda:0'))
                texel_indices_batch = texel_indices_data[batch_n].to(torch.device('cuda:0'))
                uv_coords_batch = texel_index_to_uv(texel_indices_batch)
                
                # render
                render_result = render_brdf_with_uvmap_and_textures(cam_xyzs_batch, surface_xyzs_batch, uv_coords_batch, textures, lights)

                # compute the loss for this batch
                loss = rgb_loss(render_result, rgbs_batch / 255.0)
                avg_loss_for_epoch = avg_loss_for_epoch + loss.item()
                
                # accrue gradients for batch
                # note that retain_graph is enabled here to avoid needlessly invoking the model at each batch
                loss.backward(create_graph=False, retain_graph=True)
                
            # update the model by performing one step of gradient descent
            for optimizer in optimizers.values():
                optimizer.step()
                        
            with torch.no_grad():
                # print data to track training epoch-by-epoch
                print('Epoch {}:  LOSS: {}'.format(epoch+step*number_of_inner_epochs, avg_loss_for_epoch / n_batches))
                brdf_params_test = models['brdf']()([0]).detach()
                mean_brdf = torch.mean(brdf_params_test, dim=0)
                print("Average brdf parameters:")
                print(mean_brdf)
                                                            
                # extract and test current result
                if (epoch+1) % test_frequency == 0:                
                    print("Saving brdf")
                    torch.save(brdf_params_test, '{}/brdf_{}.pt'.format(models_dir, epoch + step*number_of_inner_epochs))

                    if visualize_gradient:
                        with torch.no_grad():
                            models['brdf'].visualize_texture_gradient(epoch + step*number_of_inner_epochs, batch_n)
                    
                    textures_test = create_textures_from_brdf(brdf_params_test, normals_img)
                    test(textures_test, epoch + step*number_of_inner_epochs)
                    

def test(textures, epoch):

    print("Rendering for training views")                                                  
    render_training_views_with_uvmap_and_textures(extrinsics[:, :3, 3][:10].to(torch.device('cuda:0')), verts, bary_coords, uv, textures, posepix2face, faces, lights, epoch)
    f_name = '{}/{}_{}_{}x{}.glb'.format(gltf_dir, dataset, epoch, texture_size, texture_size)
    save_gltf(textures, uv, f_name)
    imageio.imwrite('{}/color_{}.png'.format(texture_visualization_dir, epoch),  (textures[0].cpu().numpy()).astype(np.uint8))
    imageio.imwrite('{}/roughness_{}.png'.format(texture_visualization_dir, epoch), (textures[1].cpu().numpy()).astype(np.uint8))


# precompute bary_coords-interpoloated parameters needed in training
def compute_interpolated_properties(posepix2face, bary_coords, uv, extrinsics, faces, verts):

    faces = faces.cpu()
    uv = uv.cpu()                        
    faces = faces.to(torch.device('cuda:0'))
    uv = uv.to(torch.device('cuda:0'))     
    verts = verts.to(torch.device('cuda:0'))   

    n_poses = extrinsics.shape[0]

    interpolated_face_uvs = []
    interpolated_face_xyzs = []
    for pose_i in range(n_poses):        
        print("Computing interpolated face properties for view {}... ".format(pose_i))        

        # get the bary_coords and face indices associated with pixels hit during rasterization in pose_i
        bary_coords_pose_i = bary_coords[pose_i].to(torch.device('cuda:0'))        
        pixels_hit = torch.where(posepix2face[pose_i] != -1)[0]        
        faces_hit = posepix2face[pose_i, pixels_hit]
        n_hit_faces = pixels_hit.shape[0]        
        
        bary_coords_for_pose = bary_coords_pose_i[pixels_hit].squeeze(0)
        face_verts_indices = faces[faces_hit]
        face_verts_uv = uv[face_verts_indices]

        face_verts_indices = faces[faces_hit]                
        face_verts_uv = uv[face_verts_indices]
        face_verts_xyz = verts[face_verts_indices]    
        
        # matmul needs to be batched due to inefficient memory uage
        # bmm could also be used, but requires 3D tensors, which would require some tricky indexing
        batch_size = 5000
        n_batches = math.ceil(n_hit_faces / batch_size)
        face_uv_out_pose_i = torch.zeros((n_hit_faces,2), dtype=torch.float32, device=device)
        face_xyz_out_pose_i = torch.zeros((n_hit_faces,3), dtype=torch.float32, device=device)                
        
        for batch in range(n_batches):
            start_index = batch * batch_size
            end_index = min(batch * batch_size + batch_size, n_hit_faces * (n_batches-1))        

            bary_coords_for_pose_batch = bary_coords_for_pose[start_index:end_index]
            face_verts_uv_batch = face_verts_uv[start_index:end_index]
            face_verts_xyz_batch = face_verts_xyz[start_index:end_index]
            
            # interpolate, for each face triangle in batch, their vertex uvs and xyzs to get 
            #   their uvs and xyzs at the location of the triangle barycenter
            face_uv = torch.matmul(bary_coords_for_pose_batch.unsqueeze(1), face_verts_uv_batch)
            face_xyz = torch.matmul(bary_coords_for_pose_batch.unsqueeze(1), face_verts_xyz_batch)

            face_uv_out_pose_i[start_index:end_index] = face_uv.squeeze(1)
            face_xyz_out_pose_i[start_index:end_index] = face_xyz.squeeze(1)

        interpolated_face_uvs.append(face_uv_out_pose_i)
        interpolated_face_xyzs.append(face_xyz_out_pose_i)

        face_uv_out_pose_i = face_uv_out_pose_i.cpu()
        face_xyz_out_pose_i = face_xyz_out_pose_i.cpu()
        bary_coords_pose_i = bary_coords_pose_i.cpu()


    return interpolated_face_uvs, interpolated_face_xyzs
    

def prepare_training_data(posepix2face, bary_coords, uv, extrinsics, faces, verts):
        
        face_uvs, face_xyzs = compute_interpolated_properties(posepix2face, bary_coords, uv, extrinsics, faces, verts)

        texels = {}
        
        faces = faces.cpu()
        uv = uv.cpu()                        

        n_invalid_texels = 0
        faces = faces.to(torch.device('cuda:0'))
        uv = uv.to(torch.device('cuda:0'))     
        verts = verts.to(torch.device('cuda:0'))   

        n_poses = extrinsics.shape[0]
        for pose_i in range(n_poses):

            rgb_from_photos_pose_i = rgb_from_photos[pose_i].to(torch.device('cuda:0'))
            face_uv_pose_i = face_uvs[pose_i].to(torch.device('cuda:0'))
            face_xyz_pose_i = face_xyzs[pose_i].to(torch.device('cuda:0'))
            pix2face_i = posepix2face[pose_i].to(torch.device('cuda:0'))

            print("Packing up data for view {}... ".format(pose_i))
            pose_xyz = extrinsics[pose_i, :3, 3].to(torch.device('cuda:0'))
            
            # get the barycentric coordinates for all hits
            pixels_hit = torch.where(pix2face_i != -1)[0]
            faces_hit = pix2face_i[pixels_hit]
        
            rgb_samples_for_pose = rgb_from_photos_pose_i.reshape(W*H, 3)[pixels_hit]

            # update rgb samples, camera pose samples, and texel xyz (if not initialized already) for texels seen by this pose
            for i, face_index in enumerate(faces_hit):
              
                face_uv = face_uv_pose_i[i]
                surface_xyz = face_xyz_pose_i[i]                                                
                texel_i = uv_to_texel_index(face_uv)

                if (texel_i < 0):
                    n_invalid_texels = n_invalid_texels + 1
                    print("texel invalid")
                    continue
                
                face_surface_xyz = surface_xyz
                face_pose_xyz = pose_xyz
                face_rgb_sample = rgb_samples_for_pose[i]                                
                                
                if str(texel_i.item()) not in texels.keys():
                    texels[str(texel_i.item())] = TexelSamples(texel_i.item(), [face_rgb_sample.cpu()], [face_pose_xyz.cpu()], [face_surface_xyz.cpu()], [face_index.cpu()])

                else:                    
                    texels[str(texel_i.item())].add_sample(face_rgb_sample.cpu(), face_pose_xyz.cpu(), face_surface_xyz.cpu(), face_index.int().cpu())

            face_uv_pose_i = face_uv_pose_i.cpu()
            face_xyz_pose_i = face_xyz_pose_i.cpu()
            pix2face_i = pix2face_i.cpu()
            rgb_from_photos[pose_i] = rgb_from_photos[pose_i].cpu()
            extrinsics[pose_i, :3, 3] = extrinsics[pose_i, :3, 3].cpu()
            pose_xyz = pose_xyz.cpu()

        n_texels_hit = 0
        hits = 0
        max_texels_hit = -float('inf')
        for i, (k,v) in enumerate(texels.items()):            
            n_hits_for_texel = len(v.rgb_samples)
            if n_hits_for_texel > 0:
                hits = hits + len(v.rgb_samples)
                if n_hits_for_texel > max_texels_hit:
                    max_texels_hit = n_hits_for_texel
                                
            n_texels_hit = n_texels_hit + 1
            
        print("Number of texels hit: {}".format(n_texels_hit))
        print("Max number of texels hit: {}".format(max_texels_hit))
        print("Avg hits per texel: {}".format( (hits / n_texels_hit)))
        print("Percentage of texels with >= 1 hits: {}".format((n_texels_hit / (texture_size**2))))  
        print("Number of invalid texels: {}".format(n_invalid_texels))
        return texels


if __name__ == '__main__':


    # load intrinsics/extrinsics
    with torch.no_grad():


        # flags for saving/loading rasterizer results
        load_precomputed_rasterizer_data = True
        save_rast_result = False

        H = 1440
        W = 1920        
        extrinsics = torch.from_numpy(np.load('./{}/camera_extrinsics.npy'.format(input_data_dir)))
        n_total_poses = extrinsics.shape[0]
        intrinsics = torch.from_numpy(np.load('./{}/camera_intrinsics.npy'.format(input_data_dir)))
        image_directory = '{}/{}'.format('./data/dragon_scale_large', 'color')        
        mesh_f_name = './{}/dragon_triangulated_beautified_remeshed_smooth_10_ratio_0.70_instant_meshed_uv_mapped_angle_0.80_island_margin_0.003_face_weight_0.0.ply'.format(input_data_dir)
        mesh = load_mesh(mesh_f_name)
        
        # hard coded currently
        # TODO: follow same input convension as other input data
        center = torch.tensor([0.029568, -0.275911, -0.2385711]).to(torch.device('cpu'))

        # placeholder that does the job currently smiley face
        lights = make_lights_cube(center, 0.5)        
                
        skip_every_n_pose_indices = 4
        n_considered_poses = extrinsics[::skip_every_n_pose_indices].shape[0]    
                
        max_samples_per_texel = 20
                        
        verts = mesh.verts_list()[0].cpu()
        faces = mesh.faces_list()[0].cpu()
        face_vertices = verts[faces]
        n_faces = faces.shape[0]
    
        # rasterize
        if load_precomputed_rasterizer_data:
            f_name = './{}/posepix2face_dragon_scale_test.pt'.format(input_data_dir)
            posepix2face = torch.load(f_name)
            f_name = './{}/bary_coords_dragon_scale_test.pt'.format(input_data_dir)
            bary_coords = torch.load(f_name)
        else:
            posepix2face, bary_coords = rasterize(mesh, extrinsics.cpu().numpy(), intrinsics.cpu().numpy(), H, W, save_result=save_rast_result, save_renders=True)
        
        rgb_from_photos = load_image_data(image_directory=image_directory, pose_indices=np.arange(start=0, step=skip_every_n_pose_indices, stop=n_total_poses), H=H, W=W)

        posepix2face = posepix2face[::skip_every_n_pose_indices][:n_considered_poses]
        bary_coords =  bary_coords[::skip_every_n_pose_indices][:n_considered_poses]
        extrinsics = extrinsics[::skip_every_n_pose_indices][:n_considered_poses]        
        rgb_from_photos = rgb_from_photos[:n_considered_poses]
        
        mesh = trimesh.load(mesh_f_name, process=False)
        vertices = torch.from_numpy(mesh.vertices).to(torch.float32).to(torch.device('cuda:0'))        
        
        faces = torch.from_numpy(np.asarray(mesh.faces)).cpu()
        uv = torch.from_numpy(np.asarray(mesh.visual.uv)).to(torch.float32).cpu()                

        texels = prepare_training_data(posepix2face, bary_coords, uv, extrinsics, faces, verts)
        n_texels_in_texture = texture_size**2

        # estimate initial diffuse colors and normals
        blender_normals_csv_f_name = '{}/dragon_triangulated_beautified_remeshed_smooth_10_ratio_0.70_instant_meshed_uv_mapped_angle_0.80_island_margin_0.003_face_weight_0.0.csv'.format(input_data_dir)
        face_normals = get_face_normals(faces, blender_normals_csv_f_name).cpu()
        print(face_normals)

        initial_colors, initial_normals = get_initial_colors_and_normals(texels, face_normals)
        initial_colors = initial_colors.reshape((texture_size,texture_size,3)).cpu()
        initial_colors = initial_colors.to(torch.float32) / 255.0

        initial_normals = initial_normals.reshape((texture_size,texture_size,3)).cpu()

        # pick something reasonable for initial roughness
        initial_roughness = torch.zeros((texture_size, texture_size, 3), dtype=torch.float32).cpu()
        initial_roughness[:, :, 1] = 255.0
        initial_roughness[:, :, 1] = initial_roughness[:, :, 1] * 0.9
        initial_roughness = initial_roughness / 255.0
      
        initial_textures = [
            initial_colors,
            initial_roughness,
            initial_normals,
        ]                                                
          
    # initialize models
    models = {}
    optimizers = {}            
        
    models['brdf'] = torch.compile(DisneyBRDFModel(n_texels_in_texture, initial_textures=initial_textures).to(torch.device('cuda:0')), mode='max-autotune')
    models['brdf'].to(torch.device('cuda:0'))    
    optimizers['brdf'] = torch.optim.AdamW(models['brdf'].parameters(), lr=0.001)
        
    # fit brdf parameters
    brdf_params = train(models, optimizers, texels, max_samples_per_texel, lights, visualize_gradient=True, extrinsics = extrinsics, verts=verts, uv=uv, normals_img=initial_textures[2])




    