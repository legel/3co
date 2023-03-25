import torch
import torch.nn as nn
import struct
from PIL import Image
import imageio
import numpy as np
import random, math
import open3d as o3d
import sys, os, shutil, copy, glob, json
from learn import *
from utils.camera import *
from pbr_rendering import *
from torch.autograd import Variable

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

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

import gc

set_randomness()
torch.set_float32_matmul_precision('high')
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose=False      

device = torch.device('cuda:0')
#device = torch.device('cpu')


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

    rgb_texture = TexturesVertex(verts_features=[vertex_colors_tensor]).to(device)
    mesh = Meshes(verts=[vertices_tensor], faces=[triangles_tensor], textures=rgb_texture).to(device)

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

    pixels_to_mesh_face_indices = mesh_rasterizer.forward(mesh).pix_to_face
    pixels_to_mesh_face_indices = pixels_to_mesh_face_indices[0,:,:,0] # for the 0th mesh, for all H,W pixels, get the first face index
    
    pixels_that_hit_face = len(torch.where(pixels_to_mesh_face_indices != -1)[0])
    print("          -> Pixel Raycast Hits: {:,} Faces".format(pixels_that_hit_face))

    if save_renders:
        soft_phong_shader = SoftPhongShader(device=device, cameras=cameras, lights=lights) # Shader all systems go

        # Create a Phong renderer by composing a rasterizer and a shader
        renderer = MeshRenderer(rasterizer=mesh_rasterizer, shader=soft_phong_shader) # Mesh rendering ready
        image = renderer(mesh.to(device=device))  # Boom! Render that mesh :)
        image = (image.detach().cpu().numpy()[0, ..., :3] * 255).astype(np.uint8)

        return image, pixels_to_mesh_face_indices

    else:
        return None, pixels_to_mesh_face_indices
    

def rasterize(mesh, camera_extrinsics, camera_intrinsics, H, W, save_result=False):
    
    number_of_poses = camera_extrinsics.shape[0]
    number_of_faces = mesh.num_faces_per_mesh()
    
    number_of_vertices = mesh.num_verts_per_mesh()
    number_of_rows = H
    number_of_cols = W

    nerf_camera_extrinsics = np.copy(camera_extrinsics)

    # now, we define a data structure that maps for every face, for all poses, at most 1 pixel row & pixel col index that may hit it
    # by default, we set these indices = -1, indicating that the face is not hit for a given pose     
    faces = mesh.faces_list()[0] 
    vertices = mesh.verts_list()[0]
    face_vertices = vertices[faces]
    faces_to_pixels = torch.full(size=(number_of_faces, number_of_poses, 2), fill_value=-1, dtype=torch.int32, device=device)
    face_normals = torch.zeros(number_of_faces, 3, dtype=torch.float32, device=device)    
    vertex_normals = torch.zeros(number_of_vertices, 3, dtype=torch.float32, device=device)
    posepix2face = torch.empty( (number_of_poses, H*W), dtype=torch.int32, device=device)

    #for pose_index in range(0,number_of_poses):
    for pose_index in range(0,number_of_poses):
        extrinsics = correct_camera_extrinsics_for_pytorch3d(camera_extrinsics[pose_index,:,:])
        intrinsics = camera_intrinsics[pose_index,:,:]

        intrinsics = torch.from_numpy(intrinsics) 
        extrinsics = torch.from_numpy(extrinsics)

        save_renders = True
        print("{}.png ({}.jpg)".format(pose_index, pose_index*60))

        image, pix2face = render_mesh( 
            extrinsics=extrinsics, 
            intrinsics=intrinsics, 
            mesh=mesh,
            image_size=(number_of_rows, number_of_cols),
            show_interactive_3D_figure=False,
            save_renders=save_renders
        )

        # (n_poses, H*W)        
        
        posepix2face[pose_index, :] = pix2face.clone().reshape(H*W)


        if save_renders:
            im = Image.fromarray(image)
            im.save('rast_imgs/{}.png'.format(pose_index))                                

        # get the pixel (row, col) for all raycast hits
        rows_hit, cols_hit = torch.where(pix2face != -1)
        rows_hit = rows_hit.to(dtype=torch.int32)
        cols_hit = cols_hit.to(dtype=torch.int32)

        # get the face indices
        faces_hit = pix2face[rows_hit, cols_hit]        

        # assign for all face indices that were hit, for this pose index, values for the elements of row,col (0,1)
        faces_to_pixels[faces_hit, pose_index, 0] = rows_hit
        faces_to_pixels[faces_hit, pose_index, 1] = cols_hit

        total_hits = torch.where(faces_to_pixels[:,:,0] != -1)[0]
        print("  Total surface hits = {:,}".format(total_hits.shape[0]))

        # approximate face normal by face->camera direction vector
        camera_xyz = torch.from_numpy(nerf_camera_extrinsics[pose_index, :3, 3]).to(torch.device(device))    
        
        faces_xyz = torch.mean(face_vertices[faces_hit], dim=1)

        face_normals[faces_hit] = camera_xyz.unsqueeze(0).expand(faces_xyz.size()[0], 3) - faces_xyz

        face_normals[faces_hit] = torch.nn.functional.normalize(face_normals[faces_hit], p=2, dim=1)

        vertex_normals[(faces[faces_hit])[:,0]] = face_normals[faces_hit]
        vertex_normals[(faces[faces_hit])[:,1]] = face_normals[faces_hit]
        vertex_normals[(faces[faces_hit])[:,2]] = face_normals[faces_hit]


    if save_result:
        save_rasterizer_result(
            'rast_imgs/faces_to_pixels.npy', faces_to_pixels, 
            'rast_imgs/posepix2face.npy', posepix2face
        )


    return (faces_to_pixels, posepix2face)
        

def save_rasterizer_result(f2p_f_name, faces_to_pixels, posepix2face_f_name, posepix2face):
    
    with open(f2p_f_name, 'wb') as f:
        np.save(f, faces_to_pixels.detach().cpu().numpy())            
    with open(posepix2face_f_name, 'wb') as f:
        np.save(f, posepix2face.detach().cpu().numpy())                        


###########################################################################################
################## Inverse Rendering ######################################################
###########################################################################################
class LightModel(nn.Module):

    def __init__(self, initial_intensity_scale):
        super(LightModel, self).__init__()
        intensity_scale = initial_intensity_scale.clone().detach() #, device=torch.device('cuda:0'))
        self.intensity_scale = nn.Parameter(intensity_scale, requires_grad=False)

    def forward(self, i=None):        
        intensity_scale = self.intensity_scale * 1.0
        intensity_scale = torch.clamp(intensity_scale, min=0.0, max=2.0)
        return intensity_scale
   

class GeometryModel(nn.Module):

    def __init__(self, initial_normals):
        super(GeometryModel, self).__init__()
        self.n_faces = initial_normals.size()[0]
        self.normals = nn.Parameter(initial_normals.to(torch.device('cuda:0')), requires_grad=False)
        self.r = nn.Parameter(torch.zeros(size=(self.n_faces, 3), dtype=torch.float32, device=torch.device('cuda:0')), requires_grad=True)  # (N, 3)        

    def forward(self, i=None):
        r = self.r
        delta_n = self.make_pose(r)        
        normals = delta_n @ self.normals.unsqueeze(2)
        return normals.squeeze(2)
                    
    def make_pose(self, r):
        """
        :param r:  (N, 3, ) axis-angle
        :return:   (N, 4, 4)
        """        
        R = self.Exp_batch(r)  # (N, 3, 3)
        return R            

    def Exp_batch(self, r):
        """so(3) vector to SO(3) matrix
        :param r: (N, 3, ) axis-angle
        :return:  (N, 3, 3)
        """
        batch_size = r.shape[0]
        skew_r = self.vec2skew_batch(r)  # (N, 3, 3)
        norm_r = r.norm() + torch.tensor([1e-15]).to(torch.device('cuda:0'))
        eye = torch.eye(3, dtype=torch.float32, device=r.device)    
        batch_eye = eye.repeat(batch_size, 1, 1)
        a = skew_r @ skew_r
        R = batch_eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)        
        return R    

    def vec2skew_batch(self, v):
        """
        :param v:  (N, 3, ) torch tensor
        :return:   (N, 3, 3)
        """
        number_of_samples = v.shape[0]
        zero = torch.zeros((number_of_samples,1), dtype=torch.float32, device=v.device)
        skew_v0 = torch.cat([ zero,    -v[:,2:3],   v[:,1:2]], dim=1)  # (N, 3, 1)
        skew_v1 = torch.cat([ v[:,2:3],   zero,    -v[:,0:1]], dim=1)  # (N, 3, 1)
        skew_v2 = torch.cat([-v[:,1:2],   v[:,0:1],   zero], dim=1)    # (N, 3, 1)
        skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=1)       # (N, 3, 3)
        #skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=2)       # (N, 3, 3)                        
        return skew_v  # (N, 3, 3)
    

class IRModel(nn.Module):

    def __init__(self, n_faces, initial_face_colors=None):

        super(IRModel, self).__init__()
        brdf_params = {}
        brdf_params['metallic'] = 0.0
        brdf_params['subsurface'] = 0.0  
        brdf_params['specular'] = 0.0
        brdf_params['roughness'] = 0.0
        brdf_params['specularTint'] = 0.0
        brdf_params['anisotropic'] = 0.0
        brdf_params['sheen'] = 0.0
        brdf_params['sheenTint'] = 0.0
        brdf_params['clearcoat'] = 0.0
        brdf_params['clearcoatGloss'] = 0.0
        brdf_params['red'] = torch.rand(1)
        brdf_params['green'] = torch.rand(1)
        brdf_params['blue'] = torch.rand(1)

        brdf_params_tensor = torch.tensor([
            brdf_params['metallic'], brdf_params['subsurface'], brdf_params['specular'],
            brdf_params['roughness'], brdf_params['specularTint'], brdf_params['anisotropic'],
            brdf_params['sheen'], brdf_params['sheenTint'], brdf_params['clearcoat'],
            brdf_params['clearcoatGloss'], brdf_params['red'], brdf_params['green'], brdf_params['blue'],
        ])
            
        brdf_params_tensor_expand = brdf_params_tensor.repeat( (n_faces, 1) )

        if initial_face_colors is not None:            
            brdf_params_tensor_expand[:, -3:] = initial_face_colors

        self.brdf_params = nn.Parameter(brdf_params_tensor_expand, requires_grad=True)
        

    def forward(self, i=None):

        brdf = 1.0 * self.brdf_params
                                
        with torch.no_grad():            
            brdf[:] = torch.clamp(brdf, min=0.0, max=1.0)
            #brdf[:, 0] = 0
            #brdf[:, 1] = 0
            #brdf[:, 2] = 0

            #brdf[:, 4] = 0
            #brdf[:, 5] = 0
            #brdf[:, 6] = 0
            #brdf[:, 7] = 0
            #brdf[:, 8] = 0
            #brdf[:, 9] = 0

        return brdf
    
    # save images representing brdf gradient for all poses and faces at a particular epoch
    def visualize_gradient(self, posepix2face, epoch, batch_n, H=1440, W=1920):
        brdf_params = self.brdf_params.to(torch.device('cpu'))
        grad = brdf_params.grad
        if grad == None:
            print("grad is none")
            return        

        posepix2face = posepix2face.to(torch.device('cpu'))
        print('visualizing gradient')
        
        
        vis_grad = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        brdf_param_names = self.get_brdf_param_names()

        
        os.makedirs('/home/rob/research_code/3co/research/nerf/rast_imgs/renders/gradients/', exist_ok=True)
        with open('./rast_imgs/renders/gradients/stats_{}.txt'.format(epoch), 'w') as f:        
            for pose_i in range(posepix2face.size()[0]):

                # posepix2face: (n_poses, H*W)
                
                hit_pixels = torch.where(posepix2face[pose_i] != -1)[0].to(torch.device('cpu'))                                
                hit_face_indices = posepix2face[pose_i, hit_pixels] 
                n_hit_indices = hit_face_indices.size()[0]
                        
                f.write("----- pose {} -----\n".format(pose_i))
                for p in vis_grad:

                    grad_p = grad[hit_face_indices, p]
                    rendered_grad = torch.zeros((H*W), dtype=torch.float32, device=torch.device('cpu'))
                    rendered_grad[hit_pixels] = grad_p
                    rendered_grad = rendered_grad.reshape(H, W)                                
                    min_grad = grad_p.min()                
                    max_grad = grad_p.max()
                    f.write('min grad for {}: {}\n'.format(brdf_param_names[p], min_grad.item()))
                    f.write('max grad for {}: {}\n'.format(brdf_param_names[p], max_grad.item()))
                    
                    rendered_grad = rendered_grad - min_grad
                    max_grad = rendered_grad.max()
                    rendered_grad = rendered_grad / max_grad
                    rendered_grad = (rendered_grad.cpu().numpy() * 255).astype(np.uint8)
                    f_name = './rast_imgs/renders/gradients/{}/grad_pose{}_epoch{}_batch{}.png'.format(brdf_param_names[p], pose_i, epoch, batch_n)
                    os.makedirs('/home/rob/research_code/3co/research/nerf/rast_imgs/renders/gradients/{}'.format(brdf_param_names[p]), exist_ok=True)
                    imageio.imwrite(f_name, rendered_grad)



    def get_brdf_param_names(self):
        return [
            'metallic', 'subsurface', 'specular', 'roughness', 'specularTint', 'anisotropic',
            'sheen', 'sheenTint', 'clearcoat', 'clearcoatGloss', 'red', 'green', 'blue',
        ]


class TexelSamples:

    def __init__(self, face_index, rgb_samples, cam_xyzs, texel_xyz):
        self.face_index = face_index
        self.rgb_samples = rgb_samples
        self.cam_xyzs = cam_xyzs
        self.texel_xyz = texel_xyz
        #self.normal = normal
        

    def add_sample(self, rgb_sample, cam_xyz):
        self.rgb_samples.append(rgb_sample)
        self.cam_xyzs.append(cam_xyz)


def render_brdf_on_training_views(brdf_params, cam_xyzs, surface_xyzs, lights, light_intensity_scale, face_normals, posepix2face, H, W, epoch):
    
    n_poses = cam_xyzs.size()[0]
    n_pixels = posepix2face.size()[1]    

    surface_xyzs = surface_xyzs.to(device)
    cam_xyzs = cam_xyzs.to(device)
    face_normals = face_normals.to(device)
    brdf_params = brdf_params.to(device)
    os.makedirs('/home/rob/research_code/3co/research/nerf/rast_imgs/renders/color/', exist_ok=True)
    os.makedirs('/home/rob/research_code/3co/research/nerf/rast_imgs/renders/roughness/', exist_ok=True)
    os.makedirs('/home/rob/research_code/3co/research/nerf/rast_imgs/renders/normals/', exist_ok=True)
    
    for i in range(n_poses):
        pix2face = posepix2face[i].to(device)

        hit_indices = torch.where(pix2face != -1)[0].to(device)
        n_hit_indices = hit_indices.size()[0]        
    
        rendered_img = torch.zeros((n_pixels, 3), dtype=torch.float32, device=device)
        rendered_roughness = torch.zeros((n_pixels,), dtype=torch.float32, device=device)
        rendered_normals = torch.zeros((n_pixels,3), dtype=torch.float32, device=device)
        view_cam_xyzs = cam_xyzs[i].unsqueeze(0).expand(n_hit_indices, 3).to(device)
        view_normals = face_normals[pix2face[hit_indices]].to(device)        
        view_surface_xyzs = surface_xyzs[pix2face[hit_indices]].to(device)

        view_brdf_params = brdf_params[pix2face[hit_indices]].to(device)
                  
        #render_result, good_idx = render_brdf(view_light_xyzs, view_normals, view_cam_xyzs, view_surface_xyzs, view_brdf_params)
        render_result, good_idx = render_brdf(view_normals, view_cam_xyzs, view_surface_xyzs, view_brdf_params, lights, light_intensity_scale)
        good_idx = torch.tensor(range(render_result.size()[0]), device=device)
        rendered_img[hit_indices] = render_result
        rendered_img = rendered_img.reshape(H,W,3)
        rendered_img = (rendered_img.cpu().numpy() * 255).astype(np.uint8)
        label = str(i).zfill(4)
        f_name = './rast_imgs/renders/color/view_{}_{}.png'.format(label, str(epoch).zfill(6))
        imageio.imwrite(f_name, rendered_img)
        
        rendered_roughness[hit_indices] = view_brdf_params[:,3]
        rendered_roughness[hit_indices] = torch.minimum(torch.tensor(1.0), rendered_roughness[hit_indices] * 5.0)
        rendered_roughness = rendered_roughness.reshape(H,W).unsqueeze(2).expand(H,W,3)
        rendered_roughness = (rendered_roughness.cpu().numpy() * 255).astype(np.uint8)
        label = str(i).zfill(4)
        f_name = './rast_imgs/renders/roughness/view_{}_{}.png'.format(label, str(epoch).zfill(6))
        imageio.imwrite(f_name, rendered_roughness)

        rendered_normals[hit_indices] = view_normals

        #torch.set_printoptions(precision=5)      
        #torch.set_printoptions(threshold=10_000)      
        #probed_pixels = torch.tensor(range(int(W*H*0.5) + 100*W, int(W*H*0.5) + 101*W)).to(device)
        #hit_probed_pixels = torch.where(pix2face[probed_pixels]!=-1)[0]
        #probed_normals = face_normals[pix2face[probed_pixels[hit_probed_pixels]]]
        #print (probed_normals)
        #torch.set_printoptions(precision='default')     
        #rendered_normals[probed_pixels[hit_probed_pixels]] = torch.tensor([-1.0, -1.0, -1.0]).to(device)#face_normals[pix2face[hit_probed_pixels]]

        rendered_normals = rendered_normals.reshape(H,W,3)
        rendered_normals= ( (1 + rendered_normals).cpu().numpy() * 128).astype(np.uint8)
        label = str(i).zfill(4)
        f_name = './rast_imgs/renders/normals/view_{}_{}.png'.format(label, str(epoch).zfill(6))
        imageio.imwrite(f_name, rendered_normals)
                   


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



def render_brdf_on_mesh(mesh, f_name, brdf_params, face_vertex_indices, vertices):
    colors = brdf_params[:,-3:].cpu()
    vertex_colors = torch.zeros((vertices.size()[0], 3))
    member_of_n_faces = torch.zeros((vertices.size()[0],))
    for i, face in enumerate(face_vertex_indices):
        vertex_colors[face,:] = vertex_colors[face,:] + colors[i,:]        
        member_of_n_faces[face] = member_of_n_faces[face] + 1
    
    vertex_colors = vertex_colors / member_of_n_faces.unsqueeze(1).expand(vertices.size()[0],3)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors.detach().cpu().numpy())    
    o3d.io.write_triangle_mesh(f_name, mesh, write_ascii = True)    

    

def load_image(image_path):
    image_data = imageio.imread(image_path)

    # clip out alpha channel, if it exists
    image_data = image_data[:, :, :3]  # (H, W, 3)

    # convert to torch format and normalize between 0-1.0
    image = torch.from_numpy(image_data).to(dtype=torch.uint8, device=device) # (H, W, 3) torch.float32

    return image


def load_image_data(image_directory = "./data/dragon_scale/color", n_poses=5, H=1440, W=1920, skip_every_n_images_for_training=60):

    #rgb_from_photos = torch.zeros(size=(n_poses, H, W, 3), dtype=torch.uint8, device=torch.device('cpu'))
    rgb_from_photos = torch.zeros(size=(n_poses, H, W, 3), dtype=torch.uint8, device=torch.device('cpu'))

    for pose_index in range (0, n_poses):        
        image_name = str(int(pose_index * skip_every_n_images_for_training)).zfill(6)
        image_path = "{}/{}.jpg".format(image_directory, image_name)
        rgb_from_photo = load_image(image_path)
        print("Loaded (R,G,B) data from photo {}.jpg of shape {}".format(image_name, rgb_from_photo.shape))

        rgb_from_photos[pose_index,:,:,:] = rgb_from_photo    
    return rgb_from_photos



# for this version it might be best to skip the texel representation and go straight to tensors, 
# but this allows for more flexible data storage and should make changes to training easier to implement
def unpack_texels(texels, max_samples_per_face):

    dev = torch.device('cpu')
    rgb_tensor = torch.zeros((len(texels), max_samples_per_face, 3), dtype=torch.uint8, device=dev)
    cam_xyz_tensor = torch.zeros((len(texels), max_samples_per_face, 3), dtype=torch.float32, device=dev)
    surface_xyz_tensor = torch.zeros((len(texels), 3), dtype=torch.float32, device=dev)
    texel_indices = torch.zeros((len(texels),), dtype=torch.int32, device=dev)
    distinct_sample_ns = torch.zeros((len(texels),), dtype=torch.int32, device=dev)

    for i, texel_i in enumerate(texels):
        texel = texels[texel_i]  
        

        rgb = torch.stack(texel.rgb_samples, dim=0)
                
        cam_xyz = torch.stack(texel.cam_xyzs, dim=0)
        surface_xyz = texel.texel_xyz
        
        sample_indices = torch.randint(low=0, high=rgb.size()[0], size=(max_samples_per_face,))
        rgb_tensor[i] = rgb[sample_indices]
        cam_xyz_tensor[i] = cam_xyz[sample_indices]

        surface_xyz_tensor[i] = surface_xyz
        texel_indices[i] = int(texel_i)
        distinct_sample_ns[i] = min(rgb.size()[0], max_samples_per_face)


        """
        # resample to generate additional samples for this texel for n_samples total   
        n_samples = min(max_samples_per_face, rgb.size()[0])
        n_empty_indices = max_samples_per_face - n_samples

        rgb_tensor[i, : n_samples, :] = rgb[ : n_samples]
        
        cam_xyz_tensor[i, : n_samples, :] = cam_xyz[ : n_samples : ]

        if n_empty_indices > 0:
            resample_indices = torch.randint(low=0, high=rgb.size()[0], size=(n_empty_indices,))
            rgb_tensor[i, n_samples : ] = rgb[resample_indices]
            cam_xyz_tensor[i, n_samples :] = cam_xyz[resample_indices]

        surface_xyz_tensor[i] = surface_xyz 
        texel_indices[i] = int(texel_i)     
        distinct_sample_ns[i] = rgb.size()[0]

        distinct_sample_ns[i] = max(rgb.size()[0], max_samples_per_face)
        """        

        
        

    return (rgb_tensor, cam_xyz_tensor, surface_xyz_tensor, texel_indices, distinct_sample_ns)


def rgb_loss(x1, x2, distinct_sample_ns, max_samples_per_face, light_intensity_scale):

    loss = torch.mean( ((x1 - x2)*(distinct_sample_ns / max_samples_per_face).unsqueeze(1).expand(x1.size()[0], 3) ) ** 2)    
    loss = loss + (torch.abs(1.0 - light_intensity_scale)/10.0)**2
    return loss


def make_lights_cube(center, l_distance):
    
    lights = []

    l_distance = torch.tensor(0.5, device=device)      
    lights.append(center + l_distance * torch.nn.functional.normalize(torch.tensor([ 1,  1,  1], device=device)-center, dim=0, p=2))
    lights.append(center + l_distance * torch.nn.functional.normalize(torch.tensor([ 1,  1, -1], device=device)-center, dim=0, p=2))
    lights.append(center + l_distance * torch.nn.functional.normalize(torch.tensor([ 1, -1,  1], device=device)-center, dim=0, p=2))
    lights.append(center + l_distance * torch.nn.functional.normalize(torch.tensor([ 1, -1, -1], device=device)-center, dim=0, p=2))
    lights.append(center + l_distance * torch.nn.functional.normalize(torch.tensor([-1,  1,  1], device=device)-center, dim=0, p=2))
    lights.append(center + l_distance * torch.nn.functional.normalize(torch.tensor([-1,  1, -1], device=device)-center, dim=0, p=2))
    lights.append(center + l_distance * torch.nn.functional.normalize(torch.tensor([-1, -1,  1], device=device)-center, dim=0, p=2))
    lights.append(center + l_distance * torch.nn.functional.normalize(torch.tensor([-1, -1, -1], device=device)-center, dim=0, p=2))      
    #lights.append(camera_pos + torch.tensor([0.0, 0.05, 0.0], device=device))    

    return lights

def train(models, optimizers, texels, max_samples_per_face, lights, visualize_gradient=False, posepix2face=None, faces_xyz=None, extrinsics=None, o3d_mesh=None, faces=None, verts=None):

    models['ir'].train()
    models['geometry'].train()
    models['light'].eval()
    number_of_epochs = 50000

    print('Unpacking data...')
    rgbs, cam_xyzs, surface_xyzs, texel_indices, distinct_sample_ns = unpack_texels(texels, max_samples_per_face)
    n_texels = rgbs.size()[0]
    n_batches = 2 # 70 for 122 views, 120 samples    

    # massage everything into (n_faces * max_samples_per_face, 3) tensors
    rgbs_data = rgbs.float().flatten(start_dim=0, end_dim=1)
    cam_xyzs_data = cam_xyzs.flatten(start_dim=0, end_dim=1)    

    surface_xyzs_data = surface_xyzs.unsqueeze(1).expand(n_texels, max_samples_per_face, 3).flatten(start_dim=0, end_dim=1)    
    distinct_sample_ns_data = distinct_sample_ns.unsqueeze(1).expand(n_texels, max_samples_per_face).flatten(start_dim=0, end_dim=1)

    # now data is shape [n_texels * n_samples_per_face, 3], batch it up
    batch_size = int ((np.ceil(n_texels) / n_batches) * max_samples_per_face)
    rgbs_data = torch.split(rgbs_data, batch_size)

    cam_xyzs_data = torch.split(cam_xyzs_data, batch_size)
    surface_xyzs_data = torch.split(surface_xyzs_data, batch_size)

    distinct_sample_ns_data = torch.split(distinct_sample_ns_data, batch_size)
    
    n_batches = len(rgbs_data)
        
    for epoch in range (number_of_epochs):

        for optimizer in optimizers.values():
            optimizer.zero_grad()

        total_avg_loss = 0.0
        
        for batch_n in range(n_batches):

            # possible improvement: select batch-specific indices of brdf_params so that only relevant gradients are saved
            # however, moving to expanded loss function that's not independent for each texel would no longer allow this            

            brdf_params = models['ir'](0)[texel_indices].unsqueeze(1).expand(n_texels, max_samples_per_face, 13).flatten(start_dim=0, end_dim=1)
            light_intensity_scale = models['light'](0)
            normals = models['geometry'](0)[texel_indices].unsqueeze(1).expand(n_texels, max_samples_per_face, 3).flatten(start_dim=0, end_dim=1)
            
            brdf_params_batch = brdf_params[batch_size * batch_n : batch_size * batch_n + batch_size].to(device)
            normals_batch = normals[batch_size * batch_n : batch_size * batch_n + batch_size].to(device)
            
            rgbs_batch = rgbs_data[batch_n].to(device)
                            
            cam_xyzs_batch = cam_xyzs_data[batch_n].to(device)            
            surface_xyzs_batch = surface_xyzs_data[batch_n].to(device)        

            distinct_sample_ns_batch = distinct_sample_ns_data[batch_n].to(device)            

            # append light source above camera?  lights.append(camera_pos + torch.tensor([0.0, 0.05, 0.0], device=device))
            render_result, good_idx = render_brdf(normals_batch, cam_xyzs_batch, surface_xyzs_batch, brdf_params_batch, lights, light_intensity_scale=light_intensity_scale)
            good_idx = torch.tensor(range(render_result.size()[0]), device=device)

            if good_idx.size()[0] > 0:  # no training where batch has zero good indices

                loss = rgb_loss(render_result[good_idx], rgbs_batch[good_idx] / 255.0, distinct_sample_ns_batch[good_idx], max_samples_per_face, light_intensity_scale)
                total_avg_loss = total_avg_loss + loss.item()            
                                                    
                loss.backward(create_graph=False, retain_graph=False)

                with torch.no_grad():
                    print("normal:")
                    print(normals[0])
            
                """
                if visualize_gradient and posepix2face != None:            
                    with torch.no_grad():                
                        print("--> Visualizing gradients for batch {}".format(batch_n))
                        models['ir'].visualize_gradient(posepix2face[:10], epoch, batch_n, H, W)  

                """
                

        for optimizer in optimizers.values():
            optimizer.step()        

        with torch.no_grad():
            print('Epoch {}:  LOSS: {}'.format(epoch, total_avg_loss / n_batches))
            #print("light intensity scale: ")
            #print(light_intensity_scale)
            #print("_______________________________________________________")

        if epoch % 100 == 0:                
            with torch.no_grad():
                print("--> Saving brdf")
                brdf_params_save = models['ir'](0).detach()
                torch.save(brdf_params_save, 'rast_imgs/brdf_{}.pt'.format(epoch))
                #print("--> Saving normals")
                normals_save = models['geometry'](0).detach()
                #torch.save(normals_save, 'rast_imgs/normals_{}.pt'.format(epoch))      
                print("--> Saving light intensity scale")
                light_intensity_scale_save = models['light'](0).detach()
                torch.save(light_intensity_scale_save, 'rast_imgs/light_intensity_{}.pt'.format(epoch))                      

                if visualize_gradient and posepix2face != None:            
                    with torch.no_grad():                
                        print("--> Visualizing gradients")
                        models['ir'].visualize_gradient(posepix2face[:10], epoch, batch_n, H, W)  

                print("--> Rendering brdf on mesh")
                f_name = 'brdf_on_mesh_{}.ply'.format(epoch)
                render_brdf_on_mesh(o3d_mesh, f_name, brdf_params_save, faces, verts)
                print("--> Rendering for training views")
                render_brdf_on_training_views(brdf_params_save, extrinsics[:10, :3, 3], faces_xyz, lights, light_intensity_scale_save, normals_save, posepix2face, 1440, 1920, epoch)



def print_memory_usage():

    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a
    
    print('total memory: {}'.format(t/1000000))
    print("reserved memory: {}".format(r/1000000))
    print("allocated memory: {}".format(a/1000000))
    print("reserved free memory: {}".format(f/1000000))

    mem_report()
    print("__________________________________")
    
# https://gist.github.com/Stonesjtu/368ddf5d9eb56669269ecdf9b0d21cbe
# https://gist.github.com/Stonesjtu
def mem_report():
    '''Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported'''

    def _mem_report(tensors, mem_type):
        '''Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation '''
        print('Storage on %s' %(mem_type))
        print('-'*LEN)
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()            
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()            
            total_numel += numel
            element_size = tensor.storage().element_size()            
            mem = numel*element_size /1024/1024 # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            print('%s\t\t%s\t\t%.2f' % (
                element_type,
                size,
                mem) )
        print('-'*LEN)
        print('Total Tensors: %d \tUsed Memory Space: %.2f MBytes' % (total_numel, total_mem) )
        print('-'*LEN)

    LEN = 65
    print('='*LEN)
    objects = gc.get_objects()
    print('%s\t%s\t\t\t%s' %('Element type', 'Size', 'Used MEM(MBytes)') )
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    _mem_report(cuda_tensors, 'GPU')
    _mem_report(host_tensors, 'CPU')
    print('='*LEN)


if __name__ == '__main__':


    # load intrinsics/extrinsics
    with torch.no_grad():


        # flags for saving/loading rasterizer results    
        load_precomputed_rasterizer_data = True
        save_rast_result = False                
        render_brdf_on_train_views = False
            
        """
            dynamic_args = {
                "base_directory" : '\'./data/dragon_scale_large\'',
                "number_of_samples_outward_per_raycast" : 360,
                "number_of_samples_outward_per_raycast_for_test_renders" : 360,
                "density_neural_network_parameters" : 256,               
                "color_neural_network_parameters" : 256,
                "skip_every_n_images_for_training" : 60,
                "pretrained_models_directory" : '\'./data/dragon_scale/hyperparam_experiments/from_cloud/dragon_scale_run39/models\'',            
                "start_epoch" : 500001,
                "load_pretrained_models" : True,            
            }        

            scene = SceneModel(args=parse_args(), experiment_args='dynamic', dynamic_args=dynamic_args)    
            extrinsics, intrinsics = scene.export_camera_extrinsics_and_intrinsics_from_trained_model(save=True)
        """

        H = 1440
        W = 1920
        extrinsics = torch.from_numpy(np.load('./camera_extrinsics.npy'))
        intrinsics = torch.from_numpy(np.load('./camera_intrinsics.npy'))
        image_directory = '{}/{}'.format('./data/dragon_scale_large', 'color')
        face_normals = torch.load('./face_normals_lance.pt')
        mesh_f_name = './test/dragon_scale/3/dragon_winding_fixed_unseen_faces_removed.ply'
        mesh = load_mesh(mesh_f_name)
        o3d_mesh = o3d.io.read_triangle_mesh(mesh_f_name)
        f_name = './test/dragon_scale/3/rgb_per_face_without_unseen_faces.pt'
        initial_face_colors = torch.load(f_name).to(device) / 255.0

        light_intensity_scale = torch.tensor([1.0], device=device)
        center = torch.tensor([0.029568, -0.275911, -0.2385711]).to(device)
        lights = make_lights_cube(center, 0.5)        
        
        #n_poses = extrinsics.size()[0]
        n_poses = 2
        max_samples_per_face = 10

        #scene = None
                        
        verts = mesh.verts_list()[0].cpu()
        faces = mesh.faces_list()[0].cpu()
        face_vertices = verts[faces]
        faces_xyz = torch.mean(face_vertices, dim=1)
        n_faces = faces.size()[0]
    

        # load face normals generated from blender script
        #face_normals = torch.zeros((n_faces, 3), dtype=torch.float32)
        #face_normals_f_name = 'blender_normals_im.csv'
        #with open(face_normals_f_name, 'r') as f:
        #    for i, line in enumerate(f):
        #        normals_i = line.split(',')
        #        face_normals[i] = torch.tensor([float(normals_i[0]), float(normals_i[1]), float(normals_i[2])])
        #face_normals = torch.nn.functional.normalize(face_normals, dim=1, p=2)
        
        
        # rasterize
        if load_precomputed_rasterizer_data:
            f_name = './rast_imgs/faces_to_pixels.npy'
            faces_to_pixels = torch.from_numpy(np.load(f_name))[:, :n_poses]
            f_name = './rast_imgs/posepix2face.npy'
            posepix2face = torch.from_numpy(np.load(f_name))[:]

        else:
            (faces_to_pixels, posepix2face) = rasterize(mesh, extrinsics[:n_poses].cpu().numpy(), intrinsics[:n_poses].cpu().numpy(), H, W, save_result=save_rast_result)
            faces_to_pixels = faces_to_pixels.cpu()

        
        if render_brdf_on_train_views:
            # render brdf for train images
            epoch = 300
            print('loading brdf: ')            
            brdf_params = torch.load('rast_imgs/brdf_{}.pt'.format(epoch)).to(device)            
            face_normals = torch.load('rast_imgs/normals_{}.pt'.format(epoch)).to(device)
            light_intensity_scale = torch.load('rast_imgs/light_intensity_{}.pt'.format(epoch)).to(device)
            render_brdf_on_mesh(o3d_mesh, brdf_params, faces, verts)            
            render_brdf_on_training_views(brdf_params, extrinsics[:, :3, 3], faces_xyz, lights, light_intensity_scale, face_normals, posepix2face, H, W, epoch)        
            quit()


        # prepare training data
        texels = {}

        # TODO: either do this in a faster way or implement precomputing
        faces_to_pixels = faces_to_pixels.cpu()
        rgb_from_photos = load_image_data(image_directory=image_directory, n_poses=n_poses, H=H, W=W)

        for pose_i in range(n_poses):

            print("Packing up data for view {}... ".format(pose_i))
            pose_xyz = extrinsics[pose_i, :3, 3]

            # get the rgb samples for pose_i for each face reported to be hit in pose_i                    
            hit_face_indices_for_pose = torch.where(faces_to_pixels[:, pose_i, 0] != -1)[0]
                                  
            rgb_samples_for_pose = rgb_from_photos[
                pose_i,
                faces_to_pixels[hit_face_indices_for_pose, pose_i, 0],
                faces_to_pixels[hit_face_indices_for_pose, pose_i, 1]
            ]
            
            # update rgb samples, camera pose samples, and texel xyz (if not initialized already) for texels seen by this pose
            for i in range(hit_face_indices_for_pose.size()[0]):
                texel_i = hit_face_indices_for_pose[i]
                
                if str(texel_i.item()) not in texels.keys():                
                    texels[str(texel_i.item())] = TexelSamples(texel_i.item(), [rgb_samples_for_pose[i]], [pose_xyz], faces_xyz[texel_i])
                else:
                    texels[str(texel_i.item())].add_sample(rgb_samples_for_pose[i], pose_xyz)                                    

        n_faces_hit = 0
        hits = 0
        max_faces_hit = -float('inf')
        for i, (k,v) in enumerate(texels.items()):
            n_hits_for_face = len(v.rgb_samples)
            if n_hits_for_face > 1:
                hits = hits + len(v.rgb_samples)
                if n_hits_for_face > max_faces_hit:
                    max_faces_hit = n_hits_for_face
                                
            n_faces_hit = n_faces_hit + 1
            
        print("Number of faces hit: {}".format(n_faces_hit))
        print("Max number of faces hit: {}".format(max_faces_hit))
        print("Avg hits per face: {}".format( (hits / n_faces_hit)))
        print("Percentage of faces with >= 1 hits: {}".format((n_faces_hit / n_faces)))


    # initialize models
    models = {}
    optimizers = {}
    models['ir'] = IRModel(n_faces, initial_face_colors)
    optimizers['ir'] = torch.optim.Adam(models['ir'].parameters(), lr=0.001)
    models['geometry'] = GeometryModel(face_normals)
    optimizers['geometry'] = torch.optim.Adam(models['geometry'].parameters(), lr=0.001)
    models['light'] = LightModel(light_intensity_scale)
    optimizers['light'] = torch.optim.Adam(models['light'].parameters(), lr=0.0001)    
    
    # fit brdf parameters
    brdf_params = train(models, optimizers, texels, max_samples_per_face, lights, visualize_gradient=True, posepix2face=posepix2face, faces_xyz=faces_xyz, extrinsics = extrinsics, o3d_mesh=o3d_mesh, faces=faces, verts=verts)




    