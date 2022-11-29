import torch
import numpy as np
import mcubes
import open3d as o3d

#filename = 'data/dragon_scale/hyperparam_experiments/pretrained_with_entropy_loss_200k/sdf/sdf_120meshes_vox001.npy'
filename = 'sdf_120meshes_vox001.npy'

sdf = torch.from_numpy(np.load(filename))

x = sdf[:,0].long()
y = sdf[:,1].long()
z = sdf[:,2].long()
d = sdf[:,3].float()

max_x_i = torch.max(x,dim=0)[0]
max_y_i = torch.max(y,dim=0)[0]
max_z_i = torch.max(z,dim=0)[0]

u = torch.zeros(max_x_i+1, max_y_i+1, max_z_i+1)


u[ sdf[:,0].long(), sdf[:,1].long(), sdf[:,2].long() ] = sdf[:,3].float()
print(u.size())

u = u.numpy()

# do we watch marching cube's voxel representation?
vertices, triangles = mcubes.marching_cubes(u,0.005)
print("vertices, triangles:", vertices.shape, triangles.shape)

mcubes.export_mesh(vertices, triangles, "mesh.dae", "NiceMesh")