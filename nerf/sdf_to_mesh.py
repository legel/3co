import torch
import numpy as np
import mcubes
import open3d as o3d
import sys

def sdf2mesh(in_file_name, out_file_name):
    print('Meshing', in_file_name)
    sdf_in = torch.from_numpy(np.load(in_file_name))

    sdf_indices = torch.argwhere(sdf_in == sdf_in)
    sdf = np.transpose(np.asarray([
        sdf_indices[:,0].numpy(), 
        sdf_indices[:,1].numpy(), 
        sdf_indices[:,2].numpy(), 
        sdf_in[sdf_indices[:,0].numpy(), sdf_indices[:,1].numpy(), sdf_indices[:,2].numpy()].numpy()
    ]))                      

    sdf = torch.tensor(sdf)

    #sdf = torch.from_numpy(np.load(filename))

    x = sdf[:,0].long()
    y = sdf[:,1].long()
    z = sdf[:,2].long()
    d = sdf[:,3].float()

    max_x_i = torch.max(x,dim=0)[0]
    max_y_i = torch.max(y,dim=0)[0]
    max_z_i = torch.max(z,dim=0)[0]

    u = torch.zeros(max_x_i+1, max_y_i+1, max_z_i+1)

    u[ sdf[:,0].long(), sdf[:,1].long(), sdf[:,2].long() ] = sdf[:,3].float()    

    u = u.numpy()

    # do we match marching cube's voxel representation?
    vertices, triangles = mcubes.marching_cubes(u,0.01)
    print("vertices, triangles:", vertices.shape, triangles.shape)

    print('Outputting', out_file_name)
    mcubes.export_mesh(vertices, triangles, out_file_name, "NiceMesh")

    

def merge_sdfs(in_file_names, out_file_name):

    sdf_merged = None
    for sdf_i in [torch.from_numpy(np.load(f_name)) for f_name in in_file_names]:
        #sdf_i = torch.tensor(sdf_i.clone().detach())
        if sdf_merged == None:
            sdf_merged = sdf_i
        else:
            sdf_merged = torch.min( torch.cat ( [sdf_merged.unsqueeze(1), sdf_i.unsqueeze(1)], dim=1 ), dim=1 )[0]            
    
    with open(out_file_name, "wb") as f:
        np.save(f, sdf_merged.numpy())    
    

if __name__ == '__main__':

    with torch.no_grad():
        args = sys.argv

        if args[1] == 'sdf2mesh':
            #in_file_name = 'sdfs/sdf_merged.npy'
            #out_file_name = 'mesh_merged.dae'
            in_file_name = 'sdfs/sdf_merged.npy'#'sdfs/vox001/sdf.npy'
            out_file_name = 'vox01_mesh.dae'            
            sdf2mesh(in_file_name, out_file_name)
        elif args[1] == 'merge':
            n_sdfs = 30
            in_file_names = ['sdfs/current_run/sdf_vox0.01_{}.npy'.format(i) for i in range(n_sdfs)]
            out_file_name = 'sdfs/sdf_merged.npy'
            merge_sdfs(in_file_names, out_file_name)
        else:
            print("Usage: sdf2mesh or merge")