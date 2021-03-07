import sys
from os import listdir, path, getcwd
cwd = getcwd()
sys.path.append(cwd)
import open3d as o3d
import os
import reconstruction
import csv2ply

class IrisAgent():

  def __init__(self, scanner, start_position):
    self.constructPrior()
    self.scanner = scanner
    self.resolution = scanner.resolution
    self.position = start_position
    self.t = 0
    self.workspace = "iris_workspace"
    command = "rm -r {}".format(self.workspace)
    os.system(command)
    command = "mkdir {}".format(self.workspace)
    os.system(command)

  def constructPrior(self):
    self.psi = []

  def scan (self):
    #self.scanner.render("simulated_scanner_outputs/control_test/scan_{}.png".format(self.t))
    return self.scanner.scan()

  def learn(self, obs):

    csv2ply.csv2ply("{}".format(obs), "{}/scan_{}.ply".format(self.workspace, self.t))

    command = "mv {} {}/scan_{}.csv".format(obs, self.workspace, self.t)
    os.system(command)

    dataset = "scan"
    fdir = "iris_workspace"
    fname = "{}/{}".format(fdir,dataset)
    n_files = self.t+1
    max_edge_len = 0.05
    voxel_size = 0.01
    use_im_remesh = True

    mesh = reconstruction.doReconstruction(fname, fdir, dataset, n_files, self.resolution, max_edge_len, voxel_size, use_im_remesh)
    outfname = "mesh_{}.ply".format(self.t)
    mesh.writeAsPLY(outfname)



  # return action of the form "go to [x,y,z,yaw,pitch]"
  def act(self):
    self.t = self.t + 1
    self.position[0] = self.position[0] - 0.1
    return self.position

