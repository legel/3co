import sys
import os

def csv2ply(f_in, f_out):

  # write header
  n_vertices = sum(1 for line in open(f_in)) - 1
  ply_file = open(f_out, "w")
  ply_file.write('ply\n')
  ply_file.write('format ascii 1.0\n')
  ply_file.write('element vertex ' + str(n_vertices) + '\n')
  ply_file.write('property float x\n')
  ply_file.write('property float y\n')
  ply_file.write('property float z\n')
  ply_file.write('property uchar red\n')
  ply_file.write('property uchar green\n')
  ply_file.write('property uchar blue\n')
  ply_file.write('end_header\n')

  # read vertices and write each one
  with open(f_in, "r") as csv_file:
    # skip header
    #line = csv_file.readline()
    # read vertices
    for line in csv_file:
      l = line.split(",")
      x = l[2]
      y = l[3]
      z = l[4]
      r = l[5]
      g = l[6]
      b = l[7]
      vertex = '{} {} {} {} {} {}'.format(x,y,z,r,g,b)
      ply_file.write(vertex)

  ply_file.close()
'''
def main():

    f_list = os.popen("ls simulated_scanner_outputs/chair/*.csv").read().split("\n")
    for f_in in f_list:
        if f_in != "":
            f_out = f_in.split(".csv")[0] + "_fixed.ply"
            csv2ply(f_in, f_out)


if __name__ == "__main__":
    main()
    '''
