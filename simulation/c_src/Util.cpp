#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <octomap/Pointcloud.h>
#include <iostream>
#include <fstream>
#include <string>

#include "Util.h"

using namespace std;
using namespace octomap;



  
  
Pointcloud Util::readFromPly(string fname) {

  Pointcloud pc;
  ifstream fin(fname);
  int h = 0;
  int v = 0;
  float x = 0.0;
  float y = 0.0;
  float z = 0.0;
  int r = 0;
  int g = 0;
  int b = 0;
  string line;
  int i=0;
  while (getline (fin, line)){
    //if (i>=10) {
    sscanf(line.c_str(), "%d,%d,%f,%f,%f,%d,%d,%d\n", &h, &v, &x,&y,&z,&r,&g,&b);
    pc.push_back(x,y,z);
    //}
    i++;
  }

  return pc;
}



