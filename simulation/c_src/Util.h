#ifndef Util_h
#define Util_h

#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <octomap/Pointcloud.h>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;
using namespace octomap;

class Util {
  public:
    static Pointcloud readFromPly(string fname);
};


#endif
