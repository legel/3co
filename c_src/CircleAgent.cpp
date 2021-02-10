#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <octomap/Pointcloud.h>
#include "CircleAgent.h"
#include "Util.h"
#include <math.h>

using namespace std;
using namespace octomap;

CircleAgent::CircleAgent() {

  t = 0;

  memset(pose, 0.0, sizeof(float)*5);

  tree = new OcTree(0.01);


}

void CircleAgent::setStartPose(float *start_pose) {

  memcpy(pose, start_pose, sizeof(float)*5);

}

void CircleAgent::learn(char *obs) {


  char fname[128];
  memset(fname, '\0', 128);
  sprintf(fname, "%s", obs);

  Pointcloud pc = Util::readFromPly(fname);

  tree->insertPointCloud(&pc, point3d(pose[0],pose[1],pose[2]));

  printf("pose: %f, %f, %f\n", pose[0], pose[1], pose[2]);
  sprintf(fname, "c_src/logs/circleagent_tree_%d.bt", t);
  string s(fname);
  tree->writeBinary(s);

}

float* CircleAgent::act() {

  float* action = rotate(pose[0], pose[1], pose[2], pose[3], pose[4], 30.0);
  memcpy(pose, action, sizeof(float)*5);
  return action;
}


float* CircleAgent::rotate(float x, float y, float z, float yaw, float pitch, float theta) {

  float a = (M_PI * theta) / 180.0;
  float xp = x*cos(a) - y*sin(a);
  float yp = x*sin(a) + y*cos(a);
  float zp = z;
  float yawp = yaw - theta;
  if (yawp >= 360.0)
    yawp = yawp - 360.0;
  else if (yawp < 0)
    yawp = yawp + 360.0;

  float* out = new float[5];
  out[0] = xp;
  out[1] = yp;
  out[2] = zp;
  out[3] = yawp;
  out[4] = pitch;
  return out;


}

