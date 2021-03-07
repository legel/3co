#ifndef circleagent_h
#define circleagent_h

#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <octomap/Pointcloud.h>
#include "Agent.h"

using namespace std;
using namespace octomap;

// just used for testing basic octomap functionality
class CircleAgent : Agent {

  public:
    float path[12][5];
    int t;
    float pose[5];
    OcTree* tree;


    CircleAgent();
    void setStartPose(float *pose);
    void learn(char* obs);
    float* act();
    float* rotate(float x, float y, float z, float yaw, float pitch, float theta);

};






#endif
