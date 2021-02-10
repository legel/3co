#ifndef Agent_h
#define Agent_h


#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <octomap/Pointcloud.h>


class Agent {

  public:
    virtual void learn(char* obs) = 0;
    virtual float* act() = 0;

};


class NullAgent : Agent {

  public:
    void learn(char* obs);
    float* act();

};



/*
/////////////////////////////

void NullAgent::learn(char *obs) {
  printf("I'm incapable of learning :)\n");
}

float* NullAgent::act() {
  return NULL;
}

/////////////////////////////
*/








#endif
