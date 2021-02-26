#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <string>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include "Util.h"
#include "Agent.h"
#include "CircleAgent.h"
using namespace std;
using namespace octomap;


int main(int argc, char** argv) {

  CircleAgent agent;

  if (argc != 3) {
    printf("Usage: ./iris_agent <resolution> <port>\n");
    exit(1);
  }

  float resolution = stof(argv[1]);
  int port = atoi(argv[2]);

  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
    printf("ERROR: could not open socket\n");
    exit(1);
  }

  struct hostent *server = gethostbyname("localhost");
  if (server == NULL) {
    printf("ERROR: no such host\n");
    exit(1);
  }

  struct sockaddr_in serv_addr;
  memset(&serv_addr, 0, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  memcpy((char *)server->h_addr, (char *)&serv_addr.sin_addr.s_addr, server->h_length);
  serv_addr.sin_port = htons(port);


  if (connect(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
    printf("ERROR: could not connect\n");
    exit(1);
  }

  char buffer[1024];
  int n;

  n = read(sockfd, buffer, 1024);

  float s_pose[5];
  float x,y,z,yaw,pitch;
  sscanf(buffer, "start_pos : %f,%f,%f,%f,%f", &s_pose[0], &s_pose[1],&s_pose[2],&s_pose[3],&s_pose[4]);
  agent.setStartPose(s_pose);

  memset(buffer, 0, 1024);
  sprintf(buffer, "ack");
  n = write(sockfd, buffer, strlen(buffer));


  int i = 0;
  while (strcmp(buffer, "end_session") != 0) {
    memset(buffer, '\0', 1024);
    n = read(sockfd, buffer, 1024);
    if (strcmp(buffer, "end_session") == 0) {
      printf("controller ending session\n");
      break;
    }


    // observe
    char code[1024];
    char obs[1024];
    memset(code, '\0', 1024);
    memset(obs, '\0', 1024);
    sscanf(buffer, "%s : %s.csv", code, obs);
    if (strcmp(code, "obs") != 0) {
      printf("%s\n", code);
      printf("%s\n", obs);
      printf("ERROR: received invalid obs\n");
      break;
    }

    printf("agent received obs\n");

    // learn
    printf("agent learning\n");
    agent.learn(obs);


    // act
    printf("agent acting\n");
    float* action = agent.act();
    memset(buffer, '\0', 1024);
    sprintf(buffer, "act : %f,%f,%f,%f,%f", action[0], action[1], action[2], action[3], action[4]);
    n = write(sockfd, buffer, strlen(buffer));
    delete [] action;
    i++;

  }

  close(sockfd);





  /*
  printf("initializing pc\n");
  Pointcloud pc = Util::readFromPly("../simulated_scanner_outputs/chalice_0.1/chalice_0.1_0.ply");

  printf("creating tree\n");
  OcTree tree (0.01);

  tree.insertPointCloud(&pc, point3d(4.8, 0.0, 1.7));

  tree.writeBinary("test_tree.bt");
*/

  return 0;


}

