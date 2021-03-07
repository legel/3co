import open3d as o3d
import evaluation
import socket


# Code from Rob to be refactored into system when ready

# in online mode, actions are supplied by agent client, but start position
# must be specified below
# data goes to iris_workspace/
if sim_mode == "online":
  x,y,z,theta,phi = 4.8, 0.0, 1.7, 90.0, 90.0
  start_pos = [x,y,z,theta,phi]
  scene.update(iris,start_pos)
  startOnlineSimulation(iris, scene, start_pos)

def startOfflineSimulation(iris, environment, path, dataset):
  print("---------------------------------")
  print("Begin offline scan simulation")
  print("---------------------------------\n")
  i = 0
  scaling = 1.0
  for p in path:
    x = p[0] / scaling
    y = p[1] / scaling
    z = p[2] / scaling
    yaw = p[3]
    pitch = p[4]
    f_out_name="{}/{}_{}".format(dataset,dataset,i)
    print("Moving to new state: [({}, {}, {}), ({}, {})]".format(round(x,2),round(y,2),round(z,2),round(yaw,2),round(pitch,2)))
    action = [x, y, z, yaw, pitch]
    environment.update(iris, action) 
    #scanner.move(x=x, y=y, z=z, pitch=pitch, yaw=yaw)
    print("Scanning")
    iris.scanner.scan(f_out_name=f_out_name, render_png=True)
    csv2ply.csv2ply("simulated_scanner_outputs/{}.csv".format(f_out_name), "simulated_scanner_outputs/{}.ply".format(f_out_name))
    i = i + 1

    # execute action on environment
    environment.update(iris, action)

    t = t + 1

  msg = "end_session"
  client_sock.send(msg.encode("utf-8"))
  client_sock.close()
  sock.close()
  print("------ Simulation concluded ------")

def parseAction(action):
  a = action.split(",")
  return [float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4])]

def startOnlineSimulation(iris, environment, start_pos):

  print("---------------------------------")
  print("Begin online scan simulation")
  print("---------------------------------\n")

  host = socket.gethostname()
  port = 8080

  
  sock = socket.socket()
  sock.bind((host, port))


  sock.listen(1)
  #print("Starting agent controller...")
  #os.system(controller_start_cmd)
  print("Waiting for agent controller to connect...")
  client_sock, address = sock.accept()
  print("Agent controller connected.\n")
  print("Sending start position to agent.")

  msg = "start_pos : {},{},{},{},{}".format(start_pos[0],start_pos[1],start_pos[2],start_pos[3],start_pos[4])
  client_sock.send(msg.encode("utf-8"))

  # wait for ack from agent
  data = client_sock.recv(1024).decode("utf-8")

  if data.strip() == "ack":
    print("Agent ack received.")
  else:
    print("Agent ack failed! Aborting simulation")
    client_sock.close()
    sock.close()
    quit()

  print("------ Begin simulation ------")
  

  t = 0

  status = "active"
  while status == "active":

    obs = iris.scan(t) # filename of obs

    # send obs to agent
    data = obs
    msg = "obs : {}".format(data)
    print("sending obs to agent")
    client_sock.send(msg.encode("utf-8"))

    # receive next action from agent
    data = client_sock.recv(1024).decode("utf-8")
    if not data:
      print("null data")
      client_sock.close()
      sock.close()
      break
    action_data = data.split(":")
    if action_data[0].strip() != "act":
      print("Error: received invalid action")
      client_sock.close()
      sock.close()
      quit()
    print("received action from agent")
    print(action_data)
    action = parseAction(action_data[1].strip())


"""
the following might be useful later, but needs to be updated

# Experiment runs main control loop given a constructed iris agent and environment
def experiment(iris, environment):

  print("---------------------------------")
  print("Begin scan simulation")
  print("---------------------------------\n")

  results = []
  results.append(open("experiment_results/avg_scan_error.txt","w"))
  results.append(open("experiment_results/avg_recon_error.txt","w"))
  results.append(open("experiment_results/avg_total_error.txt","w"))
  results.append(open("experiment_results/haus_scan_error.txt","w"))
  results.append(open("experiment_results/haus_recon_error.txt","w"))
  results.append(open("experiment_results/haus_total_error.txt","w"))


  # get the point cloud from ground truth mesh
  ground_truth_mesh_fn = "simulated_scanner_outputs/chalice_0.1/chalice_centered.ply"
  ground_truth_pc = o3d.io.read_point_cloud(ground_truth_mesh_fn)

  t = 0

  while iris.active():
    obs = iris.scan()
    iris.learn(obs)
    action = iris.act()
    environment.update(iris, action)

    
    if t > 0:
      # get the merged point cloud from all scans so far
      scan_pc_fn = "iris_workspace/scan_0.1_merged.ply"
      scan_pc = o3d.io.read_point_cloud(scan_pc_fn)

      # get the point cloud from the current reconstruction
      current_mesh_fn = "iris_workspace/scan_0.1_reconstructed_vcg.ply"
      current_pc = o3d.io.read_point_cloud(current_mesh_fn)

      scan_d_haus = evaluation.hausdorffDistance(scan_pc, ground_truth_pc)
      scan_d_max_avg = evaluation.maxAvgPointCloudDistance(scan_pc, ground_truth_pc)
      recon_d_haus = evaluation.hausdorffDistance(scan_pc, current_pc)
      recon_d_max_avg = evaluation.maxAvgPointCloudDistance(scan_pc, current_pc)
      final_d_haus = evaluation.hausdorffDistance(current_pc, ground_truth_pc)
      final_d_max_avg = evaluation.maxAvgPointCloudDistance(current_pc, ground_truth_pc)

      results[0].write("{}\n".format(scan_d_max_avg))
      results[1].write("{}\n".format(recon_d_max_avg))
      results[2].write("{}\n".format(final_d_max_avg))
      results[3].write("{}\n".format(scan_d_haus))
      results[4].write("{}\n".format(recon_d_haus))
      results[5].write("{}\n".format(final_d_haus))

    t = t + 1


  for f in results:
    f.close()



  print("---------------------------------")
  print("End scan simulation")
  print("---------------------------------\n")

"""


