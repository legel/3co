import commander
from robotics import Iris
import time
 	# (-180, +180) 
#commander.move({"cali_x": 2.19})	#1.225		# (0.0, 2.0) # 0.89 cali_x inverted corresponds to 0.975 iris x
#commander.move({"cali_y": 2.45})	#0.96	# 1.05 # (0.0, 2.0)
#commander.move({"cali_turn": 90})
#commander.move({"cali_z": 0.7})		# (0.0, 0.7)

# #commander.move({"x": 0.975})
# #commander.move({"y": -0.09})
#commander.move({"z": 1.05}) # 0.56


#robot = Robot()
#robot.scan(project_name="rust_crystal_4", distance=525.0, hdr_exposure_times=[15.0, 30.0, 60.0], export_to_npy=True)



# commander.move({"pitch": -90.0})
# commander.move({"yaw": -90.0})

# commander.move({'x': 1.35})
# commander.move({'y': 0.2})
# commander.move({'z': 1.2})

# all_commands = []
# command = {"x": 1.40, "y": 0.1, "z": 1.0, "pitch": -90.0, "yaw": -90.0}
# commander.move(command)
# robot = Robot()
# robot.scan(project_name="forest_crystal_1", distance=500.0, hdr_exposure_times=[35.0, 75.0, 150.0, 350.0], export_to_npy=True)

# command = {"x": 1.45, "y": 0.1, "z": 1.0, "pitch": -90.0, "yaw": -90.0}
# commander.move(command)
# robot = Robot()
# robot.scan(project_name="forest_crystal_2", distance=500.0, hdr_exposure_times=[35.0, 75.0, 150.0, 350.0], export_to_npy=True)

# command = {"x": 1.50, "y": 0.1, "z": 1.0, "pitch": -90.0, "yaw": -90.0}
# commander.move(command)
# robot = Robot()
# robot.scan(project_name="forest_crystal_3", distance=500.0, hdr_exposure_times=[35.0, 75.0, 150.0, 350.0], export_to_npy=True)

# command = {"x": 1.55, "y": 0.1, "z": 1.0, "pitch": -90.0, "yaw": -90.0}
# commander.move(command)
# robot = Robot()
# robot.scan(project_name="forest_crystal_4", distance=500.0, hdr_exposure_times=[35.0, 75.0, 150.0, 350.0], export_to_npy=True)

# command = {"x": 1.60, "y": 0.1, "z": 1.0, "pitch": -90.0, "yaw": -90.0}
# commander.move(command)
# robot = Robot()
# robot.scan(project_name="forest_crystal_5", distance=500.0, hdr_exposure_times=[35.0, 75.0, 150.0, 350.0], export_to_npy=True)

# command = {"x": 1.60, "y": 0.25, "z": 1.0, "pitch": -75.0, "yaw": -90.0}
# commander.move(command)
# robot = Robot()
# robot.scan(project_name="forest_crystal_6", distance=525.0, hdr_exposure_times=[35.0, 75.0, 150.0, 350.0], export_to_npy=True)

# command = {"x": 1.60, "y": 0.35, "z": 1.0, "pitch": -60.0, "yaw": -90.0}
# commander.move(command)
# robot = Robot()
# robot.scan(project_name="forest_crystal_6", distance=525.0, hdr_exposure_times=[35.0, 75.0, 150.0, 350.0], export_to_npy=True)

# command = {"x": 1.60, "y": 0.45, "z": 0.85, "pitch": -45.0, "yaw": -90.0}
# commander.move(command)
# robot = Robot()
# robot.scan(project_name="forest_crystal_7", distance=525.0, hdr_exposure_times=[35.0, 75.0, 150.0, 350.0], export_to_npy=True)

# command = {"x": 1.60, "y": 0.55, "z": 0.60, "pitch": -30.0, "yaw": -90.0}
# commander.move(command)
# robot = Robot()
# robot.scan(project_name="forest_crystal_8", distance=525.0, hdr_exposure_times=[35.0, 75.0, 150.0, 350.0], export_to_npy=True)

# command = {"x": 1.60, "y": 0.65, "z": 0.55, "pitch": -15.0, "yaw": -90.0}
# commander.move(command)
# robot = Robot()
# robot.scan(project_name="forest_crystal_9", distance=525.0, hdr_exposure_times=[35.0, 75.0, 150.0, 350.0], export_to_npy=True)

# command = {"x": 1.60, "y": 0.65, "z": 0.45, "pitch": 0.0, "yaw": -90.0}
# commander.move(command)
# robot = Robot()
# robot.scan(project_name="forest_crystal_10", distance=525.0, hdr_exposure_times=[35.0, 75.0, 150.0, 350.0], export_to_npy=True)

# command = {"x": 1.65, "y": 0.625, "z": 0.425, "pitch": -5.0, "yaw": -75.0}
# commander.move(command)
# robot = Robot()
# robot.scan(project_name="forest_crystal_11", distance=425.0, hdr_exposure_times=[35.0, 75.0, 150.0, 350.0], export_to_npy=True)

# command = {"x": 1.65, "y": 0.645, "z": 0.275, "pitch": 5.0, "yaw": -75.0}
# commander.move(command)
# robot = Robot()
# robot.scan(project_name="forest_crystal_12", distance=450.0, hdr_exposure_times=[35.0, 75.0, 150.0, 350.0], export_to_npy=True)

# command = {"x": 1.65, "y": 0.645, "z": 0.295, "pitch": 0.0, "yaw": -75.0}
# commander.move(command)
# robot = Robot()
# robot.scan(project_name="forest_crystal_13", distance=450.0, hdr_exposure_times=[35.0, 75.0, 150.0, 350.0], export_to_npy=True)

# robot = Iris()
# while True:
# 	commander.move({"x": 1.58, "y": -0.04, "z": 1.20, "pitch": 0.0, "yaw": -33.33})
# 	commander.move({"x": 1.58, "y": -0.04, "z": 0.820, "pitch": 0.0, "yaw": 33.33})
# 	#robot.focus_sweep()
# 	#commander.move({"x": 1.58, "y": -0.04, "z": 0.820, "pitch": 0.0, "yaw": 30.0})
# 	commander.move({"x": 1.58, "y": -0.04, "z": 0.820, "pitch": 0.0, "yaw": 0.0})
# 	commander.move({"x": 1.58, "y": -0.04, "z": 0.820, "pitch": 33.33, "yaw": 0.0})
# 	#commander.move({"x": 1.58, "y": -0.04, "z": 0.820, "pitch": -15.0, "yaw": 0.0})
# 	commander.move({"x": 1.58, "y": -0.04, "z": 0.820, "pitch": 12.5, "yaw": 0.0})
# 	robot.scan(project_name="studio", auto_focus=False, distance=600, hdr_exposure_times=[200.0])

	#robot.focus_optics(self, distance=600)
	#commander.calibrate("camera_focus")
	#commander.calibrate("projector_focus")
	#commander.calibrate("camera_aperture")



#robot = Iris()
#robot.scan(project_name="sample_fuck", red_led_current=1.0, green_led_current=1.0, blue_led_current=1.0, auto_focus=False, export_to_npy=False, process_point_cloud=False) # distance=450.0, hdr_exposure_times=[35.0, 75.0, 125.0], 
#robot.scan(project_name="green_lower_2", red_led_current=1.0, green_led_current=0.5, blue_led_current=1.0, auto_focus=False)
#robot.scan(project_name="blue_lower_2", red_led_current=1.0, green_led_current=1.0, blue_led_current=0.5, auto_focus=False)
# finder = PlaneFinder(project_name="prarie_oak_x", path_to_point_cloud_file="prarie_oak_x-0.ply")



#commander.move({"camera_polarization": 55})
#commander.move({"yaw": -90})
#commander.move({"z": 0.40})
#commander.move({"pitch": -70.0})
#commander.move({"yaw": -125.0})

# ## CALIBRATION
#commander.move({"z": 0.5})
#commander.calibrate('yaw')
# commander.calibrate('yaw') # double calibration can be necessary

#commander.calibrate('pitch')
#commander.move({"pitch": 12.0})
#commander.recoordinate("pitch")

# commander.calibrate('x')
# commander.move({'x': 0.1})
#commander.move({"yaw": -90})
#commander.calibrate('y')
#commander.move({"yaw": -90})
#commander.move({"y": 0.6})
#commander.move({"z": 0.45})
#commander.move({"pitch": -60.0})

# commander.move({"yaw": 0})
#commander.calibrate('z')
# commander.move({"y": 1.0})
# commander.move({"pitch": -90.0})
commander.move({"z": 0.05})

#commander.move({"pitch": -90.0})

# command = {"x": 1.35, "y": 0.1, "z": 1.0, "pitch": -90.0, "yaw": -90.0}
# commander.move(command)

#robot = Robot()

# yaws = [85, 90, 95]
# for i, yaw in enumerate(yaws):
# 	project_name = "ikea_chair_2_{}".format(i)
# 	commander.move({"yaw": yaw})
# 	time.sleep(10.0)
#time.sleep(10.0)

#robot.scan(project_name="board", distance=555.0, exposure_time=55.0)

# commander.move({"pitch": -90})
# commander.move({"pitch": -75})
# commander.move({"pitch": -60})
# commander.move({"pitch": -45})
# commander.move({"pitch": -30})

# FITTING CLOSEST CHARUCO POINTS TO PLANE
#	- Objective function uses average_neighboring_point_distance() to measure in pixel coordinates the distance from the selected 3D point to the ideal 3D point 
#	- Objective function uses distance matrix of all ChaRuCo points for each plane, and does not confused planes
#	- Objective function uses planar equation, forcing the final ChArUco point for all points to exist on the plane
#	- Visualizing both the plane fit and the interpolatated 3D ChAruCo point fit, and estimating final deviation in the objective function, in mm, will demonstrate value of approach
#	- Make a module that you can feed all of this into, and cleanly get results running the background
# WORKING WITH ALL CHARUCO POINTS DETECTED IN 2D
# FITTING 6D JOINT POSE GLOBAL OPTIMIZATION ALGORITHM ON TOP OF COHERENT POINT DRIFT
# 6D(PITCH) = ?
# 6D(YAW) = ?
# 6D(PITCH,YAW) = ?
# 6D(X) = ?
# 6D(Y) = ?
# 6D(Z) = ?
# 6D(X,Y,Z,PITCH,YAW) = ?



# cali_turn = 180, iris_yaw = 90, cali_x = 1.11, cali_y = 1.05
# cali_turn = 135, iris_yaw = 45, cali_x = 1.21 (+10.0cm), cali_y = 0.95 (-10cm)
# cali_turn = 90, iris_yaw = 0, cali_x = 1.31 (+10.0cm), cali_y = 1.025 (-10cm)
# 