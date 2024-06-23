from robotics import Robot
import time
robot = Robot()
#robot.move(x=0.05, z=1.0)

#robot.calibrate_axis("yaw")
# robot.calibrate_axis("y")
# robot.calibrate_axis("z")
robot.move(x=0.11, y=-0.225, z=0.79, pitch=-20.5, yaw=161.0)
robot.scan(project_name="ikea_chair", scan_index=0, distance=280, exposure_time=50.0, export_to_csv=False) # exposure_time=75.0, dual_exposure_mode=True, dual_exposure_multiplier=4.0, filter_outliers=True, outliers_std_dev=2.0


# #robot.move(x=0.01, y=0.1, z=0.12, pitch=-86.35, yaw=-175)
# base_exposure_time = 3.0
# number_of_exposures = 33
# exposure_times = [multiplier * base_exposure_time for multiplier in range(1,number_of_exposures+1)]
# for scan_index, exposure_time in enumerate(exposure_times):
#time.sleep(30.0)
#robot.scan(project_name="ikea_chair", scan_index=0, distance=300, exposure_time=50.0, export_to_csv=False) # exposure_time=75.0, dual_exposure_mode=True, dual_exposure_multiplier=4.0, filter_outliers=True, outliers_std_dev=2.0
# # 0.5, 1.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 400.0

# # 30.0, 40.0, 50.0, 60.0
# #hdr_exposure_times=[2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 35.0, 50.0, 75.0, 100.0, 150.0, 200.0]

# accessNodeColor
# accessNodeRadius
# nodeColumns
# nodeScales
# 
# Sensible default radii
# handleAccessNodeRadius:
#  -> minimum-maximum radius parameters as a function
#  -> default color: 
#  -> cluster: 
#  -> ...
#  -> ...

# auto-generate: R,G,B
# 

# handleAccessNodeColor
# ExplorerUI -> Lib. Get us on same one. 
# Haven't figured out hot reloading.

# 