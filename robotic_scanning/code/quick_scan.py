from robotics import Robot
robot = Robot()
#robot.calibrate()
#robot.move(x=0.05, y=0.0, z=0.15, pitch=-86.35, yaw=0.0)

robot.scan(project_name="x", scan_index=0, distance=555, exposure_time=75.0) # exposure_time=75.0, dual_exposure_mode=True, dual_exposure_multiplier=4.0, filter_outliers=True, outliers_std_dev=2.0

#robot.scan(project_name="test", scan_index=8, distance=440, hdr_exposure_times=[100.0, 200.0]) # exposure_time=75.0, dual_exposure_mode=True, dual_exposure_multiplier=4.0, filter_outliers=True, outliers_std_dev=2.0


# import commander
# import time

# commander.move({'x': 1.14, 'y': -0.225, 'z': 0.95, 'yaw': 0.0, 'pitch': -83.65})
# robot.scan(project_name="cry-1", distance=425, auto_hdr=True, hdr_exposure_times=[25.0, 75.0, 150.0, 275.0, 400.0]) # exposure_time=75.0, dual_exposure_mode=True, dual_exposure_multiplier=4.0, filter_outliers=True, outliers_std_dev=2.0

#commander.move({'x': 1.14, 'y': -0.07, 'z': 0.95, 'yaw': 0.0, 'pitch': -83.65})
#robot.scan(project_name="flower", distance=425, hdr_exposure_times=[400.0, 275.0, 150.0, 75.0, 25.0]) # exposure_time=75.0, dual_exposure_mode=True, dual_exposure_multiplier=4.0, filter_outliers=True, outliers_std_dev=2.0


#commander.move({'x': 1.20, 'y': -0.075, 'z': 0.50, 'yaw': 12.5, 'pitch': -55.0}) # -85.65 pitch correction..
#robot.scan(project_name="gerbara-1", distance=300, hdr_exposure_times=[5.0, 10.0, 25.0, 50.0, 100.0, 140.0, 180.0, 250.0, 300.0, 350.0]) # exposure_time=75.0, dual_exposure_mode=True, dual_exposure_multiplier=4.0, filter_outliers=True, outliers_std_dev=2.0

#commander.move({'x': 1.20, 'y': 0.0, 'z': 0.50, 'yaw': 0, 'pitch': -55.0}) # -85.65 pitch correction..
#robot.scan(project_name="gerbara-2", distance=300, hdr_exposure_times=[5.0, 10.0, 25.0, 50.0, 100.0, 140.0, 180.0, 250.0, 300.0, 350.0]) # exposure_time=75.0, dual_exposure_mode=True, dual_exposure_multiplier=4.0, filter_outliers=True, outliers_std_dev=2.0

#commander.move({'x': 1.20, 'y': 0.075, 'z': 0.50, 'yaw': -12.5, 'pitch': -55.0}) # -85.65 pitch correction..
#robot.scan(project_name="gerbara-3", distance=300, hdr_exposure_times=[5.0, 10.0, 25.0, 50.0, 100.0, 140.0, 180.0, 250.0, 300.0, 350.0]) # exposure_time=75.0, dual_exposure_mode=True, dual_exposure_multiplier=4.0, filter_outliers=True, outliers_std_dev=2.0

#commander.move({'x': 1.17, 'y': 0.16, 'z': 0.50, 'yaw': -34, 'pitch': -55.0}) # -85.65 pitch correction..
#robot = Robot()
#robot.scan(project_name="gerbara-4", distance=300, hdr_exposure_times=[5.0, 10.0, 25.0, 50.0, 100.0, 140.0, 180.0, 250.0, 300.0, 350.0]) # exposure_time=75.0, dual_exposure_mode=True, dual_exposure_multiplier=4.0, filter_outliers=True, outliers_std_dev=2.0

#commander.move({'x': 1.1, 'y': 0.30, 'z': 0.50, 'yaw': -70, 'pitch': -55.0}) # -85.65 pitch correction..
#robot = Robot()
#robot.scan(project_name="gerbara-5", distance=300, hdr_exposure_times=[5.0, 10.0, 25.0, 50.0, 100.0, 140.0, 180.0, 250.0, 300.0, 350.0]) # exposure_time=75.0, dual_exposure_mode=True, dual_exposure_multiplier=4.0, filter_outliers=True, outliers_std_dev=2.0

#commander.move({'x': 1.025, 'y': 0.35, 'z': 0.50, 'yaw': -90, 'pitch': -55.0}) # -85.65 pitch correction..

# robot = Robot()
# robot.scan(project_name="multiview_registration_sample", distance=500, exposure_time=50.0) # exposure_time=75.0, dual_exposure_mode=True, dual_exposure_multiplier=4.0, filter_outliers=True, outliers_std_dev=2.0

#hdr_exposure_times=[25.0, 50.0, 100.0, 125.0, 175.0, 225.0, 275.0, 325.0, 350.0, 400.0]
#pitchs = [i*15 - 86.35 for i in range(10)]
#for pitch in pitchs:
#	commander.move({'pitch': pitch})
#	time.sleep(5.0)


#points = robot.scan(project_name="{}".format("500"), auto_hdr=True)



####### For a sequence of long exposures (HDR), we need to flash the scanner basically in between every call.
####### Besides a different way to pass around files (fine), this opens up room for a cleaner, tighter API.
####### By making an API exclusively for the scanner, we can isolate its various instabilities on a per scan basis.
####### It is true that we have to wait a few seconds per connection, but it's not much more than that!



# SCAN 1
#commander.move({'x': 1.0, 'y': -0.15, 'z': 0.95, 'yaw': 0, 'pitch': -86.35})
# robot = Robot()
# robot.scan(project_name="w1", distance=440, hdr_exposure_times=[5.0, 10.0, 25.0, 50.0, 100.0, 140.0, 180.0, 250.0, 300.0, 350.0]) # exposure_time=75.0, dual_exposure_mode=True, dual_exposure_multiplier=4.0, filter_outliers=True, outliers_std_dev=2.0

# SCAN 2
#commander.move({'x': 1.0, 'y': -0.07, 'z': 0.95, 'yaw': 0, 'pitch': -86.35})
#robot = Robot()
#robot.scan(project_name="w2", distance=440, hdr_exposure_times=[5.0, 10.0, 25.0, 50.0, 100.0, 140.0, 180.0, 250.0, 300.0, 350.0]) # exposure_time=75.0, dual_exposure_mode=True, dual_exposure_multiplier=4.0, filter_outliers=True, outliers_std_dev=2.0

# # SCAN 3
# #commander.move({'x': 1.0, 'y': 0.03, 'z': 0.95, 'yaw': 0, 'pitch': -86.35})
# robot = Robot()
# robot.scan(project_name="w3", distance=440, hdr_exposure_times=[5.0, 10.0, 25.0, 50.0, 100.0, 140.0, 180.0, 250.0, 300.0, 350.0]) # exposure_time=75.0, dual_exposure_mode=True, dual_exposure_multiplier=4.0, filter_outliers=True, outliers_std_dev=2.0

# SCAN 4
#commander.move({'x': 0.825, 'y': 0.03, 'z': 0.95, 'yaw': 0, 'pitch': -86.35})
#robot = Robot()
#robot.scan(project_name="w4", distance=440, hdr_exposure_times=[5.0, 10.0, 25.0, 50.0, 100.0, 140.0, 180.0, 250.0, 300.0, 350.0]) # exposure_time=75.0, dual_exposure_mode=True, dual_exposure_multiplier=4.0, filter_outliers=True, outliers_std_dev=2.0

# SCAN 5
#commander.move({'x': 0.825, 'y': -0.07, 'z': 0.95, 'yaw': 0, 'pitch': -86.35})
#robot = Robot()
#robot.scan(project_name="w5", distance=440, hdr_exposure_times=[5.0, 10.0, 25.0, 50.0, 100.0, 140.0, 180.0, 250.0, 300.0, 350.0]) # exposure_time=75.0, dual_exposure_mode=True, dual_exposure_multiplier=4.0, filter_outliers=True, outliers_std_dev=2.0

# SCAN 6
# commander.move({'x': 0.825, 'y': -0.15, 'z': 0.95, 'yaw': 0, 'pitch': -86.35})
# time.sleep(20.0)
# robot = Robot()
# robot.scan(project_name="w6", distance=440, hdr_exposure_times=[5.0, 10.0, 25.0, 50.0, 100.0, 140.0, 180.0, 250.0, 300.0, 350.0]) # exposure_time=75.0, dual_exposure_mode=True, dual_exposure_multiplier=4.0, filter_outliers=True, outliers_std_dev=2.0

# SCAN 7
#commander.move({'x': 0.775, 'y': -0.45, 'z': 0.85, 'yaw': 70, 'pitch': -55.0})
# time.sleep(20.0)
# robot = Robot()
#robot.scan(project_name="w7", distance=440, hdr_exposure_times=[5.0, 10.0, 25.0, 50.0, 100.0, 140.0, 180.0, 250.0, 300.0, 350.0]) # exposure_time=75.0, dual_exposure_mode=True, dual_exposure_multiplier=4.0, filter_outliers=True, outliers_std_dev=2.0

# SCAN 8

#commander.move({'x': 0.825, 'y': -0.45, 'z': 0.85, 'yaw': 80, 'pitch': -55.0})
#time.sleep(20.0)

#robot.move(x=1.0)
#robot.scan(project_name="test", scan_index=8, distance=440, hdr_exposure_times=[100.0, 200.0]) # exposure_time=75.0, dual_exposure_mode=True, dual_exposure_multiplier=4.0, filter_outliers=True, outliers_std_dev=2.0

