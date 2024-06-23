from robotics import Robot
robot = Robot()
#robot.calibrate()

	robot.move(x=0.05, y=0.0, z=0.125, pitch=-86.35, yaw=0.0)
	robot.scan(project_name="color_checker_mini_2", scan_index=0, distance=200, hdr_exposure_times=[1.0, 100.0, 300.0], export_to_csv=True) # exposure_time=75.0, dual_exposure_mode=True, dual_exposure_multiplier=4.0, filter_outliers=True, outliers_std_dev=2.0

#robot.move(x=0.05, y=0.0, z=0.175, pitch=-86.35, yaw=0.0)
#robot.scan(project_name="color_checker_mini", scan_index=1, distance=250, hdr_exposure_times=[5.0, 10.0, 25.0, 50.0, 100.0, 140.0, 180.0, 250.0, 300.0, 350.0], export_to_csv=True) # exposure_time=75.0, dual_exposure_mode=True, dual_exposure_multiplier=4.0, filter_outliers=True, outliers_std_dev=2.0
