
# DONE - I. GET FIXED MECHANICAL REFERENCE

# [ Ensure mechanical integrity to system prior to calibration process ]

# 1. x: calibrate (mechanical end point closest to turntable)
# 2. y: calibrate (mechanical middle point)
# 3. z: calibrate (mechanical end point closest to ground)
# 4. pitch: calibrate (mechanical middle point, facing vertical center)
# 5. yaw: calibrate (mechanical middle point, facing toward turntable end of x-axis)

# [For a fixed mechanical system, now we have a fixed reference (x,y,z,pitch,yaw)]

# II. SCAN FOR PLANES USING DISTANCE

# TODO:
# 	def scan_for_planes(target_distance):
#			focus_scanner(target_distance)
#			scan( )
#			plane_finder(target_distance)
#			return plane points and metadata

# TODO: def planar_target_distance():
#			AIM FOR A ROUGH APPROXIMATION BASED ON MAXIMUM DENSITY, AVERAGE ERROR AS A FUNCTION OF DISTANCE 
#			return (minimum_planar_error, minimum_number_of_points, ...) 

# 6. Find estimate for (x=0, y=0, z=0) which is defined as the center of the turntable, exactly on its surface
#
#	V: sufficient_distance_from_ground =  0.75
# 	V: distance_to_turntable_estimate = sufficient_distance_from_ground + 0.05
#
# 	6.1		move_robot({pitch: 0, yaw: 90, z: sufficient_distance_from_ground}) # sufficient to ensure that full turntable in FOV
#	6.2 	scan_for_planes(distance_to_turntable_estimate)
#
#				Identify points of plane of turntable (probably has 2nd most points, z per point probably 2nd largest)
#
#				# TODO: def estimate_turntable_center_point():
#						fit points of plane a circle with a radius estimated by mechanical information, and center point
#						return 					turntable_center_point = Point(x=scanner_x_offset, y=scanner_y_offset, z=scanner_depth)
#
#	V: ideal_block_and_turntable_plane_measurement_distance = 0.40
#	V: turntable_origin = Point(0.0, 0.0, 0.0)
# 	V: turntable_origin.x = -1 * scanner_y_offset
#	V: turntable_origin.y = -1 * scanner_x_offset
#	V: turntable_origin.z = distance_to_turntable_estimate - scanner_depth
#
# 7. move_robot({x: turntable_origin.x, y: turntable_origin.y, z: turntable_origin.z + ideal_block_and_turntable_plane_measurement_distance})
#
# 8. Find estimate for block top plane center point
#
#		8.1 scan_for_planes(ideal_block_and_turntable_plane_measurement_distance - inches_to_mm(3)) # assuming block is sitting with 3" side vertical
#				Identify points of plane of top block (probably easiest to filter by z distance ~= ideal_block_and_turntable_plane_measurement_distance - inches_to_mm(3))
#
#				# TODO: def plane_of_top_block_center_point():
#							# method a: try to measure all three planes, get 3 corner points of top plane, triangulate middle positions, triangulate middle point
#							# method b: compute average value of all points in top plane
#							# -> := block_center_point
#							return block_top_center_point_in_scanner_coordinate_system = Point(block_center_point.x, block_center_point.y, block_center_point.z)
#							
#	V: block_center.x = -1 * block_top_center_point_in_scanner_coordinate_system.y
#	V: block_center.y = -1 * block_top_center_point_in_scanner_coordinate_system.x
# 	ASSERT that ideal_block_and_turntable_plane_measurement_distance - block_top_center_point_in_scanner_coordinate_system.z ~= 3 inches
#	V: block_center.z = turntable_origin.z + inches_to_mm(3) or turntable_origin.z + ideal_block_and_turntable_plane_measurement_distance - block_top_center_point_in_scanner_coordinate_system.z
#
# 9. move_robot({x: block_center.x, y: block_center.y}) # now the block should be directly under the scanner, and in the center of the region of interest 
#
# 10. focus_scanner(target_distance = block_top_center_point_in_scanner_coordinate_system.z)
#
# 11. Find (pitch=0) which is defined as directly perpendicular to the plane of the turntable / block
# 	
# 	V: vertical_slope = only for points in block plane, for minimum to maximum pixel columns, from i=top (+y) to bottom (-y), average(point(i+1).z - point(i).z) 
#	V: worst_case_pitch_offset_in_z_axis_in_mm = 25
#	V: average_vertical_angle = 100
#	V: vertical angle = math.arcsin((p(i).z - p(i+1).z)/(p(i).y - p(i+1).y))
#
#	while average_vertical_angle > 0.1 degrees: # or e.g. 1 millimeter
#		scanned_points = scan(filter_by_z < block_top_center_point_in_scanner_coordinate_system.z + worst_case_pitch_offset_in_z_axis_in_mm)
#		top_plane_points = plane_finder(scanned_points, block_top_center_point_in_scanner_coordinate_system.z)
#		average_vertical_angle = compute_average_vertical_angle(top_plane_points)
#		move_robot(pitch += average_vertical_angle)
#
#	SAVE: fixed reference pitch - current_pitch: new pitch = 0
#
# 12. Recenter with function with (8. Find estimate for block top center point) 
# 13. Recompute (pitch=0) in (11.)
# 14. Find fixed constant value for *roll*:
#	
#		V:	n_roll_samples = 5
#		V: 	random_sample_deviation = 0.05
#		V: 	horizontal_angle = math.arcsin((p(i).z - p(i+1).z)/(p(i).x - p(i+1).x))
#		
#		all_horizontal_angles = []
#		for roll_sample in range(n_roll_samples):
#			# next_random_sample = get random samples of +- random_sample_deviation% for changes in (x,y,z) and maybe (pitch, yaw)
#			move_robot(next_random_sample)
#			scanned_points = scan(filter_by_z < block_top_center_point_in_scanner_coordinate_system.z + worst_case_pitch_offset_in_z_axis_in_mm)
# 			top_plane_points = plane_finder(scanned_points, block_top_center_point_in_scanner_coordinate_system.z)
#			average_horizontal_angle = compute_average_horizontal_angle(top_plane_points)
#			all_horizontal_angles.apend(average_horizontal_angle)
#
# 	ASSERT that horizontal angles from different perspectives are all approximately in same space
# 
#	VALIDATE: Code a visualization of a scan of the block, which uses the horizontal angle of the block as an estimate for the roll in the system, such that all points are automatically "rolled" into exact fit
#	VALIDATE: Confirm from multiple held out perspectives of the block that its corners are always parallel to the corners (with bisected center point) of the camera sensor plane 
#	AUTOMATE: Automatically rotate all future scans indefinitely with this roll value
#
# 15. Calibrate the z-axis with respect to its relative orientation in the x-y plane
#		
#		V:	n_z_samples = 5
#		V: 	z_step = 0.01
#
#		all_horizontal_angles = []
#		all_vertical_angles = []
#		for z_sample in range(n_z_samples):
#			move_robot(next_z_sample)
#			scanned_points = scan(filter_by_z < block_top_center_point_in_scanner_coordinate_system.z + worst_case_pitch_offset_in_z_axis_in_mm)
# 			top_plane_points = plane_finder(scanned_points, block_top_center_point_in_scanner_coordinate_system.z)
#			average_horizontal_angle = compute_average_horizontal_angle(top_plane_points)
#			all_horizontal_angles.apend(average_horizontal_angle)
#			average_vertical_angle = compute_average_vertical_angle(top_plane_points)
#			all_vertical_angles.append(average_vertical_angle)
#			ASSERT all_vertical_angles values are approximately the same; all_horizontal_angles values are approximately the same;
#			ELSE we have a non-linear rotation in the z-axis motion...
#			get 4 corners of top block plane
#			measure translation in x and y directions for change in v: compute the slope_x and slope_y that can predict this translation
#			# TODO: For a sequence of different x and y directions, try to extract z-only orientation...
#  
# 16. Approaching calibrations of x axis and y axis using above type of approaches
# 
# 17. Calibration of yaw using approach along x-axis, measuring relative planar angle until the relative angle does not change below threshold
#
# 18. Rotate block around turntable with fixed scanner position to determine true center of turntable; set that true (x,y,z) = 0
# 19. With the absolute coordinate system for all axes now calibrated, capture N perspectives of block corners
# 20. Fit a mechanical model for position of (x,y,z)_scanner as a function of pitch (rotation around constant mechanical values)
# 21. Fit a mechanical model for position of (x,y,z)_scanner as a function of yaw (rotation around constant mechanical values)
# 22. Validate that predictive model can be given calibrated (x,y,z) estimate and mechanically derived adjustment with (pitch,yaw) and roll estimate to auto-rotate 3D points from one perspective into alignment with another; measure error; 
# 23. Potentially, get N (where N == very large number, e.g. 1000) perspectives and seek to fit a deep model around the most unknown / unexplained variables (but be careful relying on this idea)
# 24. Write production-level functions for taking in all calibration information, for waking up the robot, and for auto-rotating a sequence of multiple scans into one scan
# 25. Visualize!
