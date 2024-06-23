from scanning import Scanner
scanner = Scanner(reinitialize_focus=False, camera_aperture_position=0.0, move_positions=False, distance=None)
scanner.find_optimal_focus_values_and_exposure_dynamic_range_for_given_apertures_and_distances(apertures=[0.0], distances=[0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50], initial_x=0.0, initial_y=0.0, initial_z=0.05, initial_pitch=-85.65, initial_yaw=0)