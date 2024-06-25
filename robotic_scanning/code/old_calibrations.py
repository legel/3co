# network settings for each computer

computers = ['pi_1', 'pi_2', 'pi_3', 'udoo']
#ip_address = {'pi_1': '192.168.0.87', 'pi_2': '192.168.0.167', 'pi_3': '', 'udoo': ''}
network_channel_name = '3cobot'
axis_to_computer = {'x1': 'pi_1', 'x2': 'pi_1', 'x': 'pi_1', 'y': 'pi_2', 'z1': 'pi_2', 'z2': 'pi_2', 'z': 'pi_2', 'phi': 'pi_1', 'theta': 'pi_2', 'turn': 'pi_2','projector_focus': 'pi_1', 'camera_focus': 'pi_1', 'camera_aperture': 'pi_1', 'camera_polarization': 'pi_1', 'cali_z1': 'pi_3', 'cali_z2': 'pi_3', 'cali_z': 'pi_3', 'cali_x1': 'pi_3', 'cali_x2': 'pi_3', 'cali_x': 'pi_3', 'cali_y': 'pi_3', 'cali_turn': 'pi_3'}


# wiring and speed settings for each motor
motors = { 'x1':    {'inverse_coordinate': False, 'directional_correction': -1, 'positioning': 'manual_initialization', 'computer': 'pi_1', 'port': 3, 'microsteps': 32, 'max_speed': 300, 'accel': 15, 'decel': 15, 'hold_current': 70, 'run_current': 90,'acc_current': 100, 'dec_current': 100, 'motion': 'linear', 'steps_per_mm': 160, 'min': 0.0, 'max': 2.6},
           'x2':    {'inverse_coordinate': False, 'directional_correction': 1, 'positioning': 'manual_initialization', 'computer': 'pi_1', 'port': 4, 'microsteps': 32, 'max_speed': 300, 'accel': 15, 'decel': 15, 'hold_current': 70, 'run_current': 90,'acc_current': 100, 'dec_current': 100, 'motion': 'linear', 'steps_per_mm': 160, 'min': 0.0, 'max': 2.6},
           'y':     {'inverse_coordinate': False, 'directional_correction': 1, 'positioning': 'manual_initialization', 'computer': 'pi_2', 'port': 3, 'microsteps': 32, 'max_speed': 300, 'accel': 15, 'decel': 15, 'hold_current': 50, 'run_current': 50,'acc_current': 50, 'dec_current': 50,  'motion': 'linear', 'steps_per_mm': 107, 'min': -1.0, 'max': 1.0},
           'z1':    {'inverse_coordinate': True, 'directional_correction': 1, 'positioning': 'recorded_motion', 'computer': 'pi_2', 'port': 0, 'microsteps': 16, 'max_speed': 800, 'accel': 100, 'decel': 100, 'hold_current': 50, 'run_current': 90,'acc_current': 100, 'dec_current': 100, 'motion': 'linear', 'steps_per_mm': 400, 'min': 0.000, 'max': 0.850},
           'z2':    {'inverse_coordinate': True, 'directional_correction': -1, 'positioning': 'recorded_motion', 'computer': 'pi_2', 'port': 1, 'microsteps': 16, 'max_speed': 600, 'accel': 100, 'decel': 100, 'hold_current': 50, 'run_current': 90,'acc_current': 100, 'dec_current': 100, 'motion': 'linear', 'steps_per_mm': 396.5, 'min': 0.000, 'max': 0.850},
           'phi':   {'inverse_coordinate': False, 'directional_correction': 1, 'positioning': 'manual_initialization', 'computer': 'pi_1', 'port': 2, 'microsteps': 128, 'max_speed': 50, 'accel': 15, 'decel': 15, 'hold_current': 100, 'motion': 'angular', 'steps_per_degree': 444.44, 'steps_from_initialization_to_origin': 0, 'min': -90.0, 'max': 180.0},  # 1.8 per step, 128 microsteps, 50 big pulley x2, 20 small pulley
            # theta = yaw; phi = pitch
           'theta': {'inverse_coordinate': False, 'directional_correction': -1, 'positioning': 'manual_initialization', 'computer': 'pi_2', 'port': 4, 'microsteps': 128, 'max_speed': 50, 'accel': 15, 'decel': 15, 'hold_current': 60, 'run_current': 60,'acc_current': 60, 'dec_current': 60,  'motion': 'angular', 'steps_per_degree': 497.77, 'steps_from_initialization_to_origin': 0, 'min': -177.0, 'max': 177.0},
           'turn':  {'inverse_coordinate': False, 'directional_correction': 1, 'positioning': 'manual_initialization', 'computer': 'pi_2', 'port': 2, 'microsteps': 64, 'max_speed': 16, 'accel': 5, 'decel': 5, 'hold_current': 10, 'run_current': 90,'acc_current': 100, 'dec_current': 100,  'motion': 'angular', 'steps_per_degree': 1953.6, 'steps_from_initialization_to_origin': 0, 'min': -100000000.0, 'max': 100000000.0},
           'projector_focus':  {'inverse_coordinate': False, 'directional_correction': -1, 'positioning': 'recorded_motion', 'computer': 'pi_1', 'port': 5, 'microsteps': 16, 'max_speed': 1200, 'accel': 500, 'decel': 500, 'hold_current': 20, 'run_current': 25,'acc_current': 25, 'dec_current': 25,  'motion': 'angular', 'steps_per_degree': 161.2121, 'steps_from_initialization_to_origin': 0, 'min': -45.0, 'max': 45.0},
           'camera_focus':     {'inverse_coordinate': False, 'directional_correction': -1, 'positioning': 'recorded_motion', 'computer': 'pi_1', 'port': 0, 'microsteps': 8, 'max_speed': 1200, 'accel': 600, 'decel': 600, 'hold_current': 75, 'run_current': 75,'acc_current': 75, 'dec_current': 75,  'motion': 'angular', 'steps_per_degree': 115.1515, 'steps_from_initialization_to_origin': 0, 'min': -360.0, 'max': 360.0},
           'camera_aperture':  {'inverse_coordinate': False, 'directional_correction': 1, 'positioning': 'recorded_motion', 'computer': 'pi_1', 'port': 1, 'microsteps': 16, 'max_speed': 1200, 'accel': 500, 'decel': 500, 'hold_current': 45, 'run_current': 45,'acc_current': 45, 'dec_current': 45,  'motion': 'angular', 'steps_per_degree': 184.2424, 'steps_from_initialization_to_origin': 0, 'min': -180.0, 'max': 180.0},
           'camera_polarization':  {'inverse_coordinate': False, 'directional_correction': 1, 'positioning': 'recorded_motion', 'computer': 'pi_1', 'port': 6, 'microsteps': 8, 'max_speed': 5000, 'accel': 500, 'decel': 500, 'hold_current': 45, 'run_current': 45,'acc_current': 45, 'dec_current': 45,  'motion': 'angular', 'steps_per_degree': 405.1222, 'steps_from_initialization_to_origin': 0, 'min': -180.0, 'max': 180.0}, 
           'cali_x1':  {'inverse_coordinate': False, 'directional_correction': -1, 'positioning': 'recorded_motion', 'computer': 'pi_3', 'port': 3, 'microsteps': 2, 'max_speed': 100, 'accel': 10, 'decel': 10, 'hold_current': 10, 'run_current': 10,'acc_current': 10, 'dec_current': 10,  'motion': 'linear', 'steps_per_mm': 1, 'steps_from_initialization_to_origin': 0, 'min': 0.0, 'max': 2.5},  # x1
           'cali_x2':  {'inverse_coordinate': False, 'directional_correction': -1, 'positioning': 'recorded_motion', 'computer': 'pi_3', 'port': 4, 'microsteps': 2, 'max_speed': 100, 'accel': 10, 'decel': 10, 'hold_current': 10, 'run_current': 10,'acc_current': 10, 'dec_current': 10,  'motion': 'linear', 'steps_per_mm': 1, 'steps_from_initialization_to_origin': 0, 'min': 0.0, 'max': 2.5},  # x2
           'cali_y':  {'inverse_coordinate': False, 'directional_correction': -1, 'positioning': 'recorded_motion', 'computer': 'pi_3', 'port': 5, 'microsteps': 2, 'max_speed': 100, 'accel': 10, 'decel': 10, 'hold_current': 10, 'run_current': 10,'acc_current': 10, 'dec_current': 10,  'motion': 'linear', 'steps_per_mm': 1, 'steps_from_initialization_to_origin': 0, 'min': 0.0, 'max': 2.0},   # y
           'cali_z1':  {'inverse_coordinate': False, 'directional_correction': -1, 'positioning': 'recorded_motion', 'computer': 'pi_3', 'port': 0, 'microsteps': 2, 'max_speed': 100, 'accel': 10, 'decel': 10, 'hold_current': 10, 'run_current': 10,'acc_current': 10, 'dec_current': 10,  'motion': 'linear', 'steps_per_mm': 1, 'steps_from_initialization_to_origin': 0, 'min': 0.0, 'max': 0.7},  # z1
           'cali_z2':  {'inverse_coordinate': False, 'directional_correction': -1, 'positioning': 'recorded_motion', 'computer': 'pi_3', 'port': 1, 'microsteps': 2, 'max_speed': 100, 'accel': 10, 'decel': 10, 'hold_current': 10, 'run_current': 10,'acc_current': 10, 'dec_current': 10,  'motion': 'linear', 'steps_per_mm': 1, 'steps_from_initialization_to_origin': 0, 'min': 0.0, 'max': 0.7},  # z2
           'cali_turn':  {'inverse_coordinate': False, 'directional_correction': -1, 'positioning': 'recorded_motion', 'computer': 'pi_3', 'port': 2, 'microsteps': 2, 'max_speed': 100, 'accel': 10, 'decel': 10, 'hold_current': 10, 'run_current': 10,'acc_current': 10, 'dec_current': 10,  'motion': 'angular', 'steps_per_degree': 1, 'steps_from_initialization_to_origin': 0, 'min': -360.0, 'max': 360.0} # turn
           }

multi_motor_axes = { 'x': ['x1', 'x2'],
                     'z': ['z1', 'z2'],
                     'cali_x': ['cali_x1', 'cali_x2'],
                     'cali_z': ['cali_z1', 'cali_z2']
                   }

motors_to_sensors = {'x1': {'orientation_of_sensor_relative_to_axis': 1, 'sensor_address': '/dev/ttyUSB2', 'origin_mm_from_sensor': 1759.4, 'min_mm_from_sensor': 680, 'max_distance_from_sensor': 3035}, 
                     'x2': {'orientation_of_sensor_relative_to_axis': 1, 'sensor_address': '/dev/ttyUSB0', 'origin_mm_from_sensor': 1777.8, 'min_mm_from_sensor': 737, 'max_distance_from_sensor': 3068}, 
                     'y': {'orientation_of_sensor_relative_to_axis': -1, 'sensor_address': '/dev/ttyUSB1', 'origin_mm_from_sensor': 959, 'min_mm_from_sensor': 459, 'max_distance_from_sensor': 1459} 
                     }

min_camera_distance = 0.18
camera_z_offset = 0.075
field_of_view_offset = 0.30 # distance between consecutive photos, for incrementing theta

sensor_calibration_time = 10 # seconds to calibrate x & y position on boot
