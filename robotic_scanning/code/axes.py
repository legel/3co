from commander import *
import time

def calibrate_all_axes():
	start_time = time.time()
	calibrate('pitch')
	move({'pitch': -90.0})
	calibrate('yaw')
	move({'z': 1.0})
	calibrate('x')
	move({'x': 0.1})
	calibrate('y')
	move({'x': 0.45, 'y': -0.7})
	calibrate('z')
	move({'z': 0.5})
	move({'x': 0.0, 'y': 0.0})
	end_time = time.time()
	print("(x,y,z,pitch,yaw) calibrated in {:.1f} minutes".format((end_time-start_time)/60.0))

if __name__ == '__main__':
	calibrate_all_axes()