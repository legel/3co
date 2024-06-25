from commander import *
import time

def calibrate_all_axes():
	start_time = time.time()
	calibrate('theta')
	calibrate('phi')
	move({'phi': -90.0})
	move({'z': 1.0})
	calibrate('x')
	move({'x': 0.1})
	calibrate('y')
	move({'x': 0.45, 'y': -0.7})
	calibrate('z')
	move({'z': 0.5})
	move({'x': 0.0, 'y': 0.0})
	end_time = time.time()
	print("(x,y,z,pitch,yaw) calibrated in {:.1f} seconds".format(end_time-start_time))

# calibrate('phi')
# calibrate('theta')
# move({'theta': -50.0})
#move({'theta': -1.0})
# move({'theta': -50.0})
# move({'theta': -1.0})
# move({'theta': -50.0})
# move({'theta': -1.0})