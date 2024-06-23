from time import sleep
from commander import *

## initialize by moving z all the way to the top
#move({'z': 0.75})

## then calibrate x, and move system to safe x-spot for y calibration
#calibrate('x') # moves back only
#move({'x': 0.3})

## calibrate y
#calibrate('y') # first moves left, then right, then centers

# now move system to safe spot for z calibration 
#move({'x': 0.6})
##calibrate('z') # z-calibration is currently skipped in favor of recorded motion
#move({'z': 0.2})

# now calibrate pitch and yaw 
#calibrate('theta')
#move({'theta': 0})
calibrate('phi')
