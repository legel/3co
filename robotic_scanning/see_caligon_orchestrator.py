import subprocess
from commander import *

# -90, -75, -60, -45, -30, -15, 
#pitchs_to_see = [-65, -55] # -85, -75, 
#focus_distances = [700, 825] # 550, 625, 
# pitch := [-90, -70, -50]
# yaw := [-135, -90, -45, 0, 45, 90, 135, 177]

# pitchs_to_see = [-80, -60]
# focus_distances = [600, 700]

#move({'z': 1.0, 'pitch': 0, 'yaw': 0})
# print("\nRecalibrating yaw...\n")
# calibrate('yaw')

# print('\nRecalibrating pitch...\n')
# calibrate('pitch')

# pitchs_to_see = [-45, -15] # -85, -75, 
# zs_to_see = [750, 100]
# focus = 800 # 550, 625, 


#focus_distances = [900, 1100, 1300] #int(sys.argv[3])
batches = 4
trials_per_batch = 25
views = 0
for batch in range(batches):
	subprocess.call(["python3", "scan.py", "{}".format(views)])
	views += trials_per_batch