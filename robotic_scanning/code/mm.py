from time import sleep
from commander import *
#move)
#calibrate('x')
#calibrate('y')
#calibrate('theta')
#calibrate('phi')
#recoordinate(['phi'])
#free('theta')

# y -0.12 (6.88) -> -0.131 (-4.286) -> -0.1258 (0.8365)
# x 0.015 (-9.47) -> 0.0165 (-7.6) -> 0.026 (1.487)
#move({'theta': 0 , 'phi': -91.0, 'x': 0.026, 'y': -0.1258, 'z': 0.40})
# y -0.125 (3) -> -0.13 (-7) -> -0.1265 (0.88) -> -0.1258 (1.7) -> -0.1269 (1.172) -> -0.1289 (-0.92)
# x 0.015 (5.99) -> 0.022 (-3) -> 0.0199 (11.45) -> 0.026 (16.48) -> 0.008 (-1.236) -> 0.0097 (-0.025)
#move({'theta': 0})
#move({'z': 0.40})
#move({'y': -0.1276})
#move({'x': 0.0097})
#move({'phi': -87.3})

move({'x': 0.2})
move({'phi': -80})

#recoordinate('phi')
#move({'theta': 0, 'y': 0, 'x': 0, 'z': 0.5})


