import commander
import time

# # main calibration routine
#commander.move({'z': 0.7})


# #commander.move({'x': 0.0, 'y': 0.0})

# # # never calibrate z before y, unless z is lowered by 0.2 (e.g. 1.6)

# #commander.move({'x': 0.05})#, 'y': 0.0, 'z': 0.81, 'yaw': -90.0, 'pitch': -86.65}) # -85.65 pitch correction..

# commander.calibrate('x')
# commander.calibrate('y')
# commander.calibrate('z')
# commander.calibrate('pitch')
# commander.calibrate('yaw')

#commander.move({'x': 0.5})
#commander.move({'yaw': 165.0, 'pitch': 15, 'y': -0.60, 'z': 1.25}) # -85.65 pitch correction..
#commander.move({'z': 1.5}) # -85.65 pitch correction..

#commander.move({'z': 0.9}) # -85.65 pitch correction..


# -0.225
# -83.65
#commander.move({'x': 1.4, 'y': 0.0, 'z': 0.61, 'yaw': -40, 'pitch': 4.35}) # -85.65 pitch correction..

# commander.calibrate('camera_aperture')
# commander.calibrate('camera_focus')
# commander.calibrate('projector_focus')
#commander.move({'pitch': 40})


#commander.move({'x': 0.05, 'y': 0.0, 'z': 0.61, 'yaw': -157.0, 'pitch': -86.35}) # -85.65 pitch correction..
#time.sleep(10.0)
#commander.move({'x': 0.05, 'y': 0.0, 'z': 0.61, 'yaw': -157.0, 'pitch': 45.35}) # -85.65 pitch correction..


#commander.move({'yaw': -90.0})
#commander.move({'camera_polarization': 59})

# for i in range(10):           


# time.sleep(5.0)

# commander.move({'projector_focus': 14})
# commander.move({'camera_aperture': 90})
# commander.move({'camera_focus': 72})


# for i in range(50):
# 	print(i*3)
# 	commander.move({'camera_focus': i*3})
# 	time.sleep(1.0)


#commander.move({'camera_focus': 250})
#commander.move({'x': 0.0, 'y': 0.0, 'z': 0.2, 'pitch': 0, 'yaw': 0})
# # # # commander.move({'y': 0.0})
# commander.move({'y': 0.15})

# # commander.calibrate('x')
# # commander.calibrate('y')    

#commander.recoordinate('pitch')

# #commander.move({'camera_aperture': 15})

#commander.recoordinate('projector_focus')


# commander.calibrate('camera_focus')
# commander.calibrate('camera_aperture')
# commander.calibrate('projector_focus')

#commander.move({'camera_aperture': 90})
#commander.move({'camera_aperture': 0})
# commander.move({'camera_focus': 100})

#  commander.move({'projector_focus': 11})

# commander.free('camera_focus')
# commander.free('camera_aperture')
#commander.free('projector_focus')

#commander.free('projector_focus')

#commander.calibrate('camera_polarization')
#commander.calibrate('z')



#commander.move({'yaw': 175.0})
# commander.move({'z': 0.78})

# commander.move({'x': 0.01})
# commander.move({'z': 0.2})
#commander.move({'yaw': 0})
# commander.move({'pitch': 0})

#commander.move({'yaw': 5})

# #time.sleep(15.0)

# commander.move({'x': 0.00})
# commander.move({'y': 0.00})
# commander.move({'z': 0.60})
# commander.move({'pitch': 0})
#commander.move({'yaw': 0.0})


#commander.calibrate('x')


#commander.move({'z': 0.30})

# commander.move({'yaw': -95})
# commander.move({'pitch': -20})
# commander.move({'y': 0.80})
# commander.move({'y': 0.0})
# commander.move({'yaw': -165})
# commander.move({'z': 0.0})
# commander.move({'yaw': 0})
# commander.move({'pitch': 0})
# commander.calibrate('x')


# commander.move({'z': 0.0})
# commander.move({'y': 0.7})
# commander.move({'x': 1.40})
# commander.move({'yaw': -25})
# commander.move({'pitch': -10})


#commander.move({'x': 1.25})
#commander.move({'z': 0.0})

# commander.move({'z': 0.})




# #commander.move({'pitch': -90, 'yaw': 175})
# commander.calibrate('camera_focus')
# commander.calibrate('projector_focus')
# commander.calibrate('camera_aperture')

#commander.move({'pitch': -45.0})
#commander.free('y')

# # # #commander.move({'x': 1.0, 'y': -0.1, 'z': 1.70, 'pitch': -90.0, 'yaw': 0.0})
# # #commander.move({'x': 1.0, 'y': -0.1, 'z': 1.5, 'pitch': -90.0, 'yaw': 0.0})
#commander.move({'x': 0.95, 'y': -0.2, 'z': 0.6, 'pitch': -90.0, 'yaw': 0.0})

# pitch := [-90, -70, -50]
# yaw := [-135, -90, -45, 0, 45, 90, 135, 177]

#commander.move({'x': 1.0, 'y': -0.1, 'z': 0.30, 'pitch': -90.0, 'yaw': 175.0})


# free
# commander.free('pitch')

#time.sleep(10.0)
#commander.move({'x': 0.0, 'y': 0.70, 'z': 0.2, 'pitch': 45.0, 'yaw': 90.0})
#commander.move({'x': 0.0, 'y': 0.70, 'z': 0.2, 'pitch': -90.0, 'yaw': 90.0})


# focus adjustments
#commander.calibrate('camera_focus')
#commander.free('camera_focus')

#commander.calibrate('projector_focus')
#commander.free('projector_focus')

#commander.calibrate('camera_aperture')
#commander.free('camera_aperture')

#commander.move({'camera_focus': 170.0})
#commander.move({'projector_focus': 22.0})
#commander.move({'camera_aperture': 75.0})

# presenting:
#commander.move({'x': 1.7, 'y': -0.55, 'z': 0.7, 'pitch': -90, 'yaw': 150})