import time
import commander 

class Robot():
	def __init__(self,x=0.0, y=0.0, z=1.0, pitch=0.0, yaw=0.0):
		self.x
		self.y
		self.z
		self.pitch
		self.yaw
		self.move()

	def calibrate_axis(self, axis):
		commander.calibrate(axis)
		if axis == 'x':
			self.x = 0.0
		if axis == 'y':
			self.y = 0.0
		if axis == 'z':
			self.z = 0.0
		if axis == 'pitch':
			self.pitch = 0.0
		if axis == 'yaw':
			self.yaw = 0.0

	def calibrate(self):
		start_time = time.time()
		self.calibrate_axis('pitch')
		self.move(pitch = -90.0)
		self.calibrate_axis('yaw')
		self.move(z = 1.0)
		self.calibrate_axis('x')
		self.move(x = 0.1)
		self.calibrate_axis('y')
		self.move(x = 0.45, y = -0.7)
		self.calibrate_axis('z')
		self.move(z = 0.5)
		self.move(x = 0.0, y = 0.0)
		end_time = time.time()
		print("(x,y,z,pitch,yaw) calibrated in {:.1f} minutes".format((end_time-start_time)/60.0))

	def move(self, x=None, y=None, z=None, pitch=None, yaw=None):
		if x:
			self.x = x
		if y:
			self.y = y
		if z:
			self.z = z
		if pitch:
			self.pitch = pitch
		if yaw:
			self.yaw = yaw
		commander.move({'x': self.x, 'y': self.y, 'z': self.z, 'pitch': self.pitch, 'yaw': self.yaw})
