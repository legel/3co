import math

class Metrologist():
	def __init__(self, robot_positions={}):
		self.iris_x = robot_positions.get("iris_x")
		self.iris_y = robot_positions.get("iris_y")
		self.iris_z = robot_positions.get("iris_z")
		self.iris_pitch = robot_positions.get("iris_pitch")
		self.iris_yaw = robot_positions.get("iris_yaw")
		self.cali_x = robot_positions.get("cali_x")
		self.cali_y = robot_positions.get("cali_y")
		self.cali_z = robot_positions.get("cali_z")
		self.cali_turn = robot_positions.get("cali_turn")

		self.bounds = {}
		self.bounds["iris_x"] = {"min": -0.9, "max": 0.9, "mid": 0.0}
		self.bounds["iris_y"] = {"min": -0.9, "max": 0.9, "mid": 0.0}
		self.bounds["iris_z"] = {"min": 0.0, "max": 1.7, "mid": 0.61}
		self.bounds["iris_pitch"] = {"min": -90.0, "max": 55.0, "mid": 0.000}
		self.bounds["iris_yaw"] = {"min": -177.5, "max": 177.5, "mid": 0.000}

		self.bounds["cali_x"] = {"min": -1.0, "max": 1.0, "mid": 1.11, "invert": True, "conversion_to_iris_x": 0.13}
		self.bounds["cali_y"] = {"min": -1.0, "max": 1.0, "mid": 1.05, "invert": True, "conversion_to_iris_y": 0.13}
		self.bounds["cali_z"] = {"min": 0.0, "max": 0.7, "mid": 0.00, "invert": False, "conversion_to_iris_z": 0.560}
		self.bounds["cali_turn"] = {"min": -180.0, "max": 180.0, "mid": 0.000, "invert": False, "conversion_to_iris_yaw": 90.0}

		# self.bounds["cali_x"] = {"min": 0.0, "max": 2.0, "mid": 1.11, "invert": True, "conversion_to_iris_x": 0.085}
		# self.bounds["cali_y"] = {"min": 0.0, "max": 2.0, "mid": 1.05, "invert": True, "conversion_to_iris_y": -1.040}
		# self.bounds["cali_z"] = {"min": 0.0, "max": 0.7, "mid": 1.05, "invert": False, "conversion_to_iris_z": 0.560}
		# self.bounds["cali_turn"] = {"min": -180.0, "max": 180.0, "mid": 0.000, "invert": False, "conversion_to_iris_yaw": 90.0}


	def move(self, iris_x=None, iris_y=None, iris_z=None, iris_pitch=None, iris_yaw=None, cali_x=None, cali_y=None, cali_z=None, cali_turn=None):
		move_command = {}
		# iris x coordinate system is shifted by +0.975, such that iris_x = 0.00 is actually under the hood x=0.975
		if type(iris_x) != type(None):
			if iris_x >= self.bounds["iris_x"]["min"] and iris_x <= self.bounds["iris_x"]["max"]:
				x_command = iris_x + 0.975
				move_command["x"] = x_command
			else:
				print("iris x command {} out of bounds".format(iris_x))
		# iris y coordinate system is shifted by -0.09, such that iris_y = 0.00 is actually under the hood y=-0.09
		if type(iris_y) != type(None):
			if iris_y >= self.bounds["iris_y"]["min"] and iris_y <= self.bounds["iris_y"]["max"]:
				y_command = iris_y - 0.09
				move_command["y"] = y_command
			else:
				print("iris y command {} out of bounds".format(iris_y))
		# iris z coordinate system is the same
		if type(iris_z) != type(None):
			if iris_z >= self.bounds["iris_z"]["min"] and iris_z <= self.bounds["iris_z"]["max"]:
				z_command = iris_z
				move_command["z"] = z_command
			else:
				print("iris z command {} out of bounds".format(iris_z))
		# iris pitch coordinate system is the same
		if type(iris_pitch) != type(None):
			if iris_pitch >= self.bounds["iris_pitch"]["min"] and iris_pitch <= self.bounds["iris_pitch"]["max"]:
				pitch_command = iris_pitch
				move_command["pitch"] = pitch_command
			else:
				print("iris pitch command {} out of bounds".format(iris_pitch))
		# iris yaw coordinate system is the same
		if type(iris_yaw) != type(None):
			if iris_yaw >= self.bounds["iris_yaw"]["min"] and iris_yaw <= self.bounds["iris_yaw"]["max"]:
				yaw_command = iris_yaw
				move_command["yaw"] = yaw_command
			else:
				print("iris yaw command {} out of bounds".format(iris_yaw))

		if type(cali_x) != type(None):
		# cali x coordinate system is inverted by subtracting from its maximum, then shifting to make the center 0.0
			if cali_x >= self.bounds["cali_x"]["min"] and cali_x <= self.bounds["cali_x"]["max"]:
				inversion = 2.0 - cali_x
				origin_in_center = inversion - 1.00 + self.bounds["cali_x"]["conversion_to_iris_x"]
				cali_x_command = origin_in_center
				move_command["cali_x"] = cali_x_command
			else:
				print("cali x command {} out of bounds".format(cali_x))

		if type(cali_y) != type(None):
			# cali y coordinate system is inverted by subtracting from its maximum, then shifting to make the center 0.0
			if cali_y >= self.bounds["cali_y"]["min"] and cali_y <= self.bounds["cali_y"]["max"]:
					inversion = 2.0 - cali_y
					origin_in_center = inversion - 1.00 + self.bounds["cali_y"]["conversion_to_iris_y"]
					cali_y_command = origin_in_center
					move_command["cali_y"] = cali_y_command
			else:
				print("cali y command {} out of bounds".format(cali_y))
		# cali z coordinate system is the same
		if type(cali_z) != type(None):
			if cali_z >= self.bounds["cali_z"]["min"] and cali_z <= self.bounds["cali_z"]["max"]:
				z_command = cali_z
				move_command["cali_z"] = z_command
			else:
				print("cali z command {} out of bounds".format(cali_z))
		# cali turn coordinate system is shifted by 90 degrees, where yaw = 0 corresponds to cali_turn 90.0; yaw = 90 corresponds to cali_turn = 180.0; yaw = 177.5 corresponds to cali_turn = 267.5
		if type(cali_turn) != type(None):
			if cali_turn >= self.bounds["cali_turn"]["min"] and cali_turn <= self.bounds["cali_turn"]["max"]:
				shifted_cali_turn = cali_turn + 90.0
				if shifted_cali_turn >= -180.0 and shifted_cali_turn <= 270.0:
					move_command["cali_turn"] = shifted_cali_turn
			else:
				print("cali turn command {} out of bounds".format(iris_pitch))

		print(move_command)
		return move_command

if __name__ == "__main__":
	m = Metrologist()

	iris_x = 0.0
	iris_y = 0.0
	iris_z = 0.61
	#iris_yaw = 0.0
	iris_pitch = 0.0
	cali_x = 0.0
	cali_y = 0.0
	cali_z = 0.0
	cali_turn = 0.0

	yaws = []
	cali_turns = []
	cali_xs = []
	cali_ys = []
	for iris_yaw in [150]:
		focal_offset = 0.085 # meters
		f_x = -1 * math.cos(math.radians(iris_yaw)) * focal_offset
		f_y = math.sin(math.radians(iris_yaw)) * focal_offset
		view_offset = 0.09
		v_x = math.sin(math.radians(iris_yaw)) * view_offset
		v_y = math.cos(math.radians(iris_yaw)) * view_offset
		delta_x = f_x + v_x
		delta_y = f_y + v_y
		cali_turn = iris_yaw
		move_command = m.move(iris_x=iris_x, iris_y=iris_y, iris_z=iris_z, iris_yaw=iris_yaw, iris_pitch=iris_pitch, cali_x=cali_x + delta_x, cali_y=cali_y + delta_y, cali_z= cali_z, cali_turn=cali_turn)
		yaws.append(move_command["yaw"])
		cali_turns.append(move_command["cali_turn"])
		cali_xs.append(move_command["cali_x"])
		cali_ys.append(move_command["cali_y"])

	with open("dance_cali_data.csv", "w") as output_file:
		for yaw, cali_turn, cali_x, cali_y in zip(yaws, cali_turns, cali_xs, cali_ys):
			output_file.write("{},{},{},{}\n".format(yaw,cali_turn,cali_x,cali_y))