from six_dimensional import roll_correction, pitch_correction, yaw_correction


# NEED TO REIMPLEMENT IF DESIRED TO USE

	def rotation_by_rodrigues_vector(self, rotation_vector):
		# OpenCV uses Rodrigues Vectors which is not the same as Euler, it is in "axis-angle" representation (see below)
		# https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
		# https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
		rotation_matrix = np.zeros(shape=(3,3))
		cv2.Rodrigues(rotation_vector, rotation_matrix) # convert vector to matrix
		#rotation_matrix = rotation_matrix.transpose()
		#rotation_matrix = rotation_matrix * -1
		all_points = np.matrix([self.x,self.y,self.z]) # note order difference between axes X and Y are interchanged
		all_points_rotated = rotation_matrix * all_points

		self.x = np.asarray(all_points_rotated[0,:]).flatten()
		self.y = np.asarray(all_points_rotated[1,:]).flatten()
		self.z = np.asarray(all_points_rotated[2,:]).flatten()

		# i = 0
		# for i in range(len(self.x)):
		# 	x = self.x[i]
		# 	y = self.y[i]
		# 	z = self.z[i]
		# 	if i % 100000 == 0:
		# 		print("({}) Rotating point (x,y,z)=({},{},{}) by Rodrigues matrix {}".format(i, self.x[i], self.y[i], self.z[i], rotation_matrix))
		# 	point = np.matrix([[x],[y],[z]])
		# 	rotated_point = rotation_matrix * point 
		# 	self.x[i] = rotated_point[0]
		# 	self.y[i] = rotated_point[1]
		# 	self.z[i] = rotated_point[2]
		# 	if i % 100000 == 0:
		# 		print("({}) After rotation, point now (x,y,z)=({},{},{})".format(i, self.x[i], self.y[i], self.z[i]))

		#print(rotated_point)
		#xyz = np.dstack([self.x, self.y, self.z])[0]
		#rotated_xyz = xyz * rotation_matrix
		#self.x = np.take(rotated_xyz, indices=0, axis=1)
		#self.y = np.take(rotated_xyz, indices=1, axis=1)
		#self.z = np.take(rotated_xyz, indices=2, axis=1)

	def translation_by_vector(self, translation_vector):
		# OpenCV translation is with reference to camera coordinate system, where left-right in image is x-axis; up-down is y-axis; back-forward is z-axis
		print("Translating points, e.g. (x,y,z)_1=({},{},{}) at (h,v)=({},{})...".format(self.x[0], self.y[0], self.z[0], self.pixel_column[0], self.pixel_row[0]))

		self.x = self.x + translation_vector[0] # +
		self.y = self.y + translation_vector[1] # +
		self.z = self.z - translation_vector[2]

		all_points = np.matrix([self.x,self.y,self.z]) 

		print("...is translated to (x,y,z)_2=({},{},{})".format(self.x[0], self.y[0], self.z[0]))


	def homogeneous_transform_inverted(self, rotation_vector=None, translation_vector=None, rotation_matrix=None):
		# our situation is unique in that we technically have camera coordinates, (x_c, y_c, z_c), i.e. 3D coordinates with respect to each camera's coordinate system
		# and we want to convert those coordinates into world coordinates (x_w, y_w, z_w), i.e. a coordinate system in which all cameras are embedded
		# in fact, if you study the equation for a homogeneous transform, all we have to do is to compute the inverse of the transformation matrix, and then multiply that by camera coordinates
		# https://docs.opencv.org/4.4.0/d9/d0c/group__calib3d.html#ga13159f129eec8a7d9bd8501f012d5543

		# get the 3x3 rotation matrix from the Rodrigues vector, and add it to the transform; compute the rotation inverse manually, like from http://vr.cs.uiuc.edu/node81.html
		if type(rotation_matrix) == type(None):
			rotation_matrix = np.zeros(shape=(3,3))
			cv2.Rodrigues(rotation_vector, rotation_matrix)
		inverted_rotation = rotation_matrix.transpose()
		print("Inverted rotation:\n{}".format(inverted_rotation))

		# first construct the 4x4 homogeneous transformation matrix, for both rotation and translation, as from http://vr.cs.uiuc.edu/node81.html
		inverted_transform_matrix = np.zeros(shape=(4,4))
		inverted_transform_matrix[3,3] = 1.0
		inverted_transform_matrix[:3,:3] = inverted_rotation



		isolated_rotation_transform = np.zeros(shape=(4,4))
		isolated_rotation_transform[3,3] = 1.0
		isolated_rotation_transform[:3,:3] = inverted_rotation

		isolated_translate_transform = np.eye(4)
		#inverted_translate_transform = np.eye(4)

		# now we can add the inverted rotation
		#inverted_rotation_transform[:3,:3] = inverted_rotation

		#  by linear algebra convention set the last element to 1
		#inverted_rotation_transform[3,3] = 1.0

		# now for the translation, we just compute the negative of the operation
		print("Shape of translation vector: {}".format(translation_vector.shape))
		#translation_vector = translation_vector.flatten()
		negative_inverted_rotation = -inverted_rotation
		print("Negative inverted rotation:\n{}".format(negative_inverted_rotation))
		print("Translation vector:\n{}".format(translation_vector))
		matrix_translation_vector = np.matrix(translation_vector)
		print("Matrix translation vector with shape {}: {}".format(matrix_translation_vector.shape, matrix_translation_vector))
		inverted_translation_vector = negative_inverted_rotation * matrix_translation_vector #np.dot(negative_inverted_rotation, translation_vector)
		print("Negative rotated (inverted) translation vector:\n{}".format(inverted_translation_vector))

		#print("Translation vector inverse:\nFrom {}\nTo {}".format(translation_vector, inverted_translation_vector))

		# now add the translation to the right
		inverted_transform_matrix[:3,3] = np.asarray(inverted_translation_vector).flatten()

		isolated_translate_transform[:3,3] = inverted_transform_matrix[:3,3] #negative_inverted_rotation * np.matrix(translation_vector) #inverted_transform_matrix[:3,3]


		# now our transform matrix from world coordinates to camera coordinates is complete
		#transform_matrix_inverse = np.linalg.inv(transform_matrix)

		transform_matrix_original = np.zeros(shape=(4,4))
		transform_matrix_original[:3,:3] = rotation_matrix
		transform_matrix_original[:3,3] = translation_vector.flatten()
		transform_matrix_original[3,3] = 1.0

		print("\nHere is the inverted 4x4 homogeneous transformation matrix:\n{}".format(inverted_transform_matrix))
		print("\n...and the original 4x4 homogeneous transformation matrix:\n{}".format(transform_matrix_original))

		print("As a test of proper inversion, see that the dot product is zero between original matrix and manually inverted matrix")
		dot_product = np.dot(inverted_transform_matrix, transform_matrix_original)
		print("Dot product:\n{}".format(dot_product))

		numpy_transform_matrix_inverse = np.linalg.inv(transform_matrix_original)
		print("For comparison, here is the NumPy computed inverse:\n{}".format(numpy_transform_matrix_inverse))

		#print("\n...and it's inverse:\n{}".format(transform_matrix_inverse))
		#print("\n...with the following dot product for verification:\n{}".format(np.dot(transform_matrix, transform_matrix_inverse)))

		# now convert our Euclidian coordinates to homogeneous coordinates, by making 4 columns where columns are (x,y,z,1), and then every row is a separate point

		homogeneous_ones = np.ones(len(self.x))
		homogeneous_points = np.matrix([self.x, self.y, self.z, homogeneous_ones])

		print("\n...to a point cloud of shape {}, with an example point:\n{}".format(homogeneous_points.shape, homogeneous_points[:,0]))

		initial_transformed_points = isolated_translate_transform * homogeneous_points
		transformed_points = isolated_rotation_transform * initial_transformed_points

		combined_transformed_points = inverted_transform_matrix * homogeneous_points

		#transformed_points = inverted_rotation_transform * transformed_points

		print("\n...resulting in a final output of shape {}, with an example transformed point using combined method:\n{}".format(combined_transformed_points.shape, combined_transformed_points[:,0]))
		print("\n...resulting in a final output of shape {}, with an example transformed point using rotation-then-translation method:\n{}".format(transformed_points.shape, transformed_points[:,0]))

		self.x = transformed_points[0,:]
		self.y = transformed_points[1,:]
		self.z = transformed_points[2,:]
		self.x = np.asarray(self.x).flatten()
		self.y = np.asarray(self.y).flatten()
		self.z = np.asarray(self.z).flatten()

		print("Shape of X after: {}".format(self.x.shape))




	def homogeneous_transform(self, rotation_matrix, translation_matrix):
		# get the 3x3 rotation matrix from the Rodrigues vector, and add it to the transform; compute the rotation inverse manually, like from http://vr.cs.uiuc.edu/node81.html

		#inverted_rotation = rotation_matrix.transpose()
		print("Rotation matrix:\n{}".format(rotation_matrix))

		# first construct the 4x4 homogeneous transformation matrix, for both rotation and translation, as from http://vr.cs.uiuc.edu/node81.html
		transform_matrix = np.zeros(shape=(4,4))
		transform_matrix[3,3] = 1.0
		transform_matrix[:3,:3] = rotation_matrix

		# isolated_rotation_transform = np.zeros(shape=(4,4))
		# isolated_rotation_transform[3,3] = 1.0
		# isolated_rotation_transform[:3,:3] = inverted_rotation

		# isolated_translate_transform = np.eye(4)
		#inverted_translate_transform = np.eye(4)

		# now we can add the inverted rotation
		#inverted_rotation_transform[:3,:3] = inverted_rotation

		#  by linear algebra convention set the last element to 1
		#inverted_rotation_transform[3,3] = 1.0

		# now for the translation, we just compute the negative of the operation
		# print("Shape of translation vector: {}".format(translation_vector.shape))
		#translation_vector = translation_vector.flatten()
		# negative_inverted_rotation = -inverted_rotation
		# print("Negative inverted rotation:\n{}".format(negative_inverted_rotation))
		print("Translation matrix:\n{}".format(translation_matrix))

		# matrix_translation_vector = np.matrix(translation_vector)
		# # print("Matrix translation vector with shape {}: {}".format(matrix_translation_vector.shape, matrix_translation_vector))
		# inverted_translation_vector = negative_inverted_rotation * matrix_translation_vector #np.dot(negative_inverted_rotation, translation_vector)
		# print("Negative rotated (inverted) translation vector:\n{}".format(inverted_translation_vector))

		#print("Translation vector inverse:\nFrom {}\nTo {}".format(translation_vector, inverted_translation_vector))

		translation_matrix = np.asarray(translation_matrix).flatten() * 1000
		print("Translation matrix x1000:\n{}".format(translation_matrix))
		# now add the translation to the right
		transform_matrix[:3,3] = translation_matrix


		# isolated_translate_transform[:3,3] = inverted_transform_matrix[:3,3] #negative_inverted_rotation * np.matrix(translation_vector) #inverted_transform_matrix[:3,3]


		# now our transform matrix from world coordinates to camera coordinates is complete
		#transform_matrix_inverse = np.linalg.inv(transform_matrix)

		# transform_matrix_original = np.zeros(shape=(4,4))
		# transform_matrix_original[:3,:3] = rotation_matrix
		# transform_matrix_original[:3,3] = translation_vector.flatten()
		# transform_matrix_original[3,3] = 1.0

		# print("\nHere is the inverted 4x4 homogeneous transformation matrix:\n{}".format(inverted_transform_matrix))
		# print("\n...and the original 4x4 homogeneous transformation matrix:\n{}".format(transform_matrix_original))

		# print("As a test of proper inversion, see that the dot product is zero between original matrix and manually inverted matrix")
		# dot_product = np.dot(inverted_transform_matrix, transform_matrix_original)
		# print("Dot product:\n{}".format(dot_product))

		# numpy_transform_matrix_inverse = np.linalg.inv(transform_matrix_original)
		# print("For comparison, here is the NumPy computed inverse:\n{}".format(numpy_transform_matrix_inverse))

		#print("\n...and it's inverse:\n{}".format(transform_matrix_inverse))
		#print("\n...with the following dot product for verification:\n{}".format(np.dot(transform_matrix, transform_matrix_inverse)))

		# now convert our Euclidian coordinates to homogeneous coordinates, by making 4 columns where columns are (x,y,z,1), and then every row is a separate point

		homogeneous_ones = np.ones(len(self.x))
		homogeneous_points = np.matrix([self.x, self.y, self.z, homogeneous_ones])

		print("\nStartng with a point cloud of shape {}, with an example point:\n{}".format(homogeneous_points.shape, homogeneous_points[:,0]))

		transformed_points = transform_matrix * homogeneous_points

		#transformed_points = inverted_rotation_transform * transformed_points

		print("\n...resulting in a final output of shape {}, with an example transformed point using combined method:\n{}".format(transformed_points.shape, transformed_points[:,0]))

		self.x = transformed_points[0,:]
		self.y = transformed_points[1,:]
		self.z = transformed_points[2,:]
		self.x = np.asarray(self.x).flatten()
		self.y = np.asarray(self.y).flatten()
		self.z = np.asarray(self.z).flatten()

		print("Shape of X after: {}".format(self.x.shape))


	def transform(self, rotation_matrix, translation_matrix):
		#print("Rotation matrix:\n{}".format(rotation_matrix))

		#homogeneous_ones = np.ones(len(self.x))
		points = np.matrix([self.x, self.y, self.z])
		points = np.transpose(points)
		print("\nStartng with a point cloud of shape {}, with an example point:\n{}".format(points.shape, points[0,:]))

		#print("\n...to a point cloud of shape {}, with an example point:\n{}".format(homogeneous_points.shape, homogeneous_points[:,0]))

		#transformed_points = transform_matrix * homogeneous_points
		scale_factor = 1.0000 # 1.000025
		transformed_points = scale_factor * np.dot(points, rotation_matrix) + translation_matrix

		#transformed_points = inverted_rotation_transform * transformed_points

		print("\n...resulting in a final output of shape {}, with an example transformed point using combined method:\n{}".format(transformed_points.shape, transformed_points[0,:]))

		self.x = transformed_points[:,0]
		self.y = transformed_points[:,1]
		self.z = transformed_points[:,2]
		self.x = np.asarray(self.x).flatten()
		self.y = np.asarray(self.y).flatten()
		self.z = np.asarray(self.z).flatten()

		print("Shape of X after: {}".format(self.x.shape))




	def transformation(self, rotation_vector, translation_vector):
		# Specific to Open CV functions, see above

		self.x = self.x + translation_vector[0] # +
		self.y = self.y + translation_vector[1] # +
		self.z = self.z - translation_vector[2]

		rotation_matrix = np.zeros(shape=(3,3))

		if rotation_vector[2] < 0:
			rotation_vector[0] = -1 * rotation_vector[0]
			rotation_vector[1] = -1 * rotation_vector[1]
			rotation_vector[2] = -1 * rotation_vector[2]

		#rotation_vector = np.matrix([rotation_vector[1], rotation_vector[0], rotation_vector[2]])
		cv2.Rodrigues(rotation_vector, rotation_matrix) # convert vector to matrix
		#rotation_matrix = np.transpose(rotation_matrix, axes=[1,0]) # , (0, 1, 2)

		#np.hstack(rotation_matrix, translation_vector)


		# sin_x = math.sqrt(rotation_matrix[2,0] * rotation_matrix[2,0] +  rotation_matrix[2,1] * rotation_matrix[2,1])    
		# x  = math.atan2(sin_x,  rotation_matrix[2,2])     				# around x-axis

		# print(x)

		# rotation_matrix[0,:] = rotation_matrix[2,:] 
		# rotation_matrix[1,:] = rotation_matrix[0,:] 
		# rotation_matrix[2,:] = rotation_matrix[1,:] 

		#rotation_matrix = np.asarray(rotation_matrix)
		#print(rotation_matrix)
		#rotation_matrix = np.transpose(rotation_matrix, axes=[1,0]) # , (0, 1, 2)
		#rotation_matrix = rotation_matrix * -1
		#print(rotation_matrix)

		all_points = np.matrix([self.x,self.y,self.z]) # note order difference between axes X and Y are interchanged

		#rotation_matrix = np.linalg.inv(rotation_matrix)
		all_points_transformed = rotation_matrix * all_points #+ translation_vector

		#print(all_points_transformed.shape)

		self.x = np.asarray(all_points_transformed[0,:]).flatten()
		self.y = np.asarray(all_points_transformed[1,:]).flatten()
		self.z = np.asarray(all_points_transformed[2,:]).flatten()


		# self.x = self.x + translation_vector[0] # +
		# self.y = self.y + translation_vector[1] # +
		# self.z = self.z + translation_vector[2]

		#self.rotation_by_rodrigues_vector(rotation_vector=rotation_vector)

		#self.translation_by_vector(translation_vector=translation_vector)


	# def essential_from_rotations_translations(self, rotations, translations):

	# 	#

	def rotation_from_rodrigues_through_yaw_pitch_roll(self, rotation_vector):
		if rotation_vector[2] < 0:
			rotation_vector[0] = -1 * rotation_vector[0]
			rotation_vector[1] = -1 * rotation_vector[1]
			rotation_vector[2] = -1 * rotation_vector[2]

		rotation_matrix = np.zeros(shape=(3,3))
		cv2.Rodrigues(rotation_vector, rotation_matrix)

		sin_x = math.sqrt(rotation_matrix[2,0] * rotation_matrix[2,0] +  rotation_matrix[2,1] * rotation_matrix[2,1])    
		singular = sin_x < 1e-6
		if not singular:
			a1 = math.atan2(rotation_matrix[2,0], rotation_matrix[2,1])     # around z1-axis
			a2  = math.atan2(sin_x, rotation_matrix[2,2])     				# around x-axis
			a3 = math.atan2(rotation_matrix[0,2], -rotation_matrix[1,2])    # around z2-axis
		else: # gimbal lock
			a1 = 0                                         # around z1-axis
			a2  = math.atan2(sin_x, rotation_matrix[2,2])   # around x-axis
			a3 = 0                                         # around z2-axis

		angles = -180 * np.array([[a1], [a2], [a3]]) / math.pi
		#angles[0,0] = (360 - angles[0,0]) % 360 # change rotation sense if needed, comment this line otherwise
		angles[1,0] = angles[1,0] + 90

		# 1,2,3
		# 1,3,2
		# 2,3,1
		# 2,1,3
		# 3,1,2
		# 3,2,1
		a1 = math.radians(angles[0,0])
		a2 = math.radians(angles[1,0])
		a3 = math.radians(angles[2,0])

		# a1 = math.radians(0)
		# a2 = math.radians(0)
		# a3 = math.radians(0)

		yaw = angles[0,0]
		pitch = angles[1,0]
		roll = angles[2,0]
		print("YAW = {}, PITCH = {}, ROLL = {}".format(yaw, pitch, roll))


		# if pitch < -85:
		# 	pitch = -87.0
		# else:
		# 	pitch = -77.0
		#self.x, self.y, self.z = roll_correction(roll=roll, x=self.x, y=self.y, z=self.z)
		self.x, self.y, self.z = yaw_correction(yaw=yaw, x=self.x, y=self.y, z=self.z)
		self.x, self.y, self.z = pitch_correction(pitch=pitch, x=self.x, y=self.y, z=self.z)
		self.x, self.y, self.z = roll_correction(roll=roll, x=self.x, y=self.y, z=self.z)

		#self.x, self.y, self.z = yaw_correction(yaw=yaw, x=self.x, y=self.y, z=self.z)

		#yaw_raw = math.radians(yaw)
		#pitch_raw = math.radians(pitch)
		#roll_raw = math.radians(roll)

		# roll, pitch, yaw

		# yaw = roll_raw
		# pitch = pitch_raw
		# roll = yaw_raw

		# r1c1 = math.cos(a1) * math.cos(a3) - math.cos(a2) * math.sin(a1) * math.sin(a3)
		# r1c2 = - math.cos(a1) * math.sin(a3) - math.cos(a2) * math.cos(a3) * math.sin(a1)
		# r1c3 = math.sin(a1) * math.sin(a2)
		# r2c1 = math.cos(a3) * math.sin(a1) + math.cos(a1) * math.cos(a2) * math.sin(a3)
		# r2c2 = math.cos(a1) * math.cos(a2) * math.cos(a3) - math.sin(a1) * math.sin(a3)
		# r2c3 = - math.cos(a1) * math.sin(a2)
		# r3c1 = math.sin(a2) * math.sin(a3)
		# r3c2 = math.cos(a3) * math.sin(a2)
		# r3c3 = math.cos(a2)


		# r1c1 = math.cos(a1) * math.cos(a2) 
		# r1c2 = math.cos(a1) * math.sin(a2) * math.sin(a3) - math.cos(a3) * math.sin(a1)
		# r1c3 = math.sin(a1) * math.sin(a3) + math.cos(a1) * math.cos(a3) * math.sin(a2)
		# r2c1 = math.cos(a2) * math.sin(a1) 
		# r2c2 = math.cos(a1) * math.cos(a3) + math.sin(a1) * math.sin(a2) * math.sin(a3)
		# r2c3 = math.cos(a3) * math.sin(a1) * math.sin(a2) - math.cos(a1) * math.sin(a3)
		# r3c1 = - math.sin(a2)
		# r3c2 = math.cos(a2) * math.sin(a3)
		# r3c3 = math.cos(a2) * math.cos(a3)


		# r1c1 = math.cos(a1) * math.cos(a2) * math.cos(a3) - math.sin(a1) * math.sin(a3)
		# r1c2 = - math.cos(a3) * math.sin(a1) - math.cos(a1) * math.cos(a2) * math.sin(a3)
		# r1c3 = math.cos(a1) * math.sin(a2)
		# r2c1 = math.cos(a1) * math.sin(a3) + math.cos(a2) * math.cos(a3) * math.sin(a1)
		# r2c2 = math.cos(a1) * math.cos(a3) - math.cos(a2) * math.sin(a1) * math.sin(a3)
		# r2c3 = math.sin(a1) * math.sin(a2)
		# r3c1 = math.cos(a3) * math.sin(a2)
		# r3c2 = math.sin(a2) * math.sin(a3)
		# r3c3 = math.cos(a2)

		#reconstructed_3x3_rotation_matrix = np.matrix( [[r1c1,r1c2,r1c3], [r2c1,r2c2,r2c3], [r3c1,r3c2,r3c3]] )
		#print(reconstructed_3x3_rotation_matrix)

		#all_points = np.matrix([self.y,self.x,self.z]) # FLIP X AND Y

		#print("Transforming by reconstructed rotation matrix")
		#transformed_all_points = reconstructed_3x3_rotation_matrix * all_points

		#self.x = np.asarray(transformed_all_points[0,:]).flatten()
		#self.y = np.asarray(transformed_all_points[1,:]).flatten()
		#self.z = np.asarray(transformed_all_points[2,:]).flatten()
