import numpy as np
from scipy.optimize import minimize

def optimize_transformation(rotation0, translation0, source_points, target_points):
	current_step = 0
	current_error = None

	def average_transformed_point_to_point_distance(rotation, translation, source_points, target_points):
		transformed_source_points = np.dot(source_points, rotation) + translation
		average_point_to_point_difference = np.average(np.abs(transformed_source_points - target_points)) * 1000
		return average_point_to_point_difference

	def objective_function_evaluation(hypothesis, args):
		new_rotation = hypothesis[:9]
		new_rotation = np.asarray(new_rotation).reshape(3,3)
		new_translation = hypothesis[9:]
		new_translation = np.asarray(new_translation)
		source_points = args[0]
		target_points = args[1]
		average_point_to_point_difference = average_transformed_point_to_point_distance(rotation=new_rotation, translation=new_translation, source_points=source_points, target_points=target_points)
		#print("New rotation: {}".format(new_rotation))
		#print("New translation: {}".format(new_translation))
		#global current_step
		#current_step += 1
		#if average_point_to_point_difference != current_error:
		#current_error = average_point_to_point_difference
		#print(average_point_to_point_difference)
		return average_point_to_point_difference

	average_point_to_point_difference = average_transformed_point_to_point_distance(rotation=rotation0, translation=translation0, source_points=source_points, target_points=target_points)
	#print("Original average point-to-point distance = {:.1f} microns".format(average_point_to_point_difference))
	rotation0 = rotation0.flatten().tolist()
	translation0 = translation0.flatten().tolist()
	initial_guess = rotation0 + translation0
	sol = minimize(fun=objective_function_evaluation, x0=initial_guess, args=[source_points, target_points], options={'maxiter': 1000}, method='BFGS')
	transformation = sol.x
	final_rotation = transformation[:9]
	final_rotation = np.asarray(final_rotation).reshape(3,3)

	final_translation = transformation[9:]
	final_translation = np.asarray(final_translation)

	final_average_point_to_point_difference = average_transformed_point_to_point_distance(rotation=final_rotation, translation=final_translation, source_points=source_points, target_points=target_points)
	print("Average point-to-point alignment error = {:.1f} microns (after original error of {:.1f})".format(final_average_point_to_point_difference, average_point_to_point_difference))
	return final_rotation, final_translation