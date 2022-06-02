from matplotlib import pyplot as plt
import numpy as np

def polynomial_decay(current_step, total_steps, start_value, end_value, exponential_index=1, curvature_shape=1):
    return (start_value - end_value) * (1 - current_step**curvature_shape / total_steps**curvature_shape)**exponential_index + end_value

exponential_indices = [1,2,3,4,5,6,7]
curvature_shapes = [1,2,3,4,5,6,7]
total_steps = 1000
start_value = 1.0
end_value = 0.0

# fig = plt.figure()
# ax = plt.axes()

x = np.linspace(0, 1000, 1)

for exponential_index in exponential_indices:
	for curvature_shape in curvature_shapes:
		label = "k={}, N={}".format(curvature_shape, exponential_index)
		line = []
		for current_step in range(total_steps):
			value = polynomial_decay(current_step, total_steps, start_value, end_value, exponential_index, curvature_shape)
			line.append(value)
		plt.plot(np.asarray(line), label=label)

plt.title("Visualization of exponential decay curves for loss metrics, explored through hyperparameter optimization")
plt.legend()
plt.show()