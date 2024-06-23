import numpy as np
import matplotlib.pyplot as plt
import optimizer.pyqt_fit.nonparam_regression as smooth
from optimizer.pyqt_fit import npr_methods

polarization_lenses = True

# values to fit: e.g. camera_focuses (motor control value), as a function of distances
distances =  			[195.0, 		341.945831,	361.776886,	381.738800,	421.888458,	502.716736,	665.006653,	985.424744,	1189.634888]
camera_focuses = 		[72.0,			149.38,		150.98,		161.69,		172.73,		188.87,		210.04,		224.38,		228.64]
projector_focuses = 	[14.0,			23.56,		24.21,		24.89,		26.40,		29.18,		31.20,		34.05,		35.33]
exposure_times = 		[10.0,			14.0,		18.5,		20.5,		24.5,		34.5,		55.0,		117.5,		179.13]

projector_f_stop = 2.8 # current guess, contact jeremy@ajile.ca to confirm exact value
projector_circle_of_confusion = 0.0350 # mm
projector_focal_length = 12.00 # mm

class ScannerOptics():
	def __init__(self):
		self.fit_camera_focus_to_distance()
		self.fit_projector_focus_to_distance()
		self.fit_exposure_time_to_distance()

	def fit_camera_focus_to_distance(self):
		self.camera_focus_kernel = smooth.NonParamRegression(distances, camera_focuses, method=npr_methods.LocalPolynomialKernel(q=2))
		self.camera_focus_kernel.fit()

	def fit_projector_focus_to_distance(self):
		self.projector_focus_kernel = smooth.NonParamRegression(distances, projector_focuses, method=npr_methods.LocalPolynomialKernel(q=2))
		self.projector_focus_kernel.fit()

	def fit_exposure_time_to_distance(self):
		self.exposure_time_kernel = smooth.NonParamRegression(distances, exposure_times, method=npr_methods.LocalPolynomialKernel(q=2))
		self.exposure_time_kernel.fit()

	def plot_camera_focus_to_distance(self):
		grid = np.r_[min(distances):max(distances):1731.0j]
		plt.clf()
		plt.title("target distance vs. camera focus")
		plt.plot(distances, camera_focuses, 'o', alpha=0.5)
		plt.plot(grid, self.camera_focus_kernel(grid), 'y', linewidth=2)
		plt.ylabel("camera focus ring rotation (degrees)")
		plt.xlabel("distance (millimeters)")
		plt.savefig("{}_vs_{}.png".format("camera_focuses", "distances"))
		print("Camera angle (degrees) evaluated at 300mm, 400mm, 500mm: {}".format(self.camera_focus_kernel([300.0, 400.0, 500.0])))

	def plot_projector_focus_to_distance(self):
		grid = np.r_[min(distances):max(distances):1731.0j]
		plt.clf()
		plt.title("target distance vs. projector focus")
		plt.plot(distances, projector_focuses, 'o', alpha=0.5)
		plt.plot(grid, self.projector_focus_kernel(grid), 'y', linewidth=2)
		plt.ylabel("projector focus ring rotation (degrees)")
		plt.xlabel("distance (millimeters)")
		plt.savefig("{}_vs_{}.png".format("projector_focuses", "distances"))
		print("Projector angle (degrees) evaluated at 300mm, 400mm, 500mm: {}".format(self.projector_focus_kernel([300.0, 400.0, 500.0])))

	def plot_exposure_time_to_distance(self):
		grid = np.r_[min(distances):max(distances):1731.0j]
		plt.clf()
		plt.title("target distance vs. exposure times")
		plt.plot(distances, exposure_times, 'o', alpha=0.5)
		plt.plot(grid, self.exposure_time_kernel(grid), 'y', linewidth=2)
		plt.ylabel("exposure time (milliseconds)")
		plt.xlabel("distance (millimeters)")
		plt.savefig("{}_vs_{}.png".format("exposure_times", "distances"))
		print("Exposure time (ms) evaluated at 300mm, 400mm, 500mm: {}".format(self.exposure_time_kernel([300.0, 400.0, 500.0])))

	def camera_focus(self, distance):
		return self.camera_focus_kernel([distance])[0]

	def projector_focus(self, distance):
		return self.projector_focus_kernel([distance])[0]

	def exposure_time(self, distance):
		return self.exposure_time_kernel([distance])[0]

	def focus(self, distance):
		camera_focus = self.camera_focus(distance = distance)
		projector_focus = self.projector_focus(distance = distance)
		exposure_time = self.exposure_time(distance = distance)
		return [camera_focus, projector_focus, exposure_time]

	def depth_of_field(self, distance):
		if distance:
			minimum_in_focus_depth = distance * projector_focal_length**2 / (projector_focal_length**2 + projector_f_stop * projector_circle_of_confusion * distance)
			if distance > 1000.0:
				maximum_in_focus_depth = 3000.0
			else:
				maximum_in_focus_depth = distance * projector_focal_length**2 / (projector_focal_length**2 - projector_f_stop * projector_circle_of_confusion * distance)
		else:
			minimum_in_focus_depth = 0.0
			maximum_in_focus_depth = 10000.0
		return minimum_in_focus_depth, maximum_in_focus_depth


if __name__ == "__main__":
	optics = ScannerOptics()
	for distance in range(200, 1500, 50):
		minimum_in_focus_depth, maximum_in_focus_depth = optics.depth_of_field(distance=distance)
		camera_focus = optics.camera_focus(distance = distance)
		projector_focus = optics.projector_focus(distance = distance)
		exposure_time = optics.exposure_time(distance = distance)
		print("Targeting {:.1f}mm away, minimum in-focus depth = {:.1f}mm, maximum in-focus depth = {:.1f}mm, camera focus at {} degrees, projector focus at {} degrees, exposure time at {} seconds".format(distance, minimum_in_focus_depth, maximum_in_focus_depth, camera_focus, projector_focus, exposure_time))
	optics.plot_exposure_time_to_distance()
	optics.plot_projector_focus_to_distance()
	optics.plot_camera_focus_to_distance()