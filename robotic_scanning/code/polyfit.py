import time
import numpy as np
import matplotlib.pyplot as plt
import optimizer.pyqt_fit.nonparam_regression as smooth
from optimizer.pyqt_fit import npr_methods

distances = 		[275.174, 306.539, 336.916, 367.396, 407.918, 438.157, 488.493, 538.693, 613.766, 688.410, 788.626, 887.188, 1032.60, 1176.94, 1445.80, 1586.95, 1731.00]
camera_focuses = 	[136.200, 182.000, 184.500, 193.700, 204.500, 214.400, 224.500, 230.900, 238.400, 246.100, 253.100, 261.400, 267.200, 271.900, 282.200, 283.400, 289.20 ]
projector_focuses = [15.080,  17.300,  18.000,  18.700,  20.210,  20.650,  22.210,  22.580,  23.690,  24.070,  26.050,  26.560,  26.810,  27.070,  27.660,  27.830,  28.310 ]
exposure_times = 	[9.200,   9.600,   10.800,  12.900,  16.600,  17.300,  22.000,  24.700,  32.900,  40.600,  41.900,  43.200,  44.500,  45.800,  47.100,  48.400,  49.700 ] 

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
		lt.title("target distance vs. camera focus")
		plt.plot(distances, camera_focuses, 'o', alpha=0.5)
		plt.plot(grid, self.camera_focus_kernel(grid), 'y', linewidth=2)
		plt.ylabel("camera focus ring rotation (degrees)")
		plt.xlabel("distance (millimeters)")
		plt.savefig("{}_vs_{}.png".format("camera_focuses", "distances"))
		print("Camera angle (degrees) evaluated at 300mm, 400mm, 500mm: {}".format(self.camera_focus_kernel([300.0, 400.0, 500.0])))

	def plot_projector_focus_to_distance(self):
		grid = np.r_[min(distances):max(distances):1731.0j]
		lt.title("target distance vs. projector focus")
		plt.plot(distances, projector_focuses, 'o', alpha=0.5)
		plt.plot(grid, self.projector_focus_kernel(grid), 'y', linewidth=2)
		plt.ylabel("projector focus ring rotation (degrees)")
		plt.xlabel("distance (millimeters)")
		plt.savefig("{}_vs_{}.png".format("projector_focuses", "distances"))
		print("Projector angle (degrees) evaluated at 300mm, 400mm, 500mm: {}".format(self.projector_focus_kernel([300.0, 400.0, 500.0])))

	def plot_exposure_time_to_distance(self):
		grid = np.r_[min(distances):max(distances):1731.0j]
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

if __name__ == "__main__":
	start_time = time.time()
	target_distance = 1000

	optics = ScannerOptics()
	camera_focus, projector_focus, exposure_time = optics.focus(target_distance)

	print("{} degree camera focus angle for target distance {} mm away".format(camera_focus, target_distance))
	print("{} degree projector focus angle for target distance {} mm away".format(projector_focus, target_distance))
	print("{} millisecond exposure time for target distance {} mm away".format(exposure_time, target_distance))

	print(time.time() - start_time)